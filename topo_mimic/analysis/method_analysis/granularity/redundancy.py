"""Redundancy analysis using DTW distance between clusters."""

import os
import pickle as pkl
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from tqdm import tqdm

from .config import GranularityConfig
from .dm_multi_persistence import DMExtractionResult
from .utils import count_unique_clusters, union_find_groups


@dataclass
class RedundantPair:
    cluster_i: int
    cluster_j: int
    dtw_distance: float


@dataclass
class ClusteringResult:
    pers_delta: float
    n_clusters: int
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]]
    labels: np.ndarray
    windows: list[tuple[int, np.ndarray]]
    D: np.ndarray
    Dn: np.ndarray


@dataclass
class RedundancyResult:
    pers_delta: float
    n_clusters: int
    dtw_matrix: np.ndarray
    cluster_ids: list[int]
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]]
    redundant_pairs_fixed: list[RedundantPair]
    redundant_pairs_percentile: list[RedundantPair]
    n_unique_fixed: int
    n_unique_percentile: int
    top_similar_pairs: list[tuple[int, int, float]]


@dataclass
class RedundancyAnalysisOutput:
    results: dict[tuple[float, int], RedundancyResult]
    summary_df: pd.DataFrame


def compute_dtw(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
) -> float:
    """
    Compute DTW distance between two sequences.

    Args:
        seq_a: (T1, D) first sequence
        seq_b: (T2, D) second sequence

    Returns:
        Normalized DTW distance
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float('inf')

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq_a[i - 1] - seq_b[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )

    return dtw_matrix[n, m] / (n + m)


def extract_sequences_from_cluster(
    cluster_entries: list[tuple[int, np.ndarray]],
    qpos_data: np.ndarray,
    skip_root_dims: int = 7,
    min_seq_length: int = 3,
) -> list[np.ndarray]:
    """
    Extract qpos sequences from cluster entries.

    Args:
        cluster_entries: List of (clip_id, frame_indices) tuples
        qpos_data: (n_clips, n_frames, D) qpos array
        skip_root_dims: Root dims to skip
        min_seq_length: Minimum sequence length

    Returns:
        List of (T, D) sequence arrays
    """
    sequences = []

    for clip_id, frame_indices in cluster_entries:
        clip_id = int(clip_id)
        if clip_id >= len(qpos_data):
            continue

        frames = np.asarray(frame_indices).flatten()
        frames = np.sort(frames)
        valid_frames = frames[(frames >= 0) & (frames < len(qpos_data[clip_id]))]

        if len(valid_frames) < min_seq_length:
            continue

        seq = qpos_data[clip_id, valid_frames]
        if skip_root_dims > 0:
            seq = seq[:, skip_root_dims:]
        sequences.append(seq)

    return sequences


def compute_cluster_representative(
    sequences: list[np.ndarray],
    max_samples: int = 10,
    seed: int = 42,
) -> np.ndarray | None:
    """
    Get representative sequence from cluster (medoid or random sample).

    Args:
        sequences: List of sequences
        max_samples: Max sequences to consider
        seed: Random seed

    Returns:
        Representative sequence or None if empty
    """
    if not sequences:
        return None

    if len(sequences) > max_samples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(sequences), max_samples, replace=False)
        sequences = [sequences[i] for i in indices]

    if len(sequences) == 1:
        return sequences[0]

    target_len = int(np.median([len(s) for s in sequences]))
    resampled = []
    for seq in sequences:
        if len(seq) == target_len:
            resampled.append(seq)
        else:
            idx = np.linspace(0, len(seq) - 1, target_len).astype(int)
            resampled.append(seq[idx])

    return np.mean(resampled, axis=0)


def compute_pairwise_cluster_dtw(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    qpos_data: np.ndarray,
    config: GranularityConfig,
) -> tuple[np.ndarray, list[int]]:
    """
    Compute pairwise DTW distance matrix between cluster representatives.

    Args:
        cluster_dict: Dict mapping cluster_id to entries
        qpos_data: (n_clips, n_frames, D) qpos array
        config: Configuration object

    Returns:
        Tuple of (dtw_matrix, cluster_ids)
    """
    cluster_ids = sorted(cluster_dict.keys())
    n_clusters = len(cluster_ids)

    representatives = {}
    for cid in cluster_ids:
        sequences = extract_sequences_from_cluster(
            cluster_dict[cid],
            qpos_data,
            skip_root_dims=config.joints.skip_root_dims,
        )
        rep = compute_cluster_representative(sequences, seed=config.redundancy.seed)
        representatives[cid] = rep

    dtw_matrix = np.zeros((n_clusters, n_clusters))

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            rep_i = representatives[cluster_ids[i]]
            rep_j = representatives[cluster_ids[j]]

            if rep_i is None or rep_j is None:
                dtw_matrix[i, j] = dtw_matrix[j, i] = float('inf')
            else:
                dist = compute_dtw(rep_i, rep_j)
                dtw_matrix[i, j] = dtw_matrix[j, i] = dist

    return dtw_matrix, cluster_ids


def identify_redundant_pairs(
    dtw_matrix: np.ndarray,
    threshold: float | None = None,
    percentile: float | None = None,
) -> list[tuple[int, int, float]]:
    """
    Identify redundant cluster pairs based on DTW threshold.

    Args:
        dtw_matrix: (n, n) symmetric DTW distance matrix
        threshold: Fixed threshold (pairs below are redundant)
        percentile: Percentile threshold (bottom X% are redundant)

    Returns:
        List of (i, j, distance) tuples for redundant pairs
    """
    n = dtw_matrix.shape[0]
    pairs_with_dist = []

    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_matrix[i, j]
            if np.isfinite(d):
                pairs_with_dist.append((i, j, d))

    if not pairs_with_dist:
        return []

    if percentile is not None:
        distances = [d for _, _, d in pairs_with_dist]
        threshold = np.percentile(distances, percentile)

    if threshold is None:
        return []

    redundant = [(i, j, d) for i, j, d in pairs_with_dist if d < threshold]
    return sorted(redundant, key=lambda x: x[2])


def prepare_windows_from_dm(
    dm_result: DMExtractionResult,
    intention_data: np.ndarray,
    config: GranularityConfig,
    clip_indices: list[int],
    source: str = "cycle",
) -> list[tuple[int, np.ndarray]]:
    """
    Prepare windows from DM cycles or paths for clustering.

    Args:
        dm_result: DM extraction result
        intention_data: (n_clips, n_frames, D) intention array
        config: Configuration object
        clip_indices: List of clip indices
        source: "cycle" to use all_cycle_dm, "path" to use all_path_dm

    Returns:
        List of (clip_id, frame_indices) tuples
    """
    from topo_mimic.extraction.skeletonization.postprocess import (
        _expand_frames_nonconsec,
        _uniform_downsample,
    )

    if source == "path" and dm_result.all_path_dm is not None:
        all_dm = dm_result.all_path_dm
    else:
        all_dm = dm_result.all_cycle_dm

    windows = []
    takens_offset = (config.takens.m - 1) * config.takens.tau
    n_clips = len(intention_data)

    for local_idx in range(n_clips):
        if local_idx >= len(all_dm):
            continue

        segments = all_dm[local_idx]
        if not segments:
            continue

        if source == "cycle" and config.clustering.only_max:
            segments = [max(segments, key=len)]

        clip_len = intention_data.shape[1]
        for seg in segments:
            if len(seg) < 3:
                continue

            seg_frames = np.asarray(seg, dtype=int) + takens_offset
            expanded = _expand_frames_nonconsec(seg_frames, pad=0, clip_len=clip_len)
            sampled = _uniform_downsample(expanded, config.clustering.target_len)

            if sampled is not None:
                windows.append((local_idx, sampled))

    return windows


def run_clustering_for_k(
    windows: list[tuple[int, np.ndarray]],
    intention_data: np.ndarray,
    n_clusters: int,
    config: GranularityConfig,
    clip_indices: list[int],
) -> ClusteringResult | None:
    """
    Run W2-based clustering for a specific k.

    Args:
        windows: List of (clip_id, frame_indices) tuples
        intention_data: (n_clips, n_frames, D) intention array
        n_clusters: Number of clusters
        config: Configuration object
        clip_indices: List of clip indices

    Returns:
        ClusteringResult or None if clustering fails
    """
    from topo_mimic.experiment.transitions import compute_w2_distance_matrix, labels_to_cluster_dict

    if len(windows) < n_clusters:
        print(f"    Warning: Not enough windows ({len(windows)}) for {n_clusters} clusters")
        return None

    D, Dn = compute_w2_distance_matrix(
        windows, intention_data, config.clustering.target_len
    )

    mds = MDS(
        n_components=config.clustering.mds_components,
        dissimilarity='precomputed',
        random_state=config.redundancy.seed,
    )
    X_embedded = mds.fit_transform(Dn)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=config.redundancy.seed,
        n_init='auto',
    )
    labels = kmeans.fit_predict(X_embedded)

    cluster_dict = labels_to_cluster_dict(labels, windows, clip_indices)

    return ClusteringResult(
        pers_delta=0.0,
        n_clusters=n_clusters,
        cluster_dict=cluster_dict,
        labels=labels,
        windows=windows,
        D=D,
        Dn=Dn,
    )


def run_redundancy_for_config(
    clustering_result: ClusteringResult,
    qpos_data: np.ndarray,
    config: GranularityConfig,
    pers_delta: float,
    output_dir: Path,
) -> RedundancyResult:
    """
    Run redundancy analysis for a single (pers_delta, n_clusters) configuration.

    Args:
        clustering_result: Result from clustering
        qpos_data: (n_clips, n_frames, D) qpos array
        config: Configuration object
        pers_delta: Persistence value
        output_dir: Output directory

    Returns:
        RedundancyResult
    """
    n_clusters = clustering_result.n_clusters
    cache_dir = output_dir / f"pers_{pers_delta:.2f}" / f"n_clusters_{n_clusters}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dtw_cache_path = cache_dir / "dtw_matrix.pkl"

    if config.redundancy.use_cached_dtw and dtw_cache_path.exists():
        from topo_mimic.analysis.method_analysis.granularity.dm_multi_persistence import _LegacyAwareUnpickler
        with open(dtw_cache_path, "rb") as f:
            try:
                cached = _LegacyAwareUnpickler(f).load()
            except Exception:
                f.seek(0)
                cached = pkl.load(f)
        dtw_matrix = cached["D"]
        cluster_ids = cached["cluster_ids"]
        print(f"    Loaded cached DTW matrix from {dtw_cache_path}")
    else:
        dtw_matrix, cluster_ids = compute_pairwise_cluster_dtw(
            clustering_result.cluster_dict, qpos_data, config
        )

        if config.redundancy.cache_dtw_matrix:
            with open(dtw_cache_path, "wb") as f:
                pkl.dump({"D": dtw_matrix, "cluster_ids": cluster_ids}, f)

    redundant_fixed = identify_redundant_pairs(
        dtw_matrix, threshold=config.redundancy.dtw_threshold
    )
    redundant_percentile = identify_redundant_pairs(
        dtw_matrix, percentile=config.redundancy.dtw_percentile
    )

    pairs_fixed = [(i, j) for i, j, _ in redundant_fixed]
    pairs_percentile = [(i, j) for i, j, _ in redundant_percentile]

    n_unique_fixed = count_unique_clusters(pairs_fixed, len(cluster_ids))
    n_unique_percentile = count_unique_clusters(pairs_percentile, len(cluster_ids))

    all_pairs = []
    n = dtw_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_matrix[i, j]
            if np.isfinite(d):
                all_pairs.append((cluster_ids[i], cluster_ids[j], d))

    all_pairs.sort(key=lambda x: x[2])
    top_similar = all_pairs[:config.redundancy.top_k_similar_pairs]

    if redundant_fixed:
        pairs_df = pd.DataFrame([
            {"cluster_i": cluster_ids[i], "cluster_j": cluster_ids[j], "dtw_distance": d}
            for i, j, d in redundant_fixed
        ])
        pairs_df.to_csv(cache_dir / "redundant_pairs_fixed.csv", index=False)

    if redundant_percentile:
        pairs_df = pd.DataFrame([
            {"cluster_i": cluster_ids[i], "cluster_j": cluster_ids[j], "dtw_distance": d}
            for i, j, d in redundant_percentile
        ])
        pairs_df.to_csv(cache_dir / "redundant_pairs_percentile.csv", index=False)

    with open(cache_dir / "cluster_dict.pkl", "wb") as f:
        pkl.dump(clustering_result.cluster_dict, f)

    dtw_df = pd.DataFrame(
        dtw_matrix,
        index=[f"C{c}" for c in cluster_ids],
        columns=[f"C{c}" for c in cluster_ids],
    )
    dtw_df.to_csv(cache_dir / "dtw_matrix.csv")
    print(f"      Saved DTW matrix to {cache_dir / 'dtw_matrix.csv'}")

    return RedundancyResult(
        pers_delta=pers_delta,
        n_clusters=n_clusters,
        dtw_matrix=dtw_matrix,
        cluster_ids=cluster_ids,
        cluster_dict=clustering_result.cluster_dict,
        redundant_pairs_fixed=[RedundantPair(cluster_ids[i], cluster_ids[j], d) for i, j, d in redundant_fixed],
        redundant_pairs_percentile=[RedundantPair(cluster_ids[i], cluster_ids[j], d) for i, j, d in redundant_percentile],
        n_unique_fixed=n_unique_fixed,
        n_unique_percentile=n_unique_percentile,
        top_similar_pairs=top_similar,
    )


def run_redundancy_analysis(
    dm_results: dict[float, DMExtractionResult],
    intention_data: np.ndarray,
    qpos_data: np.ndarray,
    config: GranularityConfig,
    clip_indices: list[int],
    source: str = "cycle",
) -> RedundancyAnalysisOutput:
    """
    Run full redundancy analysis across persistence values and cluster counts.

    Args:
        dm_results: Dict mapping pers_delta to DMExtractionResult
        intention_data: (n_clips, n_frames, D) intention array
        qpos_data: (n_clips, n_frames, D) qpos array
        config: Configuration object
        clip_indices: List of clip indices
        source: "cycle" or "path" — which DM features to use for windowing

    Returns:
        RedundancyAnalysisOutput with all results
    """
    print(f"\n=== Redundancy Analysis (source={source}) ===")

    output_dir = Path(config.output_dir) / "redundancy"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summary_rows = []

    for pers_delta in tqdm(config.multi_persistence.pers_delta_values, desc="Persistence values"):
        dm_result = dm_results[pers_delta]

        print(f"\n  Persistence delta = {pers_delta}")
        windows = prepare_windows_from_dm(dm_result, intention_data, config, clip_indices, source=source)
        print(f"    Prepared {len(windows)} windows ({source})")

        if len(windows) < max(config.redundancy.n_clusters_range):
            print(f"    Warning: Only {len(windows)} windows, some k values may be skipped")

        for n_clusters in config.redundancy.n_clusters_range:
            if len(windows) < n_clusters:
                continue

            cache_dir = output_dir / f"pers_{pers_delta:.2f}" / f"n_clusters_{n_clusters}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            clustering_cache_path = cache_dir / "clustering_result.pkl"

            if config.redundancy.use_cached_clustering and clustering_cache_path.exists():
                from topo_mimic.analysis.method_analysis.granularity.dm_multi_persistence import _LegacyAwareUnpickler
                with open(clustering_cache_path, "rb") as f:
                    try:
                        clustering_result = _LegacyAwareUnpickler(f).load()
                    except Exception:
                        f.seek(0)
                        clustering_result = pkl.load(f)
                print(f"    Loaded cached clustering for k={n_clusters} from {clustering_cache_path}")
            else:
                print(f"    Running clustering for k={n_clusters}...")
                clustering_result = run_clustering_for_k(
                    windows, intention_data, n_clusters, config, clip_indices
                )

                if clustering_result is None:
                    continue

                clustering_result.pers_delta = pers_delta

                if config.redundancy.cache_clustering:
                    with open(clustering_cache_path, "wb") as f:
                        pkl.dump(clustering_result, f)

            if clustering_result is None:
                continue

            print(f"    Computing redundancy for k={n_clusters}...")
            result = run_redundancy_for_config(
                clustering_result, qpos_data, config, pers_delta, output_dir
            )

            all_results[(pers_delta, n_clusters)] = result

            summary_rows.append({
                "pers_delta": pers_delta,
                "n_clusters": n_clusters,
                "n_redundant_fixed": len(result.redundant_pairs_fixed),
                "n_redundant_percentile": len(result.redundant_pairs_percentile),
                "n_unique_fixed": result.n_unique_fixed,
                "n_unique_percentile": result.n_unique_percentile,
            })

            print(f"      Fixed: {len(result.redundant_pairs_fixed)} redundant pairs, "
                  f"{result.n_unique_fixed} unique clusters")
            print(f"      Percentile: {len(result.redundant_pairs_percentile)} redundant pairs, "
                  f"{result.n_unique_percentile} unique clusters")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    print(f"\n  Saved summary to {output_dir / 'summary.csv'}")

    return RedundancyAnalysisOutput(
        results=all_results,
        summary_df=summary_df,
    )
