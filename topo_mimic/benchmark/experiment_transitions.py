"""
Unified experiment runner for behavioral clustering benchmark.

Single entry point for running all clustering and state-space methods
with fixed number of clusters, integrated seed handling, and auto-generated
labels.yaml files for each method.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["JAX_ENABLE_X64"] = "True"

import sys
import functools
import multiprocessing as mp
import pickle as pkl
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from topo_mimic.dmwk.utils.load_utils import get_aggregate_data, get_clip_indices
from topo_mimic.dmwk.utils.feature_transforms import apply_dm_key_transform, get_hdf5_key_for_dm_key
from topo_mimic.dmwk.utils.dataset_config import get_dataset_preset
from topo_mimic.dmwk.skeletonization.postprocess import (
    _expand_frames_nonconsec,
    _uniform_downsample,
    _normalize_dist_matrix,
    chunk_loops_by_clustered_vpath,
)
from topo_mimic.dmwk.skeletonization.rendering import render_grid_clusters_side_by_side

try:
    import ot
    HAS_OT = True
except ImportError:
    HAS_OT = False

try:
    from tslearn.metrics import dtw as tslearn_dtw
    HAS_TSLEARN = True
except ImportError:
    HAS_TSLEARN = False

from topo_mimic.benchmark.methods import (
    get_method,
    MethodType,
    ClusteringResult,
    KPMSResult,
)

try:
    from topo_mimic.benchmark.hyperparam_sweep.visualization import (
        plot_reconstruction_timeseries,
    )
    HAS_RECON_VIZ = True
except ImportError:
    HAS_RECON_VIZ = False

try:
    from topo_mimic.benchmark.hyperparam_sweep.keypoint_loader import (
        setup_stac_reference,
        qpos_to_keypoints,
    )
    HAS_KP_LOADER = True
except ImportError:
    HAS_KP_LOADER = False


@dataclass
class DataConfig:
    dm_key: str = "intention"
    evaluation_key: str = "qpos"


@dataclass
class KeypointConfig:
    xml_path: str = "topo_mimic/dmwk/xml/rodent.xml"
    stac_data_path: str = "data/rodent_reference_clips.h5"
    kp_names: list[str] = None


@dataclass
class DMConfig:
    results_dir: str = ""


@dataclass
class DMSourceConfig:
    results_dir: str = ""
    dm_key: str = "intention"
    pca_dims: int | None = None
    xml_path: str = "topo_mimic/dmwk/xml/rodent.xml"
    stac_data_path: str = "data/rodent_reference_clips.h5"


@dataclass
class ClusteringConfig:
    min_len: int = 0
    max_len: int = 500
    target_len: int = 70
    only_max: bool = True
    skip_clips: tuple[int, ...] = ()


@dataclass
class MethodEntry:
    name: str
    type: str = "clustering"
    enabled: bool = True
    params: dict = field(default_factory=dict)
    dm_source: str = "default"


@dataclass
class RenderingConfig:
    enabled: bool = True
    pad: int = 0
    max_samples: int = 5
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)
    per_cluster_videos: bool = False


@dataclass
class KSelectionConfig:
    mode: str = "fixed"
    n_clusters: int = 10


@dataclass
class ClusterMergingConfig:
    enabled: bool = False
    dtw_threshold: float = 0.1  # Fixed threshold (pairs below are similar)
    dtw_percentile: float = 10.0  # Percentile threshold (bottom X% are similar)
    threshold_method: str = "both"  # "fixed", "percentile", or "both"
    top_k_similar_pairs: int = 5
    render_similar_pairs: bool = True
    samples_per_cluster: int = 3
    skip_root_dims: int | None = None


@dataclass
class ExperimentConfig:
    dataset: str = "rodent"
    output_dir: str = "outputs/experiment"
    data_path: str = ""
    rodent_rollout_dir: str = ""
    split: str = "train"
    n_clips: int | None = None
    behavior_labels_csv: str = ""

    data: DataConfig = None
    dm: DMConfig = None
    dm_sources: dict[str, DMSourceConfig] = field(default_factory=dict)
    clustering: ClusteringConfig = None
    keypoint: KeypointConfig = None

    k_selection: KSelectionConfig = None
    seeds: list[int] = field(default_factory=lambda: [0])
    aggregate_seeds: bool = True

    methods: list[MethodEntry] = field(default_factory=list)
    rendering: RenderingConfig = None
    cluster_merging: ClusterMergingConfig = None

    def __post_init__(self):
        from topo_mimic.dmwk.utils.dataset_config import get_dataset_preset

        preset = get_dataset_preset(self.dataset)

        if not self.data_path and not self.rodent_rollout_dir:
            self.data_path = preset.default_data_path
        elif not self.data_path:
            self.data_path = self.rodent_rollout_dir
        self.rodent_rollout_dir = self.data_path

        if self.data is None:
            self.data = DataConfig()
        if self.dm is None:
            self.dm = DMConfig()
        if self.clustering is None:
            self.clustering = ClusteringConfig()
        if not self.clustering.skip_clips and preset.skip_clips:
            self.clustering.skip_clips = preset.skip_clips
        if self.keypoint is None:
            self.keypoint = KeypointConfig(
                xml_path=preset.xml_path,
                stac_data_path=preset.stac_data_path,
            )
        if self.k_selection is None:
            self.k_selection = KSelectionConfig()
        if self.rendering is None:
            self.rendering = RenderingConfig()
        if self.cluster_merging is None:
            self.cluster_merging = ClusterMergingConfig()
        if self.cluster_merging.skip_root_dims is None:
            self.cluster_merging.skip_root_dims = preset.root_strip_dims
        if not self.dm_sources:
            self.dm_sources = {
                "default": DMSourceConfig(
                    results_dir=self.dm.results_dir,
                    dm_key=self.data.dm_key,
                )
            }

    @property
    def n_clusters(self) -> int:
        return self.k_selection.n_clusters


@dataclass
class ExperimentResults:
    results: dict
    config: ExperimentConfig


def load_config(config_path: str) -> ExperimentConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = DataConfig(**cfg.get("data", {}))
    dm_cfg = DMConfig(**cfg.get("dm", {}))

    clustering_dict = cfg.get("clustering", {})
    if "skip_clips" in clustering_dict:
        clustering_dict["skip_clips"] = tuple(clustering_dict["skip_clips"])
    clustering_cfg = ClusteringConfig(**clustering_dict)

    if "k_selection" in cfg:
        k_sel_dict = cfg["k_selection"].copy()
        for key in ["plot_dunn", "plot_silhouette", "evaluation_key", "k_min", "k_max", "save_all_k"]:
            k_sel_dict.pop(key, None)
        k_sel_dict["mode"] = "fixed"
        k_selection_cfg = KSelectionConfig(**k_sel_dict)
    elif "n_clusters" in cfg:
        k_selection_cfg = KSelectionConfig(
            mode="fixed",
            n_clusters=cfg["n_clusters"],
        )
    else:
        k_selection_cfg = KSelectionConfig()

    dm_sources = {}
    if "dm_sources" in cfg:
        for src_name, src_dict in cfg["dm_sources"].items():
            dm_sources[src_name] = DMSourceConfig(**src_dict)

    methods_list = []
    for m in cfg.get("methods", []):
        methods_list.append(MethodEntry(
            name=m["name"],
            type=m.get("type", "clustering"),
            enabled=m.get("enabled", True),
            params=m.get("params", {}),
            dm_source=m.get("dm_source", "default"),
        ))

    rendering_dict = cfg.get("rendering", {})
    if "target_size" in rendering_dict:
        rendering_dict["target_size"] = tuple(rendering_dict["target_size"])
    rendering_cfg = RenderingConfig(**rendering_dict)

    keypoint_dict = cfg.get("keypoint", {})
    keypoint_cfg = KeypointConfig(**keypoint_dict)

    cluster_merging_dict = cfg.get("cluster_merging", {})
    cluster_merging_cfg = ClusterMergingConfig(**cluster_merging_dict)

    data_path = cfg.get("data_path", cfg.get("rodent_rollout_dir", ""))

    return ExperimentConfig(
        dataset=cfg.get("dataset", "rodent"),
        output_dir=cfg.get("output_dir", ExperimentConfig.output_dir),
        data_path=data_path,
        rodent_rollout_dir=data_path,
        split=cfg.get("split", ExperimentConfig.split),
        n_clips=cfg.get("n_clips", None),
        behavior_labels_csv=cfg.get("behavior_labels_csv", ""),
        data=data_cfg,
        dm=dm_cfg,
        dm_sources=dm_sources,
        clustering=clustering_cfg,
        keypoint=keypoint_cfg,
        k_selection=k_selection_cfg,
        seeds=cfg.get("seeds", [0]),
        aggregate_seeds=cfg.get("aggregate_seeds", True),
        methods=methods_list,
        rendering=rendering_cfg,
        cluster_merging=cluster_merging_cfg,
    )


def load_data(
    data_key: str,
    rodent_rollout_dir: str,
    split: str,
    n_clips: int | None = None,
    total_clips: int = 842,
    test_size: int = 200,
) -> tuple[np.ndarray, list[int]]:
    clip_indices = get_clip_indices(split, n_clips=n_clips, total_clips=total_clips, test_size=test_size)
    work_fn = functools.partial(
        get_aggregate_data, f"/{data_key}", [], path=rodent_rollout_dir
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        data_list = list(tqdm(pool.imap(work_fn, clip_indices), total=len(clip_indices), desc=f"Loading {data_key}"))

    first_clip = data_list[0]
    if first_clip.ndim == 2:
        n_frames, n_dim = first_clip.shape
    else:
        total_size = first_clip.size
        n_dim = total_size // 500
        n_frames = 500

    data = np.array([
        clip.reshape(-1, n_dim) if clip.ndim != 2 else clip
        for clip in data_list
    ])

    min_frames = min(d.shape[0] for d in data)
    data = data[:, :min_frames, :]

    print(f"Loaded {data_key}: {data.shape} (split={split}, clips={clip_indices[0]}-{clip_indices[-1]})")
    return data, clip_indices


def load_source_data(
    source_cfg: DMSourceConfig,
    rodent_rollout_dir: str,
    split: str,
    n_clips: int | None,
    dataset: str = "rodent",
) -> tuple[np.ndarray, list[int]]:
    preset = get_dataset_preset(dataset)
    hdf5_key = get_hdf5_key_for_dm_key(source_cfg.dm_key)
    data, clip_indices = load_data(
        hdf5_key, rodent_rollout_dir, split, n_clips,
        total_clips=preset.total_clips, test_size=preset.test_size,
    )

    data = apply_dm_key_transform(
        data, source_cfg.dm_key, source_cfg.xml_path, source_cfg.stac_data_path,
        dataset=dataset,
    )
    print(f"After transform ({source_cfg.dm_key}): {data.shape}")

    if source_cfg.pca_dims is not None:
        from sklearn.decomposition import PCA
        n_clips_d, n_frames_d, n_dim_d = data.shape
        flat = data.reshape(-1, n_dim_d)
        pca = PCA(n_components=source_cfg.pca_dims)
        flat_reduced = pca.fit_transform(flat)
        data = flat_reduced.reshape(n_clips_d, n_frames_d, source_cfg.pca_dims)
        print(f"PCA reduced: {n_dim_d} -> {source_cfg.pca_dims} dims (var explained: {pca.explained_variance_ratio_.sum():.3f})")

    return data, clip_indices


def prepare_windows_for_clustering(
    all_cycle_dm: list,
    data: np.ndarray,
    clustering_cfg: ClusteringConfig,
    clip_indices: list[int] | None = None,
) -> tuple[list[tuple[int, np.ndarray]], list[np.ndarray]]:
    windows = []
    cleaned_loops = []
    n_clips = len(data)

    for local_idx in range(n_clips):
        # Convert to absolute clip ID for skip check
        abs_clip_id = clip_indices[local_idx] if clip_indices else local_idx
        if abs_clip_id in clustering_cfg.skip_clips:
            continue
        if local_idx >= len(all_cycle_dm):
            continue

        loops = [
            loop for loop in all_cycle_dm[local_idx]
            if clustering_cfg.min_len <= len(loop) <= clustering_cfg.max_len
        ]

        # only_max is handled upstream: dm_pipeline now pre-splits the longest
        # cycle into segments before pickling, so all entries are already from
        # the longest cycle. Skipping only_max filter here.

        clip_len = len(data[local_idx])
        for loop in loops:
            expanded = _expand_frames_nonconsec(loop, pad=0, clip_len=clip_len)
            sampled = _uniform_downsample(expanded, clustering_cfg.target_len)
            if sampled is None:
                continue
            windows.append((local_idx, sampled))  # Store local index for data access
            cleaned_loops.append(np.asarray(loop, dtype=int))

    if cleaned_loops:
        loop_lengths = np.array([len(loop) for loop in cleaned_loops])
        print(f"[segments] Segment length stats: mean={loop_lengths.mean():.2f}, std={loop_lengths.std():.2f}, n={len(loop_lengths)}")

    return windows, cleaned_loops


def prepare_unif_frames_for_clustering(
    data: np.ndarray,
    clustering_cfg: ClusteringConfig,
    clip_indices: list[int] | None = None,
) -> list[tuple[int, np.ndarray]]:
    """
    Prepare windows using uniformly downsampled frames from each clip.

    Creates one window per clip with frames uniformly downsampled to target_len.

    Args:
        data: Full data array (n_clips, n_frames, n_dim)
        clustering_cfg: Clustering configuration
        clip_indices: Absolute clip indices (for skip_clips check)

    Returns:
        List of (local_idx, frame_indices) tuples, one per clip
    """
    windows = []
    n_clips = len(data)

    for local_idx in range(n_clips):
        abs_clip_id = clip_indices[local_idx] if clip_indices else local_idx
        if abs_clip_id in clustering_cfg.skip_clips:
            continue

        clip_len = len(data[local_idx])
        all_frames = np.arange(clip_len)
        sampled = _uniform_downsample(all_frames, clustering_cfg.target_len)

        if sampled is not None:
            windows.append((local_idx, sampled))

    print(f"[unif baseline] Created {len(windows)} windows (1 per clip, {clustering_cfg.target_len} frames)")
    return windows


def prepare_full_frames_for_clustering(
    data: np.ndarray,
    clustering_cfg: ClusteringConfig,
    clip_indices: list[int] | None = None,
) -> list[tuple[int, np.ndarray]]:
    """
    Prepare windows using all frames from each clip (no downsampling).

    Creates one window per clip with every frame included.

    Args:
        data: Full data array (n_clips, n_frames, n_dim)
        clustering_cfg: Clustering configuration
        clip_indices: Absolute clip indices (for skip_clips check)

    Returns:
        List of (local_idx, frame_indices) tuples, one per clip
    """
    windows = []
    n_clips = len(data)

    for local_idx in range(n_clips):
        abs_clip_id = clip_indices[local_idx] if clip_indices else local_idx
        if abs_clip_id in clustering_cfg.skip_clips:
            continue

        clip_len = len(data[local_idx])
        windows.append((local_idx, np.arange(clip_len)))

    print(f"[full baseline] Created {len(windows)} windows (1 per clip, all frames)")
    return windows


def build_feature_matrix(
    windows: list[tuple[int, np.ndarray]],
    data: np.ndarray,
) -> np.ndarray:
    features = []
    for clip_id, frames in windows:
        traj = data[clip_id, frames].flatten()
        features.append(traj)
    return np.array(features)


def compute_w2_distance_matrix(
    windows: list[tuple[int, np.ndarray]],
    data: np.ndarray,
    target_len: int,
    squared_cost: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if not HAS_OT:
        raise ImportError("POT library not installed")

    M = len(windows)
    D = np.zeros((M, M), dtype=float)

    print(f"Computing W2 distance matrix for {M} windows...")
    for i in tqdm(range(M), desc="W2 distances"):
        ci, fi = windows[i]
        Xi = data[ci, fi]
        ni = len(fi)
        wi = np.full(ni, 1.0 / ni)

        for j in range(i + 1, M):
            cj, fj = windows[j]
            Xj = data[cj, fj]
            nj = len(fj)
            vj = np.full(nj, 1.0 / nj)

            C = ot.dist(Xi, Xj)
            if squared_cost:
                C = C ** 2

            val = ot.emd2(wi, vj, C)
            if squared_cost:
                val = np.sqrt(val)

            D[i, j] = D[j, i] = val

    Dn = _normalize_dist_matrix(D)
    return D, Dn


def compute_gw_distance_matrix(
    windows: list[tuple[int, np.ndarray]],
    data: np.ndarray,
    target_len: int,
    loss_fun: str = "square_loss",
    max_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Gromov-Wasserstein distance matrix between all window pairs.

    Args:
        windows: List of (clip_id, frame_indices) tuples
        data: Full data array (n_clips, n_frames, n_dim)
        target_len: Number of frames per window (for uniform weights)
        loss_fun: GW loss function ("square_loss" or "kl_loss")
        max_iter: Maximum iterations for GW solver

    Returns:
        D: Raw GW distance matrix (M, M)
        Dn: Normalized GW distance matrix (M, M)
    """
    if not HAS_OT:
        raise ImportError("POT library not installed")

    M = len(windows)
    D = np.zeros((M, M), dtype=float)

    print(f"Computing GW distance matrix for {M} windows...")
    print(f"  loss_fun={loss_fun}, max_iter={max_iter}")

    structure_matrices = []
    for clip_id, frames in tqdm(windows, desc="Precomputing structure matrices"):
        Xi = data[clip_id, frames]
        Ci = ot.dist(Xi, Xi)
        structure_matrices.append(Ci)

    w = np.full(target_len, 1.0 / target_len)

    for i in tqdm(range(M), desc="GW distances"):
        Ci = structure_matrices[i]
        for j in range(i + 1, M):
            Cj = structure_matrices[j]
            val = ot.gromov.gromov_wasserstein2(
                Ci, Cj, w, w,
                loss_fun=loss_fun,
                max_iter=max_iter,
            )
            D[i, j] = D[j, i] = val

    Dn = _normalize_dist_matrix(D)
    return D, Dn


def labels_to_cluster_dict(
    labels: np.ndarray,
    windows: list[tuple[int, np.ndarray]],
    clip_indices: list[int] | None = None,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    cluster_dict = {}
    for lbl, (local_clip_id, frames) in zip(labels, windows):
        lbl = int(lbl)
        if lbl < 0:
            continue
        abs_clip_id = clip_indices[local_clip_id] if clip_indices else local_clip_id
        cluster_dict.setdefault(lbl, []).append((abs_clip_id, frames))
    return cluster_dict


def generate_labels_yaml(
    method_name: str,
    n_clusters: int,
    seed: int,
    output_dir: str,
    experiment_id: str = None,
) -> str:
    if experiment_id is None:
        from datetime import datetime
        experiment_id = f"experiment_{datetime.now().strftime('%Y-%m-%d')}"

    labels_path = os.path.join(output_dir, f"labels_seed_{seed}.yaml")

    labels_content = {
        "experiment_id": experiment_id,
        "method": method_name,
        "n_clusters": n_clusters,
        "seed": seed,
        "labels": {i: "" for i in range(n_clusters)},
    }

    with open(labels_path, "w") as f:
        f.write("# AUTO-GENERATED - Fill in labels after watching videos\n")
        f.write("# Label format: single behavior (\"rear\"), transition (\"walk rear\"), compound (\"rear left_turn\")\n\n")
        yaml.dump(labels_content, f, default_flow_style=False, sort_keys=False)

    return labels_path


def compute_dtw_between_clusters(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    eval_data: np.ndarray,
    clip_indices: list[int],
    skip_root_dims: int = 7,
) -> tuple[np.ndarray, list[int]]:
    """
    Compute DTW distance matrix between cluster representatives.

    For each cluster, compute the mean trajectory (centroid) across all instances,
    then compute pairwise DTW distances between cluster centroids.

    Args:
        cluster_dict: Dict mapping cluster_id to list of (abs_clip_id, frame_indices)
        eval_data: Evaluation data array (n_clips, n_frames, n_dim)
        clip_indices: List of absolute clip indices
        skip_root_dims: Number of root dimensions to skip (default 7)

    Returns:
        dtw_matrix: (K, K) pairwise DTW distance matrix
        cluster_ids: List of cluster IDs in matrix order
    """
    if not HAS_TSLEARN:
        raise ImportError("tslearn library not installed. Install with: pip install tslearn")

    cluster_ids = sorted(cluster_dict.keys())
    K = len(cluster_ids)

    abs_to_local = {abs_idx: local_idx for local_idx, abs_idx in enumerate(clip_indices)}

    cluster_centroids = {}
    for cluster_id in cluster_ids:
        instances = cluster_dict[cluster_id]
        trajectories = []

        for abs_clip_id, frames in instances:
            if abs_clip_id not in abs_to_local:
                continue
            local_idx = abs_to_local[abs_clip_id]
            traj = eval_data[local_idx, frames, skip_root_dims:]
            trajectories.append(traj)

        if trajectories:
            min_len = min(t.shape[0] for t in trajectories)
            aligned_trajs = np.array([t[:min_len] for t in trajectories])
            centroid = aligned_trajs.mean(axis=0)
            cluster_centroids[cluster_id] = centroid

    dtw_matrix = np.zeros((K, K), dtype=float)

    for i, ci in enumerate(tqdm(cluster_ids, desc="Computing DTW matrix")):
        if ci not in cluster_centroids:
            continue
        for j in range(i + 1, K):
            cj = cluster_ids[j]
            if cj not in cluster_centroids:
                continue

            dist = tslearn_dtw(cluster_centroids[ci], cluster_centroids[cj])
            dtw_matrix[i, j] = dtw_matrix[j, i] = dist

    return dtw_matrix, cluster_ids


def identify_similar_cluster_pairs(
    dtw_matrix: np.ndarray,
    cluster_ids: list[int],
    threshold: float | None = None,
    percentile: float | None = None,
    top_k: int = 5,
) -> list[tuple[int, int, float]]:
    """
    Identify pairs of clusters that are similar based on DTW distance.

    Consistent with granularity/redundancy.py implementation.

    Args:
        dtw_matrix: (K, K) pairwise DTW distance matrix
        cluster_ids: List of cluster IDs in matrix order
        threshold: Fixed DTW threshold (pairs below this are similar)
        percentile: Percentile threshold (pairs below this percentile are similar)
        top_k: Return top-k most similar pairs

    Returns:
        List of (cluster_i, cluster_j, dtw_distance) tuples, sorted by distance
    """
    K = len(cluster_ids)
    pairs = []

    for i in range(K):
        for j in range(i + 1, K):
            dist = dtw_matrix[i, j]
            if np.isfinite(dist):
                pairs.append((cluster_ids[i], cluster_ids[j], dist))

    if not pairs:
        return []

    pairs.sort(key=lambda x: x[2])
    distances = [p[2] for p in pairs]

    if percentile is not None:
        threshold = np.percentile(distances, percentile)

    if threshold is None:
        return pairs[:top_k]

    similar = [p for p in pairs if p[2] < threshold]
    return similar[:top_k]


@dataclass
class ClusterMergingResult:
    dtw_matrix: np.ndarray
    cluster_ids: list[int]
    similar_pairs_fixed: list[tuple[int, int, float]]
    similar_pairs_percentile: list[tuple[int, int, float]]
    top_similar_pairs: list[tuple[int, int, float]]
    threshold_fixed: float | None
    threshold_percentile: float | None


def run_cluster_merging_analysis(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    eval_data: np.ndarray,
    clip_indices: list[int],
    config: ClusterMergingConfig,
) -> ClusterMergingResult:
    """
    Run cluster merging analysis to identify similar clusters.

    Consistent with granularity/redundancy.py - computes both fixed and percentile
    similar pairs separately.

    Args:
        cluster_dict: Dict mapping cluster_id to list of (abs_clip_id, frame_indices)
        eval_data: Evaluation data array (n_clips, n_frames, n_dim)
        clip_indices: List of absolute clip indices
        config: Cluster merging configuration

    Returns:
        ClusterMergingResult with DTW matrix and similar pairs
    """
    print("  Computing DTW distances between cluster centroids...")
    dtw_matrix, cluster_ids = compute_dtw_between_clusters(
        cluster_dict, eval_data, clip_indices, config.skip_root_dims
    )

    similar_pairs_fixed = identify_similar_cluster_pairs(
        dtw_matrix, cluster_ids,
        threshold=config.dtw_threshold,
        percentile=None,
        top_k=config.top_k_similar_pairs,
    )

    similar_pairs_percentile = identify_similar_cluster_pairs(
        dtw_matrix, cluster_ids,
        threshold=None,
        percentile=config.dtw_percentile,
        top_k=config.top_k_similar_pairs,
    )

    all_pairs = []
    K = len(cluster_ids)
    for i in range(K):
        for j in range(i + 1, K):
            d = dtw_matrix[i, j]
            if np.isfinite(d):
                all_pairs.append((cluster_ids[i], cluster_ids[j], d))
    all_pairs.sort(key=lambda x: x[2])
    top_similar = all_pairs[:config.top_k_similar_pairs]

    distances = [p[2] for p in all_pairs] if all_pairs else []
    threshold_percentile = np.percentile(distances, config.dtw_percentile) if distances else None

    print(f"  Fixed threshold ({config.dtw_threshold}): {len(similar_pairs_fixed)} similar pairs")
    threshold_str = f"{threshold_percentile:.4f}" if threshold_percentile is not None else "N/A"
    print(f"  Percentile threshold ({config.dtw_percentile}%={threshold_str}): {len(similar_pairs_percentile)} similar pairs")

    return ClusterMergingResult(
        dtw_matrix=dtw_matrix,
        cluster_ids=cluster_ids,
        similar_pairs_fixed=similar_pairs_fixed,
        similar_pairs_percentile=similar_pairs_percentile,
        top_similar_pairs=top_similar,
        threshold_fixed=config.dtw_threshold,
        threshold_percentile=threshold_percentile,
    )


def render_similar_cluster_pairs(
    similar_pairs: list[tuple[int, int, float]],
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    rodent_rollout_dir: str,
    output_dir: str,
    samples_per_cluster: int = 3,
    framerate: int = 30,
    target_size: tuple[int, int] = (320, 240),
) -> list[str]:
    """
    Render videos showing similar cluster pairs side by side.

    Creates a grid video where:
    - Rows: Different similar cluster pairs
    - Columns: Samples from each cluster (alternating between cluster i and j)

    Args:
        similar_pairs: List of (cluster_i, cluster_j, dtw_distance) tuples
        cluster_dict: Dict mapping cluster_id to list of (abs_clip_id, frame_indices)
        rodent_rollout_dir: Path to rodent rollout data
        output_dir: Output directory for videos
        samples_per_cluster: Number of samples per cluster
        framerate: Video framerate
        target_size: Target video size (width, height)

    Returns:
        List of rendered video paths
    """
    import cv2
    from topo_mimic.benchmark.analysis.dtw_visualization import render_frames_for_sequence

    os.makedirs(output_dir, exist_ok=True)
    rendered_videos = []

    if not similar_pairs:
        print("  No similar pairs to render")
        return rendered_videos

    grid_path = os.path.join(output_dir, "similar_clusters_grid.mp4")

    all_row_frames = []
    max_frames_per_row = 0

    for pair_idx, (ci, cj, dist) in enumerate(similar_pairs):
        instances_i = cluster_dict.get(ci, [])
        instances_j = cluster_dict.get(cj, [])

        if not instances_i or not instances_j:
            continue

        samples_i = instances_i[:samples_per_cluster]
        samples_j = instances_j[:samples_per_cluster]

        row_clips = []

        for sample_idx in range(samples_per_cluster):
            if sample_idx < len(samples_i):
                abs_clip_id, frames = samples_i[sample_idx]
                frames_i = render_frames_for_sequence(
                    abs_clip_id, frames, rodent_rollout_dir
                )
                row_clips.append(frames_i)
            else:
                row_clips.append(None)

            if sample_idx < len(samples_j):
                abs_clip_id, frames = samples_j[sample_idx]
                frames_j = render_frames_for_sequence(
                    abs_clip_id, frames, rodent_rollout_dir
                )
                row_clips.append(frames_j)
            else:
                row_clips.append(None)

        all_row_frames.append((ci, cj, dist, row_clips))

        for clip_frames in row_clips:
            if clip_frames is not None:
                max_frames_per_row = max(max_frames_per_row, len(clip_frames))

    if not all_row_frames:
        print("  No valid rows to render")
        return rendered_videos

    n_rows = len(all_row_frames)
    n_cols = samples_per_cluster * 2
    grid_width = n_cols * target_size[0]
    grid_height = n_rows * target_size[1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(grid_path, fourcc, framerate, (grid_width, grid_height))

    for frame_idx in range(max_frames_per_row):
        grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        for row_idx, (ci, cj, dist, row_clips) in enumerate(all_row_frames):
            row_y = row_idx * target_size[1]

            for col_idx, clip_frames in enumerate(row_clips):
                col_x = col_idx * target_size[0]

                if clip_frames is not None and frame_idx < len(clip_frames):
                    frame = clip_frames[frame_idx]
                    resized = cv2.resize(frame, target_size)
                    grid_frame[row_y:row_y + target_size[1], col_x:col_x + target_size[0]] = resized

            label = f"C{ci} vs C{cj} (DTW={dist:.3f})"
            cv2.putText(grid_frame, label, (5, row_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(grid_frame)

    out.release()
    rendered_videos.append(grid_path)
    print(f"  Rendered similar pairs grid: {grid_path}")

    return rendered_videos


def save_dtw_matrix_csv(
    dtw_matrix: np.ndarray,
    cluster_ids: list[int],
    output_path: str,
) -> None:
    """Save DTW distance matrix to CSV file."""
    import pandas as pd
    df = pd.DataFrame(
        dtw_matrix,
        index=[f"C{i}" for i in cluster_ids],
        columns=[f"C{i}" for i in cluster_ids],
    )
    df.to_csv(output_path)
    print(f"  Saved DTW matrix to {output_path}")


def plot_kpms_reconstruction(
    kpms_result: KPMSResult,
    output_dir: str,
    method_name: str,
    seed: int,
    n_dims: int = 5,
    n_clips: int = 3,
) -> dict[str, str]:
    """
    Generate reconstruction time series plots for KPMS results.

    Args:
        kpms_result: KPMSResult with original and reconstructed data
        output_dir: Directory to save plots
        method_name: Name of the method for plot titles
        seed: Random seed used
        n_dims: Number of dimensions to plot
        n_clips: Number of clips to generate plots for

    Returns:
        Dict mapping plot names to file paths
    """
    plot_paths = {}

    if kpms_result.original is None or kpms_result.reconstructed is None:
        print(f"  Warning: No reconstruction data available for {method_name}")
        return plot_paths

    os.makedirs(output_dir, exist_ok=True)

    orig = kpms_result.original
    recon = kpms_result.reconstructed

    if HAS_RECON_VIZ:
        for clip_idx in range(min(n_clips, len(orig))):
            try:
                fig = plot_reconstruction_timeseries(
                    original_data=orig,
                    reconstructed_data=recon,
                    clip_idx=clip_idx,
                    n_dims=min(n_dims, orig.shape[2]),
                    title_prefix=f"{method_name} Reconstruction",
                    max_frames=500,
                )
                plot_path = os.path.join(output_dir, f"reconstruction_clip{clip_idx}_seed{seed}.png")
                fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                plot_paths[f"reconstruction_clip{clip_idx}"] = plot_path
                print(f"  Saved reconstruction plot: {plot_path}")
            except Exception as e:
                print(f"  Warning: Failed to generate reconstruction plot for clip {clip_idx}: {e}")
    else:
        for clip_idx in range(min(n_clips, len(orig))):
            try:
                n_frames = min(orig.shape[1], recon.shape[1], 500)
                actual_n_dims = min(n_dims, orig.shape[2])
                t = np.arange(n_frames)

                fig, axes = plt.subplots(actual_n_dims, 1, figsize=(12, 2 * actual_n_dims), sharex=True)
                if actual_n_dims == 1:
                    axes = [axes]

                for dim_idx in range(actual_n_dims):
                    ax = axes[dim_idx]
                    orig_vals = orig[clip_idx, :n_frames, dim_idx]
                    recon_vals = recon[clip_idx, :n_frames, dim_idx]

                    ax.plot(t, orig_vals, label="Original", color="blue", alpha=0.7, linewidth=1)
                    ax.plot(t, recon_vals, label="Reconstructed", color="red", alpha=0.7, linewidth=1)

                    mse = np.mean((orig_vals - recon_vals) ** 2)
                    ax.set_ylabel(f"Dim {dim_idx}")
                    ax.set_title(f"Dimension {dim_idx} (MSE: {mse:.4f})", fontsize=10)
                    if dim_idx == 0:
                        ax.legend(loc="upper right")

                axes[-1].set_xlabel("Frame")
                fig.suptitle(f"{method_name} Reconstruction - Clip {clip_idx} (seed={seed})", fontsize=12)
                plt.tight_layout()

                plot_path = os.path.join(output_dir, f"reconstruction_clip{clip_idx}_seed{seed}.png")
                fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                plot_paths[f"reconstruction_clip{clip_idx}"] = plot_path
                print(f"  Saved reconstruction plot: {plot_path}")
            except Exception as e:
                print(f"  Warning: Failed to generate reconstruction plot for clip {clip_idx}: {e}")

    if kpms_result.reconstruction_mse is not None:
        summary_path = os.path.join(output_dir, f"reconstruction_summary_seed{seed}.txt")
        with open(summary_path, "w") as f:
            f.write(f"Reconstruction Summary for {method_name} (seed={seed})\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Reconstruction MSE: {kpms_result.reconstruction_mse:.6f}\n")
            f.write(f"Original shape: {orig.shape}\n")
            f.write(f"Reconstructed shape: {recon.shape}\n")
        plot_paths["summary"] = summary_path

    return plot_paths


def load_keypoint_data(
    rodent_rollout_dir: str,
    split: str,
    n_clips: int | None,
    keypoint_cfg: KeypointConfig,
    total_clips: int = 842,
    test_size: int = 200,
) -> tuple[np.ndarray, list[str]]:
    """Load qpos data and convert to keypoints using MuJoCo model."""
    if not HAS_KP_LOADER:
        raise ImportError("Keypoint loader not available. Install jax and mujoco.")

    qpos_data, clip_indices = load_data("qpos", rodent_rollout_dir, split, n_clips,
                                        total_clips=total_clips, test_size=test_size)

    print("Setting up MuJoCo model for keypoint conversion...")
    mjx_model, mjx_data, site_ids, kp_names = setup_stac_reference(
        keypoint_cfg.xml_path, keypoint_cfg.stac_data_path
    )
    print(f"Found {len(site_ids)} keypoint sites")

    print("Converting qpos to keypoints...")
    keypoint_data = qpos_to_keypoints(qpos_data, mjx_model, mjx_data, site_ids)
    print(f"Keypoints shape: {keypoint_data.shape}")

    return keypoint_data, kp_names


def run_rendering(
    cluster_dict: dict,
    rodent_rollout_dir: str,
    rendering_cfg: RenderingConfig,
    output_dir: str,
    cluster_labels: dict[int, str] | None = None,
    output_filename: str = "cluster_side_by_side.mp4",
    title: str = "Cluster Samples",
) -> None:
    cluster_chunks, cluster_clip_ids = chunk_loops_by_clustered_vpath(
        cluster_dict,
        rodent_rollout_dir,
        pad=rendering_cfg.pad,
        sample=rendering_cfg.max_samples,
        return_clip_ids=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)

    render_grid_clusters_side_by_side(
        cluster_chunks=cluster_chunks,
        out_path=out_path,
        framerate=rendering_cfg.framerate,
        max_samples_per_cluster=rendering_cfg.max_samples,
        target_size=rendering_cfg.target_size,
        title=title,
        per_tile_labels=True,
        selected_clusters=sorted(cluster_dict.keys()),
        cluster_labels=cluster_labels,
        cluster_clip_ids=cluster_clip_ids,
    )
    print(f"Video saved to: {out_path}")


def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("Unified Benchmark Experiment")
    print("=" * 60)
    print(f"Output dir: {config.output_dir}")
    print(f"n_clusters: {config.k_selection.n_clusters}")
    print(f"Seeds: {config.seeds}")
    print(f"DM sources: {list(config.dm_sources.keys())}")

    preset = get_dataset_preset(config.dataset)

    print(f"\n=== Stage 1: Loading {config.data.evaluation_key} data ===")
    eval_data, clip_indices = load_data(
        config.data.evaluation_key, config.rodent_rollout_dir, config.split, config.n_clips,
        total_clips=preset.total_clips, test_size=preset.test_size,
    )
    print(f"Loaded {len(clip_indices)} clips: {clip_indices[0]}-{clip_indices[-1]}")

    print("\n=== Stage 2: Running methods by DM source ===")
    enabled_methods = [m for m in config.methods if m.enabled]
    print(f"Enabled methods: {[m.name for m in enabled_methods]}")

    source_groups: dict[str, list[MethodEntry]] = {}
    state_space_methods: list[MethodEntry] = []
    for m in enabled_methods:
        method_obj = get_method(m.name, m.params)
        if method_obj.method_type == MethodType.STATE_SPACE:
            state_space_methods.append(m)
        else:
            source_groups.setdefault(m.dm_source, []).append(m)

    results_path = os.path.join(config.output_dir, "all_results.pkl")
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            all_results = pkl.load(f)
        print(f"Loaded existing results: {list(all_results.keys())}")
    else:
        all_results = {}

    from datetime import datetime
    experiment_id = f"experiment_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"

    for source_name, source_methods in source_groups.items():
        source_cfg = config.dm_sources.get(source_name)
        if source_cfg is None:
            raise ValueError(f"DM source '{source_name}' not found in dm_sources config")

        print(f"\n{'='*60}")
        print(f"DM Source: {source_name} (dm_key={source_cfg.dm_key})")
        print(f"{'='*60}")

        source_data, source_clip_indices = load_source_data(
            source_cfg, config.rodent_rollout_dir, config.split, config.n_clips,
            dataset=config.dataset,
        )

        min_frames = min(source_data.shape[1], eval_data.shape[1])
        source_data = source_data[:, :min_frames, :]

        if not source_cfg.results_dir:
            raise ValueError(f"results_dir required for DM source '{source_name}'")
        cycles_path = os.path.join(source_cfg.results_dir, "all_cycle_dm.pkl")
        with open(cycles_path, "rb") as f:
            all_cycle_dm = pkl.load(f)
        print(f"Loaded {len(all_cycle_dm)} cycle sets from {source_cfg.results_dir}")

        windows, cleaned_loops = prepare_windows_for_clustering(
            all_cycle_dm, source_data, config.clustering, clip_indices
        )
        print(f"Prepared {len(windows)} windows (from topological cycles)")

        if len(windows) == 0:
            print(f"WARNING: No valid windows for source '{source_name}', skipping")
            continue

        X = build_feature_matrix(windows, source_data)
        print(f"Feature matrix: {X.shape}")

        baseline_names = {"w2_unif_kmeans", "w2_full_kmeans"}
        unif_needed = any(m.name == "w2_unif_kmeans" for m in source_methods)
        full_needed = any(m.name == "w2_full_kmeans" for m in source_methods)

        windows_unif = None
        X_unif = None
        if unif_needed:
            windows_unif = prepare_unif_frames_for_clustering(source_data, config.clustering, clip_indices)
            X_unif = build_feature_matrix(windows_unif, source_data)
            print(f"Unif baseline feature matrix: {X_unif.shape}")

        windows_full = None
        X_full = None
        if full_needed:
            windows_full = prepare_full_frames_for_clustering(source_data, config.clustering, clip_indices)
            X_full = build_feature_matrix(windows_full, source_data)
            print(f"Full baseline feature matrix: {X_full.shape}")

        w2_dm_methods = [m for m in source_methods if m.name not in baseline_names and m.name != "gw_dmc_int_kmeans" and "dmp" not in m.name]
        Dn_w2 = None
        if w2_dm_methods:
            w2_cache_path = os.path.join(config.output_dir, f"w2_distance_matrix_{source_name}.pkl")
            if os.path.exists(w2_cache_path):
                print(f"Loading cached W2 distance matrix for {source_name}...")
                with open(w2_cache_path, "rb") as f:
                    w2_data = pkl.load(f)
                Dn_w2 = w2_data["Dn"]
            else:
                D_w2, Dn_w2 = compute_w2_distance_matrix(
                    windows, source_data, config.clustering.target_len
                )
                with open(w2_cache_path, "wb") as f:
                    pkl.dump({"D": D_w2, "Dn": Dn_w2}, f)
                print(f"Cached W2 matrix to {w2_cache_path}")

        Dn_w2_unif = None
        if unif_needed:
            w2_unif_cache = os.path.join(config.output_dir, f"w2_distance_matrix_unif_{source_name}.pkl")
            if os.path.exists(w2_unif_cache):
                with open(w2_unif_cache, "rb") as f:
                    w2_data = pkl.load(f)
                Dn_w2_unif = w2_data["Dn"]
            else:
                D_w2_u, Dn_w2_unif = compute_w2_distance_matrix(
                    windows_unif, source_data, config.clustering.target_len
                )
                with open(w2_unif_cache, "wb") as f:
                    pkl.dump({"D": D_w2_u, "Dn": Dn_w2_unif}, f)
                print(f"Cached W2 unif matrix to {w2_unif_cache}")

        Dn_w2_full = None
        if full_needed:
            w2_full_cache = os.path.join(config.output_dir, f"w2_distance_matrix_full_{source_name}.pkl")
            if os.path.exists(w2_full_cache):
                with open(w2_full_cache, "rb") as f:
                    w2_data = pkl.load(f)
                Dn_w2_full = w2_data["Dn"]
            else:
                D_w2_f, Dn_w2_full = compute_w2_distance_matrix(
                    windows_full, source_data, config.clustering.target_len
                )
                with open(w2_full_cache, "wb") as f:
                    pkl.dump({"D": D_w2_f, "Dn": Dn_w2_full}, f)
                print(f"Cached W2 full matrix to {w2_full_cache}")

        dmp_methods = [m for m in source_methods if "dmp" in m.name]
        windows_path = None
        X_path = None
        Dn_w2_path = None
        if dmp_methods:
            path_pkl_path = os.path.join(source_cfg.results_dir, "all_path_dm.pkl")
            if os.path.exists(path_pkl_path):
                with open(path_pkl_path, "rb") as f:
                    all_path_dm = pkl.load(f)
                print(f"Loaded {len(all_path_dm)} path sets from {source_cfg.results_dir}")
                windows_path, _ = prepare_windows_for_clustering(
                    all_path_dm, source_data, config.clustering, clip_indices
                )
                print(f"Prepared {len(windows_path)} path windows")
                if windows_path:
                    X_path = build_feature_matrix(windows_path, source_data)
                    w2_path_cache = os.path.join(config.output_dir, f"w2_distance_matrix_path_{source_name}.pkl")
                    if os.path.exists(w2_path_cache):
                        with open(w2_path_cache, "rb") as f:
                            w2_data = pkl.load(f)
                        Dn_w2_path = w2_data["Dn"]
                    else:
                        D_w2_p, Dn_w2_path = compute_w2_distance_matrix(
                            windows_path, source_data, config.clustering.target_len
                        )
                        with open(w2_path_cache, "wb") as f:
                            pkl.dump({"D": D_w2_p, "Dn": Dn_w2_path}, f)
                        print(f"Cached W2 path matrix to {w2_path_cache}")
            else:
                print(f"WARNING: all_path_dm.pkl not found at {source_cfg.results_dir}, dmp methods will be skipped")

        Dn_gw = None
        gw_methods = [m for m in source_methods if m.name == "gw_dmc_int_kmeans"]
        if gw_methods:
            gw_params = gw_methods[0].params
            loss_fun = gw_params.get("loss_fun", "square_loss")
            max_iter = gw_params.get("max_iter", 1000)
            gw_cache_path = os.path.join(config.output_dir, f"gw_distance_matrix_{source_name}.pkl")
            if os.path.exists(gw_cache_path):
                with open(gw_cache_path, "rb") as f:
                    gw_data = pkl.load(f)
                Dn_gw = gw_data["Dn"]
            else:
                D_gw, Dn_gw = compute_gw_distance_matrix(
                    windows, source_data, config.clustering.target_len,
                    loss_fun=loss_fun, max_iter=max_iter
                )
                with open(gw_cache_path, "wb") as f:
                    pkl.dump({"D": D_gw, "Dn": Dn_gw}, f)

        for method_entry in source_methods:
            print(f"\n  --- Method: {method_entry.name} (source={source_name}) ---")

            method_output_dir = os.path.join(config.output_dir, method_entry.name)
            os.makedirs(method_output_dir, exist_ok=True)

            method = get_method(method_entry.name, method_entry.params)
            method_results = {}

            if method_entry.name == "w2_unif_kmeans":
                method.params["distance_matrix"] = Dn_w2_unif
                current_windows = windows_unif
                current_X = X_unif
            elif method_entry.name == "w2_full_kmeans":
                method.params["distance_matrix"] = Dn_w2_full
                current_windows = windows_full
                current_X = X_full
            elif method_entry.name == "gw_dmc_int_kmeans":
                method.params["distance_matrix"] = Dn_gw
                current_windows = windows
                current_X = X
            elif "dmp" in method_entry.name:
                if Dn_w2_path is None or windows_path is None:
                    print(f"    Skipping {method_entry.name}: no path data available")
                    continue
                method.params["distance_matrix"] = Dn_w2_path
                current_windows = windows_path
                current_X = X_path
            else:
                method.params["distance_matrix"] = Dn_w2
                current_windows = windows
                current_X = X

            for seed in config.seeds:
                labels = method.fit_predict(current_X, config.k_selection.n_clusters, seed)
                cluster_dict = labels_to_cluster_dict(
                    labels, current_windows, clip_indices=clip_indices
                )

                method_results[seed] = {
                    "n_clusters": config.k_selection.n_clusters,
                    "cluster_dict": cluster_dict,
                    "labels": labels,
                }

                print(f"    Seed {seed}: complete")

            for seed in config.seeds:
                result_n_clusters = method_results[seed].get("n_clusters", config.k_selection.n_clusters)
                if config.behavior_labels_csv and Path(config.behavior_labels_csv).exists():
                    from topo_mimic.benchmark.auto_label import generate_auto_labels_yaml
                    generate_auto_labels_yaml(
                        method_name=method_entry.name,
                        n_clusters=result_n_clusters,
                        seed=seed,
                        output_dir=method_output_dir,
                        cluster_dict=method_results[seed]["cluster_dict"],
                        behavior_labels_csv=config.behavior_labels_csv,
                        experiment_id=experiment_id,
                    )
                else:
                    generate_labels_yaml(
                        method_name=method_entry.name,
                        n_clusters=result_n_clusters,
                        seed=seed,
                        output_dir=method_output_dir,
                        experiment_id=experiment_id,
                    )

            with open(os.path.join(method_output_dir, "results.pkl"), "wb") as f:
                pkl.dump(method_results, f)

            all_results[method_entry.name] = method_results

    dm_data_for_ss = None
    for method_entry in state_space_methods:
        print(f"\n{'='*60}")
        print(f"Method: {method_entry.name} (state-space)")
        print(f"{'='*60}")

        if dm_data_for_ss is None:
            hdf5_key = get_hdf5_key_for_dm_key(config.data.dm_key)
            dm_data_for_ss, _ = load_data(
                hdf5_key, config.rodent_rollout_dir, config.split, config.n_clips,
                total_clips=preset.total_clips, test_size=preset.test_size,
            )
            min_frames = min(dm_data_for_ss.shape[1], eval_data.shape[1])
            dm_data_for_ss = dm_data_for_ss[:, :min_frames, :]

        method_output_dir = os.path.join(config.output_dir, method_entry.name)
        os.makedirs(method_output_dir, exist_ok=True)

        method = get_method(method_entry.name, method_entry.params)
        method_results = {}

        n_clusters_to_use = config.k_selection.n_clusters
        skip_set = set(config.clustering.skip_clips)
        keep_indices = [i for i in range(dm_data_for_ss.shape[0]) if i not in skip_set]

        is_keypoint_method = method_entry.name == "kpms_keypoint"

        if is_keypoint_method:
            print(f"  Loading keypoint data for {method_entry.name}...")
            kp_data, kp_names = load_keypoint_data(
                config.rodent_rollout_dir, config.split, config.n_clips, config.keypoint,
                total_clips=preset.total_clips, test_size=preset.test_size,
            )
            kp_data_filtered = kp_data[keep_indices]
            print(f"  Filtered {kp_data.shape[0]} -> {kp_data_filtered.shape[0]} clips")
            data_for_method = kp_data_filtered
        else:
            dm_data_filtered = dm_data_for_ss[keep_indices]
            print(f"  Filtered {dm_data_for_ss.shape[0]} -> {dm_data_filtered.shape[0]} clips (skipped {len(skip_set)})")
            data_for_method = dm_data_filtered

        for seed in config.seeds:
            seed_dir = os.path.join(method_output_dir, f"seed_{seed}")

            if is_keypoint_method:
                kpms_result = method.fit(
                    data_for_method, n_clusters_to_use, seed_dir, seed,
                    kp_names=kp_names if 'kp_names' in locals() else None
                )
            else:
                kpms_result = method.fit(
                    data_for_method, n_clusters_to_use, seed_dir, seed
                )

            cluster_dict_filtered = kpms_result.transition_cluster_dict
            cluster_dict = {
                cluster_id: [
                    (clip_indices[keep_indices[clip_id]], frames)
                    for clip_id, frames in instances
                ]
                for cluster_id, instances in cluster_dict_filtered.items()
            }
            stats = kpms_result.transition_stats

            method_results[seed] = {
                "n_clusters": len(cluster_dict),
                "cluster_dict": cluster_dict,
                "labels": kpms_result.labels,
                "transition_stats": stats,
                "reconstruction_mse": kpms_result.reconstruction_mse,
            }

            if "n_clusters" in stats and "syllable_id_map" in stats:
                print(f"  Seed {seed}: {stats['n_clusters']} syllable-as-cluster groups, "
                      f"{stats.get('total_frames', 0)} total frames, "
                      f"{len(cluster_dict)} clusters retained")
            else:
                print(f"  Seed {seed}: {stats.get('n_sustained_segments', '?')} sustained segments, "
                      f"{stats.get('n_transition_types', '?')} transition types, "
                      f"{len(cluster_dict)} clusters")
            if kpms_result.reconstruction_mse is not None:
                print(f"  Seed {seed}: Reconstruction MSE = {kpms_result.reconstruction_mse:.6f}")

            print(f"\n  Generating reconstruction plots for seed {seed}...")
            plot_paths = plot_kpms_reconstruction(
                kpms_result=kpms_result,
                output_dir=seed_dir,
                method_name=method_entry.name,
                seed=seed,
                n_dims=5,
                n_clips=3,
            )
            method_results[seed]["reconstruction_plots"] = plot_paths

        for seed in config.seeds:
            result_n_clusters = method_results[seed].get("n_clusters", config.k_selection.n_clusters)
            if config.behavior_labels_csv and Path(config.behavior_labels_csv).exists():
                from topo_mimic.benchmark.auto_label import generate_auto_labels_yaml
                generate_auto_labels_yaml(
                    method_name=method_entry.name,
                    n_clusters=result_n_clusters,
                    seed=seed,
                    output_dir=method_output_dir,
                    cluster_dict=method_results[seed]["cluster_dict"],
                    behavior_labels_csv=config.behavior_labels_csv,
                    experiment_id=experiment_id,
                )
            else:
                generate_labels_yaml(
                    method_name=method_entry.name,
                    n_clusters=result_n_clusters,
                    seed=seed,
                    output_dir=method_output_dir,
                    experiment_id=experiment_id,
                )

        with open(os.path.join(method_output_dir, "results.pkl"), "wb") as f:
            pkl.dump(method_results, f)

        all_results[method_entry.name] = method_results

    if config.rendering.enabled:
        print("\n=== Stage 6: Rendering videos ===")
        for method_entry in enabled_methods:
            method_output_dir = os.path.join(config.output_dir, method_entry.name)
            method_results = all_results[method_entry.name]

            for seed in config.seeds:
                result = method_results[seed]
                result_n_clusters = result.get("n_clusters", config.k_selection.n_clusters)

                if "cluster_dict" in result:
                    cluster_dict = result["cluster_dict"]
                    cluster_labels = {i: f"C{i}" for i in cluster_dict.keys()}

                    run_rendering(
                        cluster_dict,
                        config.rodent_rollout_dir,
                        config.rendering,
                        method_output_dir,
                        cluster_labels=cluster_labels,
                        output_filename=f"clusters_k{result_n_clusters}_seed{seed}.mp4",
                        title=f"{method_entry.name} (k={result_n_clusters}, seed={seed})",
                    )

    if config.cluster_merging.enabled:
        print("\n=== Stage 7: Cluster Merging Analysis ===")
        for method_entry in enabled_methods:
            method_output_dir = os.path.join(config.output_dir, method_entry.name)
            method_results = all_results[method_entry.name]

            for seed in config.seeds:
                result = method_results[seed]
                if "cluster_dict" not in result:
                    continue

                cluster_dict = result["cluster_dict"]
                print(f"\n  {method_entry.name} (seed={seed}):")

                merging_result = run_cluster_merging_analysis(
                    cluster_dict, eval_data, clip_indices, config.cluster_merging
                )

                merging_dir = os.path.join(method_output_dir, f"cluster_merging_seed{seed}")
                os.makedirs(merging_dir, exist_ok=True)

                save_dtw_matrix_csv(
                    merging_result.dtw_matrix,
                    merging_result.cluster_ids,
                    os.path.join(merging_dir, "dtw_matrix.csv"),
                )

                similar_pairs_path = os.path.join(merging_dir, "similar_pairs.yaml")
                with open(similar_pairs_path, "w") as f:
                    yaml.dump({
                        "threshold_fixed": merging_result.threshold_fixed,
                        "threshold_percentile": merging_result.threshold_percentile,
                        "similar_pairs_fixed": [
                            {"cluster_i": int(ci), "cluster_j": int(cj), "dtw_distance": float(d)}
                            for ci, cj, d in merging_result.similar_pairs_fixed
                        ],
                        "similar_pairs_percentile": [
                            {"cluster_i": int(ci), "cluster_j": int(cj), "dtw_distance": float(d)}
                            for ci, cj, d in merging_result.similar_pairs_percentile
                        ],
                        "top_similar_pairs": [
                            {"cluster_i": int(ci), "cluster_j": int(cj), "dtw_distance": float(d)}
                            for ci, cj, d in merging_result.top_similar_pairs
                        ],
                    }, f, default_flow_style=False)
                print(f"    Saved similar pairs to {similar_pairs_path}")

                pairs_to_render = merging_result.top_similar_pairs
                if config.cluster_merging.render_similar_pairs and pairs_to_render:
                    rendered = render_similar_cluster_pairs(
                        pairs_to_render,
                        cluster_dict,
                        config.rodent_rollout_dir,
                        merging_dir,
                        samples_per_cluster=config.cluster_merging.samples_per_cluster,
                        framerate=config.rendering.framerate,
                        target_size=config.rendering.target_size,
                    )
                    result["cluster_merging"] = {
                        "similar_pairs_fixed": merging_result.similar_pairs_fixed,
                        "similar_pairs_percentile": merging_result.similar_pairs_percentile,
                        "top_similar_pairs": merging_result.top_similar_pairs,
                        "dtw_matrix": merging_result.dtw_matrix,
                        "cluster_ids": merging_result.cluster_ids,
                        "rendered_videos": rendered,
                    }
    else:
        print("\n=== Stage 7: Skipping Cluster Merging (disabled) ===")

    print("\n=== Stage 8: Saving summary ===")
    summary_path = os.path.join(config.output_dir, "experiment_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Unified Benchmark Experiment Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Experiment ID: {experiment_id}\n")
        f.write(f"n_clusters: {config.k_selection.n_clusters}\n")
        f.write(f"Seeds: {config.seeds}\n\n")

        f.write("Results by Method:\n")
        f.write("-" * 40 + "\n")

        for method_name, results in all_results.items():
            f.write(f"\n{method_name}:\n")
            for seed in config.seeds:
                if seed in results:
                    r = results[seed]
                    k = r.get("n_clusters", "N/A")
                    f.write(f"  Seed {seed}: k={k}, labels_file=labels_seed_{seed}.yaml\n")

    print(f"Saved summary to {summary_path}")

    with open(results_path, "wb") as f:
        pkl.dump(all_results, f)
    print(f"Saved all results to {results_path} (methods: {list(all_results.keys())})")

    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)

    return ExperimentResults(results=all_results, config=config)


def main(config_path: str = "configs/benchmark/experiment_transitions.yaml"):
    if os.path.exists(config_path):
        cfg = load_config(config_path)
    else:
        cfg = ExperimentConfig()

    run_experiment(cfg)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/benchmark/experiment_transitions.yaml"
    main(config_path)
