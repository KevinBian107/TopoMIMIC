"""Multi-persistence DM extraction with caching."""

import os
import pickle as pkl
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from tqdm import tqdm

from .config import GranularityConfig


@dataclass
class DMExtractionResult:
    pers_delta: float
    all_cycle_dm: list[list[np.ndarray]]
    all_G_sm: list[nx.Graph]
    cycle_stats: dict


def _load_cached_dm(cache_path: Path) -> DMExtractionResult | None:
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pkl.load(f)
    return None


def _save_cached_dm(result: DMExtractionResult, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pkl.dump(result, f)


def compute_dm_for_clip(
    points: np.ndarray,
    k: int,
    pers_delta: float,
    epsilon: float,
) -> tuple[list[np.ndarray], nx.Graph]:
    """
    Run DM extraction for a single clip.

    Args:
        points: Takens-embedded point cloud (N, D)
        k: Number of neighbors for graph construction
        pers_delta: Persistence threshold
        epsilon: Sparse Rips approximation parameter

    Returns:
        Tuple of (cycles, graph) from DM extraction
    """
    from topo_mimic.dmwk.skeletonization.skeletonization import extract_skeleton_pcd_dm

    cycles, G = extract_skeleton_pcd_dm(
        points,
        k=k,
        pers_delta=pers_delta,
        epsilon=epsilon,
        cleanup=True,
    )
    return cycles, G


def extract_dm_for_persistence(
    intention_data: np.ndarray,
    pers_delta: float,
    config: GranularityConfig,
    clip_indices: list[int],
) -> DMExtractionResult:
    """
    Extract DM cycles for all clips at a single persistence value.

    Args:
        intention_data: (n_clips, n_frames, D) intention array
        pers_delta: Persistence threshold value
        config: Configuration object
        clip_indices: List of clip indices to process

    Returns:
        DMExtractionResult with cycles and graphs for all clips
    """
    from topo_mimic.dmwk.skeletonization.preprocess import takens_points

    all_cycle_dm = []
    all_G_sm = []
    n_total_cycles = 0
    cycle_lengths = []

    n_clips = len(intention_data)
    for local_idx in tqdm(range(n_clips), desc=f"DM extraction (pers={pers_delta})"):
        clip_intention = intention_data[local_idx]
        points = takens_points(clip_intention, m=config.takens.m, tau=config.takens.tau)

        if len(points) < 10:
            all_cycle_dm.append([])
            all_G_sm.append(nx.Graph())
            continue

        try:
            cycles, G = compute_dm_for_clip(
                points,
                k=config.dm.k_neighbors,
                pers_delta=pers_delta,
                epsilon=config.dm.epsilon,
            )
            all_cycle_dm.append(cycles)
            all_G_sm.append(G)

            n_total_cycles += len(cycles)
            for c in cycles:
                cycle_lengths.append(len(c))
        except Exception as e:
            print(f"  Warning: DM extraction failed for clip {clip_indices[local_idx]}: {e}")
            all_cycle_dm.append([])
            all_G_sm.append(nx.Graph())

    cycle_stats = {
        "n_clips": n_clips,
        "n_total_cycles": n_total_cycles,
        "n_clips_with_cycles": sum(1 for c in all_cycle_dm if len(c) > 0),
        "mean_cycle_length": float(np.mean(cycle_lengths)) if cycle_lengths else 0.0,
        "std_cycle_length": float(np.std(cycle_lengths)) if cycle_lengths else 0.0,
    }

    return DMExtractionResult(
        pers_delta=pers_delta,
        all_cycle_dm=all_cycle_dm,
        all_G_sm=all_G_sm,
        cycle_stats=cycle_stats,
    )


def load_or_compute_dm(
    intention_data: np.ndarray,
    pers_delta: float,
    config: GranularityConfig,
    clip_indices: list[int],
    cache_dir: Path,
) -> DMExtractionResult:
    """
    Load cached DM result or compute if not available.

    Args:
        intention_data: (n_clips, n_frames, D) intention array
        pers_delta: Persistence threshold value
        config: Configuration object
        clip_indices: List of clip indices
        cache_dir: Directory for caching results

    Returns:
        DMExtractionResult (from cache or freshly computed)
    """
    cache_path = cache_dir / f"pers_{pers_delta:.2f}" / "dm_result.pkl"

    cached = _load_cached_dm(cache_path)
    if cached is not None:
        print(f"  Loaded cached DM result for pers_delta={pers_delta}")
        return cached

    print(f"  Computing DM extraction for pers_delta={pers_delta}...")
    result = extract_dm_for_persistence(
        intention_data, pers_delta, config, clip_indices
    )

    _save_cached_dm(result, cache_path)
    print(f"  Cached DM result to {cache_path}")

    return result


def run_dm_multi_persistence(
    intention_data: np.ndarray,
    config: GranularityConfig,
    clip_indices: list[int],
) -> dict[float, DMExtractionResult]:
    """
    Run DM extraction across multiple persistence thresholds.

    Args:
        intention_data: (n_clips, n_frames, D) intention array
        config: Configuration object
        clip_indices: List of clip indices

    Returns:
        Dict mapping pers_delta value to DMExtractionResult
    """
    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")

    original_cwd = os.getcwd()

    cache_dir = Path(config.output_dir) / "dm_cache"
    if not cache_dir.is_absolute():
        cache_dir = Path(original_cwd) / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for pers_delta in config.multi_persistence.pers_delta_values:
        result = load_or_compute_dm(
            intention_data, pers_delta, config, clip_indices, cache_dir
        )
        results[pers_delta] = result

        os.chdir(original_cwd)

        stats = result.cycle_stats
        print(f"  pers={pers_delta}: {stats['n_total_cycles']} cycles, "
              f"{stats['n_clips_with_cycles']}/{stats['n_clips']} clips with cycles, "
              f"mean_len={stats['mean_cycle_length']:.1f}")

    os.chdir(original_cwd)
    return results
