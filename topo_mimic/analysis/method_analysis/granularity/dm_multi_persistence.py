"""Multi-persistence DM extraction with caching."""

import os
import pickle as pkl
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
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
    all_path_dm: list[list[np.ndarray]] | None = None
    path_stats: dict | None = None


class _LegacyAwareUnpickler(pkl.Unpickler):
    """Resolve legacy module paths used by older granularity caches."""
    _RENAMES = {
        "topo_mimic.granularity": "topo_mimic.analysis.method_analysis.granularity",
    }
    def find_class(self, module: str, name: str):
        for old, new in self._RENAMES.items():
            if module == old or module.startswith(old + "."):
                module = new + module[len(old):]
                break
        return super().find_class(module, name)


def _load_cached_dm(cache_path: Path) -> DMExtractionResult | None:
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            try:
                return _LegacyAwareUnpickler(f).load()
            except Exception:
                f.seek(0)
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
) -> tuple[list[np.ndarray], nx.Graph, str]:
    """
    Run DM extraction for a single clip.

    Args:
        points: Takens-embedded point cloud (N, D)
        k: Number of neighbors for graph construction
        pers_delta: Persistence threshold
        epsilon: Sparse Rips approximation parameter

    Returns:
        Tuple of (cycles, graph, output_dir) from DM extraction.
        Caller is responsible for cleaning up output_dir.
    """
    from topo_mimic.extraction.skeletonization.skeletonization import extract_skeleton_pcd_dm

    clip_output_dir = tempfile.mkdtemp()
    cycles, G = extract_skeleton_pcd_dm(
        points,
        k=k,
        pers_delta=pers_delta,
        epsilon=epsilon,
        output_dir=clip_output_dir,
        cleanup=False,
    )
    return cycles, G, clip_output_dir


def extract_dm_for_persistence(
    intention_data: np.ndarray,
    pers_delta: float,
    config: GranularityConfig,
    clip_indices: list[int],
) -> DMExtractionResult:
    """
    Extract DM cycles (and optionally paths) for all clips at a single persistence value.

    Args:
        intention_data: (n_clips, n_frames, D) intention array
        pers_delta: Persistence threshold value
        config: Configuration object
        clip_indices: List of clip indices to process

    Returns:
        DMExtractionResult with cycles, graphs, and optionally paths for all clips
    """
    from topo_mimic.extraction.skeletonization.preprocess import takens_points
    from topo_mimic.extraction.skeletonization.remap import (
        parse_swr_remap,
        remap_cycle_to_original,
        split_cycle_into_segments,
    )

    use_dm_path = getattr(config.dm, "use_dm_path", False)
    if use_dm_path:
        from topo_mimic.extraction.skeletonization.dm_path import extract_dm_paths

    all_cycle_dm = []
    all_path_dm = [] if use_dm_path else None
    all_G_sm = []
    n_total_segments = 0
    segment_lengths = []
    n_total_paths = 0
    path_lengths = []

    segment_gap_limit = getattr(config.dm, "segment_gap_limit", 50)
    min_segment_len = getattr(config.dm, "min_segment_len", 50)

    n_clips = len(intention_data)
    for local_idx in tqdm(range(n_clips), desc=f"DM extraction (pers={pers_delta})"):
        clip_intention = intention_data[local_idx]
        points = takens_points(clip_intention, m=config.takens.m, tau=config.takens.tau)

        if len(points) < 10:
            all_cycle_dm.append([])
            all_G_sm.append(nx.Graph())
            if use_dm_path:
                all_path_dm.append([])
            continue

        clip_output_dir = None
        try:
            cycles, G, clip_output_dir = compute_dm_for_clip(
                points,
                k=config.dm.k_neighbors,
                pers_delta=pers_delta,
                epsilon=config.dm.epsilon,
            )
            all_G_sm.append(G)

            remap = parse_swr_remap(clip_output_dir)
            n_takens = len(points)

            remapped_cycles = []
            for cycle in cycles:
                cycle_arr = np.array(cycle, dtype=int)
                orig = remap_cycle_to_original(cycle_arr, remap, n_takens)
                if len(orig) > 0:
                    remapped_cycles.append(orig)

            if remapped_cycles:
                longest = max(remapped_cycles, key=len)
                segs = split_cycle_into_segments(longest, gap_limit=segment_gap_limit)
                segments = [seg for seg in segs if len(seg) >= min_segment_len]
            else:
                segments = []

            all_cycle_dm.append(segments)
            n_total_segments += len(segments)
            for seg in segments:
                segment_lengths.append(len(seg))

            if use_dm_path:
                path_result = extract_dm_paths(
                    G, clip_output_dir, remap, n_takens,
                    gap_limit=segment_gap_limit,
                    min_segment_len=min_segment_len,
                )
                all_path_dm.append(path_result.paths_remapped)
                n_total_paths += len(path_result.paths_remapped)
                for p in path_result.paths_remapped:
                    path_lengths.append(len(p))

        except Exception as e:
            print(f"  Warning: DM extraction failed for clip {clip_indices[local_idx]}: {e}")
            all_cycle_dm.append([])
            all_G_sm.append(nx.Graph())
            if use_dm_path:
                all_path_dm.append([])
        finally:
            if clip_output_dir is not None:
                shutil.rmtree(clip_output_dir, ignore_errors=True)

    cycle_stats = {
        "n_clips": n_clips,
        "n_total_cycles": n_total_segments,
        "n_clips_with_cycles": sum(1 for c in all_cycle_dm if len(c) > 0),
        "mean_cycle_length": float(np.mean(segment_lengths)) if segment_lengths else 0.0,
        "std_cycle_length": float(np.std(segment_lengths)) if segment_lengths else 0.0,
    }

    path_stats = None
    if use_dm_path:
        path_stats = {
            "n_clips": n_clips,
            "n_total_paths": n_total_paths,
            "n_clips_with_paths": sum(1 for p in all_path_dm if len(p) > 0),
            "mean_path_length": float(np.mean(path_lengths)) if path_lengths else 0.0,
            "std_path_length": float(np.std(path_lengths)) if path_lengths else 0.0,
        }

    return DMExtractionResult(
        pers_delta=pers_delta,
        all_cycle_dm=all_cycle_dm,
        all_G_sm=all_G_sm,
        cycle_stats=cycle_stats,
        all_path_dm=all_path_dm,
        path_stats=path_stats,
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
