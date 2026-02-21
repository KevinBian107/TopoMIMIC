'''Functions for post-processing, namely chunking and clustering'''

from typing import Callable, Any
import numpy as np
import random

from topo_mimic.dmwk.utils.rendering import render_from_saved_rollout
from topo_mimic.dmwk.utils.load_utils import load_rollout_from_source


def _expand_frames_nonconsec(
    loop_frames: np.ndarray | list[int],
    pad: int,
    clip_len: int,
) -> np.ndarray:
    if pad <= 0:
        return np.unique(np.asarray(loop_frames, dtype=int))
    chunks = [np.arange(max(0, f - pad), min(clip_len, f + pad + 1)) for f in np.asarray(loop_frames, dtype=int)]
    return np.unique(np.concatenate(chunks))


def _expand_indices(
    indices: list[int],
    pad: int,
    T: int,
) -> list[int]:
    idx = set()
    for i in indices:
        a = max(0, int(i) - pad)
        b = min(T - 1, int(i) + pad)
        idx.update(range(a, b + 1))
    return sorted(idx)


def _uniform_downsample(
    frames: np.ndarray | list[int],
    target_len: int,
    min_frames: int = 3,
) -> np.ndarray | None:
    """
    Resample frame indices to target_len via uniform sampling.

    If frames >= target_len: downsample by selecting evenly spaced indices.
    If frames < target_len: upsample by interpolating frame indices.

    Args:
        frames: Frame indices array
        target_len: Desired output length
        min_frames: Minimum frames required (returns None if fewer)

    Returns:
        Resampled frame indices of length target_len, or None if too few frames
    """
    frames = np.unique(np.asarray(frames, dtype=int))
    if len(frames) < min_frames:
        return None
    idx = np.linspace(0, len(frames) - 1, target_len)
    idx = np.rint(idx).astype(int)
    return frames[idx]


def _normalize_dist_matrix(
    D: np.ndarray,
) -> np.ndarray:
    d = D.astype(float).copy()
    np.fill_diagonal(d, np.nan)
    row_med = np.nanmedian(d, axis=1)
    col_med = np.nanmedian(d, axis=0)
    s = np.sqrt(np.outer(row_med, col_med))
    s[s == 0] = 1.0
    Dn = D / s
    Dn[np.isnan(Dn)] = 0.0
    np.fill_diagonal(Dn, 0.0)
    return Dn


def chunk_loops_by_clustered_vpath(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    rodent_rollout_dir: str,
    pad: int = 25,
    sample: int | None = None,
    rng: random.Random | None = None,
    return_clip_ids: bool = False,
) -> dict[int, list[list[np.ndarray]]] | tuple[dict[int, list[list[np.ndarray]]], dict[int, list[int]]]:
    """
    Build video-frame chunks for every behavior cluster.

    Args:
        cluster_dict: Dict mapping cluster_id to list of (clip_id, frame_indices)
        rodent_rollout_dir: Directory containing clip HDF5 files
        pad: Padding around each frame index
        sample: Maximum samples per cluster (None for all)
        rng: Random number generator for sampling
        return_clip_ids: Whether to also return clip IDs for each chunk

    Returns:
        Dict mapping cluster_id to list of frame chunks.
        If return_clip_ids=True, also returns dict mapping cluster_id to list of clip_ids.
    """
    if rng is None:
        rng = random

    cluster_chunks = {}
    cluster_clip_ids = {}

    for cid, items in cluster_dict.items():
        if sample is not None and len(items) > sample:
            items = rng.sample(items, sample)

        cluster_chunks[cid] = []
        cluster_clip_ids[cid] = []

        for clip_id, frames in items:
            rollout = load_rollout_from_source(clip_id, rodent_rollout_dir)
            full_frames = render_from_saved_rollout(rollout)
            T = len(full_frames)

            ordered = _expand_indices(frames, pad=pad, T=T) if pad and pad > 0 else sorted(int(f) for f in frames)
            chunk = [full_frames[i] for i in ordered]
            cluster_chunks[cid].append(chunk)
            cluster_clip_ids[cid].append(clip_id)

    if return_clip_ids:
        return cluster_chunks, cluster_clip_ids
    return cluster_chunks