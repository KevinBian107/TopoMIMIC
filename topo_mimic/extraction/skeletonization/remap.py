"""Shared utilities for remapping DM graph indices to original Takens space."""

import os

import numpy as np


def parse_swr_remap(output_dir: str) -> dict[int, int]:
    """Parse sparse_weighted_rips_filtration.txt to build {remapped_idx: orig_idx} mapping.

    Args:
        output_dir: Directory containing DM output files

    Returns:
        Dict mapping weight-sorted remapped index to original point index
    """
    filt_path = os.path.join(output_dir, "sparse_weighted_rips_filtration.txt")
    remap_to_orig = {}
    vertex_count = 0

    with open(filt_path) as f:
        for line in f:
            parts = line.strip().split()
            dim = int(parts[0])
            if dim == 0:
                orig_idx = int(parts[1])
                remap_to_orig[vertex_count] = orig_idx
                vertex_count += 1

    return remap_to_orig


def remap_cycle_to_original(
    indices: np.ndarray | list[int],
    remap_to_orig: dict[int, int],
    n_takens: int,
) -> np.ndarray:
    """Convert remapped DM indices to original Takens indices, filtering invalid.

    Args:
        indices: Remapped DM graph indices
        remap_to_orig: Mapping from remapped to original indices
        n_takens: Number of Takens-embedded points (for bounds check)

    Returns:
        Array of valid original Takens indices
    """
    orig = [remap_to_orig[int(i)] for i in indices if int(i) in remap_to_orig]
    return np.array([i for i in orig if i < n_takens], dtype=int)


def split_cycle_into_segments(
    cycle_orig_indices: np.ndarray,
    gap_limit: int = 50,
) -> list[np.ndarray]:
    """Split sorted cycle indices into contiguous segments based on gap limit.

    Args:
        cycle_orig_indices: Original Takens indices (will be sorted internally)
        gap_limit: Max allowed gap between consecutive indices within a segment

    Returns:
        List of arrays, each a contiguous segment of sorted indices
    """
    sorted_idx = np.sort(cycle_orig_indices)
    if len(sorted_idx) == 0:
        return []

    diffs = np.diff(sorted_idx)
    split_points = np.where(diffs > gap_limit)[0] + 1
    segments = np.split(sorted_idx, split_points)
    return [seg for seg in segments if len(seg) > 0]
