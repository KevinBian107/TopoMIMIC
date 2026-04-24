"""DM-path extraction: edge-disjoint paths between persistent density minima."""

import os
from dataclasses import dataclass

import networkx as nx
import numpy as np

from .remap import parse_swr_remap, remap_cycle_to_original, split_cycle_into_segments


@dataclass
class DMPathResult:
    min1_remapped: int | None
    min2_remapped: int | None
    paths_remapped: list[np.ndarray]
    topology_type: str
    edge_connectivity: int


def _parse_sorted_weights(output_dir: str) -> dict[int, float]:
    weights_path = os.path.join(output_dir, "sorted-weights.txt")
    weights = {}
    with open(weights_path) as f:
        for i, line in enumerate(f):
            weights[i] = float(line.strip())
    return weights


def _parse_persistence_pairs(output_dir: str) -> list[tuple[int, int, int, float]]:
    edge_path = os.path.join(output_dir, "edge_for_morse_only.txt")
    pairs = []
    with open(edge_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            v0, v1, pair_type = int(parts[0]), int(parts[1]), int(parts[2])
            pers = float("inf") if parts[3] == "inf" else float(parts[3])
            pairs.append((v0, v1, pair_type, pers))
    return pairs


def _find_persistent_minima(
    pairs: list[tuple[int, int, int, float]],
    weights: dict[int, float],
    top_k: int = 2,
) -> list[int]:
    type0_pairs = [(v0, v1, pers) for v0, v1, t, pers in pairs if t == 0]
    type0_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    seen = set()
    minima = []
    for v0, v1, pers in type0_pairs:
        w0 = weights.get(v0, float("inf"))
        w1 = weights.get(v1, float("inf"))
        min_v = v0 if w0 < w1 else v1
        if min_v not in seen:
            minima.append(min_v)
            seen.add(min_v)
        if len(minima) >= top_k:
            break
    return minima


def find_dm_minima(output_dir: str) -> tuple[int | None, int | None]:
    """
    Find the two most persistent density minima from DM output files.

    Args:
        output_dir: Directory containing DM output files
            (sorted-weights.txt, edge_for_morse_only.txt)

    Returns:
        Tuple of (min1, min2) remapped vertex indices, or None if not found
    """
    pairs_path = os.path.join(output_dir, "edge_for_morse_only.txt")
    weights_path = os.path.join(output_dir, "sorted-weights.txt")

    if not os.path.exists(pairs_path) or not os.path.exists(weights_path):
        return None, None

    pairs = _parse_persistence_pairs(output_dir)
    weights = _parse_sorted_weights(output_dir)
    minima = _find_persistent_minima(pairs, weights, top_k=2)

    min1 = minima[0] if len(minima) >= 1 else None
    min2 = minima[1] if len(minima) >= 2 else None
    return min1, min2


def extract_dm_paths(
    G_dm: nx.Graph,
    output_dir: str,
    remap_to_orig: dict[int, int],
    n_takens: int,
    gap_limit: int = 50,
    min_segment_len: int = 50,
) -> DMPathResult:
    """
    Extract edge-disjoint paths between the two most persistent density minima.

    Uses the same remap pipeline as cycles: parse_swr_remap -> remap_cycle_to_original
    -> split_cycle_into_segments.

    Args:
        G_dm: DM graph (NetworkX)
        output_dir: Directory containing DM persistence files
        remap_to_orig: Mapping from remapped to original indices
        n_takens: Number of Takens-embedded points
        gap_limit: Max gap between consecutive indices within a segment
        min_segment_len: Minimum segment length to keep

    Returns:
        DMPathResult with minima, paths, topology type, and edge connectivity
    """
    min1, min2 = find_dm_minima(output_dir)

    if min1 is None or min2 is None:
        return DMPathResult(
            min1_remapped=min1,
            min2_remapped=min2,
            paths_remapped=[],
            topology_type="insufficient_minima",
            edge_connectivity=0,
        )

    if min1 not in G_dm or min2 not in G_dm:
        return DMPathResult(
            min1_remapped=min1,
            min2_remapped=min2,
            paths_remapped=[],
            topology_type="minima_not_in_graph",
            edge_connectivity=0,
        )

    ec = nx.edge_connectivity(G_dm, min1, min2)

    try:
        raw_paths = list(nx.edge_disjoint_paths(G_dm, min1, min2))
    except (nx.NetworkXError, nx.NetworkXNoPath, nx.NetworkXUnfeasible):
        raw_paths = []

    paths_remapped = []
    for p in raw_paths:
        remapped = remap_cycle_to_original(p, remap_to_orig, n_takens)
        if len(remapped) > 0:
            segs = split_cycle_into_segments(remapped, gap_limit=gap_limit)
            for seg in segs:
                if len(seg) >= min_segment_len:
                    paths_remapped.append(seg)

    topology_type = "theta" if ec >= 2 else "single-path"

    return DMPathResult(
        min1_remapped=min1,
        min2_remapped=min2,
        paths_remapped=paths_remapped,
        topology_type=topology_type,
        edge_connectivity=ec,
    )
