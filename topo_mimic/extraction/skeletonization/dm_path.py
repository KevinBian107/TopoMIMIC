"""DM minima utilities: parse persistence/weights and find persistent density minima.

The minima-to-minima DM-path definition was removed. Shipped DM-paths are the
bridge-corridors from ``skeletonization.extract_all_corridors``.
"""

import os


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
