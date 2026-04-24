"""Shared utility functions for granularity analysis."""

import numpy as np


JOINT_NAMES = [
    "vertebra_6_twist",      # 0
    "hip_L_supinate",        # 1
    "hip_L_abduct",          # 2
    "hip_L_extend",          # 3
    "knee_L",                # 4
    "ankle_L",               # 5
    "toe_L",                 # 6
    "hip_R_supinate",        # 7
    "hip_R_abduct",          # 8
    "hip_R_extend",          # 9
    "knee_R",                # 10
    "ankle_R",               # 11
    "toe_R",                 # 12
    "vertebra_C21_bend",     # 13
    "vertebra_C19_extend",   # 14
    "vertebra_C17_bend",     # 15
    "vertebra_C15_extend",   # 16
    "vertebra_C13_bend",     # 17
    "vertebra_C11_extend",   # 18
    "vertebra_C9_bend",      # 19
    "vertebra_C7_extend",    # 20
    "vertebra_C6_extend",    # 21
    "vertebra_C6_bend",      # 22
    "vertebra_C5_extend",    # 23
    "vertebra_C5_bend",      # 24
    "vertebra_C4_extend",    # 25
    "vertebra_C4_bend",      # 26
    "vertebra_C3_extend",    # 27
    "vertebra_C3_bend",      # 28
    "vertebra_C2_extend",    # 29
    "vertebra_C2_bend",      # 30
    "vertebra_C1_extend",    # 31
    "vertebra_C1_bend",      # 32
    "vertebra_5_bend",       # 33
    "vertebra_4_extend",     # 34
    "vertebra_3_twist",      # 35
    "vertebra_2_bend",       # 36
    "vertebra_1_extend",     # 37
    "vertebra_atlant_extend",# 38
    "atlas",                 # 39
    "mandible",              # 40
    "vertebra_axis_twist",   # 41
    "vertebra_cervical_1",   # 42
    "vertebra_cervical_2",   # 43
    "vertebra_cervical_3",   # 44
    "vertebra_cervical_4",   # 45
    "vertebra_cervical_5",   # 46
    "scapula_L_supinate",    # 47
    "scapula_L_abduct",      # 48
    "scapula_L_extend",      # 49
    "shoulder_L",            # 50
    "shoulder_sup_L",        # 51
    "elbow_L",               # 52
    "wrist_L",               # 53
    "finger_L",              # 54
    "scapula_R_supinate",    # 55
    "scapula_R_abduct",      # 56
    "scapula_R_extend",      # 57
    "shoulder_R",            # 58
    "shoulder_sup_R",        # 59
    "elbow_R",               # 60
    "wrist_R",               # 61
    "finger_R",              # 62
]


def resolve_joint_indices(
    joint_names: list[str],
    skip_root_dims: int = 7,
) -> tuple[list[int], list[str]]:
    """
    Resolve joint names to indices in qpos array (after skipping root dims).

    Args:
        joint_names: List of joint names to resolve
        skip_root_dims: Number of root dimensions to skip (default 7)

    Returns:
        Tuple of (joint_indices, resolved_names) where indices are into the
        qpos array after root dims are removed
    """
    joint_indices = []
    resolved_names = []

    for name in joint_names:
        if name in JOINT_NAMES:
            joint_idx = JOINT_NAMES.index(name)
            joint_indices.append(joint_idx)
            resolved_names.append(name)
        else:
            print(f"Warning: Joint '{name}' not found in JOINT_NAMES")

    return joint_indices, resolved_names


def compute_root_displacement(
    qpos_clip: np.ndarray,
    root_indices: list[int] = [0, 1, 2],
) -> float:
    """
    Compute total root displacement for a clip.

    Args:
        qpos_clip: (T, D) qpos array for one clip
        root_indices: Indices of root position (default [0, 1, 2] for x, y, z)

    Returns:
        Total displacement as sum of frame-to-frame distances
    """
    root_positions = qpos_clip[:, root_indices]
    displacements = np.diff(root_positions, axis=0)
    frame_distances = np.linalg.norm(displacements, axis=1)
    return float(np.sum(frame_distances))


class UnionFind:
    """Union-Find data structure for grouping redundant clusters."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def union_find_groups(
    pairs: list[tuple[int, int]],
    n_items: int,
) -> list[set[int]]:
    """
    Group items using union-find based on redundant pairs.

    Args:
        pairs: List of (i, j) tuples indicating redundant pairs
        n_items: Total number of items

    Returns:
        List of sets, each containing indices of items in the same group
    """
    uf = UnionFind(n_items)
    for i, j in pairs:
        uf.union(i, j)

    groups_dict: dict[int, set[int]] = {}
    for i in range(n_items):
        root = uf.find(i)
        if root not in groups_dict:
            groups_dict[root] = set()
        groups_dict[root].add(i)

    return list(groups_dict.values())


def count_unique_clusters(
    pairs: list[tuple[int, int]],
    n_items: int,
) -> int:
    """
    Count unique clusters after merging redundant pairs.

    Args:
        pairs: List of (i, j) tuples indicating redundant pairs
        n_items: Total number of items

    Returns:
        Number of unique clusters after merging
    """
    groups = union_find_groups(pairs, n_items)
    return len(groups)
