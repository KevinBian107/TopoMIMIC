"""
Cluster alignment across seeds using Hungarian algorithm.

Aligns cluster IDs from different random seeds to a reference seed,
enabling consistent labeling and meaningful cross-seed stability analysis.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class AlignmentResult:
    mapping: dict[int, int]
    overlap_matrix: np.ndarray
    alignment_score: float
    per_cluster_scores: dict[int, float]


def compute_overlap_matrix(
    ref_labels: np.ndarray,
    target_labels: np.ndarray,
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Compute overlap matrix between reference and target cluster assignments.

    Args:
        ref_labels: Cluster labels from reference seed (N,)
        target_labels: Cluster labels from target seed (N,)

    Returns:
        Tuple of (overlap_matrix, ref_cluster_ids, target_cluster_ids)
    """
    ref_ids = sorted([x for x in np.unique(ref_labels) if x >= 0])
    tgt_ids = sorted([x for x in np.unique(target_labels) if x >= 0])

    overlap = np.zeros((len(ref_ids), len(tgt_ids)), dtype=int)

    for i, r in enumerate(ref_ids):
        ref_mask = ref_labels == r
        for j, t in enumerate(tgt_ids):
            tgt_mask = target_labels == t
            overlap[i, j] = np.sum(ref_mask & tgt_mask)

    return overlap, ref_ids, tgt_ids


def align_clusters_to_reference(
    ref_labels: np.ndarray,
    target_labels: np.ndarray,
) -> AlignmentResult:
    """
    Align target cluster IDs to reference using Hungarian algorithm.

    Args:
        ref_labels: Cluster labels from reference seed (N,)
        target_labels: Cluster labels from target seed (N,)

    Returns:
        AlignmentResult with mapping, overlap matrix, and scores
    """
    ref_labels = np.asarray(ref_labels)
    target_labels = np.asarray(target_labels)

    if len(ref_labels) != len(target_labels):
        raise ValueError(
            f"Label arrays must have same length: {len(ref_labels)} vs {len(target_labels)}"
        )

    overlap, ref_ids, tgt_ids = compute_overlap_matrix(ref_labels, target_labels)

    n_ref = len(ref_ids)
    n_tgt = len(tgt_ids)

    if n_ref == 0 or n_tgt == 0:
        return AlignmentResult(
            mapping={},
            overlap_matrix=overlap,
            alignment_score=0.0,
            per_cluster_scores={},
        )

    max_dim = max(n_ref, n_tgt)
    padded_overlap = np.zeros((max_dim, max_dim), dtype=int)
    padded_overlap[:n_ref, :n_tgt] = overlap

    row_ind, col_ind = linear_sum_assignment(-padded_overlap)

    mapping = {}
    per_cluster_scores = {}
    total_overlap = 0
    total_samples = len(ref_labels)

    for r_idx, t_idx in zip(row_ind, col_ind):
        if r_idx < n_ref and t_idx < n_tgt:
            ref_id = ref_ids[r_idx]
            tgt_id = tgt_ids[t_idx]
            mapping[tgt_id] = ref_id

            matched_overlap = overlap[r_idx, t_idx]
            total_overlap += matched_overlap

            ref_count = np.sum(ref_labels == ref_id)
            tgt_count = np.sum(target_labels == tgt_id)
            max_possible = max(ref_count, tgt_count)
            per_cluster_scores[ref_id] = matched_overlap / max_possible if max_possible > 0 else 0.0

    for t_idx, tgt_id in enumerate(tgt_ids):
        if tgt_id not in mapping:
            mapping[tgt_id] = -1

    alignment_score = total_overlap / total_samples if total_samples > 0 else 0.0

    return AlignmentResult(
        mapping=mapping,
        overlap_matrix=overlap,
        alignment_score=alignment_score,
        per_cluster_scores=per_cluster_scores,
    )


def remap_labels(
    labels: np.ndarray,
    mapping: dict[int, int],
) -> np.ndarray:
    """
    Remap label array according to alignment mapping.

    Args:
        labels: Original cluster labels (N,)
        mapping: Mapping from original IDs to aligned IDs

    Returns:
        Remapped labels array
    """
    labels = np.asarray(labels)
    remapped = np.full_like(labels, -1)

    for orig_id, new_id in mapping.items():
        if new_id >= 0:
            mask = labels == orig_id
            remapped[mask] = new_id

    return remapped


def remap_cluster_dict(
    cluster_dict: dict[int, list],
    mapping: dict[int, int],
) -> dict[int, list]:
    """
    Remap cluster_dict keys according to alignment mapping.

    Args:
        cluster_dict: Original cluster dict {cluster_id: [(clip_id, frames), ...]}
        mapping: Mapping from original IDs to aligned IDs

    Returns:
        Remapped cluster_dict with aligned keys
    """
    remapped = {}

    for orig_id, entries in cluster_dict.items():
        new_id = mapping.get(orig_id, -1)
        if new_id >= 0:
            remapped[new_id] = entries

    return remapped


def compute_alignment_summary(
    alignment_results: dict[int, AlignmentResult],
) -> dict:
    """
    Compute summary statistics across all seed alignments.

    Args:
        alignment_results: Dict mapping seed -> AlignmentResult

    Returns:
        Summary dict with mean/std alignment scores
    """
    if not alignment_results:
        return {"mean_score": 0.0, "std_score": 0.0, "min_score": 0.0, "n_seeds": 0}

    scores = [ar.alignment_score for ar in alignment_results.values()]

    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "n_seeds": len(scores),
        "per_seed_scores": {seed: ar.alignment_score for seed, ar in alignment_results.items()},
    }


def get_alignment_quality_label(score: float) -> str:
    """Return human-readable quality label for alignment score."""
    if score >= 0.9:
        return "excellent"
    elif score >= 0.7:
        return "good"
    elif score >= 0.5:
        return "moderate"
    else:
        return "poor"
