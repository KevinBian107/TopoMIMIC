"""
Labeling core (2026-04-18 rewrite).

Replaces the previous Wasserstein-distance pipeline with ground-truth majority
voting. See ``topo_mimic/experiment/auto_label.py`` for the canonical labeling
algorithm; this module is the thin adapter that builds
``LabelRecommendation`` objects from that algorithm's output so the GUI and
downstream CSVs keep working.

The old W2 code path (``_compute_w2``, ``compute_cluster_to_gt_distances``,
``extract_gt_transition_windows``, ``GTLibrary``, ``GTStatistics``) has been
deleted. The current rodent CSV is the only source of truth for transition
types.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from topo_mimic.experiment.auto_label import (
    BehaviorSegment,
    ClusterLabel,
    auto_label_clusters,
    load_behavior_segments,
)


@dataclass
class LabelRecommendation:
    """Adapter object for the GUI. ``labels`` is a list of (label, rank) pairs
    where rank = 0 is the majority vote's primary label and subsequent entries
    are alternatives in alphabetical order. ``rank`` is used in the GUI slot
    that previously held a W2 distance; lower = more preferred."""
    cluster_id: int
    labels: list[tuple[str, float]]


def build_recommendations(
    cluster_labels: dict[int, ClusterLabel],
) -> dict[int, LabelRecommendation]:
    recommendations: dict[int, LabelRecommendation] = {}
    for cid, cl in cluster_labels.items():
        entries: list[tuple[str, float]] = []
        if cl.primary:
            entries.append((cl.primary, 0.0))
        for rank, alt in enumerate(cl.alternatives, start=1):
            entries.append((alt, float(rank)))
        if not entries:
            entries.append(("", 0.0))
        recommendations[cid] = LabelRecommendation(cluster_id=cid, labels=entries)
    return recommendations


def print_recommendations(
    recommendations: dict[int, LabelRecommendation],
    method: str,
    seed: int,
) -> None:
    """Print a one-line-per-cluster summary to stdout."""
    print(f"\n=== Auto-labels for {method} (seed {seed}) ===")
    for cid in sorted(recommendations.keys()):
        rec = recommendations[cid]
        primary = rec.labels[0][0] if rec.labels else ""
        alts = [l for l, _ in rec.labels[1:]]
        alt_str = f" (alts: {alts})" if alts else ""
        print(f"  cluster {cid}: {primary!r}{alt_str}")


def save_recommendations_csv(
    recommendations: dict[int, LabelRecommendation],
    output_path: str,
) -> None:
    """CSV schema: cluster_id, rank, label. Distance column dropped under the
    new GT-majority pipeline; rank stands in."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "rank", "label"])
        for cid in sorted(recommendations.keys()):
            for rank, (label, _) in enumerate(recommendations[cid].labels):
                writer.writerow([cid, rank, label])


def write_minimal_stats_csv(
    cluster_labels: dict[int, ClusterLabel],
    output_path: str,
) -> None:
    """Replacement for the old GT-library statistics CSV: per-cluster vote
    counts and member counts."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "primary", "alternatives", "n_members", "vote_counts"])
        for cid in sorted(cluster_labels.keys()):
            cl = cluster_labels[cid]
            alts = "|".join(cl.alternatives) if cl.alternatives else ""
            vote_str = "|".join(f"{k}:{v}" for k, v in cl.vote_counts.items())
            writer.writerow([cid, cl.primary, alts, len(cl.per_clip_labels), vote_str])


__all__ = [
    "LabelRecommendation",
    "BehaviorSegment",
    "ClusterLabel",
    "auto_label_clusters",
    "load_behavior_segments",
    "build_recommendations",
    "print_recommendations",
    "save_recommendations_csv",
    "write_minimal_stats_csv",
]
