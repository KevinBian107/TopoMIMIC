"""Cluster diversity metrics for evaluating clustering quality."""

import os
from dataclasses import dataclass

import pandas as pd

from .behavior_parsing import parse_cluster_label, TurnDirection


@dataclass
class DiversityMetrics:
    """Container for cluster diversity metrics."""
    n_clusters: int
    unique_labels: int
    redundancy: float
    transition_types_found: set[tuple[str, str]]
    transition_type_coverage: float
    gt_transition_types: set[tuple[str, str]]
    clusters_with_turn: int
    turn_direction_sensitivity: float


def compute_cluster_diversity(
    cluster_labels: dict[int, str],
    gt_transition_types: set[tuple[str, str]] | None = None,
    merge_rear: bool = True,
) -> DiversityMetrics:
    """
    Compute diversity metrics for cluster labels.

    Args:
        cluster_labels: Dict mapping cluster_id to label string
        gt_transition_types: Set of (from_behavior, to_behavior) tuples from GT.
                            If None, coverage is computed against discovered types.
        merge_rear: Whether to merge Supported/Unsupported Rear

    Returns:
        DiversityMetrics with all computed metrics
    """
    n_clusters = len(cluster_labels)

    parsed_labels = []
    transition_types_found = set()
    clusters_with_turn = 0

    for cid, label in cluster_labels.items():
        parsed = parse_cluster_label(label, merge_rear=merge_rear)
        if parsed is None:
            continue

        parsed_labels.append(parsed)

        if parsed.comparison_type == "transition" and len(parsed.primitives) >= 2:
            p1, p2 = sorted(parsed.primitives[:2])
            transition_types_found.add((p1, p2))

        has_turn = (
            parsed.from_turn not in (None, TurnDirection.STRAIGHT) or
            parsed.to_turn not in (None, TurnDirection.STRAIGHT)
        )
        if has_turn:
            clusters_with_turn += 1

    unique_base_labels = set()
    for parsed in parsed_labels:
        if parsed.comparison_type == "transition" and len(parsed.primitives) >= 2:
            base = tuple(sorted(parsed.primitives[:2]))
        elif parsed.comparison_type == "single" and parsed.primitives:
            base = (parsed.primitives[0],)
        else:
            base = (parsed.original,)
        unique_base_labels.add(base)

    unique_labels = len(unique_base_labels)
    redundancy = 1.0 - (unique_labels / n_clusters) if n_clusters > 0 else 0.0

    if gt_transition_types:
        coverage = len(transition_types_found & gt_transition_types) / len(gt_transition_types)
    else:
        coverage = 1.0 if transition_types_found else 0.0

    turn_sensitivity = clusters_with_turn / n_clusters if n_clusters > 0 else 0.0

    return DiversityMetrics(
        n_clusters=n_clusters,
        unique_labels=unique_labels,
        redundancy=redundancy,
        transition_types_found=transition_types_found,
        transition_type_coverage=coverage,
        gt_transition_types=gt_transition_types or set(),
        clusters_with_turn=clusters_with_turn,
        turn_direction_sensitivity=turn_sensitivity,
    )


def get_gt_transition_types(
    behavior_labels_csv: str,
    merge_rear: bool = True,
) -> set[tuple[str, str]]:
    """
    Extract all unique transition types from GT behavior labels.

    Args:
        behavior_labels_csv: Path to behavior labels CSV
        merge_rear: Whether to merge Supported/Unsupported Rear

    Returns:
        Set of (from_behavior, to_behavior) tuples (sorted alphabetically)
    """
    from .ground_truth import find_all_transitions

    all_transitions = find_all_transitions(
        behavior_labels_csv,
        short_threshold=10,
        max_gap=10,
        merge_rear=merge_rear,
    )

    transition_types = set()
    for t in all_transitions:
        p1, p2 = sorted([t.from_behavior, t.to_behavior])
        transition_types.add((p1, p2))

    return transition_types


def print_diversity_metrics(
    metrics: DiversityMetrics,
    method_name: str = "",
    seed: int = 0,
) -> None:
    """Print diversity metrics in a formatted way."""
    print(f"\n    === Cluster Diversity Metrics ({method_name} seed {seed}) ===")
    print(f"    Total clusters: {metrics.n_clusters}")
    print(f"    Unique label types: {metrics.unique_labels}")
    print(f"    Redundancy: {metrics.redundancy:.1%} (lower is better)")
    print()
    print(f"    Transition types found ({len(metrics.transition_types_found)}):")
    for t in sorted(metrics.transition_types_found):
        print(f"      - {t[0]} <-> {t[1]}")
    print()
    if metrics.gt_transition_types:
        missing = metrics.gt_transition_types - metrics.transition_types_found
        print(f"    GT transition types: {len(metrics.gt_transition_types)}")
        print(f"    Transition Type Coverage: {metrics.transition_type_coverage:.1%}")
        if missing:
            print(f"    Missing transitions ({len(missing)}):")
            for t in sorted(missing):
                print(f"      - {t[0]} <-> {t[1]}")
    print()
    print(f"    Clusters with turn direction: {metrics.clusters_with_turn}/{metrics.n_clusters}")
    print(f"    Turn Direction Sensitivity: {metrics.turn_direction_sensitivity:.1%}")


def save_diversity_metrics(
    metrics: DiversityMetrics,
    output_dir: str,
    method_name: str = "",
    seed: int = 0,
) -> str:
    """Save diversity metrics to CSV."""
    prefix = f"{method_name}_seed{seed}" if method_name else f"seed{seed}"

    data = {
        "Metric": [
            "Total Clusters",
            "Unique Label Types",
            "Redundancy",
            "Transition Types Found",
            "GT Transition Types",
            "Transition Type Coverage",
            "Clusters with Turn Direction",
            "Turn Direction Sensitivity",
        ],
        "Value": [
            metrics.n_clusters,
            metrics.unique_labels,
            f"{metrics.redundancy:.1%}",
            len(metrics.transition_types_found),
            len(metrics.gt_transition_types) if metrics.gt_transition_types else "N/A",
            f"{metrics.transition_type_coverage:.1%}",
            metrics.clusters_with_turn,
            f"{metrics.turn_direction_sensitivity:.1%}",
        ],
    }

    df = pd.DataFrame(data)
    path = os.path.join(output_dir, f"{prefix}_diversity_metrics.csv")
    df.to_csv(path, index=False)

    return path
