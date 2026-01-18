"""Automatic label assignment module for behavioral clustering benchmark."""

from .core import (
    extract_gt_transition_windows,
    compute_cluster_to_gt_distances,
    generate_label_recommendations,
    print_gt_statistics,
    print_recommendations,
    save_recommendations_csv,
    save_gt_statistics_csv,
    GTLibrary,
    GTStatistics,
    LabelRecommendation,
)
from .main import run_labeling, load_config, LabelingConfig

__all__ = [
    "extract_gt_transition_windows",
    "compute_cluster_to_gt_distances",
    "generate_label_recommendations",
    "print_gt_statistics",
    "print_recommendations",
    "save_recommendations_csv",
    "save_gt_statistics_csv",
    "GTLibrary",
    "GTStatistics",
    "LabelRecommendation",
    "run_labeling",
    "load_config",
    "LabelingConfig",
]
