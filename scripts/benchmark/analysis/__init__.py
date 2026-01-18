"""
Analysis module for transition cluster comparison.

Provides tools for comparing discovered behavioral clusters against
ground truth behavior labels, with support for:
- Short intermediate behavior bridging
- Turn direction classification (left/right/straight)
- Merged rear categories (Supported + Unsupported)
- Transition variant ranking
- JAVD (Joint Angle Variance Difference) analysis
"""

from .config import AnalysisConfig, load_config
from .behavior_parsing import (
    PRIMITIVE_BEHAVIORS,
    COMPOSITIONAL_BEHAVIORS,
    TurnDirection,
    ParsedLabel,
    parse_cluster_label,
    normalize_behavior_name,
    merge_rear_category,
)
from .ground_truth import (
    BehaviorSegment,
    TransitionSegment,
    load_behavior_segments,
    find_gt_transition_frames,
    find_all_transitions,
    get_turn_direction_for_segment,
)
from .transition_ranking import (
    TransitionVariant,
    count_transition_variants,
    print_transition_ranking,
    run_transition_ranking_analysis,
)
from .data_loading import (
    load_cluster_dict,
    load_labels_yaml,
    load_qpos,
    load_intention,
    load_experiment_results,
)
from .javd import compute_javd_table
from .visualization import (
    plot_transition_cluster_comparison,
    plot_variance_heatmap,
)
from .main import run_analysis

__all__ = [
    "AnalysisConfig",
    "load_config",
    "PRIMITIVE_BEHAVIORS",
    "COMPOSITIONAL_BEHAVIORS",
    "TurnDirection",
    "ParsedLabel",
    "parse_cluster_label",
    "normalize_behavior_name",
    "merge_rear_category",
    "BehaviorSegment",
    "TransitionSegment",
    "load_behavior_segments",
    "find_gt_transition_frames",
    "find_all_transitions",
    "get_turn_direction_for_segment",
    "TransitionVariant",
    "count_transition_variants",
    "print_transition_ranking",
    "run_transition_ranking_analysis",
    "load_cluster_dict",
    "load_labels_yaml",
    "load_qpos",
    "load_intention",
    "load_experiment_results",
    "compute_javd_table",
    "plot_transition_cluster_comparison",
    "plot_variance_heatmap",
    "run_analysis",
]
