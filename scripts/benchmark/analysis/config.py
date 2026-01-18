"""Configuration for transition analysis."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AnalysisConfig:
    experiment_dir: str
    output_dir: str = "analysis_output"
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    behavior_csv: str = "/home/jovyan/vast/kaiwen/TopoVNL/behavior_book.csv"
    behavior_labels_csv: str = "/home/jovyan/vast/kaiwen/TopoVNL/behavior_labels.csv"

    methods: list[str] = None
    seeds: list[int] = None

    run_variance_heatmap: bool = True
    run_mds_analysis: bool = True
    run_cluster_to_label_comparison: bool = True
    run_seed_comparison: bool = True
    run_transition_comparison: bool = True
    run_javd_analysis: bool = True
    run_diversity_analysis: bool = True
    run_transition_ranking: bool = True
    render_gt_transitions: bool = False

    top_k_variance_joints: int = 5
    gt_render_max_samples: int = 5
    gt_render_framerate: int = 30
    gt_render_target_size: tuple[int, int] = (320, 240)
    mds_k_cycles: int = 50
    mds_s_top: int = 10
    mds_behaviors: list[str] = None
    mds_figsize: tuple[float, float] = (10, 9)

    body_parts: list[str] = None
    max_loops: int = 200

    transition_window: int = 5
    max_transition_points: int = 500

    short_behavior_threshold: int = 10
    merge_rear: bool = True
    max_gap_between_segments: int = 10

    def __post_init__(self):
        if self.body_parts is None:
            self.body_parts = [
                "torso", "pelvis",
                "upper_leg_L", "lower_leg_L", "foot_L",
                "upper_leg_R", "lower_leg_R", "foot_R",
                "skull", "jaw",
                "upper_arm_L", "lower_arm_L",
                "upper_arm_R", "lower_arm_R",
            ]


def load_config(config_path: str | Path) -> AnalysisConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return AnalysisConfig(
        experiment_dir=cfg.get("experiment_dir", ""),
        output_dir=cfg.get("output_dir", "analysis_output"),
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", AnalysisConfig.rodent_rollout_dir),
        behavior_csv=cfg.get("behavior_csv", AnalysisConfig.behavior_csv),
        behavior_labels_csv=cfg.get("behavior_labels_csv", AnalysisConfig.behavior_labels_csv),
        methods=cfg.get("methods", None),
        seeds=cfg.get("seeds", None),
        run_variance_heatmap=cfg.get("run_variance_heatmap", True),
        run_mds_analysis=cfg.get("run_mds_analysis", True),
        run_cluster_to_label_comparison=cfg.get("run_cluster_to_label_comparison", True),
        run_seed_comparison=cfg.get("run_seed_comparison", True),
        run_transition_comparison=cfg.get("run_transition_comparison", True),
        run_javd_analysis=cfg.get("run_javd_analysis", True),
        run_diversity_analysis=cfg.get("run_diversity_analysis", True),
        run_transition_ranking=cfg.get("run_transition_ranking", True),
        render_gt_transitions=cfg.get("render_gt_transitions", False),
        top_k_variance_joints=cfg.get("top_k_variance_joints", 5),
        gt_render_max_samples=cfg.get("gt_render_max_samples", 5),
        gt_render_framerate=cfg.get("gt_render_framerate", 30),
        gt_render_target_size=tuple(cfg.get("gt_render_target_size", [320, 240])),
        mds_k_cycles=cfg.get("mds_k_cycles", 50),
        mds_s_top=cfg.get("mds_s_top", 10),
        mds_behaviors=cfg.get("mds_behaviors", None),
        mds_figsize=tuple(cfg.get("mds_figsize", [14, 12])),
        body_parts=cfg.get("body_parts", None),
        max_loops=cfg.get("max_loops", 200),
        transition_window=cfg.get("transition_window", 5),
        max_transition_points=cfg.get("max_transition_points", 500),
        short_behavior_threshold=cfg.get("short_behavior_threshold", 10),
        merge_rear=cfg.get("merge_rear", True),
        max_gap_between_segments=cfg.get("max_gap_between_segments", 10),
    )
