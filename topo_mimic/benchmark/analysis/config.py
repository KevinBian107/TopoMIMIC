"""Configuration for transition analysis."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class AnalysisConfig:
    experiment_dir: str
    output_dir: str = "analysis_output"
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    behavior_labels_csv: str = "/home/jovyan/vast/kaiwen/TopoVNL/behavior_labels.csv"

    methods: list[str] = None
    seeds: list[int] = None

    run_mds_analysis: bool = True
    run_transition_comparison: bool = True
    run_transition_ranking: bool = True
    run_transition_graph: bool = False
    render_gt_transitions: bool = False

    gt_render_max_samples: int = 5
    gt_render_framerate: int = 30
    gt_render_target_size: tuple[int, int] = (320, 240)
    mds_k_cycles: int = 50
    mds_s_top: int = 10
    mds_behaviors: list[str] = None
    mds_figsize: tuple[float, float] = (10, 9)

    transition_window: int = 5
    max_transition_points: int = 500

    short_behavior_threshold: int = 10
    merge_rear: bool = True
    max_gap_between_segments: int = 10

    transition_graph_k: int = 3
    transition_graph_gt_top_k: int = 10
    transition_graph_layout: str = "spring"
    transition_graph_figsize: tuple[float, float] = (16, 7)


def load_config(config_path: str | Path) -> AnalysisConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return AnalysisConfig(
        experiment_dir=cfg.get("experiment_dir", ""),
        output_dir=cfg.get("output_dir", "analysis_output"),
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", AnalysisConfig.rodent_rollout_dir),
        behavior_labels_csv=cfg.get("behavior_labels_csv", AnalysisConfig.behavior_labels_csv),
        methods=cfg.get("methods", None),
        seeds=cfg.get("seeds", None),
        run_mds_analysis=cfg.get("run_mds_analysis", True),
        run_transition_comparison=cfg.get("run_transition_comparison", True),
        run_transition_ranking=cfg.get("run_transition_ranking", True),
        run_transition_graph=cfg.get("run_transition_graph", False),
        render_gt_transitions=cfg.get("render_gt_transitions", False),
        gt_render_max_samples=cfg.get("gt_render_max_samples", 5),
        gt_render_framerate=cfg.get("gt_render_framerate", 30),
        gt_render_target_size=tuple(cfg.get("gt_render_target_size", [320, 240])),
        mds_k_cycles=cfg.get("mds_k_cycles", 50),
        mds_s_top=cfg.get("mds_s_top", 10),
        mds_behaviors=cfg.get("mds_behaviors", None),
        mds_figsize=tuple(cfg.get("mds_figsize", [14, 12])),
        transition_window=cfg.get("transition_window", 5),
        max_transition_points=cfg.get("max_transition_points", 500),
        short_behavior_threshold=cfg.get("short_behavior_threshold", 10),
        merge_rear=cfg.get("merge_rear", True),
        max_gap_between_segments=cfg.get("max_gap_between_segments", 10),
        transition_graph_k=cfg.get("transition_graph_k", 3),
        transition_graph_gt_top_k=cfg.get("transition_graph_gt_top_k", 10),
        transition_graph_layout=cfg.get("transition_graph_layout", "spring"),
        transition_graph_figsize=tuple(cfg.get("transition_graph_figsize", [16, 7])),
    )
