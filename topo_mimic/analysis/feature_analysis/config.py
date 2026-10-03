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

    run_gt_sanity_check: bool = True
    run_mds_analysis: bool = True
    run_transition_comparison: bool = True
    run_transition_ranking: bool = True
    run_transition_graph: bool = False
    run_dtw_metrics: bool = True
    run_dtw_visualization: bool = False
    run_c2st_analysis: bool = True
    run_purity_analysis: bool = True
    run_mmd_analysis: bool = True
    render_gt_transitions: bool = False

    c2st_raw_h5_path: str = "data/rodent_ar.h5"
    c2st_window_frames: int = 100
    c2st_min_cluster_size: int = 20  # 5-fold CV needs ≥4 per class per fold for stable acc; 20 gives 4
    c2st_n_folds: int = 5
    c2st_models: tuple[str, ...] = ("logreg", "rf", "mlp")

    gt_sanity_top_k: int = 10
    gt_sanity_data_key: str = "qpos_rollout"
    gt_sanity_figsize: tuple[float, float] = (12, 10)
    gt_sanity_skip_root_dims: int = 7

    dtw_data_key: str = "qpos_rollout"
    dtw_max_seqs: int = 50
    dtw_max_comparisons: int = 100
    dtw_min_seq_length: int = 3
    dtw_skip_root_dims: int = 7

    kid_unit: str = "window"  # "window" (flattened resampled windows) | "frame" (pooled frames)
    kid_subset_size: int = 1000
    kid_n_subsets: int = 10

    pure_behavior_labels: list[str] = None
    pure_behavior_min_length: int = 30

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

    consistency_n_samples: int = 5

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
        run_gt_sanity_check=cfg.get("run_gt_sanity_check", True),
        run_mds_analysis=cfg.get("run_mds_analysis", True),
        run_transition_comparison=cfg.get("run_transition_comparison", True),
        run_transition_ranking=cfg.get("run_transition_ranking", True),
        run_transition_graph=cfg.get("run_transition_graph", False),
        run_dtw_metrics=cfg.get("run_dtw_metrics", True),
        run_dtw_visualization=cfg.get("run_dtw_visualization", False),
        run_c2st_analysis=cfg.get("run_c2st_analysis", True),
        run_purity_analysis=cfg.get("run_purity_analysis", True),
        run_mmd_analysis=cfg.get("run_mmd_analysis", True),
        c2st_raw_h5_path=cfg.get("c2st_raw_h5_path", "data/rodent_ar.h5"),
        c2st_window_frames=int(cfg.get("c2st_window_frames", 100)),
        c2st_min_cluster_size=int(cfg.get("c2st_min_cluster_size", 8)),
        c2st_n_folds=int(cfg.get("c2st_n_folds", 5)),
        c2st_models=tuple(cfg.get("c2st_models", ["logreg", "rf", "mlp"])),
        render_gt_transitions=cfg.get("render_gt_transitions", False),
        gt_sanity_top_k=cfg.get("gt_sanity_top_k", 10),
        gt_sanity_data_key=cfg.get("gt_sanity_data_key", "qpos_rollout"),
        gt_sanity_figsize=tuple(cfg.get("gt_sanity_figsize", [12, 10])),
        gt_sanity_skip_root_dims=cfg.get("gt_sanity_skip_root_dims", 7),
        dtw_data_key=cfg.get("dtw_data_key", "qpos_rollout"),
        dtw_max_seqs=cfg.get("dtw_max_seqs", 50),
        dtw_max_comparisons=cfg.get("dtw_max_comparisons", 100),
        dtw_min_seq_length=cfg.get("dtw_min_seq_length", 3),
        dtw_skip_root_dims=cfg.get("dtw_skip_root_dims", 7),
        kid_unit=str(cfg.get("kid_unit", "window")),
        kid_subset_size=int(cfg.get("kid_subset_size", 1000)),
        kid_n_subsets=int(cfg.get("kid_n_subsets", 10)),
        pure_behavior_labels=cfg.get("pure_behavior_labels", None),
        pure_behavior_min_length=cfg.get("pure_behavior_min_length", 30),
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
        consistency_n_samples=cfg.get("consistency_n_samples", 5),
        transition_graph_k=cfg.get("transition_graph_k", 3),
        transition_graph_gt_top_k=cfg.get("transition_graph_gt_top_k", 10),
        transition_graph_layout=cfg.get("transition_graph_layout", "spring"),
        transition_graph_figsize=tuple(cfg.get("transition_graph_figsize", [16, 7])),
    )
