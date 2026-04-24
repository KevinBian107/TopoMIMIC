"""CLI entry point for automatic label assignment.

2026-04-18 rewrite: the labeling pipeline now uses GT majority voting
(``topo_mimic/experiment/auto_label.py``) instead of Wasserstein distance.
Legacy W2 config sections (``distance``, ``gt_extraction``) are accepted but
ignored; only ``recommendations.top_k`` and the ``dominance_threshold`` knob
influence the new algorithm.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from topo_mimic.experiment.auto_label import (
    DEFAULT_DOMINANCE_THRESHOLD,
    GROOMING_LABEL,
    auto_label_clusters_directional,
    generate_auto_labels_yaml,
    load_behavior_segments,
    split_segments_by_kind,
)
from topo_mimic.labeling.core import (
    build_recommendations,
    print_recommendations,
    save_recommendations_csv,
    write_minimal_stats_csv,
)
from topo_mimic.analysis.feature_analysis.data_loading import load_cluster_dict


@dataclass
class AutoLabelConfig:
    dominance_threshold: float = DEFAULT_DOMINANCE_THRESHOLD
    grooming_label: str = GROOMING_LABEL


@dataclass
class LabelingConfig:
    experiment_dir: str = "outputs/benchmark_experiment_transitions"
    behavior_labels_csv: str = "behavior_labels/rodent_labels.csv"
    method: str = "w2_dmp_qpos_kmeans"
    seed: int = 0
    auto_label: AutoLabelConfig = field(default_factory=AutoLabelConfig)
    save_csv: bool = True
    launch_gui: bool = False
    gui_port: int = 7860
    gui_share: bool = False


def load_config(config_path: str) -> LabelingConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    auto_cfg_data: dict[str, Any] = cfg.get("auto_label", {}) or {}
    if "dominance_threshold" not in auto_cfg_data and "gt_extraction" in cfg:
        pass
    auto_cfg = AutoLabelConfig(
        dominance_threshold=float(auto_cfg_data.get("dominance_threshold", DEFAULT_DOMINANCE_THRESHOLD)),
        grooming_label=str(auto_cfg_data.get("grooming_label", GROOMING_LABEL)),
    )

    gui_cfg = cfg.get("gui", {}) or {}
    return LabelingConfig(
        experiment_dir=cfg.get("experiment_dir", LabelingConfig.experiment_dir),
        behavior_labels_csv=cfg.get("behavior_labels_csv", LabelingConfig.behavior_labels_csv),
        method=cfg.get("method", LabelingConfig.method),
        seed=int(cfg.get("seed", LabelingConfig.seed)),
        auto_label=auto_cfg,
        save_csv=bool(cfg.get("save_csv", True)),
        launch_gui=bool(cfg.get("launch_gui", False)),
        gui_port=int(gui_cfg.get("port", 7860)),
        gui_share=bool(gui_cfg.get("share", False)),
    )


def run_labeling(config: LabelingConfig) -> dict:
    print("=" * 60)
    print("Automatic Label Assignment Pipeline (GT majority vote)")
    print("=" * 60)
    print(f"Experiment dir: {config.experiment_dir}")
    print(f"Method: {config.method}")
    print(f"Seed: {config.seed}")
    print(f"Dominance threshold: {config.auto_label.dominance_threshold}")
    print(f"Grooming override label: {config.auto_label.grooming_label}")

    method_dir = os.path.join(config.experiment_dir, config.method)
    if not os.path.exists(method_dir):
        raise FileNotFoundError(f"Method directory not found: {method_dir}")

    print("\n=== Stage 1: Loading data ===")
    cluster_dict = load_cluster_dict(method_dir, config.seed)
    if not cluster_dict:
        raise ValueError(f"No cluster_dict found for {config.method} seed {config.seed}")
    print(f"Loaded {len(cluster_dict)} clusters from {method_dir}")

    print("\n=== Stage 2: Loading GT behavior segments ===")
    segments = load_behavior_segments(config.behavior_labels_csv)
    print(f"Loaded GT segments for {len(segments)} clips from {config.behavior_labels_csv}")
    if not segments:
        print("WARNING: no GT segments found; auto-labels will be empty.")

    print("\n=== Stage 3: GT directional majority-vote labeling ===")
    behav_segs, dir_segs = split_segments_by_kind(segments)
    cluster_labels = auto_label_clusters_directional(
        cluster_dict,
        behav_segs,
        dir_segs,
    )

    print("\n=== Stage 4: Building recommendations ===")
    recommendations = build_recommendations(cluster_labels)
    print_recommendations(recommendations, config.method, config.seed)

    labels_yaml_path = None
    if config.save_csv:
        print("\n=== Stage 5: Saving outputs ===")
        labeling_dir = os.path.join(method_dir, "labeling")
        os.makedirs(labeling_dir, exist_ok=True)

        rec_path = os.path.join(labeling_dir, f"recommendations_seed_{config.seed}.csv")
        save_recommendations_csv(recommendations, rec_path)
        print(f"  recommendations -> {rec_path}")

        stats_path = os.path.join(labeling_dir, f"cluster_votes_seed_{config.seed}.csv")
        write_minimal_stats_csv(cluster_labels, stats_path)
        print(f"  vote stats -> {stats_path}")

        n_clusters = max(cluster_dict.keys()) + 1 if cluster_dict else 0
        labels_yaml_path = generate_auto_labels_yaml(
            method_name=config.method,
            n_clusters=n_clusters,
            seed=config.seed,
            output_dir=method_dir,
            cluster_dict=cluster_dict,
            behavior_labels_csv=config.behavior_labels_csv,
            dominance_threshold=config.auto_label.dominance_threshold,
            grooming_label=config.auto_label.grooming_label,
        )
        print(f"  labels yaml -> {labels_yaml_path}")

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)

    return {
        "cluster_labels": cluster_labels,
        "recommendations": recommendations,
        "labels_yaml_path": labels_yaml_path,
    }


def main(config_path: str = "configs/labeling/auto_label.yaml"):
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        print(f"Config not found: {config_path}, using defaults")
        config = LabelingConfig()

    results = run_labeling(config)

    if config.launch_gui:
        from topo_mimic.labeling.gui import launch_gui
        launch_gui(config, results, port=config.gui_port, share=config.gui_share)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/labeling/auto_label.yaml"
    main(config_path)
