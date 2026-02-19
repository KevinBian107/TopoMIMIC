"""CLI entry point for automatic label assignment."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import yaml

from topo_mimic.benchmark.labeling.core import (
    extract_gt_transition_windows,
    compute_cluster_to_gt_distances,
    generate_label_recommendations,
    print_gt_statistics,
    print_recommendations,
    save_recommendations_csv,
    save_gt_statistics_csv,
)
from topo_mimic.benchmark.analysis.data_loading import (
    load_method_results,
    load_cluster_dict,
)


@dataclass
class GTExtractionConfig:
    target_len: int = 100
    short_threshold: int = 10
    max_gap: int = 10
    merge_rear: bool = True
    include_non_transitions: list[str] = None
    non_transition_min_length: int = 30

    def __post_init__(self):
        if self.include_non_transitions is None:
            self.include_non_transitions = []


@dataclass
class DistanceConfig:
    max_cycles_per_cluster: int = 20
    max_gt_samples: int = 10


@dataclass
class RecommendationsConfig:
    top_k: int = 5


@dataclass
class LabelingConfig:
    experiment_dir: str = "outputs/benchmark_experiment"
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar.h5"
    behavior_labels_csv: str = "/home/jovyan/vast/kaiwen/TopoVNL/behavior_labels.csv"

    method: str = "w2_kmeans"
    seed: int = 0

    gt_extraction: GTExtractionConfig = None
    distance: DistanceConfig = None
    recommendations: RecommendationsConfig = None

    save_csv: bool = True
    launch_gui: bool = False
    gui_port: int = 7860
    gui_share: bool = False

    def __post_init__(self):
        if self.gt_extraction is None:
            self.gt_extraction = GTExtractionConfig()
        if self.distance is None:
            self.distance = DistanceConfig()
        if self.recommendations is None:
            self.recommendations = RecommendationsConfig()


def load_config(config_path: str) -> LabelingConfig:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    gt_extraction_data = cfg.get("gt_extraction", {})
    if "include_non_transitions" in gt_extraction_data and gt_extraction_data["include_non_transitions"] is None:
        gt_extraction_data["include_non_transitions"] = []
    gt_cfg = GTExtractionConfig(**gt_extraction_data)
    dist_cfg = DistanceConfig(**cfg.get("distance", {}))
    rec_cfg = RecommendationsConfig(**cfg.get("recommendations", {}))

    gui_cfg = cfg.get("gui", {})

    return LabelingConfig(
        experiment_dir=cfg.get("experiment_dir", LabelingConfig.experiment_dir),
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", LabelingConfig.rodent_rollout_dir),
        behavior_labels_csv=cfg.get("behavior_labels_csv", LabelingConfig.behavior_labels_csv),
        method=cfg.get("method", LabelingConfig.method),
        seed=cfg.get("seed", LabelingConfig.seed),
        gt_extraction=gt_cfg,
        distance=dist_cfg,
        recommendations=rec_cfg,
        save_csv=cfg.get("save_csv", True),
        launch_gui=cfg.get("launch_gui", False),
        gui_port=gui_cfg.get("port", 7860),
        gui_share=gui_cfg.get("share", False),
    )


def load_intention_data(rodent_rollout_dir: str) -> np.ndarray:
    """Load intention data from h5 file."""
    if rodent_rollout_dir.endswith('.h5'):
        with h5py.File(rodent_rollout_dir, 'r') as f:
            return np.array(f['intention'])
    else:
        raise ValueError("Expected .h5 file for rodent_rollout_dir")


def run_labeling(config: LabelingConfig) -> dict:
    """
    Run the full labeling pipeline.

    Returns:
        Dictionary with recommendations and statistics
    """
    print("=" * 60)
    print("Automatic Label Assignment Pipeline")
    print("=" * 60)
    print(f"Experiment dir: {config.experiment_dir}")
    print(f"Method: {config.method}")
    print(f"Seed: {config.seed}")

    method_dir = os.path.join(config.experiment_dir, config.method)
    if not os.path.exists(method_dir):
        raise FileNotFoundError(f"Method directory not found: {method_dir}")

    print("\n=== Stage 1: Loading data ===")
    intention_data = load_intention_data(config.rodent_rollout_dir)
    print(f"Loaded intention data: {intention_data.shape}")

    cluster_dict = load_cluster_dict(method_dir, config.seed)
    if not cluster_dict:
        raise ValueError(f"No cluster_dict found for {config.method} seed {config.seed}")
    print(f"Loaded {len(cluster_dict)} clusters")

    print("\n=== Stage 2: Extracting GT transitions ===")
    gt_library = extract_gt_transition_windows(
        behavior_labels_csv=config.behavior_labels_csv,
        intention_data=intention_data,
        target_len=config.gt_extraction.target_len,
        short_threshold=config.gt_extraction.short_threshold,
        max_gap=config.gt_extraction.max_gap,
        merge_rear=config.gt_extraction.merge_rear,
        include_non_transitions=config.gt_extraction.include_non_transitions,
        non_transition_min_length=config.gt_extraction.non_transition_min_length,
    )
    print_gt_statistics(gt_library)

    print("\n=== Stage 3: Computing W2 distances ===")
    distances, gt_type_names, cluster_ids = compute_cluster_to_gt_distances(
        cluster_dict=cluster_dict,
        gt_library=gt_library,
        intention_data=intention_data,
        max_cycles_per_cluster=config.distance.max_cycles_per_cluster,
        max_gt_samples=config.distance.max_gt_samples,
        seed=config.seed,
    )

    print("\n=== Stage 4: Generating recommendations ===")
    recommendations = generate_label_recommendations(
        distances=distances,
        gt_type_names=gt_type_names,
        cluster_ids=cluster_ids,
        top_k=config.recommendations.top_k,
    )
    print_recommendations(recommendations, config.method, config.seed)

    if config.save_csv:
        print("\n=== Stage 5: Saving results ===")
        labeling_dir = os.path.join(method_dir, "labeling")
        os.makedirs(labeling_dir, exist_ok=True)

        rec_path = os.path.join(labeling_dir, f"recommendations_seed_{config.seed}.csv")
        save_recommendations_csv(recommendations, rec_path)

        gt_stats_path = os.path.join(labeling_dir, "gt_statistics.csv")
        save_gt_statistics_csv(gt_library, gt_stats_path)

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)

    return {
        "recommendations": recommendations,
        "gt_library": gt_library,
        "distances": distances,
        "gt_type_names": gt_type_names,
        "cluster_ids": cluster_ids,
    }


def main(config_path: str = "configs/benchmark/labeling.yaml"):
    """CLI entry point."""
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        print(f"Config not found: {config_path}, using defaults")
        config = LabelingConfig()

    results = run_labeling(config)

    if config.launch_gui:
        from topo_mimic.benchmark.labeling.gui import launch_gui
        launch_gui(config, results, port=config.gui_port, share=config.gui_share)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/benchmark/labeling.yaml"
    main(config_path)
