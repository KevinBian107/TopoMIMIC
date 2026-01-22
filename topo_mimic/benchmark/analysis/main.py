"""Main analysis runner for transition cluster comparison."""

import os
import sys

import h5py
import numpy as np

from .config import AnalysisConfig, load_config
from .data_loading import (
    load_experiment_results,
    load_cluster_dict,
    load_labels_yaml,
    load_method_results,
)
from .transition_ranking import run_transition_ranking_analysis
from .visualization import (
    plot_transition_cluster_comparison,
    plot_transition_graph,
)
from .metrics import (
    run_mmd_analysis,
    plot_mmd_comparison_table,
)


def load_trajectory_data(rodent_rollout_dir: str, data_key: str) -> np.ndarray:
    """Load trajectory data (qpos or intention) from h5 file."""
    if rodent_rollout_dir.endswith('.h5'):
        with h5py.File(rodent_rollout_dir, 'r') as f:
            available_keys = list(f.keys())
            if data_key not in available_keys:
                raise KeyError(
                    f"Key '{data_key}' not found in h5 file. "
                    f"Available keys: {available_keys}"
                )
            return np.array(f[data_key])
    else:
        raise ValueError("Expected .h5 file for rodent_rollout_dir")


def run_analysis(config: AnalysisConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("Unified Analysis Pipeline")
    print("=" * 60)

    if config.run_transition_ranking:
        print("\n=== Running Transition Variant Ranking ===")
        ranking_output = os.path.join(config.output_dir, "transition_ranking")
        run_transition_ranking_analysis(
            csv_path=config.behavior_labels_csv,
            output_dir=ranking_output,
            short_threshold=config.short_behavior_threshold,
            max_gap=config.max_gap_between_segments,
            merge_rear=config.merge_rear,
            top_k=50,
        )

    all_results = load_experiment_results(config.experiment_dir)
    print(f"\nLoaded results for {len(all_results)} methods")

    if config.methods:
        print(f"Filtering to methods: {config.methods}")
    if config.seeds:
        print(f"Filtering to seeds: {config.seeds}")

    mmd_data = None
    if config.run_mmd_metrics:
        print(f"\n=== Loading data for MMD ({config.mmd_data_key}) ===")
        try:
            mmd_data = load_trajectory_data(
                config.rodent_rollout_dir,
                config.mmd_data_key,
            )
            print(f"Loaded data: {mmd_data.shape}")
        except Exception as e:
            print(f"Failed to load data: {e}")
            config.run_mmd_metrics = False

    all_mmd_metrics = {}

    for method_name, method_results in all_results.items():
        if config.methods and method_name not in config.methods:
            continue

        print(f"\n=== Analyzing: {method_name} ===")

        method_dir = os.path.join(config.experiment_dir, method_name)
        method_output_dir = os.path.join(config.output_dir, method_name)
        os.makedirs(method_output_dir, exist_ok=True)

        seeds = sorted([k for k in method_results.keys() if isinstance(k, int)])
        if config.seeds:
            seeds = [s for s in seeds if s in config.seeds]
        if not seeds:
            print(f"  No seeds found, skipping")
            continue

        cluster_dicts = {}
        cluster_labels_per_seed = {}
        turn_directions_per_seed = {}

        for seed in seeds:
            result = method_results.get(seed, {})
            cluster_dict = result.get("cluster_dict", {})

            if not cluster_dict:
                print(f"  Seed {seed}: No cluster_dict found, skipping")
                continue

            cluster_dicts[seed] = cluster_dict
            cluster_labels, turn_directions = load_labels_yaml(method_dir, seed)
            if not cluster_labels:
                cluster_labels = {i: f"C{i}" for i in cluster_dict.keys()}
            cluster_labels_per_seed[seed] = cluster_labels
            turn_directions_per_seed[seed] = turn_directions

            seed_output_dir = os.path.join(method_output_dir, f"seed_{seed}")
            os.makedirs(seed_output_dir, exist_ok=True)

            print(f"  Seed {seed}:")

            if config.run_mds_analysis:
                if os.path.exists(config.behavior_labels_csv):
                    print(f"    Running MDS analysis...")
                    try:
                        from topo_mimic.dmwk.skeletonization.plotting import (
                            plot_cluster_subspace_mds_separate,
                        )
                        mds_dir = os.path.join(seed_output_dir, "mds")
                        os.makedirs(mds_dir, exist_ok=True)
                        target_clusters = list(cluster_dict.keys())
                        mds_save_path = os.path.join(mds_dir, f"{method_name}_seed{seed}_mds_separate.pdf")
                        plot_cluster_subspace_mds_separate(
                            cluster_cycles_by_id=cluster_dict,
                            cluster_labels=cluster_labels,
                            turn_directions=turn_directions,
                            target_clusters=target_clusters,
                            rodent_rollout_dir=config.rodent_rollout_dir,
                            behavior_csv_path=config.behavior_labels_csv,
                            behaviors=config.mds_behaviors,
                            k_cycles=config.mds_k_cycles,
                            s_top=config.mds_s_top,
                            figsize=config.mds_figsize,
                            save_path=mds_save_path,
                        )
                        print(f"    Saved MDS plot to {mds_save_path}")
                    except Exception as e:
                        print(f"    MDS analysis failed: {e}")

            if config.run_transition_comparison:
                if os.path.exists(config.behavior_labels_csv):
                    print(f"    Running transition cluster comparison...")
                    try:
                        plot_transition_cluster_comparison(
                            cluster_dict=cluster_dict,
                            cluster_labels=cluster_labels,
                            turn_directions=turn_directions,
                            behavior_labels_csv=config.behavior_labels_csv,
                            rodent_rollout_dir=config.rodent_rollout_dir,
                            output_dir=seed_output_dir,
                            max_points=config.max_transition_points,
                            method_name=method_name,
                            seed=seed,
                            short_threshold=config.short_behavior_threshold,
                            merge_rear=config.merge_rear,
                        )
                    except Exception as e:
                        print(f"    Transition comparison failed: {e}")

            if config.run_transition_graph:
                if os.path.exists(config.behavior_labels_csv):
                    print(f"    Running transition graph analysis...")
                    try:
                        plot_transition_graph(
                            cluster_dict=cluster_dict,
                            cluster_labels=cluster_labels,
                            turn_directions=turn_directions,
                            behavior_labels_csv=config.behavior_labels_csv,
                            rodent_rollout_dir=config.rodent_rollout_dir,
                            output_dir=seed_output_dir,
                            k_neighbors=config.transition_graph_k,
                            gt_top_k=config.transition_graph_gt_top_k,
                            layout=config.transition_graph_layout,
                            figsize=config.transition_graph_figsize,
                            method_name=method_name,
                            seed=seed,
                            short_threshold=config.short_behavior_threshold,
                            merge_rear=config.merge_rear,
                        )
                    except Exception as e:
                        print(f"    Transition graph analysis failed: {e}")

            if config.run_mmd_metrics and mmd_data is not None:
                if os.path.exists(config.behavior_labels_csv):
                    print(f"    Running MMD metrics ({config.mmd_data_key})...")
                    try:
                        metrics_dir = os.path.join(seed_output_dir, "mmd_metrics")
                        mmd_metrics = run_mmd_analysis(
                            cluster_dict=cluster_dict,
                            cluster_labels=cluster_labels,
                            behavior_labels_csv=config.behavior_labels_csv,
                            data=mmd_data,
                            output_dir=metrics_dir,
                            method_name=method_name,
                            seed=seed,
                            target_len=config.mmd_target_len,
                            short_threshold=config.short_behavior_threshold,
                            merge_rear=config.merge_rear,
                        )
                        method_key = f"{method_name}_seed{seed}"
                        if mmd_metrics:
                            all_mmd_metrics[method_key] = mmd_metrics
                    except Exception as e:
                        print(f"    MMD metrics failed: {e}")
                        import traceback
                        traceback.print_exc()

    if config.run_mmd_metrics and all_mmd_metrics:
        print("\n=== Generating MMD Comparison Table ===")
        comparison_dir = os.path.join(config.output_dir, "comparison_tables")
        os.makedirs(comparison_dir, exist_ok=True)

        mmd_table_path = os.path.join(comparison_dir, "mmd_comparison.pdf")
        plot_mmd_comparison_table(
            all_mmd_metrics,
            mmd_table_path,
            title="MMD Distance by Method and Transition Type",
        )

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Results saved to: {config.output_dir}")


def main(config_path: str = "configs/benchmark/analysis.yaml"):
    if os.path.exists(config_path):
        cfg = load_config(config_path)
    else:
        cfg = AnalysisConfig(experiment_dir="out/experiment")

    run_analysis(cfg)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/benchmark/analysis.yaml"
    main(config_path)
