"""Main analysis runner for transition cluster comparison."""

import os
import sys

import h5py
import numpy as np
import pandas as pd


def _should_skip_analysis(path: str) -> bool:
    """Skip rule for per-(method, seed, analysis) outputs.

    Returns True if ``path`` exists and has at least one file or subdirectory.
    Override by `rm -rf` of the output folder.
    """
    if not os.path.isdir(path):
        return False
    try:
        return any(True for _ in os.scandir(path))
    except OSError:
        return False

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
    run_dtw_analysis,
    DTWResult,
    run_internal_consistency_analysis,
    InternalConsistencyResult,
)
from .gt_sanity_check import run_gt_dtw_sanity_check
from .dtw_visualization import run_dtw_visualization


def create_dtw_comparison_table(
    all_dtw_results: dict[str, list[DTWResult]],
    output_dir: str,
) -> None:
    """
    Create a comparison table of DTW results across methods.

    Rows are (label, turn_direction) combinations, columns are methods.
    N/A for clusters that didn't appear in a method.
    """
    all_keys = set()
    for results in all_dtw_results.values():
        for r in results:
            key = (r.label, r.turn_direction or "none")
            all_keys.add(key)

    all_keys = sorted(all_keys)
    methods = sorted(all_dtw_results.keys())

    rows = []
    for label, turn in all_keys:
        row = {"Label": label, "Turn": turn}
        for method in methods:
            results = all_dtw_results.get(method, [])
            found = None
            for r in results:
                r_turn = r.turn_direction or "none"
                if r.label == label and r_turn == turn:
                    found = r
                    break
            if found:
                row[method] = f"{found.mean_dtw:.4f}"
            else:
                row[method] = "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)

    csv_path = os.path.join(output_dir, "dtw_comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved DTW comparison table to {csv_path}")

    print("\nDTW Comparison Table:")
    print("-" * 80)
    print(df.to_string(index=False))
    print("-" * 80)

    numeric_cols = [c for c in df.columns if c not in ("Label", "Turn")]
    if numeric_cols:
        print("\nMethod Averages (excluding N/A):")
        for method in numeric_cols:
            vals = []
            for v in df[method]:
                if v != "N/A":
                    vals.append(float(v))
            if vals:
                print(f"  {method}: {np.mean(vals):.4f} (n={len(vals)})")


def create_consistency_comparison_table(
    all_consistency_results: dict[str, list[InternalConsistencyResult]],
    output_dir: str,
) -> None:
    """
    Create a comparison table of internal consistency results across methods.

    Rows are (label, turn_direction) combinations, columns are methods.
    """
    all_keys = set()
    for results in all_consistency_results.values():
        for r in results:
            key = (r.label, r.turn_direction or "none")
            all_keys.add(key)

    all_keys = sorted(all_keys)
    methods = sorted(all_consistency_results.keys())

    rows = []
    for label, turn in all_keys:
        row = {"Label": label, "Turn": turn}
        for method in methods:
            results = all_consistency_results.get(method, [])
            found = None
            for r in results:
                r_turn = r.turn_direction or "none"
                if r.label == label and r_turn == turn:
                    found = r
                    break
            if found:
                row[method] = f"{found.mean_intra_dtw:.4f}"
            else:
                row[method] = "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)

    csv_path = os.path.join(output_dir, "consistency_comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved consistency comparison table to {csv_path}")

    print("\nInternal Consistency Comparison Table:")
    print("-" * 80)
    print(df.to_string(index=False))
    print("-" * 80)


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

    if config.run_gt_sanity_check:
        print("\n=== Running GT Sanity Check (DTW) ===")
        print(f"Comparing top {config.gt_sanity_top_k} GT transition sequences via DTW")
        sanity_output = os.path.join(config.output_dir, "gt_sanity_check")
        try:
            sanity_data = load_trajectory_data(
                config.rodent_rollout_dir,
                config.gt_sanity_data_key,
            )
            print(f"Loaded {config.gt_sanity_data_key}: {sanity_data.shape}")
            run_gt_dtw_sanity_check(
                behavior_labels_csv=config.behavior_labels_csv,
                data=sanity_data,
                output_dir=sanity_output,
                top_k=config.gt_sanity_top_k,
                max_seqs_per_cluster=config.dtw_max_seqs,
                max_comparisons=config.dtw_max_comparisons,
                short_threshold=config.short_behavior_threshold,
                merge_rear=config.merge_rear,
                figsize=config.gt_sanity_figsize,
                skip_root_dims=config.gt_sanity_skip_root_dims,
                min_seq_length=config.dtw_min_seq_length,
            )
        except Exception as e:
            print(f"GT DTW sanity check failed: {e}")
            import traceback
            traceback.print_exc()

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

    dtw_data = None

    if config.run_dtw_metrics:
        print(f"\n=== Loading data for DTW metrics ({config.dtw_data_key}) ===")
        try:
            dtw_data = load_trajectory_data(
                config.rodent_rollout_dir,
                config.dtw_data_key,
            )
            print(f"Loaded data: {dtw_data.shape}")
        except Exception as e:
            print(f"Failed to load data: {e}")
            config.run_dtw_metrics = False

    all_dtw_results: dict[str, list[DTWResult]] = {}
    all_consistency_results: dict[str, list[InternalConsistencyResult]] = {}

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
                mds_dir = os.path.join(seed_output_dir, "mds")
                if _should_skip_analysis(mds_dir):
                    print(f"    [skip] MDS already exists at {mds_dir}")
                elif os.path.exists(config.behavior_labels_csv):
                    print(f"    Running MDS analysis...")
                    try:
                        from topo_mimic.dmwk.skeletonization.plotting import (
                            plot_cluster_subspace_mds_separate,
                        )
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

            if config.run_dtw_metrics and dtw_data is not None:
                dtw_dir = os.path.join(seed_output_dir, "dtw_metrics")
                if _should_skip_analysis(dtw_dir):
                    print(f"    [skip] DTW metrics already exist at {dtw_dir}")
                elif os.path.exists(config.behavior_labels_csv):
                    print(f"    Running DTW analysis (sequence-based comparison)...")
                    try:
                        dtw_results = run_dtw_analysis(
                            cluster_dict=cluster_dict,
                            cluster_labels=cluster_labels,
                            turn_directions=turn_directions,
                            behavior_labels_csv=config.behavior_labels_csv,
                            data=dtw_data,
                            output_dir=dtw_dir,
                            method_name=method_name,
                            seed=seed,
                            max_samples=config.dtw_max_seqs,
                            max_comparisons=config.dtw_max_comparisons,
                            short_threshold=config.short_behavior_threshold,
                            merge_rear=config.merge_rear,
                            skip_root_dims=config.dtw_skip_root_dims,
                            min_seq_length=config.dtw_min_seq_length,
                            pure_behavior_labels=config.pure_behavior_labels,
                            pure_behavior_min_length=config.pure_behavior_min_length,
                            resample_K=config.c2st_window_frames,
                        )
                        method_key = f"{method_name}_seed{seed}"
                        all_dtw_results[method_key] = dtw_results
                    except Exception as e:
                        print(f"    DTW analysis failed: {e}")
                        import traceback
                        traceback.print_exc()

                    consistency_dir = os.path.join(seed_output_dir, "consistency")
                    if _should_skip_analysis(consistency_dir):
                        print(f"    [skip] Consistency already exists at {consistency_dir}")
                    else:
                        print(f"    Running internal consistency analysis...")
                        try:
                            consistency_results = run_internal_consistency_analysis(
                                cluster_dict=cluster_dict,
                                cluster_labels=cluster_labels,
                                turn_directions=turn_directions,
                                data=dtw_data,
                                output_dir=consistency_dir,
                                method_name=method_name,
                                seed=seed,
                                skip_root_dims=config.dtw_skip_root_dims,
                                min_seq_length=config.dtw_min_seq_length,
                                merge_rear=config.merge_rear,
                                n_samples=config.consistency_n_samples,
                                resample_K=config.c2st_window_frames,
                            )
                            method_key = f"{method_name}_seed{seed}"
                            all_consistency_results[method_key] = consistency_results
                        except Exception as e:
                            print(f"    Internal consistency analysis failed: {e}")
                            import traceback
                            traceback.print_exc()

            if config.run_dtw_visualization and dtw_data is not None:
                dtw_viz_dir = os.path.join(seed_output_dir, "dtw_visualization")
                if _should_skip_analysis(dtw_viz_dir):
                    print(f"    [skip] DTW visualization already exists at {dtw_viz_dir}")
                elif os.path.exists(config.behavior_labels_csv):
                    print(f"    Running DTW visualization...")
                    try:
                        run_dtw_visualization(
                            cluster_dict=cluster_dict,
                            cluster_labels=cluster_labels,
                            turn_directions=turn_directions,
                            behavior_labels_csv=config.behavior_labels_csv,
                            data=dtw_data,
                            rodent_rollout_dir=config.rodent_rollout_dir,
                            output_dir=dtw_viz_dir,
                            method_name=method_name,
                            seed=seed,
                            short_threshold=config.short_behavior_threshold,
                            merge_rear=config.merge_rear,
                            skip_root_dims=config.dtw_skip_root_dims,
                            min_seq_length=config.dtw_min_seq_length,
                            fps=config.gt_render_framerate,
                            target_size=config.gt_render_target_size,
                        )
                    except Exception as e:
                        print(f"    DTW visualization failed: {e}")
                        import traceback
                        traceback.print_exc()

            if config.run_mmd_analysis and dtw_data is not None:
                mmd_dir = os.path.join(seed_output_dir, "mmd")
                if _should_skip_analysis(mmd_dir):
                    print(f"    [skip] MMD analysis already exists at {mmd_dir}")
                elif os.path.exists(config.behavior_labels_csv):
                    try:
                        from .metrics import run_mmd_analysis
                        print(f"    Running MMD/KID analysis...")
                        run_mmd_analysis(
                            cluster_dict=cluster_dict,
                            cluster_labels=cluster_labels,
                            turn_directions=turn_directions,
                            behavior_labels_csv=config.behavior_labels_csv,
                            data=dtw_data,
                            output_dir=mmd_dir,
                            method_name=method_name,
                            seed=seed,
                            max_samples=config.dtw_max_seqs,
                            short_threshold=config.short_behavior_threshold,
                            merge_rear=config.merge_rear,
                            skip_root_dims=config.dtw_skip_root_dims,
                            min_seq_length=config.dtw_min_seq_length,
                            resample_K=config.c2st_window_frames,
                        )
                    except Exception as e:
                        print(f"    MMD/KID analysis failed: {e}")
                        import traceback
                        traceback.print_exc()

            if config.run_c2st_analysis:
                c2st_dir = os.path.join(seed_output_dir, "c2st")
                if _should_skip_analysis(c2st_dir):
                    print(f"    [skip] C2ST analysis already exists at {c2st_dir}")
                else:
                    try:
                        from topo_mimic.benchmark.c2st.run_c2st_analysis import (
                            run_c2st_for_method_seed,
                        )
                        print(f"    Running C2ST cluster analysis...")
                        run_c2st_for_method_seed(
                            method_dir=method_dir,
                            seed=seed,
                            raw_h5_path=config.c2st_raw_h5_path,
                            behavior_labels_csv=config.behavior_labels_csv,
                            output_dir=c2st_dir,
                            window_frames=config.c2st_window_frames,
                            min_cluster_size=config.c2st_min_cluster_size,
                            n_folds=config.c2st_n_folds,
                            models=tuple(config.c2st_models),
                        )
                    except Exception as e:
                        print(f"    C2ST cluster analysis failed: {e}")
                        import traceback
                        traceback.print_exc()

            if config.run_purity_analysis:
                purity_dir = os.path.join(seed_output_dir, "purity")
                if _should_skip_analysis(purity_dir):
                    print(f"    [skip] Purity analysis already exists at {purity_dir}")
                else:
                    try:
                        from topo_mimic.benchmark.analysis.purity import run_purity_analysis
                        print(f"    Running cluster purity analysis...")
                        run_purity_analysis(
                            cluster_dict=cluster_dict,
                            labels=cluster_labels,
                            behavior_labels_csv=config.behavior_labels_csv,
                            output_dir=purity_dir,
                            seed=seed,
                            method_name=method_name,
                        )
                    except Exception as e:
                        print(f"    Purity analysis failed: {e}")
                        import traceback
                        traceback.print_exc()

    if config.run_dtw_metrics and all_dtw_results:
        print("\n=== Generating DTW Comparison Table ===")
        create_dtw_comparison_table(all_dtw_results, config.output_dir)

    if config.run_dtw_metrics and all_consistency_results:
        print("\n=== Generating Consistency Comparison Table ===")
        create_consistency_comparison_table(all_consistency_results, config.output_dir)

    analyzed_methods = [m for m in all_results if not config.methods or m in config.methods]
    analyzed_seeds = config.seeds or [0]
    has_sample_labels = any(
        os.path.exists(os.path.join(config.experiment_dir, m, f"sample_labels_seed_{analyzed_seeds[0]}.yaml"))
        for m in analyzed_methods
    )
    if has_sample_labels:
        print("\n=== Cluster Classification Metrics (GT labels available) ===")
        from .cluster_classification import run_cluster_classification_analysis
        try:
            run_cluster_classification_analysis(
                experiment_dir=config.experiment_dir,
                methods=analyzed_methods,
                seeds=analyzed_seeds,
                output_dir=os.path.join(config.output_dir, "classification"),
            )
        except Exception as e:
            print(f"  Classification analysis failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Results saved to: {config.output_dir}")


def main(config_path: str = "configs/benchmark/analysis_transitions.yaml"):
    if os.path.exists(config_path):
        cfg = load_config(config_path)
    else:
        cfg = AnalysisConfig(experiment_dir="outputs/experiment")

    run_analysis(cfg)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/benchmark/analysis_transitions.yaml"
    main(config_path)
