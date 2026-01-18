"""Main analysis runner for transition cluster comparison."""

import os
import sys

import numpy as np

from .config import AnalysisConfig, load_config
from .data_loading import (
    load_experiment_results,
    load_cluster_dict,
    load_labels_yaml,
    load_method_results,
    load_behavior_labels_legacy,
)
from .transition_ranking import run_transition_ranking_analysis
from .visualization import (
    plot_transition_cluster_comparison,
    plot_variance_heatmap,
)
from .javd import compute_javd_table


def plot_variance_heatmaps_side_by_side(
    cluster_dicts: dict[int, dict],
    cluster_labels_per_seed: dict[int, dict[int, str]],
    rodent_rollout_dir: str,
    output_dir: str,
    body_parts: list[str],
    max_loops: int,
    method_name: str,
) -> None:
    """Plot variance heatmaps for multiple seeds side by side."""
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from .data_loading import load_qpos, QPOS_BODY_NAMES

    os.makedirs(output_dir, exist_ok=True)

    if body_parts is None:
        body_parts = [
            "torso", "pelvis",
            "upper_leg_L", "lower_leg_L", "foot_L",
            "upper_leg_R", "lower_leg_R", "foot_R",
            "skull", "jaw",
            "upper_arm_L", "lower_arm_L",
            "upper_arm_R", "lower_arm_R",
        ]

    body_indices = []
    valid_body_parts = []
    for bp in body_parts:
        if bp in QPOS_BODY_NAMES:
            body_indices.append(QPOS_BODY_NAMES.index(bp))
            valid_body_parts.append(bp)

    seeds = sorted(cluster_dicts.keys())
    n_seeds = len(seeds)

    if n_seeds < 2:
        print("  Need at least 2 seeds for side-by-side comparison")
        return

    fig, axes = plt.subplots(1, n_seeds, figsize=(6 * n_seeds, 8), sharey=True)
    if n_seeds == 1:
        axes = [axes]

    for ax_idx, seed in enumerate(seeds):
        cluster_dict = cluster_dicts[seed]
        cluster_labels = cluster_labels_per_seed.get(seed, {})

        cluster_ids = sorted(cluster_dict.keys())
        if cluster_labels:
            sorted_pairs = sorted(
                zip(cluster_ids, [cluster_labels.get(cid, f"C{cid}") for cid in cluster_ids]),
                key=lambda x: x[1]
            )
            cluster_ids = [p[0] for p in sorted_pairs]

        n_clusters = len(cluster_ids)
        n_bodies = len(valid_body_parts)
        variance_matrix = np.zeros((n_clusters, n_bodies))

        for row_idx, cid in enumerate(cluster_ids):
            entries = cluster_dict.get(cid, [])
            all_qpos_by_body = {bp: [] for bp in valid_body_parts}

            for i, (clip_id, frames) in enumerate(entries):
                if i >= max_loops:
                    break
                try:
                    qpos = load_qpos(clip_id, rodent_rollout_dir)
                except:
                    continue

                frames = np.asarray(frames, dtype=int)
                frames = frames[frames < len(qpos)]
                if len(frames) < 2:
                    continue

                for col_idx, (bp, body_idx) in enumerate(zip(valid_body_parts, body_indices)):
                    if body_idx < qpos.shape[1]:
                        all_qpos_by_body[bp].append(qpos[frames, body_idx])

            for col_idx, bp in enumerate(valid_body_parts):
                if all_qpos_by_body[bp]:
                    combined = np.concatenate(all_qpos_by_body[bp])
                    variance_matrix[row_idx, col_idx] = np.var(combined)

        cluster_names = [cluster_labels.get(cid, f"C{cid}").replace("_", " ") for cid in cluster_ids]
        body_labels = [bp.replace("_", " ") for bp in valid_body_parts]

        ax = axes[ax_idx]
        im = ax.imshow(variance_matrix, aspect="auto", cmap="plasma")

        ax.set_xticks(np.arange(n_bodies))
        ax.set_xticklabels(body_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(np.arange(n_clusters))
        ax.set_yticklabels(cluster_names, fontsize=9)

        ax.set_xlabel("Body Part", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Cluster", fontsize=11)
        ax.set_title(f"Seed {seed}", fontsize=12, fontweight="bold")

    title = f"{method_name}: Variance Heatmaps by Seed" if method_name else "Variance Heatmaps by Seed"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Variance")

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    variance_dir = os.path.join(output_dir, "variance")
    os.makedirs(variance_dir, exist_ok=True)

    save_path = os.path.join(variance_dir, f"{method_name}_variance_heatmap_seeds.pdf" if method_name else "variance_heatmap_seeds.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved side-by-side variance heatmaps to {save_path}")


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

    behavior_labels = None
    if config.run_cluster_to_label_comparison:
        if os.path.exists(config.behavior_csv):
            behavior_labels = load_behavior_labels_legacy(config.behavior_csv)
            print(f"Loaded behavior labels from {config.behavior_csv}")

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

        for seed in seeds:
            result = method_results.get(seed, {})
            cluster_dict = result.get("cluster_dict", {})

            if not cluster_dict:
                print(f"  Seed {seed}: No cluster_dict found, skipping")
                continue

            cluster_dicts[seed] = cluster_dict
            cluster_labels = load_labels_yaml(method_dir, seed)
            if not cluster_labels:
                cluster_labels = {i: f"C{i}" for i in cluster_dict.keys()}
            cluster_labels_per_seed[seed] = cluster_labels

            seed_output_dir = os.path.join(method_output_dir, f"seed_{seed}")
            os.makedirs(seed_output_dir, exist_ok=True)

            print(f"  Seed {seed}:")

            if config.run_variance_heatmap:
                print(f"    Running variance heatmap...")
                plot_variance_heatmap(
                    cluster_dict=cluster_dict,
                    cluster_labels=cluster_labels,
                    rodent_rollout_dir=config.rodent_rollout_dir,
                    output_dir=seed_output_dir,
                    body_parts=config.body_parts,
                    max_loops=config.max_loops,
                    method_name=f"{method_name}_seed{seed}",
                )

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

            if config.run_javd_analysis:
                if os.path.exists(config.behavior_labels_csv):
                    print(f"    Running JAVD analysis...")
                    try:
                        compute_javd_table(
                            cluster_dict=cluster_dict,
                            cluster_labels=cluster_labels,
                            behavior_labels_csv=config.behavior_labels_csv,
                            rodent_rollout_dir=config.rodent_rollout_dir,
                            output_dir=seed_output_dir,
                            body_parts=config.body_parts,
                            top_k=config.top_k_variance_joints,
                            max_loops=config.max_loops,
                            method_name=method_name,
                            seed=seed,
                            short_threshold=config.short_behavior_threshold,
                            max_gap=config.max_gap_between_segments,
                            merge_rear=config.merge_rear,
                            render_gt=config.render_gt_transitions,
                            gt_render_max_samples=config.gt_render_max_samples,
                            gt_render_framerate=config.gt_render_framerate,
                            gt_render_target_size=config.gt_render_target_size,
                        )
                    except Exception as e:
                        print(f"    JAVD analysis failed: {e}")

        if config.run_seed_comparison and len(cluster_dicts) > 1:
            print(f"  Running side-by-side seed comparison...")
            plot_variance_heatmaps_side_by_side(
                cluster_dicts=cluster_dicts,
                cluster_labels_per_seed=cluster_labels_per_seed,
                rodent_rollout_dir=config.rodent_rollout_dir,
                output_dir=method_output_dir,
                body_parts=config.body_parts,
                max_loops=config.max_loops,
                method_name=method_name,
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
