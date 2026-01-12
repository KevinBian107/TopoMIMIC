"""
Unified analysis module for experiment results.

Provides variance heatmaps, MDS projections, cluster-to-label comparison,
cross-method comparison, and seed stability analysis.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import pickle as pkl
import yaml
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from vnl_tda.utils.load_utils import load_rollout_from_source
from vnl_tda.pipelines.tda.skeletonization.plotting import (
    plot_cluster_subspace_mds_separate,
    _compare_clusters_axis_motion_with_union,
)


QPOS_BODY_NAMES = [
    "world", "floor", "walker", "torso", "vertebra_1", "vertebra_2", "vertebra_3",
    "vertebra_4", "vertebra_5", "vertebra_6", "pelvis", "upper_leg_L", "lower_leg_L",
    "foot_L", "toe_L", "upper_leg_R", "lower_leg_R", "foot_R", "toe_R",
    "vertebra_C1", "vertebra_C2", "vertebra_C3", "vertebra_C4", "vertebra_C5",
    "vertebra_C6", "vertebra_C7", "vertebra_C8", "vertebra_C9", "vertebra_C10",
    "vertebra_C11", "vertebra_C12", "vertebra_C13", "vertebra_C14", "vertebra_C15",
    "vertebra_C16", "vertebra_C17", "vertebra_C18", "vertebra_C19", "vertebra_C20",
    "vertebra_C21", "vertebra_C22", "vertebra_C23", "vertebra_C24", "vertebra_C25",
    "vertebra_C26", "vertebra_C27", "vertebra_C28", "vertebra_C29", "vertebra_C30",
    "vertebra_cervical_5", "vertebra_cervical_4", "vertebra_cervical_3",
    "vertebra_cervical_2", "vertebra_cervical_1", "vertebra_axis", "vertebra_atlant",
    "skull", "jaw", "scapula_L", "upper_arm_L", "lower_arm_L", "hand_L", "finger_L",
    "scapula_R", "upper_arm_R", "lower_arm_R", "hand_R", "finger_R",
]


@dataclass
class AnalysisConfig:
    experiment_dir: str
    output_dir: str = "analysis_output"
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    behavior_csv: str = "/home/jovyan/vast/kaiwen/TopoVNL/behavior_book.csv"
    behavior_labels_csv: str = "/home/jovyan/vast/kaiwen/TopoVNL/behavior_labels.csv"

    run_variance_heatmap: bool = True
    run_mds_analysis: bool = True
    run_cluster_to_label_comparison: bool = True
    run_cross_method_comparison: bool = True
    run_seed_stability_analysis: bool = True

    mds_k_cycles: int = 50
    mds_s_top: int = 10
    mds_behaviors: list[str] = None

    body_parts: list[str] = None
    max_loops: int = 200

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


def load_config(config_path: str) -> AnalysisConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return AnalysisConfig(
        experiment_dir=cfg.get("experiment_dir", ""),
        output_dir=cfg.get("output_dir", "analysis_output"),
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", AnalysisConfig.rodent_rollout_dir),
        behavior_csv=cfg.get("behavior_csv", AnalysisConfig.behavior_csv),
        behavior_labels_csv=cfg.get("behavior_labels_csv", AnalysisConfig.behavior_labels_csv),
        run_variance_heatmap=cfg.get("run_variance_heatmap", True),
        run_mds_analysis=cfg.get("run_mds_analysis", True),
        run_cluster_to_label_comparison=cfg.get("run_cluster_to_label_comparison", True),
        run_cross_method_comparison=cfg.get("run_cross_method_comparison", True),
        run_seed_stability_analysis=cfg.get("run_seed_stability_analysis", True),
        mds_k_cycles=cfg.get("mds_k_cycles", 50),
        mds_s_top=cfg.get("mds_s_top", 10),
        mds_behaviors=cfg.get("mds_behaviors", None),
        body_parts=cfg.get("body_parts", None),
        max_loops=cfg.get("max_loops", 200),
    )


def load_experiment_results(experiment_dir: str) -> dict:
    results_path = os.path.join(experiment_dir, "all_results.pkl")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results not found: {results_path}")

    with open(results_path, "rb") as f:
        return pkl.load(f)


def load_method_results(method_dir: str) -> dict:
    results_path = os.path.join(method_dir, "results.pkl")
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            return pkl.load(f)
    return {}


def load_cluster_dict(method_dir: str, seed: int = 0, use_aligned: bool = True) -> dict:
    """
    Load cluster_dict for a specific seed.

    Args:
        method_dir: Path to method output directory
        seed: Random seed to load
        use_aligned: If True, prefer aligned cluster_dict for non-primary seeds

    Returns:
        Cluster dict mapping cluster_id to list of (clip_id, frames)
    """
    results = load_method_results(method_dir)
    if seed in results:
        result = results[seed]
        if use_aligned and "cluster_dict_aligned" in result:
            return result["cluster_dict_aligned"]
        return result.get("cluster_dict", {})
    return {}


def load_labels_yaml(method_dir: str) -> dict[int, str]:
    labels_path = os.path.join(method_dir, "labels.yaml")
    if not os.path.exists(labels_path):
        return {}

    with open(labels_path, "r") as f:
        data = yaml.safe_load(f)

    labels = data.get("labels", {})
    return {int(k): v for k, v in labels.items() if v}


def load_qpos(clip_id: int, rodent_rollout_dir: str) -> np.ndarray:
    roll = load_rollout_from_source(int(clip_id), rodent_rollout_dir)
    return np.asarray(roll["qpos"])


def load_behavior_labels(behavior_csv: str) -> dict[str, list[int]]:
    df = pd.read_csv(behavior_csv)
    return {
        "walk": df[df["walk"] == 1]["clip_idx"].tolist(),
        "rear": df[df["rear"] == 1]["clip_idx"].tolist(),
        "groom": df[(df["groom_down"] == 1) | (df["groom_up"] == 1)]["clip_idx"].tolist(),
    }


def plot_variance_heatmap(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    rodent_rollout_dir: str,
    output_dir: str,
    body_parts: list[str] = None,
    max_loops: int = 200,
    figsize: tuple[int, int] = (12, 8),
    method_name: str = "",
) -> None:
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

    cluster_ids = sorted(cluster_dict.keys())
    if cluster_labels:
        sorted_pairs = sorted(zip(cluster_ids, [cluster_labels.get(cid, f"C{cid}") for cid in cluster_ids]), key=lambda x: x[1])
        cluster_ids = [p[0] for p in sorted_pairs]

    n_clusters = len(cluster_ids)
    n_bodies = len(valid_body_parts)
    variance_matrix = np.zeros((n_clusters, n_bodies))

    for row_idx, cid in enumerate(tqdm(cluster_ids, desc="Computing variance")):
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

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(variance_matrix, aspect="auto", cmap="plasma")

    ax.set_xticks(np.arange(n_bodies))
    ax.set_xticklabels(body_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(n_clusters))
    ax.set_yticklabels(cluster_names, fontsize=10)

    ax.set_xlabel("Body Part", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)
    title = f"Pose Variance by Cluster"
    if method_name:
        title = f"{method_name}: {title}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Variance", fontsize=11)

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{method_name}_variance_heatmap.pdf" if method_name else "variance_heatmap.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved variance heatmap to {save_path}")


def get_ground_truth_frames(
    cluster_label: str,
    behavior_labels: dict[str, list[int]],
) -> tuple[list[int], int]:
    behaviors = cluster_label.split()

    if len(behaviors) == 1:
        clips = behavior_labels.get(behaviors[0].lower(), [])
        return clips, len(clips)

    if is_complementary(behaviors):
        frame_sets = [set(behavior_labels.get(b.lower(), [])) for b in behaviors]
        intersection = set.intersection(*frame_sets) if frame_sets else set()
        return sorted(intersection), len(intersection)

    if is_simultaneous_compound(behaviors):
        frame_sets = [set(behavior_labels.get(b.lower(), [])) for b in behaviors]
        intersection = set.intersection(*frame_sets) if frame_sets else set()
        return sorted(intersection), len(intersection)
    else:
        frame_sets = [set(behavior_labels.get(b.lower(), [])) for b in behaviors]
        union = set.union(*frame_sets) if frame_sets else set()
        return sorted(union), len(union)


def is_complementary(behaviors: list[str]) -> bool:
    complementary_pairs = [
        {"left_turn", "right_turn"},
        {"left", "right"},
    ]
    behavior_set = set(b.lower() for b in behaviors)
    return any(behavior_set == pair for pair in complementary_pairs)


def is_simultaneous_compound(behaviors: list[str]) -> bool:
    modifiers = {"left_turn", "right_turn", "slow", "fast", "turn"}
    return any(b.lower() in modifiers for b in behaviors)


def plot_cluster_to_label_comparison(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    behavior_labels: dict[str, list[int]],
    rodent_rollout_dir: str,
    output_dir: str,
    body_parts: list[str] = None,
    max_loops: int = 200,
    method_name: str = "",
) -> dict:
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

    results = {}

    for cid, label in cluster_labels.items():
        if not label:
            continue

        entries = cluster_dict.get(cid, [])
        n_cluster_frames = sum(len(frames) for _, frames in entries[:max_loops])

        gt_clips, n_gt_clips = get_ground_truth_frames(label, behavior_labels)

        results[cid] = {
            "label": label,
            "n_cluster_frames": n_cluster_frames,
            "n_ground_truth_clips": n_gt_clips,
        }

        print(f"Cluster {cid} ({label}): {n_cluster_frames} frames, {n_gt_clips} GT clips")

    csv_path = os.path.join(output_dir, f"{method_name}_comparison.csv" if method_name else "comparison.csv")
    df = pd.DataFrame([
        {"cluster_id": k, **v} for k, v in results.items()
    ])
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison to {csv_path}")

    return results


def run_cross_method_comparison(
    experiment_dir: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    all_results = load_experiment_results(experiment_dir)

    methods = list(all_results.keys())
    if len(methods) < 2:
        print("Need at least 2 methods for comparison")
        return

    alignment_per_method = {}

    for method_name, method_results in all_results.items():
        align_sum = method_results.get("alignment_summary", {})
        if align_sum:
            alignment_per_method[method_name] = align_sum.get("mean_score", 0.0)

    if not alignment_per_method:
        print("No alignment data available for cross-method comparison")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    method_names = list(alignment_per_method.keys())
    x_pos = np.arange(len(method_names))
    scores = [alignment_per_method[m] * 100 for m in method_names]

    colors = ['green' if s >= 90 else 'yellowgreen' if s >= 70 else 'orange' if s >= 50 else 'red' for s in scores]
    bars = ax.bar(x_pos, scores, color=colors, alpha=0.7, edgecolor='black')

    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Excellent (90%)')
    ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Good (70%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel("Mean Alignment Score (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title("Cross-Method Comparison (Cluster Alignment)", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "cross_method_comparison.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-method comparison to {save_path}")


def run_seed_stability_analysis(
    experiment_dir: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    all_results = load_experiment_results(experiment_dir)

    method_names = []
    alignment_scores = []

    for method_name, method_results in all_results.items():
        seeds = [k for k in method_results.keys() if isinstance(k, int)]
        if len(seeds) < 2:
            continue

        align_sum = method_results.get("alignment_summary", {})
        if align_sum:
            method_names.append(method_name)
            alignment_scores.append(align_sum.get("mean_score", 0.0))

    if not method_names:
        print("No alignment data available for seed stability analysis")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(method_names))
    colors = ['green' if s >= 0.9 else 'yellowgreen' if s >= 0.7 else 'orange' if s >= 0.5 else 'red' for s in alignment_scores]
    bars = ax.bar(x_pos, [s * 100 for s in alignment_scores], color=colors, alpha=0.7, edgecolor='black')

    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Excellent (90%)')
    ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Good (70%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Mean Alignment Score (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title("Cluster Alignment Across Seeds (Higher = More Consistent)", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "seed_stability.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved seed stability analysis to {save_path}")


def run_analysis(config: AnalysisConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("Unified Analysis Pipeline")
    print("=" * 60)

    all_results = load_experiment_results(config.experiment_dir)
    print(f"Loaded results for {len(all_results)} methods")

    behavior_labels = None
    if config.run_cluster_to_label_comparison:
        if os.path.exists(config.behavior_csv):
            behavior_labels = load_behavior_labels(config.behavior_csv)
            print(f"Loaded behavior labels from {config.behavior_csv}")

    for method_name, method_results in all_results.items():
        print(f"\n=== Analyzing: {method_name} ===")

        method_dir = os.path.join(config.experiment_dir, method_name)
        method_output_dir = os.path.join(config.output_dir, method_name)
        os.makedirs(method_output_dir, exist_ok=True)

        primary_seed = list(method_results.keys())[0]
        if isinstance(primary_seed, str):
            for k in method_results.keys():
                if isinstance(k, int):
                    primary_seed = k
                    break

        result = method_results.get(primary_seed, {})
        cluster_dict = result.get("cluster_dict", {})

        if not cluster_dict:
            print(f"  No cluster_dict found, skipping")
            continue

        cluster_labels = load_labels_yaml(method_dir)
        if not cluster_labels:
            cluster_labels = {i: f"C{i}" for i in cluster_dict.keys()}

        if config.run_variance_heatmap:
            print(f"  Running variance heatmap...")
            plot_variance_heatmap(
                cluster_dict=cluster_dict,
                cluster_labels=cluster_labels,
                rodent_rollout_dir=config.rodent_rollout_dir,
                output_dir=method_output_dir,
                body_parts=config.body_parts,
                max_loops=config.max_loops,
                method_name=method_name,
            )

        if config.run_mds_analysis:
            if os.path.exists(config.behavior_labels_csv):
                print(f"  Running MDS analysis (separate behaviors)...")
                target_clusters = list(cluster_dict.keys())
                mds_save_path = os.path.join(method_output_dir, f"{method_name}_mds_separate.pdf")
                try:
                    plot_cluster_subspace_mds_separate(
                        cluster_cycles_by_id=cluster_dict,
                        cluster_labels=cluster_labels,
                        target_clusters=target_clusters,
                        rodent_rollout_dir=config.rodent_rollout_dir,
                        behavior_csv_path=config.behavior_labels_csv,
                        behaviors=config.mds_behaviors,
                        k_cycles=config.mds_k_cycles,
                        s_top=config.mds_s_top,
                        save_path=mds_save_path,
                    )
                    print(f"  Saved MDS plot to {mds_save_path}")
                except Exception as e:
                    print(f"  MDS analysis failed: {e}")
            else:
                print(f"  Skipping MDS: behavior_labels_csv not found at {config.behavior_labels_csv}")

        if config.run_cluster_to_label_comparison and behavior_labels:
            print(f"  Running cluster-to-label comparison...")
            plot_cluster_to_label_comparison(
                cluster_dict=cluster_dict,
                cluster_labels=cluster_labels,
                behavior_labels=behavior_labels,
                rodent_rollout_dir=config.rodent_rollout_dir,
                output_dir=method_output_dir,
                body_parts=config.body_parts,
                max_loops=config.max_loops,
                method_name=method_name,
            )

    if config.run_cross_method_comparison and len(all_results) > 1:
        print("\n=== Running cross-method comparison ===")
        run_cross_method_comparison(config.experiment_dir, config.output_dir)

    if config.run_seed_stability_analysis:
        print("\n=== Running seed stability analysis ===")
        run_seed_stability_analysis(config.experiment_dir, config.output_dir)

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
