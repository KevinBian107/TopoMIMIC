"""
Benchmark Analysis Pipeline.
Compares variance differences between labeled behaviors and each clustering method.

For each method:
1. Group clusters by behavior keyword (walk, rear, groom)
2. Compute variance on cluster cycle frames
3. Compare to labeled clip variance (ground truth from behavior_book.csv)
4. Output: mean absolute difference as single metric per method
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import numpy as np
import pickle as pkl
import yaml
import h5py
from dataclasses import dataclass, field
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd

from vnl_tda.utils.load_utils import load_from_h5py


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
class MethodConfig:
    name: str
    results_dir: str
    n_clusters: int
    cluster_labels: dict[int, str] = field(default_factory=dict)


@dataclass
class BenchmarkAnalysisConfig:
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    behavior_csv: str = "/home/jovyan/vast/kaiwen/TopoVNL/behavior_book.csv"
    output_dir: str = "out_benchmark_analysis"
    behavior_categories: list[str] = field(default_factory=lambda: ["walk", "rear", "groom"])
    body_parts: list[str] = None
    max_loops: int = 200
    methods: list[MethodConfig] = field(default_factory=list)

    def __post_init__(self):
        if self.body_parts is None:
            self.body_parts = [
                "upper_leg_R", "lower_leg_R", "upper_arm_R",
                "upper_leg_L", "lower_leg_L", "upper_arm_L",
                "skull", "torso", "pelvis",
            ]


def load_config(config_path: str) -> BenchmarkAnalysisConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    methods_list = []
    for m in cfg.get("methods", []):
        cluster_labels = m.get("cluster_labels", {})
        if cluster_labels:
            cluster_labels = {int(k): v for k, v in cluster_labels.items()}
        methods_list.append(MethodConfig(
            name=m["name"],
            results_dir=m["results_dir"],
            n_clusters=m.get("n_clusters", 10),
            cluster_labels=cluster_labels,
        ))

    return BenchmarkAnalysisConfig(
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", BenchmarkAnalysisConfig.rodent_rollout_dir),
        behavior_csv=cfg.get("behavior_csv", BenchmarkAnalysisConfig.behavior_csv),
        output_dir=cfg.get("output_dir", BenchmarkAnalysisConfig.output_dir),
        behavior_categories=cfg.get("behavior_categories", ["walk", "rear", "groom"]),
        body_parts=cfg.get("body_parts", None),
        max_loops=cfg.get("max_loops", 200),
        methods=methods_list,
    )


def load_cluster_dict(results_dir: str) -> dict | None:
    """Load cluster dictionary from results directory."""
    possible_paths = [
        os.path.join(results_dir, "clustering", "cluster_dict.pkl"),
        os.path.join(results_dir, "cluster_dict.pkl"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pkl.load(f)

    print(f"  Warning: No cluster_dict.pkl found in {results_dir}")
    return None


def load_qpos(clip_id: int, rodent_rollout_dir: str) -> np.ndarray:
    """Load qpos data for a clip."""
    with h5py.File(f"{rodent_rollout_dir}/clip_{int(clip_id)}.h5", "r") as f:
        roll = load_from_h5py(f)
    return np.asarray(roll["qpos"])


def load_behavior_labels(behavior_csv: str) -> dict[str, list[int]]:
    """Load behavior labels from CSV file."""
    df = pd.read_csv(behavior_csv)

    behavior_to_clips = {
        "walk": df[df["walk"] == 1]["clip_idx"].tolist(),
        "rear": df[df["rear"] == 1]["clip_idx"].tolist(),
        "groom": df[(df["groom_down"] == 1) | (df["groom_up"] == 1)]["clip_idx"].tolist(),
    }

    return behavior_to_clips


def group_clusters_by_behavior(
    cluster_labels: dict[int, str],
    behavior_categories: list[str],
) -> dict[str, list[int]]:
    """Group cluster IDs by behavioral category based on keyword matching."""
    grouped = {cat: [] for cat in behavior_categories}

    for cid, label in cluster_labels.items():
        label_lower = label.lower()
        for cat in behavior_categories:
            if cat in label_lower:
                grouped[cat].append(cid)
                break

    return grouped


def compute_variance_for_labeled_clips(
    clip_ids: list[int],
    rodent_rollout_dir: str,
    body_indices: list[int],
    valid_body_parts: list[str],
) -> np.ndarray:
    """Compute variance per body part for labeled clips (full clip, all frames)."""
    n_bodies = len(valid_body_parts)
    all_qpos_by_body = {bp: [] for bp in valid_body_parts}

    for clip_id in clip_ids:
        try:
            qpos = load_qpos(clip_id, rodent_rollout_dir)
        except:
            continue

        for col_idx, (bp, body_idx) in enumerate(zip(valid_body_parts, body_indices)):
            if body_idx < qpos.shape[1]:
                all_qpos_by_body[bp].append(qpos[:, body_idx])

    variances = np.zeros(n_bodies)
    for col_idx, bp in enumerate(valid_body_parts):
        if all_qpos_by_body[bp]:
            combined = np.concatenate(all_qpos_by_body[bp])
            variances[col_idx] = np.var(combined)

    return variances


def compute_variance_for_cluster_group(
    cluster_ids: list[int],
    cluster_dict: dict,
    rodent_rollout_dir: str,
    body_indices: list[int],
    valid_body_parts: list[str],
    max_loops: int = 200,
) -> np.ndarray:
    """Compute variance per body part for clusters (cycle frames only)."""
    n_bodies = len(valid_body_parts)
    all_qpos_by_body = {bp: [] for bp in valid_body_parts}

    for cid in cluster_ids:
        if cid not in cluster_dict:
            continue

        entries = cluster_dict[cid]

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

    variances = np.zeros(n_bodies)
    for col_idx, bp in enumerate(valid_body_parts):
        if all_qpos_by_body[bp]:
            combined = np.concatenate(all_qpos_by_body[bp])
            variances[col_idx] = np.var(combined)

    return variances


def compute_method_variance_diff(
    method_cfg: MethodConfig,
    labeled_variance: np.ndarray,
    labeled_clips: dict[str, list[int]],
    cfg: BenchmarkAnalysisConfig,
    body_indices: list[int],
    valid_body_parts: list[str],
) -> tuple[float, np.ndarray, np.ndarray, dict[str, list[int]]]:
    """
    Compute variance difference for a single method.

    Returns:
        Tuple of (mean_abs_diff, cluster_variance_matrix, diff_matrix, cluster_groups)
    """
    cluster_dict = load_cluster_dict(method_cfg.results_dir)

    if cluster_dict is None:
        n_categories = len(cfg.behavior_categories)
        n_bodies = len(valid_body_parts)
        return np.nan, np.zeros((n_categories, n_bodies)), np.zeros((n_categories, n_bodies)), {}

    if not method_cfg.cluster_labels:
        method_cfg.cluster_labels = {i: f"cluster_{i}" for i in cluster_dict.keys()}

    cluster_groups = group_clusters_by_behavior(
        method_cfg.cluster_labels,
        cfg.behavior_categories,
    )

    n_categories = len(cfg.behavior_categories)
    n_bodies = len(valid_body_parts)
    cluster_variance = np.zeros((n_categories, n_bodies))

    for row_idx, cat in enumerate(cfg.behavior_categories):
        cluster_ids = cluster_groups.get(cat, [])
        if cluster_ids:
            cluster_variance[row_idx] = compute_variance_for_cluster_group(
                cluster_ids, cluster_dict, cfg.rodent_rollout_dir,
                body_indices, valid_body_parts, cfg.max_loops
            )

    with np.errstate(divide='ignore', invalid='ignore'):
        diff_matrix = (cluster_variance - labeled_variance) / np.where(
            labeled_variance > 1e-10, labeled_variance, 1.0
        )
        diff_matrix = np.nan_to_num(diff_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    mean_abs_diff = np.mean(np.abs(diff_matrix))

    return mean_abs_diff, cluster_variance, diff_matrix, cluster_groups


def plot_comparison_bar_chart(
    method_names: list[str],
    mean_diffs: list[float],
    output_dir: str,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Plot bar chart comparing mean variance differences across methods."""
    os.makedirs(output_dir, exist_ok=True)

    valid_indices = [i for i, d in enumerate(mean_diffs) if not np.isnan(d)]
    valid_names = [method_names[i] for i in valid_indices]
    valid_diffs = [mean_diffs[i] for i in valid_indices]

    sorted_pairs = sorted(zip(valid_names, valid_diffs), key=lambda x: x[1])
    sorted_names = [p[0] for p in sorted_pairs]
    sorted_diffs = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_names)))
    bars = ax.barh(range(len(sorted_names)), sorted_diffs, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlabel("Mean Absolute Normalized Variance Difference", fontsize=12)
    ax.set_title("Clustering Method Comparison\n(Lower = Closer to Labeled Behavior Variance)", fontsize=14, fontweight='bold')

    for i, (bar, val) in enumerate(zip(bars, sorted_diffs)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, axis='x', alpha=0.3)

    ax.set_xlim(0, max(sorted_diffs) * 1.15)

    plt.tight_layout()

    save_path = os.path.join(output_dir, "method_comparison_bar.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison bar chart to {save_path}")


def plot_method_variance_heatmap(
    method_name: str,
    labeled_variance: np.ndarray,
    cluster_variance: np.ndarray,
    diff_matrix: np.ndarray,
    behavior_categories: list[str],
    valid_body_parts: list[str],
    output_dir: str,
    figsize: tuple[int, int] = (16, 5),
) -> None:
    """Plot variance comparison heatmaps for a single method."""
    os.makedirs(output_dir, exist_ok=True)

    body_labels = [bp.replace("_", " ") for bp in valid_body_parts]
    category_labels = [cat.capitalize() for cat in behavior_categories]

    n_categories = len(behavior_categories)
    n_bodies = len(valid_body_parts)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    im1 = axes[0].imshow(labeled_variance, aspect="auto", cmap="plasma")
    axes[0].set_xticks(np.arange(n_bodies))
    axes[0].set_xticklabels(body_labels, rotation=45, ha="right", fontsize=10)
    axes[0].set_yticks(np.arange(n_categories))
    axes[0].set_yticklabels(category_labels, fontsize=11)
    axes[0].set_xlabel("Body Part", fontsize=12)
    axes[0].set_ylabel("Behavior", fontsize=12)
    axes[0].set_title("Labeled Data Variance", fontsize=14, fontweight="bold")
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
    cbar1.set_label("Variance", fontsize=11)

    im2 = axes[1].imshow(cluster_variance, aspect="auto", cmap="plasma")
    axes[1].set_xticks(np.arange(n_bodies))
    axes[1].set_xticklabels(body_labels, rotation=45, ha="right", fontsize=10)
    axes[1].set_yticks(np.arange(n_categories))
    axes[1].set_yticklabels(category_labels, fontsize=11)
    axes[1].set_xlabel("Body Part", fontsize=12)
    axes[1].set_ylabel("Behavior", fontsize=12)
    axes[1].set_title(f"{method_name}\nCluster Variance", fontsize=14, fontweight="bold")
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
    cbar2.set_label("Variance", fontsize=11)

    vmax = np.abs(diff_matrix).max()
    vmax = min(vmax, 2.0)
    im3 = axes[2].imshow(diff_matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[2].set_xticks(np.arange(n_bodies))
    axes[2].set_xticklabels(body_labels, rotation=45, ha="right", fontsize=10)
    axes[2].set_yticks(np.arange(n_categories))
    axes[2].set_yticklabels(category_labels, fontsize=11)
    axes[2].set_xlabel("Body Part", fontsize=12)
    axes[2].set_ylabel("Behavior", fontsize=12)
    axes[2].set_title("Normalized Difference\n(Cluster - Labeled) / Labeled", fontsize=14, fontweight="bold")
    cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8)
    cbar3.set_label("Relative Diff", fontsize=11)

    plt.tight_layout()

    safe_name = method_name.replace(" ", "_").replace("+", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(output_dir, f"{safe_name}_variance_heatmap.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap to {save_path}")


def main(config_path: str = "configs/benchmark_analysis.yaml"):
    cfg = load_config(config_path)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print("Benchmark Analysis Pipeline")
    print("=" * 60)

    body_indices = []
    valid_body_parts = []
    for bp in cfg.body_parts:
        if bp in QPOS_BODY_NAMES:
            body_indices.append(QPOS_BODY_NAMES.index(bp))
            valid_body_parts.append(bp)

    print(f"\nBody parts: {valid_body_parts}")
    print(f"Behavior categories: {cfg.behavior_categories}")

    print("\n=== Stage 1: Loading labeled behavior clips ===")
    labeled_clips = load_behavior_labels(cfg.behavior_csv)
    for cat, clips in labeled_clips.items():
        print(f"  {cat}: {len(clips)} clips")

    print("\n=== Stage 2: Computing labeled variance ===")
    n_categories = len(cfg.behavior_categories)
    n_bodies = len(valid_body_parts)
    labeled_variance = np.zeros((n_categories, n_bodies))

    for row_idx, cat in enumerate(tqdm(cfg.behavior_categories, desc="Labeled variance")):
        clip_ids = labeled_clips.get(cat, [])
        if clip_ids:
            labeled_variance[row_idx] = compute_variance_for_labeled_clips(
                clip_ids, cfg.rodent_rollout_dir, body_indices, valid_body_parts
            )

    print("\n=== Stage 3: Computing method variance differences ===")
    method_names = []
    mean_diffs = []
    all_results = {}

    for method_cfg in cfg.methods:
        print(f"\nProcessing: {method_cfg.name}")
        print(f"  Results dir: {method_cfg.results_dir}")

        mean_diff, cluster_variance, diff_matrix, cluster_groups = compute_method_variance_diff(
            method_cfg,
            labeled_variance,
            labeled_clips,
            cfg,
            body_indices,
            valid_body_parts,
        )

        method_names.append(method_cfg.name)
        mean_diffs.append(mean_diff)

        all_results[method_cfg.name] = {
            "mean_abs_diff": mean_diff,
            "cluster_variance": cluster_variance,
            "diff_matrix": diff_matrix,
            "cluster_groups": cluster_groups,
        }

        print(f"  Cluster groups: {cluster_groups}")
        print(f"  Mean absolute diff: {mean_diff:.4f}")

        if not np.isnan(mean_diff):
            plot_method_variance_heatmap(
                method_cfg.name,
                labeled_variance,
                cluster_variance,
                diff_matrix,
                cfg.behavior_categories,
                valid_body_parts,
                cfg.output_dir,
            )

    print("\n=== Stage 4: Generating comparison plots ===")
    plot_comparison_bar_chart(method_names, mean_diffs, cfg.output_dir)

    print("\n=== Stage 5: Saving results ===")
    results_path = os.path.join(cfg.output_dir, "benchmark_analysis_results.pkl")
    with open(results_path, "wb") as f:
        pkl.dump({
            "labeled_variance": labeled_variance,
            "all_results": all_results,
            "behavior_categories": cfg.behavior_categories,
            "body_parts": valid_body_parts,
        }, f)
    print(f"Saved results to {results_path}")

    summary_path = os.path.join(cfg.output_dir, "benchmark_analysis_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Benchmark Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Comparison: Cluster variance vs Labeled clip variance\n")
        f.write(f"Behaviors: {cfg.behavior_categories}\n")
        f.write(f"Body parts: {valid_body_parts}\n\n")

        f.write("Ranking (lower = closer to labeled variance):\n")
        f.write("-" * 40 + "\n")

        valid_pairs = [(n, d) for n, d in zip(method_names, mean_diffs) if not np.isnan(d)]
        sorted_pairs = sorted(valid_pairs, key=lambda x: x[1])

        for rank, (name, diff) in enumerate(sorted_pairs, 1):
            f.write(f"{rank}. {name}: {diff:.4f}\n")

    print(f"Saved summary to {summary_path}")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

    print("\nFinal Rankings (lower = better):")
    valid_pairs = [(n, d) for n, d in zip(method_names, mean_diffs) if not np.isnan(d)]
    sorted_pairs = sorted(valid_pairs, key=lambda x: x[1])
    for rank, (name, diff) in enumerate(sorted_pairs, 1):
        print(f"  {rank}. {name}: {diff:.4f}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/benchmark_analysis.yaml"
    main(config_path)
