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
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import pandas as pd
import ripser
from persim import wasserstein

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


def kpms_labels_to_cluster_dict(labels_list: list[np.ndarray]) -> dict[int, list[tuple[int, np.ndarray]]]:
    """
    Convert KPMS syllable labels to cluster_dict format.

    Args:
        labels_list: List of syllable arrays per clip (n_clips x n_frames)

    Returns:
        cluster_dict[syllable_id] = [(clip_id, frame_indices), ...]
    """
    cluster_dict = {}
    for clip_idx, syllables in enumerate(labels_list):
        syllables = np.asarray(syllables)
        unique_sylls = np.unique(syllables)

        for syll in unique_sylls:
            syll = int(syll)
            frame_indices = np.where(syllables == syll)[0]

            if syll not in cluster_dict:
                cluster_dict[syll] = []
            cluster_dict[syll].append((clip_idx, frame_indices))

    return cluster_dict


def load_cluster_dict(results_dir: str, method_name: str = None) -> dict | None:
    """Load cluster dictionary from results directory or benchmark results file."""
    possible_paths = [
        os.path.join(results_dir, "clustering", "cluster_dict.pkl"),
        os.path.join(results_dir, "cluster_dict.pkl"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pkl.load(f)

    # Try loading from parent benchmark_results.pkl
    parent_dir = os.path.dirname(results_dir)
    benchmark_results_path = os.path.join(parent_dir, "benchmark_results.pkl")
    if os.path.exists(benchmark_results_path) and method_name:
        with open(benchmark_results_path, "rb") as f:
            all_results = pkl.load(f)

        # Try to find the method by matching name patterns
        for key, result in all_results.items():
            # Match by exact name or by directory basename
            dir_basename = os.path.basename(results_dir)
            if key == method_name or key == dir_basename or dir_basename in key.lower().replace(" ", "_"):
                if "cluster_dict" in result:
                    return result["cluster_dict"]

        # Try matching by directory name to method key
        name_mappings = {
            "vanilla_kmeans": "vanilla_kmeans",
            "pca_kmeans": "pca_kmeans",
            "tsne_kmeans": "tsne_kmeans",
            "umap_kmeans": "umap_kmeans",
            "dbscan": "dbscan",
            "spectral": "spectral",
        }
        dir_basename = os.path.basename(results_dir)
        if dir_basename in name_mappings:
            for key, result in all_results.items():
                if name_mappings[dir_basename] in key.lower().replace(" ", "_").replace("+", "_"):
                    if "cluster_dict" in result:
                        return result["cluster_dict"]

    # Try loading from KPMS benchmark results
    kpms_results_path = os.path.join(results_dir, "kpms_benchmark_results_full.pkl")
    if os.path.exists(kpms_results_path) and method_name:
        with open(kpms_results_path, "rb") as f:
            kpms_results = pkl.load(f)

        method_key = method_name.lower().replace("-", "").replace(" ", "")
        for key in ["arhmm", "slds"]:
            if key in method_key and key in kpms_results:
                optimal_k = kpms_results[key].get("optimal_k")
                if optimal_k and "all_labels" in kpms_results[key]:
                    labels_list = kpms_results[key]["all_labels"].get(optimal_k)
                    if labels_list is not None:
                        return kpms_labels_to_cluster_dict(labels_list)

    # Try loading from parent directory KPMS results
    parent_kpms_path = os.path.join(parent_dir, "kpms_benchmark_results_full.pkl")
    if os.path.exists(parent_kpms_path) and method_name:
        with open(parent_kpms_path, "rb") as f:
            kpms_results = pkl.load(f)

        method_key = method_name.lower().replace("-", "").replace(" ", "")
        for key in ["arhmm", "slds"]:
            if key in method_key and key in kpms_results:
                optimal_k = kpms_results[key].get("optimal_k")
                if optimal_k and "all_labels" in kpms_results[key]:
                    labels_list = kpms_results[key]["all_labels"].get(optimal_k)
                    if labels_list is not None:
                        return kpms_labels_to_cluster_dict(labels_list)

    print(f"  Warning: No cluster_dict found for {results_dir}")
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
    cluster_dict = load_cluster_dict(method_cfg.results_dir, method_cfg.name)

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


def collect_labeled_behavior_points(
    labeled_clips: dict[str, list[int]],
    rodent_rollout_dir: str,
) -> dict[str, np.ndarray]:
    """
    Collect mean qpos for each labeled behavior clip.

    Returns:
        Dict mapping behavior_name to (N, D) array of mean qpos per clip
    """
    behavior_points = {}

    for behavior, clip_ids in labeled_clips.items():
        points = []
        for clip_id in clip_ids:
            try:
                with h5py.File(f"{rodent_rollout_dir}/clip_{int(clip_id)}.h5", "r") as f:
                    roll = load_from_h5py(f)
                qpos = np.asarray(roll["qpos"])
                points.append(qpos.mean(axis=0))
            except:
                continue

        if points:
            behavior_points[behavior] = np.array(points)

    return behavior_points


def collect_cluster_points_by_behavior(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    rodent_rollout_dir: str,
    behavior_categories: list[str],
    max_loops: int = 200,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Collect cluster points grouped by behavior category.

    Returns:
        Dict mapping behavior_category to {cluster_label: points_array}
    """
    behavior_clusters = {cat: {} for cat in behavior_categories}

    for cid, entries in cluster_dict.items():
        label = cluster_labels.get(cid, f"cluster_{cid}")
        label_lower = label.lower()

        matched_behavior = None
        for cat in behavior_categories:
            if cat in label_lower:
                matched_behavior = cat
                break

        if matched_behavior is None:
            continue

        points = []
        for i, (clip_id, frames) in enumerate(entries):
            if i >= max_loops:
                break

            try:
                with h5py.File(f"{rodent_rollout_dir}/clip_{int(clip_id)}.h5", "r") as f:
                    roll = load_from_h5py(f)
                qpos = np.asarray(roll["qpos"])
            except:
                continue

            frames = np.asarray(frames, dtype=int)
            frames = frames[frames < len(qpos)]

            if len(frames) < 2:
                continue

            points.append(qpos[frames].mean(axis=0))

        if points:
            behavior_clusters[matched_behavior][label] = np.array(points)

    return behavior_clusters


def collect_labeled_qpos_for_behavior(
    clip_ids: list[int],
    rodent_rollout_dir: str,
    body_indices: list[int],
    max_frames: int = 5000,
) -> np.ndarray:
    """
    Collect qpos point cloud from labeled clips for a behavior category.

    Args:
        clip_ids: List of clip IDs for this behavior
        rodent_rollout_dir: Path to rollout data
        body_indices: Indices of body parts to include
        max_frames: Maximum total frames to collect

    Returns:
        Point cloud array of shape (N, len(body_indices))
    """
    all_points = []
    total_frames = 0

    for clip_id in clip_ids:
        if total_frames >= max_frames:
            break

        try:
            qpos = load_qpos(clip_id, rodent_rollout_dir)
        except:
            continue

        selected_qpos = qpos[:, body_indices]
        all_points.append(selected_qpos)
        total_frames += len(qpos)

    if not all_points:
        return np.array([])

    combined = np.vstack(all_points)
    if len(combined) > max_frames:
        indices = np.random.choice(len(combined), max_frames, replace=False)
        combined = combined[indices]

    return combined


def collect_cluster_qpos_for_behavior(
    cluster_ids: list[int],
    cluster_dict: dict,
    rodent_rollout_dir: str,
    body_indices: list[int],
    max_frames: int = 5000,
    max_loops: int = 200,
) -> np.ndarray:
    """
    Collect qpos point cloud from clusters for a behavior category.

    Args:
        cluster_ids: List of cluster IDs matching this behavior
        cluster_dict: Cluster dictionary mapping cluster_id -> [(clip_id, frames), ...]
        rodent_rollout_dir: Path to rollout data
        body_indices: Indices of body parts to include
        max_frames: Maximum total frames to collect
        max_loops: Maximum entries per cluster

    Returns:
        Point cloud array of shape (N, len(body_indices))
    """
    all_points = []
    total_frames = 0

    for cid in cluster_ids:
        if cid not in cluster_dict:
            continue

        entries = cluster_dict[cid]

        for i, (clip_id, frames) in enumerate(entries):
            if i >= max_loops or total_frames >= max_frames:
                break

            try:
                qpos = load_qpos(clip_id, rodent_rollout_dir)
            except:
                continue

            frames = np.asarray(frames, dtype=int)
            frames = frames[frames < len(qpos)]

            if len(frames) < 2:
                continue

            selected_qpos = qpos[frames][:, body_indices]
            all_points.append(selected_qpos)
            total_frames += len(frames)

    if not all_points:
        return np.array([])

    combined = np.vstack(all_points)
    if len(combined) > max_frames:
        indices = np.random.choice(len(combined), max_frames, replace=False)
        combined = combined[indices]

    return combined


def compute_persistence_diagrams(
    point_cloud: np.ndarray,
    max_dim: int = 1,
    max_points: int = 2000,
) -> dict:
    """
    Compute persistence diagrams using Ripser.

    Args:
        point_cloud: Point cloud array of shape (N, D)
        max_dim: Maximum homology dimension to compute
        max_points: Maximum points for computation (subsampled if exceeded)

    Returns:
        Dictionary containing diagrams and Betti statistics
    """
    if len(point_cloud) < 3:
        return {
            "diagrams": [np.array([]) for _ in range(max_dim + 1)],
            "betti": np.zeros(max_dim + 1, dtype=int),
            "total_persistence": np.zeros(max_dim + 1),
        }

    if len(point_cloud) > max_points:
        indices = np.random.choice(len(point_cloud), max_points, replace=False)
        point_cloud = point_cloud[indices]

    try:
        result = ripser.ripser(
            point_cloud,
            maxdim=max_dim,
            thresh=np.inf,
            coeff=2,
            do_cocycles=False,
        )
        diagrams = result["dgms"]

        betti = np.zeros(max_dim + 1, dtype=int)
        total_persistence = np.zeros(max_dim + 1)

        for dim, dgm in enumerate(diagrams):
            if len(dgm) == 0:
                continue

            finite_mask = np.isfinite(dgm[:, 1])
            finite_dgm = dgm[finite_mask]

            if len(finite_dgm) > 0:
                persistence = finite_dgm[:, 1] - finite_dgm[:, 0]
                threshold = np.percentile(persistence, 75) if len(persistence) > 4 else 0
                betti[dim] = np.sum(persistence > threshold)
                total_persistence[dim] = np.sum(persistence)

        return {
            "diagrams": diagrams,
            "betti": betti,
            "total_persistence": total_persistence,
        }

    except Exception as e:
        print(f"  Warning: PH computation failed: {e}")
        return {
            "diagrams": [np.array([]) for _ in range(max_dim + 1)],
            "betti": np.zeros(max_dim + 1, dtype=int),
            "total_persistence": np.zeros(max_dim + 1),
        }


def compute_betti_similarity(betti_labeled: np.ndarray, betti_cluster: np.ndarray) -> float:
    """
    Compute similarity between Betti vectors using cosine similarity.

    Args:
        betti_labeled: Betti vector for labeled data
        betti_cluster: Betti vector for cluster data

    Returns:
        Cosine similarity score (0 to 1, higher is more similar)
    """
    norm_l = np.linalg.norm(betti_labeled)
    norm_c = np.linalg.norm(betti_cluster)

    if norm_l < 1e-10 or norm_c < 1e-10:
        return 0.0

    return float(np.dot(betti_labeled, betti_cluster) / (norm_l * norm_c))


def compute_wasserstein_distance(
    dgm_labeled: np.ndarray,
    dgm_cluster: np.ndarray,
) -> float:
    """
    Compute Wasserstein distance between persistence diagrams.

    Args:
        dgm_labeled: Persistence diagram for labeled data
        dgm_cluster: Persistence diagram for cluster data

    Returns:
        Wasserstein distance (lower is more similar)
    """
    if len(dgm_labeled) == 0 and len(dgm_cluster) == 0:
        return 0.0

    if len(dgm_labeled) == 0:
        dgm_labeled = np.array([[0, 0]])
    if len(dgm_cluster) == 0:
        dgm_cluster = np.array([[0, 0]])

    try:
        return float(wasserstein(dgm_labeled, dgm_cluster))
    except Exception as e:
        print(f"  Warning: Wasserstein computation failed: {e}")
        return np.nan


@dataclass
class TopologicalMetrics:
    wasserstein_h0: dict[str, float]
    wasserstein_h1: dict[str, float]
    betti_similarity: dict[str, float]
    labeled_betti: dict[str, np.ndarray]
    cluster_betti: dict[str, np.ndarray]
    mean_wasserstein: float
    mean_betti_similarity: float


def compute_topological_metrics(
    method_cfg: "MethodConfig",
    labeled_clips: dict[str, list[int]],
    cfg: "BenchmarkAnalysisConfig",
    body_indices: list[int],
    max_frames: int = 3000,
    max_ph_points: int = 1500,
) -> TopologicalMetrics | None:
    """
    Compute topological similarity metrics between labeled data and clustering method.

    Args:
        method_cfg: Method configuration
        labeled_clips: Dict mapping behavior -> clip_ids
        cfg: Benchmark analysis configuration
        body_indices: Indices of body parts for qpos
        max_frames: Maximum frames to collect per behavior
        max_ph_points: Maximum points for PH computation

    Returns:
        TopologicalMetrics dataclass or None if cluster_dict not found
    """
    cluster_dict = load_cluster_dict(method_cfg.results_dir, method_cfg.name)

    if cluster_dict is None:
        return None

    if not method_cfg.cluster_labels:
        method_cfg.cluster_labels = {i: f"cluster_{i}" for i in cluster_dict.keys()}

    cluster_groups = group_clusters_by_behavior(
        method_cfg.cluster_labels,
        cfg.behavior_categories,
    )

    wasserstein_h0 = {}
    wasserstein_h1 = {}
    betti_similarity = {}
    labeled_betti = {}
    cluster_betti = {}

    for behavior in cfg.behavior_categories:
        clip_ids = labeled_clips.get(behavior, [])
        cluster_ids = cluster_groups.get(behavior, [])

        labeled_points = collect_labeled_qpos_for_behavior(
            clip_ids, cfg.rodent_rollout_dir, body_indices, max_frames
        )
        cluster_points = collect_cluster_qpos_for_behavior(
            cluster_ids, cluster_dict, cfg.rodent_rollout_dir,
            body_indices, max_frames, cfg.max_loops
        )

        labeled_ph = compute_persistence_diagrams(labeled_points, max_dim=1, max_points=max_ph_points)
        cluster_ph = compute_persistence_diagrams(cluster_points, max_dim=1, max_points=max_ph_points)

        wasserstein_h0[behavior] = compute_wasserstein_distance(
            labeled_ph["diagrams"][0], cluster_ph["diagrams"][0]
        )
        wasserstein_h1[behavior] = compute_wasserstein_distance(
            labeled_ph["diagrams"][1] if len(labeled_ph["diagrams"]) > 1 else np.array([]),
            cluster_ph["diagrams"][1] if len(cluster_ph["diagrams"]) > 1 else np.array([]),
        )

        labeled_betti[behavior] = labeled_ph["betti"]
        cluster_betti[behavior] = cluster_ph["betti"]
        betti_similarity[behavior] = compute_betti_similarity(
            labeled_ph["betti"], cluster_ph["betti"]
        )

    valid_w0 = [v for v in wasserstein_h0.values() if not np.isnan(v)]
    valid_w1 = [v for v in wasserstein_h1.values() if not np.isnan(v)]
    valid_betti = [v for v in betti_similarity.values() if not np.isnan(v)]

    mean_wasserstein = np.mean(valid_w0 + valid_w1) if (valid_w0 or valid_w1) else np.nan
    mean_betti_sim = np.mean(valid_betti) if valid_betti else np.nan

    return TopologicalMetrics(
        wasserstein_h0=wasserstein_h0,
        wasserstein_h1=wasserstein_h1,
        betti_similarity=betti_similarity,
        labeled_betti=labeled_betti,
        cluster_betti=cluster_betti,
        mean_wasserstein=mean_wasserstein,
        mean_betti_similarity=mean_betti_sim,
    )


def plot_topological_comparison(
    method_names: list[str],
    topo_metrics: list[TopologicalMetrics | None],
    behavior_categories: list[str],
    output_dir: str,
    figsize: tuple[int, int] = (14, 10),
) -> None:
    """
    Plot topological comparison charts.

    Args:
        method_names: List of method names
        topo_metrics: List of TopologicalMetrics for each method
        behavior_categories: List of behavior categories
        output_dir: Output directory
        figsize: Figure size
    """
    os.makedirs(output_dir, exist_ok=True)

    valid_indices = [i for i, m in enumerate(topo_metrics) if m is not None]
    if not valid_indices:
        print("  Warning: No valid topological metrics to plot")
        return

    valid_names = [method_names[i] for i in valid_indices]
    valid_metrics = [topo_metrics[i] for i in valid_indices]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Mean Wasserstein Distance (lower is better)
    ax1 = axes[0, 0]
    mean_w = [m.mean_wasserstein for m in valid_metrics]
    sorted_pairs = sorted(zip(valid_names, mean_w), key=lambda x: x[1] if not np.isnan(x[1]) else np.inf)
    sorted_names_w = [p[0] for p in sorted_pairs]
    sorted_w = [p[1] for p in sorted_pairs]

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_names_w)))
    bars = ax1.barh(range(len(sorted_names_w)), sorted_w, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(sorted_names_w)))
    ax1.set_yticklabels(sorted_names_w, fontsize=10)
    ax1.set_xlabel("Mean Wasserstein Distance", fontsize=11)
    ax1.set_title("PH Distance to Labeled Data\n(Lower = More Similar Topology)", fontsize=12, fontweight='bold')
    ax1.grid(True, axis='x', alpha=0.3)
    for bar, val in zip(bars, sorted_w):
        if not np.isnan(val):
            ax1.text(bar.get_width() + 0.01 * max(sorted_w), bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)

    # 2. Mean Betti Similarity (higher is better)
    ax2 = axes[0, 1]
    mean_b = [m.mean_betti_similarity for m in valid_metrics]
    sorted_pairs = sorted(zip(valid_names, mean_b), key=lambda x: -x[1] if not np.isnan(x[1]) else np.inf)
    sorted_names_b = [p[0] for p in sorted_pairs]
    sorted_b = [p[1] for p in sorted_pairs]

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_names_b)))
    bars = ax2.barh(range(len(sorted_names_b)), sorted_b, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(sorted_names_b)))
    ax2.set_yticklabels(sorted_names_b, fontsize=10)
    ax2.set_xlabel("Betti Vector Cosine Similarity", fontsize=11)
    ax2.set_title("Betti Number Similarity\n(Higher = More Similar Complexity)", fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1.1)
    ax2.grid(True, axis='x', alpha=0.3)
    for bar, val in zip(bars, sorted_b):
        if not np.isnan(val):
            ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)

    # 3. Wasserstein H1 per behavior (heatmap)
    ax3 = axes[1, 0]
    w1_matrix = np.zeros((len(valid_names), len(behavior_categories)))
    for i, m in enumerate(valid_metrics):
        for j, beh in enumerate(behavior_categories):
            w1_matrix[i, j] = m.wasserstein_h1.get(beh, np.nan)

    im3 = ax3.imshow(w1_matrix, aspect='auto', cmap='RdYlGn_r')
    ax3.set_xticks(range(len(behavior_categories)))
    ax3.set_xticklabels([b.capitalize() for b in behavior_categories], fontsize=10)
    ax3.set_yticks(range(len(valid_names)))
    ax3.set_yticklabels(valid_names, fontsize=10)
    ax3.set_xlabel("Behavior", fontsize=11)
    ax3.set_ylabel("Method", fontsize=11)
    ax3.set_title("Wasserstein H1 Distance per Behavior\n(Lower = Better)", fontsize=12, fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label("W1 Distance", fontsize=10)

    for i in range(len(valid_names)):
        for j in range(len(behavior_categories)):
            val = w1_matrix[i, j]
            if not np.isnan(val):
                ax3.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if val > np.nanmedian(w1_matrix) else 'black')

    # 4. Betti similarity per behavior (heatmap)
    ax4 = axes[1, 1]
    betti_matrix = np.zeros((len(valid_names), len(behavior_categories)))
    for i, m in enumerate(valid_metrics):
        for j, beh in enumerate(behavior_categories):
            betti_matrix[i, j] = m.betti_similarity.get(beh, np.nan)

    im4 = ax4.imshow(betti_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax4.set_xticks(range(len(behavior_categories)))
    ax4.set_xticklabels([b.capitalize() for b in behavior_categories], fontsize=10)
    ax4.set_yticks(range(len(valid_names)))
    ax4.set_yticklabels(valid_names, fontsize=10)
    ax4.set_xlabel("Behavior", fontsize=11)
    ax4.set_ylabel("Method", fontsize=11)
    ax4.set_title("Betti Similarity per Behavior\n(Higher = Better)", fontsize=12, fontweight='bold')
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
    cbar4.set_label("Cosine Similarity", fontsize=10)

    for i in range(len(valid_names)):
        for j in range(len(behavior_categories)):
            val = betti_matrix[i, j]
            if not np.isnan(val):
                ax4.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if val < 0.5 else 'black')

    plt.tight_layout()

    save_path = os.path.join(output_dir, "topological_comparison.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved topological comparison to {save_path}")


CLUSTER_COLORS = [
    "#E63946",  # Red
    "#2A9D8F",  # Teal
    "#E9C46A",  # Yellow
    "#264653",  # Dark blue
    "#F4A261",  # Orange
    "#8338EC",  # Purple
    "#06D6A0",  # Mint
    "#118AB2",  # Blue
    "#EF476F",  # Pink
    "#073B4C",  # Navy
]


def plot_method_behavior_overlay(
    method_name: str,
    method_cluster_dict: dict,
    method_cluster_labels: dict[int, str],
    labeled_clips: dict[str, list[int]],
    behavior_categories: list[str],
    rodent_rollout_dir: str,
    output_dir: str,
    pca_model: PCA = None,
    max_loops: int = 200,
    figsize: tuple[int, int] = (15, 5),
) -> PCA:
    """
    Plot three-panel visualization (one per behavior) with labeled data as grey outline
    and cluster overlays in distinct colors.

    Args:
        method_name: Name of the clustering method
        method_cluster_dict: Cluster dictionary for this method
        method_cluster_labels: Labels for each cluster
        labeled_clips: Dict of behavior -> clip_ids (ground truth)
        behavior_categories: List of behavior categories to plot
        rodent_rollout_dir: Path to rollout data
        output_dir: Output directory
        pca_model: Pre-fitted PCA model (for consistent projection across methods)
        max_loops: Max samples per cluster
        figsize: Figure size

    Returns:
        Fitted PCA model (for reuse)
    """
    os.makedirs(output_dir, exist_ok=True)

    labeled_points = collect_labeled_behavior_points(labeled_clips, rodent_rollout_dir)
    cluster_by_behavior = collect_cluster_points_by_behavior(
        method_cluster_dict, method_cluster_labels, rodent_rollout_dir,
        behavior_categories, max_loops
    )

    if not labeled_points:
        print(f"  Warning: No labeled points, skipping {method_name}")
        return pca_model

    all_points = []
    for pts in labeled_points.values():
        all_points.append(pts)
    for behavior_dict in cluster_by_behavior.values():
        for pts in behavior_dict.values():
            all_points.append(pts)

    if not all_points:
        return pca_model

    all_points = np.vstack(all_points)

    if pca_model is None:
        pca_model = PCA(n_components=2, random_state=0)
        pca_model.fit(all_points)

    labeled_points_2d = {beh: pca_model.transform(pts) for beh, pts in labeled_points.items()}
    cluster_points_2d = {}
    for behavior, cluster_dict in cluster_by_behavior.items():
        cluster_points_2d[behavior] = {
            label: pca_model.transform(pts) for label, pts in cluster_dict.items()
        }

    all_2d_points = []
    for pts in labeled_points_2d.values():
        all_2d_points.append(pts)
    for behavior_dict in cluster_points_2d.values():
        for pts in behavior_dict.values():
            all_2d_points.append(pts)
    all_2d = np.vstack(all_2d_points) if all_2d_points else None

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax_idx, behavior in enumerate(behavior_categories[:3]):
        ax = axes[ax_idx]

        if behavior in labeled_points_2d:
            pts = labeled_points_2d[behavior]

            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    hull_points = pts[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])
                    ax.fill(hull_points[:, 0], hull_points[:, 1],
                            color='#f0f0f0', alpha=0.6, edgecolor='#666666',
                            linewidth=2.5, linestyle='-', zorder=1)
                except:
                    pass

            ax.scatter(pts[:, 0], pts[:, 1], c='#999999', alpha=0.4, s=20,
                       edgecolors='none', zorder=1)

        cluster_handles = []
        if behavior in cluster_points_2d:
            clusters = cluster_points_2d[behavior]

            for i, (label, pts) in enumerate(sorted(clusters.items())):
                color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

                if len(pts) >= 3:
                    try:
                        hull = ConvexHull(pts)
                        hull_points = pts[hull.vertices]
                        hull_points = np.vstack([hull_points, hull_points[0]])
                        ax.fill(hull_points[:, 0], hull_points[:, 1],
                                color=color, alpha=0.35, edgecolor=color,
                                linewidth=2, zorder=2)
                    except:
                        pass

                ax.scatter(pts[:, 0], pts[:, 1], c=color, alpha=0.7, s=25,
                           edgecolors='white', linewidths=0.3, zorder=3)

                cluster_handles.append(mpatches.Patch(
                    facecolor=color, alpha=0.6, label=label,
                    edgecolor=color, linewidth=1.5
                ))

        labeled_handle = mpatches.Patch(
            facecolor='#f0f0f0', alpha=0.6, label=f'{behavior} (labeled)',
            edgecolor='#666666', linewidth=1.5
        )

        if all_2d is not None:
            margin = 0.1
            x_range = all_2d[:, 0].max() - all_2d[:, 0].min()
            y_range = all_2d[:, 1].max() - all_2d[:, 1].min()
            ax.set_xlim(all_2d[:, 0].min() - margin * x_range, all_2d[:, 0].max() + margin * x_range)
            ax.set_ylim(all_2d[:, 1].min() - margin * y_range, all_2d[:, 1].max() + margin * y_range)

        all_handles = [labeled_handle] + cluster_handles
        ax.legend(
            handles=all_handles,
            loc='upper right',
            fontsize=8,
            framealpha=0.95,
        )

        ax.set_xlabel("PC1", fontsize=11)
        ax.set_ylabel("PC2", fontsize=11)
        ax.set_title(f"{behavior.capitalize()}", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')

    fig.suptitle(f"{method_name}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    safe_name = method_name.replace(" ", "_").replace("+", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(output_dir, f"{safe_name}_behavior_overlay.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved behavior overlay to {save_path}")

    return pca_model


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
    heatmap_output_dir = os.path.join(cfg.output_dir, "heatmaps")
    os.makedirs(heatmap_output_dir, exist_ok=True)

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
                heatmap_output_dir,
            )

    print("\n=== Stage 4: Generating comparison plots ===")
    plot_comparison_bar_chart(method_names, mean_diffs, cfg.output_dir)

    print("\n=== Stage 4b: Generating behavior overlay plots ===")
    hull_output_dir = os.path.join(cfg.output_dir, "pca_hulls")
    os.makedirs(hull_output_dir, exist_ok=True)

    pca_model = None
    for method_cfg in cfg.methods:
        print(f"\n  Generating overlay for: {method_cfg.name}")
        method_cluster_dict = load_cluster_dict(method_cfg.results_dir, method_cfg.name)

        if method_cluster_dict is not None:
            if not method_cfg.cluster_labels:
                method_cfg.cluster_labels = {i: f"cluster_{i}" for i in method_cluster_dict.keys()}

            pca_model = plot_method_behavior_overlay(
                method_name=method_cfg.name,
                method_cluster_dict=method_cluster_dict,
                method_cluster_labels=method_cfg.cluster_labels,
                labeled_clips=labeled_clips,
                behavior_categories=cfg.behavior_categories,
                rodent_rollout_dir=cfg.rodent_rollout_dir,
                output_dir=hull_output_dir,
                pca_model=pca_model,
                max_loops=cfg.max_loops,
            )

    print("\n=== Stage 4c: Computing topological metrics ===")
    topo_metrics = []
    for method_cfg in tqdm(cfg.methods, desc="Topological analysis"):
        print(f"\n  Computing PH for: {method_cfg.name}")
        metrics = compute_topological_metrics(
            method_cfg,
            labeled_clips,
            cfg,
            body_indices,
            max_frames=3000,
            max_ph_points=1500,
        )
        topo_metrics.append(metrics)

        if metrics is not None:
            print(f"    Mean Wasserstein: {metrics.mean_wasserstein:.4f}")
            print(f"    Mean Betti Similarity: {metrics.mean_betti_similarity:.4f}")
            for beh in cfg.behavior_categories:
                print(f"      {beh}: W_H1={metrics.wasserstein_h1[beh]:.3f}, "
                      f"Betti_sim={metrics.betti_similarity[beh]:.3f}")

            all_results[method_cfg.name]["topological_metrics"] = {
                "wasserstein_h0": metrics.wasserstein_h0,
                "wasserstein_h1": metrics.wasserstein_h1,
                "betti_similarity": metrics.betti_similarity,
                "labeled_betti": {k: v.tolist() for k, v in metrics.labeled_betti.items()},
                "cluster_betti": {k: v.tolist() for k, v in metrics.cluster_betti.items()},
                "mean_wasserstein": metrics.mean_wasserstein,
                "mean_betti_similarity": metrics.mean_betti_similarity,
            }

    print("\n=== Stage 4d: Generating topological comparison plots ===")
    plot_topological_comparison(
        method_names,
        topo_metrics,
        cfg.behavior_categories,
        cfg.output_dir,
    )

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

        f.write("Variance Diff Ranking (lower = closer to labeled variance):\n")
        f.write("-" * 40 + "\n")

        valid_pairs = [(n, d) for n, d in zip(method_names, mean_diffs) if not np.isnan(d)]
        sorted_pairs = sorted(valid_pairs, key=lambda x: x[1])

        for rank, (name, diff) in enumerate(sorted_pairs, 1):
            f.write(f"{rank}. {name}: {diff:.4f}\n")

        f.write("\n\nTopological Similarity Rankings:\n")
        f.write("-" * 40 + "\n")

        f.write("\nWasserstein Distance (lower = more similar topology):\n")
        valid_topo = [(method_names[i], topo_metrics[i]) for i in range(len(topo_metrics))
                      if topo_metrics[i] is not None and not np.isnan(topo_metrics[i].mean_wasserstein)]
        sorted_topo_w = sorted(valid_topo, key=lambda x: x[1].mean_wasserstein)
        for rank, (name, m) in enumerate(sorted_topo_w, 1):
            f.write(f"{rank}. {name}: {m.mean_wasserstein:.4f}\n")

        f.write("\nBetti Similarity (higher = more similar complexity):\n")
        sorted_topo_b = sorted(valid_topo, key=lambda x: -x[1].mean_betti_similarity)
        for rank, (name, m) in enumerate(sorted_topo_b, 1):
            f.write(f"{rank}. {name}: {m.mean_betti_similarity:.4f}\n")

    print(f"Saved summary to {summary_path}")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

    print("\nVariance Diff Rankings (lower = better):")
    valid_pairs = [(n, d) for n, d in zip(method_names, mean_diffs) if not np.isnan(d)]
    sorted_pairs = sorted(valid_pairs, key=lambda x: x[1])
    for rank, (name, diff) in enumerate(sorted_pairs, 1):
        print(f"  {rank}. {name}: {diff:.4f}")

    print("\nTopological Rankings:")
    print("  Wasserstein Distance (lower = more similar topology):")
    valid_topo = [(method_names[i], topo_metrics[i]) for i in range(len(topo_metrics))
                  if topo_metrics[i] is not None and not np.isnan(topo_metrics[i].mean_wasserstein)]
    sorted_topo_w = sorted(valid_topo, key=lambda x: x[1].mean_wasserstein)
    for rank, (name, m) in enumerate(sorted_topo_w, 1):
        print(f"    {rank}. {name}: {m.mean_wasserstein:.4f}")

    print("  Betti Similarity (higher = more similar complexity):")
    sorted_topo_b = sorted(valid_topo, key=lambda x: -x[1].mean_betti_similarity)
    for rank, (name, m) in enumerate(sorted_topo_b, 1):
        print(f"    {rank}. {name}: {m.mean_betti_similarity:.4f}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/benchmark_analysis.yaml"
    main(config_path)
