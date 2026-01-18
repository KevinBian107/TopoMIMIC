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
import h5py
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from tqdm import tqdm

from vnl_tda.utils.load_utils import load_rollout_from_source, get_aggregate_data
from vnl_tda.utils.rendering import render_from_saved_rollout
from vnl_tda.pipelines.tda.skeletonization.plotting import (
    plot_cluster_subspace_mds_separate,
    _compare_clusters_axis_motion_with_union,
)
from vnl_tda.pipelines.tda.skeletonization.rendering import render_grid_clusters_side_by_side


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

    methods: list[str] = None
    seeds: list[int] = None

    run_variance_heatmap: bool = True
    run_mds_analysis: bool = True
    run_cluster_to_label_comparison: bool = True
    run_seed_comparison: bool = True
    run_transition_comparison: bool = True
    run_javd_analysis: bool = True
    render_gt_transitions: bool = False

    top_k_variance_joints: int = 5
    gt_render_max_samples: int = 5
    gt_render_framerate: int = 30
    gt_render_target_size: tuple[int, int] = (320, 240)
    mds_k_cycles: int = 50
    mds_s_top: int = 10
    mds_behaviors: list[str] = None
    mds_figsize: tuple[float, float] = (10, 9)

    body_parts: list[str] = None
    max_loops: int = 200

    transition_window: int = 5
    max_transition_points: int = 500

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
        methods=cfg.get("methods", None),
        seeds=cfg.get("seeds", None),
        run_variance_heatmap=cfg.get("run_variance_heatmap", True),
        run_mds_analysis=cfg.get("run_mds_analysis", True),
        run_cluster_to_label_comparison=cfg.get("run_cluster_to_label_comparison", True),
        run_seed_comparison=cfg.get("run_seed_comparison", True),
        run_transition_comparison=cfg.get("run_transition_comparison", True),
        run_javd_analysis=cfg.get("run_javd_analysis", True),
        render_gt_transitions=cfg.get("render_gt_transitions", False),
        top_k_variance_joints=cfg.get("top_k_variance_joints", 5),
        gt_render_max_samples=cfg.get("gt_render_max_samples", 5),
        gt_render_framerate=cfg.get("gt_render_framerate", 30),
        gt_render_target_size=tuple(cfg.get("gt_render_target_size", [320, 240])),
        mds_k_cycles=cfg.get("mds_k_cycles", 50),
        mds_s_top=cfg.get("mds_s_top", 10),
        mds_behaviors=cfg.get("mds_behaviors", None),
        mds_figsize=tuple(cfg.get("mds_figsize", [14, 12])),
        body_parts=cfg.get("body_parts", None),
        max_loops=cfg.get("max_loops", 200),
        transition_window=cfg.get("transition_window", 5),
        max_transition_points=cfg.get("max_transition_points", 500),
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


def load_cluster_dict(method_dir: str, seed: int = 0) -> dict:
    """
    Load cluster_dict for a specific seed.

    Args:
        method_dir: Path to method output directory
        seed: Random seed to load

    Returns:
        Cluster dict mapping cluster_id to list of (clip_id, frames)
    """
    results = load_method_results(method_dir)
    if seed in results:
        result = results[seed]
        return result.get("cluster_dict", {})
    return {}


def load_labels_yaml(method_dir: str, seed: int = 0) -> dict[int, str]:
    labels_path = os.path.join(method_dir, f"labels_seed_{seed}.yaml")
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


def load_intention(clip_id: int, rodent_rollout_dir: str) -> np.ndarray:
    if rodent_rollout_dir.endswith('.h5'):
        with h5py.File(rodent_rollout_dir, 'r') as f:
            return np.array(f['intention'][clip_id])
    else:
        path = os.path.join(rodent_rollout_dir, f"clip_{clip_id}.h5")
        with h5py.File(path, 'r') as f:
            return np.array(f['intention'])


BEHAVIOR_NAME_MAP = {
    "walking": "Walking",
    "walk": "Walking",
    "immobility": "Immobility",
    "immobile": "Immobility",
    "orienting": "Orienting",
    "orient": "Orienting",
    "grooming": "Grooming",
    "groom": "Grooming",
    "supported_rear": "Supported Rear",
    "supportedrear": "Supported Rear",
    "srear": "Supported Rear",
    "unsupported_rear": "Unsupported Rear",
    "unsupportedrear": "Unsupported Rear",
    "urear": "Unsupported Rear",
    "left_turn": "Left Turn",
    "leftturn": "Left Turn",
    "lturn": "Left Turn",
    "right_turn": "Right Turn",
    "rightturn": "Right Turn",
    "rturn": "Right Turn",
}

PRIMITIVE_BEHAVIORS = {
    "Walking", "Immobility", "Orienting", "Grooming",
    "Supported Rear", "Unsupported Rear",
}

COMPOSITIONAL_BEHAVIORS = {
    "Left Turn", "Right Turn",
}

VALID_BEHAVIORS = PRIMITIVE_BEHAVIORS | COMPOSITIONAL_BEHAVIORS


def normalize_behavior_name(name: str) -> str:
    if name in VALID_BEHAVIORS:
        return name
    name_lower = name.lower().strip().replace(" ", "_")
    return BEHAVIOR_NAME_MAP.get(name_lower, name)


@dataclass
class ParsedLabel:
    original: str
    primitives: list[str]
    compositional: list[str]
    comparison_type: str  # "transition", "intersection", or "single"

    @property
    def gt_key(self) -> tuple:
        if self.comparison_type == "transition":
            return ("transition", tuple(sorted(self.primitives[:2])))
        elif self.comparison_type == "intersection":
            return ("intersection", tuple(sorted(self.primitives + self.compositional)))
        else:
            return ("single", tuple(self.primitives + self.compositional))


def parse_cluster_label(label: str) -> ParsedLabel | None:
    if not label or not label.strip():
        return None

    parts = label.strip().split()
    primitives = []
    compositional = []

    for part in parts:
        normalized = normalize_behavior_name(part)
        if normalized in PRIMITIVE_BEHAVIORS:
            primitives.append(normalized)
        elif normalized in COMPOSITIONAL_BEHAVIORS:
            compositional.append(normalized)

    if not primitives and not compositional:
        return None

    if len(primitives) >= 2:
        comparison_type = "transition"
    elif len(primitives) == 1 and len(compositional) >= 1:
        comparison_type = "intersection"
    elif len(primitives) == 1:
        comparison_type = "single"
    elif len(compositional) >= 1:
        comparison_type = "single"
    else:
        return None

    return ParsedLabel(
        original=label,
        primitives=primitives,
        compositional=compositional,
        comparison_type=comparison_type,
    )


def load_behavior_segments_multilabel(csv_path: str) -> dict[int, dict[int, set[str]]]:
    df = pd.read_csv(csv_path)
    clip_behaviors = defaultdict(lambda: defaultdict(set))

    for _, row in df.iterrows():
        start_frame = int(row["Start_Frame"])
        end_frame = int(row["End_Frame"])
        if start_frame == end_frame == 0:
            continue

        video = str(row["Video"])
        clip_id = int(video.replace("clip_", "").replace(".mp4", ""))
        behavior = str(row["Behavior"])

        for frame in range(start_frame, min(end_frame + 1, 500)):
            clip_behaviors[clip_id][frame].add(behavior)

    return {k: dict(v) for k, v in clip_behaviors.items()}


def find_gt_transition_frames_from_segments(
    csv_path: str,
    primitives: list[str],
    max_gap: int = 10,
    balance_segments: bool = True,
) -> list[tuple[int, int]]:
    """
    Find GT transition frames from consecutive behavior segments.

    A transition is identified when two segments of the target behaviors
    occur consecutively (with minimal gap). Returns frames from BOTH
    segments, with optional balancing to match segment lengths.

    Args:
        csv_path: Path to behavior_labels.csv
        primitives: List of two primitive behaviors (e.g., ["Walking", "Immobility"])
        max_gap: Maximum allowed gap between consecutive segments to count as transition
        balance_segments: If True, truncate longer segment to match shorter one's length

    Returns:
        List of (clip_id, frame) tuples for frames in both segments
    """
    if len(primitives) < 2:
        return []

    b1 = normalize_behavior_name(primitives[0])
    b2 = normalize_behavior_name(primitives[1])
    target_pair = tuple(sorted([b1, b2]))

    df = pd.read_csv(csv_path)

    segments_by_clip = defaultdict(list)
    for _, row in df.iterrows():
        start_frame = int(row["Start_Frame"])
        end_frame = int(row["End_Frame"])
        if start_frame == end_frame == 0:
            continue

        video = str(row["Video"])
        clip_id = int(video.replace("clip_", "").replace(".mp4", ""))
        behavior = normalize_behavior_name(str(row["Behavior"]))

        if behavior in PRIMITIVE_BEHAVIORS:
            segments_by_clip[clip_id].append({
                "behavior": behavior,
                "start": start_frame,
                "end": end_frame,
            })

    transitions = []

    for clip_id, segments in segments_by_clip.items():
        segments = sorted(segments, key=lambda x: x["start"])

        for i in range(len(segments) - 1):
            seg_a = segments[i]
            seg_b = segments[i + 1]

            pair = tuple(sorted([seg_a["behavior"], seg_b["behavior"]]))
            if pair != target_pair:
                continue

            gap = seg_b["start"] - seg_a["end"]
            if gap > max_gap:
                continue

            len_a = seg_a["end"] - seg_a["start"] + 1
            len_b = seg_b["end"] - seg_b["start"] + 1

            if balance_segments:
                min_len = min(len_a, len_b)
                a_start = seg_a["end"] - min_len + 1
                a_end = seg_a["end"]
                b_start = seg_b["start"]
                b_end = seg_b["start"] + min_len - 1
            else:
                a_start = seg_a["start"]
                a_end = seg_a["end"]
                b_start = seg_b["start"]
                b_end = seg_b["end"]

            for f in range(a_start, a_end + 1):
                transitions.append((clip_id, f))
            for f in range(b_start, b_end + 1):
                transitions.append((clip_id, f))

    return list(set(transitions))


def find_gt_transition_frames(
    clip_behaviors: dict[int, dict[int, set[str]]],
    primitives: list[str],
    window: int = 5,
) -> list[tuple[int, int]]:
    """Legacy function - kept for compatibility but may have issues with sparse labels."""
    if len(primitives) < 2:
        return []

    b1, b2 = primitives[0], primitives[1]
    target_pair = tuple(sorted([b1, b2]))
    transitions = []

    for clip_id, frame_map in clip_behaviors.items():
        frames = sorted(frame_map.keys())

        for i in range(1, len(frames)):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]
            prev_behaviors = frame_map[prev_frame]
            curr_behaviors = frame_map[curr_frame]

            prev_prims = prev_behaviors & PRIMITIVE_BEHAVIORS
            curr_prims = curr_behaviors & PRIMITIVE_BEHAVIORS

            if prev_prims != curr_prims:
                for pb in prev_prims:
                    for cb in curr_prims:
                        if pb != cb:
                            pair = tuple(sorted([pb, cb]))
                            if pair == target_pair:
                                for offset in range(-window, window + 1):
                                    f = prev_frame + offset
                                    if f in frame_map:
                                        transitions.append((clip_id, f))

    return list(set(transitions))


def find_gt_intersection_frames(
    clip_behaviors: dict[int, dict[int, set[str]]],
    behaviors: list[str],
) -> list[tuple[int, int]]:
    if not behaviors:
        return []

    target_set = set(behaviors)
    intersections = []

    for clip_id, frame_map in clip_behaviors.items():
        for frame, frame_behaviors in frame_map.items():
            if target_set <= frame_behaviors:
                intersections.append((clip_id, frame))

    return intersections


def compute_density_2d(
    points: np.ndarray,
    grid_size: int = 50,
    sigma: float = 1.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    pad = 0.15

    x_edges = np.linspace(
        x_min - pad * (x_max - x_min),
        x_max + pad * (x_max - x_min),
        grid_size,
    )
    y_edges = np.linspace(
        y_min - pad * (y_max - y_min),
        y_max + pad * (y_max - y_min),
        grid_size,
    )

    H, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[x_edges, y_edges])
    H = gaussian_filter(H.T, sigma=sigma)

    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2
    return xc, yc, H


def random_subsample(points: np.ndarray, max_n: int) -> np.ndarray:
    if len(points) <= max_n:
        return points
    indices = np.random.choice(len(points), max_n, replace=False)
    return points[indices]


BEHAVIOR_GROUPS = {
    "Immobility": ["Immobility"],
    "Walking": ["Walking"],
    "Rear": ["Supported Rear", "Unsupported Rear"],
    "Orienting": ["Orienting"],
    "Grooming": ["Grooming"],
}


def get_behavior_group(behaviors: list[str]) -> str | None:
    """Determine which behavior group a transition/intersection belongs to."""
    for group_name, group_behaviors in BEHAVIOR_GROUPS.items():
        for b in behaviors:
            if b in group_behaviors:
                return group_name
    return None


def _plot_transition_groups(
    transition_data: list[dict],
    output_dir: str,
    feature_type: str,
    method_name: str,
    seed: int,
    max_points: int,
    figsize: tuple[float, float],
) -> None:
    """Helper to plot transition groups for a given feature type."""
    if not transition_data:
        return

    color_palette = [
        "#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12",
        "#1ABC9C", "#E91E63", "#00BCD4", "#FF5722", "#795548",
    ]

    all_frames = []
    for td in transition_data:
        all_frames.append(td["cluster_points"])
        all_frames.append(td["gt_points"])
    all_points = np.vstack(all_frames)

    pca = PCA(n_components=2)
    pca.fit(all_points)
    var_explained = pca.explained_variance_ratio_ * 100

    for td in transition_data:
        td["cluster_proj"] = pca.transform(td["cluster_points"])
        td["gt_proj"] = pca.transform(td["gt_points"])

    all_proj = np.vstack([td["cluster_proj"] for td in transition_data] +
                         [td["gt_proj"] for td in transition_data])
    x_min, x_max = all_proj[:, 0].min() - 2, all_proj[:, 0].max() + 2
    y_min, y_max = all_proj[:, 1].min() - 2, all_proj[:, 1].max() + 2

    groups_to_plot = defaultdict(list)
    for td in transition_data:
        group = td["behavior_group"] or "Other"
        groups_to_plot[group].append(td)

    def adjust_label_positions(positions, x_range, y_range):
        if len(positions) <= 1:
            return positions

        adjusted = []
        min_dist = max((x_range[1] - x_range[0]), (y_range[1] - y_range[0])) * 0.10

        for centroid, label, color in positions:
            new_pos = centroid.copy()
            for prev_pos, _, _ in adjusted:
                dist = np.linalg.norm(new_pos - prev_pos)
                if dist < min_dist and dist > 0:
                    direction = new_pos - prev_pos
                    direction = direction / np.linalg.norm(direction)
                    new_pos = prev_pos + direction * min_dist
            adjusted.append((new_pos, label, color))
        return adjusted

    for group_name, group_data in sorted(groups_to_plot.items()):
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 2, figure=fig, wspace=0.12)

        ax_cluster = fig.add_subplot(gs[0, 0])
        ax_gt = fig.add_subplot(gs[0, 1])

        label_positions_cluster = []
        label_positions_gt = []

        for i, td in enumerate(group_data):
            color = color_palette[i % len(color_palette)]

            cluster_proj = td["cluster_proj"]
            gt_proj = td["gt_proj"]
            display_label = td["display_label"]
            gt_label = td["gt_label"]

            pts_sub = random_subsample(cluster_proj, max_points)
            ax_cluster.scatter(
                pts_sub[:, 0], pts_sub[:, 1],
                c=color, s=30, alpha=0.5,
                edgecolors='white', linewidths=0.3,
            )

            if len(cluster_proj) > 15:
                xc, yc, H = compute_density_2d(cluster_proj)
                ax_cluster.contour(xc, yc, H, levels=4, colors=[color], alpha=0.9, linewidths=2.5)

            centroid = cluster_proj.mean(axis=0)
            label_positions_cluster.append((centroid, display_label, color))

            pts_sub = random_subsample(gt_proj, max_points)
            ax_gt.scatter(
                pts_sub[:, 0], pts_sub[:, 1],
                c=color, s=30, alpha=0.5,
                edgecolors='white', linewidths=0.3,
            )

            if len(gt_proj) > 15:
                xc, yc, H = compute_density_2d(gt_proj)
                ax_gt.contour(xc, yc, H, levels=4, colors=[color], alpha=0.9, linewidths=2.5)

            centroid = gt_proj.mean(axis=0)
            label_positions_gt.append((centroid, gt_label, color))

        adjusted_cluster = adjust_label_positions(
            label_positions_cluster, (x_min, x_max), (y_min, y_max)
        )
        adjusted_gt = adjust_label_positions(
            label_positions_gt, (x_min, x_max), (y_min, y_max)
        )

        for pos, label, color in adjusted_cluster:
            ax_cluster.annotate(
                label, pos, fontsize=9, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.9,
                          edgecolor='white', linewidth=1.5),
                color='white', zorder=10,
            )

        for pos, label, color in adjusted_gt:
            ax_gt.annotate(
                label, pos, fontsize=9, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.9,
                          edgecolor='white', linewidth=1.5),
                color='white', zorder=10,
            )

        for ax in [ax_cluster, ax_gt]:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)", fontsize=12)
            ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=10)

        total_cluster = sum(len(td["cluster_proj"]) for td in group_data)
        total_gt = sum(len(td["gt_proj"]) for td in group_data)

        ax_cluster.set_title(
            f"Discovered ({len(group_data)} types, {total_cluster} frames)",
            fontsize=13, fontweight='bold', pad=10
        )
        ax_gt.set_title(
            f"Ground Truth ({len(group_data)} types, {total_gt} frames)",
            fontsize=13, fontweight='bold', pad=10
        )

        feature_label = "Intention" if feature_type == "intention" else "Qpos"
        title = f"{method_name} (seed {seed}): {group_name} Transitions [{feature_label}]" if method_name else f"{group_name} Transitions [{feature_label}]"
        fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        safe_group = group_name.replace(" ", "_").replace("/", "-")
        save_path = os.path.join(
            output_dir,
            f"{method_name}_seed{seed}_transitions_{feature_type}_{safe_group}.pdf" if method_name else f"transitions_{feature_type}_{safe_group}.pdf"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()

        print(f"    Saved {group_name} [{feature_type}] ({len(group_data)} transitions) to {save_path}")


def plot_transition_cluster_comparison(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    behavior_labels_csv: str,
    rodent_rollout_dir: str,
    output_dir: str,
    transition_window: int = 5,
    max_points: int = 500,
    method_name: str = "",
    seed: int = 0,
    figsize: tuple[float, float] = (16, 12),
) -> None:
    """
    Create behavior-grouped comparison plots with shared PCA.

    Generates plots for BOTH intention and qpos features.
    Groups transitions by behavior type (Immobility, Walking, Rear, etc.).
    PCA is computed once on ALL frames per feature type.
    Each group gets a separate figure with left (discovered) and right (GT) panels.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    clip_behaviors = load_behavior_segments_multilabel(behavior_labels_csv)
    if not clip_behaviors:
        print("    No behavior segments loaded, skipping transition comparison")
        return

    grouped_clusters = defaultdict(lambda: {
        "cluster_ids": [],
        "original_labels": [],
        "parsed": None,
    })

    for cid, label in cluster_labels.items():
        parsed = parse_cluster_label(label)
        if parsed is None:
            continue
        if parsed.comparison_type == "single":
            continue

        gt_key = parsed.gt_key
        grouped_clusters[gt_key]["cluster_ids"].append(cid)
        grouped_clusters[gt_key]["original_labels"].append(label)
        grouped_clusters[gt_key]["parsed"] = parsed

    if not grouped_clusters:
        print("    No transition/intersection clusters found, skipping comparison")
        return

    all_clip_ids = set()
    for gt_key, info in grouped_clusters.items():
        for cid in info["cluster_ids"]:
            entries = cluster_dict.get(cid, [])
            for clip_id, _ in entries:
                all_clip_ids.add(int(clip_id))

    for clip_id in clip_behaviors:
        all_clip_ids.add(clip_id)

    print(f"    Loading features for {len(all_clip_ids)} clips...")
    intention_cache = {}
    qpos_cache = {}
    for clip_id in tqdm(sorted(all_clip_ids), desc="    Loading features"):
        try:
            intention_cache[clip_id] = load_intention(clip_id, rodent_rollout_dir)
        except Exception:
            pass
        try:
            qpos_cache[clip_id] = load_qpos(clip_id, rodent_rollout_dir)
        except Exception:
            pass

    if not intention_cache and not qpos_cache:
        print("    No features loaded, skipping comparison")
        return

    def build_transition_data(feature_cache: dict, feature_name: str) -> list[dict]:
        data = []
        for gt_key, group_info in sorted(grouped_clusters.items()):
            cluster_ids = group_info["cluster_ids"]
            original_labels = group_info["original_labels"]
            parsed = group_info["parsed"]

            comparison_type = parsed.comparison_type
            primitives = parsed.primitives
            compositional = parsed.compositional

            display_label = original_labels[0]
            if len(cluster_ids) > 1:
                display_label = f"{display_label} (+{len(cluster_ids)-1} more)"

            cluster_features = []
            for cid in cluster_ids:
                entries = cluster_dict.get(cid, [])
                for clip_id, frames in entries:
                    clip_id = int(clip_id)
                    if clip_id not in feature_cache:
                        continue
                    features = feature_cache[clip_id]
                    frames = np.asarray(frames, dtype=int)
                    frames = frames[frames < len(features)]
                    if len(frames) > 0:
                        cluster_features.append(features[frames])

            if not cluster_features:
                continue

            cluster_points = np.vstack(cluster_features)

            if comparison_type == "transition":
                gt_frames = find_gt_transition_frames(
                    clip_behaviors, primitives, window=transition_window
                )
                gt_label = f"{primitives[0]} <-> {primitives[1]}"
            else:
                all_behaviors = primitives + compositional
                gt_frames = find_gt_intersection_frames(clip_behaviors, all_behaviors)
                gt_label = " + ".join(all_behaviors)

            gt_features = []
            for clip_id, frame in gt_frames:
                if clip_id not in feature_cache:
                    continue
                features = feature_cache[clip_id]
                if frame < len(features):
                    gt_features.append(features[frame])

            if not gt_features:
                continue

            gt_points = np.array(gt_features)

            all_involved = primitives + compositional
            behavior_group = get_behavior_group(all_involved)

            data.append({
                "gt_key": gt_key,
                "cluster_points": cluster_points,
                "gt_points": gt_points,
                "display_label": display_label,
                "gt_label": gt_label,
                "behavior_group": behavior_group,
                "primitives": primitives,
                "compositional": compositional,
            })
        return data

    if intention_cache:
        print("    Generating intention-based transition plots...")
        intention_data = build_transition_data(intention_cache, "intention")
        if intention_data:
            _plot_transition_groups(
                intention_data, output_dir, "intention",
                method_name, seed, max_points, figsize
            )
        else:
            print("    No valid intention data for transitions")

    if qpos_cache:
        print("    Generating qpos-based transition plots...")
        qpos_data = build_transition_data(qpos_cache, "qpos")
        if qpos_data:
            _plot_transition_groups(
                qpos_data, output_dir, "qpos",
                method_name, seed, max_points, figsize
            )
        else:
            print("    No valid qpos data for transitions")


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


def compute_frames_variance(
    frames_list: list[tuple[int, np.ndarray]],
    rodent_rollout_dir: str,
    body_parts: list[str],
    max_loops: int = 200,
) -> dict[str, float]:
    """
    Compute variance for each body part from discovered cluster frames.

    Args:
        frames_list: List of (clip_id, frame_indices) tuples
        rodent_rollout_dir: Path to rollout data
        body_parts: List of body part names to compute variance for
        max_loops: Maximum number of entries to process

    Returns:
        Dict mapping body part name to variance value
    """
    body_indices = []
    valid_body_parts = []
    for bp in body_parts:
        if bp in QPOS_BODY_NAMES:
            body_indices.append(QPOS_BODY_NAMES.index(bp))
            valid_body_parts.append(bp)

    all_qpos_by_body = {bp: [] for bp in valid_body_parts}

    for i, (clip_id, frames) in enumerate(frames_list):
        if i >= max_loops:
            break

        try:
            qpos = load_qpos(clip_id, rodent_rollout_dir)
        except Exception:
            continue

        frames = np.asarray(frames, dtype=int)
        frames = frames[frames < len(qpos)]

        if len(frames) < 1:
            continue

        for bp, body_idx in zip(valid_body_parts, body_indices):
            if body_idx < qpos.shape[1]:
                all_qpos_by_body[bp].append(qpos[frames, body_idx])

    variance_dict = {}
    for bp in valid_body_parts:
        if all_qpos_by_body[bp]:
            combined = np.concatenate(all_qpos_by_body[bp])
            variance_dict[bp] = float(np.var(combined))
        else:
            variance_dict[bp] = 0.0

    return variance_dict


def compute_gt_frames_variance(
    gt_frames: list[tuple[int, int]],
    rodent_rollout_dir: str,
    body_parts: list[str],
) -> dict[str, float]:
    """
    Compute variance for each body part from GT transition frames.

    Args:
        gt_frames: List of (clip_id, frame) tuples
        rodent_rollout_dir: Path to rollout data
        body_parts: List of body part names to compute variance for

    Returns:
        Dict mapping body part name to variance value
    """
    body_indices = []
    valid_body_parts = []
    for bp in body_parts:
        if bp in QPOS_BODY_NAMES:
            body_indices.append(QPOS_BODY_NAMES.index(bp))
            valid_body_parts.append(bp)

    all_qpos_by_body = {bp: [] for bp in valid_body_parts}
    qpos_cache = {}

    for clip_id, frame in gt_frames:
        clip_id = int(clip_id)
        if clip_id not in qpos_cache:
            try:
                qpos_cache[clip_id] = load_qpos(clip_id, rodent_rollout_dir)
            except Exception:
                continue

        qpos = qpos_cache[clip_id]
        if frame >= len(qpos):
            continue

        for bp, body_idx in zip(valid_body_parts, body_indices):
            if body_idx < qpos.shape[1]:
                all_qpos_by_body[bp].append(qpos[frame, body_idx])

    variance_dict = {}
    for bp in valid_body_parts:
        if all_qpos_by_body[bp]:
            values = np.array(all_qpos_by_body[bp])
            variance_dict[bp] = float(np.var(values))
        else:
            variance_dict[bp] = 0.0

    return variance_dict


def render_gt_transition_grid(
    gt_frames_by_transition: dict[str, list[tuple[int, int]]],
    rodent_rollout_dir: str,
    output_dir: str,
    max_samples: int = 5,
    framerate: int = 30,
    target_size: tuple[int, int] = (320, 240),
    method_name: str = "",
    seed: int = 0,
) -> None:
    """
    Render GT transition frames in a grid video.

    Each transition includes all frames from both consecutive behavior segments.

    Args:
        gt_frames_by_transition: Dict mapping transition label to list of (clip_id, frame)
        rodent_rollout_dir: Path to rollout data
        output_dir: Directory to save output videos
        max_samples: Maximum samples per transition type
        framerate: Video framerate
        target_size: (width, height) per tile
        method_name: Name for output filename
        seed: Seed for output filename
    """
    os.makedirs(output_dir, exist_ok=True)

    cluster_dict_format = {}
    for i, (transition_label, frames) in enumerate(sorted(gt_frames_by_transition.items())):
        grouped_by_clip = defaultdict(list)
        for clip_id, frame in frames:
            grouped_by_clip[clip_id].append(frame)

        entries = []
        for clip_id, frame_list in grouped_by_clip.items():
            frame_arr = np.array(sorted(set(frame_list)), dtype=int)
            entries.append((clip_id, frame_arr))

        if max_samples and len(entries) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(entries), max_samples, replace=False)
            entries = [entries[j] for j in indices]

        cluster_dict_format[i] = entries

    if not cluster_dict_format:
        print("    No GT transitions to render")
        return

    print(f"    Loading GT transition frames for rendering...")
    cluster_chunks = {}
    cluster_clip_ids = {}

    for cid, entries in tqdm(cluster_dict_format.items(), desc="    Rendering GT"):
        cluster_chunks[cid] = []
        cluster_clip_ids[cid] = []

        for clip_id, frames in entries:
            try:
                rollout = load_rollout_from_source(clip_id, rodent_rollout_dir)
                full_frames = render_from_saved_rollout(rollout)
                T = len(full_frames)

                valid_frames = sorted([f for f in frames if 0 <= f < T])
                if valid_frames:
                    chunk = [full_frames[f] for f in valid_frames]
                    cluster_chunks[cid].append(chunk)
                    cluster_clip_ids[cid].append(clip_id)
            except Exception:
                continue

    cluster_labels = {i: label for i, label in enumerate(sorted(gt_frames_by_transition.keys()))}

    prefix = f"{method_name}_seed{seed}" if method_name else f"seed{seed}"
    out_path = os.path.join(output_dir, f"{prefix}_gt_transitions.mp4")

    try:
        render_grid_clusters_side_by_side(
            cluster_chunks=cluster_chunks,
            out_path=out_path,
            framerate=framerate,
            max_samples_per_cluster=max_samples,
            target_size=target_size,
            title=f"GT Transitions ({method_name} seed {seed})" if method_name else "GT Transitions",
            per_tile_labels=True,
            cluster_labels=cluster_labels,
            cluster_clip_ids=cluster_clip_ids,
        )
        print(f"    Saved GT transition video to {out_path}")
    except Exception as e:
        print(f"    Failed to render GT transitions: {e}")


def compute_javd_table(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    behavior_labels_csv: str,
    rodent_rollout_dir: str,
    output_dir: str,
    body_parts: list[str],
    top_k: int = 5,
    transition_window: int = 5,
    max_loops: int = 200,
    method_name: str = "",
    seed: int = 0,
    render_gt: bool = False,
    gt_render_max_samples: int = 5,
    gt_render_framerate: int = 30,
    gt_render_target_size: tuple[int, int] = (320, 240),
) -> pd.DataFrame | None:
    """
    Compute JAVD (Joint Angle Variance Difference) table for transition clusters.

    Compares variance in joint angles between discovered transition clusters
    and ground truth transition frames. For each cluster, identifies the top K
    highest-variance joints from GT and computes the variance difference.

    Args:
        cluster_dict: Dict mapping cluster_id to list of (clip_id, frames)
        cluster_labels: Dict mapping cluster_id to label string
        behavior_labels_csv: Path to behavior labels CSV file
        rodent_rollout_dir: Path to rollout data
        output_dir: Directory to save output files
        body_parts: List of body part names to analyze
        top_k: Number of top variance joints to consider per cluster
        transition_window: Window size for finding GT transition frames
        max_loops: Maximum entries to process per cluster
        method_name: Name of the method for output filenames
        seed: Seed value for output filenames
        render_gt: Whether to render GT transition videos
        gt_render_max_samples: Max samples per transition for rendering
        gt_render_framerate: Video framerate for GT rendering
        gt_render_target_size: Target tile size for GT rendering

    Returns:
        DataFrame with JAVD table, or None if no transition clusters found
    """
    os.makedirs(output_dir, exist_ok=True)

    clip_behaviors = load_behavior_segments_multilabel(behavior_labels_csv)
    if not clip_behaviors:
        print("    No behavior segments loaded, skipping JAVD analysis")
        return None

    transition_clusters = []
    for cid, label in cluster_labels.items():
        parsed = parse_cluster_label(label)
        if parsed is None:
            continue
        if parsed.comparison_type != "transition":
            continue
        if len(parsed.primitives) < 2:
            continue

        transition_clusters.append({
            "cid": cid,
            "label": label,
            "parsed": parsed,
            "primitives": parsed.primitives,
        })

    if not transition_clusters:
        print("    No transition clusters found, skipping JAVD analysis")
        return None

    print(f"    Found {len(transition_clusters)} transition clusters for JAVD analysis")

    javd_data = {}
    all_top_joints = set()
    gt_frames_by_transition = {}

    for tc in tqdm(transition_clusters, desc="    Computing JAVD"):
        cid = tc["cid"]
        primitives = tc["primitives"]
        gt_label = f"{primitives[0]} <-> {primitives[1]}"

        entries = cluster_dict.get(cid, [])
        if not entries:
            continue

        gt_frames = find_gt_transition_frames_from_segments(
            behavior_labels_csv, primitives, max_gap=10
        )
        if not gt_frames:
            continue

        if gt_label not in gt_frames_by_transition:
            gt_frames_by_transition[gt_label] = []
        gt_frames_by_transition[gt_label].extend(gt_frames)

        n_discovered_frames = sum(len(frames) for _, frames in entries[:max_loops])
        n_gt_frames = len(gt_frames)

        discovered_variance = compute_frames_variance(
            entries, rodent_rollout_dir, body_parts, max_loops
        )
        gt_variance = compute_gt_frames_variance(
            gt_frames, rodent_rollout_dir, body_parts
        )

        sorted_gt_joints = sorted(
            gt_variance.items(), key=lambda x: x[1], reverse=True
        )
        top_joints = [joint for joint, _ in sorted_gt_joints[:top_k] if gt_variance[joint] > 0]

        javd_for_cluster = {}
        javd_norm_for_cluster = {}
        gt_var_for_cluster = {}
        disc_var_for_cluster = {}
        for joint in top_joints:
            var_disc = discovered_variance.get(joint, 0.0)
            var_gt = gt_variance.get(joint, 0.0)
            javd_for_cluster[joint] = abs(var_disc - var_gt)
            if var_gt > 0:
                javd_norm_for_cluster[joint] = abs(var_disc - var_gt) / var_gt
            else:
                javd_norm_for_cluster[joint] = np.nan
            gt_var_for_cluster[joint] = var_gt
            disc_var_for_cluster[joint] = var_disc
            all_top_joints.add(joint)

        javd_data[gt_label] = {
            "javd": javd_for_cluster,
            "javd_norm": javd_norm_for_cluster,
            "gt_var": gt_var_for_cluster,
            "disc_var": disc_var_for_cluster,
            "top_joints": top_joints,
            "cid": cid,
            "n_discovered_frames": n_discovered_frames,
            "n_gt_frames": n_gt_frames,
        }

    if not javd_data:
        print("    No valid JAVD data computed")
        return None

    all_joints_list = sorted(all_top_joints)
    cluster_labels_list = sorted(javd_data.keys())

    def build_variance_table(data_key: str) -> pd.DataFrame:
        table_data = []
        for joint in all_joints_list:
            row = {"Joint": joint}
            for cluster_name in cluster_labels_list:
                cluster_info = javd_data[cluster_name]
                if joint in cluster_info[data_key]:
                    row[cluster_name] = cluster_info[data_key][joint]
                else:
                    row[cluster_name] = np.nan
            table_data.append(row)

        df = pd.DataFrame(table_data)
        df = df.set_index("Joint")

        row_means = df.mean(axis=1, skipna=True)
        df["Average"] = row_means

        col_means = df.mean(axis=0, skipna=True)
        avg_row = pd.DataFrame([col_means], index=["Average"])
        df = pd.concat([df, avg_row])
        return df

    df_javd = build_variance_table("javd")
    df_javd_norm = build_variance_table("javd_norm")
    df_gt = build_variance_table("gt_var")
    df_disc = build_variance_table("disc_var")

    prefix = f"{method_name}_seed{seed}" if method_name else f"seed{seed}"

    javd_path = os.path.join(output_dir, f"{prefix}_javd_table.csv")
    df_javd.to_csv(javd_path, float_format="%.6f")
    print(f"    Saved JAVD (difference) table to {javd_path}")

    javd_norm_path = os.path.join(output_dir, f"{prefix}_javd_normalized_table.csv")
    df_javd_norm.to_csv(javd_norm_path, float_format="%.6f")
    print(f"    Saved JAVD (normalized) table to {javd_norm_path}")

    gt_path = os.path.join(output_dir, f"{prefix}_gt_variance_table.csv")
    df_gt.to_csv(gt_path, float_format="%.6f")
    print(f"    Saved GT variance table to {gt_path}")

    disc_path = os.path.join(output_dir, f"{prefix}_discovered_variance_table.csv")
    df_disc.to_csv(disc_path, float_format="%.6f")
    print(f"    Saved Discovered variance table to {disc_path}")

    sample_counts = []
    for cluster_name in cluster_labels_list:
        info = javd_data[cluster_name]
        sample_counts.append({
            "Transition": cluster_name,
            "GT Frames": info["n_gt_frames"],
            "Discovered Frames": info["n_discovered_frames"],
            "Cluster ID": info["cid"],
        })
    df_counts = pd.DataFrame(sample_counts)

    counts_path = os.path.join(output_dir, f"{prefix}_sample_counts.csv")
    df_counts.to_csv(counts_path, index=False)
    print(f"    Saved sample counts to {counts_path}")

    print(f"\n    === Sample Counts ({method_name} seed {seed}) ===")
    print(df_counts.to_string(index=False))
    print()

    low_sample_transitions = df_counts[
        (df_counts["GT Frames"] < 50) | (df_counts["Discovered Frames"] < 50)
    ]["Transition"].tolist()
    if low_sample_transitions:
        print(f"    WARNING: Low sample count (<50 frames) for: {low_sample_transitions}")
        print()

    print(f"\n    === JAVD Table (|Discovered - GT|) ({method_name} seed {seed}) ===")
    print(df_javd.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else ""))

    print(f"\n    === JAVD Normalized Table (|Disc - GT| / GT) ({method_name} seed {seed}) ===")
    print(df_javd_norm.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else ""))

    print(f"\n    === GT Variance Table ({method_name} seed {seed}) ===")
    print(df_gt.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else ""))

    print(f"\n    === Discovered Variance Table ({method_name} seed {seed}) ===")
    print(df_disc.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else ""))
    print()

    overall_javd = df_javd.loc["Average", "Average"]
    overall_javd_norm = df_javd_norm.loc["Average", "Average"]
    overall_gt = df_gt.loc["Average", "Average"]
    overall_disc = df_disc.loc["Average", "Average"]
    print(f"    Overall Mean JAVD: {overall_javd:.6f}")
    print(f"    Overall Mean JAVD (normalized): {overall_javd_norm:.4f} ({overall_javd_norm*100:.1f}%)")
    print(f"    Overall Mean GT Variance: {overall_gt:.6f}")
    print(f"    Overall Mean Discovered Variance: {overall_disc:.6f}")

    if render_gt and gt_frames_by_transition:
        print(f"\n    Rendering GT transition videos...")
        render_gt_transition_grid(
            gt_frames_by_transition=gt_frames_by_transition,
            rodent_rollout_dir=rodent_rollout_dir,
            output_dir=output_dir,
            max_samples=gt_render_max_samples,
            framerate=gt_render_framerate,
            target_size=gt_render_target_size,
            method_name=method_name,
            seed=seed,
        )

    return df_javd


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


def plot_variance_heatmaps_side_by_side(
    cluster_dicts: dict[int, dict],
    cluster_labels_per_seed: dict[int, dict[int, str]],
    rodent_rollout_dir: str,
    output_dir: str,
    body_parts: list[str] = None,
    max_loops: int = 200,
    method_name: str = "",
) -> None:
    """
    Plot variance heatmaps for multiple seeds side by side.

    Args:
        cluster_dicts: Dict mapping seed -> cluster_dict
        cluster_labels_per_seed: Dict mapping seed -> {cluster_id: label}
        rodent_rollout_dir: Path to rollout data
        output_dir: Output directory
        body_parts: List of body parts to analyze
        max_loops: Max loops per cluster
        method_name: Method name for title
    """
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

    save_path = os.path.join(output_dir, f"{method_name}_variance_heatmap_seeds.pdf" if method_name else "variance_heatmap_seeds.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved side-by-side variance heatmaps to {save_path}")


def run_analysis(config: AnalysisConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("Unified Analysis Pipeline")
    print("=" * 60)

    all_results = load_experiment_results(config.experiment_dir)
    print(f"Loaded results for {len(all_results)} methods")

    if config.methods:
        print(f"Filtering to methods: {config.methods}")
    if config.seeds:
        print(f"Filtering to seeds: {config.seeds}")

    behavior_labels = None
    if config.run_cluster_to_label_comparison:
        if os.path.exists(config.behavior_csv):
            behavior_labels = load_behavior_labels(config.behavior_csv)
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
                    target_clusters = list(cluster_dict.keys())
                    mds_save_path = os.path.join(seed_output_dir, f"{method_name}_seed{seed}_mds_separate.pdf")
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
                            figsize=config.mds_figsize,
                            save_path=mds_save_path,
                        )
                        print(f"    Saved MDS plot to {mds_save_path}")
                    except Exception as e:
                        print(f"    MDS analysis failed: {e}")

            if config.run_cluster_to_label_comparison and behavior_labels:
                print(f"    Running cluster-to-label comparison...")
                plot_cluster_to_label_comparison(
                    cluster_dict=cluster_dict,
                    cluster_labels=cluster_labels,
                    behavior_labels=behavior_labels,
                    rodent_rollout_dir=config.rodent_rollout_dir,
                    output_dir=seed_output_dir,
                    body_parts=config.body_parts,
                    max_loops=config.max_loops,
                    method_name=f"{method_name}_seed{seed}",
                )

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
                            transition_window=config.transition_window,
                            max_points=config.max_transition_points,
                            method_name=method_name,
                            seed=seed,
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
                            transition_window=config.transition_window,
                            max_loops=config.max_loops,
                            method_name=method_name,
                            seed=seed,
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
