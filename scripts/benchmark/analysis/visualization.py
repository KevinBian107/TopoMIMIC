"""Visualization utilities for transition analysis."""

import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from tqdm import tqdm

from .behavior_parsing import (
    TurnDirection,
    ParsedLabel,
    parse_cluster_label,
    turn_direction_to_string,
)
from .ground_truth import (
    TransitionSegment,
    find_gt_transition_frames,
    load_behavior_segments_multilabel,
    find_gt_intersection_frames,
)
from .data_loading import load_intention, load_qpos


BEHAVIOR_GROUPS = {
    "Immobility": ["Immobility"],
    "Walking": ["Walking"],
    "Rear": ["Rear", "Supported Rear", "Unsupported Rear"],
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

        transitions_dir = os.path.join(output_dir, "transitions")
        os.makedirs(transitions_dir, exist_ok=True)

        safe_group = group_name.replace(" ", "_").replace("/", "-")
        save_path = os.path.join(
            transitions_dir,
            f"{method_name}_seed{seed}_transitions_{feature_type}_{safe_group}.pdf" if method_name else f"transitions_{feature_type}_{safe_group}.pdf"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
    short_threshold: int = 10,
    merge_rear: bool = True,
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

    clip_behaviors = load_behavior_segments_multilabel(behavior_labels_csv, merge_rear)
    if not clip_behaviors:
        print("    No behavior segments loaded, skipping transition comparison")
        return

    grouped_clusters = defaultdict(lambda: {
        "cluster_ids": [],
        "original_labels": [],
        "parsed": None,
    })

    for cid, label in cluster_labels.items():
        parsed = parse_cluster_label(label, merge_rear=merge_rear)
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
                gt_transitions = find_gt_transition_frames(
                    behavior_labels_csv,
                    primitives,
                    short_threshold=short_threshold,
                    merge_rear=merge_rear,
                )
                gt_frames = []
                for t in gt_transitions:
                    for f in t.all_frames:
                        gt_frames.append((t.clip_id, f))
                gt_label = f"{primitives[0]} <-> {primitives[1]}"
            else:
                all_behaviors = primitives + compositional
                gt_frames = find_gt_intersection_frames(clip_behaviors, all_behaviors, merge_rear)
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


def render_gt_transition_grid(
    gt_transitions: list,
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

    Args:
        gt_transitions: List of TransitionSegment objects grouped by label
        rodent_rollout_dir: Path to rollout data
        output_dir: Directory to save output videos
        max_samples: Maximum samples per transition type
        framerate: Video framerate
        target_size: (width, height) per tile
        method_name: Name for output filename
        seed: Seed for output filename
    """
    from collections import defaultdict
    from vnl_tda.utils.load_utils import load_rollout_from_source
    from vnl_tda.utils.rendering import render_from_saved_rollout
    from vnl_tda.pipelines.tda.skeletonization.rendering import render_grid_clusters_side_by_side

    os.makedirs(output_dir, exist_ok=True)

    gt_frames_by_transition = defaultdict(list)
    for t in gt_transitions:
        label = f"{t.from_behavior} <-> {t.to_behavior}"
        for f in t.all_frames:
            gt_frames_by_transition[label].append((t.clip_id, f))

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

    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    prefix = f"{method_name}_seed{seed}" if method_name else f"seed{seed}"
    out_path = os.path.join(videos_dir, f"{prefix}_gt_transitions.mp4")

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
    from .data_loading import QPOS_BODY_NAMES

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

    variance_dir = os.path.join(output_dir, "variance")
    os.makedirs(variance_dir, exist_ok=True)

    save_path = os.path.join(variance_dir, f"{method_name}_variance_heatmap.pdf" if method_name else "variance_heatmap.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved variance heatmap to {save_path}")
