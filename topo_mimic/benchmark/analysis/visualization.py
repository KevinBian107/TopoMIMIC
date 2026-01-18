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
    parse_turn_direction,
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
    turn_directions: dict[int, str] = None,
    behavior_labels_csv: str = "",
    rodent_rollout_dir: str = "",
    output_dir: str = "",
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

    if turn_directions is None:
        turn_directions = {}

    clip_behaviors = load_behavior_segments_multilabel(behavior_labels_csv, merge_rear)
    if not clip_behaviors:
        print("    No behavior segments loaded, skipping transition comparison")
        return

    grouped_clusters = defaultdict(lambda: {
        "cluster_ids": [],
        "original_labels": [],
        "turn_directions": [],
        "parsed": None,
    })

    for cid, label in cluster_labels.items():
        parsed = parse_cluster_label(label, merge_rear=merge_rear)
        if parsed is None:
            continue
        if parsed.comparison_type == "single":
            continue

        user_turn = turn_directions.get(cid, "")
        if user_turn:
            user_turn_enum = parse_turn_direction(user_turn)
            parsed.from_turn = user_turn_enum
            parsed.to_turn = user_turn_enum

        gt_key = parsed.gt_key_with_turns
        grouped_clusters[gt_key]["cluster_ids"].append(cid)
        grouped_clusters[gt_key]["original_labels"].append(label)
        grouped_clusters[gt_key]["turn_directions"].append(user_turn)
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
            user_turns = group_info["turn_directions"]
            parsed = group_info["parsed"]

            comparison_type = parsed.comparison_type
            primitives = parsed.primitives
            compositional = parsed.compositional

            turn_prefix = ""
            if user_turns and user_turns[0]:
                turn_short = turn_direction_to_string(parsed.from_turn)
                turn_prefix = f"[{turn_short}] "

            display_label = f"{turn_prefix}{original_labels[0]}"
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
                    include_turn_direction=True,
                )
                gt_frames = []
                gt_turn_prefix = ""
                if user_turns and user_turns[0]:
                    target_turn = parsed.from_turn
                    matching_transitions = [
                        t for t in gt_transitions
                        if t.from_turn == target_turn or t.to_turn == target_turn
                    ]
                    if matching_transitions:
                        gt_transitions = matching_transitions
                        gt_turn_prefix = f"[{turn_direction_to_string(target_turn)}] "
                for t in gt_transitions:
                    for f in t.all_frames:
                        gt_frames.append((t.clip_id, f))
                gt_label = f"{gt_turn_prefix}{primitives[0]} <-> {primitives[1]}"
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
    from topo_mimic.dmwk.utils.load_utils import load_rollout_from_source
    from topo_mimic.dmwk.utils.rendering import render_from_saved_rollout
    from topo_mimic.dmwk.skeletonization.rendering import render_grid_clusters_side_by_side

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
    turn_directions: dict[int, str] = None,
    rodent_rollout_dir: str = "",
    output_dir: str = "",
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

    if turn_directions is None:
        turn_directions = {}

    def get_cluster_display_name(cid: int) -> str:
        label = cluster_labels.get(cid, f"C{cid}").replace("_", " ")
        turn = turn_directions.get(cid, "")
        if turn:
            turn_short = {"left": "L", "right": "R", "straight": "S", "non-deterministic": "ND"}.get(turn.lower(), "")
            if turn_short:
                label = f"[{turn_short}] {label}"
        return label

    cluster_names = [get_cluster_display_name(cid) for cid in cluster_ids]
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


def compute_w2_distance_gaussian(
    points1: np.ndarray,
    points2: np.ndarray,
) -> float:
    """
    Compute W2 distance between two point sets using Gaussian approximation.

    W2²(P, Q) = ||μ_P - μ_Q||² + trace(Σ_P + Σ_Q - 2(Σ_P^{1/2} Σ_Q Σ_P^{1/2})^{1/2})
    """
    if len(points1) < 2 or len(points2) < 2:
        return np.inf

    mu1 = np.mean(points1, axis=0)
    mu2 = np.mean(points2, axis=0)

    cov1 = np.cov(points1.T)
    cov2 = np.cov(points2.T)

    if cov1.ndim == 0:
        cov1 = np.array([[cov1]])
    if cov2.ndim == 0:
        cov2 = np.array([[cov2]])

    mean_diff = np.sum((mu1 - mu2) ** 2)

    try:
        sqrt_cov1 = np.linalg.cholesky(cov1 + np.eye(cov1.shape[0]) * 1e-6)
        product = sqrt_cov1 @ cov2 @ sqrt_cov1.T
        eigvals = np.linalg.eigvalsh(product)
        eigvals = np.maximum(eigvals, 0)
        sqrt_product = np.sum(np.sqrt(eigvals))

        trace_term = np.trace(cov1) + np.trace(cov2) - 2 * sqrt_product
        trace_term = max(trace_term, 0)

        w2_squared = mean_diff + trace_term
        return np.sqrt(w2_squared)
    except np.linalg.LinAlgError:
        return np.sqrt(mean_diff)


def plot_transition_graph(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    turn_directions: dict[int, str] = None,
    behavior_labels_csv: str = "",
    rodent_rollout_dir: str = "",
    output_dir: str = "",
    k_neighbors: int = 3,
    gt_top_k: int = 10,
    layout: str = "spring",
    figsize: tuple[float, float] = (16, 7),
    method_name: str = "",
    seed: int = 0,
    short_threshold: int = 10,
    merge_rear: bool = True,
) -> None:
    """
    Create k-NN transition graphs comparing discovered clusters vs GT.

    Nodes are transition clusters (with turn directions as separate nodes).
    Edges connect k nearest neighbors based on W2 distance of intention distributions.

    Args:
        gt_top_k: Number of top GT transition types to include (by frequency)
    """
    import networkx as nx

    os.makedirs(output_dir, exist_ok=True)

    if turn_directions is None:
        turn_directions = {}

    discovered_nodes = {}
    for cid, label in cluster_labels.items():
        parsed = parse_cluster_label(label, merge_rear=merge_rear)
        if parsed is None:
            continue
        if parsed.comparison_type != "transition":
            continue
        if len(parsed.primitives) < 2:
            continue

        user_turn = turn_directions.get(cid, "")
        if user_turn:
            turn_enum = parse_turn_direction(user_turn)
            turn_short = turn_direction_to_string(turn_enum)
            node_label = f"[{turn_short}] {parsed.primitives[0]} <-> {parsed.primitives[1]}"
        else:
            node_label = f"{parsed.primitives[0]} <-> {parsed.primitives[1]}"

        entries = cluster_dict.get(cid, [])
        if not entries:
            continue

        discovered_nodes[cid] = {
            "label": node_label,
            "entries": entries,
            "primitives": parsed.primitives,
            "turn": user_turn,
        }

    if len(discovered_nodes) < 2:
        print("    Not enough transition clusters for graph, skipping")
        return

    print(f"    Loading intention data for {len(discovered_nodes)} discovered clusters...")
    intention_cache = {}
    all_clip_ids = set()
    for cid, info in discovered_nodes.items():
        for clip_id, _ in info["entries"]:
            all_clip_ids.add(int(clip_id))

    for clip_id in tqdm(sorted(all_clip_ids), desc="    Loading intentions"):
        try:
            intention_cache[clip_id] = load_intention(clip_id, rodent_rollout_dir)
        except Exception:
            pass

    discovered_features = {}
    for cid, info in discovered_nodes.items():
        features = []
        for clip_id, frames in info["entries"]:
            clip_id = int(clip_id)
            if clip_id not in intention_cache:
                continue
            intention = intention_cache[clip_id]
            frames = np.asarray(frames, dtype=int)
            frames = frames[frames < len(intention)]
            if len(frames) > 0:
                features.append(intention[frames])
        if features:
            discovered_features[cid] = np.vstack(features)

    if len(discovered_features) < 2:
        print("    Not enough clusters with features, skipping")
        return

    from .behavior_parsing import PRIMITIVE_BEHAVIORS
    from itertools import combinations

    all_primitives = list(PRIMITIVE_BEHAVIORS)
    if merge_rear:
        all_primitives = [p for p in all_primitives if p not in {"Supported Rear", "Unsupported Rear"}]
        if "Rear" not in all_primitives:
            all_primitives.append("Rear")

    print(f"    Loading all GT transitions for {len(all_primitives)} behaviors...")
    all_gt_transitions = []
    for b1, b2 in combinations(all_primitives, 2):
        transitions = find_gt_transition_frames(
            behavior_labels_csv,
            [b1, b2],
            short_threshold=short_threshold,
            merge_rear=merge_rear,
            include_turn_direction=True,
        )
        all_gt_transitions.extend(transitions)

    gt_nodes = defaultdict(list)
    for t in all_gt_transitions:
        turn = t.from_turn if t.from_turn != TurnDirection.STRAIGHT else t.to_turn
        turn_short = turn_direction_to_string(turn)
        behaviors_sorted = tuple(sorted([t.from_behavior, t.to_behavior]))
        if turn == TurnDirection.STRAIGHT:
            node_label = f"{behaviors_sorted[0]} <-> {behaviors_sorted[1]}"
        else:
            node_label = f"[{turn_short}] {behaviors_sorted[0]} <-> {behaviors_sorted[1]}"
        gt_nodes[node_label].append(t)

    gt_node_counts = {label: len(transitions) for label, transitions in gt_nodes.items()}
    top_gt_labels = sorted(gt_node_counts.keys(), key=lambda x: gt_node_counts[x], reverse=True)[:gt_top_k]
    gt_nodes = {label: gt_nodes[label] for label in top_gt_labels}

    print(f"    Selected top {len(gt_nodes)} GT transition types (by frequency)")

    for t in all_gt_transitions:
        if t.clip_id not in intention_cache:
            try:
                intention_cache[t.clip_id] = load_intention(t.clip_id, rodent_rollout_dir)
            except Exception:
                pass

    gt_features = {}
    for node_label, transitions in gt_nodes.items():
        features = []
        for t in transitions:
            if t.clip_id not in intention_cache:
                continue
            intention = intention_cache[t.clip_id]
            for frame in t.all_frames:
                if frame < len(intention):
                    features.append(intention[frame])
        if features:
            gt_features[node_label] = np.array(features)

    if len(gt_features) < 2:
        print("    Not enough GT nodes with features, skipping")
        return

    print(f"    Computing W2 distances for {len(discovered_features)} discovered nodes...")
    disc_cids = list(discovered_features.keys())
    disc_n = len(disc_cids)
    disc_dist_matrix = np.zeros((disc_n, disc_n))
    for i in range(disc_n):
        for j in range(i + 1, disc_n):
            d = compute_w2_distance_gaussian(
                discovered_features[disc_cids[i]],
                discovered_features[disc_cids[j]]
            )
            disc_dist_matrix[i, j] = d
            disc_dist_matrix[j, i] = d

    print(f"    Computing W2 distances for {len(gt_features)} GT nodes...")
    gt_labels = list(gt_features.keys())
    gt_n = len(gt_labels)
    gt_dist_matrix = np.zeros((gt_n, gt_n))
    for i in range(gt_n):
        for j in range(i + 1, gt_n):
            d = compute_w2_distance_gaussian(
                gt_features[gt_labels[i]],
                gt_features[gt_labels[j]]
            )
            gt_dist_matrix[i, j] = d
            gt_dist_matrix[j, i] = d

    def build_knn_graph(labels, dist_matrix, node_info, k):
        G = nx.Graph()
        n = len(labels)
        for i, label in enumerate(labels):
            if isinstance(label, int):
                display_label = node_info[label]["label"]
            else:
                display_label = label
            G.add_node(i, label=display_label)

        k_actual = min(k, n - 1)
        for i in range(n):
            distances = dist_matrix[i].copy()
            distances[i] = np.inf
            nearest = np.argsort(distances)[:k_actual]
            for j in nearest:
                if not G.has_edge(i, j):
                    G.add_edge(i, j, weight=distances[j])
        return G

    G_disc = build_knn_graph(disc_cids, disc_dist_matrix, discovered_nodes, k_neighbors)
    G_gt = build_knn_graph(gt_labels, gt_dist_matrix, {}, k_neighbors)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    def get_layout(G, layout_name):
        if layout_name == "spring":
            return nx.spring_layout(G, seed=42, k=2)
        elif layout_name == "kamada_kawai":
            return nx.kamada_kawai_layout(G)
        elif layout_name == "spectral":
            return nx.spectral_layout(G)
        else:
            return nx.spring_layout(G, seed=42)

    behavior_colors = {
        "Walking": "#3498DB",
        "Rear": "#E74C3C",
        "Immobility": "#95A5A6",
        "Orienting": "#9B59B6",
        "Grooming": "#2ECC71",
    }

    def get_node_color(label):
        for behavior, color in behavior_colors.items():
            if behavior in label:
                return color
        return "#F39C12"

    ax = axes[0]
    pos_disc = get_layout(G_disc, layout)
    node_colors_disc = [get_node_color(G_disc.nodes[n]["label"]) for n in G_disc.nodes()]
    node_labels_disc = {n: G_disc.nodes[n]["label"] for n in G_disc.nodes()}

    edges = G_disc.edges(data=True)
    if edges:
        weights = [e[2].get("weight", 1) for e in edges]
        max_w = max(weights) if weights else 1
        edge_widths = [2.0 * (1 - w / max_w) + 0.5 for w in weights]
    else:
        edge_widths = []

    nx.draw_networkx_edges(G_disc, pos_disc, ax=ax, width=edge_widths, alpha=0.6, edge_color="#888888")
    nx.draw_networkx_nodes(G_disc, pos_disc, ax=ax, node_color=node_colors_disc,
                           node_size=800, edgecolors="white", linewidths=2)

    for node, (x, y) in pos_disc.items():
        label = node_labels_disc[node]
        ax.annotate(label, (x, y), fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_title(f"Discovered Clusters (k={k_neighbors})", fontsize=13, fontweight="bold")
    ax.axis("off")

    ax = axes[1]
    pos_gt = get_layout(G_gt, layout)
    node_colors_gt = [get_node_color(G_gt.nodes[n]["label"]) for n in G_gt.nodes()]
    node_labels_gt = {n: G_gt.nodes[n]["label"] for n in G_gt.nodes()}

    edges = G_gt.edges(data=True)
    if edges:
        weights = [e[2].get("weight", 1) for e in edges]
        max_w = max(weights) if weights else 1
        edge_widths = [2.0 * (1 - w / max_w) + 0.5 for w in weights]
    else:
        edge_widths = []

    nx.draw_networkx_edges(G_gt, pos_gt, ax=ax, width=edge_widths, alpha=0.6, edge_color="#888888")
    nx.draw_networkx_nodes(G_gt, pos_gt, ax=ax, node_color=node_colors_gt,
                           node_size=800, edgecolors="white", linewidths=2)

    for node, (x, y) in pos_gt.items():
        label = node_labels_gt[node]
        ax.annotate(label, (x, y), fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_title(f"Ground Truth (k={k_neighbors})", fontsize=13, fontweight="bold")
    ax.axis("off")

    title = f"Transition Cluster k-NN Graph (W2 Distance)"
    if method_name:
        title = f"{method_name} seed {seed}: {title}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    graph_dir = os.path.join(output_dir, "transition_graph")
    os.makedirs(graph_dir, exist_ok=True)

    prefix = f"{method_name}_seed{seed}" if method_name else f"seed{seed}"
    save_path = os.path.join(graph_dir, f"{prefix}_transition_knn_graph.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved transition graph to {save_path}")

    disc_dist_df_data = []
    for i, cid_i in enumerate(disc_cids):
        row = {"Cluster": discovered_nodes[cid_i]["label"]}
        for j, cid_j in enumerate(disc_cids):
            row[discovered_nodes[cid_j]["label"]] = disc_dist_matrix[i, j]
        disc_dist_df_data.append(row)

    import pandas as pd
    disc_dist_df = pd.DataFrame(disc_dist_df_data).set_index("Cluster")
    disc_dist_path = os.path.join(graph_dir, f"{prefix}_discovered_w2_distances.csv")
    disc_dist_df.to_csv(disc_dist_path, float_format="%.4f")
    print(f"    Saved discovered W2 distance matrix to {disc_dist_path}")
