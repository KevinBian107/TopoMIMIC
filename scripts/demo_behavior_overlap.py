# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "h5py", "matplotlib", "scikit-learn", "scipy", "pyyaml", "umap-learn"]
# ///
"""
Visualize transition clusters with individual behavior clusters on the side.

Main panel: Shows where transition frames cluster (Immobility↔Walking, Orienting↔Walking, Immobility↔Orienting)
Side panels: Individual behavior clusters for reference
"""

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import yaml
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, ConnectionPatch
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


np.random.seed(42)

BEHAVIORS = [
    "Walking",
    "Immobility",
    "Orienting",
    "Grooming",
    "Rear",
]

BEHAVIOR_COLORS = {
    "Walking": "#4A90D9",
    "Immobility": "#7BC47F",
    "Orienting": "#F5A623",
    "Grooming": "#E91E63",
    "Rear": "#9C27B0",
}

BEHAVIOR_MAPPING = {
    "Supported Rear": "Rear",
    "Unsupported Rear": "Rear",
}

TRANSITION_PALETTE = [
    "#E74C3C", "#9B59B6", "#1ABC9C", "#E67E22", "#3498DB",
    "#2ECC71", "#F39C12", "#8E44AD", "#16A085", "#C0392B",
]


def get_transition_color(idx: int) -> str:
    return TRANSITION_PALETTE[idx % len(TRANSITION_PALETTE)]


def get_transition_label(trans_key: tuple[str, str]) -> str:
    return f"{trans_key[0]} ↔ {trans_key[1]}"


@dataclass
class Config:
    data_source: str = "data/rodent_ar_latest.h5"
    labels_path: str = "data/behavior_labels.csv"
    output_path: str = "out/behavior_clusters.png"
    n_clips: int = 200
    max_points_per_behavior: int = 1000
    max_transition_points: int = 500
    transition_window: int = 5
    top_k_transitions: int = 2
    figsize: tuple[int, int] = (16, 10)
    density_only: bool = False
    show_side_panels: bool = True
    projection: str = "pca"

    @classmethod
    def from_yaml(cls, yaml_path: Path, base_dir: Path) -> "Config":
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        config = cls(
            data_source=str(base_dir / data.get("data_source", cls.data_source)),
            labels_path=str(base_dir / data.get("labels_path", cls.labels_path)),
            output_path=str(base_dir / data.get("output_path", cls.output_path)),
            n_clips=data.get("n_clips", cls.n_clips),
            max_points_per_behavior=data.get("max_points_per_behavior", cls.max_points_per_behavior),
            max_transition_points=data.get("max_transition_points", cls.max_transition_points),
            transition_window=data.get("transition_window", cls.transition_window),
            top_k_transitions=data.get("top_k_transitions", cls.top_k_transitions),
            figsize=tuple(data.get("figsize", cls.figsize)),
            density_only=data.get("density_only", cls.density_only),
            show_side_panels=data.get("show_side_panels", cls.show_side_panels),
            projection=data.get("projection", cls.projection),
        )
        return config


def load_behavior_labels(labels_path: str) -> list[dict]:
    segments = []
    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            segments.append({
                "video": row["Video"],
                "behavior": row["Behavior"],
                "start_frame": int(row["Start_Frame"]),
                "end_frame": int(row["End_Frame"]),
            })
    return segments


def get_clip_id(video: str) -> int | None:
    try:
        return int(video.replace(".mp4", "").replace("clip_", ""))
    except:
        return None


def build_frame_behavior_map(segments: list[dict]) -> dict[int, dict[int, str]]:
    clip_behaviors = defaultdict(dict)
    for seg in segments:
        clip_id = get_clip_id(seg["video"])
        if clip_id is None:
            continue
        behavior = BEHAVIOR_MAPPING.get(seg["behavior"], seg["behavior"])
        for frame in range(seg["start_frame"], min(seg["end_frame"] + 1, 500)):
            clip_behaviors[clip_id][frame] = behavior
    return clip_behaviors


def find_transition_frames(
    clip_behaviors: dict[int, dict[int, str]],
    window: int = 5,
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Find frames near behavior transitions."""
    transitions = defaultdict(list)

    for clip_id, frame_map in clip_behaviors.items():
        frames = sorted(frame_map.keys())

        for i in range(1, len(frames)):
            prev_frame = frames[i-1]
            curr_frame = frames[i]
            prev_beh = frame_map[prev_frame]
            curr_beh = frame_map[curr_frame]

            if prev_beh != curr_beh and prev_beh in BEHAVIORS and curr_beh in BEHAVIORS:
                key = tuple(sorted([prev_beh, curr_beh]))
                for offset in range(-window, window + 1):
                    f = prev_frame + offset
                    if f in frame_map:
                        transitions[key].append((clip_id, f))

    return transitions


def is_consolidated_format(path: str) -> bool:
    return path.endswith('.h5')


def load_all_intentions(data_source: str, clip_ids: list[int]) -> dict[int, np.ndarray]:
    """Load all intentions from either folder or consolidated H5 format."""
    clip_data = {}

    if is_consolidated_format(data_source):
        with h5py.File(data_source, "r") as f:
            intention_data = f["intention"]
            for clip_id in clip_ids:
                if clip_id < intention_data.shape[0]:
                    clip_data[clip_id] = np.array(intention_data[clip_id])
    else:
        for clip_id in clip_ids:
            path = os.path.join(data_source, f"clip_{clip_id}.h5")
            if os.path.exists(path):
                with h5py.File(path, "r") as f:
                    clip_data[clip_id] = np.array(f["intention"])

    return clip_data


def random_subsample(points: np.ndarray, max_n: int) -> np.ndarray:
    if len(points) <= max_n:
        return points
    indices = np.random.choice(len(points), max_n, replace=False)
    return points[indices]


def compute_hull(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
        verts = points[hull.vertices]
        return np.vstack([verts, verts[0]])
    except:
        return None


def compute_density(points: np.ndarray, grid_size: int = 50, sigma: float = 1.2):
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    pad = 0.15

    x_edges = np.linspace(x_min - pad * (x_max - x_min), x_max + pad * (x_max - x_min), grid_size)
    y_edges = np.linspace(y_min - pad * (y_max - y_min), y_max + pad * (y_max - y_min), grid_size)

    H, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[x_edges, y_edges])
    H = gaussian_filter(H.T, sigma=sigma)

    xc = (x_edges[:-1] + x_edges[1:]) / 2
    yc = (y_edges[:-1] + y_edges[1:]) / 2
    return xc, yc, H


def plot_behavior_panel(ax, points: np.ndarray, behavior: str, color: str, density_only: bool = False):
    """Plot a single behavior cluster panel."""
    if len(points) > 20:
        xc, yc, H = compute_density(points)
        ax.contour(xc, yc, H, levels=3, colors=[color], alpha=0.7, linewidths=1.5)

    if not density_only:
        points_sub = random_subsample(points, 300)
        ax.scatter(points_sub[:, 0], points_sub[:, 1], c=color, s=25, alpha=0.7,
                   edgecolors='white', linewidths=0.4)

    centroid = points.mean(axis=0)
    ax.annotate(behavior, centroid, fontsize=11, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9, edgecolor='white'),
                color='white', zorder=10)

    ax.set_title(f"{behavior}\n({len(points)} pts)", fontsize=11, fontweight='bold', color=color)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(2)


def plot_main_panel(
    ax,
    transition_points: dict,
    all_behavior_points: dict,
    transition_colors: dict[tuple[str, str], str],
    config: Config,
):
    """Plot main panel with transition clusters."""
    if not config.density_only:
        for behavior, points in all_behavior_points.items():
            color = BEHAVIOR_COLORS[behavior]
            points_sub = random_subsample(points, 200)
            ax.scatter(points_sub[:, 0], points_sub[:, 1], c=color, s=8, alpha=0.15, edgecolors='none')

    centroids = {}
    for trans_key, points in transition_points.items():
        if len(points) > 0:
            centroids[trans_key] = points.mean(axis=0)

    for trans_key, points in transition_points.items():
        if len(points) == 0:
            continue

        color = transition_colors.get(trans_key, "#888888")
        label = get_transition_label(trans_key)

        if len(points) > 15:
            xc, yc, H = compute_density(points, grid_size=40, sigma=1.0)
            ax.contour(xc, yc, H, levels=3, colors=[color], alpha=0.7, linewidths=1.5)

        if not config.density_only:
            points_sub = random_subsample(points, config.max_transition_points)
            ax.scatter(points_sub[:, 0], points_sub[:, 1], c=color, s=50, alpha=0.8,
                       edgecolors='white', linewidths=1, label=f"{label} ({len(points)})", zorder=5)

        # Centroid label with dynamic offset to avoid overlap
        centroid = points.mean(axis=0)
        short_label = f"{trans_key[0][:3]}↔{trans_key[1][:3]}"

        # Compute offset: push label away from other centroids
        offset = np.array([0.0, 0.0])
        for other_key, other_cent in centroids.items():
            if other_key != trans_key:
                diff = centroid - other_cent
                dist = np.linalg.norm(diff)
                if dist < 3.0 and dist > 0.01:
                    offset += (diff / dist) * 1.2

        # Ensure minimum offset with random direction fallback
        if np.linalg.norm(offset) < 0.5:
            angle = hash(trans_key) % 360 * np.pi / 180
            offset = np.array([np.cos(angle), np.sin(angle)]) * 1.2

        label_pos = centroid + offset

        ax.annotate(short_label, label_pos, fontsize=10, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.95, edgecolor='white', linewidth=2),
                    color='white', zorder=10)

        # Draw line from label to centroid if offset is large
        if np.linalg.norm(offset) > 0.8:
            ax.plot([centroid[0], label_pos[0]], [centroid[1], label_pos[1]],
                    color=color, linewidth=1.5, alpha=0.6, zorder=9)

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title("Transition Clusters", fontsize=14, fontweight='bold')
    if config.density_only:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=transition_colors.get(k, "#888888"), alpha=0.6,
                  label=f"{get_transition_label(k)} ({len(v)})")
            for k, v in transition_points.items() if len(v) > 0
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)
    else:
        ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)


def plot_density_panel(ax, all_behavior_points: dict, density_only: bool = False):
    """Plot behavior clusters without transitions - clean view."""
    centroids = {b: pts.mean(axis=0) for b, pts in all_behavior_points.items() if len(pts) > 0}

    for behavior, points in all_behavior_points.items():
        color = BEHAVIOR_COLORS[behavior]

        if len(points) > 20:
            xc, yc, H = compute_density(points, grid_size=50, sigma=1.2)
            ax.contour(xc, yc, H, levels=4, colors=[color], alpha=0.8, linewidths=1.5)

        if not density_only:
            points_sub = random_subsample(points, 300)
            ax.scatter(points_sub[:, 0], points_sub[:, 1], c=color, s=20, alpha=0.6,
                       edgecolors='white', linewidths=0.3)

    # Add labels with offset to avoid overlap
    for behavior, points in all_behavior_points.items():
        if len(points) == 0:
            continue
        color = BEHAVIOR_COLORS[behavior]
        centroid = centroids[behavior]

        # Compute offset: push label away from other centroids
        offset = np.array([0.0, 0.0])
        for other_beh, other_cent in centroids.items():
            if other_beh != behavior:
                diff = centroid - other_cent
                dist = np.linalg.norm(diff)
                if dist < 2.0 and dist > 0.01:
                    offset += (diff / dist) * 0.3

        label_pos = centroid + offset
        ax.annotate(behavior, label_pos, fontsize=10, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.9, edgecolor='white', linewidth=1.5),
                    color='white', zorder=10)

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title("Behavior Clusters", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=BEHAVIOR_COLORS[b], alpha=0.6, label=b) for b in BEHAVIORS]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)


def main():
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "configs" / "scripts" / "demo_behavior_overlap.yaml"
    config = Config.from_yaml(config_path, base_dir)
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)

    print(f"Loading data from: {config.data_source}")
    segments = load_behavior_labels(config.labels_path)
    clip_behaviors = build_frame_behavior_map(segments)

    clips_to_load = sorted(clip_behaviors.keys())[:config.n_clips]
    clip_data = load_all_intentions(config.data_source, clips_to_load)
    print(f"Loaded {len(clip_data)} clips")

    # Find transition frames and get top K
    print("Finding transitions...")
    all_trans_frames = find_transition_frames(clip_behaviors, config.transition_window)

    # Sort by count and take top K
    sorted_transitions = sorted(all_trans_frames.items(), key=lambda x: len(x[1]), reverse=True)
    top_k = sorted_transitions[:config.top_k_transitions]
    trans_frames = dict(top_k)

    print(f"Top {config.top_k_transitions} transitions:")
    transition_colors = {}
    for idx, (key, frames) in enumerate(top_k):
        transition_colors[key] = get_transition_color(idx)
        print(f"  {idx+1}. {key[0]} <-> {key[1]}: {len(frames)} frames")

    # Collect all points by behavior
    behavior_intentions = {b: [] for b in BEHAVIORS}
    for clip_id, intentions in clip_data.items():
        frame_map = clip_behaviors.get(clip_id, {})
        for frame_idx in range(len(intentions)):
            beh = frame_map.get(frame_idx, "Unknown")
            if beh in BEHAVIORS:
                behavior_intentions[beh].append(intentions[frame_idx])

    for b in BEHAVIORS:
        behavior_intentions[b] = np.array(behavior_intentions[b])
        print(f"  {b}: {len(behavior_intentions[b])} points")

    # Collect transition points
    trans_intentions = {k: [] for k in trans_frames.keys()}
    for trans_key, frames in trans_frames.items():
        for clip_id, frame_idx in frames:
            if clip_id in clip_data and frame_idx < len(clip_data[clip_id]):
                trans_intentions[trans_key].append(clip_data[clip_id][frame_idx])
        trans_intentions[trans_key] = np.array(trans_intentions[trans_key]) if trans_intentions[trans_key] else np.array([])

    all_points = np.vstack([behavior_intentions[b] for b in BEHAVIORS])
    all_trans_points = [v for v in trans_intentions.values() if len(v) > 0]
    combined_points = np.vstack([all_points] + all_trans_points) if all_trans_points else all_points

    print(f"\nFitting {config.projection.upper()} on {len(combined_points)} points...")
    if config.projection == "pca":
        projector = PCA(n_components=2)
        projector.fit(combined_points)
        var_explained = projector.explained_variance_ratio_ * 100
        print(f"Explained variance: {var_explained}")
        axis_labels = (f"PC1 ({var_explained[0]:.1f}%)", f"PC2 ({var_explained[1]:.1f}%)")
        transform_fn = projector.transform
    elif config.projection == "umap":
        projector = umap.UMAP(n_components=2, random_state=42)
        all_embedded = projector.fit_transform(combined_points)
        axis_labels = ("UMAP1", "UMAP2")
        transform_fn = projector.transform
    elif config.projection == "tsne":
        projector = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_points) - 1))
        all_embedded = projector.fit_transform(combined_points)
        axis_labels = ("t-SNE1", "t-SNE2")
        transform_fn = None
    else:
        raise ValueError(f"Unknown projection method: {config.projection}")

    if config.projection == "tsne":
        idx = 0
        behavior_projected = {}
        for b in BEHAVIORS:
            n = len(behavior_intentions[b])
            behavior_projected[b] = all_embedded[idx:idx+n]
            idx += n
        trans_projected = {}
        for k, v in trans_intentions.items():
            if len(v) > 0:
                n = len(v)
                trans_projected[k] = all_embedded[idx:idx+n]
                idx += n
            else:
                trans_projected[k] = np.array([])
    else:
        behavior_projected = {b: transform_fn(behavior_intentions[b]) for b in BEHAVIORS}
        trans_projected = {}
        for k, v in trans_intentions.items():
            if len(v) > 0:
                trans_projected[k] = transform_fn(v)
            else:
                trans_projected[k] = np.array([])

    n_behaviors = len(BEHAVIORS)

    if config.show_side_panels:
        fig = plt.figure(figsize=(26, max(10, n_behaviors * 2)))
        gs = GridSpec(n_behaviors, 3, figure=fig, width_ratios=[1, 2.5, 2.5],
                      height_ratios=[1] * n_behaviors, wspace=0.18, hspace=0.15)
        behavior_axes = {}
        for i, behavior in enumerate(BEHAVIORS):
            ax = fig.add_subplot(gs[i, 0])
            plot_behavior_panel(ax, behavior_projected[behavior], behavior, BEHAVIOR_COLORS[behavior], config.density_only)
            behavior_axes[behavior] = ax
        ax_main = fig.add_subplot(gs[:, 1])
        ax_density = fig.add_subplot(gs[:, 2])
    else:
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.15)
        behavior_axes = {}
        ax_main = fig.add_subplot(gs[0, 0])
        ax_density = fig.add_subplot(gs[0, 1])

    plot_main_panel(ax_main, trans_projected, behavior_projected, transition_colors, config)
    plot_density_panel(ax_density, behavior_projected, config.density_only)

    # Synchronize main and density panel limits
    x_min = min(ax_main.get_xlim()[0], ax_density.get_xlim()[0])
    x_max = max(ax_main.get_xlim()[1], ax_density.get_xlim()[1])
    y_min = min(ax_main.get_ylim()[0], ax_density.get_ylim()[0])
    y_max = max(ax_main.get_ylim()[1], ax_density.get_ylim()[1])
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    ax_density.set_xlim(x_min, x_max)
    ax_density.set_ylim(y_min, y_max)

    ax_main.set_xlabel(axis_labels[0], fontsize=12)
    ax_main.set_ylabel(axis_labels[1], fontsize=12)
    ax_density.set_xlabel(axis_labels[0], fontsize=12)
    ax_density.set_ylabel(axis_labels[1], fontsize=12)

    if config.show_side_panels:
        for behavior, ax_small in behavior_axes.items():
            color = BEHAVIOR_COLORS[behavior]
            centroid = behavior_projected[behavior].mean(axis=0)
            arrow = ConnectionPatch(
                xyA=(1.0, 0.5), coordsA=ax_small.transAxes,
                xyB=(centroid[0], centroid[1]), coordsB=ax_main.transData,
                arrowstyle="-|>", mutation_scale=15,
                color=color, linewidth=2, alpha=0.7
            )
            fig.add_artist(arrow)

    fig.suptitle(f"Behavioral Clusters in Latent Space ({config.projection.upper()})", fontsize=18, fontweight='bold', y=0.98)

    plt.savefig(config.output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {config.output_path}")
    plt.show()


if __name__ == "__main__":
    main()
