# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "h5py", "matplotlib", "scikit-learn", "scipy"]
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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, ConnectionPatch
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA


np.random.seed(42)

BEHAVIORS = ["Walking", "Immobility", "Orienting"]

BEHAVIOR_COLORS = {
    "Walking": "#4A90D9",
    "Immobility": "#7BC47F",
    "Orienting": "#F5A623",
}

TRANSITION_COLORS = {
    ("Immobility", "Walking"): "#E74C3C",
    ("Orienting", "Walking"): "#9B59B6",
    ("Immobility", "Orienting"): "#1ABC9C",
}

TRANSITION_LABELS = {
    ("Immobility", "Walking"): "Immobility ↔ Walking",
    ("Orienting", "Walking"): "Orienting ↔ Walking",
    ("Immobility", "Orienting"): "Immobility ↔ Orienting",
}


@dataclass
class Config:
    data_dir: str = "data/rodent_ar"
    labels_path: str = "behavior_labels.csv"
    output_path: str = "out/behavior_clusters.png"
    n_clips: int = 200
    max_points_per_behavior: int = 1000
    max_transition_points: int = 500
    transition_window: int = 5
    figsize: tuple[int, int] = (16, 10)


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
        for frame in range(seg["start_frame"], min(seg["end_frame"] + 1, 500)):
            clip_behaviors[clip_id][frame] = seg["behavior"]
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


def load_all_intentions(data_dir: str, clip_ids: list[int]) -> dict[int, np.ndarray]:
    """Load all intentions without subsampling."""
    clip_data = {}
    for clip_id in clip_ids:
        path = os.path.join(data_dir, f"clip_{clip_id}.h5")
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


def plot_behavior_panel(ax, points: np.ndarray, behavior: str, color: str):
    """Plot a single behavior cluster panel."""
    points_sub = random_subsample(points, 300)

    # Density contour
    if len(points) > 20:
        xc, yc, H = compute_density(points)
        ax.contourf(xc, yc, H, levels=6, colors=[color], alpha=0.3)
        ax.contour(xc, yc, H, levels=3, colors=[color], alpha=0.7, linewidths=1.5)

    # Points
    ax.scatter(points_sub[:, 0], points_sub[:, 1], c=color, s=25, alpha=0.7,
               edgecolors='white', linewidths=0.4)

    # Label
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


def plot_main_panel(ax, transition_points: dict, all_behavior_points: dict, config: Config):
    """Plot main panel with transition clusters."""

    # First draw faded behavior backgrounds
    for behavior, points in all_behavior_points.items():
        color = BEHAVIOR_COLORS[behavior]
        points_sub = random_subsample(points, 200)
        ax.scatter(points_sub[:, 0], points_sub[:, 1], c=color, s=8, alpha=0.15, edgecolors='none')

    # Compute centroids first for label placement
    centroids = {}
    for trans_key, points in transition_points.items():
        if len(points) > 0:
            centroids[trans_key] = points.mean(axis=0)

    # Draw transition clusters prominently
    for trans_key, points in transition_points.items():
        if len(points) == 0:
            continue

        color = TRANSITION_COLORS.get(trans_key, "#888888")
        label = TRANSITION_LABELS.get(trans_key, f"{trans_key[0]} ↔ {trans_key[1]}")

        points_sub = random_subsample(points, config.max_transition_points)

        # Density contour for transition cluster
        if len(points) > 15:
            xc, yc, H = compute_density(points, grid_size=40, sigma=1.0)
            ax.contourf(xc, yc, H, levels=5, colors=[color], alpha=0.25)
            ax.contour(xc, yc, H, levels=3, colors=[color], alpha=0.7, linewidths=1.5)

        # Points
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

        # Ensure minimum offset
        if np.linalg.norm(offset) < 0.5:
            # Use fixed offsets as fallback
            fallback = {
                ("Immobility", "Walking"): np.array([0, 1.2]),
                ("Orienting", "Walking"): np.array([0, -1.2]),
                ("Immobility", "Orienting"): np.array([1.2, 0]),
            }
            offset = fallback.get(trans_key, np.array([0, 0]))

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
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)


def plot_density_panel(ax, all_behavior_points: dict):
    """Plot behavior clusters without transitions - clean view."""

    # Plot each behavior with contour lines only (no fill/shading)
    for behavior, points in all_behavior_points.items():
        color = BEHAVIOR_COLORS[behavior]

        # Contour lines only
        if len(points) > 20:
            xc, yc, H = compute_density(points, grid_size=50, sigma=1.2)
            ax.contour(xc, yc, H, levels=4, colors=[color], alpha=0.8, linewidths=1.5)

        # Scatter points
        points_sub = random_subsample(points, 300)
        ax.scatter(points_sub[:, 0], points_sub[:, 1], c=color, s=20, alpha=0.6,
                   edgecolors='white', linewidths=0.3)

        # Label at centroid
        centroid = points.mean(axis=0)
        ax.annotate(behavior, centroid, fontsize=11, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9, edgecolor='white', linewidth=2),
                    color='white', zorder=10)

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title("Behavior Clusters", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=BEHAVIOR_COLORS[b], alpha=0.6, label=b) for b in BEHAVIORS]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)


def main():
    config = Config()
    # Go up to project root (scripts/additional -> scripts -> TopoVNL)
    base_dir = Path(__file__).parent.parent.parent
    config.data_dir = str(base_dir / "data" / "rodent_ar")
    config.labels_path = str(base_dir / "behavior_labels.csv")
    config.output_path = str(base_dir / "out" / "behavior_clusters.png")
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)

    print("Loading data...")
    segments = load_behavior_labels(config.labels_path)
    clip_behaviors = build_frame_behavior_map(segments)

    clips_to_load = sorted(clip_behaviors.keys())[:config.n_clips]
    clip_data = load_all_intentions(config.data_dir, clips_to_load)
    print(f"Loaded {len(clip_data)} clips")

    # Find transition frames
    print("Finding transitions...")
    trans_frames = find_transition_frames(clip_behaviors, config.transition_window)
    for key, frames in trans_frames.items():
        print(f"  {key[0]} <-> {key[1]}: {len(frames)} frames")

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

    # Combine all for PCA
    all_points = []
    for b in BEHAVIORS:
        all_points.append(behavior_intentions[b])
    all_points = np.vstack(all_points)

    print(f"\nFitting PCA on {len(all_points)} points...")
    pca = PCA(n_components=2)
    pca.fit(all_points)
    print(f"Explained variance: {pca.explained_variance_ratio_}")

    # Project all
    behavior_projected = {b: pca.transform(behavior_intentions[b]) for b in BEHAVIORS}
    trans_projected = {}
    for k, v in trans_intentions.items():
        if len(v) > 0:
            trans_projected[k] = pca.transform(v)
        else:
            trans_projected[k] = np.array([])

    # Create figure: [small panels] [main transition] [density]
    fig = plt.figure(figsize=(26, 10))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1, 2.5, 2.5], height_ratios=[1, 1, 1],
                  wspace=0.18, hspace=0.15)

    # Store variance for axis labels
    var_explained = pca.explained_variance_ratio_ * 100

    # Left column: Small behavior panels (stacked vertically)
    behavior_axes = {}
    for i, behavior in enumerate(BEHAVIORS):
        ax = fig.add_subplot(gs[i, 0])
        plot_behavior_panel(ax, behavior_projected[behavior], behavior, BEHAVIOR_COLORS[behavior])
        behavior_axes[behavior] = ax

    # Middle: Main transition panel
    ax_main = fig.add_subplot(gs[:, 1])
    plot_main_panel(ax_main, trans_projected, behavior_projected, config)

    # Right: Behavior density panel
    ax_density = fig.add_subplot(gs[:, 2])
    plot_density_panel(ax_density, behavior_projected)

    # Synchronize main and density panel limits
    x_min = min(ax_main.get_xlim()[0], ax_density.get_xlim()[0])
    x_max = max(ax_main.get_xlim()[1], ax_density.get_xlim()[1])
    y_min = min(ax_main.get_ylim()[0], ax_density.get_ylim()[0])
    y_max = max(ax_main.get_ylim()[1], ax_density.get_ylim()[1])
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    ax_density.set_xlim(x_min, x_max)
    ax_density.set_ylim(y_min, y_max)

    # Update axis labels with variance explained
    ax_main.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)", fontsize=12)
    ax_main.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)", fontsize=12)
    ax_density.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)", fontsize=12)
    ax_density.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)", fontsize=12)

    # Draw arrows from small panels to behavior centroids in main panel
    for behavior, ax_small in behavior_axes.items():
        color = BEHAVIOR_COLORS[behavior]
        centroid = behavior_projected[behavior].mean(axis=0)
        # Arrow from right edge of small panel to centroid location in main panel
        arrow = ConnectionPatch(
            xyA=(1.0, 0.5), coordsA=ax_small.transAxes,
            xyB=(centroid[0], centroid[1]), coordsB=ax_main.transData,
            arrowstyle="-|>", mutation_scale=15,
            color=color, linewidth=2, alpha=0.7
        )
        fig.add_artist(arrow)

    fig.suptitle("Behavioral Clusters in Latent Space", fontsize=18, fontweight='bold', y=0.98)

    plt.savefig(config.output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {config.output_path}")
    plt.show()


if __name__ == "__main__":
    main()
