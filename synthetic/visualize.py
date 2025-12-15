"""
Visualization for synthetic behavioral data.
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from dataclasses import dataclass

from synthetic.generate import (
    SyntheticConfig,
    SyntheticDataset,
    generate_synthetic_dataset,
    save_synthetic_dataset,
    load_synthetic_dataset,
)


@dataclass
class VisualizeConfig:
    data_dir: str = "data/synthetic"
    output_dir: str = "out_synthetic"
    figsize: tuple[int, int] = (15, 5)
    dpi: int = 150
    alpha: float = 0.6
    point_size: int = 15
    behavior_colors: dict = None
    subcluster_markers: list[str] = None
    n_trajectory_clips: int = 5
    line_alpha: float = 0.3
    line_width: float = 1.0
    show_hulls: bool = True
    hull_alpha: float = 0.2
    pca_random_state: int = 42

    def __post_init__(self):
        if self.behavior_colors is None:
            self.behavior_colors = {
                "walk": "#2A9D8F",
                "rear": "#E63946",
                "groom": "#E9C46A",
            }
        if self.subcluster_markers is None:
            self.subcluster_markers = ["o", "s", "^", "D", "v"]


def load_config(config_path: str) -> tuple[SyntheticConfig, VisualizeConfig]:
    """Load combined config from YAML, returns both generation and visualization configs."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    gen_cfg = SyntheticConfig(
        n_clips=cfg.get("n_clips", 500),
        frames_per_clip=cfg.get("frames_per_clip", 500),
        latent_dim=cfg.get("latent_dim", 16),
        behaviors=cfg.get("behaviors", ["walk", "rear", "groom"]),
        subclusters_per_behavior=cfg.get("subclusters_per_behavior", 3),
        behavior_separation=cfg.get("behavior_separation", 8.0),
        subcluster_separation=cfg.get("subcluster_separation", 2.0),
        cluster_std=cfg.get("cluster_std", 0.5),
        dwell_time_mean=cfg.get("dwell_time_mean", 50),
        dwell_time_std=cfg.get("dwell_time_std", 20),
        dwell_time_min=cfg.get("dwell_time_min", 10),
        transition_smoothness=cfg.get("transition_smoothness", 10),
        intra_behavior_transition_prob=cfg.get("intra_behavior_transition_prob", 0.7),
        inter_behavior_transition_prob=cfg.get("inter_behavior_transition_prob", 0.3),
        noise_scale=cfg.get("noise_scale", 0.1),
        global_noise_scale=cfg.get("global_noise_scale", 0.0),
        random_seed=cfg.get("random_seed", 42),
        output_dir=cfg.get("data_dir", "data/synthetic"),
    )

    viz_cfg = VisualizeConfig(
        data_dir=cfg.get("data_dir", "data/synthetic"),
        output_dir=cfg.get("output_dir", "out_synthetic"),
        figsize=tuple(cfg.get("figsize", [15, 5])),
        dpi=cfg.get("dpi", 150),
        alpha=cfg.get("alpha", 0.6),
        point_size=cfg.get("point_size", 15),
        behavior_colors=cfg.get("behavior_colors"),
        subcluster_markers=cfg.get("subcluster_markers"),
        n_trajectory_clips=cfg.get("trajectory", {}).get("n_clips", 5),
        line_alpha=cfg.get("trajectory", {}).get("line_alpha", 0.3),
        line_width=cfg.get("trajectory", {}).get("line_width", 1.0),
        show_hulls=cfg.get("show_hulls", True),
        hull_alpha=cfg.get("hull_alpha", 0.2),
        pca_random_state=cfg.get("pca", {}).get("random_state", 42),
    )

    return gen_cfg, viz_cfg


def get_color_for_behavior(behavior: str, viz_cfg: VisualizeConfig) -> str:
    """Get color for a behavior, with fallback."""
    default_colors = ["#2A9D8F", "#E63946", "#E9C46A", "#264653", "#F4A261"]
    if behavior in viz_cfg.behavior_colors:
        return viz_cfg.behavior_colors[behavior]
    idx = hash(behavior) % len(default_colors)
    return default_colors[idx]


def plot_behavior_distribution(
    dataset: SyntheticDataset,
    pca_2d: np.ndarray,
    viz_cfg: VisualizeConfig,
    ax: plt.Axes,
) -> None:
    """Plot points colored by behavior with convex hulls."""
    behaviors = dataset.config.behaviors
    behavior_labels = dataset.all_behavior_labels

    for b_idx, behavior in enumerate(behaviors):
        mask = behavior_labels == b_idx
        pts = pca_2d[mask]
        color = get_color_for_behavior(behavior, viz_cfg)

        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=color, alpha=viz_cfg.alpha, s=viz_cfg.point_size,
            label=behavior.capitalize(), edgecolors='none'
        )

        if viz_cfg.show_hulls and len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                ax.fill(
                    hull_pts[:, 0], hull_pts[:, 1],
                    color=color, alpha=viz_cfg.hull_alpha,
                    edgecolor=color, linewidth=2
                )
            except:
                pass

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title("Distribution by Behavior", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    """Convert RGB tuple (0-1 range) to hex color."""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))


def get_subcluster_shades(base_color: str, n_shades: int) -> list[str]:
    """Generate shades of a base color for sub-clusters (light to dark)."""
    rgb = hex_to_rgb(base_color)
    shades = []

    lightness_range = [0.4, 0.7, 1.0][:n_shades] if n_shades <= 3 else \
                      [0.4 + 0.6 * i / (n_shades - 1) for i in range(n_shades)]

    for lightness in lightness_range:
        shade = tuple(max(0, min(1, c * lightness + (1 - lightness) * 0.9)) for c in rgb)
        shades.append(rgb_to_hex(shade))

    return shades


def plot_subcluster_distribution(
    dataset: SyntheticDataset,
    pca_2d: np.ndarray,
    viz_cfg: VisualizeConfig,
    ax: plt.Axes,
) -> None:
    """Plot sub-cluster coverage (hulls only, no scatter points)."""
    behaviors = dataset.config.behaviors
    behavior_labels = dataset.all_behavior_labels
    subcluster_labels = np.concatenate([c.subcluster_labels for c in dataset.clips])
    n_subclusters = dataset.config.subclusters_per_behavior

    handles = []

    for b_idx, behavior in enumerate(behaviors):
        base_color = get_color_for_behavior(behavior, viz_cfg)
        shades = get_subcluster_shades(base_color, n_subclusters)

        for sub_idx in range(n_subclusters):
            mask = (behavior_labels == b_idx) & (subcluster_labels == sub_idx)
            pts = pca_2d[mask]

            if len(pts) < 3:
                continue

            shade_color = shades[sub_idx]

            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                ax.fill(
                    hull_pts[:, 0], hull_pts[:, 1],
                    color=shade_color, alpha=0.4,
                    edgecolor=shade_color, linewidth=2, zorder=1
                )
            except:
                pass

            handles.append(mpatches.Patch(
                color=shade_color, alpha=0.6,
                label=f"{behavior.capitalize()}_{sub_idx}"
            ))

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title("Sub-cluster Coverage", fontsize=14, fontweight="bold")
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3, linestyle="--")


def plot_sample_trajectories(
    dataset: SyntheticDataset,
    pca: PCA,
    viz_cfg: VisualizeConfig,
    ax: plt.Axes,
) -> None:
    """Plot sample clip trajectories through the space."""
    behaviors = dataset.config.behaviors
    n_clips = min(viz_cfg.n_trajectory_clips, len(dataset.clips))

    rng = np.random.default_rng(42)
    clip_indices = rng.choice(len(dataset.clips), n_clips, replace=False)

    all_latents_2d = pca.transform(dataset.all_latents)
    ax.scatter(
        all_latents_2d[:, 0], all_latents_2d[:, 1],
        c='lightgray', alpha=0.2, s=5, edgecolors='none'
    )

    trajectory_colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_clips))

    for i, clip_idx in enumerate(clip_indices):
        clip = dataset.clips[clip_idx]
        clip_2d = pca.transform(clip.latents)

        points = clip_2d.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            colors=[trajectory_colors[i]],
            alpha=viz_cfg.line_alpha,
            linewidths=viz_cfg.line_width
        )
        ax.add_collection(lc)

        ax.scatter(clip_2d[0, 0], clip_2d[0, 1], c='green', s=50, marker='o', zorder=5)
        ax.scatter(clip_2d[-1, 0], clip_2d[-1, 1], c='red', s=50, marker='x', zorder=5)

    ax.scatter([], [], c='green', s=50, marker='o', label='Start')
    ax.scatter([], [], c='red', s=50, marker='x', label='End')

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title(f"Sample Trajectories (n={n_clips})", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")


def plot_cluster_centers(
    dataset: SyntheticDataset,
    pca: PCA,
    viz_cfg: VisualizeConfig,
    ax: plt.Axes,
) -> None:
    """Plot cluster centers with labels."""
    centers = np.array([c.center for c in dataset.cluster_specs])
    centers_2d = pca.transform(centers)

    for i, cluster in enumerate(dataset.cluster_specs):
        color = get_color_for_behavior(cluster.behavior, viz_cfg)
        ax.scatter(
            centers_2d[i, 0], centers_2d[i, 1],
            c=color, s=200, marker='*', edgecolors='black', linewidths=1.5, zorder=10
        )
        ax.annotate(
            cluster.name, (centers_2d[i, 0], centers_2d[i, 1]),
            xytext=(5, 5), textcoords='offset points', fontsize=8
        )

    all_latents_2d = pca.transform(dataset.all_latents)
    ax.scatter(
        all_latents_2d[:, 0], all_latents_2d[:, 1],
        c='lightgray', alpha=0.3, s=5, edgecolors='none', zorder=1
    )

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title("Cluster Centers", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")


def create_visualization(
    dataset: SyntheticDataset,
    viz_cfg: VisualizeConfig,
) -> None:
    """Create and save all visualizations."""
    os.makedirs(viz_cfg.output_dir, exist_ok=True)

    pca = PCA(n_components=2, random_state=viz_cfg.pca_random_state)
    all_latents_2d = pca.fit_transform(dataset.all_latents)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    fig, axes = plt.subplots(1, 3, figsize=viz_cfg.figsize)

    plot_behavior_distribution(dataset, all_latents_2d, viz_cfg, axes[0])
    plot_subcluster_distribution(dataset, all_latents_2d, viz_cfg, axes[1])
    plot_sample_trajectories(dataset, pca, viz_cfg, axes[2])

    plt.tight_layout()

    save_path = os.path.join(viz_cfg.output_dir, "synthetic_distribution.pdf")
    plt.savefig(save_path, dpi=viz_cfg.dpi, bbox_inches='tight')
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=viz_cfg.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plot to {save_path}")

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_cluster_centers(dataset, pca, viz_cfg, ax)
    plt.tight_layout()

    save_path = os.path.join(viz_cfg.output_dir, "cluster_centers.pdf")
    plt.savefig(save_path, dpi=viz_cfg.dpi, bbox_inches='tight')
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=viz_cfg.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster centers plot to {save_path}")


def main(config_path: str = "configs/synthetic/synthetic.yaml"):
    """Generate synthetic data and create visualizations."""
    gen_cfg, viz_cfg = load_config(config_path)

    print("=" * 60)
    print("Synthetic Data Generation and Visualization")
    print("=" * 60)

    print("\n=== Stage 1: Generating synthetic dataset ===")
    dataset = generate_synthetic_dataset(gen_cfg)
    save_synthetic_dataset(dataset, gen_cfg.output_dir)

    print("\n=== Stage 2: Creating visualizations ===")
    create_visualization(dataset, viz_cfg)

    print("\n=== Stage 3: Dataset statistics ===")
    print(f"Total clips: {len(dataset.clips)}")
    print(f"Total frames: {len(dataset.all_latents)}")
    print(f"Latent dim: {gen_cfg.latent_dim}")
    print(f"Behaviors: {gen_cfg.behaviors}")
    print(f"Clusters per behavior: {gen_cfg.subclusters_per_behavior}")
    print(f"Total clusters: {len(dataset.cluster_specs)}")

    behavior_counts = np.bincount(dataset.all_behavior_labels)
    for b_idx, behavior in enumerate(gen_cfg.behaviors):
        print(f"  {behavior}: {behavior_counts[b_idx]} frames ({100*behavior_counts[b_idx]/len(dataset.all_behavior_labels):.1f}%)")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/synthetic/synthetic.yaml"
    main(config)
