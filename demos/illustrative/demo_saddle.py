#!/usr/bin/env python3
"""
Density Basin Discrete Morse Demonstration
============================================
Demonstrates DM cycle vs DM path detection on an inverse-density surface
where dense regions form basins (low z).

Three experiments:
1. Two basins connected by two distinct routes -> cycle (theta-graph)
2. Two basins connected by only one route -> path only (no cycle)
3. Multi-basin landscape: some pairs have two routes (cycle), others only one (path)

Usage:
    python demos/illustrative/demo_saddle.py
    python demos/illustrative/demo_saddle.py --config configs/demos/illustrative/demo_saddle.yaml
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
import networkx as nx
from pathlib import Path
from dataclasses import dataclass, field
import yaml

sys.path.insert(0, "/home/jovyan/vast/kaiwen/TopoVNL")
sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")

from topo_mimic.dmwk.skeletonization.skeletonization import extract_skeleton_pcd_dm


@dataclass
class MorseConfig:
    k_neighbors: int = 5
    pers_delta: float = 0.3
    epsilon: float = 0.99


@dataclass
class PathConfig:
    n_steps: int = 600
    noise_std: float = 0.03
    seed: int = 42


@dataclass
class VisualizationConfig:
    dpi: int = 300
    surface_alpha: float = 0.25
    contour_levels: int = 15
    contour_alpha: float = 0.35
    path_linewidth: float = 1.5
    density_cmap: str = "viridis"
    cycle_color: str = "#d62728"
    cycle_linewidth: float = 3.0
    path_color: str = "#1f77b4"
    path_marker_color: str = "#1f77b4"


@dataclass
class Config:
    output_dir: str = "outputs/saddle_dm_demo"
    path: PathConfig = field(default_factory=PathConfig)
    dm: MorseConfig = field(default_factory=MorseConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    path_d = cfg_dict.get("path", {})
    path_cfg = PathConfig(**{k: v for k, v in path_d.items() if k in PathConfig.__dataclass_fields__})

    dm_d = cfg_dict.get("dm", {})
    dm_cfg = MorseConfig(**{k: v for k, v in dm_d.items() if k in MorseConfig.__dataclass_fields__})

    vis_d = cfg_dict.get("visualization", {})
    vis_cfg = VisualizationConfig(**{k: v for k, v in vis_d.items() if k in VisualizationConfig.__dataclass_fields__})

    return Config(
        output_dir=cfg_dict.get("output_dir", "outputs/saddle_dm_demo"),
        path=path_cfg,
        dm=dm_cfg,
        visualization=vis_cfg,
    )


# ── Surface generation ──────────────────────────────────────────────────────


def density_basin_surface_2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Two basins (dense regions sink down) on a flat base.

    Basin A at (-1.5, 0), Basin B at (1.5, 0). Between them a ridge
    with two low passes (at y~+0.8 and y~-0.8), creating two distinct routes.
    """
    base = np.zeros_like(x)
    basin_a = -1.2 * np.exp(-((x + 1.5)**2 + y**2) / 0.5)
    basin_b = -1.2 * np.exp(-((x - 1.5)**2 + y**2) / 0.5)
    ridge = 0.6 * np.exp(-(x**2) / 0.4) * np.exp(-(y**2) / 3.0)
    pass_upper = -0.35 * np.exp(-(x**2) / 0.6) * np.exp(-((y - 0.9)**2) / 0.3)
    pass_lower = -0.35 * np.exp(-(x**2) / 0.6) * np.exp(-((y + 0.9)**2) / 0.3)
    return base + basin_a + basin_b + ridge + pass_upper + pass_lower


def density_basin_surface_1pass(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Two basins connected by only ONE low pass (at y~+0.8). The other direction
    is blocked by a high wall, so only one route exists -> path only."""
    base = np.zeros_like(x)
    basin_a = -1.2 * np.exp(-((x + 1.5)**2 + y**2) / 0.5)
    basin_b = -1.2 * np.exp(-((x - 1.5)**2 + y**2) / 0.5)
    ridge = 0.8 * np.exp(-(x**2) / 0.4) * np.exp(-(y**2) / 3.0)
    pass_upper = -0.5 * np.exp(-(x**2) / 0.6) * np.exp(-((y - 0.9)**2) / 0.3)
    wall_lower = 0.4 * np.exp(-(x**2) / 0.5) * np.exp(-((y + 0.7)**2) / 0.4)
    return base + basin_a + basin_b + ridge + pass_upper + wall_lower


def density_basin_surface_multi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Three basins: A(-2,0), B(1,1.5), C(1,-1.5).
    A-B: two passes (cycle). A-C: one pass (path only)."""
    base = np.zeros_like(x)
    ba = -1.2 * np.exp(-((x + 2)**2 + y**2) / 0.5)
    bb = -1.2 * np.exp(-((x - 1)**2 + (y - 1.5)**2) / 0.5)
    bc = -1.2 * np.exp(-((x - 1)**2 + (y + 1.5)**2) / 0.5)
    ridge_ab = 0.5 * np.exp(-((x + 0.5)**2) / 0.5) * np.exp(-((y - 0.75)**2) / 1.5)
    pass_ab_1 = -0.35 * np.exp(-((x + 0.3)**2 + (y - 1.4)**2) / 0.4)
    pass_ab_2 = -0.35 * np.exp(-((x + 0.3)**2 + (y - 0.1)**2) / 0.4)
    ridge_ac = 0.6 * np.exp(-((x + 0.5)**2) / 0.5) * np.exp(-((y + 0.75)**2) / 1.5)
    pass_ac = -0.4 * np.exp(-((x + 0.3)**2 + (y + 0.8)**2) / 0.4)
    wall_ac_lower = 0.4 * np.exp(-((x + 0.3)**2 + (y + 1.6)**2) / 0.4)
    return base + ba + bb + bc + ridge_ab + pass_ab_1 + pass_ab_2 + ridge_ac + pass_ac + wall_ac_lower


# ── Path generation ─────────────────────────────────────────────────────────


def generate_round_trip_2routes(
    min1: np.ndarray,
    min2: np.ndarray,
    surface_fn,
    n_steps: int,
    noise_std: float,
    rng: np.random.Generator,
    y_offset: float = 0.8,
    blob_fraction: float = 0.0,
) -> np.ndarray:
    """Go from min1 to min2 via upper pass, return via lower pass -> loop.

    blob_fraction: fraction of n_steps to allocate as dense blobs at each
    minimum (0 = pure arc, 0.25 = 25% of points as blobs at each end).
    """
    n_blob = int(n_steps * blob_fraction)
    n_arc = n_steps - 2 * n_blob
    n_half = n_arc // 2

    t_out = np.linspace(0, 1, n_half)
    x_out = min1[0] + (min2[0] - min1[0]) * t_out
    y_out = min1[1] + (min2[1] - min1[1]) * t_out + y_offset * np.sin(np.pi * t_out)

    t_back = np.linspace(0, 1, n_arc - n_half)
    x_back = min2[0] + (min1[0] - min2[0]) * t_back
    y_back = min2[1] + (min1[1] - min2[1]) * t_back - y_offset * np.sin(np.pi * t_back)

    parts_x = [x_out, x_back]
    parts_y = [y_out, y_back]

    if n_blob > 0:
        parts_x.insert(0, min1[0] + rng.normal(0, 0.15, n_blob))
        parts_y.insert(0, min1[1] + rng.normal(0, 0.15, n_blob))
        parts_x.append(min2[0] + rng.normal(0, 0.15, n_blob))
        parts_y.append(min2[1] + rng.normal(0, 0.15, n_blob))

    x_path = np.concatenate(parts_x) + rng.normal(0, noise_std, n_steps)
    y_path = np.concatenate(parts_y) + rng.normal(0, noise_std, n_steps)
    z_path = surface_fn(x_path, y_path)
    return np.column_stack([x_path, y_path, z_path])


def generate_one_way_blob(
    min1: np.ndarray,
    min2: np.ndarray,
    surface_fn,
    n_steps: int,
    noise_std: float,
    rng: np.random.Generator,
    y_offset: float = 0.8,
) -> np.ndarray:
    """Dense blobs at each minimum connected by a single bridge -> path, no cycle."""
    n_blob = n_steps // 3
    n_bridge = n_steps - 2 * n_blob

    blob1_x = min1[0] + rng.normal(0, 0.2, n_blob)
    blob1_y = min1[1] + rng.normal(0, 0.2, n_blob)
    blob2_x = min2[0] + rng.normal(0, 0.2, n_blob)
    blob2_y = min2[1] + rng.normal(0, 0.2, n_blob)

    t_bridge = np.linspace(0, 1, n_bridge)
    bridge_x = min1[0] + (min2[0] - min1[0]) * t_bridge + rng.normal(0, 0.03, n_bridge)
    bridge_y = (min1[1] + (min2[1] - min1[1]) * t_bridge
                + y_offset * np.sin(np.pi * t_bridge) + rng.normal(0, 0.03, n_bridge))

    x_path = np.concatenate([blob1_x, bridge_x, blob2_x])
    y_path = np.concatenate([blob1_y, bridge_y, blob2_y])
    z_path = surface_fn(x_path, y_path)
    return np.column_stack([x_path, y_path, z_path])


def generate_one_way_arc(
    min1: np.ndarray,
    min2: np.ndarray,
    surface_fn,
    n_steps: int,
    noise_std: float,
    rng: np.random.Generator,
    y_offset: float = 0.8,
) -> np.ndarray:
    """Single arc out and back along the same route -> clean arc style, path only."""
    n_half = n_steps // 2
    t_out = np.linspace(0, 1, n_half)
    x_out = min1[0] + (min2[0] - min1[0]) * t_out
    y_out = min1[1] + (min2[1] - min1[1]) * t_out + y_offset * np.sin(np.pi * t_out)

    t_back = np.linspace(1, 0, n_steps - n_half)
    x_back = min1[0] + (min2[0] - min1[0]) * t_back
    y_back = min1[1] + (min2[1] - min1[1]) * t_back + y_offset * np.sin(np.pi * t_back)

    x_path = np.concatenate([x_out, x_back]) + rng.normal(0, noise_std * 1.5, n_steps)
    y_path = np.concatenate([y_out, y_back]) + rng.normal(0, noise_std * 1.5, n_steps)
    z_path = surface_fn(x_path, y_path)
    return np.column_stack([x_path, y_path, z_path])


# ── DM analysis ─────────────────────────────────────────────────────────────


def compute_point_density(points: np.ndarray, k: int = 10) -> np.ndarray:
    tree = KDTree(points)
    distances, _ = tree.query(points, k=min(k + 1, len(points)))
    return 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-8)


def run_dm_analysis(
    points: np.ndarray,
    config: MorseConfig,
    output_dir: str,
    experiment_name: str,
) -> tuple[list[np.ndarray], nx.Graph]:
    exp_dir = os.path.abspath(os.path.join(output_dir, experiment_name))
    os.makedirs(exp_dir, exist_ok=True)

    print(f"  Running DM analysis for {experiment_name}...")
    print(f"    Points: {len(points)}, k={config.k_neighbors}, pers_delta={config.pers_delta}")

    cycles, G_dm = extract_skeleton_pcd_dm(
        points=points,
        k=config.k_neighbors,
        pers_delta=config.pers_delta,
        epsilon=config.epsilon,
        output_dir=exp_dir,
        cleanup=False,
    )

    print(f"    DM graph: {G_dm.number_of_nodes()} nodes, {G_dm.number_of_edges()} edges")
    print(f"    Cycles found: {len(cycles)}")

    if cycles:
        cycles = sorted(cycles, key=len, reverse=True)
        print(f"    Longest cycle: {len(cycles[0])} vertices")

    return cycles, G_dm


def extract_paths_between_minima(
    G_dm: nx.Graph,
    output_dir: str,
) -> tuple[list[list[int]], int, int | None, int | None]:
    """Find the two most persistent minima and edge-disjoint paths between them."""
    from topo_mimic.dmwk.skeletonization.dm_path import find_dm_minima

    min1, min2 = find_dm_minima(output_dir)
    if min1 is None or min2 is None:
        return [], 0, min1, min2
    if min1 not in G_dm or min2 not in G_dm:
        return [], 0, min1, min2

    ec = nx.edge_connectivity(G_dm, min1, min2)
    try:
        raw_paths = [list(p) for p in nx.edge_disjoint_paths(G_dm, min1, min2)]
    except (nx.NetworkXError, nx.NetworkXNoPath, nx.NetworkXUnfeasible):
        raw_paths = []
    return raw_paths, ec, min1, min2


# ── Plotting ─────────────────────────────────────────────────────────────────


_NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 9,
    "axes.labelsize": 7,
    "axes.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "legend.fontsize": 6,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
}


def _plot_surface(ax, X, Y, Z, vis_cfg):
    ax.plot_surface(
        X, Y, Z, cmap='coolwarm', alpha=0.85,
        linewidth=0, antialiased=True, zorder=0,
    )
    ax.contour(
        X, Y, Z, levels=vis_cfg.contour_levels, colors='#666666',
        alpha=0.5, linewidths=0.4,
        offset=Z.min() - 0.15, zorder=1,
    )


def _plot_grey_path(ax, path, linewidth=1.0, alpha=0.5, z_lift=0.25, segment_colors=None):
    """segment_colors: list of (start_idx, end_idx, color) or None for uniform grey."""
    z = path[:, 2] + z_lift
    if segment_colors is None:
        for i in range(len(path) - 1):
            ax.plot(path[i:i+2, 0], path[i:i+2, 1], z[i:i+2],
                    color='#888888', linewidth=linewidth, alpha=alpha, zorder=10)
    else:
        color_map = {}
        for s, e, c in segment_colors:
            for idx in range(s, e):
                color_map[idx] = c
        for i in range(len(path) - 1):
            c = color_map.get(i, '#888888')
            ax.plot(path[i:i+2, 0], path[i:i+2, 1], z[i:i+2],
                    color=c, linewidth=linewidth, alpha=alpha, zorder=10)


def _plot_path_density(ax, path, vis_cfg, z_lift=0.25):
    density = compute_point_density(path, k=15)
    density_norm = (density - density.min()) / (density.max() - density.min() + 1e-8)
    cmap = cm.get_cmap(vis_cfg.density_cmap)
    colors = cmap(density_norm)
    z_raised = path[:, 2] + z_lift
    for i in range(len(path) - 1):
        ax.plot(
            path[i:i+2, 0], path[i:i+2, 1], z_raised[i:i+2],
            color=colors[i], linewidth=vis_cfg.path_linewidth, alpha=0.8, zorder=10,
        )
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=density.min(), vmax=density.max()))
    sm.set_array([])
    return sm


def _plot_cycle_overlay(ax, path, cycle_indices, vis_cfg, label="DM Cycle", z_lift=0.25):
    valid = [i for i in cycle_indices if 0 <= i < len(path)]
    if not valid:
        return
    n_sample = min(25, len(valid))
    sampled = [valid[i] for i in np.linspace(0, len(valid) - 1, n_sample, dtype=int)]
    pts = path[sampled]
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2] + z_lift,
        c='#e41a1c', s=45, marker='o', edgecolors='white',
        linewidths=0.6, zorder=100, **({'label': label} if label else {}),
    )


def _plot_path_overlay(ax, path, path_indices, vis_cfg, label="DM Path", z_lift=0.25):
    valid = [i for i in path_indices if 0 <= i < len(path)]
    if not valid:
        return
    n_sample = min(25, len(valid))
    sampled = [valid[i] for i in np.linspace(0, len(valid) - 1, n_sample, dtype=int)]
    pts = path[sampled]
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2] + z_lift,
        c='#377eb8', s=45, marker='D', edgecolors='white',
        linewidths=0.6, zorder=100, **({'label': label} if label else {}),
    )


def _plot_minima(ax, points, minima_indices, z_lift=0.25):
    if not minima_indices:
        return
    valid = [i for i in minima_indices if i is not None and 0 <= i < len(points)]
    if valid:
        pts = points[valid]
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2] + z_lift,
            c='#2ca02c', s=50, marker='v', edgecolors='white', linewidths=0.8,
            zorder=200,
        )


def plot_experiment(
    X, Y, Z, path, cycles, dm_paths, ec, min1, min2,
    title, vis_cfg, output_path, elev=25, azim=45, segment_colors=None,
):
    with plt.rc_context(_NATURE_RC):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5),
                                  subplot_kw={'projection': '3d'},
                                  gridspec_kw={'wspace': 0.25, 'width_ratios': [1, 1, 1]})
        ax0, ax1, ax2 = axes

        has_cycle = bool(cycles)
        has_path = bool(dm_paths)

        # Panel 1: colored surface
        _plot_surface(ax0, X, Y, Z, vis_cfg)
        ax0.set_title("Inv. density surface", fontsize=8)
        ax0.view_init(elev=elev, azim=azim)
        ax0.set_xlabel('Feature x'); ax0.set_ylabel('Feature y'); ax0.set_zlabel('Inv. density')

        # Panel 2: surface + grey trajectory + DM overlays
        _plot_surface(ax1, X, Y, Z, vis_cfg)
        _plot_grey_path(ax1, path, segment_colors=segment_colors)
        if has_cycle:
            _plot_cycle_overlay(ax1, path, cycles[0], vis_cfg)
        if has_path:
            for i, p in enumerate(dm_paths):
                _plot_path_overlay(ax1, path, p, vis_cfg, label="DM Path" if i == 0 else None)
        _plot_minima(ax1, path, [min1, min2])

        topo_label = []
        if has_cycle:
            topo_label.append("Cycle")
        if has_path:
            topo_label.append(f"Path (ec={ec})")
        if not has_cycle and not has_path:
            topo_label.append("No structure found")
        ax1.set_title(f"{title}\n{' | '.join(topo_label)}", fontsize=8, fontweight='bold')
        ax1.view_init(elev=elev, azim=azim)
        ax1.set_xlabel('Feature x'); ax1.set_ylabel('Feature y'); ax1.set_zlabel('Inv. density')
        ax1.legend(loc='upper right', fontsize=5, framealpha=0.9, edgecolor='0.7')

        # Panel 3: top-down, no colored surface, just contours
        ax2.contour(
            X, Y, Z, levels=vis_cfg.contour_levels, colors='#999999',
            alpha=0.6, linewidths=0.4, offset=Z.min() - 0.01,
        )
        _plot_grey_path(ax2, path, segment_colors=segment_colors)
        if has_cycle:
            _plot_cycle_overlay(ax2, path, cycles[0], vis_cfg)
        if has_path:
            for i, p in enumerate(dm_paths):
                _plot_path_overlay(ax2, path, p, vis_cfg, label=None)
        _plot_minima(ax2, path, [min1, min2])
        ax2.set_title("Top-down view", fontsize=8)
        ax2.view_init(elev=85, azim=0)
        ax2.set_xlabel('Feature x'); ax2.set_ylabel('Feature y')
        ax2.set_zticklabels([]); ax2.set_zlabel('')

        plt.savefig(output_path, dpi=vis_cfg.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
    print(f"  Saved {output_path}")


def plot_summary(experiments, vis_cfg, output_path):
    n = len(experiments)
    with plt.rc_context(_NATURE_RC):
        fig = plt.figure(figsize=(5 * n, 5))
        for i, exp in enumerate(experiments):
            ax = fig.add_subplot(1, n, i + 1, projection='3d')
            _plot_surface(ax, exp['X'], exp['Y'], exp['Z'], vis_cfg)
            _plot_grey_path(ax, exp['path'])
            if exp['cycles']:
                _plot_cycle_overlay(ax, exp['path'], exp['cycles'][0], vis_cfg)
            if exp['dm_paths']:
                for j, p in enumerate(exp['dm_paths']):
                    _plot_path_overlay(ax, exp['path'], p, vis_cfg, label="DM Path" if j == 0 else None)
            _plot_minima(ax, exp['path'], [exp['min1'], exp['min2']])

            topo = []
            if exp['cycles']:
                topo.append("Cycle")
            if exp['dm_paths']:
                topo.append(f"Path (ec={exp['ec']})")
            if not topo:
                topo.append("None")
            ax.set_title(f"{exp['title']}\n[{', '.join(topo)}]", fontsize=7, fontweight='bold')
            ax.view_init(elev=30, azim=45 + i * 25)
            ax.set_xlabel('Feature x'); ax.set_ylabel('Feature y'); ax.set_zlabel('Inv. density')
            ax.legend(loc='upper right', fontsize=4.5, framealpha=0.9, edgecolor='0.7')

        plt.tight_layout()
        plt.savefig(output_path, dpi=vis_cfg.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
    print(f"\nSaved summary: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def _run_experiment(
    name, surface_fn, path_fn, path_kwargs,
    x_range, y_range, resolution,
    dm_cfg, output_dir, vis_cfg,
):
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(x, y)
    Z = surface_fn(X, Y)

    path = path_fn(**path_kwargs)

    cycles, G_dm = run_dm_analysis(path, dm_cfg, output_dir, name)

    exp_dir = os.path.abspath(os.path.join(output_dir, name))
    dm_paths, ec, min1, min2 = extract_paths_between_minima(G_dm, exp_dir)

    return {
        'X': X, 'Y': Y, 'Z': Z, 'path': path,
        'cycles': cycles, 'dm_paths': dm_paths,
        'ec': ec, 'min1': min1, 'min2': min2,
    }


def main():
    parser = argparse.ArgumentParser(description="Density Basin DM Demo")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--pers_delta", type=float, default=None)
    args = parser.parse_args()

    default_config_path = Path(__file__).parent.parent.parent / "configs" / "demos" / "illustrative" / "demo_saddle.yaml"
    if args.config:
        cfg = load_config(args.config)
    elif default_config_path.exists():
        print(f"Loading config from {default_config_path}")
        cfg = load_config(str(default_config_path))
    else:
        cfg = Config()

    if args.seed is not None:
        cfg.path.seed = args.seed
    if args.k is not None:
        cfg.dm.k_neighbors = args.k
    if args.pers_delta is not None:
        cfg.dm.pers_delta = args.pers_delta

    if os.path.isabs(cfg.output_dir):
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(__file__).parent.parent.parent / cfg.output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Density Basin Discrete Morse Demonstration")
    print("=" * 70)
    print(f"  Output: {output_dir}")
    print(f"  DM params: k={cfg.dm.k_neighbors}, pers_delta={cfg.dm.pers_delta}")
    print(f"  Seed: {cfg.path.seed}")

    n_steps = cfg.path.n_steps
    noise = cfg.path.noise_std
    x_range = (-3.5, 3.5)
    y_range = (-3.5, 3.5)
    res = 120

    experiments = []

    # --- Exp 1: Two basins, two routes -> cycle ---
    print("\n[1/3] Exp 1: Two basins, two routes (expects CYCLE)")
    min_a = np.array([-1.5, 0.0])
    min_b = np.array([1.5, 0.0])
    rng1 = np.random.default_rng(cfg.path.seed)
    exp1 = _run_experiment(
        "exp1_two_routes", density_basin_surface_2,
        generate_round_trip_2routes,
        dict(min1=min_a, min2=min_b, surface_fn=density_basin_surface_2,
             n_steps=n_steps, noise_std=noise, rng=rng1, y_offset=0.9),
        x_range, y_range, res, cfg.dm, str(output_dir), cfg.visualization,
    )
    exp1['title'] = "Exp 1: Two Routes (Cycle)"
    experiments.append(exp1)
    plot_experiment(
        exp1['X'], exp1['Y'], exp1['Z'], exp1['path'],
        exp1['cycles'], exp1['dm_paths'], exp1['ec'], exp1['min1'], exp1['min2'],
        "Exp 1: Two routes between basins", cfg.visualization,
        str(output_dir / "exp1_two_routes.png"),
    )

    # --- Exp 2: Two basins, one route -> path only ---
    print("\n[2/3] Exp 2: Two basins, one route (expects PATH only)")
    rng2 = np.random.default_rng(cfg.path.seed + 100)
    exp2 = _run_experiment(
        "exp2_one_route", density_basin_surface_1pass,
        generate_one_way_blob,
        dict(min1=min_a, min2=min_b, surface_fn=density_basin_surface_1pass,
             n_steps=n_steps, noise_std=noise, rng=rng2, y_offset=0.9),
        x_range, y_range, res, cfg.dm, str(output_dir), cfg.visualization,
    )
    exp2['title'] = "Exp 2: One Route (Path)"
    experiments.append(exp2)
    plot_experiment(
        exp2['X'], exp2['Y'], exp2['Z'], exp2['path'],
        exp2['cycles'], exp2['dm_paths'], exp2['ec'], exp2['min1'], exp2['min2'],
        "Exp 2: One route between basins", cfg.visualization,
        str(output_dir / "exp2_one_route.png"),
    )

    # --- Exp 3: Three basins ---
    print("\n[3/3] Exp 3: Three basins (A-B cycle, A-C path)")
    min_A = np.array([-2.0, 0.0])
    min_B = np.array([1.0, 1.5])
    min_C = np.array([1.0, -1.5])
    rng3 = np.random.default_rng(cfg.path.seed + 200)
    n_sub = n_steps // 2

    path_AB = generate_round_trip_2routes(
        min_A, min_B, density_basin_surface_multi, n_sub, noise, rng3, y_offset=0.7)
    path_AC = generate_one_way_arc(
        min_A, min_C, density_basin_surface_multi, n_sub, noise,
        np.random.default_rng(cfg.path.seed + 201), y_offset=0.6)
    multi_path = np.vstack([path_AB, path_AC])

    x3 = np.linspace(-3.5, 3.5, res)
    y3 = np.linspace(-3.5, 3.5, res)
    X3, Y3 = np.meshgrid(x3, y3)
    Z3 = density_basin_surface_multi(X3, Y3)

    cycles3, G3 = run_dm_analysis(multi_path, cfg.dm, str(output_dir), "exp3_multi_basin")
    exp3_dir = os.path.abspath(os.path.join(str(output_dir), "exp3_multi_basin"))
    dm_paths3, ec3, m1_3, m2_3 = extract_paths_between_minima(G3, exp3_dir)

    exp3 = {
        'X': X3, 'Y': Y3, 'Z': Z3, 'path': multi_path,
        'cycles': cycles3, 'dm_paths': dm_paths3,
        'ec': ec3, 'min1': m1_3, 'min2': m2_3,
        'title': "Exp 3: Three Basins (Mixed)",
    }
    experiments.append(exp3)
    seg_colors_3 = [
        (0, len(path_AB), '#e41a1c'),
        (len(path_AB), len(path_AB) + len(path_AC), '#377eb8'),
    ]
    plot_experiment(
        X3, Y3, Z3, multi_path, cycles3, dm_paths3, ec3, m1_3, m2_3,
        "Exp 3: Multi-basin landscape", cfg.visualization,
        str(output_dir / "exp3_multi_basin.png"), elev=30, azim=60,
        segment_colors=seg_colors_3,
    )

    # --- Summary ---
    print("\nCreating summary figure...")
    plot_summary(experiments, cfg.visualization, str(output_dir / "saddle_dm_summary.png"))

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    for exp in experiments:
        n_cyc = len(exp['cycles'])
        n_pth = len(exp['dm_paths'])
        label = exp['title']
        parts = []
        if n_cyc > 0:
            parts.append(f"CYCLE ({len(exp['cycles'][0])} pts)")
        if n_pth > 0:
            parts.append(f"PATH (ec={exp['ec']}, {n_pth} disjoint)")
        if not parts:
            parts.append("nothing found")
        print(f"  {label}: {', '.join(parts)}")

    print(f"\nOutput: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
