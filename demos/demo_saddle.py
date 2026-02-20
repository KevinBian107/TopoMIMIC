#!/usr/bin/env python3
"""
Saddle Surface Discrete Morse Demonstration
=============================================
Demonstrates DM-G cycle detection on paths over a saddle surface.

Three experiments:
1. Path crossing between two local minima (detects topological loop)
2. Path staying within one basin (no loop detected)
3. Combined paths from experiments 1 and 2

Usage:
    python demos/demo_saddle.py [--config configs/scripts/demo_saddle.yaml]
    python demos/demo_saddle.py [--n_steps 500] [--seed 42]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
import networkx as nx
from pathlib import Path
from dataclasses import dataclass, field
import yaml

sys.path.insert(0, "/home/jovyan/vast/kaiwen/TopoVNL")
sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")

from topo_mimic.dmwk.skeletonization.skeletonization import extract_skeleton_pcd_dm


@dataclass
class SurfaceConfig:
    x_range: tuple[float, float] = (-2.5, 2.5)
    y_range: tuple[float, float] = (-2.5, 2.5)
    resolution: int = 100


@dataclass
class MorseConfig:
    k_neighbors: int = 5
    pers_delta: float = 0.3
    epsilon: float = 0.99


@dataclass
class PathConfig:
    n_steps: int = 500
    noise_std: float = 0.02
    seed: int = 42
    basin_radius: float = 0.5
    basin_step_size: float = 0.03
    basin_attraction: float = 0.02


@dataclass
class VisualizationConfig:
    summary_figsize: tuple[int, int] = (18, 6)
    individual_figsize: tuple[int, int] = (14, 6)
    dpi: int = 150
    perspective_elev: float = 25
    perspective_azim: float = 45
    topdown_elev: float = 90
    topdown_azim: float = 0
    surface_alpha: float = 0.35
    contour_levels: int = 15
    contour_alpha: float = 0.4
    path_linewidth: float = 2.5
    path_alpha: float = 0.9
    density_k: int = 15
    density_cmap: str = "plasma"
    cycle_color: str = "lime"
    cycle_edge_color: str = "darkgreen"
    cycle_marker_size: int = 80
    cycle_linewidth: float = 5.0
    minima_color: str = "blue"
    minima_marker: str = "v"
    minima_size: int = 120
    maxima_color: str = "red"
    maxima_marker: str = "^"
    maxima_size: int = 120
    saddle_color: str = "orange"
    saddle_marker: str = "s"
    saddle_size: int = 100


@dataclass
class Config:
    output_dir: str = "outputs/saddle_dm_demo"
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    path: PathConfig = field(default_factory=PathConfig)
    dm: MorseConfig = field(default_factory=MorseConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    surface_cfg = SurfaceConfig(
        x_range=tuple(cfg_dict.get("surface", {}).get("x_range", [-2.5, 2.5])),
        y_range=tuple(cfg_dict.get("surface", {}).get("y_range", [-2.5, 2.5])),
        resolution=cfg_dict.get("surface", {}).get("resolution", 100),
    )

    path_cfg = PathConfig(
        n_steps=cfg_dict.get("path", {}).get("n_steps", 500),
        noise_std=cfg_dict.get("path", {}).get("noise_std", 0.02),
        seed=cfg_dict.get("path", {}).get("seed", 42),
        basin_radius=cfg_dict.get("path", {}).get("basin_radius", 0.5),
        basin_step_size=cfg_dict.get("path", {}).get("basin_step_size", 0.03),
        basin_attraction=cfg_dict.get("path", {}).get("basin_attraction", 0.02),
    )

    dm_cfg = MorseConfig(
        k_neighbors=cfg_dict.get("dm", {}).get("k_neighbors", 5),
        pers_delta=cfg_dict.get("dm", {}).get("pers_delta", 0.3),
        epsilon=cfg_dict.get("dm", {}).get("epsilon", 0.99),
    )

    vis_dict = cfg_dict.get("visualization", {})
    vis_cfg = VisualizationConfig(
        summary_figsize=tuple(vis_dict.get("summary_figsize", [18, 6])),
        individual_figsize=tuple(vis_dict.get("individual_figsize", [14, 6])),
        dpi=vis_dict.get("dpi", 150),
        perspective_elev=vis_dict.get("perspective_elev", 25),
        perspective_azim=vis_dict.get("perspective_azim", 45),
        topdown_elev=vis_dict.get("topdown_elev", 90),
        topdown_azim=vis_dict.get("topdown_azim", 0),
        surface_alpha=vis_dict.get("surface_alpha", 0.35),
        contour_levels=vis_dict.get("contour_levels", 15),
        contour_alpha=vis_dict.get("contour_alpha", 0.4),
        path_linewidth=vis_dict.get("path_linewidth", 2.5),
        path_alpha=vis_dict.get("path_alpha", 0.9),
        density_k=vis_dict.get("density_k", 15),
        density_cmap=vis_dict.get("density_cmap", "plasma"),
        cycle_color=vis_dict.get("cycle_color", "lime"),
        cycle_edge_color=vis_dict.get("cycle_edge_color", "darkgreen"),
        cycle_marker_size=vis_dict.get("cycle_marker_size", 80),
        cycle_linewidth=vis_dict.get("cycle_linewidth", 5.0),
        minima_color=vis_dict.get("minima_color", "blue"),
        minima_marker=vis_dict.get("minima_marker", "v"),
        minima_size=vis_dict.get("minima_size", 120),
        maxima_color=vis_dict.get("maxima_color", "red"),
        maxima_marker=vis_dict.get("maxima_marker", "^"),
        maxima_size=vis_dict.get("maxima_size", 120),
        saddle_color=vis_dict.get("saddle_color", "orange"),
        saddle_marker=vis_dict.get("saddle_marker", "s"),
        saddle_size=vis_dict.get("saddle_size", 100),
    )

    return Config(
        output_dir=cfg_dict.get("output_dir", "outputs/saddle_dm_demo"),
        surface=surface_cfg,
        path=path_cfg,
        dm=dm_cfg,
        visualization=vis_cfg,
    )


def saddle_surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Generate a surface with two clear minima, an arch between them, and subtle complexity.

    The surface has:
    - Two valleys (local minima) on the left and right
    - A clear arch/hump in the middle
    - Small undulations to show surface complexity
    """
    z_base = 0.25 * x**2 - 0.08 * y**2

    z_minima = (
        -1.0 * np.exp(-((x + 1.3)**2 + y**2) / 0.6)
        - 1.0 * np.exp(-((x - 1.3)**2 + y**2) / 0.6)
    )

    z_arch = 0.7 * np.exp(-(x**2) / 0.6) * np.exp(-(y**2) / 2.0)

    z_texture = (
        0.08 * np.sin(2.5 * x) * np.cos(2.5 * y)
        + 0.05 * np.cos(3.0 * x + 0.5) * np.sin(3.0 * y)
        + 0.04 * np.sin(4.0 * x) * np.sin(4.0 * y)
    )

    z = z_base + z_minima + z_arch + z_texture

    return z


def find_critical_points(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> dict[str, list[tuple[float, float, float]]]:
    """Find local minima on the surface."""
    from scipy.ndimage import minimum_filter

    min_filtered = minimum_filter(Z, size=11)
    local_min_mask = (Z == min_filtered)

    minima = []
    border = 10
    min_indices = np.argwhere(local_min_mask)
    for idx in min_indices:
        i, j = idx
        if border < i < Z.shape[0] - border and border < j < Z.shape[1] - border:
            minima.append((X[i, j], Y[i, j], Z[i, j]))

    minima = sorted(minima, key=lambda p: p[2])[:2]

    return {"minima": minima}


def generate_path_crossing(
    min1: tuple[float, float, float],
    min2: tuple[float, float, float],
    n_steps: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a path that crosses over the arch: min1 -> min2 -> min1.

    The path goes from one minimum, over the arch to the other minimum,
    then returns back. The outward and return paths take slightly different
    routes (offset in y), creating a crossing pattern that DM-G can detect.
    """
    n_half = n_steps // 2

    t_out = np.linspace(0, 1, n_half)
    x_out = min1[0] + (min2[0] - min1[0]) * t_out
    y_out = 0.15 * np.sin(np.pi * t_out)

    t_back = np.linspace(0, 1, n_steps - n_half)
    x_back = min2[0] + (min1[0] - min2[0]) * t_back
    y_back = -0.15 * np.sin(np.pi * t_back)

    x_path = np.concatenate([x_out, x_back])
    y_path = np.concatenate([y_out, y_back])

    x_path += rng.normal(0, noise_std, len(x_path))
    y_path += rng.normal(0, noise_std, len(y_path))

    z_path = saddle_surface(x_path, y_path)

    path = np.column_stack([x_path, y_path, z_path])
    return path


def generate_path_no_loop(
    center: tuple[float, float, float],
    n_steps: int,
    rng: np.random.Generator,
    spread: float = 0.5,
) -> np.ndarray:
    """
    Generate a path that does NOT create topological loops.

    This is a simple linear/spiral path that never returns to previous regions.
    The path spreads out from the center in a monotonic way, ensuring
    the point cloud forms a tree-like structure with no cycles.
    """
    t = np.linspace(0, 1, n_steps)

    r = spread * t
    theta = 4 * np.pi * t

    x_path = center[0] + r * np.cos(theta)
    y_path = center[1] + r * np.sin(theta)

    noise_scale = 0.02
    x_path += rng.normal(0, noise_scale, n_steps)
    y_path += rng.normal(0, noise_scale, n_steps)

    z_path = saddle_surface(x_path, y_path)

    path = np.column_stack([x_path, y_path, z_path])
    return path


def compute_point_density(
    points: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Compute local density for each point using kNN distances."""
    tree = KDTree(points)
    distances, _ = tree.query(points, k=min(k + 1, len(points)))
    density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-8)
    return density


def run_dm_analysis(
    points: np.ndarray,
    config: MorseConfig,
    output_dir: str,
    experiment_name: str,
) -> tuple[list[np.ndarray], nx.Graph]:
    """Run discrete Morse graph analysis on point cloud."""
    exp_dir = os.path.abspath(os.path.join(output_dir, experiment_name))
    os.makedirs(exp_dir, exist_ok=True)

    print(f"  Running DM-G analysis for {experiment_name}...")
    print(f"    Points: {len(points)}, k={config.k_neighbors}, pers_delta={config.pers_delta}")
    print(f"    Output dir: {exp_dir}")

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


def plot_surface_only(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    critical_points: dict,
    vis_cfg: VisualizationConfig,
    output_path: str,
):
    """Plot and save the surface showing two minima and the arch between them."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        X, Y, Z,
        cmap='coolwarm',
        alpha=0.7,
        linewidth=0.1,
        edgecolor='none',
        antialiased=True,
        zorder=0,
    )

    minima = critical_points.get("minima", [])[:2]
    for point in minima:
        ax.scatter(
            point[0], point[1], point[2],
            c='blue',
            s=150,
            marker='o',
            edgecolors='white',
            linewidths=2.0,
            zorder=200,
        )

    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title('Surface with Two Minima and Central Arch', fontsize=12, fontweight='bold', pad=10)

    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=vis_cfg.dpi, bbox_inches='tight')
    print(f"  Saved surface plot: {output_path}")
    plt.close()


def plot_surface_with_path(
    ax: Axes3D,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    path: np.ndarray,
    cycles: list[np.ndarray],
    critical_points: dict,
    title: str,
    vis_cfg: VisualizationConfig,
    show_density: bool = True,
):
    """Plot the surface with path and highlight cycle frames."""
    ax.plot_surface(
        X, Y, Z,
        cmap='Greys',
        alpha=vis_cfg.surface_alpha,
        linewidth=0.3,
        edgecolor='gray',
        antialiased=True,
        zorder=0,
    )

    ax.contour(
        X, Y, Z,
        levels=vis_cfg.contour_levels,
        colors='gray',
        alpha=vis_cfg.contour_alpha,
        linewidths=0.5,
        offset=Z.min() - 0.1,
        zorder=1,
    )

    if show_density:
        density = compute_point_density(path, k=vis_cfg.density_k)
        density_norm = (density - density.min()) / (density.max() - density.min() + 1e-8)
        colors = cm.get_cmap(vis_cfg.density_cmap)(density_norm)

        for i in range(len(path) - 1):
            ax.plot(
                path[i:i+2, 0],
                path[i:i+2, 1],
                path[i:i+2, 2],
                color=colors[i],
                linewidth=vis_cfg.path_linewidth,
                alpha=vis_cfg.path_alpha,
                zorder=10,
            )

        sm = plt.cm.ScalarMappable(cmap=vis_cfg.density_cmap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
    else:
        ax.plot(
            path[:, 0], path[:, 1], path[:, 2],
            color='steelblue',
            linewidth=vis_cfg.path_linewidth,
            alpha=vis_cfg.path_alpha,
            zorder=10,
        )

    if cycles:
        longest_cycle = cycles[0]
        valid_indices = [idx for idx in longest_cycle if 0 <= idx < len(path)]

        if valid_indices:
            n_sample = min(20, len(valid_indices))
            if len(valid_indices) > n_sample:
                sample_idx = np.linspace(0, len(valid_indices) - 1, n_sample, dtype=int)
                sampled_indices = [valid_indices[i] for i in sample_idx]
            else:
                sampled_indices = valid_indices

            cycle_points = path[sampled_indices]

            ax.scatter(
                cycle_points[:, 0],
                cycle_points[:, 1],
                cycle_points[:, 2],
                c='red',
                s=100,
                marker='x',
                linewidths=2.5,
                zorder=100,
                label=f'DM-G loop ({len(valid_indices)} frames)',
            )
    else:
        ax.text2D(
            0.5, 0.02,
            "No cycle detected",
            transform=ax.transAxes,
            fontsize=9,
            ha='center',
            color='gray',
            fontstyle='italic',
        )

    minima = critical_points.get("minima", [])[:2]
    for point in minima:
        ax.scatter(
            point[0], point[1], point[2],
            c='blue',
            s=100,
            marker='o',
            edgecolors='white',
            linewidths=1.5,
            zorder=200,
        )

    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_zlabel('Z', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    if cycles:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)


def plot_experiment_summary(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    results: dict,
    critical_points: dict,
    vis_cfg: VisualizationConfig,
    output_path: str,
):
    """Create a summary figure with all three experiments."""
    fig = plt.figure(figsize=vis_cfg.summary_figsize)

    for i, (exp_name, exp_data) in enumerate(results.items()):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')

        plot_surface_with_path(
            ax=ax,
            X=X,
            Y=Y,
            Z=Z,
            path=exp_data['path'],
            cycles=exp_data['cycles'],
            critical_points=critical_points,
            title=exp_data['title'],
            vis_cfg=vis_cfg,
            show_density=True,
        )

        ax.view_init(elev=vis_cfg.perspective_elev, azim=vis_cfg.perspective_azim + i * 30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=vis_cfg.dpi, bbox_inches='tight')
    print(f"\nSaved summary figure to {output_path}")
    plt.close()


def plot_individual_experiment(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    path: np.ndarray,
    cycles: list[np.ndarray],
    critical_points: dict,
    title: str,
    vis_cfg: VisualizationConfig,
    output_path: str,
):
    """Create a detailed figure for a single experiment with multiple views."""
    fig = plt.figure(figsize=vis_cfg.individual_figsize)

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_surface_with_path(ax1, X, Y, Z, path, cycles, critical_points, f"{title} (Perspective)", vis_cfg, True)
    ax1.view_init(elev=vis_cfg.perspective_elev, azim=vis_cfg.perspective_azim)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_surface_with_path(ax2, X, Y, Z, path, cycles, critical_points, f"{title} (Top-Down)", vis_cfg, True)
    ax2.view_init(elev=vis_cfg.topdown_elev, azim=vis_cfg.topdown_azim)

    plt.tight_layout()
    plt.savefig(output_path, dpi=vis_cfg.dpi, bbox_inches='tight')
    print(f"  Saved {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Saddle Surface DM-G Demo")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--n_steps", type=int, default=None, help="Number of timesteps per path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--k", type=int, default=None, help="k-neighbors for DM-G")
    parser.add_argument("--pers_delta", type=float, default=None, help="Persistence threshold")
    args = parser.parse_args()

    default_config_path = Path(__file__).parent.parent / "configs" / "demos" / "demo_saddle.yaml"

    if args.config:
        cfg = load_config(args.config)
    elif default_config_path.exists():
        print(f"Loading config from {default_config_path}")
        cfg = load_config(str(default_config_path))
    else:
        cfg = Config()

    if args.n_steps is not None:
        cfg.path.n_steps = args.n_steps
    if args.seed is not None:
        cfg.path.seed = args.seed
    if args.k is not None:
        cfg.dm.k_neighbors = args.k
    if args.pers_delta is not None:
        cfg.dm.pers_delta = args.pers_delta

    if os.path.isabs(cfg.output_dir):
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / cfg.output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Saddle Surface Discrete Morse Demonstration")
    print("=" * 70)
    print(f"  Output dir: {output_dir}")
    print(f"  Path length: {cfg.path.n_steps} steps")
    print(f"  Random seed: {cfg.path.seed}")
    print(f"  DM-G params: k={cfg.dm.k_neighbors}, pers_delta={cfg.dm.pers_delta}")

    rng = np.random.default_rng(cfg.path.seed)

    print("\n[1/7] Generating saddle surface...")
    x = np.linspace(*cfg.surface.x_range, cfg.surface.resolution)
    y = np.linspace(*cfg.surface.y_range, cfg.surface.resolution)
    X, Y = np.meshgrid(x, y)
    Z = saddle_surface(X, Y)
    print(f"  Surface grid: {cfg.surface.resolution}x{cfg.surface.resolution}")

    print("\n[2/7] Finding minima...")
    critical_points = find_critical_points(X, Y, Z)
    print(f"  Found {len(critical_points['minima'])} local minima")

    print("\n[3/7] Saving surface visualization...")
    plot_surface_only(
        X, Y, Z, critical_points, cfg.visualization,
        str(output_dir / "surface.png"),
    )

    minima = critical_points['minima']
    if len(minima) >= 2:
        min1 = minima[0]
        min2 = minima[1]
    else:
        x1, x2 = -1.2, 1.2
        min1 = (x1, 0.0, float(saddle_surface(np.array([x1]), np.array([0.0]))[0]))
        min2 = (x2, 0.0, float(saddle_surface(np.array([x2]), np.array([0.0]))[0]))

    dist = np.sqrt((min2[0] - min1[0])**2 + (min2[1] - min1[1])**2)
    print(f"  Two minima (distance = {dist:.2f}):")
    print(f"    Min 1: ({min1[0]:.2f}, {min1[1]:.2f}, {min1[2]:.2f})")
    print(f"    Min 2: ({min2[0]:.2f}, {min2[1]:.2f}, {min2[2]:.2f})")

    results = {}

    print("\n[4/7] Experiment 1: Path crossing over arch (min1 -> min2 -> min1)...")
    rng_exp1 = np.random.default_rng(cfg.path.seed)
    path1 = generate_path_crossing(
        min1=min1,
        min2=min2,
        n_steps=cfg.path.n_steps,
        noise_std=cfg.path.noise_std,
        rng=rng_exp1,
    )
    cycles1, G1 = run_dm_analysis(path1, cfg.dm, str(output_dir), "experiment_1_crossing")
    results['exp1'] = {
        'path': path1,
        'cycles': cycles1,
        'graph': G1,
        'title': 'Exp 1: Crossing Path\n(Expects Cycle Detection)',
    }
    plot_individual_experiment(
        X, Y, Z, path1, cycles1, critical_points,
        "Experiment 1: Crossing Path",
        cfg.visualization,
        str(output_dir / "exp1_crossing.png"),
    )

    print("\n[5/7] Experiment 2: Path within single basin (no crossing)...")
    rng_exp2 = np.random.default_rng(cfg.path.seed + 1000)
    path2 = generate_path_no_loop(
        center=min1,
        n_steps=cfg.path.n_steps,
        rng=rng_exp2,
        spread=cfg.path.basin_radius,
    )
    cycles2, G2 = run_dm_analysis(path2, cfg.dm, str(output_dir), "experiment_2_basin")
    results['exp2'] = {
        'path': path2,
        'cycles': cycles2,
        'graph': G2,
        'title': 'Exp 2: Single Basin Path\n(Expects No Cycle)',
    }
    plot_individual_experiment(
        X, Y, Z, path2, cycles2, critical_points,
        "Experiment 2: Basin Path",
        cfg.visualization,
        str(output_dir / "exp2_basin.png"),
    )

    print("\n[6/7] Experiment 3: Combined paths (same seeds as Exp 1 & 2)...")
    rng_exp3_crossing = np.random.default_rng(cfg.path.seed)
    rng_exp3_basin = np.random.default_rng(cfg.path.seed + 1000)

    path3_crossing = generate_path_crossing(
        min1=min1,
        min2=min2,
        n_steps=cfg.path.n_steps,
        noise_std=cfg.path.noise_std,
        rng=rng_exp3_crossing,
    )
    path3_basin = generate_path_no_loop(
        center=min1,
        n_steps=cfg.path.n_steps,
        rng=rng_exp3_basin,
        spread=cfg.path.basin_radius,
    )

    path3_combined = np.vstack([path3_crossing, path3_basin])

    cycles3, G3 = run_dm_analysis(path3_combined, cfg.dm, str(output_dir), "experiment_3_combined")
    results['exp3'] = {
        'path': path3_combined,
        'cycles': cycles3,
        'graph': G3,
        'title': 'Exp 3: Combined Paths\n(Crossing + Basin)',
    }
    plot_individual_experiment(
        X, Y, Z, path3_combined, cycles3, critical_points,
        "Experiment 3: Combined Paths",
        cfg.visualization,
        str(output_dir / "exp3_combined.png"),
    )

    print("\n[7/7] Creating summary figure...")
    plot_experiment_summary(
        X, Y, Z, results, critical_points,
        cfg.visualization,
        str(output_dir / "saddle_dm_summary.png"),
    )

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"  Experiment 1 (Crossing):  {len(cycles1)} cycles detected")
    if cycles1:
        print(f"    -> Longest cycle: {len(cycles1[0])} vertices")
    print(f"  Experiment 2 (Basin):     {len(cycles2)} cycles detected")
    if cycles2:
        print(f"    -> Longest cycle: {len(cycles2[0])} vertices")
    else:
        print(f"    -> No cycle (as expected)")
    print(f"  Experiment 3 (Combined):  {len(cycles3)} cycles detected")
    if cycles3:
        print(f"    -> Longest cycle: {len(cycles3[0])} vertices")

    print(f"\nOutput saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
