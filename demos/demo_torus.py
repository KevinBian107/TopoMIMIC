#!/usr/bin/env python3
"""
Torus Discrete Morse Demonstration
===================================
Single-file demonstration of discrete Morse graph reconstruction on a torus
using the TopoVNL codebase's existing DM pipeline.

Generates an animated GIF showing:
1. Rotating torus point cloud
2. kNN graph construction
3. Discrete Morse skeleton with detected cycles

Usage:
    python demos/torus_dm_demo.py

Output:
    torus_dm_demo.gif
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import KDTree
import networkx as nx
from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass

sys.path.insert(0, "/home/jovyan/vast/kaiwen/TopoVNL")
sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")

from topo_mimic.dmwk.skeletonization.skeletonization import (
    extract_skeleton_pcd_dm,
    _knn_weights_for_persistence,
)


@dataclass
class TorusConfig:
    R: float = 3.0          # Major radius (center of tube to center of torus)
    r: float = 1.0          # Minor radius (radius of tube)
    n_points: int = 2000     # Number of points to sample
    noise_std: float = 0.1 # Gaussian noise standard deviation


@dataclass
class MorseConfig:
    k_neighbors: int = 3   # k for kNN graph
    pers_delta: float = 0.6 # Persistence threshold for DM filtering
    epsilon: float = 0.99   # Sparse Rips approximation parameter


@dataclass
class AnimConfig:
    n_frames: int = 480     # Total frames in animation
    fps: int = 5           # Frames per second (slower playback)
    figsize: Tuple[int, int] = (16, 8)  # Wider for two views
    dpi: int = 100


def generate_torus(cfg: TorusConfig, seed: int = 42) -> np.ndarray:
    """Generate uniformly sampled points on a torus surface."""
    np.random.seed(seed)

    u = np.random.uniform(0, 2 * np.pi, cfg.n_points)
    v = np.random.uniform(0, 2 * np.pi, cfg.n_points)

    x = (cfg.R + cfg.r * np.cos(v)) * np.cos(u)
    y = (cfg.R + cfg.r * np.cos(v)) * np.sin(u)
    z = cfg.r * np.sin(v)

    points = np.column_stack([x, y, z])

    if cfg.noise_std > 0:
        points += np.random.normal(0, cfg.noise_std, points.shape)

    return points


def build_knn_graph(points: np.ndarray, k: int) -> Tuple[nx.Graph, np.ndarray]:
    """Build k-nearest neighbor graph with distance-based edge weights."""
    tree = KDTree(points)
    distances, indices = tree.query(points, k=k+1)

    G = nx.Graph()
    G.add_nodes_from(range(len(points)))

    edges_with_dist = []
    for i in range(len(points)):
        for j_idx in range(1, k+1):
            j = indices[i, j_idx]
            dist = distances[i, j_idx]
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=dist)
                edges_with_dist.append((i, j, dist))

    edges_array = np.array([(e[0], e[1], e[2]) for e in edges_with_dist])
    return G, edges_array


def run_dm_pipeline(
    points: np.ndarray,
    k: int,
    pers_delta: float,
    output_dir: str,
) -> Tuple[List[np.ndarray], nx.Graph, List[Tuple[int, int]]]:
    """
    Run discrete Morse pipeline using the codebase's extract_skeleton_pcd_dm.

    Returns:
        cycles: List of cycle vertex arrays
        G_dm: Discrete Morse graph
        dm_edges: List of DM edges for visualization
    """
    print(f"      Running DM pipeline (k={k}, pers_delta={pers_delta})...")

    cycles, G_dm = extract_skeleton_pcd_dm(
        points=points,
        k=k,
        pers_delta=pers_delta,
        epsilon=0.99,
        output_dir=output_dir,
        cleanup=False,
    )

    dm_edges = list(G_dm.edges())
    print(f"      DM graph: {G_dm.number_of_nodes()} nodes, {G_dm.number_of_edges()} edges")
    print(f"      Found {len(cycles)} cycles")

    return cycles, G_dm, dm_edges


def order_cycle_vertices(points: np.ndarray, cycle_indices: np.ndarray) -> np.ndarray:
    """Order cycle vertices to form a smooth loop via nearest-neighbor traversal."""
    if len(cycle_indices) < 3:
        return cycle_indices

    cycle_points = points[cycle_indices]

    # Start from first vertex, greedily visit nearest unvisited
    ordered = [0]
    remaining = set(range(1, len(cycle_indices)))

    while remaining:
        current = ordered[-1]
        current_pt = cycle_points[current]

        # Find nearest remaining
        min_dist = float('inf')
        nearest = None
        for r in remaining:
            dist = np.linalg.norm(cycle_points[r] - current_pt)
            if dist < min_dist:
                min_dist = dist
                nearest = r

        ordered.append(nearest)
        remaining.remove(nearest)

    return cycle_indices[ordered]


def create_edge_segments(points: np.ndarray, edges: List[Tuple[int, int]]) -> np.ndarray:
    """Convert edge list to line segments for 3D plotting."""
    if not edges:
        return np.empty((0, 2, 3))
    segments = []
    for u, v in edges:
        if u < len(points) and v < len(points):
            segments.append([points[u], points[v]])
    return np.array(segments) if segments else np.empty((0, 2, 3))


def create_cycle_segments(points: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    """Convert cycle to closed line segments."""
    if len(cycle) < 2:
        return np.empty((0, 2, 3))

    # Order cycle vertices
    ordered_cycle = order_cycle_vertices(points, cycle)

    segments = []
    for i in range(len(ordered_cycle)):
        u = ordered_cycle[i]
        v = ordered_cycle[(i + 1) % len(ordered_cycle)]
        if u < len(points) and v < len(points):
            segments.append([points[u], points[v]])
    return np.array(segments) if segments else np.empty((0, 2, 3))


class TorusDMAnimation:
    """Animator class for the torus discrete Morse demonstration with dual views."""

    def __init__(
        self,
        points: np.ndarray,
        knn_edges: np.ndarray,
        dm_edges: List[Tuple[int, int]],
        cycles: List[np.ndarray],
        density: np.ndarray,
        cfg: AnimConfig
    ):
        self.points = points
        self.knn_edges = knn_edges
        self.dm_edges = dm_edges
        self.cycles = cycles
        self.density = density
        self.cfg = cfg

        # Phase boundaries (equal sections for longer duration)
        self.phase1_end = cfg.n_frames // 4      # Bare torus
        self.phase2_end = cfg.n_frames // 2      # kNN construction
        self.phase3_end = 3 * cfg.n_frames // 4  # DM skeleton
        # Phase 4: Cycle highlight

        # Sort kNN edges by distance for progressive reveal
        self.knn_edges_sorted = knn_edges[np.argsort(knn_edges[:, 2])]

        # Sort DM edges by edge length for progressive reveal
        if dm_edges:
            dm_edge_lengths = [np.linalg.norm(points[e[0]] - points[e[1]]) for e in dm_edges]
            dm_sort_idx = np.argsort(dm_edge_lengths)
            self.dm_edges_sorted = [dm_edges[i] for i in dm_sort_idx]
        else:
            self.dm_edges_sorted = []

        # Prepare ordered cycle segments for autoregressive tracing
        self.cycle_segments_ordered = []
        if cycles:
            cycle = cycles[0]
            ordered_cycle = order_cycle_vertices(points, cycle)
            for i in range(len(ordered_cycle)):
                u = ordered_cycle[i]
                v = ordered_cycle[(i + 1) % len(ordered_cycle)]
                if u < len(points) and v < len(points):
                    self.cycle_segments_ordered.append((u, v))

        self._setup_figure()

    def _setup_figure(self):
        """Initialize matplotlib figure with two 3D axes."""
        self.fig = plt.figure(figsize=self.cfg.figsize, facecolor='white')

        # Left: rotating view
        self.ax_left = self.fig.add_subplot(121, projection='3d', facecolor='white')
        # Right: top-down view
        self.ax_right = self.fig.add_subplot(122, projection='3d', facecolor='white')

        self.fig.subplots_adjust(wspace=0.05)

    def _reset_axes(self, ax, is_topdown=False):
        """Reset axis limits and properties."""
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-6.5, 6.5)
        ax.set_zlim(-2, 2)
        ax.set_box_aspect([3.25, 3.25, 1])
        ax.set_axis_off()

    def _get_phase_info(self, frame: int) -> Tuple[str, str]:
        """Get phase name and description for current frame."""
        if frame < self.phase1_end:
            return "Phase 1", "Torus Point Cloud"
        elif frame < self.phase2_end:
            return "Phase 2", "Building kNN Graph"
        elif frame < self.phase3_end:
            return "Phase 3", "Discrete Morse Skeleton"
        else:
            return "Phase 4", "Detected Cycle (H₁)"

    def _draw_on_axis(self, ax, frame: int, azimuth: float, elev: float, title_suffix: str = ""):
        """Draw the current frame on a given axis."""
        ax.clear()
        self._reset_axes(ax)
        ax.view_init(elev=elev, azim=azimuth)

        phase, description = self._get_phase_info(frame)

        # Normalize density for coloring
        d_norm = (self.density - self.density.min()) / (self.density.max() - self.density.min() + 1e-8)

        # Always draw points
        ax.scatter(
            self.points[:, 0], self.points[:, 1], self.points[:, 2],
            c=d_norm, cmap='viridis', s=10, alpha=0.5, edgecolors='none'
        )

        # Phase 2: Progressive kNN edge reveal
        if frame >= self.phase1_end:
            if frame < self.phase2_end:
                progress = (frame - self.phase1_end) / (self.phase2_end - self.phase1_end)
                n_edges_show = int(len(self.knn_edges_sorted) * progress)
            else:
                n_edges_show = len(self.knn_edges_sorted)

            if n_edges_show > 0:
                edges_to_show = self.knn_edges_sorted[:n_edges_show, :2].astype(int)
                segments = create_edge_segments(self.points, [tuple(e) for e in edges_to_show])
                if len(segments) > 0:
                    alpha = 0.15 if frame >= self.phase2_end else 0.6
                    lc = Line3DCollection(segments, colors='steelblue', linewidths=0.6, alpha=alpha)
                    ax.add_collection3d(lc)

        # Phase 3: Progressive DM skeleton reveal (hidden in Phase 4)
        if self.phase2_end <= frame < self.phase3_end and self.dm_edges_sorted:
            progress = (frame - self.phase2_end) / (self.phase3_end - self.phase2_end)
            n_dm_show = int(len(self.dm_edges_sorted) * progress)

            if n_dm_show > 0:
                dm_to_show = self.dm_edges_sorted[:n_dm_show]
                segments = create_edge_segments(self.points, dm_to_show)
                if len(segments) > 0:
                    lc = Line3DCollection(segments, colors='darkgreen', linewidths=2.5, alpha=0.9)
                    ax.add_collection3d(lc)

        # Phase 4: Autoregressive cycle tracing
        if frame >= self.phase3_end and self.cycle_segments_ordered:
            phase4_progress = (frame - self.phase3_end) / (self.cfg.n_frames - self.phase3_end)
            n_cycle_edges = int(len(self.cycle_segments_ordered) * phase4_progress) + 1
            n_cycle_edges = min(n_cycle_edges, len(self.cycle_segments_ordered))

            cycle_edges_to_show = self.cycle_segments_ordered[:n_cycle_edges]

            if cycle_edges_to_show:
                segments = create_edge_segments(self.points, cycle_edges_to_show)
                if len(segments) > 0:
                    # Glow effect
                    lc_glow = Line3DCollection(segments, colors='yellow', linewidths=12, alpha=0.4)
                    ax.add_collection3d(lc_glow)

                    # Main cycle line
                    lc = Line3DCollection(segments, colors='red', linewidths=4.5, alpha=0.95)
                    ax.add_collection3d(lc)

                # Mark current head of the trace
                if n_cycle_edges > 0:
                    last_edge = cycle_edges_to_show[-1]
                    head_pt = self.points[last_edge[1]]
                    ax.scatter(
                        [head_pt[0]], [head_pt[1]], [head_pt[2]],
                        c='yellow', s=120, alpha=1.0, edgecolors='red', linewidths=2, zorder=200, marker='o'
                    )

                # Mark visited cycle nodes
                visited_nodes = set()
                for u, v in cycle_edges_to_show:
                    visited_nodes.add(u)
                    visited_nodes.add(v)
                visited_pts = self.points[list(visited_nodes)]
                ax.scatter(
                    visited_pts[:, 0], visited_pts[:, 1], visited_pts[:, 2],
                    c='red', s=40, alpha=0.8, edgecolors='white', linewidths=0.8, zorder=100
                )

        # Title
        ax.set_title(f"{description}{title_suffix}", fontsize=12, fontweight='bold', pad=10)

    def animate(self, frame: int):
        """Animation update function for both views."""
        azimuth = (frame / self.cfg.n_frames) * 360

        # Left view: rotating perspective
        self._draw_on_axis(self.ax_left, frame, azimuth=azimuth, elev=25, title_suffix=" (Rotating)")

        # Right view: top-down (fixed)
        self._draw_on_axis(self.ax_right, frame, azimuth=0, elev=90, title_suffix=" (Top-Down)")

        # Add phase info and stats as figure title
        phase, _ = self._get_phase_info(frame)
        info_text = f"{phase} | Points: {len(self.points)} | k={self.knn_edges_sorted.shape[0] // len(self.points) if len(self.points) > 0 else 0}"
        if frame >= self.phase2_end:
            info_text += f" | DM edges: {len(self.dm_edges)}"
        if frame >= self.phase3_end and self.cycles:
            info_text += f" | Cycle: {len(self.cycles[0])} vertices"

        self.fig.suptitle(info_text, fontsize=11, color='gray', y=0.02)

        return []

    def save_gif(self, output_path: str):
        """Create and save the animation as GIF."""
        print(f"Creating animation with {self.cfg.n_frames} frames...")

        anim = FuncAnimation(
            self.fig,
            self.animate,
            frames=self.cfg.n_frames,
            interval=1000 // self.cfg.fps,
            blit=False
        )

        print(f"Saving to {output_path}...")
        anim.save(output_path, writer='pillow', fps=self.cfg.fps, dpi=self.cfg.dpi)
        print(f"Done! Saved to {output_path}")

        plt.close(self.fig)


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    output_path = script_dir / "torus_dm_demo.gif"
    dm_output_dir = script_dir / "torus_dm_output"
    dm_output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Torus Discrete Morse Demonstration")
    print("Using TopoVNL's DM Pipeline")
    print("=" * 60)

    # Configuration
    torus_cfg = TorusConfig(R=3.5, r=1.0, n_points=1000, noise_std=0.01)
    morse_cfg = MorseConfig(k_neighbors=3, pers_delta=0.6, epsilon=0.99)
    anim_cfg = AnimConfig(n_frames=300, fps=15)

    # Step 1: Generate torus
    print("\n[1/5] Generating torus point cloud...")
    points = generate_torus(torus_cfg)
    print(f"      Generated {len(points)} points")

    # Step 2: Build kNN graph (for visualization)
    print("\n[2/5] Building kNN graph...")
    G_knn, knn_edges = build_knn_graph(points, morse_cfg.k_neighbors)
    print(f"      kNN graph: {G_knn.number_of_nodes()} nodes, {G_knn.number_of_edges()} edges")

    # Step 3: Compute density (for coloring)
    print("\n[3/5] Computing point density...")
    density = _knn_weights_for_persistence(points, k=morse_cfg.k_neighbors)
    print(f"      Density range: [{density.min():.2f}, {density.max():.2f}]")

    # Step 4: Run DM pipeline
    print("\n[4/5] Running discrete Morse pipeline...")
    cycles, G_dm, dm_edges = run_dm_pipeline(
        points=points,
        k=morse_cfg.k_neighbors,
        pers_delta=morse_cfg.pers_delta,
        output_dir=str(dm_output_dir),
    )

    # Sort cycles by length
    if cycles:
        cycles = sorted(cycles, key=len, reverse=True)
        print(f"      Longest cycle has {len(cycles[0])} vertices")

    # Step 5: Create animation
    print("\n[5/5] Creating animation...")
    animator = TorusDMAnimation(
        points=points,
        knn_edges=knn_edges,
        dm_edges=dm_edges,
        cycles=cycles,
        density=density,
        cfg=anim_cfg
    )

    animator.save_gif(str(output_path))

    print("\n" + "=" * 60)
    print(f"Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
