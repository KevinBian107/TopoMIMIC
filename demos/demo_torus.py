#!/usr/bin/env python3
"""
Torus SWR Pipeline Demonstration
=================================
Single-file demonstration showing the 6 stages of the Sparse Weighted Rips
pipeline on a torus, using the TopoVNL codebase's existing DM pipeline.

Generates an animated GIF showing:
1. Rotating torus point cloud
2. Greedy permutation (farthest-point sampling)
3. Vertex weighting (k-NN RMS density)
4. SWR filtration sweep (edges in alpha order)
5. Discrete Morse skeleton
6. Cycle detection (H1 loops)

Usage:
    python demos/demo_torus.py

Output:
    torus_dm_demo.gif
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass

sys.path.insert(0, "/home/jovyan/vast/kaiwen/TopoVNL")
sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")

from topo_mimic.dmwk.skeletonization.skeletonization import extract_skeleton_pcd_dm
from topo_mimic.dmwk.skeletonization.remap import (
    parse_swr_remap,
    remap_cycle_to_original,
)


@dataclass
class TorusConfig:
    R: float = 3.0
    r: float = 1.0
    n_points: int = 2000
    noise_std: float = 0.1


@dataclass
class MorseConfig:
    k_neighbors: int = 3
    pers_delta: float = 0.6
    epsilon: float = 0.99


@dataclass
class AnimConfig:
    n_frames: int = 480
    fps: int = 5
    figsize: Tuple[int, int] = (16, 8)
    dpi: int = 100
    pause_frames: int = 10


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


def compute_greedy_permutation(
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Farthest-point sampling (greedy permutation).

    Args:
        points: (N, D) point cloud

    Returns:
        permutation: (N,) indices in farthest-point order
        lambdas: (N,) insertion radii (inf for first point)
    """
    n = len(points)
    dist_matrix = squareform(pdist(points))

    perm = [0]
    lambdas_list = [np.inf]
    min_dist_to_perm = dist_matrix[0].copy()
    min_dist_to_perm[0] = -np.inf

    for _ in range(n - 1):
        farthest = np.argmax(min_dist_to_perm)
        perm.append(farthest)
        lambdas_list.append(min_dist_to_perm[farthest])
        min_dist_to_perm[farthest] = -np.inf
        np.minimum(min_dist_to_perm, dist_matrix[farthest], out=min_dist_to_perm)

    return np.array(perm), np.array(lambdas_list)


def compute_vertex_weights(points: np.ndarray, k: int) -> np.ndarray:
    """k-NN RMS weight: sqrt(mean(k-NN distances^2))."""
    tree = KDTree(points)
    distances, _ = tree.query(points, k=k + 1)
    knn_dists = distances[:, 1:]
    return np.sqrt(np.mean(knn_dists ** 2, axis=1))


def parse_swr_edges(
    output_dir: str,
    remap_to_orig: dict[int, int],
) -> List[Tuple[int, int, float]]:
    """Parse SWR filtration file for edges with alpha values (in original index space).

    Args:
        output_dir: Directory containing DM output files
        remap_to_orig: Mapping from remapped to original indices (from parse_swr_remap)

    Returns:
        swr_edges: List of (orig_v0, orig_v1, alpha) sorted by alpha
    """
    filt_path = os.path.join(output_dir, 'sparse_weighted_rips_filtration.txt')
    raw_edges = []

    with open(filt_path) as f:
        for line in f:
            parts = line.strip().split()
            dim = int(parts[0])
            if dim == 1:
                rv0, rv1 = int(parts[1]), int(parts[2])
                alpha = float(parts[3])
                raw_edges.append((rv0, rv1, alpha))

    original_edges = []
    for rv0, rv1, alpha in raw_edges:
        if rv0 in remap_to_orig and rv1 in remap_to_orig:
            original_edges.append((remap_to_orig[rv0], remap_to_orig[rv1], alpha))

    original_edges.sort(key=lambda e: e[2])
    return original_edges


def order_cycle_vertices(points: np.ndarray, cycle_indices: np.ndarray) -> np.ndarray:
    """Order cycle vertices to form a smooth loop via nearest-neighbor traversal."""
    if len(cycle_indices) < 3:
        return cycle_indices

    cycle_points = points[cycle_indices]

    ordered = [0]
    remaining = set(range(1, len(cycle_indices)))

    while remaining:
        current_pt = cycle_points[ordered[-1]]
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


class TorusDMAnimation:
    """Animator showing the 6 stages of the SWR pipeline on a torus."""

    def __init__(
        self,
        points: np.ndarray,
        permutation: np.ndarray,
        lambdas: np.ndarray,
        weights: np.ndarray,
        swr_edges: List[Tuple[int, int, float]],
        dm_edges: List[Tuple[int, int]],
        cycles: List[np.ndarray],
        cfg: AnimConfig,
    ):
        self.points = points
        self.permutation = permutation
        self.lambdas = lambdas
        self.weights = weights
        self.swr_edges = swr_edges
        self.dm_edges = dm_edges
        self.cycles = cycles
        self.cfg = cfg
        self.n = len(points)

        # Phase boundaries (6 equal phases)
        phase_len = cfg.n_frames // 6
        self.phase_ends = [phase_len * (i + 1) for i in range(5)]

        # Normalize lambdas for coloring (replace inf with max finite)
        finite_lambdas = lambdas[np.isfinite(lambdas)]
        lambda_colors = lambdas.copy()
        lambda_colors[~np.isfinite(lambda_colors)] = finite_lambdas.max() * 1.1
        lmin, lmax = lambda_colors.min(), lambda_colors.max()
        self.lambda_norm = (lambda_colors - lmin) / (lmax - lmin + 1e-8)

        # Per-point lambda norm indexed by original point index
        self.point_lambda_norm = np.zeros(self.n)
        for i, idx in enumerate(permutation):
            self.point_lambda_norm[idx] = self.lambda_norm[i]

        # Normalize weights for coloring
        wmin, wmax = weights.min(), weights.max()
        self.weight_norm = (weights - wmin) / (wmax - wmin + 1e-8)

        # SWR edge alpha values for coloring
        if swr_edges:
            alphas = np.array([e[2] for e in swr_edges])
            amin, amax = alphas.min(), alphas.max()
            self.swr_alpha_norm = (alphas - amin) / (amax - amin + 1e-8)
        else:
            self.swr_alpha_norm = np.array([])

        # Sort DM edges by length for progressive reveal
        if dm_edges:
            dm_lengths = [np.linalg.norm(points[e[0]] - points[e[1]]) for e in dm_edges]
            sort_idx = np.argsort(dm_lengths)
            self.dm_edges_sorted = [dm_edges[i] for i in sort_idx]
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
        self.fig = plt.figure(figsize=self.cfg.figsize, facecolor='white')
        self.ax_left = self.fig.add_subplot(121, projection='3d', facecolor='white')
        self.ax_right = self.fig.add_subplot(122, projection='3d', facecolor='white')
        self.fig.subplots_adjust(wspace=0.05)

    def _reset_axes(self, ax):
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-6.5, 6.5)
        ax.set_zlim(-2, 2)
        ax.set_box_aspect([3.25, 3.25, 1])
        ax.set_axis_off()

    def _get_phase(self, frame: int) -> int:
        for i, end in enumerate(self.phase_ends):
            if frame < end:
                return i + 1
        return 6

    def _get_phase_info(self, frame: int) -> Tuple[str, str]:
        phase = self._get_phase(frame)
        descriptions = {
            1: ("Phase 1", "Point Cloud"),
            2: ("Phase 2", "Greedy Permutation"),
            3: ("Phase 3", "Vertex Weighting"),
            4: ("Phase 4", "SWR Filtration Sweep"),
            5: ("Phase 5", "DM Skeleton"),
            6: ("Phase 6", "Cycle Detection (H\u2081)"),
        }
        return descriptions[phase]

    def _phase_progress(self, frame: int) -> float:
        phase = self._get_phase(frame)
        if phase == 1:
            start, end = 0, self.phase_ends[0]
        elif phase <= 5:
            start, end = self.phase_ends[phase - 2], self.phase_ends[phase - 1]
        else:
            start, end = self.phase_ends[4], self.cfg.n_frames
        local_frame = frame - start
        anim_len = (end - start) - self.cfg.pause_frames
        if local_frame >= anim_len:
            return 1.0
        return local_frame / max(anim_len, 1)

    def _draw_on_axis(self, ax, frame: int, azimuth: float, elev: float, title_suffix: str = ""):
        ax.clear()
        self._reset_axes(ax)
        ax.view_init(elev=elev, azim=azimuth)

        phase = self._get_phase(frame)
        _, description = self._get_phase_info(frame)
        progress = self._phase_progress(frame)

        # Phase 1: bare torus, all points uniform color
        if phase == 1:
            ax.scatter(
                self.points[:, 0], self.points[:, 1], self.points[:, 2],
                c='steelblue', s=10, alpha=0.5, edgecolors='none'
            )

        # Phase 2: progressive point reveal in permutation order, colored by lambda
        elif phase == 2:
            n_show = max(1, int(self.n * progress))
            visible_indices = self.permutation[:n_show]
            visible_pts = self.points[visible_indices]
            colors = self.lambda_norm[:n_show]
            ax.scatter(
                visible_pts[:, 0], visible_pts[:, 1], visible_pts[:, 2],
                c=colors, cmap='hot', s=15, alpha=0.7, edgecolors='none',
                vmin=0, vmax=1,
            )

        # Phase 3: all points, recolored by weight (dense=bright)
        elif phase == 3:
            ax.scatter(
                self.points[:, 0], self.points[:, 1], self.points[:, 2],
                c=self.weight_norm, cmap='viridis_r', s=10, alpha=0.6, edgecolors='none',
                vmin=0, vmax=1,
            )

        # Phase 4: SWR filtration sweep - edges in alpha order
        elif phase == 4:
            ax.scatter(
                self.points[:, 0], self.points[:, 1], self.points[:, 2],
                c=self.weight_norm, cmap='viridis_r', s=8, alpha=0.4, edgecolors='none',
                vmin=0, vmax=1,
            )
            if self.swr_edges:
                n_edges_show = max(1, int(len(self.swr_edges) * progress))
                edges_to_show = self.swr_edges[:n_edges_show]
                segments = create_edge_segments(self.points, [(e[0], e[1]) for e in edges_to_show])
                if len(segments) > 0:
                    edge_colors = cm.plasma(self.swr_alpha_norm[:n_edges_show])
                    edge_colors[:, 3] = 0.6
                    lc = Line3DCollection(segments, colors=edge_colors, linewidths=0.5)
                    ax.add_collection3d(lc)

        # Phase 5: DM skeleton highlighted, SWR edges faded
        elif phase == 5:
            ax.scatter(
                self.points[:, 0], self.points[:, 1], self.points[:, 2],
                c=self.weight_norm, cmap='viridis_r', s=8, alpha=0.3, edgecolors='none',
                vmin=0, vmax=1,
            )
            if self.swr_edges:
                segments = create_edge_segments(self.points, [(e[0], e[1]) for e in self.swr_edges])
                if len(segments) > 0:
                    lc = Line3DCollection(segments, colors='steelblue', linewidths=0.3, alpha=0.1)
                    ax.add_collection3d(lc)
            if self.dm_edges_sorted:
                n_dm_show = max(1, int(len(self.dm_edges_sorted) * progress))
                dm_to_show = self.dm_edges_sorted[:n_dm_show]
                segments = create_edge_segments(self.points, dm_to_show)
                if len(segments) > 0:
                    lc = Line3DCollection(segments, colors='lime', linewidths=2.5, alpha=0.9)
                    ax.add_collection3d(lc)

        # Phase 6: autoregressive cycle tracing
        elif phase == 6:
            ax.scatter(
                self.points[:, 0], self.points[:, 1], self.points[:, 2],
                c='gray', s=6, alpha=0.2, edgecolors='none'
            )
            if self.dm_edges_sorted:
                segments = create_edge_segments(self.points, self.dm_edges_sorted)
                if len(segments) > 0:
                    lc = Line3DCollection(segments, colors='lime', linewidths=1.5, alpha=0.3)
                    ax.add_collection3d(lc)
            if self.cycle_segments_ordered:
                n_cycle_edges = min(
                    max(1, int(len(self.cycle_segments_ordered) * progress)),
                    len(self.cycle_segments_ordered),
                )
                cycle_edges_to_show = self.cycle_segments_ordered[:n_cycle_edges]
                segments = create_edge_segments(self.points, cycle_edges_to_show)
                if len(segments) > 0:
                    lc_glow = Line3DCollection(segments, colors='yellow', linewidths=12, alpha=0.4)
                    ax.add_collection3d(lc_glow)
                    lc = Line3DCollection(segments, colors='red', linewidths=4.5, alpha=0.95)
                    ax.add_collection3d(lc)

                if n_cycle_edges > 0:
                    last_edge = cycle_edges_to_show[-1]
                    head_pt = self.points[last_edge[1]]
                    ax.scatter(
                        [head_pt[0]], [head_pt[1]], [head_pt[2]],
                        c='yellow', s=120, alpha=1.0, edgecolors='red',
                        linewidths=2, zorder=200, marker='o'
                    )

                visited = set()
                for u, v in cycle_edges_to_show:
                    visited.add(u)
                    visited.add(v)
                if visited:
                    visited_pts = self.points[list(visited)]
                    ax.scatter(
                        visited_pts[:, 0], visited_pts[:, 1], visited_pts[:, 2],
                        c='red', s=40, alpha=0.8, edgecolors='white',
                        linewidths=0.8, zorder=100
                    )

        ax.set_title(f"{description}{title_suffix}", fontsize=12, fontweight='bold', pad=10)

    def animate(self, frame: int):
        azimuth = (frame / self.cfg.n_frames) * 360

        self._draw_on_axis(self.ax_left, frame, azimuth=azimuth, elev=25, title_suffix=" (Rotating)")
        self._draw_on_axis(self.ax_right, frame, azimuth=0, elev=90, title_suffix=" (Top-Down)")

        phase_name, _ = self._get_phase_info(frame)
        self.fig.suptitle(
            f"{phase_name} | Points: {self.n} | SWR edges: {len(self.swr_edges)} | DM edges: {len(self.dm_edges)}",
            fontsize=11, color='gray', y=0.02,
        )
        return []

    def save_gif(self, output_path: str):
        print(f"Creating animation with {self.cfg.n_frames} frames...")

        anim = FuncAnimation(
            self.fig,
            self.animate,
            frames=self.cfg.n_frames,
            interval=1000 // self.cfg.fps,
            blit=False,
        )

        print(f"Saving to {output_path}...")
        anim.save(output_path, writer='pillow', fps=self.cfg.fps, dpi=self.cfg.dpi)
        print(f"Done! Saved to {output_path}")

        plt.close(self.fig)


def main():
    out_dir = Path(__file__).parent.parent / "outputs" / "demo_torus"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "torus_dm_demo.gif"
    dm_output_dir = out_dir / "torus_dm_output"
    dm_output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Torus SWR Pipeline Demonstration")
    print("=" * 60)

    torus_cfg = TorusConfig(R=3.5, r=1.0, n_points=1000, noise_std=0.01)
    morse_cfg = MorseConfig(k_neighbors=3, pers_delta=0.6, epsilon=0.99)
    anim_cfg = AnimConfig(n_frames=300, fps=15)

    # Step 1: Generate torus
    print("\n[1/6] Generating torus point cloud...")
    points = generate_torus(torus_cfg)
    print(f"      Generated {len(points)} points")

    # Step 2: Greedy permutation
    print("\n[2/6] Computing greedy permutation...")
    permutation, lambdas = compute_greedy_permutation(points)
    finite_lambdas = lambdas[np.isfinite(lambdas)]
    print(f"      Lambda range: [{finite_lambdas.min():.4f}, {finite_lambdas.max():.4f}]")

    # Step 3: Run DM pipeline (produces all intermediate files)
    print("\n[3/6] Running DM pipeline...")
    cycles, G_dm = extract_skeleton_pcd_dm(
        points=points,
        k=morse_cfg.k_neighbors,
        pers_delta=morse_cfg.pers_delta,
        epsilon=morse_cfg.epsilon,
        output_dir=str(dm_output_dir),
        cleanup=False,
    )
    print(f"      DM graph: {G_dm.number_of_nodes()} nodes, {G_dm.number_of_edges()} edges")
    print(f"      Found {len(cycles)} cycles")

    # Step 4: Vertex weights
    print("\n[4/6] Computing vertex weights...")
    weights = compute_vertex_weights(points, k=morse_cfg.k_neighbors)
    print(f"      Weight range: [{weights.min():.4f}, {weights.max():.4f}]")

    # Step 5: Parse SWR edges and convert DM data to original indices
    print("\n[5/6] Parsing SWR filtration...")
    remap_to_orig = parse_swr_remap(str(dm_output_dir))
    swr_edges = parse_swr_edges(str(dm_output_dir), remap_to_orig)
    print(f"      SWR edges: {len(swr_edges)}")

    n_points = len(points)
    dm_edges_orig = []
    for u, v in G_dm.edges():
        if u in remap_to_orig and v in remap_to_orig:
            dm_edges_orig.append((remap_to_orig[u], remap_to_orig[v]))
    print(f"      DM edges (converted): {len(dm_edges_orig)}")

    cycles_orig = []
    for cycle in cycles:
        orig_cycle = remap_cycle_to_original(cycle, remap_to_orig, n_points)
        if len(orig_cycle) > 0:
            cycles_orig.append(orig_cycle)
    if cycles_orig:
        cycles_orig = sorted(cycles_orig, key=len, reverse=True)
        print(f"      Longest cycle: {len(cycles_orig[0])} vertices")

    # Step 6: Create animation
    print("\n[6/6] Creating animation...")
    animator = TorusDMAnimation(
        points=points,
        permutation=permutation,
        lambdas=lambdas,
        weights=weights,
        swr_edges=swr_edges,
        dm_edges=dm_edges_orig,
        cycles=cycles_orig,
        cfg=anim_cfg,
    )

    animator.save_gif(str(output_path))

    print("\n" + "=" * 60)
    print(f"Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
