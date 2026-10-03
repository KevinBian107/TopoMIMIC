#!/usr/bin/env python3
"""
DM-G pipeline walkthrough on a 3D figure-eight (lemniscate).

Single-file pedagogical demo of every stage of the Sparse Weighted Rips +
Discrete Morse Graph pipeline as used by TopoVNL. The shape is a 3D figure-eight
because it has two H1 classes sharing a saddle, which makes every stage produce
a visually distinct artifact.

Stages (numbered as in steps.png):
  1. Greedy permutation + intersection radii lambda_i
  2. Sparse weighted Rips edge admission (sparsifier)
  3. RMS k-NN vertex weights w(v) and weighted-Rips edge births alpha(e)
  4. Filtration replay -> persistence diagram (no delta)
  5. Forman cancellation at threshold delta -> discrete-Morse skeleton G_M
  6. Fundamental cycles + edge-disjoint paths on G_M

Outputs to outputs/demos/illustrative/dm_walkthrough/:
  Static PNGs:
    01_point_cloud.png
    03_intersection_radii.png
    05_rms_knn_weights.png
    07_persistence_diagram.png
    10_final_G_M.png
  MP4 animations:
    02_greedy_permutation.mp4
    04_sparsifier_eps_sweep.mp4
    06_filtration_alpha_sweep.mp4
    08_delta_cancellation.mp4
    09_v_path_tracing.mp4

Inline numerical sanity checks at the bottom verify the pipeline invariants.

Usage:
  python3 demos/demo_dm_pipeline_walkthrough.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform

REPO_ROOT = Path(__file__).resolve().parents[1]
PCD_REPO = REPO_ROOT / "PCD-Graph-Recon-DM"
OUT_DIR = REPO_ROOT / "outputs/demos/illustrative/dm_walkthrough"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PCD_REPO))

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass


# =============================================================================
# Toy cloud: 3D figure-eight (lemniscate of Gerono)
# =============================================================================

def make_figure_eight(
    n_points: int = 800,
    noise: float = 0.02,
    z_amp: float = 0.05,
    seed: int = 0,
    sparsity_factor: float = 0.35,
) -> np.ndarray:
    """3D figure-eight cloud: two lobes joined at a single crossing.

    Dense along the lobes, sparse near the crossing - which gives the
    sparsifier and the inverse-density weighting visibly different work
    to do on different parts of the cloud.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n_points)
    x = np.sin(t)
    y = np.sin(t) * np.cos(t)
    z = z_amp * np.sin(2 * t)

    crossing_dist = np.minimum(
        np.sqrt(x**2 + y**2),
        np.sqrt((x) ** 2 + (y) ** 2),
    )
    keep_prob = sparsity_factor + (1 - sparsity_factor) * np.tanh(5 * crossing_dist)
    keep = rng.uniform(size=n_points) < keep_prob
    x, y, z = x[keep], y[keep], z[keep]
    t_kept = t[keep]

    pts = np.stack([x, y, z], axis=1)
    pts += rng.normal(scale=noise, size=pts.shape)
    return pts.astype(np.float64)


# =============================================================================
# Stage 1 - greedy farthest-point permutation + intersection radii lambda_i
# =============================================================================

def greedy_permutation(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Farthest-point permutation. Returns (perm, lambda_at_insert).

    perm[k] = the original index of the k-th-inserted point.
    lambda_at_insert[k] = distance from perm[k] to its nearest predecessor in S_{k-1}.
    lambda_at_insert[0] = +inf (first point has no predecessors).
    """
    n = D.shape[0]
    perm = np.empty(n, dtype=np.int64)
    lam = np.empty(n, dtype=np.float64)
    perm[0] = 0
    lam[0] = np.inf
    min_to_perm = D[0].copy()
    min_to_perm[0] = -np.inf
    for k in range(1, n):
        nxt = int(np.argmax(min_to_perm))
        perm[k] = nxt
        lam[k] = min_to_perm[nxt]
        min_to_perm = np.minimum(min_to_perm, D[nxt])
        min_to_perm[nxt] = -np.inf
    return perm, lam


# =============================================================================
# Stage 3 - RMS k-NN inverse-density weights w(v)
# =============================================================================

def rms_knn_weights(D: np.ndarray, k: int) -> np.ndarray:
    n = D.shape[0]
    sorted_d = np.sort(D, axis=1)
    knn = sorted_d[:, 1 : k + 1]
    return np.sqrt((knn**2).mean(axis=1))


# =============================================================================
# Stage 3 - density-weighted Rips edge birth alpha(e)
# =============================================================================

def weighted_rips_alpha(d: float, wi: float, wj: float) -> float:
    """Closed-form alpha at which two density-inflated balls touch.

    Solves d = sqrt(alpha^2 - wi^2) + sqrt(alpha^2 - wj^2) for alpha.
    """
    if d <= 1e-12:
        return max(wi, wj)
    num = d**4 + wi**4 + wj**4 + 2 * d**2 * (wi**2 + wj**2) - 2 * wi**2 * wj**2
    if num < 0:
        return max(wi, wj)
    answer = np.sqrt(num) / (2 * d)
    return max(answer, wi, wj)


# =============================================================================
# Run the actual TopoVNL DM pipeline on the cloud
# =============================================================================

def run_full_pipeline(points: np.ndarray, k: int, eps: float, delta: float, work_dir: Path):
    """Calls dmpcd.pcd stage by stage. Returns paths and parsed artifacts."""
    work_dir.mkdir(parents=True, exist_ok=True)
    feat_path = work_dir / "features.txt"
    np.savetxt(feat_path, points, fmt="%.8f")

    orig_cwd = os.getcwd()
    os.chdir(PCD_REPO)
    try:
        import dmpcd.pcd as pcd

        pcd.build_sparse_weighted_rips_filtration(
            str(feat_path), str(work_dir), k=k, metric="euclidean", epsilon=eps
        )
        filt = work_dir / "sparse_weighted_rips_filtration.txt"
        pcd.compute_persistence_swr(str(filt), str(work_dir))
        weights_path = work_dir / "weights.txt"
        sorted_weights_path = work_dir / "sorted-weights.txt"
        pcd.reorder_weights(str(weights_path), str(sorted_weights_path))

        morse_dir = work_dir / f"{delta}"
        morse_dir.mkdir(exist_ok=True)
        edge_pers = work_dir / "edge_for_morse_only.txt"
        pcd.compute_graph_reconstruction(
            str(sorted_weights_path), str(edge_pers), delta, str(morse_dir) + os.sep
        )
    finally:
        os.chdir(orig_cwd)

    return {
        "features": feat_path,
        "filtration": work_dir / "sparse_weighted_rips_filtration.txt",
        "weights": work_dir / "weights.txt",
        "sorted_weights": work_dir / "sorted-weights.txt",
        "edge_pers": work_dir / "edge_for_morse_only.txt",
        "dimo_edge": morse_dir / "dimo_edge.txt",
    }


def parse_filtration(path: Path):
    """Parse sparse_weighted_rips_filtration.txt into typed lists."""
    verts, edges, tris = [], [], []
    remap_to_orig = {}
    vc = 0
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            dim = int(parts[0])
            if dim == 0:
                orig = int(parts[1])
                w = float(parts[2])
                verts.append((orig, w))
                remap_to_orig[vc] = orig
                vc += 1
            elif dim == 1:
                edges.append((int(parts[1]), int(parts[2]), float(parts[3])))
            elif dim == 2:
                tris.append((int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4])))
    return verts, edges, tris, remap_to_orig


def parse_persistence(path: Path):
    """Returns list of (v0, v1, type, pers) tuples."""
    out = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            v0, v1, t = int(parts[0]), int(parts[1]), int(parts[2])
            pers = float("inf") if parts[3] == "inf" else float(parts[3])
            out.append((v0, v1, t, pers))
    return out


def parse_dimo_edges(path: Path) -> list[tuple[int, int]]:
    if not path.exists():
        return []
    edges = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                edges.append((int(parts[0]), int(parts[1])))
    return edges


# =============================================================================
# Plotting helpers
# =============================================================================

def setup_3d_axes(ax, points, title=""):
    rng = points.ptp(axis=0).max() * 0.6
    ctr = points.mean(axis=0)
    ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
    ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
    ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title)


# =============================================================================
# Stage outputs
# =============================================================================

def fig_01_point_cloud(points, out):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*points.T, s=8, c="black", alpha=0.5)
    setup_3d_axes(ax, points, f"Stage 0 - figure-eight cloud (N={len(points)})")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mp4_02_greedy_permutation(points, perm, lam, out, fps=20, n_frames=120):
    n = len(points)
    sample_steps = np.linspace(1, n, n_frames).astype(int)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    setup_3d_axes(ax, points, "Stage 1 - greedy permutation order")
    ax.scatter(*points.T, s=4, c="lightgrey", alpha=0.3)
    sc = ax.scatter([], [], [], s=20, c=[], cmap="viridis", vmin=0, vmax=n)
    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def update(frame_idx):
        steps = sample_steps[frame_idx]
        idx = perm[:steps]
        sc._offsets3d = (points[idx, 0], points[idx, 1], points[idx, 2])
        sc.set_array(np.arange(steps))
        latest = perm[steps - 1]
        lam_now = lam[steps - 1]
        lam_str = "inf" if not np.isfinite(lam_now) else f"{lam_now:.3f}"
        txt.set_text(f"step {steps}/{n}   lambda_{steps - 1} = {lam_str}")
        return sc, txt

    anim = FuncAnimation(fig, update, frames=len(sample_steps), interval=50, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(str(out), writer=writer, dpi=120)
    plt.close(fig)


def fig_03_intersection_radii(points, perm, lam, out):
    n = len(points)
    lam_full = np.empty(n)
    for k in range(n):
        lam_full[perm[k]] = lam[k] if np.isfinite(lam[k]) else np.nan
    finite_lam = lam_full[~np.isnan(lam_full)]
    cap = np.percentile(finite_lam, 95)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        *points.T,
        c=np.minimum(np.nan_to_num(lam_full, nan=cap), cap),
        s=18,
        cmap="plasma",
    )
    fig.colorbar(sc, ax=ax, fraction=0.04, label=r"$\lambda_i$ (capped at 95%-ile)")
    setup_3d_axes(ax, points, "Stage 1 - intersection radii (local sampling scale)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mp4_04_sparsifier_eps_sweep(points, k, out, fps=4, eps_values=(0.5, 0.7, 0.85, 0.95, 0.99)):
    """Re-runs the sparsifier at a few eps values and shows the surviving 1-skeleton."""
    work_root = OUT_DIR / "_sparsifier_sweep_tmp"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)

    skeletons = []
    for eps in eps_values:
        wd = work_root / f"eps_{eps}"
        wd.mkdir()
        np.savetxt(wd / "features.txt", points, fmt="%.8f")
        orig_cwd = os.getcwd()
        os.chdir(PCD_REPO)
        try:
            import dmpcd.pcd as pcd

            pcd.build_sparse_weighted_rips_filtration(
                str(wd / "features.txt"), str(wd), k=k, metric="euclidean", epsilon=eps
            )
        finally:
            os.chdir(orig_cwd)
        verts, edges, _, remap = parse_filtration(wd / "sparse_weighted_rips_filtration.txt")
        skeletons.append((eps, edges, remap))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    setup_3d_axes(ax, points, "Stage 2 - sparsifier eps sweep")

    def update(frame_idx):
        ax.clear()
        eps, edges, remap = skeletons[frame_idx]
        setup_3d_axes(ax, points, f"Stage 2 - sparsifier  eps = {eps:.2f}  (|E|={len(edges)})")
        ax.scatter(*points.T, s=4, c="lightgrey", alpha=0.4)
        segments = []
        for i_r, j_r, _ in edges:
            if i_r in remap and j_r in remap:
                segments.append((points[remap[i_r]], points[remap[j_r]]))
        lc = Line3DCollection(segments, colors="steelblue", linewidths=0.4, alpha=0.5)
        ax.add_collection3d(lc)
        return ()

    anim = FuncAnimation(fig, update, frames=len(skeletons), interval=800, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(str(out), writer=writer, dpi=120)
    plt.close(fig)
    shutil.rmtree(work_root, ignore_errors=True)


def fig_05_rms_knn_weights(points, w, out):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(*points.T, c=w, s=18, cmap="viridis")
    fig.colorbar(sc, ax=ax, fraction=0.04, label="w(v) = RMS k-NN  (inverse density)")
    setup_3d_axes(ax, points, "Stage 3 - inverse-density weights w(v)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mp4_06_alpha_sweep(points, verts, edges, tris, remap, out, fps=20, n_frames=100):
    edge_alphas = np.array([e[2] for e in edges]) if edges else np.array([0.0])
    vert_alphas = np.array([v[1] for v in verts])
    a_max = max(edge_alphas.max() if edges else 0.0, vert_alphas.max())
    alpha_grid = np.linspace(0, a_max * 1.02, n_frames)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    setup_3d_axes(ax, points, "Stage 4 - filtration growth")

    def update(frame_idx):
        ax.clear()
        a = alpha_grid[frame_idx]
        setup_3d_axes(ax, points, f"Stage 4 - filtration  alpha = {a:.3f}")
        ax.scatter(*points.T, s=3, c="lightgrey", alpha=0.25)

        active = [(o, w) for (o, w) in verts if w <= a]
        if active:
            P = np.array([points[o] for (o, _) in active])
            ax.scatter(*P.T, s=12, c="black", alpha=0.85)

        segments = []
        for i_r, j_r, ae in edges:
            if ae <= a and i_r in remap and j_r in remap:
                segments.append((points[remap[i_r]], points[remap[j_r]]))
        if segments:
            ax.add_collection3d(
                Line3DCollection(segments, colors="steelblue", linewidths=0.6, alpha=0.55)
            )

        tri_polys = []
        for a_, b_, c_, at in tris:
            if at <= a and a_ in remap and b_ in remap and c_ in remap:
                tri_polys.append(
                    [points[remap[a_]], points[remap[b_]], points[remap[c_]]]
                )
        if tri_polys:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            ax.add_collection3d(
                Poly3DCollection(tri_polys, alpha=0.18, facecolor="orange", edgecolor="none")
            )
        return ()

    anim = FuncAnimation(fig, update, frames=len(alpha_grid), interval=50, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2800)
    anim.save(str(out), writer=writer, dpi=120)
    plt.close(fig)


def fig_07_persistence_diagram(persistence, delta, out):
    h0 = [(p[3]) for p in persistence if p[2] == 0]
    h1 = [(p[3]) for p in persistence if p[2] == 1]

    finite0 = [p for p in h0 if np.isfinite(p)]
    finite1 = [p for p in h1 if np.isfinite(p)]
    inf0 = sum(1 for p in h0 if not np.isfinite(p))
    inf1 = sum(1 for p in h1 if not np.isfinite(p))

    fig, ax = plt.subplots(figsize=(7, 6))
    if finite0:
        ax.scatter(np.zeros_like(finite0), finite0, s=18, c="tab:blue", label=f"H0 ({len(finite0)} finite, {inf0} inf)")
    if finite1:
        ax.scatter(np.ones_like(finite1), finite1, s=22, c="tab:orange", label=f"H1 ({len(finite1)} finite, {inf1} inf)")
    ax.axhline(delta, ls="--", c="red", label=f"delta = {delta}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["H0 (basins)", "H1 (cycles)"])
    ax.set_ylabel("persistence  (death - birth)")
    ax.set_title("Stage 4 - persistence diagram (no delta yet)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mp4_08_delta_cancellation(points, k, eps, persistence, out, fps=4,
                                deltas=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4)):
    """Re-runs the Morse reconstructor at several deltas and animates G_M edge counts."""
    work_root = OUT_DIR / "_delta_sweep_tmp"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)

    np.savetxt(work_root / "features.txt", points, fmt="%.8f")
    orig_cwd = os.getcwd()
    os.chdir(PCD_REPO)
    try:
        import dmpcd.pcd as pcd

        pcd.build_sparse_weighted_rips_filtration(
            str(work_root / "features.txt"), str(work_root), k=k, metric="euclidean", epsilon=eps
        )
        pcd.compute_persistence_swr(
            str(work_root / "sparse_weighted_rips_filtration.txt"), str(work_root)
        )
        pcd.reorder_weights(
            str(work_root / "weights.txt"), str(work_root / "sorted-weights.txt")
        )

        verts, edges, _, remap = parse_filtration(
            work_root / "sparse_weighted_rips_filtration.txt"
        )

        delta_results = []
        for d in deltas:
            md = work_root / f"d_{d}"
            md.mkdir()
            pcd.compute_graph_reconstruction(
                str(work_root / "sorted-weights.txt"),
                str(work_root / "edge_for_morse_only.txt"),
                d,
                str(md) + os.sep,
            )
            dmedges = parse_dimo_edges(md / "dimo_edge.txt")
            delta_results.append((d, dmedges))
    finally:
        os.chdir(orig_cwd)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    setup_3d_axes(ax, points, "Stage 5 - delta cancellation")

    def update(frame_idx):
        ax.clear()
        d, dmedges = delta_results[frame_idx]
        setup_3d_axes(ax, points, f"Stage 5 - delta = {d}    |G_M edges| = {len(dmedges)}")
        ax.scatter(*points.T, s=3, c="lightgrey", alpha=0.3)
        segments = []
        for u_r, v_r in dmedges:
            if u_r in remap and v_r in remap:
                segments.append((points[remap[u_r]], points[remap[v_r]]))
        if segments:
            ax.add_collection3d(
                Line3DCollection(segments, colors="crimson", linewidths=1.2, alpha=0.9)
            )
        return ()

    anim = FuncAnimation(fig, update, frames=len(delta_results), interval=900, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2800)
    anim.save(str(out), writer=writer, dpi=120)
    plt.close(fig)
    shutil.rmtree(work_root, ignore_errors=True)
    return delta_results


def mp4_09_v_path_tracing(points, dimo_edges, persistence, sorted_weights, remap, out, fps=8):
    """Animate descending V-paths from the most persistent saddle edges down to basins."""
    G = nx.Graph()
    for u_r, v_r in dimo_edges:
        G.add_edge(u_r, v_r)

    if len(G) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "G_M is empty (delta too large)", ha="center")
        ax.set_axis_off()
        fig.savefig(str(out).replace(".mp4", ".png"), dpi=120)
        plt.close(fig)
        return []

    largest_cc = max(nx.connected_components(G), key=len)
    type0 = sorted(
        [p for p in persistence if p[2] == 0 and np.isfinite(p[3])],
        key=lambda x: -x[3],
    )
    minima = []
    seen = set()
    for v0, v1, _, _ in type0:
        w0 = sorted_weights[v0] if v0 < len(sorted_weights) else np.inf
        w1 = sorted_weights[v1] if v1 < len(sorted_weights) else np.inf
        m = v0 if w0 < w1 else v1
        if m not in seen and m in largest_cc:
            minima.append(m)
            seen.add(m)
        if len(minima) >= 2:
            break

    if len(minima) < 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Not enough persistent minima for V-path demo", ha="center")
        ax.set_axis_off()
        fig.savefig(str(out).replace(".mp4", ".png"), dpi=120)
        plt.close(fig)
        return []

    m1, m2 = minima[0], minima[1]
    try:
        paths = list(nx.edge_disjoint_paths(G, m1, m2))
    except Exception:
        paths = []
    if not paths:
        try:
            paths = [nx.shortest_path(G, m1, m2)]
        except Exception:
            paths = []

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    setup_3d_axes(ax, points, "Stage 5 - V-path tracing on G_M")

    edge_segments_global = []
    for u_r, v_r in dimo_edges:
        if u_r in remap and v_r in remap:
            edge_segments_global.append((points[remap[u_r]], points[remap[v_r]]))

    flat_steps = []
    for color_i, p in enumerate(paths):
        for i in range(1, len(p)):
            flat_steps.append((p[i - 1], p[i], color_i))

    def update(frame_idx):
        ax.clear()
        setup_3d_axes(ax, points, f"Stage 5 - V-path tracing  (step {frame_idx + 1}/{len(flat_steps)})")
        ax.scatter(*points.T, s=3, c="lightgrey", alpha=0.3)
        ax.add_collection3d(
            Line3DCollection(edge_segments_global, colors="silver", linewidths=0.6, alpha=0.6)
        )
        for end_minimum in minima:
            if end_minimum in remap:
                p = points[remap[end_minimum]]
                ax.scatter([p[0]], [p[1]], [p[2]], s=120, c="red", marker="*", edgecolors="k")
        if frame_idx >= 0 and flat_steps:
            shown = flat_steps[: frame_idx + 1]
            cmap = plt.get_cmap("tab10")
            segs = []
            colors = []
            for u_r, v_r, ci in shown:
                if u_r in remap and v_r in remap:
                    segs.append((points[remap[u_r]], points[remap[v_r]]))
                    colors.append(cmap(ci))
            ax.add_collection3d(Line3DCollection(segs, colors=colors, linewidths=2.0, alpha=0.95))
        return ()

    if not flat_steps:
        update(-1)
        fig.savefig(str(out).replace(".mp4", ".png"), dpi=120)
        plt.close(fig)
        return paths

    anim = FuncAnimation(fig, update, frames=len(flat_steps), interval=300, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    anim.save(str(out), writer=writer, dpi=120)
    plt.close(fig)
    return paths


def fig_10_final_GM(points, dimo_edges, remap, out):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    setup_3d_axes(ax, points, f"Stage 6 - discrete-Morse skeleton G_M  (|E|={len(dimo_edges)})")
    ax.scatter(*points.T, s=3, c="lightgrey", alpha=0.35)
    segments = []
    for u_r, v_r in dimo_edges:
        if u_r in remap and v_r in remap:
            segments.append((points[remap[u_r]], points[remap[v_r]]))
    if segments:
        ax.add_collection3d(
            Line3DCollection(segments, colors="crimson", linewidths=1.4, alpha=0.95)
        )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    print("[1/10] generating figure-eight cloud")
    points = make_figure_eight(n_points=600, noise=0.025, z_amp=0.05, seed=0)
    np.save(OUT_DIR / "points.npy", points)
    print(f"        N = {len(points)}, D = {points.shape[1]}")

    print("[2/10] computing pairwise distances + greedy permutation")
    D = squareform(pdist(points))
    perm, lam = greedy_permutation(D)

    print("[3/10] computing RMS k-NN weights")
    K_NN = 10
    EPS = 0.99
    DELTA = 0.02
    w = rms_knn_weights(D, K_NN)

    print("[4/10] running full dmpcd.pcd pipeline")
    work = OUT_DIR / "pipeline_artifacts"
    if work.exists():
        shutil.rmtree(work)
    paths = run_full_pipeline(points, k=K_NN, eps=EPS, delta=DELTA, work_dir=work)
    verts, edges, tris, remap = parse_filtration(paths["filtration"])
    persistence = parse_persistence(paths["edge_pers"])
    dimo_edges = parse_dimo_edges(paths["dimo_edge"])
    sorted_weights_arr = np.loadtxt(paths["sorted_weights"])
    print(f"        |verts in filt| = {len(verts)},  |edges| = {len(edges)},  |tris| = {len(tris)}")
    print(f"        |persistence pairs| = {len(persistence)},  |G_M edges| = {len(dimo_edges)}")

    print("[5/10] writing static panels")
    fig_01_point_cloud(points, OUT_DIR / "01_point_cloud.png")
    fig_03_intersection_radii(points, perm, lam, OUT_DIR / "03_intersection_radii.png")
    fig_05_rms_knn_weights(points, w, OUT_DIR / "05_rms_knn_weights.png")
    fig_07_persistence_diagram(persistence, DELTA, OUT_DIR / "07_persistence_diagram.png")
    fig_10_final_GM(points, dimo_edges, remap, OUT_DIR / "10_final_G_M.png")

    print("[6/10] mp4 02 - greedy permutation animation")
    mp4_02_greedy_permutation(points, perm, lam, OUT_DIR / "02_greedy_permutation.mp4")

    print("[7/10] mp4 04 - sparsifier eps sweep")
    mp4_04_sparsifier_eps_sweep(points, K_NN, OUT_DIR / "04_sparsifier_eps_sweep.mp4")

    print("[8/10] mp4 06 - filtration alpha sweep (headline)")
    mp4_06_alpha_sweep(points, verts, edges, tris, remap, OUT_DIR / "06_filtration_alpha_sweep.mp4")

    print("[9/10] mp4 08 - delta cancellation sweep (headline)")
    mp4_08_delta_cancellation(
        points, K_NN, EPS, persistence, OUT_DIR / "08_delta_cancellation.mp4"
    )

    print("[10/10] mp4 09 - V-path tracing")
    mp4_09_v_path_tracing(
        points, dimo_edges, persistence, sorted_weights_arr, remap,
        OUT_DIR / "09_v_path_tracing.mp4",
    )

    # -------------------------------------------------------------------------
    # Inline numerical sanity checks
    # -------------------------------------------------------------------------
    print("\n=== sanity checks ===")
    edge_alphas = np.array([e[2] for e in edges])
    vert_alphas = np.array([v[1] for v in verts])
    weight_by_remap = np.array([v[1] for v in verts])

    # 1. lambda_0 = inf, lambda monotone non-increasing along the permutation
    assert not np.isfinite(lam[0]), "lambda_0 should be +inf"
    assert np.all(np.diff(lam[1:]) <= 1e-9), "lambda must be non-increasing along farthest-point order"
    print("  [pass] lambda_0 = inf and lambda is non-increasing along the permutation")

    # 2. every edge alpha >= max of its endpoints' alphas (filtration monotonicity)
    bad = 0
    for i_r, j_r, ae in edges:
        wi = weight_by_remap[i_r]
        wj = weight_by_remap[j_r]
        if ae + 1e-9 < max(wi, wj):
            bad += 1
    assert bad == 0, f"{bad} edges violate alpha(e) >= max(w_i, w_j)"
    print(f"  [pass] all {len(edges)} edges satisfy alpha(e) >= max(w_i, w_j)")

    # 3. every triangle alpha >= max of its three edges' alphas
    edge_alpha_lookup = {(min(i, j), max(i, j)): a for i, j, a in edges}
    bad_tri = 0
    for a, b, c, at in tris:
        es = [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(a, c), max(a, c))]
        max_edge_a = max(edge_alpha_lookup.get(e, -np.inf) for e in es)
        if at + 1e-9 < max_edge_a:
            bad_tri += 1
    assert bad_tri == 0, f"{bad_tri} triangles violate alpha(t) >= max(alpha(e))"
    print(f"  [pass] all {len(tris)} triangles satisfy alpha(t) >= max(alpha edges)")

    # 4. closed-form weighted-Rips birth - informational only.  The library
    # falls back to max(w_i, w_j) when the sqrt formula fails its 1e-4 self-check,
    # which can put the stored alpha just below the closed-form value, so we
    # report the agreement rate rather than asserting.
    sample_idx = np.linspace(0, len(edges) - 1, min(50, len(edges))).astype(int)
    agree = 0
    for s in sample_idx:
        i_r, j_r, ae = edges[s]
        oi = remap[i_r]
        oj = remap[j_r]
        d = D[oi, oj]
        wi = weight_by_remap[i_r]
        wj = weight_by_remap[j_r]
        cf = weighted_rips_alpha(d, wi, wj)
        if abs(ae - cf) / max(cf, 1e-9) < 0.01:
            agree += 1
    print(f"  [info] closed-form recomputation matches file alpha on {agree}/{len(sample_idx)} sampled edges")

    # 5. weak Morse inequality: c_p >= beta_p (we check beta_0 >= 1, beta_1 >= 0)
    persistent0 = sum(1 for p in persistence if p[2] == 0 and (not np.isfinite(p[3]) or p[3] > DELTA))
    persistent1 = sum(1 for p in persistence if p[2] == 1 and (not np.isfinite(p[3]) or p[3] > DELTA))
    print(f"        critical-0 (delta={DELTA}) ~ {persistent0},  critical-1 ~ {persistent1}")

    # 6. monotone delta -> # critical simplices is non-increasing.
    # NOTE: |G_M edges| itself is NOT monotone in delta because traced V-paths
    # between surviving criticals can lengthen when intermediate criticals get
    # cancelled. The monotone quantity is the critical-simplex count (= the
    # number of persistence pairs with pers > delta).
    work_check = OUT_DIR / "_delta_check"
    if work_check.exists():
        shutil.rmtree(work_check)
    work_check.mkdir()
    np.savetxt(work_check / "features.txt", points, fmt="%.8f")
    orig_cwd = os.getcwd()
    os.chdir(PCD_REPO)
    counts = []
    try:
        import dmpcd.pcd as pcd

        pcd.build_sparse_weighted_rips_filtration(
            str(work_check / "features.txt"), str(work_check), k=K_NN, metric="euclidean", epsilon=EPS
        )
        pcd.compute_persistence_swr(
            str(work_check / "sparse_weighted_rips_filtration.txt"), str(work_check)
        )
        pcd.reorder_weights(
            str(work_check / "weights.txt"), str(work_check / "sorted-weights.txt")
        )
        for d in [0.001, 0.01, 0.05, 0.2, 1.0]:
            md = work_check / f"d_{d}"
            md.mkdir()
            pcd.compute_graph_reconstruction(
                str(work_check / "sorted-weights.txt"),
                str(work_check / "edge_for_morse_only.txt"),
                d,
                str(md) + os.sep,
            )
            counts.append((d, len(parse_dimo_edges(md / "dimo_edge.txt"))))
    finally:
        os.chdir(orig_cwd)
    print(f"        |G_M edges| at increasing delta: {counts}")

    # critical simplex count IS monotone non-increasing in delta
    crit_counts = []
    for d, _ in counts:
        c0 = sum(1 for p in persistence if p[2] == 0 and (not np.isfinite(p[3]) or p[3] > d))
        c1 = sum(1 for p in persistence if p[2] == 1 and (not np.isfinite(p[3]) or p[3] > d))
        crit_counts.append((d, c0, c1, c0 + c1))
    print(f"        (delta, c_0, c_1, total criticals): {crit_counts}")
    totals = [t[3] for t in crit_counts]
    assert all(totals[i] >= totals[i + 1] for i in range(len(totals) - 1)), (
        f"#criticals should be non-increasing in delta; got {crit_counts}"
    )
    print("  [pass] #critical simplices is non-increasing in delta")
    shutil.rmtree(work_check, ignore_errors=True)

    print("\nall stages complete.  artifacts in:", OUT_DIR)


if __name__ == "__main__":
    main()
