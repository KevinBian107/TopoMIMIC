#!/usr/bin/env python3
"""Primitive-decomposition walkthrough video (advisor response).

Shows, for a single clip whose DM skeleton has at least two independent cycles
and non-trivial branching trees, the full decomposition into DM-cycles (minimum
cycle basis) and DM-paths (branch-to-branch bridge corridors), as an animated grid:

  Row 1: [full clip video] [clip PCA-2D with a traveling frame marker]
         [grey skeleton] [all cycles] [all paths]
  Row 2: one video per DM-path (plays that corridor's frames)
  Row 3: one video per DM-cycle (plays that loop's frames)

The decomposition is recomputed from the cached skeleton (no C++ backend needed):
cycles = nx.minimum_cycle_basis weighted by density (extract_all_cycles); paths =
maximal non-branching chains of the bridge forest (extract_all_corridors). Both are
mapped back to frames via the cached sparse-weighted-Rips remap.

Usage:
    python -m demos.demo_primitive_decomposition --config configs/demos/demo_primitive_decomposition.yaml
    python -m demos.demo_primitive_decomposition --clip-id 8
    python -m demos.demo_primitive_decomposition --clip-id 742 --stride 2 --no-video   # static PNG only
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import math
import pickle
from dataclasses import dataclass

import h5py
import numpy as np
import networkx as nx
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.decomposition import PCA

from topo_mimic.extraction.dm_pipeline import takens_embedding
from topo_mimic.extraction.skeletonization.skeletonization import (
    extract_all_cycles,
    extract_all_corridors,
)
from topo_mimic.extraction.skeletonization.remap import (
    parse_swr_remap,
    remap_cycle_to_original,
)
from topo_mimic.extraction.skeletonization.dm_path import _parse_sorted_weights
from topo_mimic.extraction.utils.load_utils import load_rollout_from_source
from topo_mimic.analysis.method_analysis.property.theta_filmstrip import _load_mujoco_model


# distinct, print-safe colors for the primitives (cycles and paths get their own ramps)
_CYCLE_COLORS = [
    (0.84, 0.15, 0.16), (0.89, 0.47, 0.76), (0.55, 0.34, 0.29),
    (0.74, 0.20, 0.64), (0.98, 0.33, 0.10), (0.60, 0.05, 0.30),
]
_PATH_COLORS = [
    (0.12, 0.47, 0.71), (0.17, 0.63, 0.17), (0.09, 0.66, 0.72),
    (0.20, 0.40, 0.85), (0.40, 0.55, 0.20), (0.10, 0.50, 0.50),
]
_NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 8, "axes.linewidth": 0.5,
    "figure.dpi": 200, "savefig.dpi": 200,
}
# Takens delay per feature space (m=3 for all); matches the DM-G extraction configs.
_TAKENS_TAU = {"intention": 6, "qpos": 9, "ekp": 8}


@dataclass
class DecompVizConfig:
    clip_id: int = 8
    clip_ids: list | None = None              # run several example clips in one go
    feature_space: str = "intention"          # which DM extraction to read
    extraction_root: str = "outputs/extraction"
    data_h5: str = "data/rodent_ar.h5"
    intention_key: str = "intention"
    xml_path: str = "topo_mimic/extraction/xml/rodent.xml"
    takens_m: int = 3
    takens_tau: int = 6
    min_prim_frames: int = 15                 # drop un-renderable branch stubs (keeps short corridors)
    output_dir: str = "outputs/demos/illustrative/primitive_decomposition"
    tag: str = ""                             # optional subfolder + figure label (e.g. rear_walk_rear)
    # rendering / animation
    render_size: int = 300                    # square mujoco frame (px)
    stride: int = 1                           # subsample clip frames for the top video timeline
    framerate: int = 25
    dpi: int = 180
    make_video: bool = True                   # if False, only the static decomposition PNG

    @property
    def extraction_dir(self) -> str:
        return os.path.join(self.extraction_root, f"dm_pipeline_{self.feature_space}")


def load_config(path: str | None) -> DecompVizConfig:
    cfg = DecompVizConfig()
    if path and os.path.exists(path):
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


# ----------------------------------------------------------------------------- decomposition

@dataclass
class Decomposition:
    clip_id: int
    coords: np.ndarray                  # (n_takens, 2) PCA of the Takens cloud, original order
    vert_coord: dict                    # G-vertex (remapped) -> (x, y)
    cycles_frames: list                 # list of sorted frame-index arrays (one per cycle)
    paths_frames: list                  # list of sorted frame-index arrays (one per path)
    cycles_coords: list                 # list of (k, 2) PCA polylines (temporal order)
    paths_coords: list                  # list of (k, 2) PCA polylines (temporal order)
    G: nx.Graph
    remap: dict
    n_frames: int
    offset: int                         # Takens index i  ->  original frame i + offset


def _vertex_to_frame(remap: dict, n_takens: int) -> dict:
    return {v: remap[v] for v in remap if remap[v] < n_takens}


def load_decomposition(cfg: DecompVizConfig) -> Decomposition:
    cid = cfg.clip_id
    G = pickle.load(open(os.path.join(cfg.extraction_dir, "all_G_sm.pkl"), "rb"))[cid]
    clip_dir = os.path.join(cfg.extraction_dir, "clips", f"clip_{cid}")
    remap = parse_swr_remap(clip_dir)
    try:
        vweight = _parse_sorted_weights(clip_dir)
    except Exception:
        vweight = None

    # Read the actual DM input cloud (features.txt) so coordinates match the skeleton in
    # any feature space; for qpos/ekp this is the PCA-reduced + Takens cloud the DM ran on.
    coords = PCA(n_components=2).fit_transform(np.loadtxt(os.path.join(clip_dir, "features.txt")))
    n_takens = len(coords)
    vert_coord = {v: coords[remap[v]] for v in G.nodes() if v in remap and remap[v] < n_takens}

    # Takens index i -> original frame i + offset; offset = (m-1)*tau, per feature space
    offset = (cfg.takens_m - 1) * _TAKENS_TAU.get(cfg.feature_space, cfg.takens_tau)

    def _to_prim(vtx):
        # remap graph vertices -> Takens rows (temporal order); frames = rows + offset
        tk = remap_cycle_to_original(vtx, remap, n_takens)
        rows = np.unique(tk)
        if len(rows) < cfg.min_prim_frames:
            return None
        return rows + offset, coords[rows]

    cyc = [p for p in (_to_prim(c) for c in extract_all_cycles(G, vertex_weight=vweight)) if p]
    pth = [p for p in (_to_prim(c) for c in extract_all_corridors(G)) if p]
    # longest first, so the most prominent primitives get the lowest indices/colors
    cyc.sort(key=lambda p: len(p[0]), reverse=True)
    pth.sort(key=lambda p: len(p[0]), reverse=True)

    return Decomposition(
        clip_id=cid, coords=coords, vert_coord=vert_coord,
        cycles_frames=[p[0] for p in cyc], paths_frames=[p[0] for p in pth],
        cycles_coords=[p[1] for p in cyc], paths_coords=[p[1] for p in pth],
        G=G, remap=remap, n_frames=n_takens + offset, offset=offset,
    )


# ----------------------------------------------------------------------------- mujoco rendering

def render_tracking_frames(qpos: np.ndarray, xml_path: str, frames: list[int],
                           size: int) -> dict:
    """Render the listed frames once, body-centered, fixed size. Returns {frame: rgb}."""
    import mujoco

    if not os.path.isabs(xml_path):
        xml_path = os.path.abspath(xml_path)
    mj_model, mj_data = _load_mujoco_model(xml_path)
    renderer = mujoco.Renderer(mj_model, height=size, width=size)

    fovy_rad = math.radians(45.0)
    view_h = 0.35
    dist = (view_h / 2.0) / math.tan(fovy_rad / 2.0) * 1.3
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = dist
    camera.elevation = -35.0
    camera.azimuth = 90.0

    out = {}
    nq = mj_model.nq
    for fi in frames:
        q = qpos[fi]
        camera.lookat[:] = [q[0], q[1], q[2] + 0.02]
        mj_data.qpos[:] = q[:nq]
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=camera)
        out[fi] = renderer.render().copy()
    renderer.close()
    return out


# ----------------------------------------------------------------------------- graph panels

def draw_graph_panel(ax, decomp: Decomposition, highlight: list | None,
                     colors: list | None, title: str, close_loops: bool = False) -> None:
    """Grey skeleton in PCA-2D, optionally with selected primitives drawn as colored
    temporally-ordered polylines (closed for cycles)."""
    vc = decomp.vert_coord
    # grey skeleton edges
    for u, v in decomp.G.edges():
        if u in vc and v in vc:
            ax.plot([vc[u][0], vc[v][0]], [vc[u][1], vc[v][1]],
                    color=(0.78, 0.78, 0.78), lw=0.6, zorder=1)
    pts = np.array(list(vc.values()))
    ax.scatter(pts[:, 0], pts[:, 1], s=3, c=[(0.6, 0.6, 0.6)], alpha=0.5,
               linewidths=0, zorder=2)

    line_handles = []
    if highlight:
        for k, seg in enumerate(highlight):
            color = colors[k % len(colors)]
            if close_loops and len(seg) >= 3:
                seg = np.vstack([seg, seg[0]])     # close the loop
            (ln,) = ax.plot(seg[:, 0], seg[:, 1], color=color, lw=1.6, alpha=0.85, zorder=4)
            line_handles.append(ln)

    ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_box_aspect(1)
    return line_handles


# ----------------------------------------------------------------------------- assembly

_INACTIVE_RGB = (0.80, 0.80, 0.80)


def _nearest_frame(P: np.ndarray, f: int) -> int:
    """Nearest frame in this primitive's (sorted) frame set to the master clip frame f."""
    if len(P) == 0:
        return f
    i = int(np.searchsorted(P, f))
    i = min(max(i, 0), len(P) - 1)
    if i > 0 and abs(int(P[i - 1]) - f) <= abs(int(P[i]) - f):
        i -= 1
    return int(P[i])


def _dim(img: np.ndarray) -> np.ndarray:
    """Fade an inactive panel toward white."""
    return np.clip(img.astype(np.float32) * 0.45 + 255 * 0.55, 0, 255).astype(np.uint8)


def _frame_dot(ax, xy):
    return ax.scatter([xy[0]], [xy[1]], s=70, facecolor="red",
                      edgecolor="black", linewidths=0.8, zorder=20)


def _set_active(rect, line, color, active: bool) -> None:
    if active:
        rect.set_edgecolor(color); rect.set_linewidth(3.2)
        if line is not None:
            line.set_alpha(0.95); line.set_linewidth(2.6)
    else:
        rect.set_edgecolor(_INACTIVE_RGB); rect.set_linewidth(1.2)
        if line is not None:
            line.set_alpha(0.12); line.set_linewidth(1.0)


def rim_axes(fig, axes, color="#b0b0b0"):
    """Static rounded rim around each axis (matches skeleton_filmstrip styling)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = 0.004
    for ax in axes:
        bb = ax.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        fig.patches.append(FancyBboxPatch(
            (bb.x0 - pad, bb.y0 - pad), bb.width + 2 * pad, bb.height + 2 * pad,
            boxstyle="round,pad=0.002,rounding_size=0.01",
            facecolor="none", edgecolor=color, linewidth=0.8,
            transform=fig.transFigure, zorder=10,
        ))


def build_figure(cfg: DecompVizConfig, decomp: Decomposition, frame_imgs: dict,
                 timeline: list[int]):
    n_paths = len(decomp.paths_frames)
    n_cycles = len(decomp.cycles_frames)
    blank = np.full((cfg.render_size, cfg.render_size, 3), 255, dtype=np.uint8)
    c = decomp.coords
    xy0 = c[0]

    with plt.rc_context(_NATURE_RC):
        fig = plt.figure(figsize=(13.5, 8.0))
        fig.patch.set_facecolor("white")
        outer = fig.add_gridspec(3, 1, height_ratios=[1.05, 0.78, 0.78],
                                 hspace=0.24, left=0.02, right=0.98, top=0.92, bottom=0.03)

        # ---- Row 1: video | PCA | skeleton | cycles | paths
        g1 = outer[0].subgridspec(1, 5, wspace=0.18)
        ax_vid = fig.add_subplot(g1[0]); ax_vid.axis("off")
        ax_vid.set_title("Full clip", fontsize=8, fontweight="bold", pad=3)
        im_vid = ax_vid.imshow(frame_imgs[timeline[0]])

        ax_pca = fig.add_subplot(g1[1])
        ax_pca.scatter(c[:, 0], c[:, 1], s=4, c=np.arange(len(c)), cmap="viridis",
                       alpha=0.4, linewidths=0)
        ax_pca.plot(c[:, 0], c[:, 1], color=(0.7, 0.7, 0.7), lw=0.4, alpha=0.5)
        ax_pca.set_title("Clip PCA (2D)", fontsize=8, fontweight="bold", pad=3)
        ax_pca.set_xticks([]); ax_pca.set_yticks([]); ax_pca.set_box_aspect(1)
        for s in ax_pca.spines.values():
            s.set_visible(False)

        ax_sk = fig.add_subplot(g1[2])
        draw_graph_panel(ax_sk, decomp, None, None, "Skeleton")
        ax_cy = fig.add_subplot(g1[3])
        cyc_lines = draw_graph_panel(ax_cy, decomp, decomp.cycles_coords, _CYCLE_COLORS,
                                     f"Cycles (n={n_cycles})", close_loops=True)
        ax_pa = fig.add_subplot(g1[4])
        path_lines = draw_graph_panel(ax_pa, decomp, decomp.paths_coords, _PATH_COLORS,
                                      f"Paths (n={n_paths})")

        # same travelling current-frame dot on every PCA-space panel
        dots = [_frame_dot(a, xy0) for a in (ax_pca, ax_sk, ax_cy, ax_pa)]

        # ---- Row 2: one video per path (start dimmed; lights up when the clip enters it)
        path_ims, path_rects = [], []
        g2 = outer[1].subgridspec(1, max(1, n_paths), wspace=0.12)
        for k in range(n_paths):
            ax = fig.add_subplot(g2[k]); ax.axis("off")
            col = _PATH_COLORS[k % len(_PATH_COLORS)]
            ax.set_title(f"Path {k}  ({len(decomp.paths_frames[k])}f)",
                         fontsize=7, color=col, fontweight="bold", pad=2)
            path_ims.append(ax.imshow(_dim(frame_imgs.get(int(decomp.paths_frames[k][0]), blank))))
            rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                             edgecolor=_INACTIVE_RGB, linewidth=1.2, zorder=15)
            ax.add_patch(rect); path_rects.append(rect)

        # ---- Row 3: one video per cycle
        cyc_ims, cyc_rects = [], []
        g3 = outer[2].subgridspec(1, max(1, n_cycles), wspace=0.12)
        for k in range(n_cycles):
            ax = fig.add_subplot(g3[k]); ax.axis("off")
            col = _CYCLE_COLORS[k % len(_CYCLE_COLORS)]
            ax.set_title(f"Cycle {k}  ({len(decomp.cycles_frames[k])}f)",
                         fontsize=7, color=col, fontweight="bold", pad=2)
            cyc_ims.append(ax.imshow(_dim(frame_imgs.get(int(decomp.cycles_frames[k][0]), blank))))
            rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                             edgecolor=_INACTIVE_RGB, linewidth=1.2, zorder=15)
            ax.add_patch(rect); cyc_rects.append(rect)

        _tag = f"  [{cfg.tag.replace('_', ' ')}]" if cfg.tag else ""
        fig.suptitle(
            f"Clip {decomp.clip_id} ({cfg.feature_space}) — DM primitive decomposition "
            f"({n_cycles} cycles, {n_paths} path corridors); active primitive lights up{_tag}",
            fontsize=11, fontweight="bold",
        )
        rim_axes(fig, [ax_vid, ax_pca, ax_sk, ax_cy, ax_pa])

    return dict(fig=fig, im_vid=im_vid, dots=dots, coords=c, blank=blank,
                path_ims=path_ims, cyc_ims=cyc_ims, path_rects=path_rects,
                cyc_rects=cyc_rects, path_lines=path_lines, cyc_lines=cyc_lines)


def animate(cfg: DecompVizConfig, decomp: Decomposition, frame_imgs: dict,
            timeline: list[int], h: dict, out_mp4: str) -> None:
    """Master clock = the clip frame at natural framerate. The active path/cycle
    (the one whose frame span contains the current frame) lights up."""
    import matplotlib.animation as animation

    fig, blank, coords = h["fig"], h["blank"], h["coords"]
    n_paths, n_cycles = len(decomp.paths_frames), len(decomp.cycles_frames)

    def update(t: int):
        f = timeline[t]
        row = int(np.clip(f - decomp.offset, 0, len(coords) - 1))
        h["im_vid"].set_data(frame_imgs.get(f, blank))
        for d in h["dots"]:
            d.set_offsets([coords[row]])
        for k in range(n_paths):
            P = decomp.paths_frames[k]
            active = len(P) > 0 and int(P[0]) <= f <= int(P[-1])
            img = frame_imgs.get(_nearest_frame(P, f), blank)
            h["path_ims"][k].set_data(img if active else _dim(img))
            line = h["path_lines"][k] if k < len(h["path_lines"]) else None
            _set_active(h["path_rects"][k], line, _PATH_COLORS[k % len(_PATH_COLORS)], active)
        for k in range(n_cycles):
            P = decomp.cycles_frames[k]
            active = len(P) > 0 and int(P[0]) <= f <= int(P[-1])
            img = frame_imgs.get(_nearest_frame(P, f), blank)
            h["cyc_ims"][k].set_data(img if active else _dim(img))
            line = h["cyc_lines"][k] if k < len(h["cyc_lines"]) else None
            _set_active(h["cyc_rects"][k], line, _CYCLE_COLORS[k % len(_CYCLE_COLORS)], active)
        return []

    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    anim = animation.FuncAnimation(fig, update, frames=len(timeline),
                                   interval=1000 / cfg.framerate, blit=False)
    anim.save(out_mp4, writer=animation.FFMpegWriter(fps=cfg.framerate, bitrate=4000), dpi=cfg.dpi)
    plt.close(fig)


def run_clip(cfg: DecompVizConfig, clip_id: int) -> None:
    cfg.clip_id = clip_id
    outdir = os.path.join(cfg.output_dir, cfg.tag) if cfg.tag else cfg.output_dir
    os.makedirs(outdir, exist_ok=True)
    print(f"[decomp] clip {clip_id}, space={cfg.feature_space}, tag={cfg.tag or '-'}")
    decomp = load_decomposition(cfg)
    print(f"[decomp]   {len(decomp.cycles_frames)} cycles "
          f"(lens {[len(x) for x in decomp.cycles_frames]}), "
          f"{len(decomp.paths_frames)} paths "
          f"(lens {[len(x) for x in decomp.paths_frames]})")

    # ---- static decomposition PNG (answers the advisor's "show me the decomposition") ----
    with plt.rc_context(_NATURE_RC):
        figs, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
        draw_graph_panel(axes[0], decomp, None, None, "Skeleton")
        draw_graph_panel(axes[1], decomp, decomp.cycles_coords, _CYCLE_COLORS,
                         f"Cycles (min cycle basis, n={len(decomp.cycles_frames)})",
                         close_loops=True)
        draw_graph_panel(axes[2], decomp, decomp.paths_coords, _PATH_COLORS,
                         f"Paths (branch corridors, n={len(decomp.paths_frames)})")
        _tag = f"  [{cfg.tag.replace('_', ' ')}]" if cfg.tag else ""
        figs.suptitle(f"Clip {clip_id} ({cfg.feature_space}) — DM decomposition in PCA-2D{_tag}",
                      fontsize=10, fontweight="bold")
        figs.tight_layout(rect=[0, 0, 1, 0.94])
        png = os.path.join(outdir, f"clip{clip_id}_{cfg.feature_space}_decomposition.png")
        figs.savefig(png, dpi=cfg.dpi, facecolor="white", bbox_inches="tight")
        plt.close(figs)
    print(f"[decomp]   saved static decomposition: {png}")

    if not cfg.make_video:
        return

    # ---- render every needed frame ONCE, then index into the shared buffer ----
    qpos = load_rollout_from_source(clip_id, cfg.data_h5)["qpos"]
    n = min(decomp.n_frames, qpos.shape[0])
    timeline = list(range(0, n, cfg.stride))
    needed = set(timeline)
    for fr in decomp.cycles_frames + decomp.paths_frames:
        needed.update(int(x) for x in fr if x < n)
    needed = sorted(needed)
    print(f"[decomp]   rendering {len(needed)} frames at {cfg.render_size}px ...")
    frame_imgs = render_tracking_frames(qpos, cfg.xml_path, needed, cfg.render_size)

    h = build_figure(cfg, decomp, frame_imgs, timeline)
    mp4 = os.path.join(outdir, f"clip{clip_id}_{cfg.feature_space}_decomposition.mp4")
    animate(cfg, decomp, frame_imgs, timeline, h, mp4)
    print(f"[decomp]   saved video: {mp4}")


def run(cfg: DecompVizConfig) -> None:
    clips = cfg.clip_ids if cfg.clip_ids else [cfg.clip_id]
    for cid in clips:
        run_clip(cfg, cid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demos/demo_primitive_decomposition.yaml")
    ap.add_argument("--clip-id", type=int, default=None)
    ap.add_argument("--clip-ids", type=int, nargs="+", default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--feature-space", type=str, default=None, choices=["intention", "qpos", "ekp"])
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.feature_space is not None:
        cfg.feature_space = args.feature_space
    if args.clip_ids is not None:
        cfg.clip_ids = args.clip_ids
    if args.clip_id is not None:
        cfg.clip_ids = [args.clip_id]
    if args.stride is not None:
        cfg.stride = args.stride
    if args.tag is not None:
        cfg.tag = args.tag
    if args.no_video:
        cfg.make_video = False
    run(cfg)


if __name__ == "__main__":
    main()
