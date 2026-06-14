#!/usr/bin/env python3
"""Schematic DM-primitive renderers (addresses Yusu's notes).

ONE DM skeleton, laid out so its tree part IS a split tree (Yusu's convention:
root = global MIN at the bottom, peaks = maxima at the top, internal nodes are
UP-FORK saddles), with a loop attached. The two primitives are parts of this one
skeleton, drawn in the same style:
  DM-cycle = the loop;
  DM-path  = the split-tree part, with every persistence pair marked as a
             leaf->saddle path (+ the essential global-max<->global-min path).
So skeleton = DM-cycle (loop) + DM-path (split tree).
"""
from __future__ import annotations

from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image
from shapely.geometry import LineString

GOLD = "#f5a623"
ORD_COLORS = [("#ccebc5", "#4daf4a"), ("#fbb4ae", "#e41a1c")]
ESS_COLOR = ("#a6cee3", "#1f78b4")


def _fig_to_pil(fig, dpi: int = 200) -> Image.Image:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def dm_skeleton():
    """The skeleton = split tree (tree edges) + a loop (loop edges).

    pos[node]=(x, f) with the tree height-ordered (root min at bottom, peaks at
    top). pairs lists the DM-paths on the split tree (leaf->saddle, essential)."""
    pos = {
        "r": (0.0, 0.0),      # root = global min
        "s1": (0.0, 1.0),     # saddle (up-fork)
        "s2": (0.95, 2.0),    # saddle (up-fork)
        "ma": (-1.7, 2.9),    # peak (max)
        "mb": (0.15, 3.45),   # peak (max) = global max
        "mc": (1.7, 2.8),     # peak (max); loop attaches here
        "p0": (2.5, 3.25), "p1": (3.05, 2.5), "p2": (2.4, 1.95),  # loop
    }
    tree_edges = [("r", "s1"), ("s1", "ma"), ("s1", "s2"), ("s2", "mb"), ("s2", "mc")]
    loop_edges = [("mc", "p0"), ("p0", "p1"), ("p1", "p2"), ("p2", "mc")]
    # elder rule: s2 kills mc(2.8)<mb -> (mc,s2); s1 kills ma(2.9)<mb -> (ma,s1);
    # mb (global max) -> root essential.
    pairs = [
        (["mc", "s2"], "ordinary"),
        (["ma", "s1"], "ordinary"),
        (["mb", "s2", "s1", "r"], "essential"),
    ]
    return pos, tree_edges, loop_edges, pairs


def _enc(ax, pts, face, edge, r, alpha):
    poly = LineString(pts).buffer(r, cap_style=1, join_style=1, quad_segs=18)
    geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    for g in geoms:
        xy = np.asarray(g.exterior.coords)
        ax.add_patch(MplPolygon(xy, closed=True, facecolor=face, edgecolor=edge,
                                lw=1.3, alpha=alpha, joinstyle="round", zorder=2))


def _draw_skeleton(ax, mode: str = "plain", labels: bool = True,
                   node_s: int = 95, fs: float = 1.0) -> None:
    """Draw the skeleton (one part highlighted) into a given axes. node_s is the
    node marker size; fs scales fonts, line widths, and label offsets."""
    pos, tree_edges, loop_edges, pairs = dm_skeleton()

    def edge(u, v, color, lw, z=4):
        xy = np.array([pos[u], pos[v]])
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw * fs, solid_capstyle="round", zorder=z)

    if mode == "plain":
        for u, v in tree_edges + loop_edges:
            edge(u, v, "black", 1.8)
    elif mode == "cycle":
        for u, v in tree_edges:
            edge(u, v, "#cfcfcf", 1.6, z=2)
        for u, v in loop_edges:
            edge(u, v, "#d62728", 5.5, z=3)
            edge(u, v, "black", 1.8, z=4)
    elif mode == "path":
        for u, v in loop_edges:
            edge(u, v, "#cfcfcf", 1.6, z=2)
        oi = 0
        for seq, kind in pairs:
            pts = [pos[n] for n in seq]
            if kind == "essential":
                _enc(ax, pts, ESS_COLOR[0], ESS_COLOR[1], 0.22, 0.65)
            else:
                _enc(ax, pts, ORD_COLORS[oi][0], ORD_COLORS[oi][1], 0.18, 0.82)
                oi += 1
        for u, v in tree_edges:
            edge(u, v, "black", 1.7, z=4)
        if labels:
            ax.add_patch(FancyArrowPatch((-2.5, 0.0), (-2.5, 3.55), arrowstyle="-|>",
                                         mutation_scale=14 * fs, lw=1.3 * fs, color="black"))
            ax.text(-2.42, 3.55, "$f$", fontsize=15 * fs, style="italic", va="top")
            ax.text(-2.72, 1.8, "density", rotation=90, va="center", ha="right", fontsize=11 * fs)
            ax.annotate("peaks (maxima)", pos["mb"], textcoords="offset points",
                        xytext=(-6, 15 * fs), fontsize=11 * fs, ha="center")
            ax.annotate("saddle", pos["s2"], textcoords="offset points",
                        xytext=(9, -6), fontsize=10 * fs, color="#555")
            ax.annotate("root (global min)", pos["r"], textcoords="offset points",
                        xytext=(-15 * fs, -4), fontsize=10 * fs, ha="right", color="#555")

    for _n, (x, y) in pos.items():
        ax.scatter([x], [y], s=node_s, facecolor=GOLD, edgecolor="black",
                   linewidths=1.0, zorder=6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.03)


def _shared_legend_handles():
    """(cycle, dm-path-tuple, essential) handles + labels for the shared legend."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    cyc = Line2D([0], [0], color="#d62728", lw=5)
    g = Patch(facecolor=ORD_COLORS[0][0], edgecolor=ORD_COLORS[0][1])
    r = Patch(facecolor=ORD_COLORS[1][0], edgecolor=ORD_COLORS[1][1])
    e = Patch(facecolor=ESS_COLOR[0], edgecolor=ESS_COLOR[1])
    handles = [cyc, (g, r), e]
    names = ["DM-cycle ($H_1$ loop)", "DM-paths (peak, saddle)",
             "essential (max $\\leftrightarrow$ min)"]
    return handles, names


def render_skeleton(target_w_px: int = 1000, dpi: int = 200, mode: str = "plain",
                    legend: bool = False, labels: bool = True) -> Image.Image:
    fig_w = target_w_px / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.95))
    _draw_skeleton(ax, mode, labels)
    if legend and mode == "path":
        from matplotlib.legend_handler import HandlerTuple
        h, n = _shared_legend_handles()
        ax.legend(h[1:], n[1:], handler_map={tuple: HandlerTuple(ndivide=None)},
                  loc="lower right", frameon=False, fontsize=10.5, bbox_to_anchor=(1.0, 0.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return _fig_to_pil(fig, dpi)


def render_dm_cycle(target_w_px: int = 1000, dpi: int = 200) -> Image.Image:
    return render_skeleton(target_w_px, dpi, mode="cycle")


def render_dm_path_2d(target_w_px: int = 1000, dpi: int = 200,
                      legend: bool = False, labels: bool = True) -> Image.Image:
    return render_skeleton(target_w_px, dpi, mode="path", legend=legend, labels=labels)


def render_pair(target_w_px: int = 2200, dpi: int = 200) -> Image.Image:
    """DM-cycle and DM-path side by side with a shared legend (paper Figure 2)."""
    from matplotlib.legend_handler import HandlerTuple
    fig = plt.figure(figsize=(target_w_px / dpi, target_w_px / dpi * 0.46))
    gs = fig.add_gridspec(1, 2, wspace=0.0)
    ax0 = fig.add_subplot(gs[0])
    _draw_skeleton(ax0, "cycle", labels=False, node_s=230, fs=2.4)
    # large title pad keeps the "DM-path"/"DM-cycle" titles clear of the
    # "peaks (maxima)" label that sits just above the raised drawing.
    ax0.set_title("DM-cycle", fontsize=34, pad=70)
    ax1 = fig.add_subplot(gs[1])
    _draw_skeleton(ax1, "path", labels=True, node_s=230, fs=2.4)
    ax1.set_title("DM-path", fontsize=34, pad=70)
    # raise both equal-aspect drawings toward the top of panel D: anchor them to
    # the top of their cells and reserve a wide bottom band so the cells (and
    # the drawings in them) sit high above the legend. Scoped to render_pair so
    # the flow figure is unchanged.
    ax0.set_anchor("N")
    ax1.set_anchor("N")

    h, n = _shared_legend_handles()
    fig.legend(h, n, handler_map={tuple: HandlerTuple(ndivide=None)},
               loc="lower center", ncol=3, frameon=False, fontsize=28,
               bbox_to_anchor=(0.5, 0.03))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.26)
    return _fig_to_pil(fig, dpi)


def main() -> None:
    from pathlib import Path
    out = Path(__file__).resolve().parents[1] / "paper/figures"
    out.mkdir(parents=True, exist_ok=True)
    render_skeleton().save(out / "dm_skeleton.png")
    render_dm_cycle().save(out / "dm_cycle.png")
    render_dm_path_2d(legend=True).save(out / "dm_path.png")
    render_pair().save(out / "dm_primitives_pair.png")
    print(f"wrote {out}/dm_skeleton.png, dm_cycle.png, dm_path.png, dm_primitives_pair.png")


if __name__ == "__main__":
    main()
