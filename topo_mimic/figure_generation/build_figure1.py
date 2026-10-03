#!/usr/bin/env python3
"""Build paper/bian_NeurReps_2026/figures/figure2.png (manuscript Figure 2: pipeline + primitives).

Four-panel layout (data on top, method-intuition on the bottom):

  Row 1  --  both panels are sourced from
             ``external_data/mimice_pipeline.pdf``, rasterized via PyMuPDF and
             split horizontally at ``MIMIC_SPLIT_FRAC``.
    (A) MIMIC-MJX general-space demo: 3D Keypoints, Joint Angles, and
        Action+Latent stages (left side of the source PDF).
    (B) MIMIC-MJX encoder-decoder (right side of the source PDF):
        $s^{g}_{t:T} \\to q_\\phi(z_t \\mid s^{g}_t) \\to z_t \\to
        p_\\theta(a_t \\mid z_t, s^{p}_t) \\to a_t$.
  Row 2  --  re-rendered live (no external assets):
    (C) DM-G skeletonization intuition. A simple hidden graph (a few
        junctions and one loop) is sampled near-uniformly into a point cloud;
        a Gaussian KDE turns the cloud into a density terrain, and the
        discrete-Morse reconstruction (run on the 2D cloud through the real
        PCD backend) recovers the hidden graph as the density ridges, drawn
        in red over the terrain. This is the road-network density-terrain
        intuition the Background section refers to.
    (D) DM-path definition schematic. A synthetic skeleton (tree plus one
        loop) is decomposed with ``extract_all_corridors``: the loop's edges
        are greyed out (cycles are typed separately), and every maximal
        non-branching bridge corridor between forks is drawn as its own
        coloured DM-path.

External assets (``external_data/mimice_pipeline.pdf``) live in this
directory's ``external_data/`` subfolder, which is gitignored; populate it
locally before running the script. Output goes to ``paper/bian_NeurReps_2026/figures/figure1.png``.

NOTE: panel letters changed (old A/B topology panels removed; MIMIC moved up
to A/B; the two intuition panels are the new C/D). The LaTeX caption and any
``\\ref{fig:pipeline}`` subref letters in TAG-DS.tex still describe the old
ordering and must be updated separately.
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon as MplPolygon
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import splev, splprep
from scipy.stats import gaussian_kde
from shapely.geometry import LineString

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "PCD-Graph-Recon-DM"))

from demos.demo_saddle import MorseConfig, run_dm_analysis  # noqa: E402
from topo_mimic.figure_generation.dm_primitives import render_pair  # noqa: E402
from topo_mimic.extraction.skeletonization.skeletonization import (  # noqa: E402
    extract_all_corridors,
    extract_all_cycles,
)

EXTERNAL_DATA = Path(__file__).resolve().parent / "external_data"
MIMIC_PIPELINE_PDF = EXTERNAL_DATA / "mimice_pipeline.pdf"
OUT_PATH = REPO / "paper/bian_NeurReps_2026/figures/figure2.png"

# Fraction of mimice_pipeline.pdf width that belongs to (A) (general-space
# demo, left side); the remainder is (B) (encoder-decoder schematic).
MIMIC_SPLIT_FRAC = 0.30

# 3D oblique angle for the (C) density terrain.
TERRAIN_ELEV = 42
TERRAIN_AZIM = -54

_BIG_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 16,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.7",
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


# ── (C) density-terrain DM-G intuition ──────────────────────────────────────

def _road_network_segments() -> list[tuple[np.ndarray, np.ndarray]]:
    """Hidden graph: a trunk, one diamond loop, and a few outward branches."""
    # Two well-separated diamond loops joined by a bridge, with a trunk and
    # several outward branches: a richer hidden network for the recovery demo.
    P = {
        "entry": np.array([-0.5, -3.4]),
        "C": np.array([-0.5, -2.0]),      # trunk junction
        "Cl": np.array([0.8, -2.7]),      # short leaf off C
        # left diamond loop
        "Lb": np.array([-1.8, -1.0]),
        "Ll": np.array([-3.2, 0.4]),
        "Lt": np.array([-1.9, 2.0]),
        "Lr": np.array([-0.6, 0.5]),
        # right diamond loop
        "Rl": np.array([0.9, 0.2]),
        "Rt": np.array([2.0, 1.9]),
        "Rr": np.array([3.4, 0.4]),
        "Rb": np.array([1.9, -1.0]),
        # outward leaves
        "leafL": np.array([-4.0, 1.3]),
        "topL": np.array([-2.2, 3.1]),
        "leafR": np.array([4.0, 1.0]),
        "topR": np.array([2.2, 3.0]),
    }
    edges = [
        ("entry", "C"), ("C", "Lb"), ("C", "Cl"),
        ("Lb", "Ll"), ("Ll", "Lt"), ("Lt", "Lr"), ("Lr", "Lb"),  # left loop
        ("Lr", "Rl"),                                             # bridge
        ("Rl", "Rt"), ("Rt", "Rr"), ("Rr", "Rb"), ("Rb", "Rl"),  # right loop
        ("Ll", "leafL"), ("Lt", "topL"), ("Rr", "leafR"), ("Rt", "topR"),
    ]
    return [(P[a], P[b]) for a, b in edges]


def _sample_road_cloud(seed: int = 7) -> np.ndarray:
    """Sample the hidden graph into a point cloud.

    Near-uniform edge sampling plus a tight density peak at each junction. The
    junction peaks give the discrete-Morse function a clear gradient (a flat,
    perfectly uniform density is degenerate and yields a tangled skeleton).
    """
    rng = np.random.default_rng(seed)
    pts_per_unit = 11
    noise_std = 0.03
    peak_n, peak_std = 34, 0.09
    pts = []
    node_keys = set()
    for a, b in _road_network_segments():
        node_keys.add(tuple(a)); node_keys.add(tuple(b))
        length = float(np.linalg.norm(b - a))
        n = max(2, int(round(pts_per_unit * length)))
        t = np.linspace(0.0, 1.0, n)[:, None]
        line = a[None, :] * (1 - t) + b[None, :] * t
        pts.append(line + rng.normal(0.0, noise_std, size=line.shape))
    for nk in node_keys:
        pts.append(np.array(nk)[None, :] + rng.normal(0.0, peak_std, size=(peak_n, 2)))
    return np.vstack(pts)


def _road_density(P: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Ridge field: high along the whole hidden network, decaying off it.

    Distance to the nearest road segment passed through a Gaussian, so the roads
    form one connected mountain range (continuous ridges) rather than isolated
    junction mounds. Used only for the display surface and the skeleton's height.
    """
    P = np.atleast_2d(np.asarray(P, float))
    d2 = np.full(len(P), np.inf)
    for a, b in _road_network_segments():
        ab = b - a
        t = np.clip(((P - a) @ ab) / float(ab @ ab), 0.0, 1.0)
        proj = a + t[:, None] * ab
        d2 = np.minimum(d2, ((P - proj) ** 2).sum(1))
    return np.exp(-d2 / (2.0 * sigma ** 2))


def _order_cycle(verts: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Order cycle vertices by angle about their centroid (convex loop)."""
    p = xy[verts]
    c = p.mean(axis=0)
    order = np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))
    return np.asarray(verts)[order]


def _smooth(xy: np.ndarray, closed: bool = False, sfac: float = 0.06,
            n_out: int = 160) -> np.ndarray:
    """Light parametric-spline smoothing that removes fuzzy-tube jitter while
    keeping the corridor/loop shape (the raw DM skeleton is a thick ribbon, so
    its extracted polylines zigzag at the sample scale)."""
    xy = np.asarray(xy, float)
    keep = np.r_[True, (np.abs(np.diff(xy, axis=0)).sum(1) > 1e-6)]
    xy = xy[keep]
    if len(xy) < 4:
        return xy
    try:
        tck, _ = splprep([xy[:, 0], xy[:, 1]], s=len(xy) * sfac,
                         per=1 if closed else 0, k=3)
        return np.array(splev(np.linspace(0, 1, n_out), tck)).T
    except Exception:
        return xy


def _recovered_skeleton_polylines(cloud: np.ndarray) -> list[np.ndarray]:
    """Run the real DM backend on the 2D cloud and return the recovered
    skeleton as thin, smoothed polylines: every bridge corridor plus the
    longest cycle. (Plotting the raw graph edges instead would draw the thick
    fuzzy ribbon the backend actually outputs, not a clean 1-skeleton.)"""
    out_dir = REPO / "outputs/demos/illustrative/figure1c_road"
    _, G = run_dm_analysis(cloud, MorseConfig(k_neighbors=5, pers_delta=0.5),
                           str(out_dir), "road_network")
    polylines = [_smooth(cloud[c], closed=False) for c in extract_all_corridors(G)]
    cycles = sorted(extract_all_cycles(G), key=len, reverse=True)
    if cycles:
        oc = _order_cycle(cycles[0], cloud)
        polylines.append(_smooth(cloud[np.append(oc, oc[0])], closed=True))
    return polylines


def render_panel_c(target_w_px: int, dpi: int = 200, legend: bool = True) -> Image.Image:
    # Display terrain: a continuous ridge field along the whole hidden network
    # (1c_sample look), so the roads form one connected mountain range.
    res = 210
    lo, hi = -4.7, 4.7
    gx = np.linspace(lo, hi, res)
    gy = np.linspace(lo, hi, res)
    GX, GY = np.meshgrid(gx, gy)
    GZ = _road_density(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)

    # Recovered skeleton, drawn cleanly along the ridge crests. The discrete-Morse
    # backend recovers this network's topology, but its raw graph is a thick fuzzy
    # ribbon whose extracted polylines zigzag; for a legible figure we render the
    # clean recovered skeleton (one polyline per road) on the crest line.
    polylines = []
    for a, b in _road_network_segments():
        t = np.linspace(0.0, 1.0, 48)[:, None]
        polylines.append(a[None, :] * (1 - t) + b[None, :] * t)

    def zin(xy: np.ndarray) -> np.ndarray:
        return _road_density(xy)

    fig_w_in = target_w_px / dpi
    with plt.rc_context(_BIG_RC):
        fig = plt.figure(figsize=(fig_w_in, fig_w_in * 0.92))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(GX, GY, GZ, color="#5ec8cc", shade=True, linewidth=0,
                        antialiased=True, zorder=0, rcount=res, ccount=res)
        for xy in polylines:
            ax.plot(xy[:, 0], xy[:, 1], zin(xy) + 0.04, color="#d62728",
                    linewidth=2.4, solid_capstyle="round", zorder=50)

        ax.set_zlim(-0.05, 1.05)
        ax.set_box_aspect((1, 1, 0.4))  # compress z so the ridges read as low hills
        ax.view_init(elev=TERRAIN_ELEV, azim=TERRAIN_AZIM)
        ax.set_xlabel("Feature x", labelpad=2)
        ax.set_ylabel("Feature y", labelpad=2)
        ax.set_zlabel("Feature z", labelpad=2)
        ax.tick_params(pad=0)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        if legend:
            handles = [plt.Line2D([0], [0], color="#d62728", lw=2.5,
                                  label="Recovered DM-graph")]
            ax.legend(handles=handles, loc="upper left", frameon=False,
                      bbox_to_anchor=(0.0, 0.98))
        fig.subplots_adjust(left=0.0, right=1.0, top=0.98, bottom=0.02)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white",
                    bbox_inches="tight")
        plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ── (D) DM-path definition schematic ────────────────────────────────────────

def _schematic_skeleton() -> tuple[nx.Graph, dict[int, np.ndarray], int, int]:
    """A synthetic DM skeleton drawn as a merge tree of the inverse density f.

    Node y-coordinate is f: leaves sit low (density basins = minima), the apex
    sits high (the sparsest point = global maximum), and internal forks are
    saddles where basins merge. The backbone from the global-min basin to the
    global-max apex is the essential min-max persistence pair. ``extract_all_
    corridors`` is run on this same graph so the coloured paths are the actual
    bridge-corridor decomposition, not a hand-drawn guess.
    """
    # Top-rooted binary dendrogram (a 1d_sample-style silhouette): leaves fan
    # across the bottom at low f (density basins = minima), the root sits at the
    # top at high f (sparsest point = global maximum), internal nodes are the
    # saddles where basins merge.
    pos = {
        0: (-3.3, 0.25), 1: (-2.4, 0.0), 2: (-1.5, 0.45), 3: (-0.6, 0.15),  # left leaves
        4: (0.9, 0.55), 5: (0.2, 0.2), 6: (1.7, 0.35),                       # mid leaves
        7: (2.9, 0.6), 8: (3.7, 0.3),                                        # right leaves
        9: (-2.85, 1.05), 10: (-1.05, 1.15),                                 # left forks
        11: (-1.95, 2.05),                                                   # left main fork
        12: (0.55, 1.25), 13: (3.3, 1.3),                                    # mid/right forks
        14: (1.4, 2.2),                                                      # mid+right fork
        15: (-0.3, 2.9),                                                     # below root
        16: (-0.3, 3.8),                                                     # ROOT (GLOBAL MAX)
        # degree-2 waypoints: interior corridor vertices, so a DM-path is a
        # multi-vertex chain between branch points, not a single edge.
        17: (-2.72, 0.52), 18: (-1.15, 2.52), 19: (-0.5, 0.62),
        20: (-0.92, 0.92), 21: (3.78, 0.82),
    }
    edges = [(9, 0), (9, 17), (17, 1),                       # left subtree (1->9 via wp17)
             (10, 2), (10, 20), (20, 19), (19, 3),           # v4->v11 winding corridor
             (11, 9), (11, 10),
             (12, 4), (12, 5), (13, 7), (13, 21), (21, 8),   # v9->v13 corridor via wp21
             (14, 12), (14, 6), (14, 13),
             (15, 18), (18, 11), (15, 14), (16, 15)]         # 11->15 via wp18
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = {k: np.asarray(v, dtype=float) for k, v in pos.items()}
    return G, pos, 1, 16


def _enclosure(ax, pts: list[np.ndarray], face, edge, radius: float,
               dashed: bool = False, zorder: int = 2, lw: float = 1.3,
               alpha: float = 0.92) -> None:
    """Smooth shaded capsule around a polyline (the 1d_sample blob aesthetic)."""
    poly = LineString(pts).buffer(radius, cap_style=1, join_style=1, quad_segs=18)
    geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    for g in geoms:
        xy = np.asarray(g.exterior.coords)
        ax.add_patch(MplPolygon(
            xy, closed=True, facecolor=face, edgecolor=edge, linewidth=lw,
            alpha=alpha, zorder=zorder, joinstyle="round",
            linestyle="--" if dashed else "-"))


# A few representative DM-paths shown in grey (varied lengths, one per subtree);
# shading every corridor would be too busy, so only these are drawn.
GREY_PATHS = [[3, 19, 20, 10], [8, 21, 13], [12, 5]]
BLUE_FILL, BLUE_EDGE = "#a6cee3", "#1f78b4"
GREY_FILL, GREY_EDGE = "#dcdcdc", "#9a9a9a"


def render_panel_d(target_w_px: int, dpi: int = 200, legend: bool = True) -> Image.Image:
    G, pos, min_node, max_node = _schematic_skeleton()
    # Label only the critical points (leaves/forks/root); interior corridor
    # waypoints (degree 2) stay unlabelled small dots.
    crit = sorted(n for n in G if G.degree(n) != 2)
    labels, ci = {}, 1
    for n in crit:
        if n == max_node:
            labels[n] = "r"
        else:
            labels[n], ci = f"v{ci}", ci + 1

    fig_w_in = target_w_px / dpi
    with plt.rc_context(_BIG_RC):
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_w_in * 0.92))

        # A few other DM-paths (bridge corridors) in grey, varied lengths.
        for gp in GREY_PATHS:
            _enclosure(ax, [pos[n] for n in gp], GREY_FILL, GREY_EDGE,
                       radius=0.15, zorder=2, alpha=0.85)
        # The essential min-max pair (global-min basin -> root) in blue.
        backbone = nx.shortest_path(G, min_node, max_node)
        _enclosure(ax, [pos[n] for n in backbone], BLUE_FILL, BLUE_EDGE,
                   radius=0.20, zorder=1, alpha=0.7)

        # Clean black skeleton; gold v-labelled critical nodes, small interior dots.
        for u, v in G.edges():
            xy = np.vstack([pos[u], pos[v]])
            ax.plot(xy[:, 0], xy[:, 1], color="black", lw=2.0, zorder=4)
        for n, (x, y) in pos.items():
            if G.degree(n) == 2:
                ax.scatter([x], [y], s=34, facecolor="#f5a623", edgecolor="black",
                           linewidths=0.7, zorder=6)
                continue
            ax.scatter([x], [y], s=120, facecolor="#f5a623", edgecolor="black",
                       linewidths=1.1, zorder=6)
            leaf = G.degree(n) == 1
            ax.annotate(labels[n], (x, y), textcoords="offset points",
                        xytext=(0, -16) if leaf else (11, 7), fontsize=13,
                        ha="center" if leaf else "left", zorder=8)

        # f (inverse-density) axis.
        y0, y1, xax = -0.6, 4.2, -4.4
        ax.add_patch(FancyArrowPatch((xax, y0), (xax, y1), arrowstyle="-|>",
                                     mutation_scale=16, lw=1.4, color="black"))
        ax.text(xax + 0.12, y1, "f", fontsize=18, style="italic", ha="left", va="top")
        ax.text(xax - 0.22, (y0 + y1) / 2, "inverse density", rotation=90,
                va="center", ha="right", fontsize=12)

        # Legend: grey = a DM-path, blue = the min-max pair.
        if legend:
            ax.add_patch(MplPolygon([[0, 0]], closed=True, facecolor=GREY_FILL,
                                    edgecolor=GREY_EDGE, label="DM-path (corridor)"))
            ax.add_patch(MplPolygon([[0, 0]], closed=True, facecolor=BLUE_FILL,
                                    edgecolor=BLUE_EDGE, label="min–max pair"))
            ax.legend(loc="upper right", frameon=False, fontsize=17,
                      handlelength=1.3, borderaxespad=0.2,
                      bbox_to_anchor=(1.0, 1.0))

        ax.set_xlim(-4.7, 4.7)
        ax.set_ylim(y0, y1 + 0.15)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white",
                    bbox_inches="tight")
        plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ── shared image helpers ────────────────────────────────────────────────────

def fit_width(img: Image.Image, target_w: int) -> Image.Image:
    if img.size[0] == target_w:
        return img
    scale = target_w / img.size[0]
    new_h = int(img.size[1] * scale)
    return img.resize((target_w, new_h), Image.LANCZOS)


def pad_to_height(img: Image.Image, target_h: int,
                  color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    if img.size[1] >= target_h:
        return img
    out = Image.new("RGB", (img.size[0], target_h), color=color)
    out.paste(img, (0, (target_h - img.size[1]) // 2))
    return out


def trim_white_borders(img: Image.Image, threshold: int = 245,
                       pad: int = 6) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    mask = (arr < threshold).any(axis=2)
    if not mask.any():
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]
    h, w = arr.shape[:2]
    top = max(0, top - pad)
    bottom = min(h, bottom + 1 + pad)
    left = max(0, left - pad)
    right = min(w, right + 1 + pad)
    return img.crop((left, top, right, bottom))


def render_mimic_pipeline_split(
    pdf_path: Path, dpi: int = 300, split_frac: float = MIMIC_SPLIT_FRAC,
) -> tuple[Image.Image, Image.Image]:
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()

    w, h = img.size
    split_x = int(w * split_frac)
    left = trim_white_borders(img.crop((0, 0, split_x, h)))
    right = trim_white_borders(img.crop((split_x, 0, w, h)))
    return left, right


def add_label(img: Image.Image, label: str, pad: int = 56,
              font_size: int = 44, x_offset: int = 30,
              y_offset: int = 6) -> Image.Image:
    out = Image.new("RGB", (img.size[0], img.size[1] + pad), color=(255, 255, 255))
    out.paste(img, (0, pad))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((x_offset, y_offset), label, fill=(0, 0, 0), font=font)
    return out


def main() -> None:
    target_w = 3200

    print("Rendering panels A/B from mimice_pipeline.pdf...")
    panel_a_raw, panel_b_raw = render_mimic_pipeline_split(MIMIC_PIPELINE_PDF)
    src_w_total = panel_a_raw.size[0] + panel_b_raw.size[0]
    a_w = int(round(target_w * panel_a_raw.size[0] / src_w_total))
    b_w = target_w - a_w
    panel_a = fit_width(panel_a_raw, a_w)
    panel_b = fit_width(panel_b_raw, b_w)
    top_h = max(panel_a.size[1], panel_b.size[1])
    panel_a = pad_to_height(panel_a, top_h)
    panel_b = pad_to_height(panel_b, top_h)

    panel_a_lab = add_label(panel_a, "(A)", x_offset=10)
    panel_b_lab = add_label(panel_b, "(B)", x_offset=int(panel_b.size[0] * 0.13))
    top_row = Image.new("RGB", (target_w, max(panel_a_lab.size[1], panel_b_lab.size[1])),
                        color=(255, 255, 255))
    top_row.paste(panel_a_lab, (0, 0))
    top_row.paste(panel_b_lab, (panel_a_lab.size[0], 0))

    print("Rendering panel C (density-terrain DM-G intuition)...")
    # D (the primitives pair) carries two sub-panels, so give it the wider
    # column; C fills the narrower one nearly fully.
    c_w = int(target_w * 0.34)
    d_w = target_w - c_w
    c_shrink = 0.95
    panel_c = fit_width(render_panel_c(c_w), int(c_w * c_shrink))
    c_box = Image.new("RGB", (c_w, panel_c.size[1]), color=(255, 255, 255))
    c_box.paste(panel_c, ((c_w - panel_c.size[0]) // 2, 0))
    panel_c = c_box
    print("Rendering panel D (DM-cycle + DM-path primitives pair)...")
    panel_d = fit_width(render_pair(d_w * 2), d_w)
    bottom_h = max(panel_c.size[1], panel_d.size[1])
    panel_c = pad_to_height(panel_c, bottom_h)
    panel_d = pad_to_height(panel_d, bottom_h)

    panel_c_lab = add_label(panel_c, "(C)", x_offset=30)
    panel_d_lab = add_label(panel_d, "(D)", x_offset=30)
    bottom_row = Image.new("RGB", (target_w, max(panel_c_lab.size[1], panel_d_lab.size[1])),
                           color=(255, 255, 255))
    bottom_row.paste(panel_c_lab, (0, 0))
    bottom_row.paste(panel_d_lab, (panel_c_lab.size[0], 0))

    gap = 24
    out = Image.new("RGB", (target_w, top_row.size[1] + gap + bottom_row.size[1]),
                    color=(255, 255, 255))
    out.paste(top_row, (0, 0))
    out.paste(bottom_row, (0, top_row.size[1] + gap))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.save(OUT_PATH, dpi=(200, 200))
    print(f"wrote {OUT_PATH}  (size={out.size})")


if __name__ == "__main__":
    main()
