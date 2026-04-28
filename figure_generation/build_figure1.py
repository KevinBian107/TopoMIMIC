#!/usr/bin/env python3
"""Build paper/figures/figure1.png.

Four-panel layout:
  Row 1
    (A) Two views of the multi-basin DM landscape (3D oblique + top-down
        projection) re-rendered statically (not upscaled from a gif) so the
        in-panel labels stay publication-readable. Generated from
        ``demos.demo_saddle`` with the ``exp3_multi_basin`` configuration; one
        DM-cycle ($H_1$, red) and one DM-path ($H_0$, blue) are overlaid on
        the inverse-density surface, with stars marking basin minima.
    (B) DM cancellation-tree schematic (loaded from
        ``external_data/dm_tree.png``): two trees $T_u$, $T_{u'}$ joined by a
        critical edge $e$ that the cancellation procedure cannot pair away.
  Row 2  --  both panels are sourced from
             ``external_data/mimice_pipeline.pdf``, rasterized via PyMuPDF and
             split horizontally at ``MIMIC_SPLIT_FRAC``.
    (C) MIMIC-MJX general-space demo: 3D Keypoints, Joint Angles, and
        Action+Latent stages (left side of the source PDF).
    (D) MIMIC-MJX encoder-decoder (right side of the source PDF):
        $s^{g}_{t:T} \\to q_\\phi(z_t \\mid s^{g}_t) \\to z_t \\to
        p_\\theta(a_t \\mid z_t, s^{p}_t) \\to a_t$.

External assets (``external_data/dm_tree.png`` and
``external_data/mimice_pipeline.pdf``) live in this directory's
``external_data/`` subfolder, which is gitignored; populate it locally before
running the script. Output goes to ``paper/figures/figure1.png``.
"""
from __future__ import annotations

import os
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "PCD-Graph-Recon-DM"))

from demos.demo_saddle import (  # noqa: E402
    Config,
    density_basin_surface_multi,
    extract_paths_between_minima,
    generate_one_way_arc,
    generate_round_trip_2routes,
    run_dm_analysis,
    _plot_surface,
    _plot_grey_path,
    _plot_cycle_overlay,
    _plot_path_overlay,
    _plot_minima,
)

EXTERNAL_DATA = Path(__file__).resolve().parent / "external_data"
DM_TREE = EXTERNAL_DATA / "dm_tree.png"
MIMIC_PIPELINE_PDF = EXTERNAL_DATA / "mimice_pipeline.pdf"
OUT_PATH = REPO / "paper/figures/figure1.png"

# Fraction of mimice_pipeline.pdf width that belongs to (C) (general-space
# demo, left side); the remainder is (D) (encoder-decoder schematic).
MIMIC_SPLIT_FRAC = 0.30

# Single 3D rotation angle for the middle panel (chosen to expose both the
# cycle's two arcs and the third-basin path).
MID_AZIM = 310


# Larger-than-Nature rcParams; used to ensure every glyph in the (A) row
# stays readable when figure1.png is dropped into a NeurIPS column.
_BIG_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
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


def build_exp3():
    """Re-build the multi-basin experiment data exactly as demo_saddle does."""
    cfg = Config()
    n_steps = cfg.path.n_steps
    noise = cfg.path.noise_std
    res = 80

    min_A = np.array([-2.0, 0.0])
    min_B = np.array([1.0, 1.5])
    min_C = np.array([1.0, -1.5])
    rng3 = np.random.default_rng(cfg.path.seed + 200)
    n_sub = n_steps // 2

    path_AB = generate_round_trip_2routes(
        min_A, min_B, density_basin_surface_multi, n_sub, noise, rng3, y_offset=0.7,
    )
    path_AC = generate_one_way_arc(
        min_A, min_C, density_basin_surface_multi, n_sub, noise,
        np.random.default_rng(cfg.path.seed + 201), y_offset=0.6,
    )
    multi_path = np.vstack([path_AB, path_AC])

    x3 = np.linspace(-3.5, 3.5, res)
    y3 = np.linspace(-3.5, 3.5, res)
    X3, Y3 = np.meshgrid(x3, y3)
    Z3 = density_basin_surface_multi(X3, Y3)

    out_dir = REPO / "outputs/demos/illustrative/saddle_dm_demo"
    cycles3, G3 = run_dm_analysis(multi_path, cfg.dm, str(out_dir), "exp3_multi_basin")
    exp3_dir = os.path.abspath(out_dir / "exp3_multi_basin")
    dm_paths3, ec3, m1_3, m2_3 = extract_paths_between_minima(G3, exp3_dir)

    seg_colors = [
        (0, len(path_AB), "#e41a1c"),
        (len(path_AB), len(path_AB) + len(path_AC), "#377eb8"),
    ]
    return dict(
        X=X3, Y=Y3, Z=Z3, path=multi_path, cycles=cycles3,
        dm_paths=dm_paths3, ec=ec3, min1=m1_3, min2=m2_3,
        seg_colors=seg_colors, vis=cfg.visualization,
    )


def render_panel_a(exp: dict, target_w_px: int, dpi: int = 200) -> Image.Image:
    """Two views of the multi-basin landscape (3D oblique + top-down xy plane),
    rendered with publication-readable text."""
    fig_w_in = target_w_px / dpi
    fig_h_in = fig_w_in * 0.55

    with plt.rc_context(_BIG_RC):
        fig, axes = plt.subplots(
            1, 2, figsize=(fig_w_in, fig_h_in),
            subplot_kw={"projection": "3d"},
            gridspec_kw={"wspace": 0.10},
        )
        ax_3d, ax_top = axes

        for ax in (ax_3d, ax_top):
            _plot_surface(ax, exp["X"], exp["Y"], exp["Z"], exp["vis"])
            _plot_grey_path(ax, exp["path"], linewidth=1.4, alpha=0.55,
                            segment_colors=exp["seg_colors"])
            if exp["cycles"]:
                _plot_cycle_overlay(ax, exp["path"], exp["cycles"][0], exp["vis"])
            for i, p in enumerate(exp["dm_paths"]):
                _plot_path_overlay(ax, exp["path"], p, exp["vis"],
                                   label="DM Path" if i == 0 else None)
            _plot_minima(ax, exp["path"], [exp["min1"], exp["min2"]])

        ax_3d.view_init(elev=30, azim=MID_AZIM)
        ax_3d.set_xlabel("Feature x", labelpad=4)
        ax_3d.set_ylabel("Feature y", labelpad=4)
        ax_3d.set_zlabel("Inv. density", labelpad=2)
        ax_3d.tick_params(pad=1)

        # Top-down xy view: look straight down the z-axis so the path and
        # cycle sit on the xy plane.
        ax_top.view_init(elev=89.99, azim=-90)
        ax_top.set_xlabel("Feature x", labelpad=4)
        ax_top.set_ylabel("Feature y", labelpad=4)
        ax_top.set_zticklabels([])
        ax_top.set_zlabel("")
        ax_top.tick_params(pad=1)

        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", markersize=10,
                       color="#e41a1c", markeredgecolor="white", label="DM Cycle ($H_1$)"),
            plt.Line2D([0], [0], marker="D", linestyle="", markersize=9,
                       color="#377eb8", markeredgecolor="white", label="DM Path ($H_0$)"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=2,
                   frameon=False, fontsize=18, bbox_to_anchor=(0.5, 1.02))

        fig.subplots_adjust(top=0.88, bottom=0.05, left=0.02, right=0.98)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def fit_width(img: Image.Image, target_w: int) -> Image.Image:
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


def render_mimic_pipeline_split(
    pdf_path: Path, dpi: int = 300, split_frac: float = MIMIC_SPLIT_FRAC,
) -> tuple[Image.Image, Image.Image]:
    """Rasterize ``mimice_pipeline.pdf`` and split it into (C) and (D).

    The PDF lays the general-space demo (3D Keypoints / Joint Angles /
    Action+Latent thumbnails) on the left and the encoder-decoder schematic
    on the right. We split horizontally at ``split_frac`` of the width.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()

    w, h = img.size
    split_x = int(w * split_frac)
    left = img.crop((0, 0, split_x, h))
    right = img.crop((split_x, 0, w, h))
    return left, right


def add_label(img: Image.Image, label: str, pad: int = 56,
              font_size: int = 44, x_offset: int = 30) -> Image.Image:
    out = Image.new("RGB", (img.size[0], img.size[1] + pad), color=(255, 255, 255))
    out.paste(img, (0, pad))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((x_offset, 6), label, fill=(0, 0, 0), font=font)
    return out


def main() -> None:
    print("Building exp3 multi-basin data...")
    exp = build_exp3()

    target_w = 3200
    a_w = int(target_w * 0.65)
    b_w = target_w - a_w

    print("Rendering panel A (3 rotation frames)...")
    panel_a = render_panel_a(exp, target_w_px=a_w)
    panel_a = fit_width(panel_a, a_w)

    panel_b = Image.open(DM_TREE).convert("RGB")
    panel_b = fit_width(panel_b, b_w)

    top_h = max(panel_a.size[1], panel_b.size[1])
    panel_a = pad_to_height(panel_a, top_h)
    panel_b = pad_to_height(panel_b, top_h)

    panel_a_lab = add_label(panel_a, "(A)")
    panel_b_lab = add_label(panel_b, "(B)")

    top_row = Image.new("RGB", (target_w, panel_a_lab.size[1]), color=(255, 255, 255))
    top_row.paste(panel_a_lab, (0, 0))
    top_row.paste(panel_b_lab, (panel_a_lab.size[0], 0))

    print("Rendering panels C/D from mimice_pipeline.pdf...")
    panel_c_raw, panel_d_raw = render_mimic_pipeline_split(MIMIC_PIPELINE_PDF)

    # Split the bottom row in proportion to the source widths so neither
    # panel is squeezed.
    src_w_total = panel_c_raw.size[0] + panel_d_raw.size[0]
    c_w = int(round(target_w * panel_c_raw.size[0] / src_w_total))
    d_w = target_w - c_w
    panel_c = fit_width(panel_c_raw, c_w)
    panel_d = fit_width(panel_d_raw, d_w)
    bottom_h = max(panel_c.size[1], panel_d.size[1])
    panel_c = pad_to_height(panel_c, bottom_h)
    panel_d = pad_to_height(panel_d, bottom_h)

    panel_c_lab = add_label(panel_c, "(C)")
    # Push (D) inward so it sits over the encoder-decoder schematic, not over
    # the whitespace that the source PDF leaves on the left of that panel.
    panel_d_lab = add_label(panel_d, "(D)", x_offset=int(panel_d.size[0] * 0.18))

    bottom_row = Image.new(
        "RGB",
        (target_w, max(panel_c_lab.size[1], panel_d_lab.size[1])),
        color=(255, 255, 255),
    )
    bottom_row.paste(panel_c_lab, (0, 0))
    bottom_row.paste(panel_d_lab, (panel_c_lab.size[0], 0))

    gap = 24
    out = Image.new(
        "RGB",
        (target_w, top_row.size[1] + gap + bottom_row.size[1]),
        color=(255, 255, 255),
    )
    out.paste(top_row, (0, 0))
    out.paste(bottom_row, (0, top_row.size[1] + gap))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.save(OUT_PATH, dpi=(200, 200))
    print(f"wrote {OUT_PATH}  (size={out.size})")


if __name__ == "__main__":
    main()
