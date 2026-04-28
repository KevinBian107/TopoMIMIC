#!/usr/bin/env python3
"""Build paper/appendix_figures/appendix_stationary.png.

Panel A: stationary-vs-moving line plot (DM-loop length grows with pooled
         moving clips, stays flat for stationary-only pools), re-rendered with
         the appendix palette to match panels B/C.
Panel B: body-part variance bar chart (orange = with-cycles, blue = no-cycles).
Panel C: 8-clip filmstrip grid of representative no-cycle clips (kept from the
         existing stationary_clips run).

Top row holds (A) and (B) side-by-side; (C) sits beneath at full width. Big
"(A)" / "(B)" / "(C)" panel labels sit in the top-left of each.
"""
from __future__ import annotations

import pickle as pkl
import sys
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from topo_mimic.analysis.method_analysis.property.stationary import (  # noqa: E402
    compute_body_variance,
    find_cycle_and_no_cycle_clips,
)

FILMSTRIP_SRC = REPO / "outputs/analysis/method_analysis/stationary_clips/stationary_filmstrip_grid.png"
OUT_PATH = REPO / "paper/appendix_figures/appendix_stationary.png"
H5_PATH = REPO / "data/rodent_ar.h5"
DM_RESULTS_DIR = REPO / "outputs/extraction/dm_pipeline_intention"
BATCH_PKL = REPO / "outputs/analysis/method_analysis/batch_incompressibility/all_results.pkl"

# Figure 3 palette: orange = active (with-cycles / moving), blue = quiet (no-cycles / stationary).
WITH_CYCLES_COLOR = "#E6A382"
NO_CYCLES_COLOR   = "#7E9FCB"

BODY_PARTS = [
    "torso", "pelvis", "upper_leg_L", "lower_leg_L", "foot_L",
    "upper_leg_R", "lower_leg_R", "foot_R", "skull",
    "upper_arm_L", "upper_arm_R",
]


def render_stationary_vs_moving_line_png(target_w_px: int, dpi: int = 200) -> Image.Image:
    with open(BATCH_PKL, "rb") as f:
        cached = pkl.load(f)
    results_by_n = cached["stationary_vs_moving"]

    n_values = sorted(results_by_n.keys())
    stat_m, stat_s, mov_m, mov_s = [], [], [], []
    for n in n_values:
        results = results_by_n[n]
        sf = [r["stationary_frames"] for r in results]
        mf = [r["moving_frames_total"] for r in results]
        stat_m.append(np.mean(sf)); stat_s.append(np.std(sf) / np.sqrt(len(sf)))
        mov_m.append(np.mean(mf)); mov_s.append(np.std(mf) / np.sqrt(len(mf)))
    stat_m, stat_s = np.array(stat_m), np.array(stat_s)
    mov_m, mov_s = np.array(mov_m), np.array(mov_s)

    fig_w_in = target_w_px / dpi
    fig_h_in = fig_w_in * 0.95
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)

    ax.fill_between(n_values, mov_m - mov_s, mov_m + mov_s,
                    color=WITH_CYCLES_COLOR, alpha=0.30, linewidth=0)
    ax.plot(n_values, mov_m, "s-", color=WITH_CYCLES_COLOR, linewidth=1.6,
            markersize=7, markeredgecolor="white", markeredgewidth=0.8,
            label="Moving (sum)", zorder=3)

    ax.fill_between(n_values, stat_m - stat_s, stat_m + stat_s,
                    color=NO_CYCLES_COLOR, alpha=0.30, linewidth=0)
    ax.plot(n_values, stat_m, "o-", color=NO_CYCLES_COLOR, linewidth=1.6,
            markersize=7, markeredgecolor="white", markeredgewidth=0.8,
            label="Stationary", zorder=3)

    ax.set_xlabel("Number of trajectories (n)", fontsize=14)
    ax.set_ylabel("Frames in longest DM-loop", fontsize=14)
    ax.tick_params(direction="out", length=4, width=0.8, labelsize=12)
    ax.set_xticks(n_values)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95,
              edgecolor="#cccccc", fontsize=11, handlelength=1.5)

    n_trials = len(results_by_n[n_values[0]])
    ax.text(0.95, 0.95, f"n = {n_trials} trials per point",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontstyle="italic", color="#555555")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_body_variance_png(target_w_px: int = 3204, dpi: int = 200) -> Image.Image:
    cycle_clips, no_cycle_clips = find_cycle_and_no_cycle_clips(
        str(DM_RESULTS_DIR), num_clips=500
    )
    np.random.seed(42)
    n_take = min(50, len(cycle_clips), len(no_cycle_clips))
    sampled_cycle = np.random.choice(cycle_clips, n_take, replace=False).tolist()
    sampled_no_cycle = np.random.choice(no_cycle_clips, n_take, replace=False).tolist()

    cy_var, body_parts = compute_body_variance(
        sampled_cycle, str(H5_PATH), BODY_PARTS, max_clips=n_take
    )
    nc_var, _ = compute_body_variance(
        sampled_no_cycle, str(H5_PATH), BODY_PARTS, max_clips=n_take
    )

    fig_w_in = target_w_px / dpi
    fig_h_in = fig_w_in * 0.55
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)

    x = np.arange(len(body_parts))
    w = 0.40
    ax.bar(x - w/2, cy_var, w, color=WITH_CYCLES_COLOR, edgecolor="none",
           label=f"With cycles (n={n_take})")
    ax.bar(x + w/2, nc_var, w, color=NO_CYCLES_COLOR, edgecolor="none",
           label=f"No cycles (n={n_take})")

    body_labels = [bp.replace("_", "\n") for bp in body_parts]
    ax.set_xticks(x)
    ax.set_xticklabels(body_labels, fontsize=12, rotation=45, ha="right")
    ax.set_xlabel("Body part", fontsize=14)
    ax.set_ylabel("Variance (rad²)", fontsize=14)
    ax.tick_params(direction="out", length=4, width=0.8, labelsize=12)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(11)
    ax.legend(fontsize=12, frameon=True, framealpha=0.95, edgecolor="#cccccc")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()

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


def add_label(img: Image.Image, label: str, pad: int = 70,
              font_size: int = 60) -> Image.Image:
    out = Image.new("RGB", (img.size[0], img.size[1] + pad), color=(255, 255, 255))
    out.paste(img, (0, pad))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((40, 8), label, fill=(0, 0, 0), font=font)
    return out


def main() -> None:
    film = Image.open(FILMSTRIP_SRC).convert("RGB")
    target_w = film.size[0]

    line_w_target = int(target_w * 0.36)
    var_w_target = target_w - line_w_target

    line_img = render_stationary_vs_moving_line_png(target_w_px=line_w_target)
    var_img = render_body_variance_png(target_w_px=var_w_target)

    line_img = fit_width(line_img, line_w_target)
    var_img = fit_width(var_img, var_w_target)

    top_h = max(line_img.size[1], var_img.size[1])
    line_img = pad_to_height(line_img, top_h)
    var_img = pad_to_height(var_img, top_h)

    line_labeled = add_label(line_img, "(A)", pad=70, font_size=60)
    var_labeled = add_label(var_img, "(B)", pad=70, font_size=60)
    film_labeled = add_label(film, "(C)", pad=70, font_size=60)

    top_row = Image.new("RGB", (target_w, line_labeled.size[1]),
                        color=(255, 255, 255))
    top_row.paste(line_labeled, (0, 0))
    top_row.paste(var_labeled, (line_labeled.size[0], 0))

    gap = 16
    out = Image.new(
        "RGB",
        (target_w, top_row.size[1] + gap + film_labeled.size[1]),
        color=(255, 255, 255),
    )
    out.paste(top_row, (0, 0))
    out.paste(film_labeled, (0, top_row.size[1] + gap))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.save(OUT_PATH, dpi=(200, 200))
    print(f"wrote {OUT_PATH}  (size={out.size})")


if __name__ == "__main__":
    main()
