#!/usr/bin/env python3
"""Build paper/bian_NeurReps_2026/appendix_figures/appendix_stationary.png.

Three panels, in one row at a single shared height:
  (A) Root displacement per frame ($|\\Delta x|, |\\Delta y|, |\\Delta z|$),
      with-cycle vs no-cycle clips ($n=50$ each).
  (B) Pool size $n=5$ bar: moving vs stationary mean longest DM-loop length.
  (C) Stationary-vs-moving line plot, sweeping pool size $n\\in\\{2,3,4,5\\}$
      with shaded SEM.

Panels A and B inherit their data from the old main Figure 3 (root
displacement bars and pool=5 stationary-vs-moving bar); C is the original
line-sweep panel. A filmstrip grid of no-cycle clips was panel D; it restated
the same claim as A-C in a weaker form and was dropped.
"""
from __future__ import annotations

import pickle as pkl
import sys
import yaml
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from topo_mimic.analysis.method_analysis.property.stationary import (  # noqa: E402
    compute_body_variance,
    compute_delta_xyz_stats,
    find_cycle_and_no_cycle_clips,
)

# Panel width used to be inherited from the (now dropped) filmstrip grid.
PANEL_ROW_WIDTH = 3204
OUT_PATH = REPO / "paper/bian_NeurReps_2026/appendix_figures/appendix_stationary.png"
NEURREPS_OUT_PATH = REPO / "paper/bian_NeurReps_2026/appendix_figures/appendix_stationary.png"
H5_PATH = REPO / "data/rodent_ar.h5"
DM_RESULTS_DIR = REPO / "outputs/extraction/dm_pipeline_intention"
BATCH_PKL = REPO / "outputs/analysis/method_analysis/batch_incompressibility/all_results.pkl"
BATCH_RESULTS_YAML = REPO / "outputs/analysis/method_analysis/batch_incompressibility/results.yaml"

# Figure 3 palette: orange = active (with-cycles / moving), blue = quiet (no-cycles / stationary).
WITH_CYCLES_COLOR = "#E6A382"
NO_CYCLES_COLOR   = "#7E9FCB"

BODY_PARTS = [
    "torso", "pelvis", "upper_leg_L", "lower_leg_L", "foot_L",
    "upper_leg_R", "lower_leg_R", "foot_R", "skull",
    "upper_arm_L", "upper_arm_R",
]


def _figure_to_image(fig: plt.Figure, dpi: int) -> Image.Image:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _sample_cycle_split(n_take: int = 50) -> tuple[list[int], list[int], int]:
    cycle_clips, no_cycle_clips = find_cycle_and_no_cycle_clips(
        str(DM_RESULTS_DIR), num_clips=500
    )
    np.random.seed(42)
    n_take = min(n_take, len(cycle_clips), len(no_cycle_clips))
    sampled_cycle = np.random.choice(cycle_clips, n_take, replace=False).tolist()
    sampled_no_cycle = np.random.choice(no_cycle_clips, n_take, replace=False).tolist()
    return sampled_cycle, sampled_no_cycle, n_take


def render_root_displacement_png(target_w_px: int, target_h_px: int | None = None,
                                  dpi: int = 200) -> Image.Image:
    sampled_cycle, sampled_no_cycle, n_take = _sample_cycle_split(50)
    cy_mean, cy_std = compute_delta_xyz_stats(sampled_cycle, str(H5_PATH), n_take)
    nc_mean, nc_std = compute_delta_xyz_stats(sampled_no_cycle, str(H5_PATH), n_take)

    fig_w_in = target_w_px / dpi
    fig_h_in = (target_h_px / dpi) if target_h_px is not None else fig_w_in * 0.95
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)

    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2, cy_mean, w, yerr=cy_std,
           color=WITH_CYCLES_COLOR, edgecolor="none", capsize=3.0,
           error_kw={"linewidth": 0.7, "capthick": 0.7},
           label=f"With cycles (n={n_take})")
    ax.bar(x + w / 2, nc_mean, w, yerr=nc_std,
           color=NO_CYCLES_COLOR, edgecolor="none", capsize=3.0,
           error_kw={"linewidth": 0.7, "capthick": 0.7},
           label=f"No cycles (n={n_take})")

    ax.set_xticks(x)
    ax.set_xticklabels([r"$|\Delta x|$", r"$|\Delta y|$", r"$|\Delta z|$"],
                       fontsize=12)
    ax.set_xlabel("Root displacement axis", fontsize=14)
    ax.set_ylabel("Mean |Δ| per frame", fontsize=14)
    ax.tick_params(direction="out", length=4, width=0.8, labelsize=12)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(11)
    ax.legend(fontsize=11, frameon=True, framealpha=0.95,
              edgecolor="#cccccc", loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return _figure_to_image(fig, dpi)


def render_pool5_bar_png(target_w_px: int, target_h_px: int | None = None,
                          dpi: int = 200) -> Image.Image:
    with open(BATCH_RESULTS_YAML, "r") as f:
        data = yaml.safe_load(f)
    sv = data["stationary_vs_moving"]["n_5"]
    moving_mean, moving_std = sv["moving_total_mean"], sv["moving_total_std"]
    stat_mean, stat_std = sv["stationary_mean"], sv["stationary_std"]

    fig_w_in = target_w_px / dpi
    fig_h_in = (target_h_px / dpi) if target_h_px is not None else fig_w_in * 0.95
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)

    x = np.arange(2)
    ax.bar(
        x,
        [moving_mean, stat_mean],
        yerr=[moving_std, stat_std],
        color=[WITH_CYCLES_COLOR, NO_CYCLES_COLOR],
        width=0.55, capsize=4.0,
        error_kw={"linewidth": 0.8, "capthick": 0.8},
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["moving\n(n=5)", "stationary\n(n=5)"], fontsize=12)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xlabel("Pool composition", fontsize=14)
    ax.set_ylabel("Longest DM-loop (frames)", fontsize=14)
    ax.tick_params(direction="out", length=4, width=0.8, labelsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return _figure_to_image(fig, dpi)


def render_stationary_vs_moving_line_png(target_w_px: int,
                                          target_h_px: int | None = None,
                                          dpi: int = 200) -> Image.Image:
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
    fig_h_in = (target_h_px / dpi) if target_h_px is not None else fig_w_in * 0.95
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
    return _figure_to_image(fig, dpi)


def render_body_variance_png(target_w_px: int, dpi: int = 200) -> Image.Image:
    sampled_cycle, sampled_no_cycle, n_take = _sample_cycle_split(50)

    cy_var, body_parts = compute_body_variance(
        sampled_cycle, str(H5_PATH), BODY_PARTS, max_clips=n_take
    )
    nc_var, _ = compute_body_variance(
        sampled_no_cycle, str(H5_PATH), BODY_PARTS, max_clips=n_take
    )

    fig_w_in = target_w_px / dpi
    fig_h_in = fig_w_in * 0.40
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
    return _figure_to_image(fig, dpi)


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
    target_w = PANEL_ROW_WIDTH

    # A (root displacement) + B (pool=5 bar) + C (n in {2..5} line).
    # All three rendered at the same absolute pixel height so axes line up.
    panel_a_w = target_w // 3
    panel_b_w = target_w // 4
    panel_c_w = target_w - panel_a_w - panel_b_w
    # Pin the shared height to the smallest panel's natural square-ish ratio
    # so B (the narrowest) is not stretched.
    panel_h = int(panel_b_w * 0.95)

    a_img = render_root_displacement_png(target_w_px=panel_a_w, target_h_px=panel_h)
    b_img = render_pool5_bar_png(target_w_px=panel_b_w, target_h_px=panel_h)
    c_img = render_stationary_vs_moving_line_png(target_w_px=panel_c_w, target_h_px=panel_h)

    a_img = fit_width(a_img, panel_a_w)
    b_img = fit_width(b_img, panel_b_w)
    c_img = fit_width(c_img, panel_c_w)
    top_h = max(a_img.size[1], b_img.size[1], c_img.size[1])
    a_img = pad_to_height(a_img, top_h)
    b_img = pad_to_height(b_img, top_h)
    c_img = pad_to_height(c_img, top_h)

    a_labeled = add_label(a_img, "(A)", pad=70, font_size=60)
    b_labeled = add_label(b_img, "(B)", pad=70, font_size=60)
    c_labeled = add_label(c_img, "(C)", pad=70, font_size=60)

    out = Image.new("RGB", (target_w, a_labeled.size[1]), color=(255, 255, 255))
    out.paste(a_labeled, (0, 0))
    out.paste(b_labeled, (a_labeled.size[0], 0))
    out.paste(c_labeled, (a_labeled.size[0] + b_labeled.size[0], 0))

    for path in (OUT_PATH, NEURREPS_OUT_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        out.save(path, dpi=(200, 200))
        print(f"wrote {path}  (size={out.size})")


if __name__ == "__main__":
    main()
