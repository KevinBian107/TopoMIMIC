#!/usr/bin/env python3
"""Build the new 4-panel Figure 3.

Layout (aspect ratio preserved, NO sub-panel titles, panel labels A/B/C/D):

    Row 1: (A) body-part variance bar chart (cycle vs no-cycle, sampled clips)
           (B) pool-size bar (n=5): moving vs stationary mean DM-loop length
           (C) cycle transition-signature heatmap (per-clip cycles, intention DM)
    Row 2: (D) PCA + t-SNE + UMAP of the 5-clip union, behavior-coloured;
               non-DM frames dimmed.

Outputs paper/figures/figure3.png.
"""
from __future__ import annotations

import sys
import yaml
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from topo_mimic.analysis.method_analysis.property.stationary import (  # noqa: E402
    compute_delta_xyz_stats,
    find_cycle_and_no_cycle_clips,
)

CLIPS = [250, 253, 279, 300, 599]
STATIONARY_CLIP = 279

H5_PATH = REPO / "data/rodent_ar.h5"
LABELS_CSV = REPO / "behavior_labels/rodent_labels.csv"
BATCH_RESULTS = REPO / "outputs/analysis/method_analysis/batch_incompressibility/results.yaml"
INCOMP_DIR = REPO / "outputs/analysis/method_analysis/incompressibility"
SIG_CSV = REPO / "outputs/analysis/method_analysis/cycle_transition_signature/cycle_signatures.csv"
ASYM_DIR = REPO / "outputs/analysis/method_analysis/behavioral_asymmetry"
DM_RESULTS_DIR = REPO / "outputs/extraction/dm_pipeline_intention"
OUT_PATH = REPO / "paper/figures/figure3.png"

BEHAVIOR_COLORS = {
    "Walk":     "#4C72B0",
    "Immobile": "#C0C0C0",
    "Turn":     "#55A868",
    "Rear":     "#DD8452",
}

ACTIVE_COLOR  = "#E6A382"   # orange: with-cycle / asymmetric / moving (always on left)
QUIET_COLOR   = "#7E9FCB"   # blue:   no-cycle  / symmetric  / stationary (always on right)

CYCLE_COLOR    = ACTIVE_COLOR
NO_CYCLE_COLOR = QUIET_COLOR

POOL_MOVING_COLOR     = ACTIVE_COLOR
POOL_STATIONARY_COLOR = QUIET_COLOR

UNORDERED_PAIRS = [
    ("Immobile", "Walk"),
    ("Immobile", "Turn"),
    ("Immobile", "Rear"),
    ("Walk", "Turn"),
    ("Walk", "Rear"),
    ("Turn", "Rear"),
]

BODY_PARTS = [
    "torso", "pelvis", "upper_leg_L", "lower_leg_L", "foot_L",
    "upper_leg_R", "lower_leg_R", "foot_R", "skull",
    "upper_arm_L", "upper_arm_R",
]


def load_qpos() -> np.ndarray:
    with h5py.File(H5_PATH, "r") as h:
        return h["qpos_rollout"][:]


def load_labels() -> pd.DataFrame:
    return pd.read_csv(LABELS_CSV)


def per_frame_labels(labels_df: pd.DataFrame, clip_id: int, n_frames: int) -> np.ndarray:
    sub = labels_df[(labels_df["Video"] == f"clip_{clip_id}.mp4") &
                    (labels_df["Track"] == "behavior")]
    out = np.array(["Immobile"] * n_frames, dtype=object)
    for _, r in sub.iterrows():
        s, e = int(r["Start_Frame"]), int(r["End_Frame"])
        out[s:min(e + 1, n_frames)] = r["Label"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Panel A: root motion (|Δx|, |Δy|, |Δz|), cycle vs no-cycle
# ─────────────────────────────────────────────────────────────────────────────

def render_panel_a(ax: plt.Axes) -> None:
    cycle_clips, no_cycle_clips = find_cycle_and_no_cycle_clips(
        str(DM_RESULTS_DIR), num_clips=500
    )
    np.random.seed(42)
    n_take = min(50, len(cycle_clips), len(no_cycle_clips))
    sampled_cycle = np.random.choice(cycle_clips, n_take, replace=False).tolist()
    sampled_no_cycle = np.random.choice(no_cycle_clips, n_take, replace=False).tolist()

    cy_mean, cy_std = compute_delta_xyz_stats(sampled_cycle, str(H5_PATH), n_take)
    nc_mean, nc_std = compute_delta_xyz_stats(sampled_no_cycle, str(H5_PATH), n_take)

    x = np.arange(3)
    w = 0.36
    ax.bar(x - w/2, cy_mean, w, yerr=cy_std,
           color=CYCLE_COLOR, edgecolor="none", capsize=2.5,
           error_kw={"linewidth": 0.6, "capthick": 0.6},
           label=f"With cycles (n={n_take})")
    ax.bar(x + w/2, nc_mean, w, yerr=nc_std,
           color=NO_CYCLE_COLOR, edgecolor="none", capsize=2.5,
           error_kw={"linewidth": 0.6, "capthick": 0.6},
           label=f"No cycles (n={n_take})")

    ax.set_xticks(x)
    ax.set_xticklabels([r"$|\Delta x|$", r"$|\Delta y|$", r"$|\Delta z|$"], fontsize=8)
    ax.set_xlabel("root displacement axis", fontsize=8)
    ax.set_ylabel("mean |Δ| per frame", fontsize=8)
    ax.tick_params(direction="out", length=2.0, width=0.4, labelsize=7)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(7)
    ax.legend(fontsize=6, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 1.0),
              handlelength=1.2, handletextpad=0.4, borderpad=0.0,
              borderaxespad=0.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Panel B: pool=5 only, moving vs stationary mean DM-loop length
# ─────────────────────────────────────────────────────────────────────────────

def render_panel_b(ax: plt.Axes) -> None:
    with open(BATCH_RESULTS, "r") as f:
        data = yaml.safe_load(f)
    sv = data["stationary_vs_moving"]["n_5"]
    moving_mean, moving_std = sv["moving_total_mean"], sv["moving_total_std"]
    stat_mean, stat_std = sv["stationary_mean"], sv["stationary_std"]

    x = np.arange(2)
    ax.bar(
        x,
        [moving_mean, stat_mean],
        yerr=[moving_std, stat_std],
        color=[POOL_MOVING_COLOR, POOL_STATIONARY_COLOR],
        width=0.55, capsize=3.0,
        error_kw={"linewidth": 0.6, "capthick": 0.6},
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["moving\n(n=5)", "stationary\n(n=5)"], fontsize=7)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xlabel("pool composition", fontsize=8)
    ax.set_ylabel("longest DM-loop (frames)", fontsize=8)
    ax.tick_params(direction="out", length=2.0, width=0.4, labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Panel C: cycle transition-signature heatmap
# ─────────────────────────────────────────────────────────────────────────────

def render_panel_c(ax: plt.Axes, cax: plt.Axes | None = None,
                   cbar_side: str = "left") -> None:
    if not SIG_CSV.exists():
        ax.text(0.5, 0.5, "cycle_transition_signature missing",
                ha="center", va="center", transform=ax.transAxes, fontsize=7)
        ax.axis("off")
        return
    df = pd.read_csv(SIG_CSV)
    edge_cols = [f"edge_{p[0]}_{p[1]}" for p in UNORDERED_PAIRS]
    X = df[edge_cols].to_numpy(dtype=float)
    rowsum = X.sum(axis=1, keepdims=True)
    keep = (rowsum.squeeze() > 0)
    X = X[keep]
    rowsum = rowsum[keep]
    Xn = X / rowsum
    sub = df[keep].reset_index(drop=True)

    grouped, group_sizes, group_keys = [], [], []
    for pair in UNORDERED_PAIRS:
        key = f"{pair[0]}_{pair[1]}"
        sel = (sub["dominant_pair"] == key).to_numpy()
        if not sel.any():
            continue
        block = Xn[sel]
        order = np.argsort(-block[:, UNORDERED_PAIRS.index(pair)])
        grouped.append(block[order])
        group_sizes.append(block.shape[0])
        group_keys.append(pair)
    M = np.concatenate(grouped, axis=0)

    im = ax.imshow(M, aspect="auto", cmap="magma_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(UNORDERED_PAIRS)))
    ax.set_xticklabels([f"{p[0][:3]}↔{p[1][:3]}" for p in UNORDERED_PAIRS],
                       rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks([])
    ax.tick_params(direction="out", length=2.0, width=0.4, labelsize=6)
    # Group separators removed — keep the heatmap continuous.
    yc = 0
    n_total = M.shape[0]
    for pair, n in zip(group_keys, group_sizes):
        y_axes = 1.0 - (yc + n / 2) / n_total
        ax.text(-0.06, y_axes,
                f"{pair[0][:3]}↔{pair[1][:3]}\n(n={n})",
                ha="right", va="center", fontsize=6, color="#333333",
                transform=ax.transAxes)
        yc += n
    ax.set_xlabel(f"DM-cycles (n={M.shape[0]})", fontsize=7, labelpad=4)
    if cax is not None:
        # Force colorbar height to match the heatmap's square box.
        ax_pos = ax.get_position()
        side = min(ax_pos.height, ax_pos.width)
        bottom = ax_pos.y0 + (ax_pos.height - side) / 2
        cax_pos = cax.get_position()
        new_x0 = ax_pos.x1 + 0.005
        cax.set_position([new_x0, bottom, cax_pos.width, side])
        cbar = ax.figure.colorbar(im, cax=cax)
        cbar.set_label("transition-edge fraction", fontsize=7)
        cbar.ax.tick_params(labelsize=6, length=2.0, width=0.4)
        if cbar_side == "left":
            cax.yaxis.set_ticks_position("left")
            cax.yaxis.set_label_position("left")


# ─────────────────────────────────────────────────────────────────────────────
# Panel D: PCA + t-SNE + UMAP of 5-clip union
# ─────────────────────────────────────────────────────────────────────────────

DM_TAKENS_M = 3
DM_TAKENS_TAU = 6
DM_TAKENS_OFFSET = (DM_TAKENS_M - 1) * DM_TAKENS_TAU


def _unused_collect_union_points_and_dm_mask() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qpos = load_qpos()
    labels_df = load_labels()
    qpos_concat: list[np.ndarray] = []
    label_concat: list[np.ndarray] = []
    clip_concat: list[np.ndarray] = []
    n_frames = qpos.shape[1]
    for cid in CLIPS:
        if cid >= qpos.shape[0]:
            continue
        qpos_concat.append(qpos[cid, :, 7:])
        label_concat.append(per_frame_labels(labels_df, cid, n_frames))
        clip_concat.append(np.full(n_frames, cid, dtype=int))
    points = np.concatenate(qpos_concat, axis=0)
    labels_arr = np.concatenate(label_concat, axis=0)
    clip_arr = np.concatenate(clip_concat, axis=0)

    dm_pkl = INCOMP_DIR / "dm_results.pkl"
    dm_mask = np.zeros(points.shape[0], dtype=bool)
    if not dm_pkl.exists():
        return points, labels_arr, dm_mask, clip_arr

    with open(dm_pkl, "rb") as f:
        dm_obj = pickle.load(f)
    cycles = dm_obj.get("cycles", []) if isinstance(dm_obj, dict) else (getattr(dm_obj, "cycles", []) or [])
    clip_ranges_raw = dm_obj.get("clip_ranges", []) if isinstance(dm_obj, dict) else getattr(dm_obj, "clip_ranges", [])
    clip_ids_raw = dm_obj.get("clip_ids", []) if isinstance(dm_obj, dict) else getattr(dm_obj, "clip_ids", [])

    # Build (vertex_lo, vertex_hi, clip_id) from dm clip_ranges + clip_ids.
    intervals = []
    for cid, rng in zip(clip_ids_raw, clip_ranges_raw):
        intervals.append((int(rng[0]), int(rng[1]), int(cid)))

    cid_to_concat_offset = {cid: i * n_frames for i, cid in enumerate(CLIPS)}

    def _vertex_to_concat(v: int) -> tuple[int, int] | None:
        for lo, hi, cid in intervals:
            if lo <= v < hi:
                if cid not in cid_to_concat_offset:
                    return None
                orig_frame = DM_TAKENS_OFFSET + (v - lo)
                if 0 <= orig_frame < n_frames:
                    return (cid, cid_to_concat_offset[cid] + orig_frame)
                return None
        return None

    for cyc in cycles:
        clips_touched = set()
        concat_idxs = []
        for v in cyc:
            r = _vertex_to_concat(int(v))
            if r is None:
                continue
            cid, idx = r
            clips_touched.add(cid)
            concat_idxs.append(idx)
        if len(clips_touched) < 2:
            continue  # drop single-clip (stationary-only) cycles
        for idx in concat_idxs:
            dm_mask[idx] = True
    return points, labels_arr, dm_mask, clip_arr


def clip_dominant_transition(label_seq: np.ndarray) -> str:
    counts = {p: 0 for p in UNORDERED_PAIRS}
    for i in range(len(label_seq) - 1):
        a, b = label_seq[i], label_seq[i + 1]
        if a == b:
            continue
        pair = tuple(sorted((a, b)))
        if pair in counts:
            counts[pair] += 1
    best_pair, best_n = max(counts.items(), key=lambda kv: kv[1])
    if best_n == 0:
        return "stationary"
    return f"{best_pair[0][:3]}↔{best_pair[1][:3]}"


CLIP_PALETTE = {
    250: "#4C72B0",
    253: "#DD8452",
    279: "#7F7F7F",
    300: "#55A868",
    599: "#8172B3",
}


def render_panel_d(ax: plt.Axes) -> None:
    points, labels_arr, dm_mask, clip_arr = collect_union_points_and_dm_mask()
    print(f"  Panel D: {points.shape[0]} frames, {dm_mask.sum()} DM-marked")

    pca50 = PCA(n_components=min(50, points.shape[1])).fit_transform(points)
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(pca50)

    clip_label: dict[int, str] = {}
    for cid in CLIPS:
        sel = clip_arr == cid
        if not sel.any():
            continue
        clip_label[cid] = clip_dominant_transition(labels_arr[sel])

    for cid in CLIPS:
        sel = clip_arr == cid
        if not sel.any():
            continue
        ax.scatter(emb[sel, 0], emb[sel, 1], s=4,
                   c=CLIP_PALETTE.get(cid, "#888888"), alpha=0.65,
                   edgecolors="none",
                   label=f"C{cid}: {clip_label[cid]}")
    if dm_mask.any():
        ax.scatter(emb[dm_mask, 0], emb[dm_mask, 1], s=14, marker="x",
                   c="#D62728", linewidths=0.6, label="DM-cycle frames")

    ax.set_xlabel("t-SNE 1", fontsize=7)
    ax.set_ylabel("t-SNE 2", fontsize=7)
    ax.tick_params(direction="out", length=2.0, width=0.4, labelsize=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6, frameon=False, loc="upper left",
              handletextpad=0.2, borderpad=0.0, borderaxespad=0.0,
              labelspacing=0.3, ncol=1)


# ─────────────────────────────────────────────────────────────────────────────
# Panel D: behavioral asymmetry (sym vs asym mean cycle counts), one bar
# subplot per feature space, each at its own y-scale (EKP magnitudes are an
# order of magnitude smaller than intention / qpos).
# ─────────────────────────────────────────────────────────────────────────────

SYM_COLOR  = QUIET_COLOR    # symmetric → blue (always right)
ASYM_COLOR = ACTIVE_COLOR   # asymmetric → orange (always left)


def _bootstrap_means(values: np.ndarray, n_boot: int = 1000,
                     seed: int = 0) -> np.ndarray:
    if values.size == 0:
        return np.array([])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True)
    return draws.mean(axis=1)


def render_panel_c_violins(axes: list[plt.Axes]) -> None:
    """Behavioral asymmetry violins per feature space (own y-scale).

    Reproduces `asymmetry_violins_combined_aggregated.png`: violins are over
    bootstrap means (n=1000) of per-triplet `cycle_count_norm`, blue=symmetric
    / orange=asymmetric, mean ± std overlaid.
    """
    spaces = ["intention", "qpos", "ekp"]
    for ax, sp in zip(axes, spaces):
        df = pd.read_csv(ASYM_DIR / f"observations_{sp}.csv")
        s_raw = df[df["symmetry"] == "symmetric"]["cycle_count_norm"].to_numpy()
        a_raw = df[df["symmetry"] == "asymmetric"]["cycle_count_norm"].to_numpy()
        s = _bootstrap_means(s_raw)
        a = _bootstrap_means(a_raw)

        # Asymmetric (orange) on the LEFT, symmetric (blue) on the RIGHT.
        parts = ax.violinplot(
            [a, s], positions=[0, 1], widths=0.8, showextrema=False,
        )
        for pc, color in zip(parts["bodies"], [ASYM_COLOR, SYM_COLOR]):
            pc.set_facecolor(color)
            pc.set_edgecolor("#222222")
            pc.set_linewidth(0.6)
            pc.set_alpha(1.0)

        for xpos, vals in zip([0, 1], [a, s]):
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            ax.errorbar(xpos, m, yerr=sd, fmt="o", color="#222222",
                        markersize=3.0, elinewidth=0.7, capsize=2.0,
                        capthick=0.7, zorder=5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["asym.", "sym."], fontsize=7)
        ax.set_xlim(-0.5, 1.5)
        ax.set_xlabel(r"A$\to$B$\to$A symmetry", fontsize=8)
        ax.set_title(sp, fontsize=9, fontweight="bold", pad=4)
        if ax is axes[0]:
            ax.set_ylabel("Cycles per 100 frames", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.tick_params(direction="out", length=2.0, width=0.4, labelsize=7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Composite Figure 3
# ─────────────────────────────────────────────────────────────────────────────

PANEL_LABEL_KW = dict(fontsize=11, fontweight="bold", va="top", ha="left")


def main() -> None:
    # Single-row layout: (B) narrowed (only two bars), (A) and (C) widened so
    # the overall figure occupies more area without overlapping content.
    fig = plt.figure(figsize=(15.5, 3.8))

    gs_outer = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[1.25, 0.45, 3.6],
        wspace=0.10,
        left=0.045, right=0.995, top=0.86, bottom=0.20,
    )
    ax_a = fig.add_subplot(gs_outer[0, 0])
    ax_b = fig.add_subplot(gs_outer[0, 1])
    gs_c = gs_outer[0, 2].subgridspec(1, 3, wspace=0.07)
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c2 = fig.add_subplot(gs_c[0, 1])
    ax_c3 = fig.add_subplot(gs_c[0, 2])

    render_panel_a(ax_a)
    render_panel_b(ax_b)
    render_panel_c_violins([ax_c1, ax_c2, ax_c3])

    fig.text(0.012, 0.97, "(A)", **PANEL_LABEL_KW)
    fig.text(0.243, 0.97, "(B)", **PANEL_LABEL_KW)
    fig.text(0.350, 0.97, "(C)", **PANEL_LABEL_KW)

    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
