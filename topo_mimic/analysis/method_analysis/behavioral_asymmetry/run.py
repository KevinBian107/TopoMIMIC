#!/usr/bin/env python3
"""
Behavioral asymmetry verification on A -> B -> A triplets.

For every GT A -> B -> A triplet, count the DM-cycles whose vertex-set
overlaps the triplet's frame range. Classify triplets as

  symmetric   -> direction label on first A == direction label on second A
  asymmetric  -> directions differ

Then produce Nature-style paired violin plots (bootstrap error bars)
comparing cycle counts in symmetric vs asymmetric triplets, both
aggregated and split by (A, B) transition type, in absolute units and
normalized per 100 frames.

Usage:
    python -m topo_mimic.analysis.method_analysis.behavioral_asymmetry.run \
           configs/analysis/method_analysis/behavioral_asymmetry.yaml
"""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Patch


BEHAVIORS = {"Immobile", "Turn", "Rear", "Walk"}
DIRECTIONS = {"Left", "Right", "Straight"}

SYM_COLOR = "#4C72B0"
ASYM_COLOR = "#DD8452"


@dataclass
class AsymmetryConfig:
    behavior_labels_csv: str = "behavior_labels/rodent_labels.csv"
    feature_spaces: list[str] = field(default_factory=lambda: ["intention", "qpos", "ekp"])
    extraction_dirs: dict[str, str] = field(default_factory=lambda: {
        "intention": "outputs/extraction/dm_pipeline_intention",
        "qpos":      "outputs/extraction/dm_pipeline_qpos",
        "ekp":       "outputs/extraction/dm_pipeline_ekp",
    })
    output_dir: str = "outputs/analysis/method_analysis/behavioral_asymmetry"
    n_bootstrap: int = 1000
    bootstrap_seed: int = 0
    min_support: int = 3
    skip_clips: list[int] = field(default_factory=list)


@dataclass
class TripletRecord:
    clip_id: int
    a_label: str
    b_label: str
    triplet_start: int
    triplet_end: int
    dir_from: str
    dir_to: str

    @property
    def pair_key(self) -> str:
        return f"{self.a_label}->{self.b_label}->{self.a_label}"

    @property
    def short_key(self) -> str:
        abbrev = {"Immobile": "Im", "Turn": "Tu", "Rear": "Re", "Walk": "Wk"}
        a = abbrev.get(self.a_label, self.a_label[:2])
        b = abbrev.get(self.b_label, self.b_label[:2])
        return f"{a}-{b}-{a}"

    @property
    def is_symmetric(self) -> bool:
        return self.dir_from == self.dir_to and self.dir_from != "None"

    @property
    def duration(self) -> int:
        return self.triplet_end - self.triplet_start + 1


def load_config(path: str | None) -> AsymmetryConfig:
    cfg = AsymmetryConfig()
    if not path or not os.path.exists(path):
        return cfg
    with open(path) as f:
        blob = yaml.safe_load(f) or {}
    for k, v in blob.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def parse_clip_id(video: str) -> int | None:
    """clip_153.mp4 -> 153"""
    name = Path(str(video)).stem
    if name.startswith("clip_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return None
    return None


def find_triplets(csv_path: str, skip_clips: set[int]) -> list[TripletRecord]:
    df = pd.read_csv(csv_path)
    df_b = df[df["Label"].isin(BEHAVIORS)].sort_values(
        ["Video", "Start_Frame"]
    ).reset_index(drop=True)
    df_d = df[df["Label"].isin(DIRECTIONS)].copy()
    dir_by_vid: dict[str, pd.DataFrame] = {
        v: g.reset_index(drop=True) for v, g in df_d.groupby("Video")
    }

    def seg_dir(vid: str, start: int, end: int) -> str:
        rows = dir_by_vid.get(vid)
        if rows is None or rows.empty:
            return "None"
        best_label, best_overlap = "None", 0
        for _, r in rows.iterrows():
            lo = max(start, int(r["Start_Frame"]))
            hi = min(end, int(r["End_Frame"]))
            overlap = max(0, hi - lo + 1)
            if overlap > best_overlap:
                best_overlap, best_label = overlap, str(r["Label"])
        return best_label

    out: list[TripletRecord] = []
    for vid, group in df_b.groupby("Video"):
        cid = parse_clip_id(vid)
        if cid is None or cid in skip_clips:
            continue
        rows = group.reset_index(drop=True)
        for i in range(len(rows) - 2):
            a = rows.loc[i,   "Label"]
            b = rows.loc[i+1, "Label"]
            c = rows.loc[i+2, "Label"]
            if a != c or a == b:
                continue
            s1 = int(rows.loc[i,   "Start_Frame"])
            e2 = int(rows.loc[i+2, "End_Frame"])
            d1 = seg_dir(vid, s1, int(rows.loc[i,   "End_Frame"]))
            d2 = seg_dir(vid, int(rows.loc[i+2, "Start_Frame"]), e2)
            out.append(TripletRecord(
                clip_id=cid, a_label=a, b_label=b,
                triplet_start=s1, triplet_end=e2,
                dir_from=d1, dir_to=d2,
            ))
    return out


def load_cycles_for_space(extraction_dir: str) -> dict[int, list[list[int]]]:
    """Load all_cycle_dm.pkl (cycles in Takens-point space per clip)."""
    pkl = Path(extraction_dir) / "all_cycle_dm.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"missing {pkl}")
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return {int(k): list(v) for k, v in data.items()}
    return {i: list(c) for i, c in enumerate(data)}


def count_overlapping_cycles(
    cycles: list[list[int]],
    triplet: TripletRecord,
) -> int:
    """Cycles with at least one vertex in the triplet's frame range."""
    lo, hi = triplet.triplet_start, triplet.triplet_end
    count = 0
    for cyc in cycles:
        for v in cyc:
            if lo <= int(v) <= hi:
                count += 1
                break
    return count


def assemble_observations(
    triplets: list[TripletRecord],
    cycles_by_clip: dict[int, list[list[int]]],
) -> pd.DataFrame:
    rows = []
    for t in triplets:
        cycles = cycles_by_clip.get(t.clip_id, [])
        n_abs = count_overlapping_cycles(cycles, t)
        n_norm = 100.0 * n_abs / t.duration if t.duration > 0 else 0.0
        rows.append({
            "clip_id": t.clip_id,
            "pair": t.pair_key,
            "pair_short": t.short_key,
            "symmetry": "symmetric" if t.is_symmetric else "asymmetric",
            "a_label": t.a_label,
            "b_label": t.b_label,
            "dir_from": t.dir_from,
            "dir_to": t.dir_to,
            "triplet_start": t.triplet_start,
            "triplet_end": t.triplet_end,
            "duration_frames": t.duration,
            "cycle_count_abs": n_abs,
            "cycle_count_norm": n_norm,
        })
    return pd.DataFrame(rows)


def bootstrap_means(
    values: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if values.size == 0:
        return np.array([])
    draws = rng.choice(values, size=(n_boot, values.size), replace=True)
    return draws.mean(axis=1)


def _nature_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=8, width=0.8, length=3)


def _draw_violin(ax, pos: float, vals: np.ndarray, color: str, label_below: str):
    if vals.size == 0:
        ax.text(pos, 0, "n=0", ha="center", va="bottom", fontsize=6,
                color="#777", rotation=0)
        return
    vp = ax.violinplot([vals], positions=[pos], widths=0.8, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(0.55)
        body.set_edgecolor("#222")
        body.set_linewidth(0.7)
    mu = float(vals.mean())
    lo, hi = np.percentile(vals, [2.5, 97.5])
    ax.errorbar([pos], [mu], yerr=[[mu - lo], [hi - mu]],
                fmt="o", markersize=3.5, color="#222",
                capsize=2.5, linewidth=0.9, zorder=5)
    ax.text(pos, ax.get_ylim()[0], label_below, ha="center", va="top",
            fontsize=6, color="#444")


def paired_violin_panel(
    ax,
    groups: list[tuple[str, np.ndarray, np.ndarray, int, int]],
    # each: (group_label, sym_boot, asym_boot, n_sym_obs, n_asym_obs)
    ylabel: str,
    title: str,
    legend_handles: list | None = None,
):
    _nature_axis(ax)
    # Collect all values to set a sensible y lim before drawing n=0 text.
    all_vals = [v for _, s, a, *_ in groups for v in (list(s) + list(a))]
    if all_vals:
        ymax = max(all_vals) * 1.12
        ax.set_ylim(0, max(ymax, 0.05))
    else:
        ax.set_ylim(0, 1)

    x_ticks = []
    x_labels = []
    for i, (label, sym, asym, n_sym_obs, n_asym_obs) in enumerate(groups):
        x0 = 2.0 * i
        x1 = x0 + 0.9
        _draw_violin(ax, x0, sym, SYM_COLOR, f"n={n_sym_obs}")
        _draw_violin(ax, x1, asym, ASYM_COLOR, f"n={n_asym_obs}")
        x_ticks.append((x0 + x1) / 2)
        x_labels.append(label)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.axhline(0, color="#bbb", linewidth=0.4, zorder=0)
    if legend_handles:
        ax.legend(handles=legend_handles, frameon=False, fontsize=7.5,
                  loc="upper right")


def render_figure_for_space(
    df: pd.DataFrame,
    feature_space: str,
    cfg: AsymmetryConfig,
    out_path: Path,
) -> None:
    rng = np.random.default_rng(cfg.bootstrap_seed)

    def boot(sub_df: pd.DataFrame, col: str) -> np.ndarray:
        return bootstrap_means(sub_df[col].to_numpy(dtype=float),
                               cfg.n_bootstrap, rng)

    # Aggregated (all triplets pooled)
    sym_all = df[df["symmetry"] == "symmetric"]
    asym_all = df[df["symmetry"] == "asymmetric"]
    agg_abs = [(
        "All A->B->A",
        boot(sym_all, "cycle_count_abs"),
        boot(asym_all, "cycle_count_abs"),
        len(sym_all), len(asym_all),
    )]
    agg_norm = [(
        "All A->B->A",
        boot(sym_all, "cycle_count_norm"),
        boot(asym_all, "cycle_count_norm"),
        len(sym_all), len(asym_all),
    )]

    # Per-pair (sorted by total support desc)
    per_pair_totals = df.groupby("pair_short").size().sort_values(ascending=False)
    per_pair_abs = []
    per_pair_norm = []
    for pair in per_pair_totals.index:
        sub = df[df["pair_short"] == pair]
        s_sub = sub[sub["symmetry"] == "symmetric"]
        a_sub = sub[sub["symmetry"] == "asymmetric"]
        n_sym = len(s_sub)
        n_asym = len(a_sub)
        if max(n_sym, n_asym) < cfg.min_support:
            continue
        per_pair_abs.append((pair, boot(s_sub, "cycle_count_abs"),
                             boot(a_sub, "cycle_count_abs"), n_sym, n_asym))
        per_pair_norm.append((pair, boot(s_sub, "cycle_count_norm"),
                              boot(a_sub, "cycle_count_norm"), n_sym, n_asym))

    n_pairs = len(per_pair_abs)
    fig_w = 4 + max(5.5, n_pairs * 1.3)
    fig = plt.figure(figsize=(fig_w, 8.5))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1, max(1.8, n_pairs * 0.55)],
        hspace=0.65, wspace=0.28,
    )
    ax_abs_agg  = fig.add_subplot(gs[0, 0])
    ax_abs_pair = fig.add_subplot(gs[0, 1])
    ax_nrm_agg  = fig.add_subplot(gs[1, 0])
    ax_nrm_pair = fig.add_subplot(gs[1, 1])

    legend_handles = [
        Patch(facecolor=SYM_COLOR, alpha=0.55, edgecolor="#222",
              label="symmetric (same direction)"),
        Patch(facecolor=ASYM_COLOR, alpha=0.55, edgecolor="#222",
              label="asymmetric (different direction)"),
    ]

    paired_violin_panel(ax_abs_agg, agg_abs,
                        ylabel="Cycles per triplet (abs.)",
                        title="Aggregated",
                        legend_handles=legend_handles)
    paired_violin_panel(ax_abs_pair, per_pair_abs,
                        ylabel="", title="Per A->B->A type (abs.)")
    paired_violin_panel(ax_nrm_agg, agg_norm,
                        ylabel="Cycles per 100 frames",
                        title="Aggregated (normalized)")
    paired_violin_panel(ax_nrm_pair, per_pair_norm,
                        ylabel="",
                        title="Per A->B->A type (normalized)")

    fig.suptitle(
        f"Behavioral asymmetry vs DM-cycle count — feature space: {feature_space}",
        fontsize=12, fontweight="bold", y=1.005,
    )
    plt.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"  wrote {out_path} (+.png)")


def render_combined_normalized_figure(
    dfs: dict[str, pd.DataFrame],
    cfg: AsymmetryConfig,
    out_path: Path,
) -> None:
    """One figure, normalized only, 1 row x 3 cols (one column per feature space)."""
    feature_spaces = list(dfs.keys())
    n_cols = len(feature_spaces)

    rng = np.random.default_rng(cfg.bootstrap_seed)

    def boot(sub: pd.DataFrame, col: str) -> np.ndarray:
        return bootstrap_means(sub[col].to_numpy(dtype=float),
                               cfg.n_bootstrap, rng)

    # Pair ordering fixed across panels, based on pooled support.
    pooled = pd.concat(list(dfs.values()), ignore_index=True)
    totals = pooled.groupby("pair_short").size().sort_values(ascending=False)

    per_space_groups: dict[str, list[tuple[str, np.ndarray, np.ndarray, int, int]]] = {}
    for fs, df in dfs.items():
        groups = []
        # Aggregated as the first group in each panel.
        s_all = df[df["symmetry"] == "symmetric"]
        a_all = df[df["symmetry"] == "asymmetric"]
        groups.append((
            "All",
            boot(s_all, "cycle_count_norm"),
            boot(a_all, "cycle_count_norm"),
            len(s_all), len(a_all),
        ))
        for pair in totals.index:
            sub = df[df["pair_short"] == pair]
            s_sub = sub[sub["symmetry"] == "symmetric"]
            a_sub = sub[sub["symmetry"] == "asymmetric"]
            if max(len(s_sub), len(a_sub)) < cfg.min_support:
                continue
            groups.append((
                pair,
                boot(s_sub, "cycle_count_norm"),
                boot(a_sub, "cycle_count_norm"),
                len(s_sub), len(a_sub),
            ))
        per_space_groups[fs] = groups

    n_groups = max(len(g) for g in per_space_groups.values())
    fig_w = n_cols * max(4.5, n_groups * 0.85)
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, 5.0), squeeze=False)
    axes = axes[0]

    legend_handles = [
        Patch(facecolor=SYM_COLOR, alpha=0.55, edgecolor="#222",
              label="symmetric (same direction)"),
        Patch(facecolor=ASYM_COLOR, alpha=0.55, edgecolor="#222",
              label="asymmetric (different direction)"),
    ]

    y_tops = []
    for ax, fs in zip(axes, feature_spaces):
        paired_violin_panel(
            ax, per_space_groups[fs],
            ylabel="Cycles per 100 frames" if fs == feature_spaces[0] else "",
            title=f"{fs}",
            legend_handles=None,
        )
        y_tops.append(ax.get_ylim()[1])

    shared_top = max(y_tops) if y_tops else 1.0
    for ax in axes:
        ax.set_ylim(0, shared_top)

    fig.suptitle(
        "Behavioral asymmetry vs DM-cycle count (normalized, per 100 frames)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.legend(handles=legend_handles, loc="upper center", ncol=2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.98))

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"  wrote {out_path} (+.png)")


def _per_pair_summary(df: pd.DataFrame, feature_space: str) -> pd.DataFrame:
    rows = []
    for pair, sub in df.groupby("pair"):
        s = sub[sub["symmetry"] == "symmetric"]
        a = sub[sub["symmetry"] == "asymmetric"]
        rows.append({
            "feature_space": feature_space,
            "pair": pair,
            "n_sym": len(s), "n_asym": len(a),
            "mean_sym_abs":  float(s["cycle_count_abs"].mean())  if len(s) else np.nan,
            "mean_asym_abs": float(a["cycle_count_abs"].mean())  if len(a) else np.nan,
            "mean_sym_norm":  float(s["cycle_count_norm"].mean())  if len(s) else np.nan,
            "mean_asym_norm": float(a["cycle_count_norm"].mean())  if len(a) else np.nan,
        })
    return pd.DataFrame(rows)


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading triplets from {cfg.behavior_labels_csv}...")
    triplets = find_triplets(cfg.behavior_labels_csv, set(cfg.skip_clips))
    n_sym = sum(1 for t in triplets if t.is_symmetric)
    n_asym = len(triplets) - n_sym
    print(f"  total triplets: {len(triplets)}  "
          f"(symmetric={n_sym}, asymmetric={n_asym})")

    per_space_summaries = []
    per_pair_summaries = []
    dfs_by_space: dict[str, pd.DataFrame] = {}
    for fs in cfg.feature_spaces:
        print(f"\nFeature space: {fs}")
        ext_dir = cfg.extraction_dirs[fs]
        cycles_by_clip = load_cycles_for_space(ext_dir)
        print(f"  loaded cycles for {len(cycles_by_clip)} clips")

        df = assemble_observations(triplets, cycles_by_clip)
        csv_path = out_dir / f"observations_{fs}.csv"
        df.to_csv(csv_path, index=False)

        m_sym_abs = float(df[df["symmetry"] == "symmetric"]["cycle_count_abs"].mean())
        m_asym_abs = float(df[df["symmetry"] == "asymmetric"]["cycle_count_abs"].mean())
        m_sym_norm = float(df[df["symmetry"] == "symmetric"]["cycle_count_norm"].mean())
        m_asym_norm = float(df[df["symmetry"] == "asymmetric"]["cycle_count_norm"].mean())
        print(f"  mean abs  cycles (sym={m_sym_abs:.3f}, asym={m_asym_abs:.3f})")
        print(f"  mean norm cycles (sym={m_sym_norm:.3f}, asym={m_asym_norm:.3f})")

        fig_path = out_dir / f"asymmetry_violins_{fs}.pdf"
        render_figure_for_space(df, fs, cfg, fig_path)
        dfs_by_space[fs] = df

        per_space_summaries.append({
            "feature_space": fs,
            "n_triplets": len(df),
            "n_sym": n_sym, "n_asym": n_asym,
            "mean_sym_abs": m_sym_abs, "mean_asym_abs": m_asym_abs,
            "mean_sym_norm": m_sym_norm, "mean_asym_norm": m_asym_norm,
        })
        per_pair_summaries.append(_per_pair_summary(df, fs))

    pd.DataFrame(per_space_summaries).to_csv(
        out_dir / "summary_by_feature_space.csv", index=False,
    )
    pd.concat(per_pair_summaries, ignore_index=True).to_csv(
        out_dir / "summary_by_pair.csv", index=False,
    )

    combined_path = out_dir / "asymmetry_violins_combined_normalized.pdf"
    render_combined_normalized_figure(dfs_by_space, cfg, combined_path)

    print(f"\nAll results written to {out_dir}")


if __name__ == "__main__":
    cfg_path = (
        sys.argv[1] if len(sys.argv) > 1
        else "configs/analysis/method_analysis/behavioral_asymmetry.yaml"
    )
    main(cfg_path)
