"""Coverage--quality boxplot panels.

Small-multiples grid: ``n_families`` rows x 2 columns. Each row is a
method family (DM-WKC, Mapper, KP-MoSeq); each column is a quality
metric (negated internal consistency on the left, negated MMD on the
right). Within a panel, one box per coverage tick pools per-(transition,
seed) cells from that family at that coverage. Y-scale shared per
column so families can be compared by eye-tracking down a column.

Both axes are negated so up-and-right is the ideal regime: tighter
clusters, broader coverage. The earlier module name ``method_scatter``
was a holdover from a literal scatter prototype; the rendered figure has
been boxplot panels for some time.

Reads:
  outputs/analysis/feature_analysis/dtw_comparison_table.csv
  outputs/analysis/feature_analysis/consistency_comparison_table.csv
  outputs/analysis/feature_analysis/<method>/seed_<N>/mmd/<method>_seed<N>_mmd_results.csv

Writes:
  outputs/analysis/feature_analysis/coverage_quality_panels/coverage_quality_panels.{png,pdf}
  outputs/analysis/feature_analysis/coverage_quality_panels/coverage_quality_panels_long.csv
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
FA_ROOT = REPO / "outputs/analysis/feature_analysis"


# Method-variant -> family map. DMC and DMP kept separate per the paper's
# best-of-family convention (Table 1).
FAMILY_OF: dict[str, str] = {
    "w2_dmc_int_kmeans":  "DM-cycle",
    "w2_dmc_qpos_kmeans": "DM-cycle",
    "w2_dmc_ekp_kmeans":  "DM-cycle",
    "w2_dmp_int_kmeans":  "DM-path",
    "w2_dmp_qpos_kmeans": "DM-path",
    "w2_dmp_ekp_kmeans":  "DM-path",
    "mapper_int":   "Mapper",
    "mapper_qpos":  "Mapper",
    "mapper_ekp":   "Mapper",
    "kpms_intention": "KP-MoSeq",
    "kpms_keypoint":  "KP-MoSeq",
}

FAMILY_ORDER = ["DM-cycle", "DM-path", "Mapper", "KP-MoSeq"]
FAMILY_COLOR = {
    "DM-cycle":  "#1F4E79",  # deep blue ("ours", cycles)
    "DM-path":   "#2E86AB",  # teal blue ("ours", paths)
    "Mapper":    "#B22222",  # red
    "KP-MoSeq":  "#2E7D32",  # green
}

# Re-key KPMS rows out of the legacy short label to the published name.
_PUBLIC_FAMILY_RENAME = {"KPMS": "KP-MoSeq", "Keypoint-MoSeq": "KP-MoSeq"}


@dataclass
class PanelsConfig:
    out_dir: Path = FA_ROOT / "coverage_quality_panels"
    figsize: tuple[float, float] = (10.5, 4.2)
    dpi: int = 220


def _parse_method_seed(col: str) -> tuple[str, int] | None:
    """Split a wide-CSV column like 'w2_dmc_int_kmeans_seed0' into
    (method, seed)."""
    if "_seed" not in col:
        return None
    method, seed_str = col.rsplit("_seed", 1)
    try:
        return method, int(seed_str)
    except ValueError:
        return None


def _read_wide_metric(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Some tables key on (Label, Turn); ensure both exist.
    if "Turn" not in df.columns:
        df["Turn"] = ""
    return df


def _read_per_seed_mmd_kid(method: str, seed: int) -> pd.DataFrame | None:
    p = FA_ROOT / method / f"seed_{seed}" / "mmd" / f"{method}_seed{seed}_mmd_results.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return None
    if df.empty:
        return None
    # Normalise column names so we can join on (Label, Turn).
    df = df.rename(columns={"Turn_Direction": "Turn"})
    return df


def collect_long(verbose: bool = False) -> pd.DataFrame:
    dtw = _read_wide_metric(FA_ROOT / "dtw_comparison_table.csv")
    cons = _read_wide_metric(FA_ROOT / "consistency_comparison_table.csv")

    method_cols = [c for c in dtw.columns
                   if c not in ("Label", "Turn") and "unif" not in c]

    # Long-form per (method, seed, transition).
    rows: list[dict] = []
    for col in method_cols:
        parsed = _parse_method_seed(col)
        if parsed is None:
            continue
        method, seed = parsed
        if method not in FAMILY_OF:
            continue
        family = FAMILY_OF[method]
        mmd_df = _read_per_seed_mmd_kid(method, seed)
        if mmd_df is None:
            if verbose:
                print(f"  skip {method} seed {seed}: no MMD/KID csv")
            continue

        for _, row in dtw.iterrows():
            label, turn = row["Label"], row.get("Turn", "")
            dtw_val = row[col]
            cons_match = cons[(cons["Label"] == label) & (cons["Turn"] == turn)]
            if cons_match.empty or col not in cons.columns:
                continue
            cons_val = cons_match.iloc[0][col]
            mk = mmd_df[(mmd_df["Label"] == label) & (mmd_df["Turn"] == turn)]
            if mk.empty:
                continue

            # Coerce; treat the CSV's "N/A" string as missing.
            try:
                dtw_v = float(dtw_val)
                cons_v = float(cons_val)
                mmd_v = float(mk.iloc[0]["MMD_RBF"])
                kid_v = float(mk.iloc[0]["KID"])
            except (ValueError, TypeError):
                continue
            if any(np.isnan([dtw_v, cons_v, mmd_v, kid_v])):
                continue

            rows.append({
                "method": method, "seed": seed, "family": family,
                "transition": f"{label} ({turn})",
                "dtw": dtw_v, "consistency": cons_v,
                "mmd": mmd_v, "kid": kid_v,
            })
    return pd.DataFrame(rows)


def _per_variant_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One dot per (family, variant, seed, transition). Coverage = number
    of unique transitions that (variant, seed) discovers (constant across
    all of that pair's transitions); metric values are the raw per-cell
    (variant, seed, transition) value -- no averaging across transitions."""
    sub = df.dropna(subset=["dtw", "mmd", "kid", "consistency"]).copy()
    if sub.empty:
        return sub
    cov = (sub.groupby(["family", "method", "seed"], as_index=False)
              .size().rename(columns={"size": "coverage"}))
    return sub.merge(cov, on=["family", "method", "seed"])


def render(df: pd.DataFrame, cfg: PanelsConfig) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    summary = _per_variant_summary(df)

    panels = [
        ("consistency", "internal consistency",
         "Negated internal consistency"),
        ("mmd", "MMD",
         "Similarity to ground truth"),
    ]

    fam_visible = [f for f in FAMILY_ORDER if (summary["family"] == f).any()]
    n_fams = len(fam_visible)

    BOX_WIDTH = 0.55
    BOX_ALPHA = 0.75  # solid-ish, no overlap to fight with

    cov_values = sorted(summary["coverage"].unique())

    # n_fams rows × 2 cols; share y within each metric column so the family
    # rows can be compared by eye-tracking down the column. Total figure
    # height is held to the original single-row size so the published
    # figure footprint doesn't grow.
    fig, axes = plt.subplots(
        n_fams, 2,
        figsize=(cfg.figsize[0] * 1.1, cfg.figsize[1]),
        dpi=cfg.dpi, sharex="col", sharey="col",
        squeeze=False,
    )

    panel_labels = ["(A)", "(B)"]
    for col_idx, (key, title, ylabel) in enumerate(panels):
        for row_idx, fam in enumerate(fam_visible):
            ax = axes[row_idx, col_idx]
            color = FAMILY_COLOR[fam]
            sub = summary[summary["family"] == fam]
            for cov_val in sorted(sub["coverage"].unique()):
                cells = -sub.loc[sub["coverage"] == cov_val,
                                 key].astype(float).to_numpy()
                if cells.size == 0:
                    continue
                bp = ax.boxplot(
                    [cells], positions=[cov_val], widths=BOX_WIDTH,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.4),
                    whiskerprops=dict(color=color, linewidth=1.0),
                    capprops=dict(color=color, linewidth=1.0),
                    boxprops=dict(linewidth=1.0, edgecolor=color),
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(BOX_ALPHA)

            ax.set_xticks(cov_values)
            ax.set_xticklabels(cov_values, fontsize=10)
            if cov_values:
                ax.set_xlim(min(cov_values) - BOX_WIDTH,
                             max(cov_values) + BOX_WIDTH)
            ax.tick_params(axis="y", labelsize=10, length=3, width=0.8)
            ax.grid(True, axis="y", color="0.92", linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)

            if row_idx == n_fams - 1:
                ax.set_xlabel("Coverage", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(fam, fontsize=10, fontweight="bold",
                              color=color, labelpad=8)
            if row_idx == 0:
                ax.set_title(ylabel, fontsize=11, pad=6)
                ax.text(-0.12, 1.12, panel_labels[col_idx],
                        transform=ax.transAxes,
                        ha="left", va="bottom",
                        fontsize=13, fontweight="bold")
            ax.tick_params(axis="y", labelsize=9, length=2.5, width=0.7)
            ax.tick_params(axis="x", labelsize=9, length=2.5, width=0.7)

    fig.tight_layout(w_pad=1.4, h_pad=0.25)

    png = cfg.out_dir / "coverage_quality_panels.png"
    pdf = cfg.out_dir / "coverage_quality_panels.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def build_coverage_quality_panels(out_dir: Path | None = None) -> Path:
    cfg = PanelsConfig()
    if out_dir is not None:
        cfg.out_dir = Path(out_dir)
    df = collect_long(verbose=True)
    print(f"  collected {len(df)} dots: "
          + ", ".join(f"{fam}={int((df.family==fam).sum())}"
                      for fam in FAMILY_ORDER))
    if df.empty:
        raise RuntimeError("no scatter data; check that all four metric "
                           "tables exist under outputs/analysis/feature_analysis/")
    csv_out = cfg.out_dir / "coverage_quality_panels_long.csv"
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"  wrote {csv_out}")
    return render(df, cfg)


if __name__ == "__main__":
    p = build_coverage_quality_panels()
    print(f"wrote {p}")
