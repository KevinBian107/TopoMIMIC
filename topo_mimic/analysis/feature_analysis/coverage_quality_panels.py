"""Coverage--quality boxplot panels.

Small-multiples grid: ``n_families`` rows x 2 columns. Each row is a
method family (DM-WKC, Mapper, KP-MoSeq); each column is a quality
metric (internal inconsistency = within-cluster DTW distance on the
left, MMD distance to ground truth on the right). Within a panel, one
box per coverage tick pools per-(transition, seed) cells from that
family at that coverage. Y-scale shared per column so families can be
compared by eye-tracking down a column.

Both axes are raw distances, so lower is better: down-and-right is the
ideal regime (tighter clusters / closer to ground truth, broader
coverage). The earlier module name ``method_scatter`` was a holdover
from a literal scatter prototype; the rendered figure has been boxplot
panels for some time.

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
        "font.size": 18,
        "axes.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    summary = _per_variant_summary(df)

    # (metric key, y-axis label). Metric goes on the y-axis; each row is one
    # family (a single color), named only in the legend below.
    panels = [
        ("consistency", "Internal inconsistency (DTW distance)"),
        ("mmd", "GT alignment difference (MMD distance)"),
    ]

    fam_visible = [f for f in FAMILY_ORDER if (summary["family"] == f).any()]
    n_fams = len(fam_visible)
    cov_values = sorted(summary["coverage"].unique())

    BOX_WIDTH = 0.55

    # small-multiples grid restored: rows = families (one color each, see legend),
    # cols = metrics; y shared per column so families compare down a column.
    fig, axes = plt.subplots(
        n_fams, 2,
        figsize=(cfg.figsize[0] * 1.2, cfg.figsize[1] * 1.2),
        dpi=cfg.dpi, sharex="col", sharey="col", squeeze=False,
    )

    panel_labels = ["(A)", "(B)"]
    for col_idx, (key, _ylab) in enumerate(panels):
        for row_idx, fam in enumerate(fam_visible):
            ax = axes[row_idx, col_idx]
            color = FAMILY_COLOR[fam]
            sub = summary[summary["family"] == fam]
            for cov_val in sorted(sub["coverage"].unique()):
                # Raw distances (DTW consistency, MMD): lower is better, no negation.
                cells = sub.loc[sub["coverage"] == cov_val,
                                key].astype(float).to_numpy()
                cells = cells[~np.isnan(cells)]
                if cells.size == 0:
                    continue
                bp = ax.boxplot(
                    [cells], positions=[cov_val], widths=BOX_WIDTH,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.6),
                    whiskerprops=dict(color=color, linewidth=1.2),
                    capprops=dict(color=color, linewidth=1.2),
                    boxprops=dict(linewidth=1.2, edgecolor=color),
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.78)

            ax.set_xticks(cov_values)
            if cov_values:
                ax.set_xlim(min(cov_values) - BOX_WIDTH, max(cov_values) + BOX_WIDTH)
            ax.grid(True, axis="y", color="0.92", linewidth=0.7, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", labelsize=13, length=3.0, width=0.8)
            ax.tick_params(axis="x", labelsize=13, length=3.0, width=0.8)

            if row_idx == n_fams - 1:
                ax.set_xticklabels(cov_values, fontsize=13)
                ax.set_xlabel("Coverage", fontsize=16)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if row_idx == 0:
                ax.text(-0.04, 1.12, panel_labels[col_idx], transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=12, fontweight="bold")

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=FAMILY_COLOR[f], edgecolor=FAMILY_COLOR[f],
                     alpha=0.78, label=f) for f in fam_visible]
    fig.legend(handles=handles, loc="lower center", ncol=len(fam_visible),
               frameon=False, fontsize=14, bbox_to_anchor=(0.5, -0.005),
               handlelength=1.3, columnspacing=1.8)

    fig.tight_layout(rect=[0.05, 0.04, 1, 0.99], h_pad=0.3, w_pad=2.0)

    # Metric name centered on each column's y-axis, placed just left of the actual
    # tick labels (measured via get_tightbbox so it never overlaps the numbers).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for col_idx, (_key, ylab) in enumerate(panels):
        boxes = [axes[r, col_idx].get_position() for r in range(n_fams)]
        y_c = (min(b.y0 for b in boxes) + max(b.y1 for b in boxes)) / 2.0
        x_left = min(
            inv.transform(axes[r, col_idx].get_tightbbox(renderer))[0][0]
            for r in range(n_fams)
        )
        fig.text(x_left - 0.012, y_c, ylab, rotation="vertical",
                 va="center", ha="center", fontsize=16)

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
