"""Build the 3 undirected transition × method_family tables.

Reads every cluster_c2st_seed_{N}.csv under outputs/benchmark_analysis/ and
emits, for each model in {logreg, rf, mlp}, a pivot where:
  - rows = UNDIRECTED_PAIRS (6 canonical A↔B)
  - cols = method families (6)
  - cells = mean C2ST accuracy across all clusters (method, seed, cluster_id)
           whose undirected key matches the row, averaged per family.

Also emits matching n_clusters tables (how many clusters contributed).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from topo_mimic.experiment.c2st import FAMILY_MEMBERS, UNDIRECTED_PAIRS
from topo_mimic.experiment.c2st.models import MODEL_DISPLAY_NAMES


def _collect_rows(analysis_dir: str) -> pd.DataFrame:
    frames = []
    method_to_family = {
        m: fam for fam, members in FAMILY_MEMBERS.items() for m in members
    }
    base = Path(analysis_dir)
    for method_dir in sorted(base.iterdir()):
        if not method_dir.is_dir() or method_dir.name in ("transition_ranking", "tables"):
            continue
        method = method_dir.name
        family = method_to_family.get(method)
        if family is None:
            continue
        for seed_dir in sorted(method_dir.iterdir()):
            if not seed_dir.name.startswith("seed_"):
                continue
            for f in seed_dir.glob("c2st/cluster_c2st_seed_*.csv"):
                try:
                    df = pd.read_csv(f)
                except pd.errors.EmptyDataError:
                    continue
                if df.empty:
                    continue
                df["method"] = method
                df["family"] = family
                df["seed"] = int(seed_dir.name.split("_")[-1])
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _is_insufficient(col) -> pd.Series:
    """Cast the insufficient_data column defensively. After CSV roundtrip,
    True/False may arrive as strings; `.astype(bool)` on 'False' returns
    True (non-empty string is truthy). Compare to known falsy tokens."""
    s = col.astype(str).str.strip().str.lower()
    return s.isin({"true", "1"})


def _pivot(
    df: pd.DataFrame, model: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (mean, std, n_seeds) per (undirected, family) across SEEDS.

    Within each seed we first average all cluster-level accuracies matching
    that (undirected, family) cell; error bars are std of those per-seed
    point estimates. std is NaN when only one seed contributes; n_seeds
    reports how many seeds landed in each cell (used to annotate "1 seed"
    in the rendered table).
    """
    col = f"acc_{model}"
    if col not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    sub = df[~_is_insufficient(df["insufficient_data"])].copy()
    sub = sub[sub["undirected"] != ""]
    per_seed = (
        sub.groupby(["undirected", "family", "seed"])[col]
        .mean()
        .reset_index()
    )
    mean_pivot = per_seed.pivot_table(
        index="undirected", columns="family", values=col, aggfunc="mean",
    )
    std_pivot = per_seed.pivot_table(
        index="undirected", columns="family", values=col, aggfunc="std",
    )
    n_seeds_pivot = per_seed.pivot_table(
        index="undirected", columns="family", values=col, aggfunc="count",
    )
    mean_pivot = mean_pivot.reindex(
        index=UNDIRECTED_PAIRS, columns=list(FAMILY_MEMBERS.keys()),
    )
    std_pivot = std_pivot.reindex(
        index=UNDIRECTED_PAIRS, columns=list(FAMILY_MEMBERS.keys()),
    )
    n_seeds_pivot = n_seeds_pivot.reindex(
        index=UNDIRECTED_PAIRS, columns=list(FAMILY_MEMBERS.keys()),
    )
    return mean_pivot, std_pivot, n_seeds_pivot


def _pivot_n(df: pd.DataFrame, model: str) -> pd.DataFrame:
    sub = df[~_is_insufficient(df["insufficient_data"])].copy()
    sub = sub[sub["undirected"] != ""]
    sub = sub[sub[f"acc_{model}"].notna()]
    pivot = sub.pivot_table(
        index="undirected", columns="family", values=f"acc_{model}", aggfunc="count",
    )
    pivot = pivot.reindex(index=UNDIRECTED_PAIRS, columns=list(FAMILY_MEMBERS.keys()))
    return pivot.fillna(0).astype(int)


def _render(
    pivot: pd.DataFrame,
    n_pivot: pd.DataFrame,
    title: str,
    out_png: str,
    std_pivot: pd.DataFrame | None = None,
    n_seeds_pivot: pd.DataFrame | None = None,
) -> None:
    data = pivot.values.astype(float)
    n_data = n_pivot.values
    std_data = (
        std_pivot.reindex(index=pivot.index, columns=pivot.columns).values.astype(float)
        if std_pivot is not None else None
    )
    n_seeds_data = (
        n_seeds_pivot.reindex(index=pivot.index, columns=pivot.columns).values.astype(float)
        if n_seeds_pivot is not None else None
    )
    has_std = std_data is not None
    row_pitch = 1.0 if has_std else 0.6
    fig, ax = plt.subplots(
        figsize=(1.7 * len(pivot.columns) + 2, row_pitch * len(pivot) + 1.6)
    )
    ax.axis("off")
    rows_text = []
    for i, lbl in enumerate(pivot.index):
        row = [lbl]
        for j, c in enumerate(pivot.columns):
            v = data[i, j]
            n = int(n_data[i, j])
            if np.isfinite(v) and n > 0:
                if has_std and np.isfinite(std_data[i, j]):
                    row.append(f"{v:.3f} ± {std_data[i, j]:.3f}\n(n={n})")
                elif (n_seeds_data is not None
                      and np.isfinite(n_seeds_data[i, j])
                      and int(n_seeds_data[i, j]) == 1):
                    row.append(f"{v:.3f}\n(n={n}, 1 seed)")
                else:
                    row.append(f"{v:.3f}\n(n={n})")
            else:
                row.append("—")
        rows_text.append(row)
    tbl = ax.table(
        cellText=rows_text,
        colLabels=["transition"] + list(pivot.columns),
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(pivot.columns) + 1)))
    tbl.scale(1.1, 2.8 if has_std else 1.6)
    for (_r, _c), _cell in tbl.get_celld().items():
        if _r > 0:
            _cell.PAD = 0.1
    for j in range(len(pivot.columns) + 1):
        tbl[(0, j)].set_text_props(weight="bold", color="white")
        tbl[(0, j)].set_facecolor("#404040")
    # color: interpret accuracy as "closer to 0.5 = better blend". Shade
    # warm if far from 0.5 (bad), cool if near 0.5 (good).
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            v = data[i, j]
            n = int(n_data[i, j])
            if not (np.isfinite(v) and n > 0):
                continue
            dist = min(max(v - 0.5, 0.0), 0.5) * 2.0  # 0..1
            tbl[(i + 1, j + 1)].set_facecolor((1.0, 1.0 - dist * 0.5, 1.0 - dist * 0.7))
    plt.title(title, fontsize=11, pad=10, weight="bold")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(analysis_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = _collect_rows(analysis_dir)
    if df.empty:
        print(f"no cluster_c2st_seed_*.csv found under {analysis_dir}")
        return
    flat_csv = Path(out_dir) / "c2st_all_clusters.csv"
    df.to_csv(flat_csv, index=False)
    print(f"wrote {flat_csv}  ({len(df)} rows)")

    for model, display in MODEL_DISPLAY_NAMES.items():
        pivot, std_pivot, n_seeds_pivot = _pivot(df, model)
        n_pivot = _pivot_n(df, model)
        acc_path = Path(out_dir) / f"c2st_{model}_accuracy.csv"
        std_path = Path(out_dir) / f"c2st_{model}_accuracy_std.csv"
        n_path = Path(out_dir) / f"c2st_{model}_n_clusters.csv"
        seeds_path = Path(out_dir) / f"c2st_{model}_n_seeds.csv"
        pivot.to_csv(acc_path)
        std_pivot.to_csv(std_path)
        n_pivot.to_csv(n_path)
        n_seeds_pivot.to_csv(seeds_path)
        png_path = Path(out_dir) / f"c2st_{model}_accuracy.png"
        _render(
            pivot, n_pivot, f"C2ST accuracy — {display}", str(png_path),
            std_pivot=std_pivot, n_seeds_pivot=n_seeds_pivot,
        )
        print(f"wrote {acc_path} / {std_path} / {n_path} / {seeds_path} / {png_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", default="outputs/benchmark_analysis")
    p.add_argument("--out-dir", default="outputs/benchmark_analysis/tables")
    a = p.parse_args()
    main(a.analysis_dir, a.out_dir)
