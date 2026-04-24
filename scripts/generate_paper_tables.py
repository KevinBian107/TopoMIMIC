#!/usr/bin/env python3
"""Generate all paper tables from analysis results.

Outputs:
  tables/figure4_summary.pdf       — Concise summary for Figure 4
  tables/appendix_dtw_detailed.pdf — Full DTW table (Appendix A5)
  tables/appendix_consistency.pdf  — Full consistency table (Appendix A6)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topo_mimic.analysis._tmp_tables import METHOD_DISPLAY_NAMES

ANALYSIS_DIR = "outputs/benchmark_analysis"
OUTPUT_DIR = "tables"

METHOD_ORDER = [
    "w2_dmc_int_kmeans", "w2_dmc_qpos_kmeans", "w2_dmc_ekp_kmeans",
    "w2_dmp_int_kmeans", "w2_dmp_qpos_kmeans", "w2_dmp_ekp_kmeans",
    "kpms_intention", "kpms_keypoint", "custom_vae",
]

SHORT_NAMES = {
    "w2_dmc_int_kmeans": "DMC-Int",
    "w2_dmc_qpos_kmeans": "DMC-Qpos",
    "w2_dmc_ekp_kmeans": "DMC-EKP",
    "w2_dmp_int_kmeans": "DMP-Int",
    "w2_dmp_qpos_kmeans": "DMP-Qpos",
    "w2_dmp_ekp_kmeans": "DMP-EKP",
    "kpms_intention": "KPMS-Int",
    "kpms_keypoint": "KPMS-KP",
    "custom_vae": "VAE",
}


def _col_key(col: str) -> str:
    return col.replace("_seed0", "")


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    method_cols = [c for c in df.columns if c not in ("Label", "Turn")]
    ordered = []
    for m in METHOD_ORDER:
        matches = [c for c in method_cols if c.startswith(m)]
        ordered.extend(matches)
    for c in method_cols:
        if c not in ordered:
            ordered.append(c)
    return df[["Label", "Turn"] + ordered]


def _rename_cols(cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        key = _col_key(c)
        out.append(SHORT_NAMES.get(key, METHOD_DISPLAY_NAMES.get(key, c)))
    return out


def _render_table(
    df: pd.DataFrame,
    output_path: str,
    title: str,
    highlight_best: bool = True,
    dpi: int = 300,
    font_size: int = 8,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
    })

    n_rows = len(df) + 1
    n_cols = len(df.columns) + 1
    fig_w = max(10, 1.5 * n_cols + 2)
    fig_h = max(3, 0.38 * n_rows + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    cell_text = []
    cell_colors = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    nd_rows = set()
    for idx, row_name in enumerate(df.index):
        if "non-deterministic" in str(row_name):
            nd_rows.add(idx)

    for idx, (row_name, row) in enumerate(df.iterrows()):
        is_nd = idx in nd_rows
        default_bg = "#ffcdd2" if is_nd else "white"
        row_text = []
        row_colors = []

        row_vals = row[numeric_cols].dropna()
        row_min = row_vals.min() if len(row_vals) > 0 else None

        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                row_text.append("—")
                row_colors.append(default_bg)
            elif col in numeric_cols:
                row_text.append(f"{val:.3f}")
                if highlight_best and row_min is not None and val == row_min and not is_nd:
                    row_colors.append("#c8e6c9")
                else:
                    row_colors.append(default_bg)
            else:
                row_text.append(str(val))
                row_colors.append(default_bg)
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    row_labels = [str(i) for i in df.index]
    col_labels = list(df.columns)

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    table.scale(1.3, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", fontsize=font_size + 1)
            cell.set_facecolor("#e3f2fd")
        if col == -1:
            cell.set_facecolor("#f5f5f5")
            cell.set_width(cell.get_width() * 1.5)

    ax.set_title(title, fontsize=font_size + 3, fontweight="bold", pad=15)

    fig.text(0.5, 0.01,
             "Green = best per row. Red rows = non-deterministic turn direction.",
             ha="center", fontsize=font_size - 1, style="italic", color="gray")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(output_path.replace(".pdf", f".{ext}"),
                    dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {output_path}")


def generate_figure4_summary() -> None:
    """Concise summary table: mean DTW + mean consistency per method."""
    dtw = pd.read_csv(f"{ANALYSIS_DIR}/dtw_comparison_table.csv")
    cons = pd.read_csv(f"{ANALYSIS_DIR}/consistency_comparison_table.csv")
    method_cols = [c for c in dtw.columns if c not in ("Label", "Turn")]

    rows = []
    for c in method_cols:
        key = _col_key(c)
        if key not in SHORT_NAMES:
            continue
        dtw_vals = dtw[c].dropna()
        cons_vals = cons[c].dropna()
        rows.append({
            "Method": SHORT_NAMES[key],
            "Mean DTW ↓": dtw_vals.mean(),
            "DTW Std": dtw_vals.std(),
            "# Trans. (DTW)": len(dtw_vals),
            "Mean Consist. ↓": cons_vals.mean(),
            "Consist. Std": cons_vals.std(),
            "# Trans. (Cons.)": len(cons_vals),
        })

    df = pd.DataFrame(rows).set_index("Method")

    # Determine which rows are "ours"
    ours_methods = {"DMC-Int", "DMC-Qpos", "DMC-EKP", "DMP-Int", "DMP-Qpos", "DMP-EKP"}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
    })

    n_rows = len(df) + 1
    n_cols = len(df.columns) + 1
    fig, ax = plt.subplots(figsize=(12, 0.45 * n_rows + 1.5))
    ax.axis("off")

    cell_text = []
    cell_colors = []

    best_dtw = df["Mean DTW ↓"].min()
    best_cons = df["Mean Consist. ↓"].min()

    for method, row in df.iterrows():
        row_text = []
        row_colors = []
        for col in df.columns:
            val = row[col]
            if col == "Mean DTW ↓":
                row_text.append(f"{val:.3f}")
                row_colors.append("#c8e6c9" if val == best_dtw else "white")
            elif col == "Mean Consist. ↓":
                row_text.append(f"{val:.3f}")
                row_colors.append("#c8e6c9" if val == best_cons else "white")
            elif col in ("DTW Std", "Consist. Std"):
                row_text.append(f"±{val:.3f}")
                row_colors.append("white")
            else:
                row_text.append(f"{int(val)}")
                row_colors.append("white")
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    row_labels = list(df.index)
    col_labels = list(df.columns)

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    table.scale(1.4, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", fontsize=10)
            cell.set_facecolor("#e3f2fd")
        if col == -1:
            cell.set_facecolor("#f5f5f5")
            cell.set_text_props(fontweight="bold" if row_labels[row - 1] in ours_methods else "normal")
            cell.set_width(cell.get_width() * 1.3)

    ax.set_title("Quantitative Comparison: DTW Distance and Internal Consistency",
                 fontsize=11, fontweight="bold", pad=15)
    fig.text(0.5, 0.01,
             "Green = best. Bold method names = ours. Lower is better for all metrics.",
             ha="center", fontsize=8, style="italic", color="gray")

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/figure4_summary"
    for ext in ("pdf", "png"):
        plt.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {out}.pdf")


def generate_appendix_dtw() -> None:
    """Full DTW table for Appendix."""
    dtw = pd.read_csv(f"{ANALYSIS_DIR}/dtw_comparison_table.csv")
    dtw = _order_columns(dtw)

    method_cols = [c for c in dtw.columns if c not in ("Label", "Turn")]
    dtw["Transition"] = dtw["Label"] + " [" + dtw["Turn"] + "]"
    pivot = dtw.set_index("Transition")[method_cols]
    pivot.columns = _rename_cols(method_cols)

    # Add mean row
    mean_row = pivot.mean(numeric_only=True)
    mean_row.name = "Mean"
    pivot = pd.concat([pivot, mean_row.to_frame().T])

    _render_table(
        pivot,
        f"{OUTPUT_DIR}/appendix_dtw_detailed.pdf",
        "DTW Distance to Ground Truth by Transition (↓ lower is better)",
        highlight_best=True,
        font_size=7,
    )


def generate_appendix_consistency() -> None:
    """Full consistency table for Appendix."""
    cons = pd.read_csv(f"{ANALYSIS_DIR}/consistency_comparison_table.csv")
    cons = _order_columns(cons)

    method_cols = [c for c in cons.columns if c not in ("Label", "Turn")]
    cons["Transition"] = cons["Label"] + " [" + cons["Turn"] + "]"
    pivot = cons.set_index("Transition")[method_cols]
    pivot.columns = _rename_cols(method_cols)

    mean_row = pivot.mean(numeric_only=True)
    mean_row.name = "Mean"
    pivot = pd.concat([pivot, mean_row.to_frame().T])

    _render_table(
        pivot,
        f"{OUTPUT_DIR}/appendix_consistency.pdf",
        "Internal Cluster Consistency by Transition (↓ lower is better)",
        highlight_best=True,
        font_size=7,
    )


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=== Figure 4: Summary Table ===")
    generate_figure4_summary()

    print("\n=== Appendix: Detailed DTW Table ===")
    generate_appendix_dtw()

    print("\n=== Appendix: Detailed Consistency Table ===")
    generate_appendix_consistency()

    print(f"\nAll tables saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
