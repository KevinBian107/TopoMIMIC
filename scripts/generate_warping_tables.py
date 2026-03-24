#!/usr/bin/env python3
"""Generate DTW warping heaviness tables across all methods.

Shows how much temporal warping DTW needs to achieve its alignment scores.
The DTW-to-frame-sync ratio exposes methods that rely on temporal rearrangement
to match postures at wrong timepoints.

Outputs:
  tables/appendix_warping_*.tex        — LaTeX tables (full + simplified)
  tables/appendix_warping_*.{pdf,png}  — Rendered table images
  outputs/warping_analysis/*.csv       — Raw data
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topo_mimic.benchmark.analysis.data_loading import (
    load_experiment_results,
    load_labels_yaml,
)
from topo_mimic.benchmark.analysis.metrics import (
    run_warping_analysis,
    WarpingResult,
)

EXPERIMENT_DIR = "outputs/benchmark_experiment_transitions"
DATA_PATH = "data/rodent_ar.h5"
DATA_KEY = "qpos_rollout"
BEHAVIOR_LABELS_CSV = "behavior_labels.csv"
OUTPUT_DIR = "outputs/warping_analysis"

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

OURS = {"DMC-Int", "DMC-Qpos", "DMC-EKP", "DMP-Int", "DMP-Qpos", "DMP-EKP"}


def load_data() -> np.ndarray:
    with h5py.File(DATA_PATH, "r") as f:
        return np.array(f[DATA_KEY])


def run_all_methods(data: np.ndarray) -> dict[str, list[WarpingResult]]:
    all_results = load_experiment_results(EXPERIMENT_DIR)
    warping_results = {}

    for method_name in METHOD_ORDER:
        if method_name not in all_results:
            print(f"Skipping {method_name} (not found)")
            continue

        method_data = all_results[method_name]
        seed = 0
        if seed not in method_data:
            print(f"Skipping {method_name} (no seed 0)")
            continue

        result = method_data[seed]
        cluster_dict = result.get("cluster_dict", {})
        if not cluster_dict:
            continue

        method_dir = os.path.join(EXPERIMENT_DIR, method_name)
        cluster_labels, turn_directions = load_labels_yaml(method_dir, seed)
        if not cluster_labels:
            cluster_labels = {i: f"C{i}" for i in cluster_dict.keys()}

        short_name = SHORT_NAMES.get(method_name, method_name)
        print(f"\n=== {short_name} ({method_name}) ===")

        results = run_warping_analysis(
            cluster_dict=cluster_dict,
            cluster_labels=cluster_labels,
            turn_directions=turn_directions,
            behavior_labels_csv=BEHAVIOR_LABELS_CSV,
            data=data,
            method_name=method_name,
            max_samples=20,
            max_comparisons=50,
            short_threshold=10,
            merge_rear=True,
            skip_root_dims=7,
            min_seq_length=3,
        )

        warping_results[short_name] = results
        for r in results:
            turn = f" [{r.turn_direction}]" if r.turn_direction else ""
            print(f"  {r.label}{turn}: DTW={r.mean_dtw:.3f} "
                  f"sync={r.mean_sync_dist:.3f} "
                  f"ratio={r.mean_dtw_sync_ratio:.3f} "
                  f"non-diag={r.mean_non_diag_frac:.3f}")

    return warping_results


def build_full_table(
    warping_results: dict[str, list[WarpingResult]],
    metric: str = "mean_non_diag_frac",
) -> pd.DataFrame:
    all_keys = set()
    for results in warping_results.values():
        for r in results:
            turn = r.turn_direction or "none"
            all_keys.add((r.label, turn))

    all_keys = sorted(all_keys)
    methods = [m for m in SHORT_NAMES.values() if m in warping_results]

    rows = []
    for label, turn in all_keys:
        row = {"Label": label, "Turn": turn}
        for method in methods:
            found = None
            for r in warping_results.get(method, []):
                r_turn = r.turn_direction or "none"
                if r.label == label and r_turn == turn:
                    found = r
                    break
            row[method] = getattr(found, metric) if found else None
        rows.append(row)

    return pd.DataFrame(rows)


def build_simple_table(
    warping_results: dict[str, list[WarpingResult]],
    metric: str = "mean_non_diag_frac",
) -> pd.DataFrame:
    methods = [m for m in SHORT_NAMES.values() if m in warping_results]

    method_by_label = {}
    for method in methods:
        by_label = defaultdict(list)
        for r in warping_results.get(method, []):
            by_label[r.label].append(getattr(r, metric))
        method_by_label[method] = {k: np.mean(v) for k, v in by_label.items()}

    all_labels = set()
    for d in method_by_label.values():
        all_labels.update(d.keys())
    all_labels = sorted(all_labels)

    rows = []
    for label in all_labels:
        row = {"Transition": label}
        for method in methods:
            val = method_by_label[method].get(label)
            row[method] = val
        rows.append(row)

    return pd.DataFrame(rows)


def format_latex_full(
    df: pd.DataFrame,
    metric_name: str,
    higher_is_worse: bool = True,
) -> str:
    methods = [c for c in df.columns if c not in ("Label", "Turn")]
    ours_marks = {m: "$^*$" if m in OURS else "" for m in methods}

    col_spec = "l" + "c" * len(methods)
    header = " & ".join(f"{m}{ours_marks[m]}" for m in methods)

    arrow = r"$\uparrow$" if higher_is_worse else r"$\downarrow$"
    direction = "more" if higher_is_worse else "less"

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{\\textbf{{{metric_name} ({arrow} = {direction} warping).}} "
        r"\textbf{Bold} = worst per row; \underline{underline} = second worst. "
        r"--- = not discovered. $^*$Ours.}",
        f"\\label{{tab:warping-{metric_name.lower().replace(' ', '-')}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.9}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"Transition & {header} \\\\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        label = row["Label"]
        turn = row["Turn"]
        display = f"{label} [{turn}]" if turn and turn != "none" else label

        vals = {m: row[m] for m in methods if pd.notna(row[m])}

        if vals:
            sorted_vals = sorted(vals.values(), reverse=higher_is_worse)
            worst = sorted_vals[0]
            second = sorted_vals[1] if len(sorted_vals) > 1 else None
        else:
            worst, second = None, None

        cells = []
        for m in methods:
            v = row[m]
            if pd.isna(v):
                cells.append("---")
            else:
                formatted = f"{v:.3f}"
                if v == worst:
                    formatted = f"\\textbf{{{formatted}}}"
                elif second is not None and v == second:
                    formatted = f"\\underline{{{formatted}}}"
                cells.append(formatted)

        lines.append(f"{display} & {' & '.join(cells)} \\\\")

    mean_vals = {m: df[m].dropna().mean() for m in methods if len(df[m].dropna()) > 0}
    if mean_vals:
        sorted_means = sorted(mean_vals.values(), reverse=higher_is_worse)
        worst_mean = sorted_means[0]
        second_mean = sorted_means[1] if len(sorted_means) > 1 else None

        mean_cells = []
        for m in methods:
            if m in mean_vals:
                v = mean_vals[m]
                formatted = f"{v:.3f}"
                if v == worst_mean:
                    formatted = f"\\textbf{{{formatted}}}"
                elif second_mean is not None and v == second_mean:
                    formatted = f"\\underline{{{formatted}}}"
                mean_cells.append(formatted)
            else:
                mean_cells.append("---")
        lines.append(r"\midrule")
        lines.append(f"\\textit{{Mean}} & {' & '.join(mean_cells)} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def format_latex_simple(
    df: pd.DataFrame,
    metric_name: str,
    higher_is_worse: bool = True,
) -> str:
    methods = [c for c in df.columns if c != "Transition"]
    ours_marks = {m: "$^*$" if m in OURS else "" for m in methods}

    col_spec = "l" + "c" * len(methods)
    header = " & ".join(f"{m}{ours_marks[m]}" for m in methods)

    lines = [
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f" & {header} \\\\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        vals = {m: row[m] for m in methods if pd.notna(row[m])}

        if vals:
            sorted_vals = sorted(vals.values(), reverse=higher_is_worse)
            worst = sorted_vals[0]
            second = sorted_vals[1] if len(sorted_vals) > 1 else None
        else:
            worst, second = None, None

        cells = []
        for m in methods:
            v = row[m]
            if pd.isna(v):
                cells.append("---")
            else:
                formatted = f"{v:.3f}"
                if v == worst:
                    formatted = f"\\textbf{{{formatted}}}"
                elif second is not None and v == second:
                    formatted = f"\\underline{{{formatted}}}"
                cells.append(formatted)

        lines.append(f"{row['Transition']} & {' & '.join(cells)} \\\\")

    mean_vals = {m: df[m].dropna().mean() for m in methods if len(df[m].dropna()) > 0}
    if mean_vals:
        sorted_means = sorted(mean_vals.values(), reverse=higher_is_worse)
        worst_mean = sorted_means[0]
        second_mean = sorted_means[1] if len(sorted_means) > 1 else None

        mean_cells = []
        for m in methods:
            if m in mean_vals:
                v = mean_vals[m]
                formatted = f"{v:.3f}"
                if v == worst_mean:
                    formatted = f"\\textbf{{{formatted}}}"
                elif second_mean is not None and v == second_mean:
                    formatted = f"\\underline{{{formatted}}}"
                mean_cells.append(formatted)
            else:
                mean_cells.append("---")
        lines.append(r"\midrule")
        lines.append(f"\\textit{{Mean}} & {' & '.join(mean_cells)} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def render_table_png(
    df: pd.DataFrame,
    output_path: str,
    title: str,
    higher_is_worse: bool = True,
    font_size: int = 8,
    dpi: int = 300,
) -> None:
    """Render a DataFrame as a publication-style PNG/PDF table."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
    })

    n_rows = len(df) + 2
    n_cols = len(df.columns) + 1
    fig_w = max(10, 1.5 * n_cols + 2)
    fig_h = max(3, 0.38 * n_rows + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    cell_text = []
    cell_colors = []

    for _, row in df.iterrows():
        row_vals = row[numeric_cols].dropna()
        if higher_is_worse:
            row_worst = row_vals.max() if len(row_vals) > 0 else None
        else:
            row_worst = row_vals.min() if len(row_vals) > 0 else None

        row_text = []
        row_colors = []
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                row_text.append("—")
                row_colors.append("white")
            elif col in numeric_cols:
                row_text.append(f"{val:.3f}")
                if row_worst is not None and val == row_worst:
                    row_colors.append("#ffcdd2")
                else:
                    row_colors.append("white")
            else:
                row_text.append(str(val))
                row_colors.append("white")
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    mean_row_text = []
    mean_row_colors = []
    mean_numeric = {}
    for col in df.columns:
        if col in numeric_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                m = vals.mean()
                mean_numeric[col] = m
                mean_row_text.append(f"{m:.3f}")
            else:
                mean_row_text.append("—")
            mean_row_colors.append("#f5f5f5")
        else:
            mean_row_text.append("Mean")
            mean_row_colors.append("#f5f5f5")

    if mean_numeric:
        if higher_is_worse:
            worst_col_val = max(mean_numeric.values())
        else:
            worst_col_val = min(mean_numeric.values())
        for i, col in enumerate(df.columns):
            if col in mean_numeric and mean_numeric[col] == worst_col_val:
                mean_row_colors[i] = "#ffcdd2"

    cell_text.append(mean_row_text)
    cell_colors.append(mean_row_colors)

    row_labels = [str(i) for i in df.index] + [""]
    col_labels = list(df.columns)

    for i, c in enumerate(col_labels):
        if c in OURS:
            col_labels[i] = f"{c}*"

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
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

    ax.set_title(title, fontsize=font_size + 3, fontweight="bold", pad=15)

    direction = "Red = worst per row." if higher_is_worse else "Red = worst per row."
    fig.text(0.5, 0.01,
             f"{direction} *Ours. — = not discovered.",
             ha="center", fontsize=font_size - 1, style="italic", color="gray")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(output_path.replace(".png", f".{ext}"),
                    dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Rendered {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading trajectory data...")
    data = load_data()
    print(f"Data shape: {data.shape}")

    print("\nRunning warping analysis across all methods...")
    warping_results = run_all_methods(data)

    metrics = [
        ("mean_non_diag_frac", "Non-Diagonal Fraction", True),
        ("mean_temporal_dev", "Temporal Deviation", True),
        ("mean_max_run", "Max Run Length", True),
        ("mean_dtw_sync_ratio", "DTW-to-Frame-Sync Ratio", False),
    ]

    for metric, label, higher_is_worse in metrics:
        print(f"\n=== Building tables for: {label} ===")

        full_df = build_full_table(warping_results, metric)
        simple_df = build_simple_table(warping_results, metric)

        safe_metric = metric.replace("mean_", "")

        full_csv = os.path.join(OUTPUT_DIR, f"warping_{safe_metric}_full.csv")
        simple_csv = os.path.join(OUTPUT_DIR, f"warping_{safe_metric}_simple.csv")
        full_df.to_csv(full_csv, index=False, float_format="%.4f")
        simple_df.to_csv(simple_csv, index=False, float_format="%.4f")
        print(f"  Saved {full_csv}")
        print(f"  Saved {simple_csv}")

        full_tex = format_latex_full(full_df, label, higher_is_worse)
        simple_tex = format_latex_simple(simple_df, label, higher_is_worse)

        full_tex_path = os.path.join(OUTPUT_DIR, f"appendix_warping_{safe_metric}_full.tex")
        simple_tex_path = os.path.join(OUTPUT_DIR, f"appendix_warping_{safe_metric}_simple.tex")
        with open(full_tex_path, "w") as f:
            f.write(full_tex)
        with open(simple_tex_path, "w") as f:
            f.write(simple_tex)
        print(f"  Saved {full_tex_path}")
        print(f"  Saved {simple_tex_path}")

        simple_display = simple_df.copy()
        simple_display = simple_display.set_index("Transition")
        render_table_png(
            simple_display,
            os.path.join(OUTPUT_DIR, f"appendix_warping_{safe_metric}_simple.png"),
            f"{label} by Transition Type",
            higher_is_worse=higher_is_worse,
        )

        full_display = full_df.copy()
        full_display["Transition"] = full_display["Label"]
        mask = (full_display["Turn"] != "none") & full_display["Turn"].notna()
        full_display.loc[mask, "Transition"] = (
            full_display.loc[mask, "Label"] + " [" + full_display.loc[mask, "Turn"] + "]"
        )
        full_display = full_display.drop(columns=["Label", "Turn"]).set_index("Transition")
        render_table_png(
            full_display,
            os.path.join(OUTPUT_DIR, f"appendix_warping_{safe_metric}_full.png"),
            f"{label} by Transition (Full)",
            higher_is_worse=higher_is_worse,
        )

    print("\n=== Summary ===")
    for metric, label, _ in metrics:
        simple = build_simple_table(warping_results, metric)
        methods = [c for c in simple.columns if c != "Transition"]
        print(f"\n{label}:")
        for m in methods:
            vals = simple[m].dropna()
            if len(vals) > 0:
                print(f"  {m}: {vals.mean():.3f} (n={len(vals)} transitions)")

    print("\nDone.")


if __name__ == "__main__":
    main()
