"""LaTeX-rendered results table for the information-loss experiment.

Reads the MIResult list, aggregates over seeds per (method, feature_space),
and emits both ``information_loss_table.tex`` (paper-ready LaTeX with the
green/yellow/red highlight from the paper preamble) and an
``information_loss_table.png`` rendered from the TeX via matplotlib.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from topo_mimic.analysis.method_analysis.information_loss.metrics import MIResult


_METRIC_HIGHER_BETTER = {
    "H_T":         False,  # less capacity used at the same I is better
    "I_TY":        True,
    "NMI":         True,
    "AMI":         True,
    "efficiency":  True,
    "H_Y_given_T": False,  # lower information loss is better
    "lift":        True,
}


def _aggregate(results: list[MIResult]) -> list[dict]:
    by_key = defaultdict(list)
    for r in results:
        by_key[(r.method_name, r.family, r.feature_space)].append(r)

    rows = []
    for (name, family, feat), rs in by_key.items():
        H_T = np.array([r.H_T for r in rs])
        I_TY = np.array([r.I_TY for r in rs])
        NMI = np.array([r.NMI for r in rs])
        AMI = np.array([r.AMI for r in rs])
        eff = np.array([r.efficiency for r in rs])
        H_YgT = np.array([r.H_Y_given_T for r in rs])
        lift = I_TY - np.array([r.I_TY_shuffled for r in rs])
        rows.append({
            "method": name,
            "family": family,
            "feature_space": feat or "",
            "n_seeds": len(rs),
            "H_T_mean": float(H_T.mean()), "H_T_std": float(H_T.std()),
            "I_TY_mean": float(I_TY.mean()), "I_TY_std": float(I_TY.std()),
            "NMI_mean": float(NMI.mean()), "NMI_std": float(NMI.std()),
            "AMI_mean": float(AMI.mean()), "AMI_std": float(AMI.std()),
            "eff_mean": float(eff.mean()), "eff_std": float(eff.std()),
            "H_Y_given_T_mean": float(H_YgT.mean()), "H_Y_given_T_std": float(H_YgT.std()),
            "lift_mean": float(lift.mean()), "lift_std": float(lift.std()),
        })
    rows.sort(key=lambda r: (r["family"], r["method"]))
    return rows


def _rank_colors(values: np.ndarray, higher_better: bool) -> list[str]:
    """Return per-row LaTeX colour for best / second / worst."""
    order = np.argsort(-values) if higher_better else np.argsort(values)
    colors = [""] * len(values)
    if len(values) >= 1:
        colors[order[0]] = "bestgreen"
    if len(values) >= 2:
        colors[order[1]] = "secondyellow"
    if len(values) >= 3:
        colors[order[-1]] = "worstred"
    return colors


def render_latex_table(rows: list[dict], output_tex: Path, granularity: str) -> str:
    metrics = [
        ("H_T_mean",        r"$H(T)$",          "H_T"),
        ("I_TY_mean",       r"$I(T;Y)$",        "I_TY"),
        ("lift_mean",       r"lift over floor", "lift"),
        ("NMI_mean",        r"NMI",             "NMI"),
        ("AMI_mean",        r"AMI",             "AMI"),
        ("eff_mean",        r"efficiency",      "efficiency"),
        ("H_Y_given_T_mean",r"$H(Y\mid T)$",    "H_Y_given_T"),
    ]
    if not rows:
        output_tex.write_text("")
        return ""

    values_mat = {k: np.array([r[k] for r in rows]) for k, _, _ in metrics}
    stds_mat = {k: np.array([r[k.replace("_mean", "_std")] for r in rows])
                for k, _, _ in metrics}
    color_mat = {
        rank_key: _rank_colors(values_mat[k], _METRIC_HIGHER_BETTER[rank_key])
        for k, _, rank_key in metrics
    }

    n_cols = 1 + len(metrics)
    lines = []
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "r" * len(metrics) + "}")
    lines.append(r"\toprule")
    headers = ["Method"] + [m[1] for m in metrics]
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for i, row in enumerate(rows):
        cells = [row["method"]]
        for k, _, rank_key in metrics:
            v = values_mat[k][i]
            s = stds_mat[k][i]
            cell = rf"\meanstd{{{v:.3f}}}{{{s:.3f}}}" if s > 0 else f"{v:.3f}"
            color = color_mat[rank_key][i]
            if color:
                cell = rf"\textcolor{{{color}}}{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")

    tex = "\n".join(lines)
    output_tex.write_text(tex)
    return tex


# ---------------------------------------------------------------------------
# PNG via matplotlib
# ---------------------------------------------------------------------------


def render_png_table(rows: list[dict], output_png: Path) -> None:
    if not rows:
        return

    metrics = [
        ("H_T_mean",        "H(T)",             "H_T"),
        ("I_TY_mean",       "I(T;Y)",           "I_TY"),
        ("lift_mean",       "lift",             "lift"),
        ("NMI_mean",        "NMI",              "NMI"),
        ("AMI_mean",        "AMI",              "AMI"),
        ("eff_mean",        "I/H(T)",           "efficiency"),
        ("H_Y_given_T_mean","H(Y|T)",           "H_Y_given_T"),
    ]
    BEST = "#2E7D32"
    SECOND = "#F9A825"
    WORST = "#C62828"

    values_mat = {k: np.array([r[k] for r in rows]) for k, _, _ in metrics}
    color_grid = []
    text_grid = []
    for i, row in enumerate(rows):
        row_colors = []
        row_text = [row["method"]]
        for k, _, rank_key in metrics:
            v = values_mat[k][i]
            s = row[k.replace("_mean", "_std")]
            row_text.append(f"{v:.2f}" + (f"\n±{s:.2f}" if s > 0 else ""))
            order = (np.argsort(-values_mat[k]) if _METRIC_HIGHER_BETTER[rank_key]
                     else np.argsort(values_mat[k]))
            if i == order[0]:
                row_colors.append(BEST)
            elif len(order) > 1 and i == order[1]:
                row_colors.append(SECOND)
            elif i == order[-1]:
                row_colors.append(WORST)
            else:
                row_colors.append("white")
        color_grid.append([None] + row_colors)
        text_grid.append(row_text)

    n_rows = len(rows)
    n_cols = 1 + len(metrics)

    with mpl.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
    }):
        fig, ax = plt.subplots(figsize=(1.0 + 1.05 * n_cols, 0.45 + 0.42 * n_rows))
        ax.set_axis_off()

        headers = ["Method"] + [m[1] for m in metrics]
        tbl = ax.table(
            cellText=[headers] + text_grid,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for col in range(n_cols):
            tbl.auto_set_column_width(col)

        for c in range(n_cols):
            cell = tbl[(0, c)]
            cell.set_facecolor("#EFEFEF")
            cell.set_text_props(weight="bold", color="#222")
            cell.set_height(0.075)

        for r in range(n_rows):
            for c in range(n_cols):
                cell = tbl[(r + 1, c)]
                cell.set_height(0.08)
                bg = color_grid[r][c]
                if bg is not None and bg != "white":
                    cell.set_facecolor(bg)
                    cell.set_text_props(color="white", weight="bold")
                else:
                    cell.set_facecolor("white")

        fig.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


def render_all(out_dir: Path, results: list[MIResult], granularity: str) -> None:
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    rows = _aggregate(results)
    render_latex_table(rows, figs_dir / "information_loss_table.tex", granularity)
    render_png_table(rows, figs_dir / "information_loss_table.png")
