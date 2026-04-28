#!/usr/bin/env python3
"""Build paper/appendix_figures/appendix_asymmetry.png.

Three violin panels (one per feature space) of bootstrap-mean
cycle-counts-per-100-frames for asymmetric vs. symmetric A->B->A triplets.
Reads the cached observations CSVs from the behavioral_asymmetry analysis.
Each panel sits on its own y-axis because EKP magnitudes are an order of
magnitude smaller than intention / qpos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ASYM_DIR = REPO / "outputs/analysis/method_analysis/behavioral_asymmetry"
OUT_PATH = REPO / "paper/appendix_figures/appendix_asymmetry.png"

ACTIVE_COLOR = "#E6A382"  # asymmetric (orange)
QUIET_COLOR  = "#7E9FCB"  # symmetric (blue)


def _bootstrap_means(values: np.ndarray, n_boot: int = 1000,
                     seed: int = 0) -> np.ndarray:
    if values.size == 0:
        return np.array([])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True)
    return draws.mean(axis=1)


def render_panel(ax: plt.Axes, feature_space: str, show_ylabel: bool) -> None:
    df = pd.read_csv(ASYM_DIR / f"observations_{feature_space}.csv")
    s_raw = df[df["symmetry"] == "symmetric"]["cycle_count_norm"].to_numpy()
    a_raw = df[df["symmetry"] == "asymmetric"]["cycle_count_norm"].to_numpy()
    s = _bootstrap_means(s_raw)
    a = _bootstrap_means(a_raw)

    parts = ax.violinplot([a, s], positions=[0, 1], widths=0.8,
                          showextrema=False)
    for pc, color in zip(parts["bodies"], [ACTIVE_COLOR, QUIET_COLOR]):
        pc.set_facecolor(color)
        pc.set_edgecolor("#222222")
        pc.set_linewidth(0.7)
        pc.set_alpha(1.0)

    for xpos, vals in zip([0, 1], [a, s]):
        m = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        ax.errorbar(xpos, m, yerr=sd, fmt="o", color="#222222",
                    markersize=3.5, elinewidth=0.8, capsize=2.5,
                    capthick=0.8, zorder=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["asym.", "sym."], fontsize=11)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xlabel(r"A$\to$B$\to$A symmetry", fontsize=12)
    ax.set_title(feature_space, fontsize=13, fontweight="bold", pad=4)
    if show_ylabel:
        ax.set_ylabel("Cycles per 100 frames", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.tick_params(direction="out", length=3.0, width=0.6, labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), dpi=200)
    for ax, sp in zip(axes, ["intention", "qpos", "ekp"]):
        render_panel(ax, sp, show_ylabel=(ax is axes[0]))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.18, wspace=0.30)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
