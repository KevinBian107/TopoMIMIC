"""Nature-style figure for the variance-frame tradeoff."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "DM-cycle": "#1F4E79",   # deep blue
    "DM-path":  "#2E86AB",   # teal
    "Random":   "#BDBDBD",   # grey null
}
SPACE_TITLE = {"intention": "Intention", "qpos": "Qpos", "ekp": "EKP"}

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def _binned_median(x: np.ndarray, y: np.ndarray, edges: np.ndarray):
    """Median + IQR of y within bins of x; returns (centers, med, q1, q3)."""
    idx = np.digitize(x, edges) - 1
    centers, med, q1, q3 = [], [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 5:
            continue
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        med.append(np.median(y[m])); q1.append(np.percentile(y[m], 25)); q3.append(np.percentile(y[m], 75))
    return np.array(centers), np.array(med), np.array(q1), np.array(q3)


def render(df: pd.DataFrame, out_dir: Path) -> Path:
    spaces = [s for s in ("intention", "qpos", "ekp") if s in df["space"].unique()]
    with mpl.rc_context(_RC):
        fig, axes = plt.subplots(2, len(spaces), figsize=(2.7 * len(spaces), 5.0),
                                 squeeze=False)
        for j, sp in enumerate(spaces):
            sub = df[df["space"] == sp]
            nmax = np.percentile(sub["n_frames"], 99) if len(sub) else 1
            edges = np.linspace(sub["n_frames"].min(), nmax, 11)

            # --- top: variance-frame tradeoff (paths, cycles, random null) ---
            ax = axes[0][j]
            for typ in ("DM-cycle", "DM-path"):
                t = sub[sub["type"] == typ]
                if t.empty:
                    continue
                x, y = t["n_frames"].to_numpy(), t["var_captured"].to_numpy()
                ax.scatter(x, y, s=4, c=COLORS[typ], alpha=0.12, linewidths=0, zorder=1)
                c, m, q1, q3 = _binned_median(x, y, edges)
                ax.fill_between(c, q1, q3, color=COLORS[typ], alpha=0.18, lw=0, zorder=2)
                ax.plot(c, m, color=COLORS[typ], lw=1.8, label=typ, zorder=3)
            # random null (over all primitives in this space)
            x, y = sub["n_frames"].to_numpy(), sub["var_random"].to_numpy()
            c, m, _, _ = _binned_median(x, y, edges)
            ax.plot(c, m, color=COLORS["Random"], lw=1.6, ls="--", label="Random window", zorder=2)
            ax.set_title(SPACE_TITLE.get(sp, sp), fontsize=9, pad=4)
            if j == 0:
                ax.set_ylabel("qpos variance captured (rad$^2$)")
            ax.set_xlabel("frames used")
            if j == len(spaces) - 1:
                ax.legend(loc="lower right", handlelength=1.3)

            # --- bottom: efficiency ratio (captured / matched random) ---
            ax2 = axes[1][j]
            data, cols, labs = [], [], []
            for typ in ("DM-cycle", "DM-path"):
                t = sub[sub["type"] == typ]["efficiency"].dropna()
                if len(t):
                    data.append(t.to_numpy()); cols.append(COLORS[typ]); labs.append(typ.replace("DM-", ""))
            if data:
                bp = ax2.boxplot(data, positions=range(len(data)), widths=0.6,
                                 patch_artist=True, showfliers=False,
                                 medianprops=dict(color="white", lw=1.3),
                                 whiskerprops=dict(color="0.3", lw=0.8),
                                 capprops=dict(color="0.3", lw=0.8),
                                 boxprops=dict(lw=0.8, edgecolor="0.3"))
                for patch, col in zip(bp["boxes"], cols):
                    patch.set_facecolor(col)
                ax2.set_xticks(range(len(labs))); ax2.set_xticklabels(labs)
            ax2.axhline(1.0, color="0.5", lw=0.9, ls=":")  # parity with random
            if j == 0:
                ax2.set_ylabel("efficiency\n(captured / random)")
            # annotate median efficiency + fraction above 1
            for k, typ in enumerate(("DM-cycle", "DM-path")):
                t = sub[sub["type"] == typ]["efficiency"].dropna()
                if len(t):
                    ax2.text(k, ax2.get_ylim()[1], f"{np.median(t):.2f}",
                             ha="center", va="bottom", fontsize=6.5, color=COLORS[typ])

        # panel letters
        for lab, ax in zip("abcdef", [a for row in axes for a in row]):
            ax.text(-0.18, 1.06, lab, transform=ax.transAxes, fontsize=11,
                    fontweight="bold", va="bottom", ha="left")
        fig.tight_layout(w_pad=1.2, h_pad=1.6)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / "variance_frame_tradeoff"
        fig.savefig(str(base) + ".png")
        fig.savefig(str(base) + ".pdf")
        plt.close(fig)
    return Path(str(base) + ".png")


def render_appendix(df: pd.DataFrame, out_dir: Path) -> Path:
    """Appendix figure: efficiency ratio (captured / matched random window) per
    feature space, DM-cycle vs DM-path, labelled a-c."""
    spaces = [s for s in ("intention", "qpos", "ekp") if s in df["space"].unique()]
    with mpl.rc_context(_RC):
        fig, axes = plt.subplots(1, len(spaces), figsize=(2.5 * len(spaces), 2.7),
                                 squeeze=False)
        for j, sp in enumerate(spaces):
            sub = df[df["space"] == sp]
            ax = axes[0][j]
            data, cols, labs = [], [], []
            for typ in ("DM-cycle", "DM-path"):
                t = sub[sub["type"] == typ]["efficiency"].dropna()
                if len(t):
                    data.append(t.to_numpy()); cols.append(COLORS[typ]); labs.append(typ.replace("DM-", ""))
            if data:
                bp = ax.boxplot(data, positions=range(len(data)), widths=0.6,
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color="white", lw=1.3),
                                whiskerprops=dict(color="0.3", lw=0.8),
                                capprops=dict(color="0.3", lw=0.8),
                                boxprops=dict(lw=0.8, edgecolor="0.3"))
                for patch, col in zip(bp["boxes"], cols):
                    patch.set_facecolor(col)
                ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)
            ax.axhline(1.0, color="0.5", lw=0.9, ls=":")
            ax.set_title(SPACE_TITLE.get(sp, sp), fontsize=9, pad=4)
            if j == 0:
                ax.set_ylabel("efficiency\n(captured / random)")
            for k, typ in enumerate(("DM-cycle", "DM-path")):
                t = sub[sub["type"] == typ]["efficiency"].dropna()
                if len(t):
                    ax.text(k, ax.get_ylim()[1], f"{np.median(t):.2f}",
                            ha="center", va="bottom", fontsize=6.5, color=COLORS[typ])
        for lab, ax in zip("abc", axes[0]):
            ax.text(-0.20, 1.06, lab, transform=ax.transAxes, fontsize=11,
                    fontweight="bold", va="bottom", ha="left")
        fig.tight_layout(w_pad=1.2)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / "variance_frame_efficiency_appendix"
        fig.savefig(str(base) + ".png")
        fig.savefig(str(base) + ".pdf")
        plt.close(fig)
    return Path(str(base) + ".png")
