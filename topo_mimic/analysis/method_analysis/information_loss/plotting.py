"""Figures for the information-loss experiment.

Three outputs:
    ib_plane.{png,pdf}        -- PRIMARY figure: H(T) vs I(T;Y) scatter,
                                 one marker per method+seed, family colours,
                                 95%-CI ellipses, theoretical bounds drawn
                                 as light reference lines. Nature single-
                                 column polish.
    mi_bar.{png,pdf}          -- companion bar comparison of I(T;Y), NMI,
                                 AMI side by side. Bootstrap-CI whiskers.
    joint_heatmaps.{png,pdf}  -- per-method joint P(Y|T) heatmaps for the
                                 appendix.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from topo_mimic.analysis.method_analysis.information_loss.labels import TransitionLabels
from topo_mimic.analysis.method_analysis.information_loss.metrics import MIResult


_FAMILY_COLOR = {
    "DMC":    "#1f77b4",
    "DMP":    "#17a2b8",
    "KPMS":   "#2ca02c",
    "VAE":    "#ff7f0e",
    "Mapper": "#d62728",
}

_FAMILY_MARKER = {
    "DMC":    "o",
    "DMP":    "o",
    "KPMS":   "o",
    "VAE":    "o",
    "Mapper": "o",
}

_FEATURE_LINESTYLE = {
    "intention": "-",
    "qpos":      "--",
    "ekp":       ":",
    None:        "-",
}


def _rcparams():
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }


def _save_both(fig: plt.Figure, base: Path) -> None:
    fig.savefig(str(base) + ".png")
    fig.savefig(str(base) + ".pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# IB plane (primary)
# ---------------------------------------------------------------------------


def _ci_ellipse(
    xs: np.ndarray,
    ys: np.ndarray,
    n_std: float = 1.96,
    **kwargs,
) -> Ellipse | None:
    if xs.size < 3 or ys.size < 3:
        return None
    cov = np.cov(np.stack([xs, ys]))
    if np.any(np.isnan(cov)) or np.linalg.matrix_rank(cov) < 2:
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    width, height = 2.0 * n_std * np.sqrt(np.maximum(eigvals, 0))
    return Ellipse(
        xy=(float(xs.mean()), float(ys.mean())),
        width=float(width), height=float(height),
        angle=angle, **kwargs,
    )


def plot_ib_plane(
    results: list[MIResult],
    labels: TransitionLabels,
    output_base: Path,
) -> None:
    if not results:
        return
    with mpl.rc_context(_rcparams()):
        fig, ax = plt.subplots(figsize=(2.9, 3.1))

        H_T = np.array([r.H_T for r in results])
        I_TY = np.array([r.I_TY for r in results])
        x_lo = max(0.0, float(np.min(H_T)) - 0.2)
        x_hi = float(np.max(H_T)) + 0.2
        y_lo = max(0.0, float(np.min(I_TY)) - 0.05)
        y_hi = float(np.max(I_TY)) * 1.18 + 0.02

        from collections import defaultdict
        by_method = defaultdict(list)
        for r in results:
            by_method[r.method_name].append(r)

        legend_handles = {}
        for method_name, rs in by_method.items():
            family = rs[0].family
            color = _FAMILY_COLOR.get(family, "0.4")
            xs = np.array([r.H_T for r in rs])
            ys = np.array([r.I_TY for r in rs])

            ax.scatter(xs, ys,
                       marker="o", s=70,
                       facecolor=color, edgecolor="white",
                       linewidth=1.0, alpha=0.92, zorder=4)

            if family not in legend_handles:
                legend_handles[family] = ax.scatter(
                    [], [], marker="o", s=70,
                    facecolor=color, edgecolor="white",
                    linewidth=1.0, label=family,
                )

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(r"$H(T)$ — representation capacity (bits)")
        ax.set_ylabel(r"$I(T;\,Y)$ — transition info preserved (bits)")
        ax.legend(handles=list(legend_handles.values()),
                  loc="upper left", fontsize=7, handlelength=1.0,
                  borderpad=0.4, labelspacing=0.4)
        _save_both(fig, output_base)


# ---------------------------------------------------------------------------
# Bar chart (companion, appendix)
# ---------------------------------------------------------------------------


def plot_mi_bar(
    results: list[MIResult],
    output_base: Path,
    metrics: tuple[str, ...] = ("I_TY", "NMI", "AMI"),
) -> None:
    """Box-and-whisker per family for each metric.

    Each family aggregates all of its (method, feature-space, seed) points
    into one distribution: DMC (3 feature spaces), DMP (3), KPMS (2 variants
    x 3 seeds = 6), VAE (1 x 3 seeds = 3), Mapper (3 feature spaces x 3
    seeds = 9). Family order is the same in every panel.
    """
    if not results:
        return
    family_order = ["DMC", "DMP", "KPMS", "VAE", "Mapper"]
    attr_map = {"I_TY": "I_TY", "NMI": "NMI", "AMI": "AMI"}
    ylabel_map = {"I_TY": r"$I(T;Y)$ (bits)", "NMI": "NMI", "AMI": "AMI"}
    title_map = {"I_TY": r"$I(T;Y)$", "NMI": "NMI", "AMI": "AMI"}

    with mpl.rc_context(_rcparams()):
        fig, axes = plt.subplots(
            1, len(metrics),
            figsize=(2.4 + 1.7 * len(metrics), 3.4),
            sharey=False,
        )
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            box_data: list[np.ndarray] = []
            box_colors: list[str] = []
            for fam in family_order:
                fam_vals = np.array([
                    getattr(r, attr_map[metric])
                    for r in results
                    if r.family == fam
                ])
                box_data.append(fam_vals if fam_vals.size else np.array([np.nan]))
                box_colors.append(_FAMILY_COLOR.get(fam, "0.4"))

            x_step = 1.4
            positions = np.arange(len(family_order)) * x_step

            bp = ax.boxplot(
                box_data,
                positions=positions,
                widths=0.7,
                patch_artist=True,
                showmeans=False,
                showfliers=True,
                medianprops=dict(color="white", linewidth=1.4),
                whiskerprops=dict(color="0.25", linewidth=0.9),
                capprops=dict(color="0.25", linewidth=0.9),
                flierprops=dict(marker="o", markersize=3,
                                markerfacecolor="0.4",
                                markeredgecolor="none", alpha=0.7),
            )
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_edgecolor("0.25")
                patch.set_linewidth(0.7)
                patch.set_alpha(0.85)

            for i, (vals, color) in enumerate(zip(box_data, box_colors)):
                if vals.size and not np.all(np.isnan(vals)):
                    jitter = np.random.RandomState(0).uniform(
                        -0.15, 0.15, size=vals.size
                    )
                    ax.scatter(np.full_like(vals, positions[i]) + jitter,
                               vals, s=10, color="0.15",
                               alpha=0.6, linewidth=0, zorder=3)

            ax.set_xticks(positions)
            ax.set_xticklabels(family_order, fontsize=8)
            ax.set_xlim(positions[0] - x_step * 0.7,
                        positions[-1] + x_step * 0.7)
            ax.set_ylabel(ylabel_map[metric])
            ax.set_ylim(bottom=0)
            ax.set_title(title_map[metric])
            ax.margins(x=0.02)

        fig.tight_layout()
        _save_both(fig, output_base)


# ---------------------------------------------------------------------------
# Joint heatmaps (appendix)
# ---------------------------------------------------------------------------


def plot_joint_heatmaps(
    results: list[MIResult],
    labels: TransitionLabels,
    joint_dir: Path,
    output_base: Path,
    seed_for_panel: int | None = 0,
) -> None:
    panels = []
    for r in results:
        if r.seed not in (None, seed_for_panel):
            continue
        npz_path = (joint_dir
                    / f"{r.method_name.replace('-', '_').lower()}_seed{0 if r.seed is None else r.seed}.npz")
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=True)
        joint = data["joint"]
        T_vals = data["T_values"]
        Y_vals = data["Y_values"]
        class_names = list(data["class_names"]) if "class_names" in data else labels.class_names
        panels.append((r.method_name, r.family, joint, T_vals, Y_vals, class_names))

    if not panels:
        return

    n = len(panels)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))

    with mpl.rc_context(_rcparams()):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.8 * n_cols, 3.4 * n_rows),
            gridspec_kw=dict(hspace=0.95, wspace=0.55),
            constrained_layout=False,
        )
        axes = np.atleast_2d(axes)

        for idx, (name, family, joint, T_vals, Y_vals, class_names) in enumerate(panels):
            ax = axes[idx // n_cols, idx % n_cols]
            if joint.size == 0:
                ax.set_axis_off()
                continue
            row_sums = joint.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1)
            cond = joint / row_sums

            im = ax.imshow(cond, aspect="auto", cmap="magma",
                           vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(name, fontsize=9, pad=8,
                         color=_FAMILY_COLOR.get(family, "0.2"))
            ax.set_xlabel("$Y$ (transition class)", fontsize=7.5, labelpad=4)
            ax.set_ylabel("$T$ (primitive id)", fontsize=7.5, labelpad=4)

            tick_step_y = max(1, T_vals.size // 8)
            ax.set_yticks(range(0, T_vals.size, tick_step_y))
            ax.set_yticklabels(T_vals[::tick_step_y], fontsize=6)

            ax.set_xticks(range(Y_vals.size))
            yt = [class_names[i] if i < len(class_names) else str(i) for i in Y_vals]
            ax.set_xticklabels(yt, rotation=55, ha="right", fontsize=6)

        for j in range(len(panels), n_rows * n_cols):
            axes[j // n_cols, j % n_cols].set_axis_off()

        fig.subplots_adjust(left=0.07, right=0.90, top=0.93, bottom=0.10,
                            hspace=0.95, wspace=0.55)
        cax = fig.add_axes([0.92, 0.20, 0.012, 0.55])
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_label(r"$p(Y|T)$", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        fig.suptitle(
            "Conditional distribution of transition class given primitive",
            fontsize=10, y=0.985,
        )
        _save_both(fig, output_base)


# ---------------------------------------------------------------------------
# Entry point used by main.py
# ---------------------------------------------------------------------------


def plot_combined_main(
    results: list[MIResult],
    labels: TransitionLabels,
    output_base: Path,
) -> None:
    """Three-panel paper-ready figure: IB plane | I(T;Y) box | AMI box.

    Same data as ``plot_ib_plane`` + ``plot_mi_bar`` restricted to I_TY and AMI.
    """
    if not results:
        return

    family_order = ["DMC", "DMP", "KPMS", "VAE", "Mapper"]
    from collections import defaultdict

    with mpl.rc_context(_rcparams()):
        fig = plt.figure(figsize=(9.0, 3.5))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.0], wspace=0.35)
        ax_ib = fig.add_subplot(gs[0, 0])
        ax_i = fig.add_subplot(gs[0, 1])
        ax_ami = fig.add_subplot(gs[0, 2])

        def _panel_label(ax, text):
            ax.text(
                -0.18, 1.02, text,
                transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="bottom", ha="left",
            )

        _panel_label(ax_ib, "(A)")
        _panel_label(ax_i, "(B)")
        _panel_label(ax_ami, "(C)")

        # ----- IB plane (left) -----
        # Full-clip control stays in the info-loss table but is not shown on the
        # IB plane (kevin).
        ib_results = [r for r in results if r.family != "Full"]
        H_T = np.array([r.H_T for r in ib_results])
        I_TY = np.array([r.I_TY for r in ib_results])
        x_lo = max(0.0, float(np.min(H_T)) - 0.2)
        x_hi = float(np.max(H_T)) + 0.2
        y_lo = max(0.0, float(np.min(I_TY)) - 0.05)
        y_hi = float(np.max(I_TY)) * 1.18 + 0.02

        by_method = defaultdict(list)
        for r in ib_results:
            by_method[r.method_name].append(r)

        legend_handles = {}
        for method_name, rs in by_method.items():
            family = rs[0].family
            color = _FAMILY_COLOR.get(family, "0.4")
            xs = np.array([r.H_T for r in rs])
            ys = np.array([r.I_TY for r in rs])

            ax_ib.scatter(xs, ys, marker="o", s=70,
                           facecolor=color, edgecolor="white",
                           linewidth=1.0, alpha=0.92, zorder=4)

            if family not in legend_handles:
                legend_handles[family] = ax_ib.scatter(
                    [], [], marker="o", s=70,
                    facecolor=color, edgecolor="white",
                    linewidth=1.0, label=family,
                )

        ax_ib.set_xlim(x_lo, x_hi)
        ax_ib.set_ylim(y_lo, y_hi)
        ax_ib.set_xlabel(r"$H(T)$ — representation capacity (bits)")
        ax_ib.set_ylabel(r"$I(T;\,Y)$ — transition info preserved (bits)")
        ax_ib.legend(handles=list(legend_handles.values()),
                     loc="upper left", fontsize=7, handlelength=1.0,
                     borderpad=0.4, labelspacing=0.4)

        # ----- Box helpers (middle + right) -----
        def boxes(ax, metric_attr, ylabel, title):
            box_data: list[np.ndarray] = []
            box_colors: list[str] = []
            for fam in family_order:
                vals = np.array([
                    getattr(r, metric_attr)
                    for r in results
                    if r.family == fam
                ])
                box_data.append(vals if vals.size else np.array([np.nan]))
                box_colors.append(_FAMILY_COLOR.get(fam, "0.4"))

            x_step = 1.4
            positions = np.arange(len(family_order)) * x_step

            bp = ax.boxplot(
                box_data, positions=positions, widths=0.7,
                patch_artist=True, showfliers=True,
                medianprops=dict(color="white", linewidth=1.4),
                whiskerprops=dict(color="0.25", linewidth=0.9),
                capprops=dict(color="0.25", linewidth=0.9),
                flierprops=dict(marker="o", markersize=3,
                                markerfacecolor="0.4",
                                markeredgecolor="none", alpha=0.7),
            )
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_edgecolor("0.25")
                patch.set_linewidth(0.7)
                patch.set_alpha(0.85)

            for i, (vals, color) in enumerate(zip(box_data, box_colors)):
                if vals.size and not np.all(np.isnan(vals)):
                    jitter = np.random.RandomState(0).uniform(
                        -0.15, 0.15, size=vals.size
                    )
                    ax.scatter(np.full_like(vals, positions[i]) + jitter,
                               vals, s=10, color="0.15",
                               alpha=0.6, linewidth=0, zorder=3)

            ax.set_xticks(positions)
            ax.set_xticklabels(family_order, fontsize=8)
            ax.set_xlim(positions[0] - x_step * 0.7,
                        positions[-1] + x_step * 0.7)
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
            ax.set_title(title)
            ax.margins(x=0.02)

        boxes(ax_i, "I_TY", r"$I(T;Y)$ (bits)", r"$I(T;Y)$")
        boxes(ax_ami, "AMI", "AMI", "AMI")

        _save_both(fig, output_base)


def render_all(
    out_dir: Path,
    results: list[MIResult],
    labels: TransitionLabels,
) -> None:
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    plot_ib_plane(results, labels, figs_dir / "ib_plane")
    plot_mi_bar(results, figs_dir / "mi_bar")
    plot_combined_main(results, labels, figs_dir / "main_panel")
    plot_joint_heatmaps(
        results, labels,
        joint_dir=out_dir / "data" / "joint_distributions",
        output_base=figs_dir / "joint_heatmaps",
    )
