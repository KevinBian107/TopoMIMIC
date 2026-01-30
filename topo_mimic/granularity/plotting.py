"""Nature-style plotting functions for granularity analysis."""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .config import GranularityConfig
from .stationary_analysis import (
    StationaryAnalysisOutput,
    JointVarianceResult,
    CycleSurvivalResult,
    CycleLifetimeResult,
)
from .redundancy import RedundancyAnalysisOutput, RedundancyResult


def setup_nature_style() -> None:
    """Configure matplotlib for Nature-style plots."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 10,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.frameon': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def _get_joint_base_name(joint_name: str) -> str:
    """Extract base name from joint (e.g., 'elbow_L' -> 'elbow')."""
    if joint_name.endswith("_L") or joint_name.endswith("_R"):
        return joint_name[:-2]
    return joint_name


def _group_joints_by_base(joint_names: list[str]) -> dict[str, list[str]]:
    """Group joints by their base name (L/R pairs together)."""
    groups = {}
    for name in joint_names:
        base = _get_joint_base_name(name)
        if base not in groups:
            groups[base] = []
        groups[base].append(name)
    for base in groups:
        groups[base].sort()
    return groups


def plot_joint_variance_boxplots(
    stationary_results: StationaryAnalysisOutput,
    config: GranularityConfig,
    output_dir: Path,
) -> dict[str, Path]:
    """
    Plot joint variance boxplots across persistence values.
    Each joint group (L/R pairs) gets its own PDF.

    Args:
        stationary_results: Results from stationary analysis
        config: Configuration object
        output_dir: Directory to save plots

    Returns:
        Dict mapping plot name to file path
    """
    setup_nature_style()
    output_dir = Path(output_dir)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    paths = {}

    all_results = {}
    for pers_delta, results in stationary_results.head_joint_results.items():
        if pers_delta not in all_results:
            all_results[pers_delta] = []
        all_results[pers_delta].extend(results)
    for pers_delta, results in stationary_results.arm_joint_results.items():
        if pers_delta not in all_results:
            all_results[pers_delta] = []
        all_results[pers_delta].extend(results)

    if not all_results:
        return paths

    pers_values = sorted(all_results.keys())
    joint_names = [r.joint_name for r in all_results[pers_values[0]]]
    joint_groups = _group_joints_by_base(joint_names)

    for base_name, group_joints in joint_groups.items():
        group_results = {}
        for pers_delta in pers_values:
            group_results[pers_delta] = [
                r for r in all_results[pers_delta] if r.joint_name in group_joints
            ]

        save_path = output_dir / "plots" / f"{base_name}_variance_boxplots.pdf"
        paths[base_name] = _plot_single_joint_group(
            group_results,
            base_name,
            group_joints,
            config,
            save_path,
        )

    return paths


def _plot_single_joint_group(
    joint_results: dict[float, list[JointVarianceResult]],
    base_name: str,
    group_joints: list[str],
    config: GranularityConfig,
    save_path: Path,
) -> Path:
    """
    Plot boxplots for a single joint group (e.g., elbow L/R) as its own PDF.

    Args:
        joint_results: Dict mapping pers_delta to list of JointVarianceResult
        base_name: Base joint name (e.g., 'elbow')
        group_joints: List of joint names in this group (e.g., ['elbow_L', 'elbow_R'])
        config: Configuration object
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    pers_values = sorted(joint_results.keys())
    if not pers_values:
        return save_path

    n_sides = len(group_joints)

    fig, ax = plt.subplots(figsize=config.plotting.figsize)

    side_colors = {"_L": "#1f77b4", "_R": "#ff7f0e", "": "#2ca02c"}
    box_width = 0.35 if n_sides == 2 else 0.6
    offsets = np.linspace(-0.2, 0.2, n_sides) if n_sides > 1 else [0]

    legend_handles = []

    for side_idx, joint_name in enumerate(group_joints):
        suffix = joint_name[len(base_name):] if joint_name != base_name else ""
        color = side_colors.get(suffix, side_colors[""])

        data_for_boxplot = []
        positions = []

        for i, pers in enumerate(pers_values):
            results_at_pers = joint_results[pers]
            joint_result = next((r for r in results_at_pers if r.joint_name == joint_name), None)

            if joint_result and joint_result.variances:
                data_for_boxplot.append(joint_result.variances)
                positions.append(i + offsets[side_idx])

        if data_for_boxplot:
            bp = ax.boxplot(
                data_for_boxplot,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
            )

            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            for element in ['whiskers', 'caps', 'medians']:
                for item in bp[element]:
                    item.set_color('black')
                    item.set_linewidth(0.8)

            label = "Left" if suffix == "_L" else ("Right" if suffix == "_R" else joint_name)
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7, label=label))

    ax.set_xticks(range(len(pers_values)))
    ax.set_xticklabels([f"{p:.2f}" for p in pers_values], rotation=45, ha='right')
    ax.set_xlabel("Persistence δ")
    ax.set_ylabel("Variance")
    ax.set_title(f"{base_name.replace('_', ' ').title()} Variance in DM Cycles", fontweight='bold')

    if len(legend_handles) > 1:
        ax.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=config.plotting.dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved {save_path}")
    return save_path


def plot_cycle_survival_comparison(
    survival_results: list[CycleSurvivalResult],
    config: GranularityConfig,
    output_dir: Path,
) -> dict[str, Path]:
    """
    Plot survival comparison: lost vs survived cycles at each persistence transition.
    Each joint gets its own PDF.

    Args:
        survival_results: List of CycleSurvivalResult
        config: Configuration object
        output_dir: Directory to save plots

    Returns:
        Dict mapping joint name to file path
    """
    setup_nature_style()
    output_dir = Path(output_dir)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    if not survival_results:
        return {}

    joint_names = list(set(r.joint_name for r in survival_results))
    joint_groups = _group_joints_by_base(joint_names)
    transitions = sorted(set(r.transition for r in survival_results))

    paths = {}

    for base_name, group_joints in joint_groups.items():
        save_path = output_dir / "plots" / f"{base_name}_survival_comparison.pdf"

        n_transitions = len(transitions)
        fig, ax = plt.subplots(figsize=(config.plotting.figsize[0] * 1.5, config.plotting.figsize[1]))

        lost_color = "#e74c3c"
        survived_color = "#27ae60"
        box_width = 0.35
        n_sides = len(group_joints)

        for t_idx, transition in enumerate(transitions):
            for side_idx, joint_name in enumerate(group_joints):
                result = next(
                    (r for r in survival_results
                     if r.transition == transition and r.joint_name == joint_name),
                    None
                )

                if result is None:
                    continue

                base_pos = t_idx * 2.5
                side_offset = (side_idx - (n_sides - 1) / 2) * 0.8

                if result.lost_variances:
                    bp_lost = ax.boxplot(
                        [result.lost_variances],
                        positions=[base_pos + side_offset - 0.2],
                        widths=box_width,
                        patch_artist=True,
                        showfliers=False,
                    )
                    for patch in bp_lost['boxes']:
                        patch.set_facecolor(lost_color)
                        patch.set_alpha(0.7)
                    for element in ['whiskers', 'caps', 'medians']:
                        for item in bp_lost[element]:
                            item.set_color('black')
                            item.set_linewidth(0.8)

                if result.survived_variances:
                    bp_surv = ax.boxplot(
                        [result.survived_variances],
                        positions=[base_pos + side_offset + 0.2],
                        widths=box_width,
                        patch_artist=True,
                        showfliers=False,
                    )
                    for patch in bp_surv['boxes']:
                        patch.set_facecolor(survived_color)
                        patch.set_alpha(0.7)
                    for element in ['whiskers', 'caps', 'medians']:
                        for item in bp_surv[element]:
                            item.set_color('black')
                            item.set_linewidth(0.8)

        ax.set_xticks([t_idx * 2.5 for t_idx in range(n_transitions)])
        ax.set_xticklabels([f"δ {t[0]:.2f}→{t[1]:.2f}" for t in transitions], rotation=45, ha='right')
        ax.set_xlabel("Persistence Transition")
        ax.set_ylabel("Variance")
        ax.set_title(f"{base_name.replace('_', ' ').title()}: Lost vs Survived Cycles", fontweight='bold')

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, fc=lost_color, alpha=0.7, label="Lost"),
            plt.Rectangle((0, 0), 1, 1, fc=survived_color, alpha=0.7, label="Survived"),
        ]
        ax.legend(handles=legend_handles, loc='upper right', frameon=False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=config.plotting.dpi, bbox_inches='tight')
        plt.close()

        print(f"  Saved {save_path}")
        paths[base_name] = save_path

    return paths


def plot_cycle_lifetime(
    lifetime_results: list[CycleLifetimeResult],
    config: GranularityConfig,
    output_dir: Path,
) -> dict[str, Path]:
    """
    Plot cycle lifetime: variance vs max persistence level where cycle survived.
    Each joint gets its own PDF.

    Args:
        lifetime_results: List of CycleLifetimeResult
        config: Configuration object
        output_dir: Directory to save plots

    Returns:
        Dict mapping joint name to file path
    """
    setup_nature_style()
    output_dir = Path(output_dir)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    if not lifetime_results:
        return {}

    joint_names = [r.joint_name for r in lifetime_results]
    joint_groups = _group_joints_by_base(joint_names)

    paths = {}

    for base_name, group_joints in joint_groups.items():
        save_path = output_dir / "plots" / f"{base_name}_cycle_lifetime.pdf"

        fig, ax = plt.subplots(figsize=config.plotting.figsize)

        side_colors = {"_L": "#1f77b4", "_R": "#ff7f0e", "": "#2ca02c"}

        all_pers_values = set()
        for joint_name in group_joints:
            result = next((r for r in lifetime_results if r.joint_name == joint_name), None)
            if result:
                all_pers_values.update(result.max_persistence)

        pers_values = sorted(all_pers_values)
        n_pers = len(pers_values)
        n_sides = len(group_joints)
        box_width = 0.35 if n_sides == 2 else 0.6

        legend_handles = []

        for side_idx, joint_name in enumerate(group_joints):
            result = next((r for r in lifetime_results if r.joint_name == joint_name), None)
            if result is None or not result.max_persistence:
                continue

            suffix = joint_name[len(base_name):] if joint_name != base_name else ""
            color = side_colors.get(suffix, side_colors[""])

            data_by_pers = {p: [] for p in pers_values}
            for mp, var in zip(result.max_persistence, result.variances):
                data_by_pers[mp].append(var)

            data_for_boxplot = []
            positions = []

            for p_idx, pers in enumerate(pers_values):
                if data_by_pers[pers]:
                    data_for_boxplot.append(data_by_pers[pers])
                    offset = (side_idx - (n_sides - 1) / 2) * 0.25
                    positions.append(p_idx + offset)

            if data_for_boxplot:
                bp = ax.boxplot(
                    data_for_boxplot,
                    positions=positions,
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                )

                for patch in bp['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                for element in ['whiskers', 'caps', 'medians']:
                    for item in bp[element]:
                        item.set_color('black')
                        item.set_linewidth(0.8)

                label = "Left" if suffix == "_L" else ("Right" if suffix == "_R" else joint_name)
                legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7, label=label))

        ax.set_xticks(range(n_pers))
        ax.set_xticklabels([f"{p:.2f}" for p in pers_values], rotation=45, ha='right')
        ax.set_xlabel("Max Persistence Survived")
        ax.set_ylabel("Variance")
        ax.set_title(f"{base_name.replace('_', ' ').title()}: Variance vs Cycle Lifetime", fontweight='bold')

        if len(legend_handles) > 1:
            ax.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=config.plotting.dpi, bbox_inches='tight')
        plt.close()

        print(f"  Saved {save_path}")
        paths[base_name] = save_path

    return paths


def plot_filtered_variance_by_stage(
    survival_results: list[CycleSurvivalResult],
    config: GranularityConfig,
    output_dir: Path,
) -> dict[str, Path]:
    """
    Plot joint variance of cycles that got filtered out at each persistence stage.
    Each joint gets its own PDF.

    Args:
        survival_results: List of CycleSurvivalResult
        config: Configuration object
        output_dir: Directory to save plots

    Returns:
        Dict mapping joint name to file path
    """
    setup_nature_style()
    output_dir = Path(output_dir)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    if not survival_results:
        return {}

    joint_names = list(set(r.joint_name for r in survival_results))
    joint_groups = _group_joints_by_base(joint_names)
    transitions = sorted(set(r.transition for r in survival_results))

    paths = {}

    for base_name, group_joints in joint_groups.items():
        save_path = output_dir / "plots" / f"{base_name}_filtered_variance_by_stage.pdf"

        n_transitions = len(transitions)
        n_sides = len(group_joints)

        fig, ax = plt.subplots(figsize=(config.plotting.figsize[0] * 1.2, config.plotting.figsize[1]))

        side_colors = {"_L": "#1f77b4", "_R": "#ff7f0e", "": "#e74c3c"}
        box_width = 0.35 if n_sides == 2 else 0.5

        legend_handles = []

        for side_idx, joint_name in enumerate(group_joints):
            suffix = joint_name[len(base_name):] if joint_name != base_name else ""
            color = side_colors.get(suffix, side_colors[""])

            data_for_boxplot = []
            positions = []

            for t_idx, transition in enumerate(transitions):
                result = next(
                    (r for r in survival_results
                     if r.transition == transition and r.joint_name == joint_name),
                    None
                )

                if result and result.lost_variances:
                    data_for_boxplot.append(result.lost_variances)
                    offset = (side_idx - (n_sides - 1) / 2) * 0.4
                    positions.append(t_idx + offset)

            if data_for_boxplot:
                bp = ax.boxplot(
                    data_for_boxplot,
                    positions=positions,
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                )

                for patch in bp['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                for element in ['whiskers', 'caps', 'medians']:
                    for item in bp[element]:
                        item.set_color('black')
                        item.set_linewidth(0.8)

                label = "Left" if suffix == "_L" else ("Right" if suffix == "_R" else base_name)
                legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7, label=label))

        ax.set_xticks(range(n_transitions))
        ax.set_xticklabels([f"δ {t[0]:.2f}→{t[1]:.2f}" for t in transitions], rotation=45, ha='right')
        ax.set_xlabel("Persistence Transition (Filtered At)")
        ax.set_ylabel("Variance")
        ax.set_title(f"{base_name.replace('_', ' ').title()}: Variance of Filtered Cycles", fontweight='bold')

        if len(legend_handles) > 1:
            ax.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=config.plotting.dpi, bbox_inches='tight')
        plt.close()

        print(f"  Saved {save_path}")
        paths[base_name] = save_path

    return paths


def plot_redundancy_heatmap(
    dtw_matrix: np.ndarray,
    cluster_ids: list[int],
    threshold: float | None,
    title: str,
    save_path: Path,
    config: GranularityConfig,
) -> Path:
    """
    Plot DTW distance heatmap between clusters.

    Args:
        dtw_matrix: (n, n) DTW distance matrix
        cluster_ids: List of cluster IDs
        threshold: Redundancy threshold (for annotation)
        title: Plot title
        save_path: Path to save figure
        config: Configuration object

    Returns:
        Path to saved figure
    """
    setup_nature_style()

    n = len(cluster_ids)
    fig, ax = plt.subplots(figsize=(max(4, n * 0.4), max(4, n * 0.4)))

    mask = ~np.isfinite(dtw_matrix)
    dtw_display = np.where(mask, np.nan, dtw_matrix)

    im = ax.imshow(dtw_display, cmap='YlOrRd', aspect='equal')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"C{c}" for c in cluster_ids], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([f"C{c}" for c in cluster_ids], fontsize=8)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("DTW Distance", fontsize=10)

    if threshold is not None:
        for i in range(n):
            for j in range(n):
                if i != j and np.isfinite(dtw_matrix[i, j]):
                    if dtw_matrix[i, j] < threshold:
                        ax.add_patch(plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, edgecolor='blue', linewidth=2
                        ))

    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("Cluster", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=config.plotting.dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved {save_path}")
    return save_path


def plot_unique_clusters_curve(
    redundancy_output: RedundancyAnalysisOutput,
    config: GranularityConfig,
    output_dir: Path,
    method: str = "fixed",
) -> Path:
    """
    Plot unique clusters vs n_clusters for each persistence value.

    Args:
        redundancy_output: Results from redundancy analysis
        config: Configuration object
        output_dir: Directory to save plot
        method: "fixed" or "percentile" threshold method

    Returns:
        Path to saved figure
    """
    setup_nature_style()
    output_dir = Path(output_dir)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    save_path = output_dir / "plots" / f"unique_clusters_curve_{method}.pdf"

    df = redundancy_output.summary_df
    if df.empty:
        return save_path

    unique_col = f"n_unique_{method}"
    if unique_col not in df.columns:
        unique_col = "n_unique_fixed"

    pers_values = sorted(df["pers_delta"].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(pers_values)))

    fig, ax = plt.subplots(figsize=config.plotting.figsize)

    for pers, color in zip(pers_values, colors):
        subset = df[df["pers_delta"] == pers].sort_values("n_clusters")
        ax.plot(
            subset["n_clusters"],
            subset[unique_col],
            marker='o',
            markersize=6,
            linewidth=1.5,
            color=color,
            label=f"δ={pers:.2f}",
        )

    ax.plot(
        config.redundancy.n_clusters_range,
        config.redundancy.n_clusters_range,
        '--',
        color='gray',
        alpha=0.5,
        label='No redundancy',
    )

    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Unique Clusters After Merging")
    ax.set_title(f"Cluster Redundancy Analysis ({method.title()} Threshold)", fontweight='bold')
    ax.legend(loc='lower right', frameon=False)

    ax.set_xlim(min(config.redundancy.n_clusters_range) - 1, max(config.redundancy.n_clusters_range) + 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=config.plotting.dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved {save_path}")
    return save_path


def plot_top_similar_pairs_bar(
    result: RedundancyResult,
    config: GranularityConfig,
    output_dir: Path,
    prefix: str = "",
) -> Path:
    """
    Plot bar chart of top similar cluster pairs.

    Args:
        result: RedundancyResult for a specific configuration
        config: Configuration object
        output_dir: Directory to save plot
        prefix: Prefix for filename

    Returns:
        Path to saved figure
    """
    setup_nature_style()
    output_dir = Path(output_dir)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    if prefix:
        save_path = output_dir / "plots" / f"{prefix}_top_similar_pairs.pdf"
    else:
        save_path = output_dir / "plots" / f"top_similar_pairs_pers{result.pers_delta:.2f}_k{result.n_clusters}.pdf"

    pairs = result.top_similar_pairs
    if not pairs:
        return save_path

    fig, ax = plt.subplots(figsize=(config.plotting.figsize[0] * 1.2, config.plotting.figsize[1]))

    labels = [f"C{p[0]}-C{p[1]}" for p in pairs]
    distances = [p[2] for p in pairs]

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.6, len(pairs)))

    y_pos = np.arange(len(pairs))
    bars = ax.barh(y_pos, distances, color=colors, edgecolor='white', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("DTW Distance")
    ax.set_title(f"Most Similar Cluster Pairs\n(δ={result.pers_delta:.2f}, k={result.n_clusters})",
                 fontweight='bold')

    for bar, dist in zip(bars, distances):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{dist:.3f}", va='center', fontsize=8)

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.plotting.dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved {save_path}")
    return save_path


def plot_dtw_matrix_heatmap(
    result: RedundancyResult,
    config: GranularityConfig,
    output_dir: Path,
) -> Path:
    """
    Plot DTW distance matrix as a heatmap.

    Args:
        result: RedundancyResult for a specific configuration
        config: Configuration object
        output_dir: Directory to save plot

    Returns:
        Path to saved figure
    """
    setup_nature_style()
    output_dir = Path(output_dir)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    save_path = output_dir / "plots" / f"dtw_heatmap_pers{result.pers_delta:.2f}_k{result.n_clusters}.pdf"

    dtw_matrix = result.dtw_matrix
    cluster_ids = result.cluster_ids
    n = len(cluster_ids)

    fig, ax = plt.subplots(figsize=(max(5, n * 0.5), max(4, n * 0.45)))

    mask = ~np.isfinite(dtw_matrix)
    dtw_display = np.where(mask, np.nan, dtw_matrix)

    im = ax.imshow(dtw_display, cmap='YlOrRd', aspect='equal')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"C{c}" for c in cluster_ids], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([f"C{c}" for c in cluster_ids], fontsize=8)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("DTW Distance", fontsize=10)

    if n <= 15:
        for i in range(n):
            for j in range(n):
                if i != j and np.isfinite(dtw_matrix[i, j]):
                    val = dtw_matrix[i, j]
                    color = 'white' if val > np.nanmedian(dtw_matrix) else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)

    threshold = np.percentile(
        dtw_matrix[np.isfinite(dtw_matrix) & (dtw_matrix > 0)],
        config.redundancy.dtw_percentile
    ) if np.any(np.isfinite(dtw_matrix) & (dtw_matrix > 0)) else None

    if threshold is not None:
        for i in range(n):
            for j in range(n):
                if i != j and np.isfinite(dtw_matrix[i, j]) and dtw_matrix[i, j] < threshold:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor='blue', linewidth=2
                    ))

    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("Cluster", fontsize=11)
    ax.set_title(f"DTW Distance Matrix (δ={result.pers_delta:.2f}, k={result.n_clusters})",
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=config.plotting.dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved {save_path}")
    return save_path


def plot_dtw_summary_across_k(
    redundancy_output: RedundancyAnalysisOutput,
    config: GranularityConfig,
    output_dir: Path,
) -> Path:
    """
    Plot summary of DTW statistics across different k values.

    Args:
        redundancy_output: Results from redundancy analysis
        config: Configuration object
        output_dir: Directory to save plot

    Returns:
        Path to saved figure
    """
    setup_nature_style()
    output_dir = Path(output_dir)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    save_path = output_dir / "plots" / "dtw_summary_across_k.pdf"

    rows = []
    for (pers_delta, n_clusters), result in redundancy_output.results.items():
        dtw = result.dtw_matrix
        upper_tri = dtw[np.triu_indices_from(dtw, k=1)]
        valid = upper_tri[np.isfinite(upper_tri)]

        if len(valid) > 0:
            rows.append({
                "pers_delta": pers_delta,
                "n_clusters": n_clusters,
                "mean_dtw": np.mean(valid),
                "min_dtw": np.min(valid),
                "max_dtw": np.max(valid),
                "std_dtw": np.std(valid),
                "median_dtw": np.median(valid),
            })

    if not rows:
        return save_path

    df = pd.DataFrame(rows)
    pers_values = sorted(df["pers_delta"].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(pers_values)))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for pers, color in zip(pers_values, colors):
        subset = df[df["pers_delta"] == pers].sort_values("n_clusters")
        axes[0].plot(subset["n_clusters"], subset["mean_dtw"], 'o-', color=color, label=f"δ={pers:.2f}")
        axes[0].fill_between(
            subset["n_clusters"],
            subset["mean_dtw"] - subset["std_dtw"],
            subset["mean_dtw"] + subset["std_dtw"],
            alpha=0.2, color=color
        )

    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Mean DTW Distance")
    axes[0].set_title("Mean DTW ± Std", fontweight='bold')
    axes[0].legend(frameon=False, fontsize=8)

    for pers, color in zip(pers_values, colors):
        subset = df[df["pers_delta"] == pers].sort_values("n_clusters")
        axes[1].plot(subset["n_clusters"], subset["min_dtw"], 'o-', color=color, label=f"δ={pers:.2f}")

    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Min DTW Distance")
    axes[1].set_title("Minimum DTW (Most Similar Pair)", fontweight='bold')

    for pers, color in zip(pers_values, colors):
        subset = df[df["pers_delta"] == pers].sort_values("n_clusters")
        axes[2].plot(subset["n_clusters"], subset["median_dtw"], 'o-', color=color, label=f"δ={pers:.2f}")

    axes[2].set_xlabel("Number of Clusters (k)")
    axes[2].set_ylabel("Median DTW Distance")
    axes[2].set_title("Median DTW", fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=config.plotting.dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved {save_path}")
    return save_path


def generate_stationary_plots(
    stationary_results: StationaryAnalysisOutput,
    config: GranularityConfig,
) -> dict[str, Path]:
    """
    Generate plots for stationary analysis results.

    Args:
        stationary_results: Results from stationary analysis
        config: Configuration object

    Returns:
        Dict mapping plot name to file path
    """
    print("  Generating stationary analysis plots...")
    paths = {}
    output_dir = Path(config.output_dir)

    joint_paths = plot_joint_variance_boxplots(
        stationary_results, config, output_dir / "stationary"
    )
    paths.update({f"joint_variance_{k}": v for k, v in joint_paths.items()})

    if stationary_results.survival_results:
        survival_paths = plot_cycle_survival_comparison(
            stationary_results.survival_results, config, output_dir / "stationary"
        )
        paths.update({f"survival_{k}": v for k, v in survival_paths.items()})

        filtered_paths = plot_filtered_variance_by_stage(
            stationary_results.survival_results, config, output_dir / "stationary"
        )
        paths.update({f"filtered_{k}": v for k, v in filtered_paths.items()})

    if stationary_results.lifetime_results:
        lifetime_paths = plot_cycle_lifetime(
            stationary_results.lifetime_results, config, output_dir / "stationary"
        )
        paths.update({f"lifetime_{k}": v for k, v in lifetime_paths.items()})

    return paths


def generate_redundancy_plots(
    redundancy_output: RedundancyAnalysisOutput,
    config: GranularityConfig,
) -> dict[str, Path]:
    """
    Generate plots for redundancy analysis results.

    Args:
        redundancy_output: Results from redundancy analysis
        config: Configuration object

    Returns:
        Dict mapping plot name to file path
    """
    print("  Generating redundancy analysis plots...")
    paths = {}
    output_dir = Path(config.output_dir)

    paths["unique_clusters_fixed"] = plot_unique_clusters_curve(
        redundancy_output, config, output_dir / "redundancy", method="fixed"
    )
    paths["unique_clusters_percentile"] = plot_unique_clusters_curve(
        redundancy_output, config, output_dir / "redundancy", method="percentile"
    )

    paths["dtw_summary"] = plot_dtw_summary_across_k(
        redundancy_output, config, output_dir / "redundancy"
    )

    for (pers_delta, n_clusters), result in redundancy_output.results.items():
        if result.top_similar_pairs:
            key = f"similar_pairs_pers{pers_delta:.2f}_k{n_clusters}"
            paths[key] = plot_top_similar_pairs_bar(
                result, config, output_dir / "redundancy"
            )

        key = f"dtw_heatmap_pers{pers_delta:.2f}_k{n_clusters}"
        paths[key] = plot_dtw_matrix_heatmap(
            result, config, output_dir / "redundancy" / f"pers_{pers_delta:.2f}" / f"n_clusters_{n_clusters}"
        )

    return paths


def generate_all_plots(
    stationary_results: StationaryAnalysisOutput | None,
    redundancy_output: RedundancyAnalysisOutput | None,
    config: GranularityConfig,
) -> dict[str, Path]:
    """
    Generate all granularity analysis plots (legacy function for compatibility).

    Args:
        stationary_results: Results from stationary analysis (or None)
        redundancy_output: Results from redundancy analysis (or None)
        config: Configuration object

    Returns:
        Dict mapping plot name to file path
    """
    paths = {}

    if stationary_results is not None:
        paths.update(generate_stationary_plots(stationary_results, config))

    if redundancy_output is not None:
        paths.update(generate_redundancy_plots(redundancy_output, config))

    return paths
