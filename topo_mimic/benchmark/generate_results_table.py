"""Generate Nature-style results table from DTW analysis results."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class TableConfig:
    analysis_dir: str = "outputs/benchmark_analysis"
    output_dir: str = "outputs/benchmark_analysis/tables"
    table_name: str = "dtw_results_table"
    dpi: int = 300
    font_size: int = 10
    header_font_size: int = 11
    figsize: tuple[float, float] = (10, 4)


METHOD_DISPLAY_NAMES = {
    "w2_dmc_int_kmeans": "W₂+DMC+Int (ours)",
    "w2_dmc_qpos_kmeans": "W₂+DMC+qpos (ours)",
    "w2_dmc_ekp_kmeans": "W₂+DMC+EKP (ours)",
    "w2_dmp_int_kmeans": "W₂+DMP+Int (ours)",
    "w2_dmp_qpos_kmeans": "W₂+DMP+qpos (ours)",
    "w2_dmp_ekp_kmeans": "W₂+DMP+EKP (ours)",
    "w2_unif_kmeans": "W₂+Unif+Int",
    "w2_full_kmeans": "W₂+Full+Int",
    "gw_dmc_int_kmeans": "GW+DMC",
    "kpms_intention": "KPMS-Int",
    "kpms_keypoint": "KPMS-KP",
    "custom_vae": "VAE",
    "mapper_int": "Mapper-Int",
    "mapper_qpos": "Mapper-qpos",
    "mapper_ekp": "Mapper-EKP",
}


def load_dtw_comparison_table(analysis_dir: str) -> pd.DataFrame:
    csv_path = Path(analysis_dir) / "dtw_comparison_table.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"DTW comparison table not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_individual_dtw_results(analysis_dir: str) -> dict[str, pd.DataFrame]:
    results = {}
    analysis_path = Path(analysis_dir)

    for method_dir in analysis_path.iterdir():
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name

        for seed_dir in method_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue

            dtw_metrics_dir = seed_dir / "dtw_metrics"
            if not dtw_metrics_dir.exists():
                continue

            for csv_file in dtw_metrics_dir.glob("*_dtw_results.csv"):
                df = pd.read_csv(csv_file)
                key = csv_file.stem.replace("_dtw_results", "")
                results[key] = df

    return results


def aggregate_results(
    individual_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Aggregate individual DTW results into a single DataFrame.

    Args:
        individual_results: Dict mapping result_key to DataFrame
    """
    all_rows = []

    for result_key, df in individual_results.items():
        parts = result_key.rsplit("_seed", 1)
        method_name = parts[0]
        seed = int(parts[1]) if len(parts) > 1 else 0

        for _, row in df.iterrows():
            all_rows.append({
                "Method": method_name,
                "Seed": seed,
                "Label": row["Label"],
                "Turn": row.get("Turn_Direction", "none"),
                "Mean_DTW": row["Mean_DTW"],
                "Std_DTW": row["Std_DTW"],
                "N_Discovered": row.get("N_Discovered_Seqs", 0),
                "N_GT": row.get("N_GT_Seqs", 0),
            })

    return pd.DataFrame(all_rows)


def create_summary_table(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Per method: mean ± std across SEEDS of the per-seed mean DTW.

    For each seed, we first average all transitions → one DTW number per seed.
    Error bar is std of those per-seed numbers (n = n_seeds, typically 3).
    """
    summary_rows = []

    for method in agg_df["Method"].unique():
        method_df = agg_df[agg_df["Method"] == method]
        per_seed = method_df.groupby("Seed")["Mean_DTW"].mean()
        mean_dtw = per_seed.mean()
        std_dtw = per_seed.std()
        n_seeds = per_seed.count()
        n_transitions = method_df["Label"].nunique()

        summary_rows.append({
            "Method": method,
            "Mean DTW ↓": mean_dtw,
            "Std": std_dtw,
            "# Seeds": int(n_seeds),
            "# Transitions": int(n_transitions),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("Mean DTW ↓", ascending=True)

    return summary_df


def _preferred_method_order(columns) -> list:
    method_order = []
    for m in [
        "w2_dmc_int_kmeans", "w2_dmc_qpos_kmeans", "w2_dmc_ekp_kmeans",
        "w2_dmp_int_kmeans", "w2_dmp_qpos_kmeans", "w2_dmp_ekp_kmeans",
        "w2_unif_kmeans", "w2_full_kmeans",
        "kpms_intention", "kpms_keypoint", "custom_vae", "gw_dmc_int_kmeans",
        "mapper_int", "mapper_qpos", "mapper_ekp",
    ]:
        if m in columns:
            method_order.append(m)
    for m in columns:
        if m not in method_order:
            method_order.append(m)
    return method_order


def _seed_mean_std_pivot(
    df: pd.DataFrame,
    index_col: str,
    values_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """For each (index_col, Method) cell, return (mean, std, n_seeds) across seeds.

    Within each seed we first average the rows (there may be multiple rows per
    (transition, method, seed) depending on upstream aggregation); then mean
    and std are computed across the per-seed point estimates. std is NaN when
    only one seed contributes; n_seeds pivot reports how many seeds landed in
    each cell so callers can annotate single-seed results.
    """
    per_seed = (
        df.groupby([index_col, "Method", "Seed"])[values_col]
        .mean()
        .reset_index()
    )
    mean_df = per_seed.pivot_table(
        index=index_col, columns="Method", values=values_col, aggfunc="mean",
    )
    std_df = per_seed.pivot_table(
        index=index_col, columns="Method", values=values_col, aggfunc="std",
    )
    n_seeds_df = per_seed.pivot_table(
        index=index_col, columns="Method", values=values_col, aggfunc="count",
    )
    return mean_df, std_df, n_seeds_df


def create_pivot_table(
    agg_df: pd.DataFrame,
    include_turn: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (mean, std, n_seeds) pivots over seeds."""
    df = agg_df.copy()

    if include_turn:
        df["Transition"] = df["Label"] + " [" + df["Turn"] + "]"
    else:
        df["Transition"] = df["Label"]

    mean_df, std_df, n_seeds_df = _seed_mean_std_pivot(df, "Transition", "Mean_DTW")
    method_order = _preferred_method_order(mean_df.columns)
    mean_df = mean_df[method_order]
    std_df = std_df.reindex(columns=method_order)
    n_seeds_df = n_seeds_df.reindex(columns=method_order)

    mean_df.columns = [METHOD_DISPLAY_NAMES.get(c, c) for c in mean_df.columns]
    std_df.columns = [METHOD_DISPLAY_NAMES.get(c, c) for c in std_df.columns]
    n_seeds_df.columns = [METHOD_DISPLAY_NAMES.get(c, c) for c in n_seeds_df.columns]
    return mean_df, std_df, n_seeds_df


def create_simple_pivot_table(
    agg_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simplified pivot grouped by Label only — returns (mean, std, n_seeds) over seeds."""
    df = agg_df.copy()

    mean_df, std_df, n_seeds_df = _seed_mean_std_pivot(df, "Label", "Mean_DTW")
    method_order = _preferred_method_order(mean_df.columns)
    mean_df = mean_df[method_order]
    std_df = std_df.reindex(columns=method_order)
    n_seeds_df = n_seeds_df.reindex(columns=method_order)

    mean_df.columns = [METHOD_DISPLAY_NAMES.get(c, c) for c in mean_df.columns]
    std_df.columns = [METHOD_DISPLAY_NAMES.get(c, c) for c in std_df.columns]
    n_seeds_df.columns = [METHOD_DISPLAY_NAMES.get(c, c) for c in n_seeds_df.columns]
    return mean_df, std_df, n_seeds_df


def render_nature_table(
    df: pd.DataFrame,
    output_path: str,
    config: TableConfig,
    title: str = "DTW Distance to Ground Truth (↓ lower is better)",
    highlight_best: bool = True,
    show_mean_row: bool = True,
    std_df: pd.DataFrame | None = None,
    n_seeds_df: pd.DataFrame | None = None,
    higher_is_better: bool = False,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": config.font_size,
    })

    display_df = df.copy()
    display_std = std_df.copy() if std_df is not None else None
    display_nseeds = n_seeds_df.copy() if n_seeds_df is not None else None

    if show_mean_row:
        mean_row = display_df.mean(numeric_only=True)
        mean_row.name = "Mean"
        display_df = pd.concat([display_df, mean_row.to_frame().T])
        if display_std is not None:
            std_mean_row = pd.Series(
                {c: np.nan for c in display_std.columns}, name="Mean"
            )
            display_std = pd.concat([display_std, std_mean_row.to_frame().T])
        if display_nseeds is not None:
            display_nseeds = pd.concat([
                display_nseeds,
                pd.Series({c: np.nan for c in display_nseeds.columns}, name="Mean").to_frame().T,
            ])

    n_rows = len(display_df) + 1
    n_cols = len(display_df.columns) + 1

    has_std = display_std is not None
    row_pitch = 0.95 if has_std else 0.4
    fig_height = max(config.figsize[1], row_pitch * n_rows + 1.8)
    fig_width = max(config.figsize[0], 2.0 * n_cols + 3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    cell_text = []
    cell_colors = []

    numeric_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()

    nd_row_indices = set()
    for idx, row_name in enumerate(display_df.index):
        if "[non-deterministic]" in str(row_name):
            nd_row_indices.add(idx)

    for idx, (row_name, row) in enumerate(display_df.iterrows()):
        row_text = []
        row_colors = []
        is_nd = idx in nd_row_indices

        row_numeric_vals = row[numeric_cols].dropna()
        if len(row_numeric_vals) == 0:
            row_best = None
        else:
            row_best = row_numeric_vals.max() if higher_is_better else row_numeric_vals.min()
        row_min = row_best  # retained for downstream comparisons (name kept)

        default_bg = "#ffcdd2" if is_nd else "white"

        for col in display_df.columns:
            val = row[col]
            std_val = (
                display_std.loc[row_name, col]
                if (display_std is not None and col in display_std.columns
                    and row_name in display_std.index)
                else np.nan
            )
            n_seeds_val = (
                display_nseeds.loc[row_name, col]
                if (display_nseeds is not None and col in display_nseeds.columns
                    and row_name in display_nseeds.index)
                else np.nan
            )
            if pd.isna(val):
                row_text.append("—")
                row_colors.append(default_bg)
            elif col in numeric_cols:
                if pd.notna(std_val):
                    row_text.append(f"{val:.3f}\n ± {std_val:.3f}")
                elif pd.notna(n_seeds_val) and int(n_seeds_val) == 1:
                    row_text.append(f"{val:.3f}\n(1 seed)")
                else:
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

    row_labels = [str(idx) for idx in display_df.index]
    col_labels = list(display_df.columns)

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(config.font_size)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    table.scale(1.4, 2.6 if has_std else 1.5)

    for (row, col), cell in table.get_celld().items():
        if has_std and col >= 0 and row > 0:
            cell.PAD = 0.08
        if row == 0:
            cell.set_text_props(fontweight="bold", fontsize=config.header_font_size)
            cell.set_facecolor("#e3f2fd")
            cell.set_width(cell.get_width() * 1.3)
        if col == -1:
            cell.set_text_props(fontweight="bold" if row == len(cell_text) else "normal")
            if (row - 1) in nd_row_indices:
                cell.set_facecolor("#ffcdd2")
            else:
                cell.set_facecolor("#f5f5f5")
            cell.set_width(cell.get_width() * 1.5)

        if row == len(cell_text) and col >= 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#fff3e0")

    ax.set_title(title, fontsize=config.header_font_size + 2, fontweight="bold", pad=20)

    fig.text(
        0.5, 0.02,
        "Green = best (lowest) per row. Red = non-deterministic. Lower DTW = closer match to ground truth.",
        ha="center",
        fontsize=config.font_size - 1,
        style="italic",
        color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved table to {output_path}")


def render_summary_table(
    summary_df: pd.DataFrame,
    output_path: str,
    config: TableConfig,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": config.font_size,
    })

    display_df = summary_df.copy()
    display_df["Method"] = display_df["Method"].map(
        lambda x: METHOD_DISPLAY_NAMES.get(x, x)
    )
    display_df = display_df.set_index("Method")

    n_rows = len(display_df) + 1
    n_cols = len(display_df.columns) + 1

    fig_width = max(8, 1.5 * n_cols)
    fig_height = max(3, 0.5 * n_rows + 1)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    cell_text = []
    cell_colors = []

    best_dtw_idx = display_df["Mean DTW ↓"].idxmin()

    for idx, (row_name, row) in enumerate(display_df.iterrows()):
        row_text = []
        row_colors = []

        for col in display_df.columns:
            val = row[col]
            if pd.isna(val):
                row_text.append("—")
                row_colors.append("white")
            elif col == "Mean DTW ↓":
                row_text.append(f"{val:.4f}")
                if row_name == best_dtw_idx:
                    row_colors.append("#c8e6c9")
                else:
                    row_colors.append("white")
            elif col == "Std":
                row_text.append(f"±{val:.4f}")
                row_colors.append("white")
            elif col in ("# Transitions", "# Seeds"):
                row_text.append(f"{int(val)}")
                row_colors.append("white")
            else:
                row_text.append(str(val))
                row_colors.append("white")

        cell_text.append(row_text)
        cell_colors.append(row_colors)

    row_labels = list(display_df.index)
    col_labels = list(display_df.columns)

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(config.font_size)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    table.scale(1.5, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", fontsize=config.header_font_size)
            cell.set_facecolor("#e3f2fd")
            cell.set_width(cell.get_width() * 1.3)
        if col == -1:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f5f5f5")
            cell.set_width(cell.get_width() * 1.5)

    ax.set_title(
        "Behavior Segmentation: DTW Distance to Ground Truth",
        fontsize=config.header_font_size + 2,
        fontweight="bold",
        pad=20,
    )

    fig.text(
        0.5, 0.02,
        "Green = best. Lower DTW indicates closer match to human-annotated transitions.",
        ha="center",
        fontsize=config.font_size - 1,
        style="italic",
        color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved summary table to {output_path}")


def load_individual_mmd_results(analysis_dir: str) -> dict[str, pd.DataFrame]:
    """Glob ``mmd/*_mmd_results.csv`` under each method/seed dir."""
    results = {}
    analysis_path = Path(analysis_dir)

    for method_dir in analysis_path.iterdir():
        if not method_dir.is_dir():
            continue
        for seed_dir in method_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            mmd_subdir = seed_dir / "mmd"
            if not mmd_subdir.exists():
                continue
            for csv_file in mmd_subdir.glob("*_mmd_results.csv"):
                try:
                    df = pd.read_csv(csv_file)
                except pd.errors.EmptyDataError:
                    continue
                key = csv_file.stem.replace("_mmd_results", "")
                results[key] = df
    return results


def aggregate_mmd_results(
    individual_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Flatten per-(label, turn) MMD rows, tag with (Method, Seed)."""
    all_rows = []
    for result_key, df in individual_results.items():
        if "_seed" in result_key:
            method_name, seed_str = result_key.rsplit("_seed", 1)
            try:
                seed = int(seed_str)
            except ValueError:
                seed = 0
        else:
            method_name, seed = result_key, 0
        for _, row in df.iterrows():
            all_rows.append({
                "Method": method_name,
                "Seed": seed,
                "Label": row["Label"],
                "Turn": row.get("Turn_Direction", "none"),
                "KID": row.get("KID", float("nan")),
                "MMD_RBF": row.get("MMD_RBF", float("nan")),
                "N_Cluster_Seqs": row.get("N_Cluster_Seqs", 0),
                "N_GT_Seqs": row.get("N_GT_Seqs", 0),
            })
    return pd.DataFrame(all_rows)


def create_mmd_pivot_table(
    agg_df: pd.DataFrame,
    metric: str = "KID",
    include_turn: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rows = transitions with direction, cols = methods, values = mean of
    the named ``metric`` column across seeds. Returns (mean, std, n_seeds).
    """
    df = agg_df.copy()
    if include_turn:
        df["Transition"] = df["Label"] + " [" + df["Turn"] + "]"
    else:
        df["Transition"] = df["Label"]

    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])
    if df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    mean_df, std_df, n_seeds_df = _seed_mean_std_pivot(df, "Transition", metric)
    method_order = _preferred_method_order(mean_df.columns)
    mean_df = mean_df[method_order]
    std_df = std_df.reindex(columns=method_order)
    n_seeds_df = n_seeds_df.reindex(columns=method_order)

    mean_df.columns = [METHOD_DISPLAY_NAMES.get(c, c) for c in mean_df.columns]
    std_df.columns = [METHOD_DISPLAY_NAMES.get(c, c) for c in std_df.columns]
    n_seeds_df.columns = [METHOD_DISPLAY_NAMES.get(c, c) for c in n_seeds_df.columns]
    return mean_df, std_df, n_seeds_df


def load_individual_purity_results(analysis_dir: str) -> dict[str, pd.DataFrame]:
    """Glob ``purity/*_purity.csv`` under each method/seed dir."""
    results = {}
    analysis_path = Path(analysis_dir)

    for method_dir in analysis_path.iterdir():
        if not method_dir.is_dir():
            continue
        for seed_dir in method_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            purity_dir = seed_dir / "purity"
            if not purity_dir.exists():
                continue
            for csv_file in purity_dir.glob("*_purity.csv"):
                try:
                    df = pd.read_csv(csv_file)
                except pd.errors.EmptyDataError:
                    continue
                key = csv_file.stem.replace("_purity", "")
                results[key] = df
    return results


def aggregate_purity_results(
    individual_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Flatten per-cluster purity rows, tagging each with (Method, Seed)."""
    all_rows = []
    for result_key, df in individual_results.items():
        # result_key looks like "{method_name}_seed{seed}"
        if "_seed" in result_key:
            method_name, seed_str = result_key.rsplit("_seed", 1)
            try:
                seed = int(seed_str)
            except ValueError:
                seed = 0
        else:
            method_name, seed = result_key, 0
        for _, row in df.iterrows():
            all_rows.append({
                "Method": method_name,
                "Seed": seed,
                "Cluster_ID": row.get("cluster_id", -1),
                "Label": row.get("Label", ""),
                "From_To": row.get("From_To", ""),
                "Turn": row.get("Turn_Direction", "none"),
                "N_Samples": row.get("N_Samples", 0),
                "Mean_Purity": row.get("Mean_Purity", float("nan")),
            })
    return pd.DataFrame(all_rows)


def create_purity_pivot_table(
    agg_df: pd.DataFrame,
    include_turn: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rows = "From_To [turn]" (or "From_To"), cols = methods, values = mean
    purity across seeds. Within each (transition, method, seed) we first take
    the mean purity across all clusters that received the label; then mean
    and std are taken across seeds.

    Rows where ``From_To`` is empty (unparseable label) are dropped.
    """
    df = agg_df[agg_df["From_To"] != ""].copy()
    if df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    if include_turn:
        df["Transition"] = df["From_To"] + " [" + df["Turn"] + "]"
    else:
        df["Transition"] = df["From_To"]

    mean_df, std_df, n_seeds_df = _seed_mean_std_pivot(df, "Transition", "Mean_Purity")
    method_order = _preferred_method_order(mean_df.columns)
    mean_df = mean_df[method_order]
    std_df = std_df.reindex(columns=method_order)
    n_seeds_df = n_seeds_df.reindex(columns=method_order)

    mean_df.columns = [METHOD_DISPLAY_NAMES.get(c, c) for c in mean_df.columns]
    std_df.columns = [METHOD_DISPLAY_NAMES.get(c, c) for c in std_df.columns]
    n_seeds_df.columns = [METHOD_DISPLAY_NAMES.get(c, c) for c in n_seeds_df.columns]
    return mean_df, std_df, n_seeds_df


def load_individual_consistency_results(analysis_dir: str) -> dict[str, pd.DataFrame]:
    results = {}
    analysis_path = Path(analysis_dir)

    for method_dir in analysis_path.iterdir():
        if not method_dir.is_dir():
            continue

        for seed_dir in method_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue

            consistency_dir = seed_dir / "consistency"
            if not consistency_dir.exists():
                continue

            for csv_file in consistency_dir.glob("*_consistency.csv"):
                df = pd.read_csv(csv_file)
                key = csv_file.stem.replace("_consistency", "")
                results[key] = df

    return results


def aggregate_consistency_results(
    individual_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    all_rows = []

    for result_key, df in individual_results.items():
        parts = result_key.rsplit("_seed", 1)
        method_name = parts[0]
        seed = int(parts[1]) if len(parts) > 1 else 0

        for _, row in df.iterrows():
            all_rows.append({
                "Method": method_name,
                "Seed": seed,
                "Label": row["Label"],
                "Turn": row.get("Turn_Direction", "none"),
                "Mean_Intra_DTW": row["Mean_Intra_DTW"],
                "Avg_Frames": row.get("Avg_Frames", 0),
            })

    return pd.DataFrame(all_rows)


def create_consistency_pivot_table(
    agg_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rows = "Label [Turn]", cols = methods, values = Mean_Intra_DTW.

    Returns (mean, std, n_seeds) pivots across seeds and appends an
    "Avg Frames" row (std/n_seeds NaN there — it is a summary row, not a
    seed-varying metric).
    """
    df = agg_df.copy()
    df["Transition"] = df["Label"] + " [" + df["Turn"] + "]"

    mean_df, std_df, n_seeds_df = _seed_mean_std_pivot(df, "Transition", "Mean_Intra_DTW")
    method_order = _preferred_method_order(mean_df.columns)
    mean_df = mean_df[method_order]
    std_df = std_df.reindex(columns=method_order)
    n_seeds_df = n_seeds_df.reindex(columns=method_order)

    avg_frames_row = {}
    for method in mean_df.columns:
        method_df = df[df["Method"] == method]
        if len(method_df) > 0:
            avg_frames_row[method] = method_df["Avg_Frames"].mean()
        else:
            avg_frames_row[method] = np.nan
    avg_frames_series = pd.Series(avg_frames_row, name="Avg Frames")
    mean_df = pd.concat([mean_df, avg_frames_series.to_frame().T])
    std_df = pd.concat([
        std_df,
        pd.Series({c: np.nan for c in std_df.columns}, name="Avg Frames").to_frame().T,
    ])
    n_seeds_df = pd.concat([
        n_seeds_df,
        pd.Series({c: np.nan for c in n_seeds_df.columns}, name="Avg Frames").to_frame().T,
    ])

    mean_df.columns = [METHOD_DISPLAY_NAMES.get(c, c) for c in mean_df.columns]
    std_df.columns = [METHOD_DISPLAY_NAMES.get(c, c) for c in std_df.columns]
    n_seeds_df.columns = [METHOD_DISPLAY_NAMES.get(c, c) for c in n_seeds_df.columns]
    return mean_df, std_df, n_seeds_df


def _enumerate_all_gt_transitions(
    behavior_labels_csv: str | None = None,
) -> pd.Series:
    """Return a Series of GT instance counts indexed by "A -> B [turn]" keys
    covering EVERY directional transition × turn direction that appears at
    least once in the ground-truth CSV. Used to expand the DTW table to show
    transitions no method discovered (cells become '—').
    """
    if behavior_labels_csv is None:
        behavior_labels_csv = "behavior_labels/rodent_labels.csv"
    if not os.path.exists(behavior_labels_csv):
        return pd.Series(dtype=int)
    from topo_mimic.benchmark.analysis.ground_truth import find_all_transitions
    from topo_mimic.benchmark.analysis.behavior_parsing import TurnDirection

    tx = find_all_transitions(behavior_labels_csv, merge_rear=True)
    from collections import Counter
    c: Counter = Counter()
    for t in tx:
        turn_str = str(t.turn) if t.turn != TurnDirection.STRAIGHT else "straight"
        key = f"{t.from_behavior} -> {t.to_behavior} [{turn_str}]"
        c[key] += 1
    return pd.Series(c, dtype=int).sort_index()


def create_count_pivot_table(
    agg_df: pd.DataFrame,
    include_turn: bool = True,
    full_gt_index: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cluster-count table as percentage of GT — returns (mean, std) over seeds.

    Each cell shows mean ± std of N_Discovered/N_GT across seeds as a
    percentage. A separate "GT (n)" column shows the raw GT count and has no
    std (NaN).
    """
    df = agg_df.copy()

    if include_turn:
        df["Transition"] = df["Label"] + " [" + df["Turn"] + "]"
    else:
        df["Transition"] = df["Label"]

    gt_counts = df.groupby("Transition")["N_GT"].first()
    if full_gt_index is not None and len(full_gt_index):
        all_idx = pd.Index(sorted(set(gt_counts.index) | set(full_gt_index.index)))
        gt_counts = gt_counts.reindex(all_idx).fillna(full_gt_index.reindex(all_idx))

    per_seed = (
        df.groupby(["Transition", "Method", "Seed"])["N_Discovered"]
        .sum()
        .reset_index()
    )
    mean_n = per_seed.pivot_table(
        index="Transition", columns="Method", values="N_Discovered", aggfunc="mean",
    )
    std_n = per_seed.pivot_table(
        index="Transition", columns="Method", values="N_Discovered", aggfunc="std",
    )
    n_seeds_df = per_seed.pivot_table(
        index="Transition", columns="Method", values="N_Discovered", aggfunc="count",
    )

    method_order = _preferred_method_order(mean_n.columns)
    mean_n = mean_n[method_order]
    std_n = std_n.reindex(columns=method_order)
    n_seeds_df = n_seeds_df.reindex(columns=method_order)

    if full_gt_index is not None and len(full_gt_index):
        mean_n = mean_n.reindex(gt_counts.index)
        std_n = std_n.reindex(gt_counts.index)
        n_seeds_df = n_seeds_df.reindex(gt_counts.index)

    mean_pct = mean_n.copy().astype(float)
    std_pct = std_n.copy().astype(float)
    for transition in mean_pct.index:
        n_gt = gt_counts.get(transition, np.nan)
        if pd.notna(n_gt) and n_gt > 0:
            mean_pct.loc[transition] = (mean_n.loc[transition] / n_gt * 100).round(1)
            std_pct.loc[transition] = (std_n.loc[transition] / n_gt * 100).round(1)
        else:
            mean_pct.loc[transition] = np.nan
            std_pct.loc[transition] = np.nan

    mean_pct.insert(0, "GT (n)", gt_counts.reindex(mean_pct.index).astype(float).fillna(0).astype(int))
    std_pct.insert(0, "GT (n)", np.nan)
    n_seeds_df.insert(0, "GT (n)", np.nan)

    display_cols = ["GT (n)"] + [
        METHOD_DISPLAY_NAMES.get(c, c) for c in mean_pct.columns[1:]
    ]
    mean_pct.columns = display_cols
    std_pct.columns = display_cols
    n_seeds_df.columns = display_cols
    return mean_pct, std_pct, n_seeds_df


def render_plain_table(
    df: pd.DataFrame,
    output_path: str,
    config: TableConfig,
    title: str = "",
    show_mean_row: bool = False,
    std_df: pd.DataFrame | None = None,
    n_seeds_df: pd.DataFrame | None = None,
) -> None:
    """Render a table without any color highlighting. If ``std_df`` is given,
    percentage cells are rendered as ``mean ± std`` (percent)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": config.font_size,
    })

    display_df = df.copy()
    display_std = std_df.copy() if std_df is not None else None
    display_nseeds = n_seeds_df.copy() if n_seeds_df is not None else None
    if show_mean_row:
        numeric_cols = display_df.select_dtypes(include=[np.number]).columns
        mean_row = display_df[numeric_cols].mean()
        mean_row.name = "Mean"
        display_df = pd.concat([display_df, mean_row.to_frame().T])
        if display_std is not None:
            display_std = pd.concat([
                display_std,
                pd.Series({c: np.nan for c in display_std.columns}, name="Mean").to_frame().T,
            ])
        if display_nseeds is not None:
            display_nseeds = pd.concat([
                display_nseeds,
                pd.Series({c: np.nan for c in display_nseeds.columns}, name="Mean").to_frame().T,
            ])

    n_rows = len(display_df) + 1
    n_cols = len(display_df.columns) + 1

    has_std = display_std is not None
    row_pitch = 0.95 if has_std else 0.4
    fig_height = max(config.figsize[1], row_pitch * n_rows + 1.8)
    fig_width = max(config.figsize[0], 2.0 * n_cols + 3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    cell_text = []
    for row_name, row in display_df.iterrows():
        row_text = []
        for col in display_df.columns:
            val = row[col]
            std_val = (
                display_std.loc[row_name, col]
                if (display_std is not None and col in display_std.columns
                    and row_name in display_std.index)
                else np.nan
            )
            n_seeds_val = (
                display_nseeds.loc[row_name, col]
                if (display_nseeds is not None and col in display_nseeds.columns
                    and row_name in display_nseeds.index)
                else np.nan
            )
            if pd.isna(val):
                row_text.append("—")
            elif isinstance(val, float):
                if val == int(val) and col == "GT (n)":
                    row_text.append(f"{int(val)}")
                elif pd.notna(std_val):
                    row_text.append(f"{val:.1f}%\n ± {std_val:.1f}%")
                elif pd.notna(n_seeds_val) and int(n_seeds_val) == 1:
                    row_text.append(f"{val:.1f}%\n(1 seed)")
                else:
                    row_text.append(f"{val:.1f}%")
            else:
                row_text.append(str(val))
        cell_text.append(row_text)

    cell_colors = [["white"] * len(display_df.columns) for _ in range(len(display_df))]

    row_labels = [str(idx) for idx in display_df.index]
    col_labels = list(display_df.columns)

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(config.font_size)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    table.scale(1.4, 2.6 if has_std else 1.5)

    for (row, col), cell in table.get_celld().items():
        if has_std and col >= 0 and row > 0:
            cell.PAD = 0.08
        if row == 0:
            cell.set_text_props(fontweight="bold", fontsize=config.header_font_size)
            cell.set_facecolor("#e3f2fd")
            cell.set_width(cell.get_width() * 1.3)
        if col == -1:
            cell.set_facecolor("#f5f5f5")
            cell.set_width(cell.get_width() * 1.5)

    if title:
        ax.set_title(title, fontsize=config.header_font_size + 2, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved table to {output_path}")


def generate_all_tables(config: TableConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {config.analysis_dir}...")

    individual_results = load_individual_dtw_results(config.analysis_dir)
    if not individual_results:
        print("No DTW results found!")
        return

    print(f"Found {len(individual_results)} result files")

    agg_df = aggregate_results(individual_results)
    print(f"Aggregated {len(agg_df)} rows")

    print("\n=== Summary Table ===")
    summary_df = create_summary_table(agg_df)
    print(summary_df.to_string(index=False))

    summary_path = output_dir / f"{config.table_name}_summary.png"
    render_summary_table(summary_df, str(summary_path), config)

    summary_pdf_path = output_dir / f"{config.table_name}_summary.pdf"
    render_summary_table(summary_df, str(summary_pdf_path), config)

    print("\n=== Simple Pivot Table (by Label only) ===")
    simple_mean_df, simple_std_df, simple_nseeds_df = create_simple_pivot_table(agg_df)
    print(simple_mean_df.to_string())

    simple_pivot_path = output_dir / f"{config.table_name}_simple.png"
    render_nature_table(
        simple_mean_df,
        str(simple_pivot_path),
        config,
        title="DTW Distance by Transition Type (↓ lower is better)",
        std_df=simple_std_df,
        n_seeds_df=simple_nseeds_df,
    )

    simple_pivot_pdf_path = output_dir / f"{config.table_name}_simple.pdf"
    render_nature_table(
        simple_mean_df,
        str(simple_pivot_pdf_path),
        config,
        title="DTW Distance by Transition Type (↓ lower is better)",
        std_df=simple_std_df,
        n_seeds_df=simple_nseeds_df,
    )

    simple_mean_df.to_csv(output_dir / f"{config.table_name}_simple.csv", float_format="%.4f")
    simple_std_df.to_csv(output_dir / f"{config.table_name}_simple_std.csv", float_format="%.4f")

    print("\n=== Detailed Pivot Table (with turn direction) ===")
    pivot_mean_df, pivot_std_df, pivot_nseeds_df = create_pivot_table(agg_df, include_turn=True)
    print(pivot_mean_df.to_string())

    pivot_path = output_dir / f"{config.table_name}_detailed.png"
    render_nature_table(
        pivot_mean_df,
        str(pivot_path),
        config,
        title="DTW Distance by Transition + Direction (↓ lower is better)",
        std_df=pivot_std_df,
        n_seeds_df=pivot_nseeds_df,
    )

    pivot_pdf_path = output_dir / f"{config.table_name}_detailed.pdf"
    render_nature_table(
        pivot_mean_df,
        str(pivot_pdf_path),
        config,
        title="DTW Distance by Transition + Direction (↓ lower is better)",
        std_df=pivot_std_df,
        n_seeds_df=pivot_nseeds_df,
    )

    pivot_mean_df.to_csv(output_dir / f"{config.table_name}_detailed.csv", float_format="%.4f")
    pivot_std_df.to_csv(output_dir / f"{config.table_name}_detailed_std.csv", float_format="%.4f")

    print("\n=== Cluster Count Table (% of GT) ===")
    full_gt_index = _enumerate_all_gt_transitions()
    count_mean_df, count_std_df, count_nseeds_df = create_count_pivot_table(
        agg_df, include_turn=True, full_gt_index=full_gt_index,
    )
    print(count_mean_df.to_string())

    count_png_path = output_dir / f"{config.table_name}_counts.png"
    render_plain_table(
        count_mean_df,
        str(count_png_path),
        config,
        title="Discovered Examples per Transition (% of GT count)",
        std_df=count_std_df,
        n_seeds_df=count_nseeds_df,
    )

    count_pdf_path = output_dir / f"{config.table_name}_counts.pdf"
    render_plain_table(
        count_mean_df,
        str(count_pdf_path),
        config,
        title="Discovered Examples per Transition (% of GT count)",
        std_df=count_std_df,
        n_seeds_df=count_nseeds_df,
    )

    count_csv_path = output_dir / f"{config.table_name}_counts.csv"
    count_mean_df.to_csv(count_csv_path, float_format="%.1f")
    count_std_csv_path = output_dir / f"{config.table_name}_counts_std.csv"
    count_std_df.to_csv(count_std_csv_path, float_format="%.1f")
    print(f"Saved count CSV to {count_csv_path} (std to {count_std_csv_path})")

    csv_path = output_dir / f"{config.table_name}_summary.csv"
    summary_df_export = summary_df.copy()
    summary_df_export["Method"] = summary_df_export["Method"].map(
        lambda x: METHOD_DISPLAY_NAMES.get(x, x)
    )
    summary_df_export.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved CSV to {csv_path}")

    consistency_results = load_individual_consistency_results(config.analysis_dir)
    if consistency_results:
        print(f"\n=== Internal Consistency Table ===")
        print(f"Found {len(consistency_results)} consistency result files")

        consistency_agg_df = aggregate_consistency_results(consistency_results)
        print(f"Aggregated {len(consistency_agg_df)} consistency rows")

        consistency_mean_df, consistency_std_df, consistency_nseeds_df = (
            create_consistency_pivot_table(consistency_agg_df)
        )
        print(consistency_mean_df.to_string())

        consistency_png_path = output_dir / "consistency_table.png"
        render_nature_table(
            consistency_mean_df,
            str(consistency_png_path),
            config,
            title="Internal Cluster Consistency (↓ lower = more consistent)",
            show_mean_row=False,
            std_df=consistency_std_df,
            n_seeds_df=consistency_nseeds_df,
        )

        consistency_pdf_path = output_dir / "consistency_table.pdf"
        render_nature_table(
            consistency_mean_df,
            str(consistency_pdf_path),
            config,
            title="Internal Cluster Consistency (↓ lower = more consistent)",
            show_mean_row=False,
            std_df=consistency_std_df,
            n_seeds_df=consistency_nseeds_df,
        )

        consistency_csv_path = output_dir / "consistency_table.csv"
        consistency_mean_df.to_csv(consistency_csv_path, float_format="%.4f")
        consistency_std_csv_path = output_dir / "consistency_table_std.csv"
        consistency_std_df.to_csv(consistency_std_csv_path, float_format="%.4f")
        print(f"Saved consistency CSV to {consistency_csv_path} (std to {consistency_std_csv_path})")
    else:
        print("\nNo consistency results found, skipping consistency table.")

    mmd_results = load_individual_mmd_results(config.analysis_dir)
    if mmd_results:
        print(f"\n=== MMD / KID Tables ===")
        print(f"Found {len(mmd_results)} MMD result files")
        mmd_agg_df = aggregate_mmd_results(mmd_results)
        for metric, pretty, title in (
            ("KID", "kid", "KID (polynomial-kernel MMD², ↓ lower = closer to real)"),
            ("MMD_RBF", "mmd_rbf", "MMD² (RBF kernel, ↓ lower = closer to real)"),
        ):
            m_mean_df, m_std_df, m_nseeds_df = create_mmd_pivot_table(
                mmd_agg_df, metric=metric, include_turn=True,
            )
            if m_mean_df.empty:
                print(f"  {metric}: no rows; skipping.")
                continue
            print(f"\n  {metric}:")
            print(m_mean_df.to_string())

            png = output_dir / f"{pretty}_table.png"
            pdf = output_dir / f"{pretty}_table.pdf"
            render_nature_table(
                m_mean_df, str(png), config,
                title=title, show_mean_row=False, highlight_best=True,
                higher_is_better=False,
                std_df=m_std_df, n_seeds_df=m_nseeds_df,
            )
            render_nature_table(
                m_mean_df, str(pdf), config,
                title=title, show_mean_row=False, highlight_best=True,
                higher_is_better=False,
                std_df=m_std_df, n_seeds_df=m_nseeds_df,
            )
            m_mean_df.to_csv(output_dir / f"{pretty}_table.csv", float_format="%.6f")
            m_std_df.to_csv(output_dir / f"{pretty}_table_std.csv", float_format="%.6f")
            print(f"  saved {pretty}_table.{{csv,png,pdf}} (std to {pretty}_table_std.csv)")
    else:
        print("\nNo MMD results found, skipping MMD/KID tables.")

    purity_results = load_individual_purity_results(config.analysis_dir)
    if purity_results:
        print(f"\n=== Cluster Purity Table ===")
        print(f"Found {len(purity_results)} purity result files")
        purity_agg_df = aggregate_purity_results(purity_results)
        purity_mean_df, purity_std_df, purity_nseeds_df = create_purity_pivot_table(
            purity_agg_df, include_turn=True,
        )
        if not purity_mean_df.empty:
            print(purity_mean_df.to_string())

            purity_png_path = output_dir / "purity_table.png"
            render_nature_table(
                purity_mean_df,
                str(purity_png_path),
                config,
                title="Cluster Purity by Transition + Direction (↑ higher is better)",
                show_mean_row=False,
                highlight_best=True,
                higher_is_better=True,
                std_df=purity_std_df,
                n_seeds_df=purity_nseeds_df,
            )

            purity_pdf_path = output_dir / "purity_table.pdf"
            render_nature_table(
                purity_mean_df,
                str(purity_pdf_path),
                config,
                title="Cluster Purity by Transition + Direction (↑ higher is better)",
                show_mean_row=False,
                highlight_best=True,
                higher_is_better=True,
                std_df=purity_std_df,
                n_seeds_df=purity_nseeds_df,
            )

            purity_csv_path = output_dir / "purity_table.csv"
            purity_mean_df.to_csv(purity_csv_path, float_format="%.4f")
            purity_std_csv_path = output_dir / "purity_table_std.csv"
            purity_std_df.to_csv(purity_std_csv_path, float_format="%.4f")
            print(f"Saved purity CSV to {purity_csv_path} (std to {purity_std_csv_path})")
        else:
            print("Purity results had no parseable rows; skipping table.")
    else:
        print("\nNo purity results found, skipping purity table.")

    print(f"\nAll tables saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Nature-style results table from DTW analysis"
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default="outputs/benchmark_analysis",
        help="Directory containing analysis results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/benchmark_analysis/tables",
        help="Directory to save output tables",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default="dtw_results_table",
        help="Base name for output files",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output images",
    )

    args = parser.parse_args()

    config = TableConfig(
        analysis_dir=args.analysis_dir,
        output_dir=args.output_dir,
        table_name=args.table_name,
        dpi=args.dpi,
    )

    generate_all_tables(config)


if __name__ == "__main__":
    main()
