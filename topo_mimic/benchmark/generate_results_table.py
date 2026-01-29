"""Generate Nature-style results table from DTW analysis results."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class TableConfig:
    analysis_dir: str = "out/benchmark_analysis"
    output_dir: str = "out/tables"
    table_name: str = "dtw_results_table"
    dpi: int = 300
    font_size: int = 10
    header_font_size: int = 11
    figsize: tuple[float, float] = (10, 4)


METHOD_DISPLAY_NAMES = {
    "w2_dm_kmeans": "W₂+DM",
    "w2_unif_kmeans": "W₂+Unif",
    "kpms_intention": "KPMS-Int",
    "kpms_keypoint": "KPMS-KP",
    "gw_dm_kmeans": "GW+DM",
    "custom_vae": "VAE",
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


def load_per_joint_dtw_results(analysis_dir: str) -> dict[str, pd.DataFrame]:
    """Load per-joint DTW results from analysis directory."""
    results = {}
    analysis_path = Path(analysis_dir)

    for method_dir in analysis_path.iterdir():
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name

        for seed_dir in method_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue

            per_joint_dir = seed_dir / "per_joint_dtw"
            if not per_joint_dir.exists():
                continue

            for csv_file in per_joint_dir.glob("*_per_joint_dtw.csv"):
                df = pd.read_csv(csv_file)
                key = csv_file.stem.replace("_per_joint_dtw", "")
                results[key] = df

    return results


def load_per_joint_summary(analysis_dir: str) -> pd.DataFrame | None:
    """Load the per-joint DTW summary (J x methods) if it exists."""
    summary_path = Path(analysis_dir) / "per_joint_dtw_summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return None


def aggregate_per_joint_results(
    per_joint_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Aggregate per-joint DTW results into J x methods summary.

    For each method, sum the DTW values across all clusters for each joint.
    """
    if not per_joint_results:
        return pd.DataFrame()

    joint_cols = None
    method_sums = {}

    for result_key, df in per_joint_results.items():
        parts = result_key.rsplit("_seed", 1)
        method_name = parts[0]

        dtw_cols = [c for c in df.columns if c.startswith("DTW_")]
        if joint_cols is None:
            joint_cols = [c.replace("DTW_", "") for c in dtw_cols]

        method_key = f"{method_name}_seed{parts[1]}" if len(parts) > 1 else method_name

        sums = {}
        for col in dtw_cols:
            joint_name = col.replace("DTW_", "")
            vals = df[col].dropna()
            vals = vals[np.isfinite(vals)]
            sums[joint_name] = vals.sum() if len(vals) > 0 else 0.0

        method_sums[method_key] = sums

    if not joint_cols or not method_sums:
        return pd.DataFrame()

    rows = []
    for joint in joint_cols:
        row = {"Joint": joint}
        for method, sums in method_sums.items():
            row[method] = sums.get(joint, 0.0)
        rows.append(row)

    return pd.DataFrame(rows)


def create_per_joint_pivot_table(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Create pivot table for per-joint DTW with display names."""
    if agg_df.empty:
        return agg_df

    df = agg_df.set_index("Joint")

    method_order = []
    for m in ["w2_dm_kmeans", "w2_unif_kmeans", "kpms_intention", "kpms_keypoint", "custom_vae", "gw_dm_kmeans"]:
        matching = [c for c in df.columns if c.startswith(m)]
        method_order.extend(matching)
    for c in df.columns:
        if c not in method_order:
            method_order.append(c)

    df = df[method_order]

    new_cols = []
    for c in df.columns:
        base_method = c.rsplit("_seed", 1)[0]
        display = METHOD_DISPLAY_NAMES.get(base_method, base_method)
        if "_seed" in c:
            seed = c.rsplit("_seed", 1)[1]
            display = f"{display} (s{seed})"
        new_cols.append(display)

    df.columns = new_cols
    return df


def aggregate_results(
    individual_results: dict[str, pd.DataFrame],
    expand_nd: bool = True,
) -> pd.DataFrame:
    """
    Aggregate individual DTW results into a single DataFrame.

    Args:
        individual_results: Dict mapping result_key to DataFrame
        expand_nd: If True, copy non-deterministic values to left/right/straight
                   entries to fill in the table
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

    df = pd.DataFrame(all_rows)

    if expand_nd and len(df) > 0:
        df = _expand_non_deterministic(df)

    return df


def _expand_non_deterministic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand non-deterministic entries to fill left/right/straight.

    When a method has Turn="non-deterministic" for a Label, copy that
    DTW value to left, right, and straight entries (if they don't exist).
    This helps fill in the comparison table.
    """
    expanded_rows = []
    directions_to_fill = ["left", "right", "straight"]

    for (method, seed, label), group in df.groupby(["Method", "Seed", "Label"]):
        existing_turns = set(group["Turn"].values)

        for _, row in group.iterrows():
            expanded_rows.append(row.to_dict())

            if row["Turn"] == "non-deterministic":
                for direction in directions_to_fill:
                    if direction not in existing_turns:
                        new_row = row.to_dict()
                        new_row["Turn"] = direction
                        new_row["_from_nd"] = True
                        expanded_rows.append(new_row)

    result_df = pd.DataFrame(expanded_rows)

    if "_from_nd" in result_df.columns:
        result_df = result_df.drop(columns=["_from_nd"])

    return result_df


def create_summary_table(agg_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for method in agg_df["Method"].unique():
        method_df = agg_df[agg_df["Method"] == method]

        mean_dtw = method_df["Mean_DTW"].mean()
        std_dtw = method_df["Mean_DTW"].std()
        n_transitions = len(method_df)

        summary_rows.append({
            "Method": method,
            "Mean DTW ↓": mean_dtw,
            "Std": std_dtw,
            "# Transitions": n_transitions,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("Mean DTW ↓", ascending=True)

    return summary_df


def create_pivot_table(
    agg_df: pd.DataFrame,
    include_turn: bool = True,
    exclude_nd: bool = True,
) -> pd.DataFrame:
    """
    Create pivot table with methods as columns and transitions as rows.

    Args:
        agg_df: Aggregated results DataFrame
        include_turn: If True, show turn direction in row labels
        exclude_nd: If True, exclude "non-deterministic" rows (since expanded)
    """
    df = agg_df.copy()

    if exclude_nd:
        df = df[df["Turn"] != "non-deterministic"]

    if include_turn:
        df["Transition"] = df["Label"] + " [" + df["Turn"] + "]"
    else:
        df["Transition"] = df["Label"]

    pivot = df.pivot_table(
        index="Transition",
        columns="Method",
        values="Mean_DTW",
        aggfunc="mean",
    )

    method_order = []
    for m in ["w2_dm_kmeans", "w2_unif_kmeans", "kpms_intention", "kpms_keypoint", "custom_vae", "gw_dm_kmeans"]:
        if m in pivot.columns:
            method_order.append(m)
    for m in pivot.columns:
        if m not in method_order:
            method_order.append(m)

    pivot = pivot[method_order]
    pivot.columns = [METHOD_DISPLAY_NAMES.get(c, c) for c in pivot.columns]

    return pivot


def create_simple_pivot_table(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simplified pivot table grouped by Label only (no turn direction).

    Averages across all turn directions for each method.
    """
    df = agg_df.copy()

    pivot = df.pivot_table(
        index="Label",
        columns="Method",
        values="Mean_DTW",
        aggfunc="mean",
    )

    method_order = []
    for m in ["w2_dm_kmeans", "w2_unif_kmeans", "kpms_intention", "kpms_keypoint", "custom_vae", "gw_dm_kmeans"]:
        if m in pivot.columns:
            method_order.append(m)
    for m in pivot.columns:
        if m not in method_order:
            method_order.append(m)

    pivot = pivot[method_order]
    pivot.columns = [METHOD_DISPLAY_NAMES.get(c, c) for c in pivot.columns]

    return pivot


def render_nature_table(
    df: pd.DataFrame,
    output_path: str,
    config: TableConfig,
    title: str = "DTW Distance to Ground Truth (↓ lower is better)",
    highlight_best: bool = True,
    show_mean_row: bool = True,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": config.font_size,
    })

    display_df = df.copy()

    if show_mean_row:
        mean_row = display_df.mean(numeric_only=True)
        mean_row.name = "Mean"
        display_df = pd.concat([display_df, mean_row.to_frame().T])

    n_rows = len(display_df) + 1
    n_cols = len(display_df.columns) + 1

    fig_height = max(config.figsize[1], 0.4 * n_rows + 1.5)
    fig_width = max(config.figsize[0], 1.8 * n_cols + 3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    cell_text = []
    cell_colors = []

    numeric_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()

    for idx, (row_name, row) in enumerate(display_df.iterrows()):
        row_text = []
        row_colors = []

        row_numeric_vals = row[numeric_cols].dropna()
        row_min = row_numeric_vals.min() if len(row_numeric_vals) > 0 else None

        for col in display_df.columns:
            val = row[col]
            if pd.isna(val):
                row_text.append("—")
                row_colors.append("white")
            elif col in numeric_cols:
                row_text.append(f"{val:.3f}")

                if highlight_best and row_min is not None and val == row_min:
                    row_colors.append("#c8e6c9")
                else:
                    row_colors.append("white")
            else:
                row_text.append(str(val))
                row_colors.append("white")

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
    table.scale(1.4, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", fontsize=config.header_font_size)
            cell.set_facecolor("#e3f2fd")
            cell.set_width(cell.get_width() * 1.3)
        if col == -1:
            cell.set_text_props(fontweight="bold" if row == len(cell_text) else "normal")
            cell.set_facecolor("#f5f5f5")
            cell.set_width(cell.get_width() * 1.5)

        if row == len(cell_text) and col >= 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#fff3e0")

    ax.set_title(title, fontsize=config.header_font_size + 2, fontweight="bold", pad=20)

    fig.text(
        0.5, 0.02,
        "Green = best (lowest) per row. Lower DTW = closer match to ground truth.",
        ha="center",
        fontsize=config.font_size - 1,
        style="italic",
        color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved table to {output_path}")


def render_per_joint_table(
    df: pd.DataFrame,
    output_path: str,
    config: TableConfig,
    title: str = "Per-Joint DTW Distance (sum across clusters, ↓ lower is better)",
) -> None:
    """Render per-joint DTW table (J x methods)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": config.font_size,
    })

    display_df = df.copy()

    total_row = display_df.select_dtypes(include=[np.number]).sum()
    total_row.name = "Total"
    display_df = pd.concat([display_df, total_row.to_frame().T])

    n_rows = len(display_df) + 1
    n_cols = len(display_df.columns) + 1

    fig_height = max(config.figsize[1], 0.4 * n_rows + 1.5)
    fig_width = max(config.figsize[0], 1.8 * n_cols + 3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    cell_text = []
    cell_colors = []

    numeric_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()

    for idx, (row_name, row) in enumerate(display_df.iterrows()):
        row_text = []
        row_colors = []

        row_numeric_vals = row[numeric_cols].dropna()
        row_min = row_numeric_vals.min() if len(row_numeric_vals) > 0 else None

        for col in display_df.columns:
            val = row[col]
            if pd.isna(val):
                row_text.append("—")
                row_colors.append("white")
            elif col in numeric_cols:
                row_text.append(f"{val:.3f}")
                if row_min is not None and val == row_min:
                    row_colors.append("#c8e6c9")
                else:
                    row_colors.append("white")
            else:
                row_text.append(str(val))
                row_colors.append("white")

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
    table.scale(1.4, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", fontsize=config.header_font_size)
            cell.set_facecolor("#e3f2fd")
            cell.set_width(cell.get_width() * 1.3)
        if col == -1:
            cell.set_text_props(fontweight="bold" if row == len(cell_text) else "normal")
            cell.set_facecolor("#f5f5f5")
            cell.set_width(cell.get_width() * 1.5)

        if row == len(cell_text) and col >= 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#fff3e0")

    ax.set_title(title, fontsize=config.header_font_size + 2, fontweight="bold", pad=20)

    fig.text(
        0.5, 0.02,
        "Green = best (lowest) per row. Values are summed DTW across all clusters.",
        ha="center",
        fontsize=config.font_size - 1,
        style="italic",
        color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved per-joint table to {output_path}")


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
            elif col == "# Transitions":
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
    simple_pivot_df = create_simple_pivot_table(agg_df)
    print(simple_pivot_df.to_string())

    simple_pivot_path = output_dir / f"{config.table_name}_simple.png"
    render_nature_table(
        simple_pivot_df,
        str(simple_pivot_path),
        config,
        title="DTW Distance by Transition Type (↓ lower is better)",
    )

    simple_pivot_pdf_path = output_dir / f"{config.table_name}_simple.pdf"
    render_nature_table(
        simple_pivot_df,
        str(simple_pivot_pdf_path),
        config,
        title="DTW Distance by Transition Type (↓ lower is better)",
    )

    print("\n=== Detailed Pivot Table (with turn direction) ===")
    pivot_df = create_pivot_table(agg_df, include_turn=True, exclude_nd=True)
    print(pivot_df.to_string())

    pivot_path = output_dir / f"{config.table_name}_detailed.png"
    render_nature_table(
        pivot_df,
        str(pivot_path),
        config,
        title="DTW Distance by Transition + Direction (↓ lower is better)",
    )

    pivot_pdf_path = output_dir / f"{config.table_name}_detailed.pdf"
    render_nature_table(
        pivot_df,
        str(pivot_pdf_path),
        config,
        title="DTW Distance by Transition + Direction (↓ lower is better)",
    )

    csv_path = output_dir / f"{config.table_name}_summary.csv"
    summary_df_export = summary_df.copy()
    summary_df_export["Method"] = summary_df_export["Method"].map(
        lambda x: METHOD_DISPLAY_NAMES.get(x, x)
    )
    summary_df_export.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved CSV to {csv_path}")

    print("\n=== Per-Joint DTW Results ===")
    per_joint_results = load_per_joint_dtw_results(config.analysis_dir)

    if per_joint_results:
        print(f"Found {len(per_joint_results)} per-joint result files")

        per_joint_agg = aggregate_per_joint_results(per_joint_results)
        if not per_joint_agg.empty:
            per_joint_pivot = create_per_joint_pivot_table(per_joint_agg)
            print("\nPer-Joint DTW (J x Methods):")
            print(per_joint_pivot.to_string())

            per_joint_png_path = output_dir / f"{config.table_name}_per_joint.png"
            render_per_joint_table(
                per_joint_pivot,
                str(per_joint_png_path),
                config,
                title="Per-Joint DTW Distance (sum across clusters, ↓ lower is better)",
            )

            per_joint_pdf_path = output_dir / f"{config.table_name}_per_joint.pdf"
            render_per_joint_table(
                per_joint_pivot,
                str(per_joint_pdf_path),
                config,
                title="Per-Joint DTW Distance (sum across clusters, ↓ lower is better)",
            )

            per_joint_csv_path = output_dir / f"{config.table_name}_per_joint.csv"
            per_joint_pivot.to_csv(per_joint_csv_path, float_format="%.4f")
            print(f"Saved per-joint CSV to {per_joint_csv_path}")
    else:
        per_joint_summary = load_per_joint_summary(config.analysis_dir)
        if per_joint_summary is not None:
            print("Using pre-computed per-joint summary")
            joint_col = per_joint_summary.columns[0]
            per_joint_pivot = per_joint_summary.set_index(joint_col)

            method_cols = [c for c in per_joint_pivot.columns if c != joint_col]
            new_cols = []
            for c in method_cols:
                base_method = c.rsplit("_seed", 1)[0]
                display = METHOD_DISPLAY_NAMES.get(base_method, base_method)
                if "_seed" in c:
                    seed = c.rsplit("_seed", 1)[1]
                    display = f"{display} (s{seed})"
                new_cols.append(display)
            per_joint_pivot.columns = new_cols

            print("\nPer-Joint DTW (J x Methods):")
            print(per_joint_pivot.to_string())

            per_joint_png_path = output_dir / f"{config.table_name}_per_joint.png"
            render_per_joint_table(
                per_joint_pivot,
                str(per_joint_png_path),
                config,
            )

            per_joint_pdf_path = output_dir / f"{config.table_name}_per_joint.pdf"
            render_per_joint_table(
                per_joint_pivot,
                str(per_joint_pdf_path),
                config,
            )
        else:
            print("No per-joint DTW results found (run analysis with run_per_joint_dtw: true)")

    print(f"\nAll tables saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Nature-style results table from DTW analysis"
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default="out/benchmark_analysis",
        help="Directory containing analysis results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/tables",
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
