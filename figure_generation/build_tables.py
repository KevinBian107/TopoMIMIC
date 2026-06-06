#!/usr/bin/env python3
"""Generate every .tex table under paper/tables/ from the master CSVs.

Usage:
    python paper/build_tables.py              # writes all tables
    python paper/build_tables.py --check      # prints what would change
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FA   = REPO / "outputs/analysis/feature_analysis"
WA   = REPO / "outputs/analysis/tables/warping_analysis"
SS   = REPO / "outputs/analysis/scale_sweep/v1/v1/cfg_01f5a4b3/discovery"
OUT  = REPO / "paper/tables"

# Drop the uniform-Wasserstein control + all classifier columns.
DROP_METHODS = {"w2_unif_kmeans", "classification"}

# The 4 macro-behaviors and the 6 unordered pairs.
BEHAVIORS  = ["Walk", "Immobile", "Turn", "Rear"]
ORDERED_PAIRS = [
    ("Immobile", "Rear"),
    ("Immobile", "Turn"),
    ("Immobile", "Walk"),
    ("Rear", "Turn"),
    ("Rear", "Walk"),
    ("Turn", "Walk"),
]
DIRECTIONS = ["left", "right", "straight"]

METHOD_DISPLAY = {
    "w2_dmc_int_kmeans"  : "DMC-Int$^*$",
    "w2_dmc_qpos_kmeans" : "DMC-Qpos$^*$",
    "w2_dmc_ekp_kmeans"  : "DMC-EKP$^*$",
    "w2_dmp_int_kmeans"  : "DMP-Int$^*$",
    "w2_dmp_qpos_kmeans" : "DMP-Qpos$^*$",
    "w2_dmp_ekp_kmeans"  : "DMP-EKP$^*$",
    "mapper_int"         : "Mapper-Int",
    "mapper_qpos"        : "Mapper-Qpos",
    "mapper_ekp"         : "Mapper-EKP",
    "kpms_intention"     : "KPMS-Int",
    "kpms_keypoint"      : "KPMS-KP",
    "custom_vae"         : "VAE",
    "w2_full_kmeans"     : "Full-Int",
}
METHOD_ORDER_FULL = list(METHOD_DISPLAY.keys())

# Family groupings used to pick best-of-family per row in the main tables.
# DM cycles (DMC) and paths (DMP) are reported as separate columns; the VAE
# baseline is dropped from the main tables.
FAMILIES = {
    "DMC$^*$" : ["w2_dmc_int_kmeans", "w2_dmc_qpos_kmeans", "w2_dmc_ekp_kmeans"],
    "DMP$^*$" : ["w2_dmp_int_kmeans", "w2_dmp_qpos_kmeans", "w2_dmp_ekp_kmeans"],
    "Mapper"  : ["mapper_int", "mapper_qpos", "mapper_ekp"],
    "KPMS"    : ["kpms_intention", "kpms_keypoint"],
    "Full"    : ["w2_full_kmeans"],
}
FAMILY_ORDER = ["DMC$^*$", "DMP$^*$", "Mapper", "KPMS", "Full"]

# Aliases used by the wide MMD/KID/warping CSVs (already pre-aggregated).
ALIAS_WIDE = {
    # MMD / KID style
    "W₂+DMC+Int (ours)"  : "w2_dmc_int_kmeans",
    "W₂+DMC+qpos (ours)" : "w2_dmc_qpos_kmeans",
    "W₂+DMC+EKP (ours)"  : "w2_dmc_ekp_kmeans",
    "W₂+DMP+Int (ours)"  : "w2_dmp_int_kmeans",
    "W₂+DMP+qpos (ours)" : "w2_dmp_qpos_kmeans",
    "W₂+DMP+EKP (ours)"  : "w2_dmp_ekp_kmeans",
    "W₂+Unif+Int"        : "w2_unif_kmeans",
    "W₂+Full+Int"        : "w2_full_kmeans",
    "KPMS-Int"           : "kpms_intention",
    "KPMS-KP"            : "kpms_keypoint",
    "VAE"                : "custom_vae",
    "Mapper-Int"         : "mapper_int",
    "Mapper-qpos"        : "mapper_qpos",
    "Mapper-EKP"         : "mapper_ekp",
    # Warping CSV short-name style
    "DMC-Int"            : "w2_dmc_int_kmeans",
    "DMC-Qpos"           : "w2_dmc_qpos_kmeans",
    "DMC-EKP"            : "w2_dmc_ekp_kmeans",
    "DMP-Int"            : "w2_dmp_int_kmeans",
    "DMP-Qpos"           : "w2_dmp_qpos_kmeans",
    "DMP-EKP"            : "w2_dmp_ekp_kmeans",
    "Mapper-EKP-pca_1d-n5"          : "mapper_ekp",
    "Mapper-INT-pca_1d-n25"         : "mapper_int",
    "Mapper-Qpos-eccentricity-n15"  : "mapper_qpos",
    "Mapper-Qpos-knn_density-n15"   : "mapper_qpos_knn",  # secondary mapper-qpos variant; will be dropped if ambiguous
}

DIRECTION_TAG = {"left": "[L]", "right": "[R]", "straight": "[S]", "non-det.": "[N]"}

# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _seed_aggregate(df: pd.DataFrame, value_label: str) -> pd.DataFrame:
    """Mean-ignore-NaN over seed columns; return long form (Label, Turn, method, value)."""
    df = df.replace("N/A", np.nan)
    method_cols = [c for c in df.columns if c not in ("Label", "Turn")]
    out = pd.DataFrame({"Label": df["Label"], "Turn": df["Turn"]})
    seen_methods: dict[str, list[str]] = {}
    for c in method_cols:
        for suffix in ("_seed0", "_seed1", "_seed2"):
            if c.endswith(suffix):
                base = c[: -len(suffix)]
                seen_methods.setdefault(base, []).append(c)
                break
        else:
            seen_methods.setdefault(c, []).append(c)
    for method, cols in seen_methods.items():
        if method in DROP_METHODS:
            continue
        vals = df[cols].astype(float)
        out[method] = vals.mean(axis=1, skipna=True)
    out = out.melt(id_vars=["Label", "Turn"], var_name="method", value_name=value_label)
    out = out.dropna(subset=[value_label])
    return out


def _seed_aggregate_with_std(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Like _seed_aggregate, but returns (mean_long, std_long).

    std is NaN when fewer than 2 seeds populate the cell.
    """
    df = df.replace("N/A", np.nan)
    method_cols = [c for c in df.columns if c not in ("Label", "Turn")]
    seen: dict[str, list[str]] = {}
    for c in method_cols:
        matched = False
        for suffix in ("_seed0", "_seed1", "_seed2"):
            if c.endswith(suffix):
                seen.setdefault(c[: -len(suffix)], []).append(c)
                matched = True
                break
        if not matched:
            seen.setdefault(c, []).append(c)
    mean_df = pd.DataFrame({"Label": df["Label"], "Turn": df["Turn"]})
    std_df  = pd.DataFrame({"Label": df["Label"], "Turn": df["Turn"]})
    for method, cols in seen.items():
        if method in DROP_METHODS:
            continue
        vals = df[cols].astype(float)
        mean_df[method] = vals.mean(axis=1, skipna=True)
        n_pop = vals.notna().sum(axis=1)
        s = vals.std(axis=1, skipna=True, ddof=1)
        s = s.where(n_pop >= 2)  # std undefined for <2 populated seeds
        std_df[method] = s
    mean_long = mean_df.melt(id_vars=["Label", "Turn"], var_name="method", value_name="value").dropna(subset=["value"])
    std_long  = std_df.melt(id_vars=["Label", "Turn"], var_name="method", value_name="std")
    return mean_long, std_long


def load_dtw_long() -> pd.DataFrame:
    return _seed_aggregate(pd.read_csv(FA / "dtw_comparison_table.csv"), "value")


def load_consistency_long() -> pd.DataFrame:
    return _seed_aggregate(pd.read_csv(FA / "consistency_comparison_table.csv"), "value")


def load_dtw_meanstd() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _seed_aggregate_with_std(pd.read_csv(FA / "dtw_comparison_table.csv"))


def load_consistency_meanstd() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _seed_aggregate_with_std(pd.read_csv(FA / "consistency_comparison_table.csv"))


def load_sync_ratio_meanstd() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _seed_aggregate_with_std(pd.read_csv(FA / "sync_ratio_comparison_table.csv"))


def load_non_diag_meanstd() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _seed_aggregate_with_std(pd.read_csv(FA / "non_diag_comparison_table.csv"))


def load_wide_meanstd(mean_csv: Path, std_csv: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean+std wide CSV pair (e.g. mmd_rbf_table + mmd_rbf_table_std)."""
    mean_df = load_wide(mean_csv)
    if std_csv is None or not std_csv.exists():
        std_df = mean_df.copy()
        for c in [c for c in std_df.columns if c not in ("Label", "Turn")]:
            std_df[c] = np.nan
    else:
        std_df = load_wide(std_csv)
    return mean_df, std_df


def load_wide(csv: Path) -> pd.DataFrame:
    """Pre-aggregated wide CSV (MMD, KID, warping)."""
    df = pd.read_csv(csv)
    if "Transition" in df.columns:
        # Split "Imm -> Rear [left]" into Label + Turn.
        labels, turns = [], []
        for s in df["Transition"]:
            s = s.strip()
            if s.endswith("]"):
                base, turn = s.rsplit(" [", 1)
                turns.append(turn.rstrip("]"))
                labels.append(base)
            else:
                labels.append(s); turns.append("")
        df = df.drop(columns=["Transition"])
        df.insert(0, "Label", labels); df.insert(1, "Turn", turns)
    rename = {k: v for k, v in ALIAS_WIDE.items() if k in df.columns}
    df = df.rename(columns=rename)
    drop = [c for c in df.columns if c in DROP_METHODS or c == "mapper_qpos_knn"]
    df = df.drop(columns=drop)
    # When the same canonical key exists twice (e.g. two mapper-qpos sweep variants),
    # collapse by row-wise mean.
    dup_cols = df.columns[df.columns.duplicated()].unique()
    for c in dup_cols:
        merged = df.loc[:, df.columns == c].astype(float).mean(axis=1, skipna=True)
        df = df.drop(columns=[c]).assign(**{c: merged})
    # Normalize the Turn column to lowercase and merge duplicate (Label, Turn) rows
    # — the warping CSVs sometimes spell directions as "Left" for Mapper and
    # "left" for DM/KPMS, so case-folding lets a single appendix row populate
    # cells from both source streams.
    if "Turn" in df.columns:
        df["Turn"] = df["Turn"].astype(str).str.lower()
        method_cols = [c for c in df.columns if c not in ("Label", "Turn")]
        df = df.groupby(["Label", "Turn"], as_index=False)[method_cols].mean(numeric_only=True)
    return df


def long_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    return long_df.pivot_table(
        index=["Label", "Turn"], columns="method", values="value", aggfunc="mean"
    ).reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Direction collapse + transition canonicalization
# ─────────────────────────────────────────────────────────────────────────────

def _canonical_pair(label: str) -> str:
    """'Walk -> Rear' -> 'Rear<->Walk' (alphabetic-pair regardless of order)."""
    if "->" not in label:
        return label
    a, b = [s.strip() for s in label.split("->")]
    a, b = sorted([a, b])
    return f"{a}<->{b}"


def collapse_direction(wide: pd.DataFrame, value_aggregate: str = "mean") -> pd.DataFrame:
    """Collapse turn directions and order direction into 6 canonical pairs."""
    wide = wide.copy()
    wide["Pair"] = wide["Label"].map(_canonical_pair)
    method_cols = [c for c in wide.columns if c not in ("Label", "Turn", "Pair")]
    grp = wide.groupby("Pair")[method_cols].mean(numeric_only=True).reset_index()
    return grp


# ─────────────────────────────────────────────────────────────────────────────
# Row formatting (color coding)
# ─────────────────────────────────────────────────────────────────────────────

GREEN = r"\textcolor{bestgreen}"
YELL  = r"\textcolor{secondyellow}"
RED   = r"\textcolor{worstred}"

def _fmt_cell(v: float, rank: str | None, fmt: str = "{:.3f}", std: float | None = None,
              std_fmt: str = "{:.2f}", single_seed: bool = False) -> str:
    """Format one cell. If single_seed is True, a small dagger marks the value."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "---"
    val_s = fmt.format(v)
    if single_seed:
        val_s = val_s + r"$^{\dagger}$"
    std_s = ""
    if std is not None and not (isinstance(std, float) and np.isnan(std)):
        std_s = std_fmt.format(std)
    body = val_s
    if std_s:
        body = val_s + r"{\scriptsize$\,\pm\,$" + std_s + "}"
    if rank == "best":   return r"\textbf{" + body + "}"
    if rank == "second": return r"\underline{" + body + "}"
    return body


def _row_ranks(values: list[float | None], lower_better: bool) -> list[str | None]:
    pairs = [(i, v) for i, v in enumerate(values) if v is not None and not (isinstance(v, float) and np.isnan(v))]
    ranks: list[str | None] = [None] * len(values)
    if not pairs:
        return ranks
    pairs.sort(key=lambda p: p[1], reverse=not lower_better)
    # Even a single populated cell counts as "best" — there's nothing to lose to.
    ranks[pairs[0][0]] = "best"
    if len(pairs) >= 2: ranks[pairs[1][0]] = "second"
    if len(pairs) >= 3: ranks[pairs[-1][0]] = "worst"
    return ranks


def _emit_color_table(
    rows: list[tuple[str, list[float]]],
    methods: list[str],
    lower_better: bool,
    caption: str,
    label: str,
    *,
    extra_block_columns: list[tuple[list[str], list[list[float]]]] | None = None,
    block_titles: list[str] | None = None,
    fmt: str = "{:.3f}",
    tabular_size: str = r"\tiny",
    col_sep: str = "1.5pt",
    row_stretch: str = "0.95",
    row_label_col: str = "Transition",
    table_env: bool = True,
    caption_first: bool = True,
    include_mean: bool = False,
) -> str:
    """Generic per-row color-coded numeric table."""
    blocks = [(methods, [vals for _, vals in rows])]
    if extra_block_columns:
        blocks.extend(extra_block_columns)
    block_titles = block_titles or [None] * len(blocks)

    n_cols = sum(len(b[0]) for b in blocks)
    col_spec = "l" + "|" + "|".join("c" * len(b[0]) for b in blocks)

    out = []
    # Always emit color definitions so \input'd snippets render correctly.
    out.append(r"\definecolor{bestgreen}{HTML}{2E7D32}")
    out.append(r"\definecolor{secondyellow}{HTML}{F9A825}")
    out.append(r"\definecolor{worstred}{HTML}{C62828}")
    if table_env:
        out.append(r"\begin{table}[h]")
        out.append(r"\centering")
        if caption_first:
            out.append(r"\caption{" + caption + r"}")
            out.append(r"\label{" + label + r"}")
    out.append(tabular_size)
    out.append(r"\setlength{\tabcolsep}{" + col_sep + r"}")
    out.append(r"\renewcommand{\arraystretch}{" + row_stretch + r"}")
    out.append(r"\resizebox{\textwidth}{!}{%")
    out.append(r"\begin{tabular}{" + col_spec + r"}")
    out.append(r"\toprule")

    if any(t for t in block_titles):
        header = [""]
        for (m_cols, _), title in zip(blocks, block_titles):
            tag = (r"\multicolumn{" + str(len(m_cols)) + r"}{c|}{\textbf{"
                   + (title or "") + r"}}")
            header.append(tag)
        # Replace last `c|` with `c` if it is the rightmost block.
        line = " & ".join(header)
        line = line[::-1].replace("|}c{nmuloctlum\\", "}c{nmuloctlum\\", 1)[::-1]
        out.append(line + r" \\")

    method_header = [row_label_col]
    for (m_cols, _) in blocks:
        method_header.extend(METHOD_DISPLAY.get(m, m) for m in m_cols)
    out.append(" & ".join(method_header) + r" \\")
    out.append(r"\midrule")

    n_rows = len(rows)
    for ridx, (rname, _) in enumerate(rows):
        cells = [rname]
        for (m_cols, mat) in blocks:
            vals = mat[ridx]
            ranks = _row_ranks(vals, lower_better=lower_better)
            cells.extend(_fmt_cell(v, r, fmt=fmt) for v, r in zip(vals, ranks))
        out.append(" & ".join(cells) + r" \\")

    if include_mean:
        out.append(r"\midrule")
        mean_cells = [r"\textit{Mean}"]
        for (m_cols, mat) in blocks:
            means = []
            for ci in range(len(m_cols)):
                col_vals = [mat[ri][ci] for ri in range(n_rows)
                            if mat[ri][ci] is not None and not (isinstance(mat[ri][ci], float) and np.isnan(mat[ri][ci]))]
                means.append(np.mean(col_vals) if col_vals else np.nan)
            ranks = _row_ranks(means, lower_better=lower_better)
            mean_cells.extend(_fmt_cell(v, r, fmt=fmt) for v, r in zip(means, ranks))
        out.append(" & ".join(mean_cells) + r" \\")

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}%")
    out.append(r"}")

    if table_env:
        if not caption_first:
            out.append(r"\caption{" + caption + r"}")
            out.append(r"\label{" + label + r"}")
        out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def _pretty_pair(t: str) -> str:
    return t.replace("<->", r"$\leftrightarrow$")


def _build_table_rows(
    df_wide: pd.DataFrame,
    methods: list[str],
    transitions: list[str],
    pair_col: str = "Pair",
) -> list[tuple[str, list[float]]]:
    rows = []
    for t in transitions:
        sel = df_wide[df_wide[pair_col] == t]
        rname = _pretty_pair(t)
        if sel.empty:
            rows.append((rname, [np.nan] * len(methods)))
            continue
        vals = []
        for m in methods:
            if m not in sel.columns:
                vals.append(np.nan)
            else:
                v = sel.iloc[0][m]
                vals.append(float(v) if pd.notna(v) else np.nan)
        rows.append((rname, vals))
    return rows


def _build_table_rows_with_std(
    mean_wide: pd.DataFrame,
    std_wide: pd.DataFrame | None,
    methods: list[str],
    transitions: list[str],
    pair_col: str = "Pair",
) -> tuple[list[tuple[str, list[float]]], list[list[float]]]:
    rows: list[tuple[str, list[float]]] = []
    stds: list[list[float]] = []
    for t in transitions:
        m_sel = mean_wide[mean_wide[pair_col] == t]
        s_sel = std_wide[std_wide[pair_col] == t] if std_wide is not None else pd.DataFrame()
        rname = _pretty_pair(t)
        vals: list[float] = []
        rstds: list[float] = []
        for m in methods:
            if m_sel.empty or m not in m_sel.columns:
                vals.append(np.nan); rstds.append(np.nan); continue
            v = m_sel.iloc[0][m]
            vals.append(float(v) if pd.notna(v) else np.nan)
            if s_sel is None or s_sel.empty or m not in s_sel.columns:
                rstds.append(np.nan)
            else:
                sv = s_sel.iloc[0][m]
                rstds.append(float(sv) if pd.notna(sv) else np.nan)
        rows.append((rname, vals)); stds.append(rstds)
    return rows, stds


def _direction_to_tag(turn: str) -> str:
    return DIRECTION_TAG.get(turn, f"[{turn}]")


def _label_with_dir(label: str, turn: str) -> str:
    pretty = label.replace(" -> ", r"$\rightarrow$")
    return f"{pretty} {_direction_to_tag(turn)}"


# ─────────────────────────────────────────────────────────────────────────────
# Direction-aware row builders (for appendix tables)
# ─────────────────────────────────────────────────────────────────────────────

DIRECTION_ORDER = {"straight": 0, "left": 1, "right": 2, "non-det.": 3, "": 4}

def _sorted_directional_rows(wide: pd.DataFrame) -> list[tuple[str, str]]:
    seen = set()
    items = []
    for _, r in wide.iterrows():
        key = (r["Label"], r["Turn"])
        if key in seen:
            continue
        seen.add(key)
        items.append(key)
    items.sort(key=lambda lt: (lt[0], DIRECTION_ORDER.get(lt[1], 9)))
    return items


def _wide_rows_for_methods(
    wide: pd.DataFrame, methods: list[str]
) -> list[tuple[str, list[float]]]:
    rows = []
    items = _sorted_directional_rows(wide)
    for label, turn in items:
        sel = wide[(wide["Label"] == label) & (wide["Turn"] == turn)]
        vals = []
        for m in methods:
            if m not in sel.columns:
                vals.append(np.nan)
            else:
                v = sel.iloc[0][m]
                vals.append(float(v) if pd.notna(v) else np.nan)
        # Keep row only if at least one cell is populated.
        if all((isinstance(v, float) and np.isnan(v)) for v in vals):
            continue
        rows.append((_label_with_dir(label, turn), vals))
    return rows


def _wide_rows_for_methods_with_std(
    wide_mean: pd.DataFrame,
    wide_std: pd.DataFrame | None,
    methods: list[str],
) -> tuple[list[tuple[str, list[float]]], list[list[float]]]:
    rows: list[tuple[str, list[float]]] = []
    stds: list[list[float]] = []
    items = _sorted_directional_rows(wide_mean)
    for label, turn in items:
        sel = wide_mean[(wide_mean["Label"] == label) & (wide_mean["Turn"] == turn)]
        s_sel = wide_std[(wide_std["Label"] == label) & (wide_std["Turn"] == turn)] \
            if wide_std is not None else None
        vals: list[float] = []; rstds: list[float] = []
        for m in methods:
            if m not in sel.columns:
                vals.append(np.nan); rstds.append(np.nan); continue
            v = sel.iloc[0][m]
            vals.append(float(v) if pd.notna(v) else np.nan)
            if s_sel is None or s_sel.empty or m not in s_sel.columns:
                rstds.append(np.nan)
            else:
                sv = s_sel.iloc[0][m]
                rstds.append(float(sv) if pd.notna(sv) else np.nan)
        if all((isinstance(v, float) and np.isnan(v)) for v in vals):
            continue
        rows.append((_label_with_dir(label, turn), vals)); stds.append(rstds)
    return rows, stds


# ─────────────────────────────────────────────────────────────────────────────
# Table writers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TableSpec:
    name: str          # output file stem under paper/tables/
    text: str

def write_tables(specs: list[TableSpec], dry: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for s in specs:
        path = OUT / f"{s.name}.tex"
        if dry:
            print(f"[dry] would write {path} ({len(s.text)} chars)")
            continue
        path.write_text(s.text)
        print(f"wrote {path}")


def _family_best_per_row(
    wide: pd.DataFrame,
    transitions: list[str],
    lower_better: bool,
    pair_col: str = "Pair",
    std_wide: pd.DataFrame | None = None,
) -> tuple[list[tuple[str, list[float]]], list[list[float]], dict[str, list[str]]]:
    """For each transition row and each family, pick the variant with the best mean value.

    Returns (rows, stds, picks). stds is per-row list aligned to family order; std comes
    from the same variant chosen as best mean.
    """
    rows: list[tuple[str, list[float]]] = []
    stds: list[list[float]] = []
    picks: dict[str, list[str]] = {}
    families = [FAMILIES[f] for f in FAMILY_ORDER]
    for t in transitions:
        sel = wide[wide[pair_col] == t]
        s_sel = std_wide[std_wide[pair_col] == t] if std_wide is not None else None
        rname = _pretty_pair(t)
        best_vals: list[float] = []
        best_stds: list[float] = []
        best_picks: list[str] = []
        for fam_methods in families:
            cands = []
            for m in fam_methods:
                if m in sel.columns and not sel.empty:
                    v = sel.iloc[0][m]
                    if pd.notna(v):
                        sv = np.nan
                        if s_sel is not None and not s_sel.empty and m in s_sel.columns:
                            raw = s_sel.iloc[0][m]
                            sv = float(raw) if pd.notna(raw) else np.nan
                        cands.append((float(v), sv, m))
            if not cands:
                best_vals.append(np.nan); best_stds.append(np.nan); best_picks.append("---")
            else:
                best = min(cands, key=lambda x: x[0]) if lower_better else max(cands, key=lambda x: x[0])
                best_vals.append(best[0]); best_stds.append(best[1])
                best_picks.append(METHOD_DISPLAY[best[2]])
        rows.append((rname, best_vals)); stds.append(best_stds)
        picks[t] = best_picks
    return rows, stds, picks


def _picks_table(picks_per_block: list[dict[str, list[str]]], transitions: list[str]) -> str:
    """Compact 'best variant chosen' annotation table for the caption."""
    # Flatten unique sets per family.
    out_lines = []
    for tname in transitions:
        for blk_picks in picks_per_block:
            pass
    return ""


def _emit_three_block_with_std(
    block_specs: list[tuple[str, str, list[tuple[str, list[float]]], list[list[float]], bool]],
) -> str:
    """Emit a side-by-side multi-block tabular. Main-table cells show value only (no inline ±);
    a $^\dagger$ marks single-seed cells; two summary rows give per-column mean and std of ±."""
    n_m = len(FAMILY_ORDER)
    cspec = "|".join(["c" * n_m] * len(block_specs))
    titles = [
        rf"\multicolumn{{{n_m}}}{{c{'|' if i < len(block_specs)-1 else ''}}}{{\textbf{{{h}}}}}"
        for i, (h, _, _, _, _) in enumerate(block_specs)
    ]
    out = [
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|" + cspec + r"}",
        r"\toprule",
        " & " + " & ".join(titles) + r" \\",
        "Transition & " + " & ".join(FAMILY_ORDER * len(block_specs)) + r" \\",
        r"\midrule",
    ]
    n_rows = len(block_specs[0][2])
    for ri in range(n_rows):
        rname = block_specs[0][2][ri][0]
        cells = [rname]
        for (_, _, rows, stds, lb) in block_specs:
            vals = rows[ri][1]
            ranks = _row_ranks(vals, lower_better=lb)
            for v, s, r in zip(vals, stds[ri], ranks):
                # single-seed = the value exists but its std is NaN.
                single = (v is not None and not (isinstance(v, float) and np.isnan(v))
                          and (s is None or (isinstance(s, float) and np.isnan(s))))
                cells.append(_fmt_cell(v, r, fmt="{:.3f}", std=None, single_seed=single))
        out.append(" & ".join(cells) + r" \\")

    # Two summary rows per block, each summarizing the per-cell ± distribution per column.
    out.append(r"\midrule")
    mean_cells = [r"\textit{Mean of $\pm$}"]
    std_cells  = [r"\textit{Std of $\pm$}"]
    for (_, _, rows, stds, _lb) in block_specs:
        for ci in range(n_m):
            col_stds = [stds[ri][ci] for ri in range(n_rows)
                        if stds[ri][ci] is not None
                        and not (isinstance(stds[ri][ci], float) and np.isnan(stds[ri][ci]))]
            m = float(np.mean(col_stds)) if col_stds else float("nan")
            s = float(np.std(col_stds, ddof=1)) if len(col_stds) >= 2 else float("nan")
            mean_cells.append("---" if np.isnan(m) else f"{m:.3f}")
            std_cells .append("---" if np.isnan(s) else f"{s:.3f}")
    out.append(" & ".join(mean_cells) + r" \\")
    out.append(" & ".join(std_cells) + r" \\")

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}%")
    out.append(r"}")
    return "\n".join(out) + "\n"


def main_alignment_dtw_mmd_kid() -> TableSpec:
    """figure4_alignment_rodent.tex — three panels: DTW + MMD + KID, family-best with ±std."""
    dtw_mean_long, dtw_std_long = load_dtw_meanstd()
    dtw_mean = collapse_direction(long_to_wide(dtw_mean_long))
    dtw_std  = collapse_direction(long_to_wide(dtw_std_long.rename(columns={"std": "value"})))
    mmd_m, mmd_s = load_wide_meanstd(FA / "tables/mmd_rbf_table.csv",
                                      FA / "tables/mmd_rbf_table_std.csv")
    kid_m, kid_s = load_wide_meanstd(FA / "tables/kid_table.csv",
                                      FA / "tables/kid_table_std.csv")
    mmd_m, mmd_s = collapse_direction(mmd_m), collapse_direction(mmd_s)
    kid_m, kid_s = collapse_direction(kid_m), collapse_direction(kid_s)
    transitions = [f"{a}<->{b}" for a, b in ORDERED_PAIRS if (a, b) != ("Turn", "Walk")]

    dtw_rows, dtw_stds, _ = _family_best_per_row(dtw_mean, transitions, lower_better=True, std_wide=dtw_std)
    mmd_rows, mmd_stds, _ = _family_best_per_row(mmd_m,    transitions, lower_better=True, std_wide=mmd_s)
    kid_rows, kid_stds, _ = _family_best_per_row(kid_m,    transitions, lower_better=True, std_wide=kid_s)

    body = _emit_three_block_with_std([
        (r"DTW to GT ($\downarrow$)", "down", dtw_rows, dtw_stds, True),
        (r"MMD ($\downarrow$)",       "down", mmd_rows, mmd_stds, True),
        (r"KID ($\downarrow$)",       "down", kid_rows, kid_stds, True),
    ])
    return TableSpec("figure4_alignment_rodent", body)


def long_to_wide_std(std_long: pd.DataFrame) -> pd.DataFrame:
    return std_long.pivot_table(
        index=["Label", "Turn"], columns="method", values="std", aggfunc="mean"
    ).reset_index()


def main_warping_table() -> TableSpec:
    """figure4_warping_rodent.tex — Consistency + Sync ratio + Non-diag, family-best with ±std.

    All three blocks now come from per-seed feature_analysis CSVs so the
    per-cell std (across 3 random seeds) is available for every block. The
    Sync-ratio and Non-diag CSVs are produced by the same per-seed loop in
    ``main_transitions.run_analysis`` that writes ``dtw_comparison_table.csv``,
    so any (Label, Turn) cell that exists in the DTW table also exists here.
    """
    cons_mean_long, cons_std_long = load_consistency_meanstd()
    cons_m = collapse_direction(long_to_wide(cons_mean_long))
    cons_s = collapse_direction(long_to_wide(cons_std_long.rename(columns={"std": "value"})))

    sync_mean_long, sync_std_long = load_sync_ratio_meanstd()
    sync_m = collapse_direction(long_to_wide(sync_mean_long))
    sync_s = collapse_direction(long_to_wide(sync_std_long.rename(columns={"std": "value"})))

    nd_mean_long, nd_std_long = load_non_diag_meanstd()
    nd_m = collapse_direction(long_to_wide(nd_mean_long))
    nd_s = collapse_direction(long_to_wide(nd_std_long.rename(columns={"std": "value"})))

    transitions = [f"{a}<->{b}" for a, b in ORDERED_PAIRS]

    cons_rows, cons_stds, _ = _family_best_per_row(cons_m, transitions, lower_better=True, std_wide=cons_s)
    sync_rows, sync_stds, _ = _family_best_per_row(sync_m, transitions, lower_better=False, std_wide=sync_s)
    nd_rows,   nd_stds,   _ = _family_best_per_row(nd_m,   transitions, lower_better=False, std_wide=nd_s)

    body = _emit_three_block_with_std([
        (r"Consistency ($\downarrow$)",  "down", cons_rows, cons_stds, True),
        (r"Sync ratio ($\uparrow$)",     "up",   sync_rows, sync_stds, False),
        (r"Non-diag ($\uparrow$)",       "up",   nd_rows,   nd_stds,   False),
    ])
    return TableSpec("figure4_warping_rodent", body)


def appendix_directional_table(
    name: str,
    wide: pd.DataFrame,
    *,
    metric_name: str,
    label: str,
    lower_better: bool,
    caption_extra: str = "",
    fmt: str = "{:.3f}",
    wide_std: pd.DataFrame | None = None,
) -> TableSpec:
    methods = METHOD_ORDER_FULL
    rows, stds = _wide_rows_for_methods_with_std(wide, wide_std, methods)
    arrow = r"\downarrow" if lower_better else r"\uparrow"
    direction = "lower" if lower_better else "higher"
    has_std = any(any(not (isinstance(s, float) and np.isnan(s)) for s in row) for row in stds)
    std_note = ""
    if has_std:
        std_note = (r" $^\dagger$ marks cells populated by only a single seed (no $\pm$"
                    r" available). The two summary rows under the table give the per-column"
                    r" mean and standard deviation of the per-cell $\pm$ (std across 3"
                    r" random seeds), aggregated over the populated rows.")
    caption = (
        rf"\textbf{{{metric_name} by directed transition (${arrow}$ {direction} is better).}} "
        rf"\textbf{{Bold}} = best per row over populated cells; \underline{{underline}} = second. "
        rf"--- = method did not discover this transition. "
        rf"Direction tags: [L] left, [R] right, [S] straight. $^*$Ours.{caption_extra}{std_note}"
    )
    body = _emit_color_table_with_std(
        rows=rows, stds=stds, methods=methods, lower_better=lower_better,
        caption=caption, label=label, fmt=fmt,
    )
    return TableSpec(name, body)


def _emit_color_table_with_std(
    rows, stds, methods, lower_better, caption, label, fmt="{:.3f}"
) -> str:
    n_m = len(methods)
    out = [
        r"\definecolor{bestgreen}{HTML}{2E7D32}",
        r"\definecolor{secondyellow}{HTML}{F9A825}",
        r"\definecolor{worstred}{HTML}{C62828}",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\tiny",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|" + "c" * n_m + r"}",
        r"\toprule",
        "Transition & " + " & ".join(METHOD_DISPLAY.get(m, m) for m in methods) + r" \\",
        r"\midrule",
    ]
    # Per-cell: value only (no inline ±); $^\dagger$ marks single-seed cells. Color from per-row rank.
    for (rname, vals), rstds in zip(rows, stds):
        ranks = _row_ranks(vals, lower_better=lower_better)
        line = [rname]
        for v, s, r in zip(vals, rstds, ranks):
            single = (v is not None and not (isinstance(v, float) and np.isnan(v))
                      and (s is None or (isinstance(s, float) and np.isnan(s))))
            line.append(_fmt_cell(v, r, fmt=fmt, std=None, single_seed=single))
        out.append(" & ".join(line) + r" \\")
    # Two summary rows per column, both about the per-cell std (±) values:
    # mean of ± across populated rows, and std of ± across populated rows.
    col_pm_mean: list[float] = []
    col_pm_std: list[float] = []
    for ci in range(n_m):
        col_stds = [stds[ri][ci] for ri in range(len(stds))
                    if stds[ri][ci] is not None
                    and not (isinstance(stds[ri][ci], float) and np.isnan(stds[ri][ci]))]
        col_pm_mean.append(float(np.mean(col_stds)) if col_stds else float("nan"))
        col_pm_std.append(float(np.std(col_stds, ddof=1)) if len(col_stds) >= 2 else float("nan"))

    def _fmt_summary(v: float, f: str = "{:.3f}") -> str:
        return "---" if (v is None or np.isnan(v)) else f.format(v)

    out.append(r"\midrule")
    mean_cells = [r"\textit{Mean of $\pm$}"] + [_fmt_summary(v) for v in col_pm_mean]
    std_cells  = [r"\textit{Std of $\pm$}"]  + [_fmt_summary(v) for v in col_pm_std]
    out.append(" & ".join(mean_cells) + r" \\")
    out.append(" & ".join(std_cells) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}%")
    out.append(r"}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def appendix_dtw_detailed() -> TableSpec:
    mean_long, std_long = load_dtw_meanstd()
    wide_m = long_to_wide(mean_long)
    wide_s = long_to_wide(std_long.rename(columns={"std": "value"}))
    return appendix_directional_table(
        "appendix_dtw_detailed", wide_m,
        metric_name="DTW distance to ground truth",
        label="tab:dtw-detailed", lower_better=True,
        wide_std=wide_s,
    )


def appendix_consistency() -> TableSpec:
    mean_long, std_long = load_consistency_meanstd()
    wide_m = long_to_wide(mean_long)
    wide_s = long_to_wide(std_long.rename(columns={"std": "value"}))
    return appendix_directional_table(
        "appendix_consistency", wide_m,
        metric_name="Internal cluster consistency",
        label="tab:consistency-detailed", lower_better=True,
        wide_std=wide_s,
    )


def appendix_mmd() -> TableSpec:
    mean_w, std_w = load_wide_meanstd(FA / "tables/mmd_rbf_table.csv",
                                       FA / "tables/mmd_rbf_table_std.csv")
    return appendix_directional_table(
        "appendix_mmd", mean_w,
        metric_name="Maximum Mean Discrepancy (RBF kernel)",
        label="tab:mmd-detailed", lower_better=True,
        wide_std=std_w,
    )


def appendix_kid() -> TableSpec:
    mean_w, std_w = load_wide_meanstd(FA / "tables/kid_table.csv",
                                       FA / "tables/kid_table_std.csv")
    return appendix_directional_table(
        "appendix_kid", mean_w,
        metric_name="Kernel Inception Distance",
        label="tab:kid-detailed", lower_better=True,
        wide_std=std_w,
    )


def appendix_dtw_sync_ratio() -> TableSpec:
    mean_long, std_long = load_sync_ratio_meanstd()
    wide_m = long_to_wide(mean_long)
    wide_s = long_to_wide(std_long.rename(columns={"std": "value"}))
    return appendix_directional_table(
        "appendix_dtw_sync_ratio", wide_m,
        metric_name="DTW-to-frame-sync ratio",
        label="tab:warping-dtw-sync", lower_better=False,
        caption_extra=" Higher means less reliance on warping rescue.",
        wide_std=wide_s,
    )


def appendix_non_diag() -> TableSpec:
    mean_long, std_long = load_non_diag_meanstd()
    wide_m = long_to_wide(mean_long)
    wide_s = long_to_wide(std_long.rename(columns={"std": "value"}))
    return appendix_directional_table(
        "appendix_non_diag_frac", wide_m,
        metric_name="Non-diagonal fraction of DTW warping path",
        label="tab:warping-non-diag", lower_better=False,
        caption_extra=" Higher means a more compressed primitive stretched to longer GT.",
        wide_std=wide_s,
    )


def appendix_coverage() -> TableSpec:
    """Discovery indicator (✓/×) for each method on the 6 macro-transitions."""
    dtw = collapse_direction(long_to_wide(load_dtw_long()))
    transitions = [f"{a}<->{b}" for a, b in ORDERED_PAIRS]
    methods = METHOD_ORDER_FULL

    # ground-truth instance counts (collapsed direction). Restrict to the eval split
    # (clips evaluated by the methods); use the union of clips appearing in the DTW CSV
    # so the count reflects what evaluation actually covered.
    gt_csv = REPO / "behavior_labels/rodent_labels.csv"
    df = pd.read_csv(gt_csv)
    df = df[df["Track"] == "behavior"].reset_index(drop=True)
    eval_clips = set()
    raw = pd.read_csv(FA / "dtw_comparison_table.csv")
    method_cols = [c for c in raw.columns if c not in ("Label", "Turn")]
    # The DTW CSV does not name clips; fall back to the test-set range from rodent.csv labels.
    # We use clips whose Video name parses to clip_<id>.mp4 with id in [642, 841].
    for v in df["Video"].unique():
        try:
            cid = int(v.replace("clip_", "").replace(".mp4", ""))
            if 642 <= cid <= 841:
                eval_clips.add(v)
        except Exception:
            pass
    sub_df = df[df["Video"].isin(eval_clips)] if eval_clips else df
    gt_counts: dict[str, int] = {f"{a}<->{b}": 0 for a, b in ORDERED_PAIRS}
    for vid, sub in sub_df.groupby("Video"):
        sub = sub.sort_values("Start_Frame").reset_index(drop=True)
        for i in range(len(sub) - 1):
            a, b = sub.loc[i, "Label"], sub.loc[i + 1, "Label"]
            if a == b: continue
            key = f"{a}<->{b}" if (a, b) in ORDERED_PAIRS else f"{b}<->{a}"
            if key in gt_counts:
                gt_counts[key] += 1

    out = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{\textbf{Behavioral coverage by method (4-behavior set).} Each cell indicates whether the method discovered ($\checkmark$) or missed ($\times$) at least one cluster majority-voted to the given unordered transition. GT $n$ = number of ground-truth transition instances in \texttt{behavior\_labels/rodent\_labels.csv}.}",
        r"\label{tab:coverage}",
        r"\tiny",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lr" + "c" * len(methods) + r"}",
        r"\toprule",
        "Transition & GT $n$ & " + " & ".join(METHOD_DISPLAY[m] for m in methods) + r" \\",
        r"\midrule",
    ]
    totals = [0] * len(methods)
    for t in transitions:
        sel = dtw[dtw["Pair"] == t]
        cells = [t.replace("<->", r"$\leftrightarrow$"), str(gt_counts.get(t, 0))]
        for i, m in enumerate(methods):
            ok = (m in sel.columns) and (not sel.empty) and pd.notna(sel.iloc[0][m])
            if ok:
                cells.append(r"$\checkmark$")
                totals[i] += 1
            else:
                cells.append(r"$\times$")
        out.append(" & ".join(cells) + r" \\")
    out.append(r"\midrule")
    out.append(r"\textit{Total discovered} & & " + " & ".join(str(t) for t in totals) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}%")
    out.append(r"}")
    out.append(r"\vspace{2pt}")
    out.append(r"\\\scriptsize{$^*$Ours.}")
    out.append(r"\end{table}")
    return TableSpec("appendix_coverage", "\n".join(out) + "\n")


def appendix_scale_sweep() -> TableSpec:
    """Per-transition discovery across KPMS and Mapper sweep variants (new 4-behavior set)."""
    df = pd.read_csv(SS / "discovery_connections_binary_wide.csv")
    transitions = [f"{a} <-> {b}" for a, b in ORDERED_PAIRS]  # match CSV format
    df = df.set_index("transition")

    kpms_cols = [c for c in df.columns if c.startswith("kpms_")]
    mapper_cols = [c for c in df.columns if c.startswith("mapper_")]

    # Drop columns that fire zero canonical transitions.
    surviving_kpms = [c for c in kpms_cols if df.loc[transitions, c].sum() > 0]
    surviving_mapper = [c for c in mapper_cols if df.loc[transitions, c].sum() > 0]

    def short_kpms(c: str) -> str:
        # kpms_k1e3_n30 -> $(10^{3},30)$
        try:
            _, ke, ns = c.split("_")
            assert ke.startswith("k1e")
            kexp = ke[3:]
            n = ns[1:]
            return rf"$(10^{{{kexp}}},{n})$"
        except Exception:
            return c

    def short_mapper(c: str) -> str:
        # mapper_intention_eccentricity_n10 -> "int / ecc / 10"
        parts = c.split("_")
        space = parts[1][:3]
        n = parts[-1][1:]
        lens = "_".join(parts[2:-1])
        lens_short = {"eccentricity": "ecc", "knn_density": "knn", "pca_1d": "pca"}.get(lens, lens)
        return f"{space}/{lens_short}/{n}"

    def render_block(cols: list[str], short_fn) -> str:
        if not cols:
            return ""
        out = [r"\begin{tabular}{l" + "c" * len(cols) + r"}",
               r"\toprule",
               "Transition & " + " & ".join(short_fn(c) for c in cols) + r" \\",
               r"\midrule"]
        totals = [0] * len(cols)
        for t in transitions:
            cells = [t.replace("<->", r"$\leftrightarrow$")]
            for i, c in enumerate(cols):
                v = int(df.loc[t, c]) if t in df.index else 0
                cells.append(r"$\checkmark$" if v else "")
                totals[i] += v
            out.append(" & ".join(cells) + r" \\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        return "\n".join(out)

    kpms_block = render_block(surviving_kpms, short_kpms)

    # Compact Mapper block: aggregate by (feature_space, lens) — a row counts as
    # discovered if ANY n in that (space,lens) sweep discovered it.
    space_map = {"intention": "Intention", "qpos": "Qpos", "ekp": "EKP"}
    lens_map  = {"eccentricity": "Eccentricity", "knn_density": "kNN-density",
                 "pca_1d": "PCA-1D"}
    sweep_groups: dict[tuple[str, str], list[str]] = {}
    for c in surviving_mapper:
        parts = c.split("_")
        space = parts[1]
        lens = "_".join(parts[2:-1])
        sweep_groups.setdefault((space, lens), []).append(c)
    # Order: intention then qpos then ekp; eccentricity, knn_density, pca_1d within.
    space_order = ["intention", "qpos", "ekp"]
    lens_order  = ["eccentricity", "knn_density", "pca_1d"]
    grp_keys = sorted(sweep_groups.keys(),
                      key=lambda k: (space_order.index(k[0]), lens_order.index(k[1])))

    out_lines = [r"\begin{tabular}{l" + "c" * len(grp_keys) + r"}",
                 r"\toprule",
                 "Transition & " + " & ".join(
                     rf"{space_map[s]}/{lens_map[l]}" for (s, l) in grp_keys
                 ) + r" \\",
                 r"\midrule"]
    for t in transitions:
        cells = [t.replace("<->", r"$\leftrightarrow$")]
        for (s, l) in grp_keys:
            cols = sweep_groups[(s, l)]
            any_disc = any(int(df.loc[t, c]) if t in df.index else 0 for c in cols)
            cells.append(r"$\checkmark$" if any_disc else "")
        out_lines.append(" & ".join(cells) + r" \\")
    out_lines.append(r"\bottomrule")
    out_lines.append(r"\end{tabular}")
    mapper_block = "\n".join(out_lines)

    # Union counts.
    union_kpms = sum(int(df.loc[transitions, surviving_kpms].max(axis=1).any()) for _ in [0])  # placeholder
    union_kpms = int((df.loc[transitions, surviving_kpms].sum(axis=1) > 0).sum())
    union_mapper = int((df.loc[transitions, surviving_mapper].sum(axis=1) > 0).sum())
    union_combined = int((df.loc[transitions, surviving_kpms + surviving_mapper].sum(axis=1) > 0).sum())
    n_total = len(transitions)

    out = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{\textbf{Baseline scale-hyperparameter sweep, per-transition discovery (4-behavior set).} Each cell indicates whether a baseline variant discovered ($\checkmark$) the canonical unordered transition on that row. (a) KPMS stickiness $\kappa$ spans five orders of magnitude (a timescale knob, larger $\kappa$ $\Rightarrow$ longer expected syllable dwell). (b) Mapper variants are aggregated by (feature space, lens): a cell is checked if any $n_{\text{intervals}} \in \{5, 10, 15, 20, 25\}$ in that group discovered the transition. The full per-$n$ grid is in \texttt{outputs/analysis/scale\_sweep/v1/v1/cfg\_01f5a4b3/discovery/}.}",
        r"\label{tab:scale-sweep}",
        "",
        r"\textit{(a) KPMS surviving variants. Columns: $(\kappa, N_{\text{states}})$.}",
        r"\\[2pt]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        kpms_block,
        "",
        r"\vspace{8pt}",
        "",
        r"\textit{(b) Mapper surviving variants, aggregated by (feature space, lens). Cell is $\checkmark$ if any $n_{\text{intervals}}$ in $\{5,10,15,20,25\}$ discovered the row.}",
        r"\\[2pt]",
        r"\resizebox{\textwidth}{!}{%",
        r"\tiny",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        mapper_block,
        r"}",
        r"\vspace{4pt}",
        rf"\scriptsize{{Union over surviving KPMS variants: {union_kpms} of {n_total} canonical transitions. Union over surviving Mapper variants: {union_mapper} of {n_total}. Combined baseline union: {union_combined} of {n_total}.}}",
        r"\end{table}",
    ]
    return TableSpec("appendix_scale_sweep", "\n".join(out) + "\n")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="dry-run; print but don't write")
    args = p.parse_args()

    specs: list[TableSpec] = [
        main_alignment_dtw_mmd_kid(),
        main_warping_table(),
        appendix_dtw_detailed(),
        appendix_consistency(),
        appendix_mmd(),
        appendix_kid(),
        appendix_dtw_sync_ratio(),
        appendix_non_diag(),
        appendix_coverage(),
        appendix_scale_sweep(),
    ]
    write_tables(specs, dry=args.check)


if __name__ == "__main__":
    main()
