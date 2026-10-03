#!/usr/bin/env python3
"""Generate every .tex table under paper/tables/ from the master CSVs.

Usage:
    python -m topo_mimic.figure_generation.build_tables           # writes all tables
    python -m topo_mimic.figure_generation.build_tables --check   # prints what would change
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FA   = REPO / "outputs/analysis/feature_analysis"
WA   = REPO / "outputs/analysis/tables/warping_analysis"
def _latest_scale_sweep_discovery() -> Path:
    """Most recently written scale-sweep discovery directory.

    The sweep writes to a config-hash directory whose name and nesting depth both
    change between runs, so hardcoding a path silently rebuilds Table 3 from stale
    data. Resolve it instead.

    Prefers `discovery_all` (every swept variant scored) over `discovery`, which
    the shipped pipeline restricts to the EML-selected best-per-kappa KPMS fits
    and so cannot support an oracle-union claim.
    """
    root = REPO / "outputs/analysis/scale_sweep"
    cands = [p for p in root.glob("**/cfg_*/discovery_all")
             if (p / "discovery_connections_binary_wide.csv").exists()]
    if not cands:
        cands = [p for p in root.glob("**/cfg_*/discovery")
                 if (p / "discovery_connections_binary_wide.csv").exists()]
    if not cands:
        raise FileNotFoundError(f"no scale-sweep discovery output under {root}")
    return max(cands, key=lambda p: (p / "discovery_connections_binary_wide.csv").stat().st_mtime)


SS   = _latest_scale_sweep_discovery()
KS   = REPO / "outputs/analysis/method_analysis/k_sweep"
PPS  = REPO / "outputs/analysis/method_analysis/path_persistence_sweep"
OUT  = REPO / "paper/bian_NeurReps_2026/tables"

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
    # custom_vae deliberately omitted: the GRU-VAE baseline is posterior-collapsed
    # and excluded from the paper (2026-08 audit). Keeping it here would re-add a
    # VAE column to every appendix table.
    "w2_full_kmeans"     : "Full-Int",
}
METHOD_ORDER_FULL = list(METHOD_DISPLAY.keys())

# Family groupings used to pick best-of-family per row in the main tables.
# DM cycles (DMC) and paths (DMP) are reported as separate columns; the VAE and
# full-clip (Full) baselines are dropped from the main tables (full detail in the
# appendix per-variant tables).
FAMILIES = {
    "DMC$^*$" : ["w2_dmc_int_kmeans", "w2_dmc_qpos_kmeans", "w2_dmc_ekp_kmeans"],
    "DMP$^*$" : ["w2_dmp_int_kmeans", "w2_dmp_qpos_kmeans", "w2_dmp_ekp_kmeans"],
    "Mapper"  : ["mapper_int", "mapper_qpos", "mapper_ekp"],
    "KPMS"    : ["kpms_intention", "kpms_keypoint"],
}
FAMILY_ORDER = ["DMC$^*$", "DMP$^*$", "Mapper", "KPMS"]

# One fixed variant per family, held constant down every column of Tables 1-2.
# Chosen by transition information I(T;Y) (tab:information-loss): DMC-Int 0.873,
# DMP-Qpos 0.895, Mapper-Qpos 0.322, KPMS-KP 0.525. Mapper moved ekp -> qpos when
# its inputs were aligned with DM-G's (root strip / egocentric keypoints + PCA-10)
# and the cover switched to tda-mapper's standard cubical algorithm.
# Column headers name the fixed variant, so a column can never be misread as a
# family-best mixture (it was, before 2026-08).
def _two_line(disp: str) -> str:
    """Header as family on one line, feature space parenthesised and small below.

    Saves a lot of horizontal room in tables whose columns are method-variants,
    which is what forces the \\resizebox on the wide grids.
    """
    star = "$^*$" if "$^*$" in disp else ""
    d = disp.replace("$^*$", "")
    if "-" not in d:
        return disp
    fam, space = d.split("-", 1)
    return r"\shortstack{" + fam + star + r"\\{\tiny (" + space + r")}}"


FAMILY_HEADER = {
    "DMC$^*$": _two_line("DMC-Int$^*$"),
    "DMP$^*$": _two_line("DMP-Qpos$^*$"),
    "Mapper" : _two_line("Mapper-Qpos"),
    "KPMS"   : _two_line("KPMS-KP"),
}

FAMILY_FIXED = {
    "DMC$^*$": "w2_dmc_int_kmeans",
    "DMP$^*$": "w2_dmp_qpos_kmeans",
    "Mapper" : "mapper_qpos",
    "KPMS"   : "kpms_keypoint",
}

# Table 1 row rule. "multi_family" keeps a directed transition only when more
# than one of FAMILIES (any variant) yields alignment metrics for it, the rule stated in
# the paper; "any_fixed" keeps every row that one of the four fixed variants
# populates. Rows none of the fixed variants populates are dropped either way.
TABLE1_ROW_RULE = "multi_family"

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


# ── Directed-transition rows for the main tables (one row per directed
#    transition + turn direction, like the appendix full tables) ──────────────
_DIR_TAG = {"left": "[L]", "right": "[R]", "straight": "[S]", "none": "", "": ""}


def _add_rowkey(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each (Label, Turn) row with a directed RowKey 'Label | turn'."""
    df = df.copy()
    df["RowKey"] = (df["Label"].astype(str).str.strip() + " | "
                    + df["Turn"].astype(str).str.strip().str.lower())
    return df


def _pretty_directed(key: str) -> str:
    """'Immobile -> Rear | left' -> 'Immobile$\\to$Rear [L]'."""
    if " | " not in key:
        return _pretty_pair(key)
    lab, turn = key.split(" | ", 1)
    parts = [s.strip() for s in lab.split("->")]
    name = f"{parts[0]}$\\to${parts[1]}" if len(parts) == 2 else lab
    tag = _DIR_TAG.get(turn.strip().lower(), "")
    return (name + (" " + tag if tag else "")).strip()


def _n_families_populated(wide: pd.DataFrame, row_key: str) -> int:
    """Number of FAMILIES entries with at least one variant populated at `row_key`."""
    row = wide[wide["RowKey"] == row_key]
    if row.empty:
        return 0
    n = 0
    for variants in FAMILIES.values():
        cols = [c for c in variants if c in row.columns]
        if any(not pd.isna(row.iloc[0][c]) for c in cols):
            n += 1
    return n


def _directed_keys(wide: pd.DataFrame) -> list[str]:
    """Ordered list of directed RowKeys present in `wide`, grouped by canonical
    pair (ORDERED_PAIRS) then direction."""
    pair_rank = {f"{a}<->{b}": i for i, (a, b) in enumerate(ORDERED_PAIRS)}
    dir_rank = {"left": 0, "right": 1, "straight": 2, "none": 3, "": 4}
    rows, seen = [], set()
    for _, r in wide.iterrows():
        lab = str(r["Label"]).strip()
        turn = str(r["Turn"]).strip().lower()
        pair = _canonical_pair(lab)
        key = f"{lab} | {turn}"
        if key in seen:
            continue
        seen.add(key)
        rows.append((pair_rank.get(pair, 99), pair, dir_rank.get(turn, 9), key))
    rows.sort()
    return [k[-1] for k in rows]


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
    """One FIXED variant per family, held constant down every column.

    Each family is represented by the variant with the highest transition information
    (Table~\\ref{tab:information-loss}); see FAMILY_FIXED. A row where that variant is
    unpopulated is reported as missing rather than being backfilled by a sibling.

    This deliberately does NOT pick the best variant per row. Doing so is a
    within-family oracle: it lets a column be assembled from whichever feature space
    happens to win each transition, which inflates every family, ours included, and
    makes the column correspond to no runnable configuration.

    Returns (rows, stds, picks); stds is aligned to family order.
    """
    rows: list[tuple[str, list[float]]] = []
    stds: list[list[float]] = []
    picks: dict[str, list[str]] = {}
    families = [FAMILIES[f] for f in FAMILY_ORDER]
    for t in transitions:
        sel = wide[wide[pair_col] == t]
        s_sel = std_wide[std_wide[pair_col] == t] if std_wide is not None else None
        rname = _pretty_directed(t)
        best_vals: list[float] = []
        best_stds: list[float] = []
        best_picks: list[str] = []
        # `lower_better` is unused: the variant is fixed per family, not chosen per row.
        for fam_name, fam_methods in zip(FAMILY_ORDER, families):
            m = FAMILY_FIXED.get(fam_name)
            if m is None or m not in fam_methods:
                m = fam_methods[0]
            v = sel.iloc[0][m] if (m in sel.columns and not sel.empty) else np.nan
            if pd.isna(v):
                best_vals.append(np.nan); best_stds.append(np.nan); best_picks.append("---")
                continue
            sv = np.nan
            if s_sel is not None and not s_sel.empty and m in s_sel.columns:
                raw = s_sel.iloc[0][m]
                sv = float(raw) if pd.notna(raw) else np.nan
            best_vals.append(float(v)); best_stds.append(sv)
            best_picks.append(METHOD_DISPLAY[m])
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
        "Transition & " + " & ".join(
            [FAMILY_HEADER[f] for f in FAMILY_ORDER] * len(block_specs)
        ) + r" \\",
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


def main_combined_table() -> TableSpec:
    """main_combined.tex — the single main-text results table.

    Merges the former alignment and warping tables and keeps only the three
    metrics the text argues from: DTW, KID and consistency. Rows where none of
    the four reported variants discovers anything are dropped, which matters now
    that each column is pinned to one variant rather than a family-best mixture.
    """
    dtw_mean_long, dtw_std_long = load_dtw_meanstd()
    dtw_mean = _add_rowkey(long_to_wide(dtw_mean_long))
    dtw_std = _add_rowkey(long_to_wide(dtw_std_long.rename(columns={"std": "value"})))
    mmd_m, mmd_s = load_wide_meanstd(FA / "tables/kid_table.csv",
                                     FA / "tables/kid_table_std.csv")
    mmd_m, mmd_s = _add_rowkey(mmd_m), _add_rowkey(mmd_s)
    cons_mean_long, cons_std_long = load_consistency_meanstd()
    cons_m = _add_rowkey(long_to_wide(cons_mean_long))
    cons_s = _add_rowkey(long_to_wide(cons_std_long.rename(columns={"std": "value"})))
    nd_mean_long, nd_std_long = load_non_diag_meanstd()
    nd_m = _add_rowkey(long_to_wide(nd_mean_long))
    nd_s = _add_rowkey(long_to_wide(nd_std_long.rename(columns={"std": "value"})))

    transitions = _directed_keys(dtw_mean)
    d_r, d_s, _ = _family_best_per_row(dtw_mean, transitions, True, "RowKey", dtw_std)
    m_r, m_s, _ = _family_best_per_row(mmd_m, transitions, True, "RowKey", mmd_s)
    c_r, c_s, _ = _family_best_per_row(cons_m, transitions, True, "RowKey", cons_s)
    n_r, n_s, _ = _family_best_per_row(nd_m, transitions, True, "RowKey", nd_s)

    keep = [i for i in range(len(transitions))
            if any(not pd.isna(v) for rows in (d_r, m_r, c_r, n_r) for v in rows[i][1])]
    if TABLE1_ROW_RULE == "multi_family":
        keep = [i for i in keep if _n_families_populated(dtw_mean, transitions[i]) > 1]
    elif TABLE1_ROW_RULE != "any_fixed":
        raise ValueError(f"unknown TABLE1_ROW_RULE {TABLE1_ROW_RULE!r}")
    sel = lambda rows: [rows[i] for i in keep]
    # every block is lower-is-better, so the table reads in one direction throughout
    body = _emit_three_block_with_std([
        (r"DTW to GT ($\downarrow$)", "down", sel(d_r), sel(d_s), True),
        (r"Consistency ($\downarrow$)", "down", sel(c_r), sel(c_s), True),
        (r"Non-diag ($\downarrow$)", "down", sel(n_r), sel(n_s), True),
        (r"KID ($\downarrow$)", "down", sel(m_r), sel(m_s), True),
    ])
    return TableSpec("main_combined", body)


def main_alignment_dtw_mmd_kid() -> TableSpec:
    """figure4_alignment_rodent.tex — three panels: DTW + MMD + KID, family-best with ±std."""
    dtw_mean_long, dtw_std_long = load_dtw_meanstd()
    dtw_mean = _add_rowkey(long_to_wide(dtw_mean_long))
    dtw_std  = _add_rowkey(long_to_wide(dtw_std_long.rename(columns={"std": "value"})))
    mmd_m, mmd_s = load_wide_meanstd(FA / "tables/mmd_rbf_table.csv",
                                      FA / "tables/mmd_rbf_table_std.csv")
    kid_m, kid_s = load_wide_meanstd(FA / "tables/kid_table.csv",
                                      FA / "tables/kid_table_std.csv")
    mmd_m, mmd_s = _add_rowkey(mmd_m), _add_rowkey(mmd_s)
    kid_m, kid_s = _add_rowkey(kid_m), _add_rowkey(kid_s)
    transitions = _directed_keys(dtw_mean)   # one row per directed transition

    dtw_rows, dtw_stds, _ = _family_best_per_row(dtw_mean, transitions, lower_better=True, pair_col="RowKey", std_wide=dtw_std)
    mmd_rows, mmd_stds, _ = _family_best_per_row(mmd_m,    transitions, lower_better=True, pair_col="RowKey", std_wide=mmd_s)
    kid_rows, kid_stds, _ = _family_best_per_row(kid_m,    transitions, lower_better=True, pair_col="RowKey", std_wide=kid_s)

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
    cons_m = _add_rowkey(long_to_wide(cons_mean_long))
    cons_s = _add_rowkey(long_to_wide(cons_std_long.rename(columns={"std": "value"})))

    sync_mean_long, sync_std_long = load_sync_ratio_meanstd()
    sync_m = _add_rowkey(long_to_wide(sync_mean_long))
    sync_s = _add_rowkey(long_to_wide(sync_std_long.rename(columns={"std": "value"})))

    nd_mean_long, nd_std_long = load_non_diag_meanstd()
    nd_m = _add_rowkey(long_to_wide(nd_mean_long))
    nd_s = _add_rowkey(long_to_wide(nd_std_long.rename(columns={"std": "value"})))

    transitions = _directed_keys(cons_m)   # one row per directed transition

    cons_rows, cons_stds, _ = _family_best_per_row(cons_m, transitions, lower_better=True, pair_col="RowKey", std_wide=cons_s)
    sync_rows, sync_stds, _ = _family_best_per_row(sync_m, transitions, lower_better=False, pair_col="RowKey", std_wide=sync_s)
    nd_rows,   nd_stds,   _ = _family_best_per_row(nd_m,   transitions, lower_better=False, pair_col="RowKey", std_wide=nd_s)

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
    exclude_labels: frozenset = frozenset(),
) -> TableSpec:
    methods = METHOD_ORDER_FULL
    if exclude_labels:
        wide = wide[~wide["Label"].isin(exclude_labels)]
        if wide_std is not None:
            wide_std = wide_std[~wide_std["Label"].isin(exclude_labels)]
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
        "Transition & " + " & ".join(_two_line(METHOD_DISPLAY.get(m, m)) for m in methods) + r" \\",
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
        label="tab:warping-non-diag", lower_better=True,
        caption_extra=(" Lower means the warping path runs closer to the diagonal, so the "
                       "primitive matches the annotated window at its own timescale rather "
                       "than being stretched or compressed onto it."),
        wide_std=wide_s,
    )


# Table 2 row set. "discovered" lists the GT-populated classes that at least one
# benchmarked method discovers (the published table); "gt_populated" lists every
# class with a GT instance in the evaluation split, so undiscovered classes show
# as all-x rows and the coverage denominator is the true class count.
COVERAGE_ROW_SET = "discovered"


def appendix_coverage() -> TableSpec:
    """Discovery indicator (✓/×) per method for each DIRECTED, direction-turn
    transition class (Imm->Turn distinct from Turn->Imm; [L]/[R]/[S] split)."""
    import sys as _sys
    if str(REPO) not in _sys.path:
        _sys.path.insert(0, str(REPO))
    from topo_mimic.analysis.feature_analysis.ground_truth import find_all_transitions
    wide = long_to_wide(load_dtw_long())
    methods = METHOD_ORDER_FULL

    # Directed + direction-turn GT instance counts over the eval split (clips
    # 642-841), keyed by (directed "A -> B" label, turn in {left,right,straight}).
    _TN = {"LEFT": "left", "RIGHT": "right", "STRAIGHT": "straight"}
    gt_counts: dict[tuple[str, str], int] = {}
    for t in find_all_transitions(str(REPO / "behavior_labels/rodent_labels.csv")):
        if not (642 <= int(t.clip_id) <= 841):
            continue
        key = (f"{t.from_behavior} -> {t.to_behavior}",
               _TN.get(getattr(t.turn, "name", ""), "straight"))
        gt_counts[key] = gt_counts.get(key, 0) + 1

    out = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{\textbf{Behavioral coverage by method (directed, direction-turn).} Each cell indicates whether the method discovered ($\checkmark$) or missed ($\times$) at least one cluster majority-voted to the given \emph{directed} transition and turn direction (Immobile$\to$Turn is distinct from Turn$\to$Immobile). Direction tags: [L] left, [R] right, [S] straight. GT $n$ = ground-truth instances in the evaluation split.@@ROWNOTE@@}",
        r"\label{tab:coverage}",
        r"\tiny",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lr" + "c" * len(methods) + r"}",
        r"\toprule",
        "Transition & GT $n$ & " + " & ".join(_two_line(METHOD_DISPLAY[m]) for m in methods) + r" \\",
        r"\midrule",
    ]
    totals = [0] * len(methods)
    if COVERAGE_ROW_SET == "gt_populated":
        row_keys = sorted(gt_counts, key=lambda lt: (lt[0], DIRECTION_ORDER.get(lt[1], 9)))
    elif COVERAGE_ROW_SET == "discovered":
        row_keys = _sorted_directional_rows(wide)
    else:
        raise ValueError(f"unknown COVERAGE_ROW_SET {COVERAGE_ROW_SET!r}")
    n_rows = 0
    for label, turn in row_keys:
        gtn = gt_counts.get((label, turn), 0)
        if gtn == 0:  # no GT instances of this directed+directional class in the eval split
            continue
        n_rows += 1
        sel = wide[(wide["Label"] == label) & (wide["Turn"] == turn)]
        cells = [_label_with_dir(label, turn), str(gtn)]
        for i, m in enumerate(methods):
            ok = (not sel.empty) and (m in sel.columns) and pd.notna(sel.iloc[0][m])
            if ok:
                cells.append(r"$\checkmark$")
                totals[i] += 1
            else:
                cells.append(r"$\times$")
        out.append(" & ".join(cells) + r" \\")
    if COVERAGE_ROW_SET == "discovered":
        row_note = (f" Only the {n_rows} classes that at least one method discovers are listed;"
                    f" the evaluation split contains {len(gt_counts)} classes with ground-truth instances.")
    else:
        row_note = ""
    out = [line.replace("@@ROWNOTE@@", row_note) for line in out]
    out.append(r"\midrule")
    out.append(r"\textit{Total discovered} & & " + " & ".join(str(t) for t in totals) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}%")
    out.append(r"}")
    out.append(r"\vspace{2pt}")
    out.append(r"\\\scriptsize{$^*$Ours.}")
    out.append(r"\end{table}")
    return TableSpec("appendix_coverage", "\n".join(out) + "\n")


MDS_SWEEP_DIMS = [3, 5, 10, 20, 40, 60]
MDS_SWEEP_ROOT = REPO / "outputs/analysis/mds_sweep"
DM_VARIANTS = [
    "w2_dmc_int_kmeans", "w2_dmc_qpos_kmeans", "w2_dmc_ekp_kmeans",
    "w2_dmp_int_kmeans", "w2_dmp_qpos_kmeans", "w2_dmp_ekp_kmeans",
]


def _mds_series(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Seed-averaged per-transition values, indexed by (Label, Turn)."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return pd.Series(dtype=float)
    s = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    s.index = df[["Label", "Turn"]].astype(str).agg("|".join, axis=1)
    return s[~s.index.duplicated()]


def _mds_metric_series(dim: int, variant: str, metric: str) -> pd.Series:
    """Seed-averaged per-transition MMD or KID from the sweep's per-seed CSVs."""
    frames = []
    for seed in (0, 1, 2):
        f = (MDS_SWEEP_ROOT / f"mds{dim}" / variant / f"seed_{seed}" / "mmd"
             / f"{variant}_seed{seed}_mmd_results.csv")
        if f.exists():
            d = pd.read_csv(f)
            d.index = d["Label"].astype(str) + "|" + d["Turn_Direction"].astype(str)
            frames.append(pd.to_numeric(d[metric], errors="coerce"))
    if not frames:
        return pd.Series(dtype=float)
    out = pd.concat(frames, axis=1).mean(axis=1)
    return out[~out.index.duplicated()]


def mds_sweep_grid() -> tuple[dict, dict, dict, dict, dict, list[int]]:
    """Return (consistency, dtw, mmd, kid, coverage) grids keyed [variant][dim].

    Means are taken over the row set a variant populates at EVERY dimension, so a
    setting cannot look better merely by discovering a different set of transitions.
    """
    dims = [d for d in MDS_SWEEP_DIMS
            if (MDS_SWEEP_ROOT / f"mds{d}" / "dtw_comparison_table.csv").exists()]
    cons, dtw, mmd, kid, cov = {}, {}, {}, {}, {}
    for v in DM_VARIANTS:
        sc, sd, sm, sk, nc = {}, {}, {}, {}, {}
        for d in dims:
            base = MDS_SWEEP_ROOT / f"mds{d}"
            sd[d] = _mds_series(pd.read_csv(base / "dtw_comparison_table.csv"), v + "_seed")
            sc[d] = _mds_series(pd.read_csv(base / "consistency_comparison_table.csv"), v + "_seed")
            sm[d] = _mds_metric_series(d, v, "MMD_RBF")
            sk[d] = _mds_metric_series(d, v, "KID")
            nc[d] = float(sd[d].notna().sum())

        def fixed_mean(store):
            common = set.intersection(*[set(x.dropna().index) for x in store.values()]) if store else set()
            common = sorted(common)
            return {d: (store[d].loc[common].mean() if common else float("nan")) for d in dims}

        dtw[v], cons[v] = fixed_mean(sd), fixed_mean(sc)
        mmd[v], kid[v] = fixed_mean(sm), fixed_mean(sk)
        cov[v] = nc
    return cons, dtw, mmd, kid, cov, dims


def select_mds_dims() -> dict[str, int]:
    """Per-variant MDS dimension, chosen by minimum within-cluster consistency.

    Consistency is computed only from the discovered clusters and never references
    the ground-truth labels, so this selection is independent of the alignment
    metrics the paper reports.
    """
    cons, _, _, _, _, dims = mds_sweep_grid()
    return {v: min(dims, key=lambda d: cons[v][d]) for v in DM_VARIANTS}


def appendix_mds_sweep() -> TableSpec:
    cons, dtw, mmd, kid, cov, dims = mds_sweep_grid()
    sel = {v: min(dims, key=lambda d: cons[v][d]) for v in DM_VARIANTS}
    disp = {v: METHOD_DISPLAY[v].replace("$^*$", "") for v in DM_VARIANTS}

    def block(grid, fmt, higher_better=False):
        """One block; the best cell in each row is bolded (ties all bolded)."""
        out = []
        for v in DM_VARIANTS:
            vals = {d: grid[v][d] for d in dims if not pd.isna(grid[v][d])}
            best = (max if higher_better else min)(vals.values()) if vals else None
            cells = []
            for d in dims:
                x = grid[v][d]
                if pd.isna(x):
                    cells.append("--")
                    continue
                t = format(x, fmt)
                cells.append(r"\textbf{" + t + "}" if x == best else t)
            out.append(f"{disp[v]} & " + " & ".join(cells) + r" \\")
        return "\n".join(out)

    head = ("Variant & " + " & ".join(str(d) for d in dims) + r" \\")
    rule = r"\cmidrule(lr){2-%d}" % (1 + len(dims))
    out = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{\textbf{Selection of the MDS embedding dimension.} Sweep over "
        r"$\{3,5,10,20,40,60\}$ at three seeds. The dimension is selected per variant by "
        r"minimising within-cluster consistency (block a), which uses no ground-truth labels; "
        r"blocks (b)--(d) take no part in the choice. Best per row in \textbf{bold}. Means are "
        r"over the transitions a variant populates at every dimension. DMP-Int shares only two "
        r"such transitions, so its block (c) row is noisy.}",
        r"\label{tab:mds-sweep}", r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{0.85}",
        r"\begin{tabular}{l" + "c" * len(dims) + "}", r"\toprule",
        r"& \multicolumn{%d}{c}{MDS embedding dimension} \\" % len(dims), rule, head, r"\midrule",
        r"\multicolumn{%d}{l}{\textit{(a) Within-cluster consistency $\downarrow$ (selection criterion)}} \\" % (1+len(dims)),
        block(cons, ".3f"), r"\midrule",
        r"\multicolumn{%d}{l}{\textit{(b) DTW to ground truth $\downarrow$}} \\" % (1+len(dims)),
        block(dtw, ".3f"), r"\midrule",
        r"\multicolumn{%d}{l}{\textit{(c) KID $\downarrow$}} \\" % (1+len(dims)),
        block(kid, ".3f"), r"\midrule",
        r"\multicolumn{%d}{l}{\textit{(d) Transitions discovered (union over 3 seeds)}} \\" % (1+len(dims)),
        block(cov, ".1f", higher_better=True),
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    return TableSpec("appendix_mds_sweep", "\n".join(out) + "\n")


def appendix_family_breakdown() -> TableSpec:
    """Distributional fidelity (MMD, KID) split by the behaviour family of the
    transition, under the fixed per-family variants used in Tables 1-2."""
    def load(f):
        d = pd.read_csv(FA / "tables" / f, index_col=0)
        # Fixed variants, matching Tables 1-2. Taking a per-row min over every DM
        # column is a within-family oracle and it manufactured the "rearing-only"
        # story this table used to tell.
        dmc = next(c for c in d.columns if "DMC" in c and "Int" in c)
        kpc = next(c for c in d.columns if "KPMS-KP" in c)
        A = pd.to_numeric(d[dmc], errors="coerce")
        B = pd.to_numeric(d[kpc], errors="coerce")
        m = A.notna() & B.notna()
        return A[m], B[m]

    def family(name: str) -> str:
        if "Rear" in name: return "Rear"
        if "Walk" in name: return "Walk"
        return "Turn"

    mmd_a, mmd_b = load("mmd_rbf_table.csv")
    kid_a, kid_b = load("kid_table.csv")
    fams = ["Rear", "Turn", "Walk"]
    rows = []
    for fam in fams + ["All"]:
        sel = (lambda i: True) if fam == "All" else (lambda i, f=fam: family(i) == f)
        idx = [i for i in mmd_a.index if sel(i)]
        kidx = [i for i in kid_a.index if sel(i)]
        if not idx:
            continue
        ma, mb = mmd_a.loc[idx].mean(), mmd_b.loc[idx].mean()
        ka, kb = kid_a.loc[kidx].mean(), kid_b.loc[kidx].mean()
        wins = int((mmd_a.loc[idx] < mmd_b.loc[idx]).sum())
        label = r"\textit{All}" if fam == "All" else fam
        rows.append(rf"{label} & {len(idx)} & {ma:.3f} & {mb:.3f} & {ma-mb:+.3f} & "
                    rf"{ka:.4f} & {kb:.4f} & {ka-kb:+.4f} & {wins}/{len(idx)} \\")

    out = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{\textbf{Distributional fidelity by transition family.} MMD and KID "
        r"(both $\downarrow$) for DMC-Int and KPMS-KP, the same fixed variants used in "
        r"Tables~\ref{tab:alignment-rat} and~\ref{tab:warping-rat}, averaged within the "
        r"behaviour family the transition involves. Only rows both variants discover are "
        r"included. The MMD difference is small and does not single out any one behaviour "
        r"family, and on KID the two are effectively tied. Few rows survive per family, so "
        r"the split is indicative rather than conclusive. \emph{DM wins} counts MMD rows.}",
        r"\label{tab:family-breakdown}", r"\small",
        r"\begin{tabular}{lcccccccc}", r"\toprule",
        r"& & \multicolumn{3}{c}{MMD} & \multicolumn{3}{c}{KID} & \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Family & Rows & DM & KPMS & $\Delta$ & DM & KPMS & $\Delta$ & DM wins \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    return TableSpec("appendix_family_breakdown", "\n".join(out) + "\n")


def appendix_scale_sweep() -> TableSpec:
    """Baseline scale sweep: count of canonical transitions discovered per variant.

    Compact layout requested 2026-08: feature space is a grouping row, the swept
    setting is nested beneath it, and columns are the second swept axis. Each cell
    gives three counts at nested label resolutions, unordered / directed /
    direction-turn, all scored under the same rule so they are mutually coherent.
    """
    df = pd.read_csv(SS / "discovery_aligned_binary_wide.csv").set_index("transition")
    # Restrict to exactly the classes Table 1 prints, so the two tables share a
    # denominator. Table 1's row set is survivor-defined (a class appears only if
    # some benchmark method found it), so this is a subset of the GT classes.
    _tag = {"left": "L", "right": "R", "straight": "S"}
    _t1 = [f"{lab} [{_tag[turn]}]"
           for lab, turn in _sorted_directional_rows(long_to_wide(load_dtw_long()))]
    df = df.reindex([c for c in _t1 if c in df.index]).fillna(0).astype(int)
    n_total = len(df)

    def count(col: str) -> int:
        return int(df[col].sum()) if col in df.columns else -1

    def cell(col: str) -> str:
        return "--" if col not in df.columns else str(int(df[col].sum()))

    # --- (a) KPMS: rows = kappa (grouped by modality), cols = num_states -----
    kpms_cols = [c for c in df.columns if c.startswith("kpms")]
    kp_pat = re.compile(r"kpms_(?:(kp|int)_)?k1e(\d+)_n(\d+)")
    kpms_cells: dict[tuple[str, str], dict[str, int]] = {}
    kappas, states = set(), set()
    for c in kpms_cols:
        m = kp_pat.fullmatch(c)
        if not m:
            continue
        mod = {"kp": "Keypoint", "int": "Intention", None: "Keypoint"}[m.group(1)]
        ke, ns = m.group(2), m.group(3)
        kpms_cells.setdefault((mod, ke), {})[ns] = cell(c)
        kappas.add(ke); states.add(ns)
    states_order = sorted(states, key=int)
    kappa_order = sorted(kappas, key=int)

    a = [r"\begin{tabular}{ll" + "c" * len(states_order) + "}", r"\toprule",
         r"& & \multicolumn{%d}{c}{$N_{\text{states}}$} \\" % len(states_order),
         r"\cmidrule(lr){3-%d}" % (2 + len(states_order)),
         "Feature space & $\\kappa$ & " + " & ".join(states_order) + r" \\", r"\midrule"]
    for mod in ("Intention", "Keypoint"):
        present = [k for k in kappa_order if (mod, k) in kpms_cells]
        if not present:
            continue
        a.append(rf"\multicolumn{{{2+len(states_order)}}}{{l}}{{\textit{{{mod}}}}} \\")
        for k in present:
            vals = kpms_cells[(mod, k)]
            cells = [vals.get(n, "--") for n in states_order]
            a.append(rf"\quad & $10^{{{k}}}$ & " + " & ".join(cells) + r" \\")
    a += [r"\bottomrule", r"\end{tabular}"]
    kpms_block = "\n".join(a)

    # --- (b) Mapper: rows = lens (grouped by feature space), cols = n_intervals ---
    mp_pat = re.compile(r"mapper_(intention|qpos|ekp)_(.+)_n(\d+)")
    mp_cells: dict[tuple[str, str], dict[str, int]] = {}
    ns_set = set()
    for c in [c for c in df.columns if c.startswith("mapper")]:
        m = mp_pat.fullmatch(c)
        if not m:
            continue
        mp_cells.setdefault((m.group(1), m.group(2)), {})[m.group(3)] = cell(c)
        ns_set.add(m.group(3))
    n_order = sorted(ns_set, key=int)
    space_map = {"intention": "Intention", "qpos": "Qpos", "ekp": "EKP"}
    lens_map = {"fiedler": "Fiedler", "knn_density": "kNN-density",
                "pca_1d": "PCA-1D", "eccentricity": "Eccentricity"}

    b = [r"\begin{tabular}{ll" + "c" * len(n_order) + "}", r"\toprule",
         r"& & \multicolumn{%d}{c}{$n_{\text{intervals}}$} \\" % len(n_order),
         r"\cmidrule(lr){3-%d}" % (2 + len(n_order)),
         "Feature space & Lens & " + " & ".join(n_order) + r" \\", r"\midrule"]
    for sp in ("intention", "qpos", "ekp"):
        lenses = [l for l in ("fiedler", "knn_density", "pca_1d", "eccentricity") if (sp, l) in mp_cells]
        if not lenses:
            continue
        b.append(rf"\multicolumn{{{2+len(n_order)}}}{{l}}{{\textit{{{space_map[sp]}}}}} \\")
        for l in lenses:
            vals = mp_cells[(sp, l)]
            cells = [vals.get(n, "--") for n in n_order]
            b.append(rf"\quad & {lens_map[l]} & " + " & ".join(cells) + r" \\")
    b += [r"\bottomrule", r"\end{tabular}"]
    mapper_block = "\n".join(b)

    all_kpms = [c for c in kpms_cols if kp_pat.fullmatch(c)]
    all_mapper = [c for c in df.columns if mp_pat.fullmatch(c)]

    def union(cols: list[str]) -> int:
        return int((df[cols].sum(axis=1) > 0).sum())

    u_k, u_m = union(all_kpms), union(all_mapper)
    u_c = union(all_kpms + all_mapper)

    out = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{\textbf{Baseline scale-hyperparameter sweep.} Each cell is the number of "
        r"the %d directed, direction-turn transition classes of Table~\ref{tab:coverage} that the "
        r"variant discovers, over the same rows and the same evaluation split, so the two tables "
        r"share a denominator. "
        r"\textbf{(a)} Keypoint-MoSeq over stickiness $\kappa$ and $N_{\text{states}}$ on both "
        r"feature streams (50 fits). \textbf{(b)} Mapper over lens and $n_{\text{intervals}}$ on "
        r"all three feature spaces (60 variants). Primitives are built exactly as for the benchmark "
        r"baselines, from sustained syllable or node pairs, and each cluster is majority-voted to a "
        r"class by the same labeller used for Table~\ref{tab:coverage}. Every fit is scored with no "
        r"model-selection criterion applied first, so the union line is a true oracle union over all "
        r"110 variants. Remaining settings are held at a common default rather than at the "
        r"per-stream tuned values of Appendix~\ref{app:baseline-hp}, and no swept setting improves "
        r"on the tuned configurations reported in Table~\ref{tab:coverage}.}"
        % n_total,
        r"\label{tab:scale-sweep}", "",
        # Side by side. The size command is scoped to each tabular with braces so it
        # cannot leak past the sub-caption, which previously rendered (a) and (b) at
        # different sizes.
        r"\begin{minipage}[t]{0.46\textwidth}", r"\centering",
        r"\textit{(a) Keypoint-MoSeq.}\\[2pt]",
        r"{\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        kpms_block, r"}", r"\end{minipage}\hfill",
        r"\begin{minipage}[t]{0.52\textwidth}", r"\centering",
        r"\textit{(b) Mapper.}\\[2pt]",
        r"{\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        mapper_block, r"}", r"\end{minipage}", "", r"\vspace{4pt}", "",
        rf"\scriptsize{{Oracle union over all {len(all_kpms)} KPMS variants: {u_k} of {n_total}. "
        rf"Over all {len(all_mapper)} Mapper variants: {u_m} of {n_total}. "
        rf"Combined baseline union: {u_c} of {n_total}.}}",
        r"\end{table}",
    ]
    return TableSpec("appendix_scale_sweep", "\n".join(out) + "\n")

# ─────────────────────────────────────────────────────────────────────────────

def appendix_scale_sweep_directed() -> TableSpec:
    """Sweep coverage at Table 1's directed + direction-turn resolution.

    Columns are per-family ORACLE UNIONS over every swept variant, so a check
    means "some setting in this family reached the class", not that any single
    configuration did. The scoring rule is deliberately more permissive than
    Table~\\ref{tab:coverage}'s majority vote, so these are upper bounds and must
    not be read head-to-head against it (see the caption).
    """
    d = pd.read_csv(SS / "discovery_aligned_binary_wide.csv").set_index("transition")
    gt = pd.read_csv(SS / "gt_directed_turn_counts.csv", index_col=0).iloc[:, 0]
    fams = [("KPMS-Int", "kpms_int"), ("KPMS-KP", "kpms_kp"),
            ("Mapper-Int", "mapper_intention"), ("Mapper-Qpos", "mapper_qpos"),
            ("Mapper-EKP", "mapper_ekp")]
    U = {name: (d[[c for c in d.columns if c.startswith(pre)]].max(axis=1) == 1)
         for name, pre in fams}
    U["Any"] = (d.max(axis=1) == 1)
    order = sorted(d.index, key=lambda t: -int(gt.get(t, 0)))

    def fmt(t: str) -> str:
        a, rest = t.split(" -> ")
        b, tag = rest.rsplit(" [", 1)
        return rf"{a}$\rightarrow${b} [{tag[:-1]}]"

    cols = [n for n, _ in fams] + ["Any"]
    body = []
    for t in order:
        cells = [(r"\checkmark" if U[c][t] else r"$\times$") for c in cols]
        body.append(f"{fmt(t)} & {int(gt.get(t,0))} & " + " & ".join(cells) + r" \\")
    tot = ["--"] + [str(int(U[c].sum())) for c in cols]

    out = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{\textbf{Baseline sweep coverage at directed, direction-turn "
        r"resolution.} Each column is the \emph{oracle union} over every swept variant "
        r"in that family, so $\checkmark$ means some setting reached the class, not that "
        r"any one configuration did. Rows are the %d directed, direction-turn classes "
        r"present in the evaluation split, ordered by ground-truth instance count (GT $n$); "
        r"direction tags [L]/[R]/[S] as in Table~\ref{tab:coverage}. A class counts as "
        r"reached if two temporally adjacent pure-state clusters straddle a ground-truth "
        r"transition of that class in at least three clips. This rule is deliberately more "
        r"permissive than the majority vote of Table~\ref{tab:coverage}: the same Mapper "
        r"configuration scores 6 here and 4 there. These columns are therefore upper bounds "
        r"on baseline coverage and are not comparable cell-for-cell with "
        r"Table~\ref{tab:coverage}.}" % len(order),
        r"\label{tab:scale-sweep-directed}", r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{0.9}",
        r"\begin{tabular}{lr" + "c" * len(cols) + "}", r"\toprule",
        r"Transition & GT $n$ & " + " & ".join(cols) + r" \\", r"\midrule",
        "\n".join(body), r"\midrule",
        r"\textit{Classes reached} & " + " & ".join(tot) + r" \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    return TableSpec("appendix_scale_sweep_directed", "\n".join(out) + "\n")

# ─────────────────────────────────────────────────────────────────────────────

def appendix_k_sweep() -> TableSpec:
    """appendix_k_sweep.tex — DM kNN-$k$ sensitivity. Per (space, $k$): label-free
    structural DM-cycles per clip + giant-CC, and DM-cycle DTW/MMD/KID/coverage, read from
    the k-sweep CSVs (outputs/analysis/method_analysis/k_sweep/)."""
    struct = pd.read_csv(KS / "structural_stats.csv")
    down = pd.read_csv(KS / "downstream_metrics_vs_k.csv")
    down = down[down["primitive"] == "cycle"]
    spaces = [("intention", "Intention"), ("qpos", "Qpos"), ("ekp", "EKP")]
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Space & $k$ & cycles/clip & giant-CC & DTW & KID & cov \\",
             r"\midrule"]
    for si, (sp, disp) in enumerate(spaces):
        for ki, k in enumerate([3, 5, 8]):
            s = struct[(struct.space == sp) & (struct.k == k)]
            d = down[(down.space == sp) & (down.k == k)]
            if s.empty or d.empty:
                continue
            s, d = s.iloc[0], d.iloc[0]
            kc = rf"\textbf{{{k}}}" if k == 3 else f"{k}"
            cell = disp if ki == 0 else ""
            lines.append(
                f"{cell} & {kc} & {s.n_cycles_mean:.2f} & {s.giant_cc_frac_mean:.2f} & "
                f"{d.dtw_mean:.2f} & {d.kid_mean:.3f} & "
                f"{d.coverage_mean:.1f} \\\\")
        if si < len(spaces) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return TableSpec("appendix_k_sweep", "\n".join(lines) + "\n")


def _pp_table(df, value_cols: list[tuple[str, str, str]], header: str) -> str:
    """Path-persistence table body grouped by space, one row per delta multiplier
    (production mult=0 in bold). value_cols = [(csv_col, fmt, _)]."""
    spaces = [("intention", "Intention"), ("qpos", "Qpos"), ("ekp", "EKP")]
    ncol = "c" * (len(value_cols))
    lines = [rf"\begin{{tabular}}{{ll{ncol}}}", r"\toprule", header, r"\midrule"]
    for si, (sp, disp) in enumerate(spaces):
        sub = df[df.space == sp].sort_values("mult")
        for ki, (_, r) in enumerate(sub.iterrows()):
            m = r["mult"]
            mc = rf"\textbf{{{m:g}}}" if m == 0 else f"{m:g}"
            cell = disp if ki == 0 else ""
            vals = " & ".join(fmt.format(r[c]) for c, fmt, _ in value_cols)
            lines.append(f"{cell} & {mc} & {vals} \\\\")
        if si < len(spaces) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def appendix_path_persistence_ami() -> TableSpec:
    """appendix_path_persistence_ami.tex — transition information of the DMP
    coarse-graining vs the path-selection persistence multiplier."""
    df = pd.read_csv(PPS / "ami_vs_delta.csv")
    header = r"Space & $\delta_{\mathrm{path}}/\delta$ & paths & $I(T;Y)$ & AMI & $H(T)$ \\"
    text = _pp_table(df, [("n_paths", "{:.0f}", ""), ("I_TY", "{:.3f}", ""),
                          ("AMI", "{:.3f}", ""), ("H_T", "{:.2f}", "")], header)
    return TableSpec("appendix_path_persistence_ami", text)


def appendix_path_persistence_down() -> TableSpec:
    """appendix_path_persistence_down.tex — downstream DMP alignment vs the
    path-selection persistence multiplier (mult=0 = production DMP)."""
    df = pd.read_csv(PPS / "downstream_vs_delta.csv")
    header = r"Space & $\delta_{\mathrm{path}}/\delta$ & DTW & KID & cov \\"
    text = _pp_table(df, [("dtw", "{:.3f}", ""),
                          ("kid", "{:.3f}", ""), ("cov", "{:.1f}", "")], header)
    return TableSpec("appendix_path_persistence_down", text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="dry-run; print but don't write")
    args = p.parse_args()

    specs: list[TableSpec] = [
        main_combined_table(),

        appendix_dtw_detailed(),
        appendix_consistency(),
        appendix_kid(),
        appendix_dtw_sync_ratio(),
        appendix_non_diag(),
        appendix_coverage(),
        appendix_scale_sweep(),
        appendix_mds_sweep(),
        appendix_k_sweep(),
        appendix_path_persistence_ami(),
        appendix_path_persistence_down(),
    ]
    write_tables(specs, dry=args.check)


if __name__ == "__main__":
    main()
