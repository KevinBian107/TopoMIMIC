#!/usr/bin/env python3
"""Cycle transition-signature analysis.

For every DM-cycle in the per-clip intention extraction
(`outputs/extraction/dm_pipeline_intention/all_cycle_dm.pkl`), build a
signature describing which behavior-pair transitions the cycle traces:

    behavior_frac[b]              = fraction of cycle frames labelled b
    transition_edge_count[(a,b)]  = #(adjacent label changes a<->b in cycle order)

Cycles are then embedded in 2D (PCA on transition-edge histogram, normalised)
and coloured by their dominant unordered transition pair. Output:

    outputs/analysis/method_analysis/cycle_transition_signature/
        cycle_signatures.csv
        cycle_signatures.pkl
        scatter_pca.pdf
        heatmap.pdf
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[4]

H5_PATH = REPO / "data/rodent_ar.h5"
LABELS_CSV = REPO / "behavior_labels/rodent_labels.csv"
CYCLES_PKL = REPO / "outputs/extraction/dm_pipeline_intention/all_cycle_dm.pkl"
PATHS_PKL = REPO / "outputs/extraction/dm_pipeline_intention/all_path_dm.pkl"
OUT_DIR = REPO / "outputs/analysis/method_analysis/cycle_transition_signature"

# Mirror intention-space Takens config (configs/extraction/dm_pipeline_intention.yaml).
TAKENS_M = 3
TAKENS_TAU = 6
TAKENS_OFFSET = (TAKENS_M - 1) * TAKENS_TAU

BEHAVIORS = ["Walk", "Immobile", "Turn", "Rear"]
BEHAVIOR_COLORS = {
    "Walk":     "#4C72B0",
    "Immobile": "#C0C0C0",
    "Turn":     "#55A868",
    "Rear":     "#DD8452",
}

UNORDERED_PAIRS = [
    ("Immobile", "Walk"),
    ("Immobile", "Turn"),
    ("Immobile", "Rear"),
    ("Walk", "Turn"),
    ("Walk", "Rear"),
    ("Turn", "Rear"),
]
PAIR_COLORS = {
    ("Immobile", "Walk"): "#3B7AB8",
    ("Immobile", "Turn"): "#3FA76A",
    ("Immobile", "Rear"): "#D77443",
    ("Walk", "Turn"):     "#7E6BC4",
    ("Walk", "Rear"):     "#B05A8E",
    ("Turn", "Rear"):     "#C9A23A",
}


@dataclass
class CycleSignature:
    clip_id: int
    feature_idx_in_clip: int
    feature_kind: str  # "cycle" or "path"
    n_frames_in_feature: int
    behavior_fraction: dict[str, float]
    transition_edge_count: dict[tuple[str, str], int]
    dominant_pair: tuple[str, str] | None


def per_frame_labels(labels_df: pd.DataFrame, clip_id: int, n_frames: int) -> np.ndarray:
    sub = labels_df[(labels_df["Video"] == f"clip_{clip_id}.mp4") &
                    (labels_df["Track"] == "behavior")]
    out = np.array(["Immobile"] * n_frames, dtype=object)
    for _, r in sub.iterrows():
        s, e = int(r["Start_Frame"]), int(r["End_Frame"])
        out[s:min(e + 1, n_frames)] = r["Label"]
    return out


def feature_to_label_seq(vertices: np.ndarray, label_arr: np.ndarray) -> np.ndarray | None:
    """Vertices index Takens points; original frame = vertex + TAKENS_OFFSET."""
    frames = np.asarray(vertices, dtype=int) + TAKENS_OFFSET
    valid = frames[(frames >= 0) & (frames < label_arr.shape[0])]
    if valid.size < 2:
        return None
    return label_arr[valid]


def signature_from_label_seq(seq: np.ndarray, kind: str) -> tuple[dict[str, float], dict[tuple[str, str], int]]:
    """For cycles, wrap around (i -> (i+1) % n); for paths, walk linearly."""
    behavior_frac = {b: float(np.sum(seq == b)) / seq.shape[0] for b in BEHAVIORS}
    edge_count = {p: 0 for p in UNORDERED_PAIRS}
    n = seq.shape[0]
    end = n if kind == "cycle" else n - 1
    for i in range(end):
        a, b = seq[i], seq[(i + 1) % n] if kind == "cycle" else seq[i + 1]
        if a == b:
            continue
        pair = tuple(sorted((a, b)))
        if pair in edge_count:
            edge_count[pair] += 1
    return behavior_frac, edge_count


def collect_signatures() -> list[CycleSignature]:
    print(f"Loading cycles from {CYCLES_PKL}...")
    with open(CYCLES_PKL, "rb") as f:
        all_cycles = pickle.load(f)
    print(f"Loading paths from {PATHS_PKL}...")
    with open(PATHS_PKL, "rb") as f:
        all_paths = pickle.load(f)
    n_clips_with_cycles = sum(1 for cs in all_cycles if cs)
    n_total_cycles = sum(len(cs) for cs in all_cycles if cs)
    n_clips_with_paths = sum(1 for ps in all_paths if ps)
    n_total_paths = sum(len(ps) for ps in all_paths if ps)
    print(f"  cycles: {n_total_cycles} across {n_clips_with_cycles} clips")
    print(f"  paths : {n_total_paths} across {n_clips_with_paths} clips")

    labels_df = pd.read_csv(LABELS_CSV)
    with h5py.File(H5_PATH, "r") as h:
        n_frames = h["qpos_rollout"].shape[1]

    sigs: list[CycleSignature] = []
    n_clips = max(len(all_cycles), len(all_paths))
    for clip_id in range(n_clips):
        clip_cycles = all_cycles[clip_id] if clip_id < len(all_cycles) else []
        clip_paths = all_paths[clip_id] if clip_id < len(all_paths) else []
        if not clip_cycles and not clip_paths:
            continue
        label_arr = per_frame_labels(labels_df, clip_id, n_frames)
        for kind, features in (("cycle", clip_cycles), ("path", clip_paths)):
            for fi, feature in enumerate(features):
                seq = feature_to_label_seq(feature, label_arr)
                if seq is None:
                    continue
                beh_frac, edge_count = signature_from_label_seq(seq, kind)
                dominant = max(edge_count.items(), key=lambda kv: kv[1])
                dominant_pair = dominant[0] if dominant[1] > 0 else None
                sigs.append(CycleSignature(
                    clip_id=clip_id,
                    feature_idx_in_clip=fi,
                    feature_kind=kind,
                    n_frames_in_feature=int(seq.shape[0]),
                    behavior_fraction=beh_frac,
                    transition_edge_count=edge_count,
                    dominant_pair=dominant_pair,
                ))
    print(f"  built {len(sigs)} signatures (cycles + paths)")
    return sigs


def signatures_to_dataframe(sigs: list[CycleSignature]) -> pd.DataFrame:
    rows = []
    for s in sigs:
        row = {
            "clip_id": s.clip_id,
            "feature_idx_in_clip": s.feature_idx_in_clip,
            "feature_kind": s.feature_kind,
            "n_frames_in_feature": s.n_frames_in_feature,
        }
        for b in BEHAVIORS:
            row[f"frac_{b}"] = s.behavior_fraction[b]
        for p in UNORDERED_PAIRS:
            row[f"edge_{p[0]}_{p[1]}"] = s.transition_edge_count[p]
        row["dominant_pair"] = (
            f"{s.dominant_pair[0]}_{s.dominant_pair[1]}" if s.dominant_pair else "none"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def render_pca_scatter(df: pd.DataFrame, out_path: Path) -> None:
    edge_cols = [f"edge_{p[0]}_{p[1]}" for p in UNORDERED_PAIRS]
    X = df[edge_cols].to_numpy(dtype=float)
    rowsum = X.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    Xn = X / rowsum

    keep = X.sum(axis=1) > 0
    Xn = Xn[keep]
    sub = df[keep].reset_index(drop=True)
    if Xn.shape[0] < 2:
        print("  not enough cycles with transitions for PCA")
        return

    pca = PCA(n_components=2)
    emb = pca.fit_transform(Xn)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for pair in UNORDERED_PAIRS:
        key = f"{pair[0]}_{pair[1]}"
        sel = sub["dominant_pair"] == key
        if not sel.any():
            continue
        ax.scatter(emb[sel, 0], emb[sel, 1], s=14,
                   c=PAIR_COLORS[pair], alpha=0.75, edgecolors="none",
                   label=f"{pair[0][:3]}↔{pair[1][:3]}")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=8)
    ax.tick_params(direction="out", length=2.5, width=0.5, labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6, frameon=False, loc="best", ncol=2,
              handletextpad=0.3, columnspacing=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def render_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    edge_cols = [f"edge_{p[0]}_{p[1]}" for p in UNORDERED_PAIRS]
    X = df[edge_cols].to_numpy(dtype=float)
    rowsum = X.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    Xn = X / rowsum

    grouped = []
    group_labels = []
    for pair in UNORDERED_PAIRS:
        key = f"{pair[0]}_{pair[1]}"
        sel = df["dominant_pair"] == key
        if not sel.any():
            continue
        block = Xn[sel.to_numpy()]
        order = np.argsort(-block[:, UNORDERED_PAIRS.index(pair)])
        grouped.append(block[order])
        group_labels.append((key, block.shape[0]))
    if not grouped:
        print("  no grouped cycles for heatmap")
        return
    M = np.concatenate(grouped, axis=0)

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    im = ax.imshow(M, aspect="auto", cmap="magma_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(UNORDERED_PAIRS)))
    ax.set_xticklabels([f"{p[0][:3]}↔{p[1][:3]}" for p in UNORDERED_PAIRS],
                       rotation=45, ha="right", fontsize=7)
    ax.set_yticks([])
    ax.set_ylabel(f"cycles ({M.shape[0]}), grouped by dominant pair", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("transition-edge fraction", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cum = 0
    for _, n in group_labels[:-1]:
        cum += n
        ax.axhline(cum - 0.5, color="white", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sigs = collect_signatures()
    df = signatures_to_dataframe(sigs)

    csv_path = OUT_DIR / "cycle_signatures.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    pkl_path = OUT_DIR / "cycle_signatures.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(sigs, f)
    print(f"wrote {pkl_path}")

    render_pca_scatter(df, OUT_DIR / "scatter_pca.pdf")
    render_heatmap(df, OUT_DIR / "heatmap.pdf")


if __name__ == "__main__":
    main()
