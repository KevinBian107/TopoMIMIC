"""Cluster evaluation against the transition classifier.

For each discovered cluster with auto-label L in TOP_8, this module:

1. Pulls the member windows (clip_id, frames) from `sample_labels_seed_{N}.yaml`.
2. Extracts qpos for those frame ranges, preprocesses (handle_rotation +
   exclude_xy), resamples each to K frames.
3. Runs the classifier ensemble to get per-window softmax over the 8 classes.
4. Compares to real GT L-transition windows (also resampled to K):
   - match_rate       = fraction of cluster windows with argmax == L
   - mean_p_L         = mean classifier confidence on cluster windows for L
   - mean_p_L_real    = same, computed on real L-transitions
   - p_L_gap          = mean_p_L_real - mean_p_L (smaller is better — cluster blends in)
   - rank_fraction    = fraction of cluster windows whose p_L lands at or above
                        the 5th percentile of real L-transitions' p_L
                        (retrievability floor)
   - confusion_row    = mean 8-dim softmax vector across cluster windows
5. Writes per-cluster metrics CSV + a heatmap of all clusters vs the 8 classes.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from topo_mimic.benchmark.auto_label import (
    load_behavior_segments,
    split_segments_by_kind,
)
from topo_mimic.benchmark.classifier import LABEL_TO_IDX, TOP_8_TRANSITIONS
from topo_mimic.benchmark.classifier.data import build_transition_instances
from topo_mimic.benchmark.classifier.labels import (
    extract_instance_window,
    find_directional_transitions,
    resample_to_K,
)
from topo_mimic.benchmark.classifier.predict import LoadedClassifier
from topo_mimic.benchmark.classifier.preprocessing import preprocess_qpos


def _plot_base_vs_cluster(
    labels: list[str],
    base: dict[str, float],
    cluster_mean: dict[str, float],
    output_png: str,
) -> None:
    """Side-by-side bars per label: classifier recall on real L (ceiling)
    vs mean match_rate across clusters labeled L. A cluster bar close to
    the base bar means cluster members score as well as real GT windows;
    a much lower bar means cluster windows are less-recognizable as L."""
    base_arr = np.array([base.get(l, np.nan) for l in labels], dtype=np.float64)
    cm_arr = np.array([cluster_mean.get(l, np.nan) for l in labels], dtype=np.float64)
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(labels) + 2.0), 3.6))
    ax.bar(x - w / 2, base_arr, width=w, label="base (real L recall)",
           color="#4878CF", edgecolor="none")
    ax.bar(x + w / 2, cm_arr, width=w, label="cluster mean match_rate",
           color="#C44E52", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("fraction of windows argmax == L")
    ax.set_ylim(0, 1.05)
    ax.set_title("Base classification rate vs cluster match rate")
    ax.legend(fontsize=8, frameon=True, edgecolor="#cccccc")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, (b, c) in enumerate(zip(base_arr, cm_arr)):
        if np.isfinite(b):
            ax.text(i - w / 2, b + 0.02, f"{b:.2f}", ha="center", va="bottom", fontsize=7)
        if np.isfinite(c):
            ax.text(i + w / 2, c + 0.02, f"{c:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _strip_direction_from_label(label: str) -> str:
    """Cluster labels are 'A -> B Direction'. We only care about the 'A -> B' part."""
    tokens = label.strip().split()
    if "->" not in tokens:
        return label.strip()
    arrow_i = tokens.index("->")
    if arrow_i < 1 or arrow_i + 1 >= len(tokens):
        return label.strip()
    return f"{tokens[arrow_i - 1]} -> {tokens[arrow_i + 1]}"


def _load_cluster_members(method_dir: str, seed: int) -> dict[int, tuple[str, list[tuple[int, np.ndarray]]]]:
    """Return {cluster_id: (primary_label, [(clip_id, frames), ...])}.

    `primary_label` is read from `labels_seed_{seed}.yaml`; members from
    `sample_labels_seed_{seed}.yaml`.
    """
    with open(os.path.join(method_dir, f"labels_seed_{seed}.yaml")) as f:
        labels_blob = yaml.safe_load(f)
    with open(os.path.join(method_dir, f"sample_labels_seed_{seed}.yaml")) as f:
        samples_blob = yaml.safe_load(f)

    labels_map = labels_blob.get("labels", {}) or {}
    cluster_members: dict[int, tuple[str, list[tuple[int, np.ndarray]]]] = {}
    for cid_str, meta in (samples_blob.get("clusters", {}) or {}).items():
        cid = int(cid_str)
        members: list[tuple[int, np.ndarray]] = []
        for s in meta.get("samples", []):
            frames = np.arange(int(s["frame_start"]), int(s["frame_end"]) + 1, dtype=np.int64)
            members.append((int(s["clip_id"]), frames))
        primary_label = labels_map.get(cid) or labels_map.get(str(cid)) or ""
        cluster_members[cid] = (primary_label or "", members)
    return cluster_members


def _member_windows(
    qpos_preproc: np.ndarray,
    members: list[tuple[int, np.ndarray]],
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Each member -> ONE K-frame window (resampled) + the member's clip_id.

    Returns (X: (N, K, D), clip_ids: (N,) int64).
    """
    if not members:
        return (
            np.zeros((0, K, qpos_preproc.shape[-1]), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    out = np.empty((len(members), K, qpos_preproc.shape[-1]), dtype=np.float32)
    cids = np.empty((len(members),), dtype=np.int64)
    for i, (clip_id, frames) in enumerate(members):
        cids[i] = int(clip_id)
        if len(frames) == 0:
            out[i] = 0.0
            continue
        lo = int(frames.min())
        hi = int(frames.max()) + 1
        slab = qpos_preproc[int(clip_id), lo:hi]
        if slab.shape[0] == 0:
            out[i] = 0.0
            continue
        out[i] = resample_to_K(slab, K)
    return out, cids


def _real_transition_windows(
    instances: list,
    qpos_preproc: np.ndarray,
    K: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """K-frame windows for all real GT instances matching `label`, plus their
    per-window clip_ids for leave-clip-out inference."""
    matched = [inst for inst in instances if inst.label == label]
    if not matched:
        return (
            np.zeros((0, K, qpos_preproc.shape[-1]), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    out = np.empty((len(matched), K, qpos_preproc.shape[-1]), dtype=np.float32)
    cids = np.empty((len(matched),), dtype=np.int64)
    for i, inst in enumerate(matched):
        out[i] = extract_instance_window(qpos_preproc[inst.clip_id], inst, K=K)
        cids[i] = int(inst.clip_id)
    return out, cids


def evaluate_clusters_with_classifier(
    cluster_dict: dict,                     # from experiment results (abs_clip_id, frames)
    method_dir: str,
    seed: int,
    raw_h5_path: str,
    behavior_labels_csv: str,
    classifier_ckpt_dir: str,
    output_dir: str,
    window_frames: int = 100,
) -> dict:
    """Primary entry invoked by `main_transitions.py`. Writes outputs under
    `output_dir`. Returns a dict of aggregate metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    ckpt_root = Path(classifier_ckpt_dir)
    versions = sorted([p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("v")])
    if not versions:
        raise FileNotFoundError(f"no classifier checkpoint under {ckpt_root}")
    clf = LoadedClassifier(versions[-1])
    K = clf.K

    from topo_mimic.benchmark.classifier.data import load_raw_qpos
    raw_qpos = load_raw_qpos(raw_h5_path)
    qpos_preproc = preprocess_qpos(raw_qpos, exclude_xy=True, do_handle_rotation=True)

    # real-transition instances (filtered to TOP_8)
    raw_segs = load_behavior_segments(behavior_labels_csv)
    behav_segs, _ = split_segments_by_kind(raw_segs)
    real_instances = find_directional_transitions(behav_segs, allowed_labels=TOP_8_TRANSITIONS)

    # Precompute classifier outputs on REAL windows per label (held-out
    # inference so clips from training don't inflate the distribution).
    # Store the full softmax to derive both per-class argmax-recall
    # ("base_match_rate"[L]) and the confidence distribution (real_p_L[L]).
    real_softmax: dict[str, np.ndarray] = {}
    real_p_L: dict[str, np.ndarray] = {}
    base_match_rate: dict[str, float] = {}
    for label in TOP_8_TRANSITIONS:
        Xr, cids = _real_transition_windows(real_instances, qpos_preproc, K=K, label=label)
        if Xr.shape[0] == 0:
            real_softmax[label] = np.zeros((0, len(TOP_8_TRANSITIONS)), dtype=np.float32)
            real_p_L[label] = np.zeros((0,), dtype=np.float32)
            base_match_rate[label] = float("nan")
            continue
        soft, _n_folds = clf.held_out_softmax(Xr, cids)
        real_softmax[label] = soft.astype(np.float32)
        real_p_L[label] = soft[:, LABEL_TO_IDX[label]].astype(np.float32)
        base_match_rate[label] = float((soft.argmax(axis=1) == LABEL_TO_IDX[label]).mean())

    # load cluster members from the labels YAMLs (which have frame_start/end + label)
    cluster_members = _load_cluster_members(method_dir, seed)

    rows: list[dict] = []
    confusion_rows: list[np.ndarray] = []
    confusion_cluster_ids: list[int] = []
    confusion_labels: list[str] = []

    for cid, (primary_label, members) in sorted(cluster_members.items()):
        if not members:
            continue
        base_label = _strip_direction_from_label(primary_label)
        if base_label not in LABEL_TO_IDX:
            # cluster's directed label is not in TOP_8; skip
            continue
        L = base_label
        L_idx = LABEL_TO_IDX[L]

        Xc, c_cids = _member_windows(qpos_preproc, members, K=K)
        if Xc.shape[0] == 0:
            continue
        cluster_soft, n_folds_used = clf.held_out_softmax(Xc, c_cids)
        cluster_pred = cluster_soft.argmax(axis=1)
        match_rate = float((cluster_pred == L_idx).mean())
        mean_p_L = float(cluster_soft[:, L_idx].mean())
        frac_heldout = float((n_folds_used < clf.metadata.get("n_folds", 5)).mean())

        mean_p_L_real = float(real_p_L[L].mean()) if real_p_L[L].size else float("nan")
        real_p5 = float(np.quantile(real_p_L[L], 0.05)) if real_p_L[L].size else 0.0
        rank_fraction = float((cluster_soft[:, L_idx] >= real_p5).mean())
        real_rate_L = base_match_rate[L]
        match_rate_gap = (real_rate_L - match_rate) if np.isfinite(real_rate_L) else float("nan")
        norm_match_rate = (match_rate / real_rate_L) if (np.isfinite(real_rate_L) and real_rate_L > 0) else float("nan")

        confusion = cluster_soft.mean(axis=0)
        rows.append({
            "cluster_id": int(cid),
            "primary_label": primary_label,
            "base_label": L,
            "n_members": int(Xc.shape[0]),
            "match_rate": round(match_rate, 4),
            "base_match_rate": round(real_rate_L, 4) if np.isfinite(real_rate_L) else float("nan"),
            "match_rate_gap": round(match_rate_gap, 4) if np.isfinite(match_rate_gap) else float("nan"),
            "normalized_match_rate": round(norm_match_rate, 4) if np.isfinite(norm_match_rate) else float("nan"),
            "mean_p_L": round(mean_p_L, 4),
            "mean_p_L_real": round(mean_p_L_real, 4),
            "p_L_gap": round(mean_p_L_real - mean_p_L, 4) if np.isfinite(mean_p_L_real) else float("nan"),
            "rank_fraction_vs_real_p5": round(rank_fraction, 4),
            "pct_members_held_out": round(frac_heldout * 100.0, 2),
            **{f"soft_{lab}": round(float(confusion[i]), 4)
               for i, lab in enumerate(TOP_8_TRANSITIONS)},
        })
        confusion_rows.append(confusion)
        confusion_cluster_ids.append(int(cid))
        confusion_labels.append(L)

    df = pd.DataFrame(rows)
    out_csv = Path(output_dir) / f"cluster_classifier_metrics_seed_{seed}.csv"
    df.to_csv(out_csv, index=False)

    # Per-label base rates (one row per label, not per cluster — shared
    # across all clusters of that label). Written separately so the paper
    # can reference the "classifier recall on real L" number directly.
    base_rows = [
        {
            "label": lbl,
            "n_real_windows": int(real_softmax[lbl].shape[0]),
            "base_match_rate": round(base_match_rate[lbl], 4) if np.isfinite(base_match_rate[lbl]) else float("nan"),
        }
        for lbl in TOP_8_TRANSITIONS
    ]
    base_df = pd.DataFrame(base_rows)
    base_df.to_csv(Path(output_dir) / f"base_match_rate_seed_{seed}.csv", index=False)

    # Per-label aggregated bar: base (real) vs cluster (mean across clusters of that label).
    # If no clusters were scorable (all primary_labels were singletons or outside TOP_8),
    # df is empty — still emit the base-only plot so the ceiling is visible.
    per_label_cluster_mean: dict[str, float] = {}
    for lbl in TOP_8_TRANSITIONS:
        if len(df) and "base_label" in df.columns:
            sub = df[df["base_label"] == lbl]
            per_label_cluster_mean[lbl] = float(sub["match_rate"].mean()) if len(sub) else float("nan")
        else:
            per_label_cluster_mean[lbl] = float("nan")
    _plot_base_vs_cluster(
        labels=TOP_8_TRANSITIONS,
        base=base_match_rate,
        cluster_mean=per_label_cluster_mean,
        output_png=str(Path(output_dir) / f"base_vs_cluster_match_rate_seed_{seed}.png"),
    )

    def _safe_mean(col: str) -> float:
        if not len(df) or col not in df.columns:
            return float("nan")
        return float(df[col].mean())

    # aggregate
    aggregate: dict = {
        "n_scored_clusters": int(len(rows)),
        "mean_match_rate": _safe_mean("match_rate"),
        "median_match_rate": float(df["match_rate"].median()) if len(df) and "match_rate" in df.columns else float("nan"),
        "mean_base_match_rate": float(np.nanmean([base_match_rate[l] for l in TOP_8_TRANSITIONS])),
        "mean_match_rate_gap": _safe_mean("match_rate_gap"),
        "mean_normalized_match_rate": _safe_mean("normalized_match_rate"),
        "mean_p_L_gap": _safe_mean("p_L_gap"),
        "mean_rank_fraction": _safe_mean("rank_fraction_vs_real_p5"),
        "base_match_rate_per_label": {l: (round(v, 4) if np.isfinite(v) else float("nan"))
                                       for l, v in base_match_rate.items()},
    }
    with open(Path(output_dir) / f"aggregate_seed_{seed}.yaml", "w") as f:
        yaml.safe_dump(aggregate, f, default_flow_style=False)

    if confusion_rows:
        cm = np.stack(confusion_rows, axis=0)
        fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.25 * len(confusion_rows) + 1.5)))
        im = ax.imshow(cm, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(TOP_8_TRANSITIONS)))
        ax.set_xticklabels(TOP_8_TRANSITIONS, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(confusion_rows)))
        ax.set_yticklabels(
            [f"c{cid} ({lab})" for cid, lab in zip(confusion_cluster_ids, confusion_labels)],
            fontsize=7,
        )
        fig.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title(f"Classifier softmax per cluster (seed {seed})", fontsize=9)
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"cluster_softmax_heatmap_seed_{seed}.png",
                    dpi=160, bbox_inches="tight")
        plt.close(fig)

    return aggregate
