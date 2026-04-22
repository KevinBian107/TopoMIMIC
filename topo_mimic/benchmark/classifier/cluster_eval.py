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

    # precompute classifier softmax on real windows per label. Use
    # held-out-only inference so clips that were in training don't inflate
    # the confidence distribution.
    real_p_L: dict[str, np.ndarray] = {}
    for label in TOP_8_TRANSITIONS:
        Xr, cids = _real_transition_windows(real_instances, qpos_preproc, K=K, label=label)
        if Xr.shape[0] == 0:
            real_p_L[label] = np.zeros((0,), dtype=np.float32)
            continue
        soft, _n_folds = clf.held_out_softmax(Xr, cids)
        real_p_L[label] = soft[:, LABEL_TO_IDX[label]].astype(np.float32)

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

        confusion = cluster_soft.mean(axis=0)
        rows.append({
            "cluster_id": int(cid),
            "primary_label": primary_label,
            "base_label": L,
            "n_members": int(Xc.shape[0]),
            "match_rate": round(match_rate, 4),
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

    # aggregate
    aggregate: dict = {
        "n_scored_clusters": int(len(rows)),
        "mean_match_rate": float(df["match_rate"].mean()) if len(df) else float("nan"),
        "median_match_rate": float(df["match_rate"].median()) if len(df) else float("nan"),
        "mean_p_L_gap": float(df["p_L_gap"].mean()) if len(df) else float("nan"),
        "mean_rank_fraction": float(df["rank_fraction_vs_real_p5"].mean()) if len(df) else float("nan"),
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
