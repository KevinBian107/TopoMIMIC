"""Per-(method, seed) C2ST runner invoked from main_transitions.py.

For each cluster in the method's sample_labels yaml:
  1. Parse the directional primary label (strip trailing L/R/Straight token).
  2. Extract cluster windows (K frames each, preprocessed).
  3. Extract real GT windows for the same directional label.
  4. Balance real to match |cluster|.
  5. Run 3 binary classifiers via 5-fold CV; record accuracy per model.

Writes:
  - cluster_c2st_seed_{N}.csv  (one row per cluster, cols = model accuracies)
  - aggregate_c2st_seed_{N}.yaml
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from topo_mimic.experiment.c2st import TRANSITION_LABELS, undirected_key
from topo_mimic.experiment.c2st.cluster_c2st import run_c2st_single
from topo_mimic.experiment.c2st.data import (
    balance_pair,
    build_real_transitions,
    load_raw_qpos,
    stack_cluster_windows,
    stack_real_windows,
)
from topo_mimic.experiment.c2st.preprocessing import preprocess_qpos


_DIRECTION_TOKENS = {"Left", "Right", "Straight"}


def _strip_direction(label: str) -> str:
    """Drop a trailing Left/Right/Straight token from a cluster label.

    'Immobile -> Turn Left'   -> 'Immobile -> Turn'
    'Immobile Straight'       -> 'Immobile'          (singleton; not in TRANSITION_LABELS)
    'Immobile -> Turn'        -> 'Immobile -> Turn'  (no-op)
    """
    if not label:
        return ""
    toks = label.strip().split()
    if toks and toks[-1] in _DIRECTION_TOKENS:
        toks = toks[:-1]
    return " ".join(toks)


def _load_members(method_dir: str, seed: int) -> dict[int, tuple[str, list[tuple[int, np.ndarray]]]]:
    """Return {cluster_id: (directional_primary_label, members)}."""
    labels_path = os.path.join(method_dir, f"labels_seed_{seed}.yaml")
    samples_path = os.path.join(method_dir, f"sample_labels_seed_{seed}.yaml")
    if not (os.path.exists(labels_path) and os.path.exists(samples_path)):
        return {}
    with open(labels_path) as f:
        labels_blob = yaml.safe_load(f) or {}
    with open(samples_path) as f:
        samples_blob = yaml.safe_load(f) or {}
    labels_map = labels_blob.get("labels", {}) or {}
    out: dict[int, tuple[str, list[tuple[int, np.ndarray]]]] = {}
    for cid_str, meta in (samples_blob.get("clusters", {}) or {}).items():
        cid = int(cid_str)
        primary = labels_map.get(cid) or labels_map.get(str(cid)) or ""
        members: list[tuple[int, np.ndarray]] = []
        for s in meta.get("samples", []) or []:
            lo = int(s["frame_start"])
            hi = int(s["frame_end"])
            frames = np.arange(lo, hi + 1, dtype=np.int64)
            members.append((int(s["clip_id"]), frames))
        out[cid] = (primary, members)
    return out


def run_c2st_for_method_seed(
    method_dir: str,
    seed: int,
    raw_h5_path: str,
    behavior_labels_csv: str,
    output_dir: str,
    window_frames: int = 100,
    min_cluster_size: int = 8,
    n_folds: int = 5,
    models: tuple[str, ...] = ("logreg", "rf", "mlp"),
) -> dict:
    """Run C2ST for every directionally-labeled transition cluster in a
    (method, seed). Returns an aggregate dict and writes per-cluster CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_qpos = load_raw_qpos(raw_h5_path)
    qpos_preproc = preprocess_qpos(raw_qpos, skip_root_dims=7)

    real_by_label = build_real_transitions(behavior_labels_csv, TRANSITION_LABELS)
    # Pre-stack real windows per label so we reuse them across clusters.
    real_stack_by_label: dict[str, np.ndarray] = {
        lbl: stack_real_windows(qpos_preproc, insts, K=window_frames)
        for lbl, insts in real_by_label.items()
    }

    members_per_cluster = _load_members(method_dir, seed)

    rows: list[dict] = []
    rng = np.random.default_rng(seed + 17)
    for cid, (primary_label, members) in sorted(members_per_cluster.items()):
        if not members:
            continue
        directed = _strip_direction(primary_label)
        if directed not in TRANSITION_LABELS:
            continue
        X_cluster = stack_cluster_windows(qpos_preproc, members, K=window_frames)
        n = X_cluster.shape[0]
        if n < min_cluster_size:
            rows.append({
                "cluster_id": cid, "primary_label": primary_label,
                "directed_label": directed,
                "undirected": undirected_key(directed) or "",
                "n_cluster": n, "n_real_pool": int(real_stack_by_label[directed].shape[0]),
                **{f"acc_{m}": float("nan") for m in models},
                "insufficient_data": True,
            })
            continue

        X_real = real_stack_by_label[directed]
        if X_real.shape[0] == 0:
            rows.append({
                "cluster_id": cid, "primary_label": primary_label,
                "directed_label": directed,
                "undirected": undirected_key(directed) or "",
                "n_cluster": n, "n_real_pool": 0,
                **{f"acc_{m}": float("nan") for m in models},
                "insufficient_data": True,
            })
            continue

        # balance_pair may subsample X_cluster when |real| < |cluster|;
        # capture BOTH return values so their lengths match in the CV pool.
        X_cluster_bal, X_real_balanced = balance_pair(X_cluster, X_real, rng)
        if X_cluster_bal.shape[0] < min_cluster_size:
            rows.append({
                "cluster_id": cid, "primary_label": primary_label,
                "directed_label": directed,
                "undirected": undirected_key(directed) or "",
                "n_cluster": n, "n_real_pool": int(X_real.shape[0]),
                **{f"acc_{m}": float("nan") for m in models},
                "insufficient_data": True,
            })
            continue
        accs = run_c2st_single(
            X_cluster=X_cluster_bal, X_real=X_real_balanced,
            model_names=models, n_folds=n_folds, seed=seed,
        )
        rows.append({
            "cluster_id": cid, "primary_label": primary_label,
            "directed_label": directed,
            "undirected": undirected_key(directed) or "",
            "n_cluster": int(X_cluster_bal.shape[0]),
            "n_real_pool": int(X_real.shape[0]),
            **{f"acc_{m}": round(accs[m], 4) for m in models},
            "insufficient_data": False,
        })

    df = pd.DataFrame(rows)
    out_csv = Path(output_dir) / f"cluster_c2st_seed_{seed}.csv"
    df.to_csv(out_csv, index=False)

    usable = df[~df["insufficient_data"]] if len(df) else df
    aggregate = {
        "n_total_clusters": int(len(df)),
        "n_usable_clusters": int(len(usable)),
        "n_insufficient": int(df["insufficient_data"].sum()) if len(df) else 0,
        "mean_acc_per_model": {
            m: float(usable[f"acc_{m}"].mean()) if len(usable) else float("nan")
            for m in models
        },
    }
    with open(Path(output_dir) / f"aggregate_c2st_seed_{seed}.yaml", "w") as f:
        yaml.safe_dump(aggregate, f, default_flow_style=False)
    return aggregate
