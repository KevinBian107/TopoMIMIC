"""Window extraction for C2ST.

Produces (N, K, D) float32 window arrays for:
  - a cluster's member windows (set A)
  - the real GT windows of the same directional label (set B)

Both go through the SAME `extract_and_resample` path from preprocessing.py.
Real-GT pool is global (no clip exclusion) per design.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

from topo_mimic.benchmark.auto_label import (
    BehaviorSegment,
    load_behavior_segments,
    split_segments_by_kind,
)
from topo_mimic.benchmark.c2st.preprocessing import (
    extract_and_resample,
    preprocess_qpos,
)


def load_raw_qpos(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if "qpos_rollout" in f:
            return np.asarray(f["qpos_rollout"][:])
        if "qpos" in f:
            return np.asarray(f["qpos"][:])
        raise KeyError(f"no qpos_rollout/qpos in {h5_path}")


@dataclass
class RealTransitionInstance:
    clip_id: int
    directed_label: str   # "A -> B"
    start: int            # inclusive — a.start
    end: int              # exclusive — b.end (== a.end for merged runs)


def build_real_transitions(
    behavior_labels_csv: str,
    allowed_labels: list[str],
) -> dict[str, list[RealTransitionInstance]]:
    """Enumerate every adjacent behavior->behavior transition from GT whose
    directional label is in ``allowed_labels``. Same merging logic as the
    classifier pipeline: consecutive same-behavior segments are collapsed
    before pairing so only true changes register.
    """
    raw = load_behavior_segments(behavior_labels_csv)
    behav, _dirs = split_segments_by_kind(raw)
    allowed = set(allowed_labels)
    by_label: dict[str, list[RealTransitionInstance]] = {lbl: [] for lbl in allowed}

    for clip_id, segs in behav.items():
        if len(segs) < 2:
            continue
        segs_sorted = sorted(segs, key=lambda s: s.start)
        merged: list[BehaviorSegment] = [segs_sorted[0]]
        for s in segs_sorted[1:]:
            if s.behavior == merged[-1].behavior:
                merged[-1] = BehaviorSegment(
                    clip=merged[-1].clip, behavior=merged[-1].behavior,
                    start=merged[-1].start, end=s.end,
                )
            else:
                merged.append(s)
        for a, b in zip(merged[:-1], merged[1:]):
            lbl = f"{a.behavior} -> {b.behavior}"
            if lbl in allowed:
                by_label[lbl].append(RealTransitionInstance(
                    clip_id=int(clip_id), directed_label=lbl,
                    start=int(a.start), end=int(b.end),
                ))
    return by_label


MIN_MEMBER_FRAMES = 5


def cluster_windows_from_member(
    qpos_preproc: np.ndarray,
    clip_id: int,
    frames: np.ndarray,
    K: int,
) -> np.ndarray | None:
    """One member -> one (K, D) window. Rejects degenerate members:
    - empty frames
    - too-short frame span (< MIN_MEMBER_FRAMES) — would produce a near-
      constant 7400-dim vector after resample and trivially separate.
    Returns None when the member should be dropped.
    """
    if frames is None or len(frames) == 0:
        return None
    lo = int(np.asarray(frames).min())
    hi = int(np.asarray(frames).max()) + 1
    if hi - lo < MIN_MEMBER_FRAMES:
        return None
    return extract_and_resample(qpos_preproc, int(clip_id), lo, hi, K)


def stack_cluster_windows(
    qpos_preproc: np.ndarray,
    members: list[tuple[int, np.ndarray]],
    K: int,
) -> np.ndarray:
    """Stack (n_members_kept, K, D) for a cluster; degenerate members are dropped."""
    if not members:
        return np.zeros((0, K, qpos_preproc.shape[-1]), dtype=np.float32)
    kept: list[np.ndarray] = []
    for cid, frames in members:
        w = cluster_windows_from_member(qpos_preproc, cid, frames, K)
        if w is not None:
            kept.append(w)
    if not kept:
        return np.zeros((0, K, qpos_preproc.shape[-1]), dtype=np.float32)
    return np.stack(kept, axis=0)


def stack_real_windows(
    qpos_preproc: np.ndarray,
    instances: list[RealTransitionInstance],
    K: int,
) -> np.ndarray:
    """Stack (n_instances, K, D) for a list of real GT transitions."""
    if not instances:
        return np.zeros((0, K, qpos_preproc.shape[-1]), dtype=np.float32)
    out = np.empty((len(instances), K, qpos_preproc.shape[-1]), dtype=np.float32)
    for i, inst in enumerate(instances):
        out[i] = extract_and_resample(qpos_preproc, inst.clip_id, inst.start, inst.end, K)
    return out


def balance_pair(
    X_cluster: np.ndarray,
    X_real: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Match `|X_real| = |X_cluster|` by SUBSAMPLING the larger side.

    - If `|real| > |cluster|`: subsample real without replacement (common).
    - If `|real| < |cluster|`: subsample CLUSTER to `|real|`. This avoids
      bootstrapping the real side (duplicate real rows leaking across
      train/val in the downstream StratifiedKFold); trade-off is a smaller
      C2ST test.
    """
    n = X_cluster.shape[0]
    m = X_real.shape[0]
    if m == 0 or n == 0:
        return X_cluster, X_real
    if m == n:
        return X_cluster, X_real
    if m > n:
        idx = rng.choice(m, size=n, replace=False)
        return X_cluster, X_real[idx]
    idx_c = rng.choice(n, size=m, replace=False)
    return X_cluster[idx_c], X_real
