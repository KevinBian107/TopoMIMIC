"""Dataset building and clip-level CV splits for the transition classifier.

Flow:
    load raw qpos (N, T, D_qpos=74)
    -> preprocess_qpos: (N, T, 74)
    -> per clip: find TransitionInstance objects from rodent_labels.csv
    -> per instance: extract_instance_window -> (K, D=74) resampled window
    -> class balance via upsampling (each class to target_count via sampling-with-replacement)
    -> (X, y, clip_id) training arrays

Normalization is computed on TRAIN ONLY and returned so eval/inference uses the
same statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from topo_mimic.benchmark.auto_label import (
    BehaviorSegment,
    load_behavior_segments,
    split_segments_by_kind,
)
from topo_mimic.benchmark.classifier import LABEL_TO_IDX, TOP_8_TRANSITIONS
from topo_mimic.benchmark.classifier.labels import (
    TransitionInstance,
    extract_instance_window,
    find_directional_transitions,
    tally_transitions,
)
from topo_mimic.benchmark.classifier.preprocessing import (
    compute_joint_normalization,
    get_joint_start_index,
    normalize_joints,
    preprocess_qpos,
)


@dataclass
class ClassifierDataset:
    X: np.ndarray           # (N, K, D) preprocessed + z-scored windows
    y: np.ndarray           # (N,) int class indices
    clip_ids: np.ndarray    # (N,) clip_id per window
    K: int
    D: int
    norm_mean: np.ndarray
    norm_std: np.ndarray
    joint_start: int

    def class_counts(self) -> np.ndarray:
        counts = np.zeros(len(TOP_8_TRANSITIONS), dtype=np.int64)
        for lbl in self.y:
            counts[int(lbl)] += 1
        return counts


def load_raw_qpos(h5_path: str) -> np.ndarray:
    """Return (N_clips, T, 74) qpos array from the consolidated rodent_ar.h5."""
    with h5py.File(h5_path, "r") as f:
        if "qpos_rollout" in f:
            return np.asarray(f["qpos_rollout"][:])
        if "qpos" in f:
            return np.asarray(f["qpos"][:])
        raise KeyError(f"no qpos_rollout/qpos in {h5_path}")


def build_transition_instances(
    behavior_labels_csv: str,
) -> tuple[list[TransitionInstance], dict]:
    raw_segs = load_behavior_segments(behavior_labels_csv)
    behav_segs, _dir_segs = split_segments_by_kind(raw_segs)
    instances = find_directional_transitions(behav_segs, allowed_labels=TOP_8_TRANSITIONS)
    counts = tally_transitions(instances)
    return instances, {
        "per_label_counts": {lbl: int(counts.get(lbl, 0)) for lbl in TOP_8_TRANSITIONS},
        "total_instances": len(instances),
        "n_labeled_clips": len({i.clip_id for i in instances}),
    }


def resample_instances(
    instances: list[TransitionInstance],
    qpos_preproc: np.ndarray,  # (N_clips, T, D)
    K: int,
    pad_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, clip_ids) for every instance. X is pre-normalization."""
    X = np.empty((len(instances), K, qpos_preproc.shape[-1]), dtype=np.float32)
    y = np.empty((len(instances),), dtype=np.int64)
    clip_ids = np.empty((len(instances),), dtype=np.int64)
    for i, inst in enumerate(instances):
        X[i] = extract_instance_window(qpos_preproc[inst.clip_id], inst, K=K, pad_frames=pad_frames)
        y[i] = LABEL_TO_IDX[inst.label]
        clip_ids[i] = inst.clip_id
    return X, y, clip_ids


def upsample_to_balance(
    X: np.ndarray, y: np.ndarray, clip_ids: np.ndarray,
    target_count: int | None = None,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class sampling-with-replacement up to the majority class count (or
    ``target_count``). Never downsample; the majority class keeps all its
    members.
    """
    rng = np.random.default_rng(rng_seed)
    n_classes = len(TOP_8_TRANSITIONS)
    counts = np.array([(y == c).sum() for c in range(n_classes)], dtype=np.int64)
    if target_count is None:
        target_count = int(counts.max())
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    c_parts: list[np.ndarray] = []
    for c in range(n_classes):
        mask = (y == c)
        n = int(mask.sum())
        if n == 0:
            continue
        idx = np.where(mask)[0]
        if n >= target_count:
            keep = idx
        else:
            extra = rng.choice(idx, size=target_count - n, replace=True)
            keep = np.concatenate([idx, extra])
        X_parts.append(X[keep])
        y_parts.append(y[keep])
        c_parts.append(clip_ids[keep])
    X_bal = np.concatenate(X_parts, axis=0)
    y_bal = np.concatenate(y_parts, axis=0)
    c_bal = np.concatenate(c_parts, axis=0)
    return X_bal, y_bal, c_bal


def normalize_with_train_stats(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    joint_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-joint mean/std on X_train (all windows flattened over time),
    apply to both. Non-joint dims (pos + 6D rotation) left untouched.
    """
    mean, std = compute_joint_normalization(X_train, joint_start)
    X_train_n = normalize_joints(X_train, joint_start, mean, std)
    X_eval_n = normalize_joints(X_eval, joint_start, mean, std)
    return X_train_n, X_eval_n, mean, std


def build_full_dataset(
    behavior_labels_csv: str,
    raw_h5_path: str,
    K: int = 100,
    exclude_xy: bool = True,
    do_handle_rotation: bool = True,
    pad_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[TransitionInstance], dict]:
    """Build (X_unnorm, y, clip_ids, instances, stats) over all labeled clips.

    `X_unnorm` is preprocessed (handle_rotation + exclude_xy applied) but NOT
    joint-normalized — normalization happens at fold-time using train-split
    stats only.
    """
    raw_qpos = load_raw_qpos(raw_h5_path)
    qpos_preproc = preprocess_qpos(
        raw_qpos,
        exclude_xy=exclude_xy,
        do_handle_rotation=do_handle_rotation,
    )
    instances, stats = build_transition_instances(behavior_labels_csv)
    X, y, clip_ids = resample_instances(instances, qpos_preproc, K=K, pad_frames=pad_frames)
    stats["D_preprocessed"] = int(qpos_preproc.shape[-1])
    stats["K"] = int(K)
    stats["n_windows"] = int(len(y))
    return X, y, clip_ids, instances, stats
