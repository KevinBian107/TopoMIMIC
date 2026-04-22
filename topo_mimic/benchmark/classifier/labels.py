"""Label-space utilities for the transition classifier.

The classifier targets the TOP_8 directional transitions observed in
`rodent_labels.csv`. Each training window is a temporal slice of a GT
transition, resampled (down/upsampled) to a fixed number of frames K.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from topo_mimic.benchmark.auto_label import (
    BEHAVIOR_LABELS,
    BehaviorSegment,
)
from topo_mimic.benchmark.classifier import LABEL_TO_IDX, TOP_8_TRANSITIONS


@dataclass
class TransitionInstance:
    """One GT transition: A -> B with frame ranges within a clip."""
    clip_id: int
    label: str            # e.g. "Immobile -> Turn"
    a_start: int          # inclusive
    a_end: int            # exclusive (end of A segment)
    b_start: int          # inclusive (== a_end)
    b_end: int            # exclusive
    a_behavior: str
    b_behavior: str

    @property
    def start(self) -> int:
        return self.a_start

    @property
    def end(self) -> int:
        return self.b_end


def find_directional_transitions(
    behavior_segments: dict[int, list[BehaviorSegment]],
    allowed_labels: list[str] = TOP_8_TRANSITIONS,
) -> list[TransitionInstance]:
    """Enumerate every adjacent behavior->behavior transition in the GT whose
    directional label is in ``allowed_labels``.

    Uses only BEHAVIOR_LABELS segments (caller must have already split out
    direction segments). Adjacent same-behavior segments are merged before
    detection so only true behavior changes register as transitions.
    """
    instances: list[TransitionInstance] = []
    allowed = set(allowed_labels)

    for clip_id, raw_segs in behavior_segments.items():
        segs = [s for s in raw_segs if s.behavior in BEHAVIOR_LABELS]
        if len(segs) < 2:
            continue
        segs = sorted(segs, key=lambda s: s.start)

        # merge consecutive same-behavior runs into single segments
        merged: list[BehaviorSegment] = [BehaviorSegment(
            clip=segs[0].clip, behavior=segs[0].behavior,
            start=segs[0].start, end=segs[0].end,
        )]
        for s in segs[1:]:
            if s.behavior == merged[-1].behavior:
                merged[-1] = BehaviorSegment(
                    clip=merged[-1].clip, behavior=merged[-1].behavior,
                    start=merged[-1].start, end=s.end,
                )
            else:
                merged.append(BehaviorSegment(
                    clip=s.clip, behavior=s.behavior,
                    start=s.start, end=s.end,
                ))

        for a, b in zip(merged[:-1], merged[1:]):
            label = f"{a.behavior} -> {b.behavior}"
            if label not in allowed:
                continue
            instances.append(TransitionInstance(
                clip_id=int(clip_id), label=label,
                a_start=int(a.start), a_end=int(a.end),
                b_start=int(b.start), b_end=int(b.end),
                a_behavior=a.behavior, b_behavior=b.behavior,
            ))
    return instances


def tally_transitions(instances: list[TransitionInstance]) -> Counter:
    c: Counter = Counter()
    for inst in instances:
        c[inst.label] += 1
    return c


def resample_to_K(
    arr: np.ndarray,
    K: int,
) -> np.ndarray:
    """Linearly resample a (T, D) array along axis 0 to K frames.

    Down-sampling (T > K) and up-sampling (T < K) are the same linear
    interpolation path. For T == K returns a copy.
    """
    if arr.ndim != 2:
        raise ValueError(f"expected (T, D) array, got shape {arr.shape}")
    T, D = arr.shape
    if T == K:
        return arr.astype(np.float32, copy=True)
    if T == 1:
        return np.broadcast_to(arr, (K, D)).astype(np.float32, copy=True)
    # linear interpolate in time
    src_idx = np.linspace(0, T - 1, K, dtype=np.float64)
    lo = np.floor(src_idx).astype(np.int64)
    hi = np.clip(lo + 1, 0, T - 1)
    frac = (src_idx - lo).astype(np.float32)
    out = (1.0 - frac[:, None]) * arr[lo] + frac[:, None] * arr[hi]
    return out.astype(np.float32, copy=True)


def extract_instance_window(
    qpos_preproc: np.ndarray,
    inst: TransitionInstance,
    K: int,
    pad_frames: int = 0,
) -> np.ndarray:
    """Slice `qpos_preproc[inst.start - pad : inst.end + pad]` and resample to (K, D)."""
    T_max = qpos_preproc.shape[0]
    lo = max(0, inst.start - pad_frames)
    hi = min(T_max, inst.end + pad_frames)
    slab = qpos_preproc[lo:hi]
    if slab.shape[0] == 0:
        return np.zeros((K, qpos_preproc.shape[1]), dtype=np.float32)
    return resample_to_K(slab, K)
