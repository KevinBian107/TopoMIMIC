"""Cluster purity analysis.

For each cluster, measure how faithfully its member samples actually match
the cluster's auto-assigned label. Per-sample purity = fraction of the
sample's frames whose GT (behavior, direction) matches the cluster label's
(behavior-set, direction). Cluster purity = mean across samples.

Label parsing:
  - "X -> Y Z"  →  behaviors = {X, Y}, direction = Z   (transition cluster)
  - "X Z"       →  behaviors = {X},    direction = Z   (single-behavior)

A sample frame "belongs" if:
  - The frame's concurrent GT behavior ∈ cluster behaviors, AND
  - The frame's concurrent GT direction == cluster direction.

Frames with no GT behavior overlap count as not-belonging. Frames with a
behavior match but no direction overlap fall back to direction = "Straight"
(to match the auto-labeler's zero-overlap fallback in
``direction_tag_from_window``).
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from topo_mimic.benchmark.auto_label import (
    BEHAVIOR_LABELS,
    DIRECTION_LABELS,
    load_behavior_segments,
    split_segments_by_kind,
)


@dataclass
class PurityResult:
    cluster_id: int
    label: str                    # full "X -> Y Z" or "X Z"
    from_to_key: str              # "X -> Y" or "X"
    direction: str                # "Left" / "Right" / "Straight"
    n_samples: int
    mean_purity: float
    std_purity: float
    min_purity: float
    max_purity: float


def _parse_cluster_label(label: str) -> tuple[set[str], str, str] | None:
    """Return (behavior_set, direction, from_to_key) or None if unparseable.

    ``from_to_key`` is the human-readable transition key without direction:
    "X -> Y" for transitions, "X" for singletons. Direction defaults to
    "Straight" when absent.
    """
    if not label:
        return None
    label = label.strip()
    if " -> " in label:
        from_part, to_part = label.split(" -> ", 1)
        to_tokens = to_part.strip().split()
        if not to_tokens:
            return None
        to_b = to_tokens[0]
        direction = to_tokens[1] if len(to_tokens) > 1 else "Straight"
        from_b = from_part.strip()
        if from_b not in BEHAVIOR_LABELS or to_b not in BEHAVIOR_LABELS:
            return None
        return {from_b, to_b}, direction, f"{from_b} -> {to_b}"
    tokens = label.strip().split()
    if not tokens:
        return None
    from_b = tokens[0]
    if from_b not in BEHAVIOR_LABELS:
        return None
    direction = tokens[1] if len(tokens) > 1 else "Straight"
    return {from_b}, direction, from_b


def _build_frame_arrays(
    clip_id: int,
    behavior_segments: dict,
    direction_segments: dict,
    max_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (behavior, direction) arrays of length ``max_frame``.

    Unassigned frames store empty string. When segments overlap, the earliest-
    start wins (matches ``_count_frames_per_behavior`` semantics in auto_label).
    """
    beh = np.full(max_frame, "", dtype=object)
    dirn = np.full(max_frame, "", dtype=object)

    for s in behavior_segments.get(int(clip_id), []):
        lo = max(0, s.start)
        hi = min(max_frame, s.end)
        if lo >= hi:
            continue
        mask_empty = beh[lo:hi] == ""
        beh[lo:hi] = np.where(mask_empty, s.behavior, beh[lo:hi])

    for s in direction_segments.get(int(clip_id), []):
        lo = max(0, s.start)
        hi = min(max_frame, s.end)
        if lo >= hi:
            continue
        mask_empty = dirn[lo:hi] == ""
        dirn[lo:hi] = np.where(mask_empty, s.behavior, dirn[lo:hi])

    return beh, dirn


def compute_cluster_purity(
    cluster_id: int,
    label: str,
    members: list[tuple[int, np.ndarray]],
    behavior_segments: dict,
    direction_segments: dict,
    frame_cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> PurityResult | None:
    """Compute mean per-sample purity for one cluster. Returns None if the
    cluster has no parseable label or no usable samples."""
    parsed = _parse_cluster_label(label)
    if parsed is None:
        return None
    behav_set, direction, from_to_key = parsed

    per_sample_purities: list[float] = []
    for clip_id, frames in members:
        cid = int(clip_id)
        if frames is None or len(frames) == 0:
            continue
        if frame_cache is not None and cid in frame_cache:
            beh_arr, dir_arr = frame_cache[cid]
        else:
            max_frame = int(max(
                (s.end for s in behavior_segments.get(cid, [])),
                default=0,
            ))
            max_frame_d = int(max(
                (s.end for s in direction_segments.get(cid, [])),
                default=0,
            ))
            max_frame = max(max_frame, max_frame_d, int(frames.max()) + 1)
            beh_arr, dir_arr = _build_frame_arrays(
                cid, behavior_segments, direction_segments, max_frame,
            )
            if frame_cache is not None:
                frame_cache[cid] = (beh_arr, dir_arr)

        idx = np.asarray(frames, dtype=int)
        idx = idx[(idx >= 0) & (idx < len(beh_arr))]
        if idx.size == 0:
            continue
        beh_sel = beh_arr[idx]
        dir_sel = dir_arr[idx]
        # Behavior match
        beh_match = np.array([b in behav_set for b in beh_sel], dtype=bool)
        # Direction match: empty direction string falls back to "Straight".
        dir_effective = np.where(dir_sel == "", "Straight", dir_sel)
        dir_match = dir_effective == direction
        n_match = int(np.sum(beh_match & dir_match))
        per_sample_purities.append(n_match / float(len(idx)))

    if not per_sample_purities:
        return None
    arr = np.asarray(per_sample_purities, dtype=float)
    return PurityResult(
        cluster_id=int(cluster_id),
        label=label,
        from_to_key=from_to_key,
        direction=direction,
        n_samples=len(per_sample_purities),
        mean_purity=float(arr.mean()),
        std_purity=float(arr.std(ddof=0)),
        min_purity=float(arr.min()),
        max_purity=float(arr.max()),
    )


def run_purity_analysis(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    labels: dict[int, str],
    behavior_labels_csv: str,
    output_dir: str,
    seed: int,
    method_name: str,
) -> pd.DataFrame:
    """Run purity analysis for every cluster in ``cluster_dict``. Writes
    ``{method}_seed{seed}_purity.csv`` to ``output_dir`` and returns the
    per-cluster DataFrame.
    """
    all_segs = load_behavior_segments(behavior_labels_csv)
    behav_segs, dir_segs = split_segments_by_kind(all_segs)

    frame_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    rows: list[dict] = []
    for cid, members in sorted(cluster_dict.items()):
        label = labels.get(cid, "") or labels.get(str(cid), "")
        result = compute_cluster_purity(
            cluster_id=cid,
            label=label,
            members=members,
            behavior_segments=behav_segs,
            direction_segments=dir_segs,
            frame_cache=frame_cache,
        )
        if result is None:
            rows.append({
                "cluster_id": cid,
                "Label": label,
                "From_To": "",
                "Turn_Direction": "",
                "N_Samples": 0,
                "Mean_Purity": float("nan"),
                "Std_Purity": float("nan"),
                "Min_Purity": float("nan"),
                "Max_Purity": float("nan"),
            })
            continue
        rows.append({
            "cluster_id": result.cluster_id,
            "Label": result.label,
            "From_To": result.from_to_key,
            "Turn_Direction": result.direction.lower(),
            "N_Samples": result.n_samples,
            "Mean_Purity": round(result.mean_purity, 6),
            "Std_Purity": round(result.std_purity, 6),
            "Min_Purity": round(result.min_purity, 6),
            "Max_Purity": round(result.max_purity, 6),
        })

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    out_csv = Path(output_dir) / f"{method_name}_seed{seed}_purity.csv"
    df.to_csv(out_csv, index=False)
    return df
