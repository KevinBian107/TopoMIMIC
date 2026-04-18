"""
Auto-labeling clusters from ground-truth behavior annotations.

New algorithm (2026-04-18, replaces the W2-distance path in
`topo_mimic/benchmark/labeling/core.py`):

For each cluster, each member is a (clip_id, frame_indices) pair. The label
for a cluster is computed in two steps.

**Per-member (clip-level) labeling**:
  1. Look up the clip's GT behavior segments from the behavior-labels CSV.
  2. Restrict to the frames in `frame_indices` (not the whole clip).
  3. Count frames per behavior within those indices.
  4. If the top behavior is `Grooming`, emit the singleton label `Grooming`
     (grooming override; matches the user spec that groom is not a transition).
  5. Else, compute the fraction of the 2nd-top behavior. If it is < 20% of the
     member's frames, emit the singleton top behavior. Otherwise, emit the
     canonical transition label `(behav_a, behav_b)` with behav_a < behav_b
     alphabetically, rendered as the string ``"behav_a <-> behav_b"``.

**Cluster-level labeling**:
  1. Majority vote across all member labels.
  2. On tie, return the set of tied labels as `alternatives`; `primary` is the
     first alphabetically (deterministic for I/O) but the caller is expected
     to present the alternatives for manual review.

No Wasserstein distance, no MDS, no interpolation of primitive windows.
Frame counts alone drive the label.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


GROOMING_LABEL = "Grooming"
DEFAULT_DOMINANCE_THRESHOLD = 0.2


@dataclass
class BehaviorSegment:
    clip: int
    behavior: str
    start: int  # inclusive
    end: int    # exclusive


@dataclass
class ClusterLabel:
    primary: str
    alternatives: list[str] = field(default_factory=list)
    per_clip_labels: list[str] = field(default_factory=list)
    vote_counts: dict[str, int] = field(default_factory=dict)


_CLIP_ID_PATTERN = re.compile(r"clip_(\d+)")


def _parse_clip_id(video_name: str) -> int | None:
    match = _CLIP_ID_PATTERN.search(video_name)
    if match is None:
        return None
    return int(match.group(1))


def load_behavior_segments(csv_path: str) -> dict[int, list[BehaviorSegment]]:
    """
    Load behavior segments from a rodent-style CSV.

    Supports both schemas in the repo:
      - New schema (behavior_labels/rodent.csv): columns Video, Behavior,
        Start_Frame, End_Frame. `clip_id` parsed from "clip_<N>" in Video.
      - Old schema (with explicit Clip column): columns Clip, Behavior,
        Start_Frame, End_Frame.

    Returns a dict {clip_id -> sorted list of BehaviorSegment}. End is
    treated as an EXCLUSIVE upper bound; if End_Frame is the last included
    frame (inclusive), we store end = end_frame + 1.
    """
    segments: dict[int, list[BehaviorSegment]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "Clip" in row and row.get("Clip"):
                try:
                    clip_id = int(row["Clip"])
                except (ValueError, TypeError):
                    continue
            elif "Video" in row:
                clip_id = _parse_clip_id(row["Video"])
                if clip_id is None:
                    continue
            else:
                continue

            try:
                start = int(row["Start_Frame"])
                end_inclusive = int(row["End_Frame"])
            except (KeyError, ValueError):
                continue

            seg = BehaviorSegment(
                clip=clip_id,
                behavior=(row.get("Behavior") or "").strip(),
                start=start,
                end=end_inclusive + 1,
            )
            if not seg.behavior:
                continue
            segments.setdefault(seg.clip, []).append(seg)

    for segs in segments.values():
        segs.sort(key=lambda s: s.start)
    return segments


def _count_frames_per_behavior(
    frame_indices: Iterable[int],
    clip_segments: list[BehaviorSegment],
) -> tuple[Counter, int]:
    """Return (Counter of behavior -> frame count, n_assigned frames)."""
    if not clip_segments:
        return Counter(), 0
    counter: Counter = Counter()
    frames_arr = np.asarray(list(frame_indices), dtype=int)
    if frames_arr.size == 0:
        return counter, 0

    assigned = 0
    starts = np.array([s.start for s in clip_segments])
    ends = np.array([s.end for s in clip_segments])
    behavior_names = [s.behavior for s in clip_segments]

    for f in frames_arr:
        mask = (starts <= f) & (ends > f)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            continue
        counter[behavior_names[int(idxs[0])]] += 1
        assigned += 1

    return counter, assigned


def _canonical_transition(a: str, b: str) -> str:
    """Canonical transition string ``"behav_a <-> behav_b"`` with a <= b alphabetically."""
    a, b = sorted([a, b])
    return f"{a} <-> {b}"


def label_member_gt(
    frame_indices: np.ndarray,
    clip_segments: list[BehaviorSegment],
    dominance_threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
    grooming_label: str = GROOMING_LABEL,
) -> str:
    """
    Label a single cluster member (clip_id, frame_indices) using the GT
    majority rule described in the module docstring.

    Returns the empty string if no GT segments overlap the member's frames.
    """
    if frame_indices is None or len(frame_indices) == 0 or not clip_segments:
        return ""

    counts, n_assigned = _count_frames_per_behavior(frame_indices, clip_segments)
    if n_assigned == 0:
        return ""

    ranked = counts.most_common()
    top_behavior, top_count = ranked[0]
    if top_behavior == grooming_label:
        return grooming_label

    if len(ranked) < 2:
        return top_behavior

    second_behavior, second_count = ranked[1]
    if second_count / n_assigned < dominance_threshold:
        return top_behavior

    return _canonical_transition(top_behavior, second_behavior)


def auto_label_clusters(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    segments: dict[int, list[BehaviorSegment]],
    dominance_threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
    grooming_label: str = GROOMING_LABEL,
) -> dict[int, ClusterLabel]:
    """
    Apply the GT majority-vote labeling to every cluster.

    Returns a dict {cluster_id -> ClusterLabel}. ClusterLabel.alternatives is
    non-empty iff the top vote is a tie; in that case caller should expose
    the alternatives for manual picking.
    """
    cluster_labels: dict[int, ClusterLabel] = {}

    for cluster_id, members in sorted(cluster_dict.items()):
        per_clip_labels: list[str] = []
        for abs_clip_id, frames in members:
            clip_segs = segments.get(int(abs_clip_id), [])
            lbl = label_member_gt(
                frames,
                clip_segs,
                dominance_threshold=dominance_threshold,
                grooming_label=grooming_label,
            )
            per_clip_labels.append(lbl)

        non_empty = [l for l in per_clip_labels if l]
        if not non_empty:
            cluster_labels[cluster_id] = ClusterLabel(
                primary="",
                alternatives=[],
                per_clip_labels=per_clip_labels,
                vote_counts={},
            )
            continue

        votes = Counter(non_empty)
        top_count = max(votes.values())
        tied_labels = sorted([l for l, c in votes.items() if c == top_count])
        primary = tied_labels[0]
        alternatives = tied_labels[1:] if len(tied_labels) > 1 else []

        cluster_labels[cluster_id] = ClusterLabel(
            primary=primary,
            alternatives=alternatives,
            per_clip_labels=per_clip_labels,
            vote_counts=dict(votes),
        )

    return cluster_labels


def generate_auto_labels_yaml(
    method_name: str,
    n_clusters: int,
    seed: int,
    output_dir: str,
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    behavior_labels_csv: str,
    experiment_id: str | None = None,
    dominance_threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
    grooming_label: str = GROOMING_LABEL,
) -> str:
    """
    Auto-label every cluster and write ``labels_seed_{seed}.yaml`` +
    ``sample_labels_seed_{seed}.yaml`` in output_dir.

    The labels YAML preserves the legacy schema (method, seed, n_clusters,
    labels) so existing readers (GUI, analysis) continue to work. A new
    top-level ``alternatives`` dict is added for clusters with a tie.
    """
    if experiment_id is None:
        from datetime import datetime
        experiment_id = f"experiment_{datetime.now().strftime('%Y-%m-%d')}"

    segments = load_behavior_segments(behavior_labels_csv)
    cluster_labels = auto_label_clusters(
        cluster_dict,
        segments,
        dominance_threshold=dominance_threshold,
        grooming_label=grooming_label,
    )

    all_labels = {i: cluster_labels.get(i, ClusterLabel(primary="")).primary for i in range(n_clusters)}
    alternatives_map = {
        i: cluster_labels[i].alternatives
        for i in range(n_clusters)
        if i in cluster_labels and cluster_labels[i].alternatives
    }
    vote_counts_map = {
        i: cluster_labels[i].vote_counts
        for i in range(n_clusters)
        if i in cluster_labels and cluster_labels[i].vote_counts
    }

    labels_content = {
        "experiment_id": experiment_id,
        "method": method_name,
        "n_clusters": n_clusters,
        "seed": seed,
        "labels": all_labels,
        "alternatives": alternatives_map,
        "vote_counts": vote_counts_map,
        "dominance_threshold": float(dominance_threshold),
    }
    labels_path = Path(output_dir) / f"labels_seed_{seed}.yaml"
    with open(labels_path, "w") as f:
        f.write("# AUTO-GENERATED from ground truth behavior labels (GT majority vote)\n")
        f.write("# Tied clusters expose their alternatives under `alternatives`.\n\n")
        yaml.dump(labels_content, f, default_flow_style=False, sort_keys=False)

    sample_detail: dict = {}
    for cluster_id in sorted(cluster_dict.keys()):
        members = cluster_dict[cluster_id]
        cl = cluster_labels.get(cluster_id, ClusterLabel(primary=""))
        entries = []
        for i, (abs_clip_id, frames) in enumerate(members):
            lbl = cl.per_clip_labels[i] if i < len(cl.per_clip_labels) else ""
            entries.append({
                "clip_id": int(abs_clip_id),
                "frame_start": int(frames.min()),
                "frame_end": int(frames.max()),
                "n_frames": int(len(frames)),
                "label": lbl,
            })
        sample_detail[int(cluster_id)] = {
            "majority_label": cl.primary,
            "alternatives": cl.alternatives,
            "n_samples": len(members),
            "samples": entries,
        }

    sample_content = {
        "experiment_id": experiment_id,
        "method": method_name,
        "seed": seed,
        "clusters": sample_detail,
    }
    sample_path = Path(output_dir) / f"sample_labels_seed_{seed}.yaml"
    with open(sample_path, "w") as f:
        yaml.dump(sample_content, f, default_flow_style=False, sort_keys=False)

    return str(labels_path)
