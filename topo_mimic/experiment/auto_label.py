"""
Auto-labeling clusters from ground-truth behavior + direction annotations.

2026-04-21 rewrite. The rodent_labels.csv ground-truth is now a mix of
**behavior labels** and **direction flags**:

    behaviors  = {Immobile, Turn, Rear, Walk}     (mutually exclusive)
    directions = {Left, Right, Straight}          (supplemental overlay)

Each frame is labeled with AT MOST one behavior and AT MOST one direction.
The two label kinds are stored in the same CSV (column ``Label``); callers
must split them via ``split_segments_by_kind``.

Two labelers live here:

1. ``auto_label_clusters_directional`` — PRIMARY, used by the benchmark and
   GUI. Emits labels of the form ``"A -> B Direction"`` (transition) or
   ``"A Direction"`` (single behavior). Per-member, the ordered pair is found
   by first-appearance within the member's frames; the cluster's
   (from, to) is a majority vote across members. Direction is now assigned
   via a per-sample vote: each member's window gets its own direction tag
   under the frame-count rule (Left/Right > Straight; longer of L/R wins
   ties → Left), and the cluster direction is the plain majority across
   sample tags (ties → Left > Right > Straight). The GT side uses the same
   per-window rule via ``direction_tag_from_window`` so cluster and GT
   labels are computed symmetrically.

2. ``auto_label_clusters`` — LEGACY canonical labeler, kept for the
   scale-sweep discovery logic. Emits ``"A <-> B"`` (alphabetical) or a
   singleton ``"A"``. Ignores direction labels entirely.

Both labelers share ``ClusterLabel``, ``BehaviorSegment``,
``load_behavior_segments``, and ``split_segments_by_kind``.
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


BEHAVIOR_LABELS: frozenset[str] = frozenset({"Immobile", "Turn", "Rear", "Walk"})
DIRECTION_LABELS: frozenset[str] = frozenset({"Left", "Right", "Straight"})

GROOMING_LABEL = "Grooming"
DEFAULT_DOMINANCE_THRESHOLD = 0.2

DIRECTIONAL_SEP = " -> "
CANONICAL_SEP = " <-> "


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
    Load segments from a behavior-labels CSV. Returns ALL segments, mixing
    behavior labels and direction labels in the same dict. Use
    ``split_segments_by_kind`` if you need them separately.

    Supports the following schemas (columns detected by name):
      - New rodent schema (behavior_labels/rodent_labels.csv):
          columns Video, Track, Label, Start_Time, End_Time, Start_Frame, End_Frame
      - Legacy rodent schema (columns Video, Behavior, Start_Frame, End_Frame)

    End is EXCLUSIVE on return (End_Frame in the CSV is inclusive).
    """
    segments: dict[int, list[BehaviorSegment]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Clip"):
                try:
                    clip_id = int(row["Clip"])
                except (ValueError, TypeError):
                    continue
            elif row.get("Video"):
                parsed = _parse_clip_id(row["Video"])
                if parsed is None:
                    continue
                clip_id = parsed
            else:
                continue

            label = (row.get("Label") or row.get("Behavior") or "").strip()
            if not label:
                continue

            try:
                start = int(row["Start_Frame"])
                end_inclusive = int(row["End_Frame"])
            except (KeyError, ValueError):
                continue

            segments.setdefault(clip_id, []).append(BehaviorSegment(
                clip=clip_id, behavior=label, start=start, end=end_inclusive + 1,
            ))

    for segs in segments.values():
        segs.sort(key=lambda s: s.start)
    return segments


def split_segments_by_kind(
    segments: dict[int, list[BehaviorSegment]],
) -> tuple[dict[int, list[BehaviorSegment]], dict[int, list[BehaviorSegment]]]:
    """Split a mixed segments dict into (behavior_segments, direction_segments).

    Segments whose label is in neither ``BEHAVIOR_LABELS`` nor
    ``DIRECTION_LABELS`` are treated as behaviors so legacy CSVs still work
    unchanged.
    """
    behav: dict[int, list[BehaviorSegment]] = {}
    dirs: dict[int, list[BehaviorSegment]] = {}
    for cid, segs in segments.items():
        for s in segs:
            if s.behavior in DIRECTION_LABELS:
                dirs.setdefault(cid, []).append(s)
            else:
                behav.setdefault(cid, []).append(s)
    return behav, dirs


def _count_frames_per_behavior(
    frame_indices: Iterable[int],
    clip_segments: list[BehaviorSegment],
) -> tuple[Counter, int]:
    """Count how many of ``frame_indices`` fall inside each segment.

    Returns (Counter behavior->count, n_assigned_frames). If multiple
    segments cover the same frame, the FIRST (earliest-start) wins. Callers
    that need unambiguous counts must filter segments to a single kind first
    (see ``split_segments_by_kind``).
    """
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


def _count_and_first_frame_per_behavior(
    frame_indices: Iterable[int],
    clip_segments: list[BehaviorSegment],
) -> tuple[Counter, dict[str, int], int]:
    """Same as ``_count_frames_per_behavior`` but also tracks the earliest
    frame at which each behavior is seen. Used for first-appearance ordering
    in the directional labeler."""
    if not clip_segments:
        return Counter(), {}, 0
    counter: Counter = Counter()
    first_frame: dict[str, int] = {}
    frames_arr = np.asarray(list(frame_indices), dtype=int)
    if frames_arr.size == 0:
        return counter, first_frame, 0

    starts = np.array([s.start for s in clip_segments])
    ends = np.array([s.end for s in clip_segments])
    behavior_names = [s.behavior for s in clip_segments]

    assigned = 0
    for f in np.sort(frames_arr):
        mask = (starts <= f) & (ends > f)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            continue
        b = behavior_names[int(idxs[0])]
        counter[b] += 1
        if b not in first_frame:
            first_frame[b] = int(f)
        assigned += 1

    return counter, first_frame, assigned


def _canonical_transition(a: str, b: str) -> str:
    """Legacy canonical transition string ``"behav_a <-> behav_b"`` with
    a <= b alphabetically. Used by the scale-sweep discovery logic."""
    a, b = sorted([a, b])
    return f"{a}{CANONICAL_SEP}{b}"


def _directional_transition(a: str, b: str) -> str:
    """Directional transition string ``"a -> b"`` preserving order."""
    return f"{a}{DIRECTIONAL_SEP}{b}"


def format_directional_label(
    from_b: str, to_b: str | None, direction: str,
) -> str:
    """Render a (from, to, direction) triple as a whitespace-parseable string.

    Examples:
        ("Immobile", "Turn", "Left")   -> "Immobile -> Turn Left"
        ("Immobile", None,   "Left")   -> "Immobile Left"
        ("Immobile", None,   "Straight") -> "Immobile Straight"
    """
    if to_b:
        return f"{from_b}{DIRECTIONAL_SEP}{to_b} {direction}"
    return f"{from_b} {direction}"


def is_transition_label(label: str) -> bool:
    """True if the label encodes a behavior transition (A -> B or A <-> B)."""
    return DIRECTIONAL_SEP in label or CANONICAL_SEP in label


def split_transition_label(label: str) -> tuple[str, str] | None:
    """Return (from, to) if ``label`` is a transition; else None."""
    if DIRECTIONAL_SEP in label:
        a, b = label.split(DIRECTIONAL_SEP, 1)
        return a.strip(), b.strip().split()[0] if b.strip().split() else ""
    if CANONICAL_SEP in label:
        a, b = label.split(CANONICAL_SEP, 1)
        return a.strip(), b.strip()
    return None


# ---------------------------------------------------------------------------
# Directional labeler (PRIMARY, used by benchmark + GUI)
# ---------------------------------------------------------------------------

def label_member_directional(
    frame_indices: np.ndarray,
    behavior_segs: list[BehaviorSegment],
) -> tuple[str, str | None]:
    """Per-member (clip, frame_indices) label under the new rule.

    Returns (from_behavior, to_behavior | None). If the member's frames
    overlap GT segments for only one behavior, ``to_behavior`` is None and
    the cluster will render as a single-behavior label. If no overlap at
    all, returns ("", None).

    Ordering rule: top-2 behaviors by frame count; the one whose earliest
    covered frame occurs first is "from", the other is "to".
    """
    if frame_indices is None or len(frame_indices) == 0 or not behavior_segs:
        return "", None

    counts, first_frame, n_assigned = _count_and_first_frame_per_behavior(
        frame_indices, behavior_segs
    )
    if n_assigned == 0:
        return "", None

    ranked = counts.most_common()
    if len(ranked) == 1:
        return ranked[0][0], None

    a, b = ranked[0][0], ranked[1][0]
    fa, fb = first_frame.get(a, 10**9), first_frame.get(b, 10**9)
    if fa <= fb:
        return a, b
    return b, a


def direction_tag_from_window(
    frames: np.ndarray,
    direction_segs: list[BehaviorSegment],
) -> str | None:
    """Per-WINDOW direction tag (shared rule, used by both cluster samples AND
    GT transition instances — this is the single source of truth).

    Counts Left/Right/Straight frames in the window:
      - Any Left or Right → pick whichever has more frames (ties → Left).
      - Both L and R zero but Straight > 0 → Straight.
      - No direction overlap at all → None (abstain).
    """
    if not direction_segs or frames.size == 0:
        return None
    counts, n = _count_frames_per_behavior(frames, direction_segs)
    if n == 0:
        return None
    l = int(counts.get("Left", 0))
    r = int(counts.get("Right", 0))
    if l == 0 and r == 0:
        return "Straight"
    return "Left" if l >= r else "Right"


def aggregate_direction_counts(
    members: list[tuple[int, np.ndarray]],
    direction_segments_by_clip: dict[int, list[BehaviorSegment]],
) -> Counter:
    """Per-SAMPLE direction votes, tallied into a Counter.

    Each cluster member sample gets ONE direction tag via
    ``direction_tag_from_window`` applied to its own frame window. Samples
    with zero direction overlap abstain. Returned Counter's keys are the
    three direction strings, values are sample-vote counts.
    """
    counter: Counter = Counter()
    for clip_id, frames in members:
        segs = direction_segments_by_clip.get(int(clip_id), [])
        if not segs:
            continue
        tag = direction_tag_from_window(frames, segs)
        if tag is not None:
            counter[tag] += 1
    return counter


def choose_cluster_direction(direction_counts: Counter) -> str:
    """Cluster direction = plain majority vote of per-sample direction tags.

    With the per-sample-vote aggregator above, ``direction_counts`` holds
    "number of samples whose window voted X". The cluster picks the direction
    with the most sample votes (ties broken Left > Right > Straight). When no
    sample voted at all, default to Straight.
    """
    l = int(direction_counts.get("Left", 0))
    r = int(direction_counts.get("Right", 0))
    s = int(direction_counts.get("Straight", 0))
    if l == 0 and r == 0 and s == 0:
        return "Straight"
    return max((l, "Left"), (r, "Right"), (s, "Straight"), key=lambda x: x[0])[1]


def auto_label_clusters_directional(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    behavior_segments: dict[int, list[BehaviorSegment]],
    direction_segments: dict[int, list[BehaviorSegment]],
) -> dict[int, ClusterLabel]:
    """Directional labeler.

    Algorithm:
      1. Per-member: ``(from, to) = label_member_directional(...)`` using
         first-appearance ordering of the top-2 behaviors.
      2. Cluster-level (from, to): majority vote across per-member pairs.
         ``(from, None)`` pairs vote for the single-behavior cluster variant.
      3. Cluster direction: aggregate Left/Right/Straight frame counts
         across ALL members, then apply ``choose_cluster_direction``.
      4. Label string: ``format_directional_label(from, to, direction)``.

    ClusterLabel.vote_counts stores the per-pair votes (including
    single-behavior variants), which is what the GUI and tie-break logic
    consume.
    """
    cluster_labels: dict[int, ClusterLabel] = {}

    for cluster_id, members in sorted(cluster_dict.items()):
        per_clip_pairs: list[tuple[str, str | None]] = []
        per_clip_labels: list[str] = []

        for abs_clip_id, frames in members:
            behav_segs = behavior_segments.get(int(abs_clip_id), [])
            from_b, to_b = label_member_directional(frames, behav_segs)
            per_clip_pairs.append((from_b, to_b))
            per_clip_labels.append(
                format_directional_label(from_b, to_b, "Straight") if from_b else ""
            )

        non_empty_pairs = [p for p in per_clip_pairs if p[0]]
        if not non_empty_pairs:
            cluster_labels[cluster_id] = ClusterLabel(
                primary="",
                alternatives=[],
                per_clip_labels=per_clip_labels,
                vote_counts={},
            )
            continue

        votes = Counter(non_empty_pairs)
        top_count = max(votes.values())
        tied_pairs = sorted(
            [p for p, c in votes.items() if c == top_count],
            key=lambda p: (p[0], p[1] or ""),
        )
        winning_pair = tied_pairs[0]

        direction_counts = aggregate_direction_counts(members, direction_segments)
        direction = choose_cluster_direction(direction_counts)

        primary = format_directional_label(winning_pair[0], winning_pair[1], direction)
        alternatives = [
            format_directional_label(p[0], p[1], direction) for p in tied_pairs[1:]
        ]

        # vote_counts key is the per-pair string WITHOUT the cluster direction
        # suffix — the direction is global to the cluster, so attaching it per
        # pair would falsely attribute the cluster's chosen direction to every
        # voted pair (see audit 2026-04-21).
        serialized_votes: dict[str, int] = {}
        for pair, cnt in votes.items():
            if pair[1]:
                key = f"{pair[0]}{DIRECTIONAL_SEP}{pair[1]}"
            else:
                key = pair[0]
            serialized_votes[key] = serialized_votes.get(key, 0) + int(cnt)

        cluster_labels[cluster_id] = ClusterLabel(
            primary=primary,
            alternatives=alternatives,
            per_clip_labels=per_clip_labels,
            vote_counts=serialized_votes,
        )

    return cluster_labels


# ---------------------------------------------------------------------------
# Legacy canonical labeler (used by scale-sweep)
# ---------------------------------------------------------------------------

def label_member_gt(
    frame_indices: np.ndarray,
    clip_segments: list[BehaviorSegment],
    dominance_threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
    grooming_label: str = GROOMING_LABEL,
) -> str:
    """Canonical per-member labeler. Returns:
      - ``""`` if no GT overlap.
      - ``"Grooming"`` if top behavior is Grooming.
      - Top behavior as a singleton if the second-top fraction is below
        ``dominance_threshold``.
      - ``"behav_a <-> behav_b"`` (alphabetical) otherwise.
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
    """Canonical (alphabetical, direction-agnostic) cluster labeler.

    Consumers: scale-sweep discovery logic. The ``segments`` dict must
    contain behavior segments only; pass the direction segments separately
    if needed. Callers should apply ``split_segments_by_kind`` before
    invoking this function on the new rodent_labels.csv schema, otherwise
    direction labels leak into the "behavior" tally.
    """
    cluster_labels: dict[int, ClusterLabel] = {}

    for cluster_id, members in sorted(cluster_dict.items()):
        per_clip_labels: list[str] = []
        for abs_clip_id, frames in members:
            clip_segs = segments.get(int(abs_clip_id), [])
            per_clip_labels.append(label_member_gt(
                frames, clip_segs,
                dominance_threshold=dominance_threshold,
                grooming_label=grooming_label,
            ))

        non_empty = [l for l in per_clip_labels if l]
        if not non_empty:
            cluster_labels[cluster_id] = ClusterLabel(
                primary="", alternatives=[], per_clip_labels=per_clip_labels,
                vote_counts={},
            )
            continue

        votes = Counter(non_empty)
        top_count = max(votes.values())
        tied_labels = sorted([l for l, c in votes.items() if c == top_count])
        cluster_labels[cluster_id] = ClusterLabel(
            primary=tied_labels[0],
            alternatives=tied_labels[1:] if len(tied_labels) > 1 else [],
            per_clip_labels=per_clip_labels,
            vote_counts=dict(votes),
        )

    return cluster_labels


# ---------------------------------------------------------------------------
# YAML writers
# ---------------------------------------------------------------------------

def _write_labels_yaml(
    output_dir: str,
    method_name: str,
    n_clusters: int,
    seed: int,
    experiment_id: str,
    cluster_labels: dict[int, ClusterLabel],
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    extra_header_lines: list[str] | None = None,
    extra_top_level: dict | None = None,
) -> str:
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

    labels_content: dict = {
        "experiment_id": experiment_id,
        "method": method_name,
        "n_clusters": n_clusters,
        "seed": seed,
        "labels": all_labels,
        "alternatives": alternatives_map,
        "vote_counts": vote_counts_map,
    }
    if extra_top_level:
        labels_content.update(extra_top_level)

    labels_path = Path(output_dir) / f"labels_seed_{seed}.yaml"
    with open(labels_path, "w") as f:
        if extra_header_lines:
            for line in extra_header_lines:
                f.write(f"# {line}\n")
            f.write("\n")
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

    sample_path = Path(output_dir) / f"sample_labels_seed_{seed}.yaml"
    with open(sample_path, "w") as f:
        yaml.dump({
            "experiment_id": experiment_id, "method": method_name, "seed": seed,
            "clusters": sample_detail,
        }, f, default_flow_style=False, sort_keys=False)

    return str(labels_path)


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
    mode: str = "directional",
) -> str:
    """Auto-label every cluster and write ``labels_seed_{seed}.yaml`` +
    ``sample_labels_seed_{seed}.yaml`` in ``output_dir``.

    ``mode``:
      - ``"directional"`` (default, new): ``"A -> B Direction"`` / ``"A Direction"``
      - ``"canonical"`` (legacy): ``"A <-> B"`` / ``"A"``, respects
        ``dominance_threshold`` and ``grooming_label``.
    """
    if experiment_id is None:
        from datetime import datetime
        experiment_id = f"experiment_{datetime.now().strftime('%Y-%m-%d')}"

    segments = load_behavior_segments(behavior_labels_csv)

    if mode == "directional":
        behav_segs, dir_segs = split_segments_by_kind(segments)
        cluster_labels = auto_label_clusters_directional(
            cluster_dict, behav_segs, dir_segs,
        )
        header = [
            "AUTO-GENERATED (directional mode) from ground truth behavior + direction labels.",
            "Label format: 'A -> B Direction' (transition) or 'A Direction' (single).",
            "Tied (from, to) pairs are listed under `alternatives` with the cluster's chosen direction appended.",
        ]
        extra = {"mode": "directional"}
    elif mode == "canonical":
        behav_segs, _ = split_segments_by_kind(segments)
        cluster_labels = auto_label_clusters(
            cluster_dict, behav_segs,
            dominance_threshold=dominance_threshold,
            grooming_label=grooming_label,
        )
        header = [
            "AUTO-GENERATED (canonical mode) from ground truth behavior labels (GT majority vote).",
            "Label format: 'A <-> B' (alphabetical) or 'A' (singleton). Direction labels ignored.",
            "Tied clusters expose their alternatives under `alternatives`.",
        ]
        extra = {
            "mode": "canonical",
            "dominance_threshold": float(dominance_threshold),
        }
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'directional' or 'canonical'.")

    return _write_labels_yaml(
        output_dir=output_dir, method_name=method_name, n_clusters=n_clusters,
        seed=seed, experiment_id=experiment_id,
        cluster_labels=cluster_labels, cluster_dict=cluster_dict,
        extra_header_lines=header, extra_top_level=extra,
    )
