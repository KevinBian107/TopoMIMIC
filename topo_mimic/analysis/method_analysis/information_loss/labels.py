"""Per-frame transition-label construction for the information-loss experiment.

Y_t = directed transition class (or background) for frame t of clip c. We
reuse ``feature_analysis.ground_truth.find_all_transitions`` as the single
source of truth for transition windows so that this experiment never disagrees
with the rest of the paper about what counts as a transition.

Label space (default ``directed``):
    0 = background (no transition window covers this frame)
    1..K = one class per directed (from, to) pair retained after frequency
           filter, ordered by descending frequency for stable indices.

Two other granularities are available for ablation but are not the default:
    ``unordered``     -- collapse (A->B) and (B->A) onto one class
    ``directed_turn`` -- add window-level turn direction (Left/Right/Straight)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np

from topo_mimic.analysis.feature_analysis.ground_truth import (
    TransitionSegment,
    find_all_transitions,
)


Granularity = Literal["unordered", "directed", "directed_turn"]


@dataclass(frozen=True)
class TransitionLabels:
    """Per-frame ground-truth transition class.

    Y[c, t] = 0 if no transition window covers frame t of clip c, else the
    class index assigned by ``class_names``. Frames in clips with no GT labels
    at all are tagged with ``-1`` and must be dropped before MI estimation.
    """

    Y: np.ndarray  # (n_clips, frames_per_clip) int. 0 = background, -1 = no-GT.
    class_names: list[str]  # length = K+1, index 0 is "background"
    granularity: Granularity
    labeled_clip_ids: list[int]
    n_transitions_per_class: dict[str, int]
    csv_path: str

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def restrict_to_labeled(self) -> tuple[np.ndarray, np.ndarray]:
        """Flatten and drop unlabeled frames.

        Returns:
            Y_flat: (n_labeled_frames,) int array of Y values.
            mask:   (n_clips, frames_per_clip) bool array, True where labeled.
        """
        mask = self.Y >= 0
        return self.Y[mask], mask


def _segment_key(seg: TransitionSegment, granularity: Granularity) -> str:
    if granularity == "unordered":
        a, b = sorted([seg.from_behavior, seg.to_behavior])
        return f"{a}-{b}"
    if granularity == "directed":
        return f"{seg.from_behavior}->{seg.to_behavior}"
    if granularity == "directed_turn":
        return f"{seg.from_behavior}->{seg.to_behavior} [{seg.turn.value}]"
    raise ValueError(f"Unknown granularity: {granularity!r}")


def load_transition_labels(
    csv_path: str,
    n_clips: int,
    frames_per_clip: int,
    granularity: Granularity = "directed",
    min_class_count: int = 5,
    short_threshold: int = 10,
    max_gap: int = 10,
    merge_rear: bool = True,
    window_frames: int = 100,
) -> TransitionLabels:
    """Build the per-frame transition label array.

    A frame is in class ``k`` iff it lies inside the from-or-to window of a
    TransitionSegment whose canonical key maps to class ``k``. Frames not in
    any retained transition window are class ``0`` (background). Frames in
    clips absent from the CSV are ``-1``.
    """
    segments = find_all_transitions(
        csv_path=csv_path,
        short_threshold=short_threshold,
        max_gap=max_gap,
        merge_rear=merge_rear,
        window_frames=window_frames,
    )

    key_counts = Counter(_segment_key(s, granularity) for s in segments)
    retained_keys = [k for k, n in key_counts.most_common() if n >= min_class_count]
    key_to_class = {k: i + 1 for i, k in enumerate(retained_keys)}
    class_names = ["background"] + retained_keys

    Y = np.full((n_clips, frames_per_clip), -1, dtype=np.int32)
    labeled_clip_ids: set[int] = {seg.clip_id for seg in segments}

    for clip_id in labeled_clip_ids:
        if 0 <= clip_id < n_clips:
            Y[clip_id] = 0

    for seg in segments:
        clip_id = seg.clip_id
        if not (0 <= clip_id < n_clips):
            continue
        key = _segment_key(seg, granularity)
        cls = key_to_class.get(key)
        if cls is None:
            continue

        from_lo = max(0, seg.from_start)
        from_hi = min(frames_per_clip, seg.from_end + 1)
        to_lo = max(0, seg.to_start)
        to_hi = min(frames_per_clip, seg.to_end + 1)

        if from_hi > from_lo:
            Y[clip_id, from_lo:from_hi] = cls
        if to_hi > to_lo:
            Y[clip_id, to_lo:to_hi] = cls

    counts = {name: 0 for name in class_names}
    flat = Y.reshape(-1)
    for i, name in enumerate(class_names):
        counts[name] = int(np.sum(flat == i))

    return TransitionLabels(
        Y=Y,
        class_names=class_names,
        granularity=granularity,
        labeled_clip_ids=sorted(labeled_clip_ids),
        n_transitions_per_class=counts,
        csv_path=csv_path,
    )


def summarize_labels(labels: TransitionLabels) -> str:
    """Human-readable summary for sanity-check logging."""
    Y_flat, _ = labels.restrict_to_labeled()
    n_total = Y_flat.size
    lines = [
        f"granularity: {labels.granularity}",
        f"n_classes:   {labels.n_classes} (including background)",
        f"n_labeled_clips:  {len(labels.labeled_clip_ids)}",
        f"n_labeled_frames: {n_total}",
        "",
        "class distribution (over labeled frames):",
    ]
    for i, name in enumerate(labels.class_names):
        n = labels.n_transitions_per_class[name]
        frac = (n / n_total) if n_total else 0.0
        lines.append(f"  [{i:2d}] {name:<40s}  n={n:6d}  ({frac:.2%})")
    return "\n".join(lines)
