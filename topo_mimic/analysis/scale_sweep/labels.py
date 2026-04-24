"""Enumerate real canonical transitions from rodent.csv for the configured clip range.

The resulting list is FROZEN before any clustering method runs, so the discovery-table
rows are method-independent.

The vocabulary is the exact set of labels the cluster auto-labeler could emit (pairs +
singletons + Grooming), drawn only from transitions/behaviors that actually occur in
the labeled ground-truth data for the configured clip range.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import yaml

from topo_mimic.experiment.auto_label import (
    BehaviorSegment,
    GROOMING_LABEL,
    _canonical_transition,
)

log = logging.getLogger(__name__)


DIRECTED_SEP = " -> "


def directed_transition(a: str, b: str) -> str:
    """Directed transition string ``"a -> b"``."""
    return f"{a}{DIRECTED_SEP}{b}"


def directed_to_canonical(directed: str) -> str:
    """Convert a directed transition row label back to its canonical (sorted) form
    so it can be matched against `auto_label_clusters` cluster labels.

    For singletons (no DIRECTED_SEP, e.g. 'Grooming'), returns the input unchanged.
    """
    if DIRECTED_SEP not in directed:
        return directed
    a, b = directed.split(DIRECTED_SEP, 1)
    return _canonical_transition(a, b)


def enumerate_real_transitions(
    segments_by_clip: dict[int, list[BehaviorSegment]],
    clip_ids: list[int],
    groom_singleton: bool = True,
    include_other_singletons: bool = False,
    directed: bool = False,
) -> tuple[list[str], dict[str, int]]:
    """Enumerate the transition vocabulary for the clips in ``clip_ids``.

    Defaults to directed transitions (both ``a -> b`` and ``b -> a`` rows for any
    pair (a, b) that occurs in either direction across clips), plus the Grooming
    singleton when present. Other behavior singletons are excluded by default —
    set ``include_other_singletons=True`` to keep them.

    Returns (sorted_labels, occurrence_counts).

    occurrence_counts semantics:
      - directed transition: total directed-pair occurrences in that direction
      - singleton: # clips containing the behavior
    """
    directed_counts: Counter[str] = Counter()
    singleton_counts: Counter[str] = Counter()
    groom_clip_count = 0
    pairs_seen: set[tuple[str, str]] = set()

    for clip_id in clip_ids:
        segs = segments_by_clip.get(clip_id, [])
        if not segs:
            continue

        behaviors_in_clip: set[str] = set()
        for s in segs:
            behaviors_in_clip.add(s.behavior)

        segs_sorted = sorted(segs, key=lambda s: s.start)
        for a, b in zip(segs_sorted[:-1], segs_sorted[1:]):
            if a.behavior == b.behavior:
                continue
            if directed:
                directed_counts[directed_transition(a.behavior, b.behavior)] += 1
                pairs_seen.add(tuple(sorted((a.behavior, b.behavior))))
            else:
                directed_counts[_canonical_transition(a.behavior, b.behavior)] += 1

        if GROOMING_LABEL in behaviors_in_clip:
            groom_clip_count += 1

        if include_other_singletons:
            for behav in behaviors_in_clip:
                if behav == GROOMING_LABEL:
                    continue
                singleton_counts[behav] += 1

    labels: dict[str, int] = {}
    labels.update(directed_counts)

    # In directed mode, ensure both directions of every observed pair appear in
    # the vocabulary (even if one direction never occurred in data) so the table
    # is symmetric and easy to read.
    if directed:
        for a, b in pairs_seen:
            for src, dst in ((a, b), (b, a)):
                key = directed_transition(src, dst)
                if key not in labels:
                    labels[key] = 0

    if groom_singleton and groom_clip_count > 0:
        labels[GROOMING_LABEL] = groom_clip_count
    if include_other_singletons:
        labels.update(singleton_counts)

    sorted_labels = sorted(labels.keys())
    log.info(
        f"Enumerated {len(sorted_labels)} real transitions: "
        f"{sum(1 for k in labels if DIRECTED_SEP in k)} directed transitions, "
        f"{1 if GROOMING_LABEL in labels else 0} Grooming singleton, "
        f"{sum(1 for k in singleton_counts)} other singletons"
    )
    return sorted_labels, dict(labels)


def write_real_transitions_yaml(
    out_path: Path,
    labels: list[str],
    occurrence_counts: dict[str, int],
    clip_range: tuple[int, int],
    n_clips_used: int,
) -> None:
    """Persist the frozen row list with provenance."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clip_range": list(clip_range),
        "n_clips_used": n_clips_used,
        "n_transitions": len(labels),
        "transitions": [
            {"label": lbl, "occurrence_count": int(occurrence_counts.get(lbl, 0))}
            for lbl in labels
        ],
    }
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    log.info(f"Wrote real_transitions.yaml to {out_path}")


def load_real_transitions_yaml(path: Path) -> tuple[list[str], dict[str, int]]:
    payload = yaml.safe_load(path.read_text())
    labels = [row["label"] for row in payload["transitions"]]
    counts = {row["label"]: int(row["occurrence_count"]) for row in payload["transitions"]}
    return labels, counts
