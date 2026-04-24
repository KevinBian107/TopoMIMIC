"""State-connection discovery: which canonical transitions does each variant
recover by directly connecting two pure-state clusters?

Complementary to single-state `discovery.py` — that one asks "is there a cluster
auto-labeled as transition T?". This one asks "are there two pure-state clusters
A and B, with primary labels behav_a and behav_b, such that the per-frame
sequence has A→B (or B→A) consecutive in at least N clips?"

A cluster is "pure-state" if its auto-label primary is a singleton (no '<->').
Transition-labeled clusters are excluded as their own state — they are the
single-state discovery channel.

Outputs (alongside discovery/):
  - discovery_connections_long.csv   # primary (transition, variant) long format
  - discovery_connections_binary_wide.csv
  - discovery_connections_counts_wide.csv
  - discovery_connections_pairs.csv  # per (variant, state_pair) detail for diagnosis
"""

from __future__ import annotations

import logging
import pickle
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from topo_mimic.experiment.auto_label import (
    BehaviorSegment,
    ClusterLabel,
    auto_label_clusters,
)
from topo_mimic.analysis.scale_sweep.config import DiscoveryCfg
from topo_mimic.experiment.auto_label import _canonical_transition

log = logging.getLogger(__name__)


def _is_singleton_label(label: str) -> bool:
    return bool(label) and " <-> " not in label


def _extract_state_pair_occurrences(
    labels_list: list[np.ndarray],
    valid_cluster_ids: set[int],
    clip_ids: list[int],
) -> dict[tuple[int, int], set[int]]:
    """Walk each clip's per-frame label sequence and collect (A, B) consecutive
    pairs of distinct states (both in valid_cluster_ids).

    Returns: {(A, B) -> set of clip_ids}. Pair tuples are stored UNORDERED
    (smaller cluster_id first) so A->B and B->A aggregate together — matching
    the canonical-transition convention.
    """
    pair_clips: dict[tuple[int, int], set[int]] = defaultdict(set)
    for i, per_clip in enumerate(labels_list):
        clip_id = clip_ids[i]
        seq = np.asarray(per_clip)
        if seq.size < 2:
            continue
        valid_mask = np.array([int(s) in valid_cluster_ids for s in seq])
        prev_state = None
        for f, s in enumerate(seq):
            if not valid_mask[f]:
                prev_state = None
                continue
            cur = int(s)
            if prev_state is not None and prev_state != cur:
                key = tuple(sorted((prev_state, cur)))
                pair_clips[key].add(clip_id)
            prev_state = cur
    return dict(pair_clips)


@dataclass
class StatePairCandidate:
    state_a: int
    state_b: int
    behav_a: str
    behav_b: str
    canonical_transition: str
    n_clips_observed: int
    passes_threshold: bool


@dataclass
class ConnectionDiscoveryRow:
    transition: str
    is_singleton: bool
    method: str
    feature_space: str
    variant_id: str
    kappa: float | None
    num_states: int | None
    lens: str | None
    n_intervals: int | None
    discovered: int
    n_state_pairs_matching: int
    n_state_pairs_passing: int
    best_pair_n_clips: int
    n_pure_state_clusters: int
    n_transition_state_clusters: int


def _evaluate_variant_connections(
    variant_meta: dict[str, Any],
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    cluster_labels: dict[int, ClusterLabel],
    labels_list: list[np.ndarray],
    clip_ids: list[int],
    transitions: list[str],
    min_clip_occurrences: int,
) -> tuple[list[ConnectionDiscoveryRow], list[StatePairCandidate]]:
    pure_clusters = {
        cid: lbl.primary for cid, lbl in cluster_labels.items()
        if _is_singleton_label(lbl.primary)
    }
    n_pure = len(pure_clusters)
    n_transition_state = sum(
        1 for lbl in cluster_labels.values() if " <-> " in lbl.primary
    )

    pair_clips = _extract_state_pair_occurrences(
        labels_list=labels_list,
        valid_cluster_ids=set(pure_clusters.keys()),
        clip_ids=clip_ids,
    )

    candidates: list[StatePairCandidate] = []
    for (a, b), clips in pair_clips.items():
        if a == b:
            continue
        behav_a = pure_clusters[a]
        behav_b = pure_clusters[b]
        if behav_a == behav_b:
            continue
        candidates.append(StatePairCandidate(
            state_a=a, state_b=b,
            behav_a=behav_a, behav_b=behav_b,
            canonical_transition=_canonical_transition(behav_a, behav_b),
            n_clips_observed=len(clips),
            passes_threshold=len(clips) >= min_clip_occurrences,
        ))

    rows: list[ConnectionDiscoveryRow] = []
    for T in transitions:
        matching = [c for c in candidates if c.canonical_transition == T]
        passing = [c for c in matching if c.passes_threshold]
        best_n_clips = max((c.n_clips_observed for c in passing), default=0)
        rows.append(ConnectionDiscoveryRow(
            transition=T,
            is_singleton=(" <-> " not in T),
            method=variant_meta["method"],
            feature_space=variant_meta.get("feature_space", ""),
            variant_id=variant_meta["variant_id"],
            kappa=variant_meta.get("kappa"),
            num_states=variant_meta.get("num_states"),
            lens=variant_meta.get("lens"),
            n_intervals=variant_meta.get("n_intervals"),
            discovered=1 if passing else 0,
            n_state_pairs_matching=len(matching),
            n_state_pairs_passing=len(passing),
            best_pair_n_clips=best_n_clips,
            n_pure_state_clusters=n_pure,
            n_transition_state_clusters=n_transition_state,
        ))

    return rows, candidates


def build_connection_discovery_tables(
    transitions: list[str],
    transition_counts: dict[str, int],
    variant_metas: list[dict[str, Any]],
    segments: dict[int, list[BehaviorSegment]],
    clip_ids: list[int],
    cfg: DiscoveryCfg,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    long_rows: list[ConnectionDiscoveryRow] = []
    pair_rows: list[dict[str, Any]] = []

    for meta in variant_metas:
        cluster_dict_path = meta["cluster_dict_path"]
        labels_path = Path(cluster_dict_path).parent / "labels_list.pkl"
        if not labels_path.exists():
            log.warning(
                f"[connection_discovery] missing labels_list.pkl for "
                f"{meta['variant_id']}; skipping (run phase 5 to backfill)"
            )
            continue
        with Path(cluster_dict_path).open("rb") as f:
            cluster_dict = pickle.load(f)
        with labels_path.open("rb") as f:
            labels_list = pickle.load(f)

        cluster_labels = auto_label_clusters(
            cluster_dict, segments,
            dominance_threshold=cfg.second_behavior_threshold,
        )

        rows, candidates = _evaluate_variant_connections(
            variant_meta=meta,
            cluster_dict=cluster_dict,
            cluster_labels=cluster_labels,
            labels_list=labels_list,
            clip_ids=clip_ids,
            transitions=transitions,
            min_clip_occurrences=cfg.min_member_clips,
        )
        long_rows.extend(rows)
        for c in candidates:
            pair_rows.append({
                "variant_id": meta["variant_id"],
                "method": meta["method"],
                "feature_space": meta.get("feature_space", ""),
                "state_a": c.state_a, "state_b": c.state_b,
                "behav_a": c.behav_a, "behav_b": c.behav_b,
                "canonical_transition": c.canonical_transition,
                "n_clips_observed": c.n_clips_observed,
                "passes_threshold": int(c.passes_threshold),
            })

    long_df = pd.DataFrame([asdict(r) for r in long_rows])
    long_df.to_csv(output_dir / "discovery_connections_long.csv", index=False)
    log.info(
        f"[connection_discovery] wrote discovery_connections_long.csv ({len(long_df)} rows)"
    )

    if not long_df.empty:
        binary_wide = long_df.pivot(
            index="transition", columns="variant_id", values="discovered"
        ).fillna(0).astype(int)
        counts_wide = long_df.pivot(
            index="transition", columns="variant_id", values="n_state_pairs_passing"
        ).fillna(0).astype(int)
        binary_wide.to_csv(output_dir / "discovery_connections_binary_wide.csv")
        counts_wide.to_csv(output_dir / "discovery_connections_counts_wide.csv")
        log.info(
            f"[connection_discovery] wrote wide CSVs "
            f"({binary_wide.shape[0]} transitions × {binary_wide.shape[1]} variants)"
        )

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(output_dir / "discovery_connections_pairs.csv", index=False)
    log.info(
        f"[connection_discovery] wrote discovery_connections_pairs.csv ({len(pair_df)} rows)"
    )
    return long_df
