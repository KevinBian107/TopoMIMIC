"""Discovery table: which real transitions does each variant recover under thresholds.

Emits:
- discovery_long.csv — primary artifact, one row per (transition, variant).
- discovery_binary_wide.csv — rows=transitions, columns=variants, {0,1}.
- discovery_counts_wide.csv — rows=transitions, columns=variants, integer cluster_count_passing.
- transition_metadata.csv — per-transition occurrence counts + singleton flag.
- variant_metadata.yaml — per-variant hyperparameters + fit metrics.

Discovery rule (for transition T and variant V):
  V 'discovers' T if at least one cluster C in V satisfies ALL of:
    (1) auto_label(C).primary == T
    (2) cluster has >= min_member_clips UNIQUE clips
    (3) cluster's majority fraction >= min_majority_fraction
    (4) cluster has >= min_total_frames total frames across all members
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from topo_mimic.experiment.auto_label import (
    BehaviorSegment,
    ClusterLabel,
    auto_label_clusters,
)
from topo_mimic.analysis.scale_sweep.labels import (
    DIRECTED_SEP,
    directed_to_canonical,
)
from topo_mimic.analysis.scale_sweep.config import DiscoveryCfg
from topo_mimic.analysis.scale_sweep.kpms_scale import KPMSVariantResult
from topo_mimic.analysis.scale_sweep.mapper_scale import MapperVariantResult

log = logging.getLogger(__name__)


@dataclass
class DiscoveryThresholds:
    min_member_clips: int
    min_majority_fraction: float
    min_total_frames: int
    second_behavior_threshold: float


def _load_cluster_dict(path: str | Path) -> dict[int, list[tuple[int, np.ndarray]]]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _clip_vote_majority_fraction(label: ClusterLabel) -> float:
    total = sum(label.vote_counts.values())
    if total == 0:
        return 0.0
    return label.vote_counts.get(label.primary, 0) / total


def _frame_purity(
    members: list[tuple[int, np.ndarray]],
    primary_label: str,
    segments: dict[int, list[BehaviorSegment]],
) -> tuple[float, int]:
    """Fraction of cluster frames whose GT behavior is consistent with the primary label.

    For a canonical transition "A <-> B", consistent = GT is A or B.
    For a singleton "X", consistent = GT is X.
    Frames with no GT coverage are excluded from both numerator and denominator.

    Returns (purity, n_frames_with_gt).
    """
    if " <-> " in primary_label:
        a, b = primary_label.split(" <-> ", 1)
        accepted = {a, b}
    else:
        accepted = {primary_label}

    total_frames_with_gt = 0
    matching_frames = 0
    for clip_id, frames in members:
        clip_segs = segments.get(int(clip_id), [])
        if not clip_segs:
            continue
        starts = np.array([s.start for s in clip_segs])
        ends = np.array([s.end for s in clip_segs])
        for f in np.asarray(frames):
            mask = (starts <= f) & (ends > f)
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                continue
            total_frames_with_gt += 1
            if clip_segs[int(idxs[0])].behavior in accepted:
                matching_frames += 1
    if total_frames_with_gt == 0:
        return 0.0, 0
    return matching_frames / total_frames_with_gt, total_frames_with_gt


def _cluster_summary(
    cluster_id: int,
    members: list[tuple[int, np.ndarray]],
    label: ClusterLabel,
    segments: dict[int, list[BehaviorSegment]],
) -> dict[str, Any]:
    unique_clips = {int(cid) for cid, _ in members}
    total_frames = sum(int(frames.size) for _, frames in members)
    clip_vote_fraction = _clip_vote_majority_fraction(label)
    if label.primary:
        frame_purity, n_gt_frames = _frame_purity(members, label.primary, segments)
    else:
        frame_purity, n_gt_frames = 0.0, 0
    return {
        "cluster_id": int(cluster_id),
        "primary_label": label.primary,
        "alternatives": list(label.alternatives),
        "n_member_clips": len(unique_clips),
        "total_frames": int(total_frames),
        "clip_vote_majority_fraction": float(clip_vote_fraction),
        "frame_purity": float(frame_purity),
        "n_gt_frames": int(n_gt_frames),
        "vote_counts": dict(label.vote_counts),
    }


def _passes_thresholds(summary: dict[str, Any], th: DiscoveryThresholds) -> bool:
    """Threshold check uses FRAME PURITY (frames whose GT is consistent with the
    cluster's primary label), not clip-vote majority. Clip-vote determines WHICH
    label the cluster gets (via auto_label); frame purity determines WHETHER the
    cluster is a clean discovery."""
    return (
        summary["n_member_clips"] >= th.min_member_clips
        and summary["frame_purity"] >= th.min_majority_fraction
        and summary["total_frames"] >= th.min_total_frames
    )


@dataclass
class VariantDiscoveryRow:
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
    cluster_count_matching: int
    cluster_count_passing: int
    best_cluster_id: int
    best_frame_purity: float
    best_clip_vote_majority_fraction: float
    best_n_member_clips: int
    best_total_frames: int
    total_clusters_in_variant: int
    variant_passing_clusters: int


def _variant_cluster_summaries(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    segments: dict[int, list[BehaviorSegment]],
    second_behavior_threshold: float,
) -> list[dict[str, Any]]:
    labels = auto_label_clusters(
        cluster_dict,
        segments,
        dominance_threshold=second_behavior_threshold,
    )
    out: list[dict[str, Any]] = []
    for cid, members in cluster_dict.items():
        out.append(_cluster_summary(cid, members, labels[cid], segments))
    return out


def _evaluate_variant(
    variant_meta: dict[str, Any],
    cluster_summaries: list[dict[str, Any]],
    transitions: list[str],
    th: DiscoveryThresholds,
) -> list[VariantDiscoveryRow]:
    total_clusters = len(cluster_summaries)
    variant_passing = sum(1 for s in cluster_summaries if _passes_thresholds(s, th))

    rows: list[VariantDiscoveryRow] = []
    for T in transitions:
        # Cluster labels from auto_label_clusters are canonical (sorted with
        # " <-> "). When the row T is a directed transition like "A -> B", a
        # cluster with primary == "A <-> B" (canonical) matches both "A -> B"
        # and "B -> A" directional rows.
        T_canonical = directed_to_canonical(T)
        matching = [s for s in cluster_summaries if s["primary_label"] == T_canonical]
        passing = [s for s in matching if _passes_thresholds(s, th)]

        if passing:
            best = max(passing, key=lambda s: (s["frame_purity"], s["total_frames"]))
        else:
            best = {
                "cluster_id": -1,
                "frame_purity": 0.0,
                "clip_vote_majority_fraction": 0.0,
                "n_member_clips": 0,
                "total_frames": 0,
            }

        rows.append(VariantDiscoveryRow(
            transition=T,
            is_singleton=(" <-> " not in T and DIRECTED_SEP not in T),
            method=variant_meta["method"],
            feature_space=variant_meta.get("feature_space", ""),
            variant_id=variant_meta["variant_id"],
            kappa=variant_meta.get("kappa"),
            num_states=variant_meta.get("num_states"),
            lens=variant_meta.get("lens"),
            n_intervals=variant_meta.get("n_intervals"),
            discovered=1 if passing else 0,
            cluster_count_matching=len(matching),
            cluster_count_passing=len(passing),
            best_cluster_id=int(best["cluster_id"]),
            best_frame_purity=float(best["frame_purity"]),
            best_clip_vote_majority_fraction=float(best["clip_vote_majority_fraction"]),
            best_n_member_clips=int(best["n_member_clips"]),
            best_total_frames=int(best["total_frames"]),
            total_clusters_in_variant=total_clusters,
            variant_passing_clusters=variant_passing,
        ))
    return rows


def assemble_variant_metas(
    kpms_results: list,
    kpms_selections: dict,
    mapper_results: list,
) -> list[dict[str, Any]]:
    """Assemble the per-variant metadata records used by both the single-state
    and the connection-based discovery tables."""
    out: list[dict[str, Any]] = []
    for vr in kpms_results:
        m = _kpms_variant_meta(vr, kpms_selections)
        if m is not None:
            out.append(m)
    for vr in mapper_results:
        out.append(_mapper_variant_meta(vr))
    return out


def _kpms_variant_meta(
    vr: KPMSVariantResult,
    kpms_selections: dict[float, Any],
) -> dict[str, Any] | None:
    """Only include KPMS variants that were SELECTED as best-per-kappa."""
    sel = kpms_selections.get(vr.kappa)
    if sel is None or sel.selected_variant_id != vr.variant_id:
        return None
    return {
        "variant_id": vr.variant_id,
        "method": "kpms",
        "feature_space": "keypoint",
        "kappa": float(vr.kappa),
        "num_states": int(vr.num_states),
        "lens": None,
        "n_intervals": None,
        "cluster_dict_path": vr.cluster_dict_path,
        "eml": float(vr.eml),
        "usage_ratio": float(vr.usage_ratio),
        "active_syllables": int(vr.active_syllables),
        "duration_mean": float(vr.duration_mean),
        "duration_median": float(vr.duration_median),
        "transition_entropy": float(vr.transition_entropy),
        "selector_used": sel.selector_used,
    }


def _mapper_variant_meta(vr: MapperVariantResult) -> dict[str, Any]:
    return {
        "variant_id": vr.variant_id,
        "method": "mapper",
        "feature_space": vr.feature_space,
        "kappa": None,
        "num_states": None,
        "lens": vr.lens,
        "n_intervals": int(vr.n_intervals),
        "cluster_dict_path": vr.cluster_dict_path,
        "n_nodes": int(vr.n_nodes),
        "n_edges": int(vr.n_edges),
        "n_clusters": int(vr.n_clusters),
        "n_noise_frames": int(vr.n_noise_frames),
        "dbscan_eps": float(vr.dbscan_eps),
    }


def build_discovery_tables(
    transitions: list[str],
    transition_counts: dict[str, int],
    kpms_results: list[KPMSVariantResult],
    kpms_selections: dict[float, Any],
    mapper_results: list[MapperVariantResult],
    segments: dict[int, list[BehaviorSegment]],
    cfg: DiscoveryCfg,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    th = DiscoveryThresholds(
        min_member_clips=cfg.min_member_clips,
        min_majority_fraction=cfg.min_majority_fraction,
        min_total_frames=cfg.min_total_frames,
        second_behavior_threshold=cfg.second_behavior_threshold,
    )

    variant_metas: list[dict[str, Any]] = []
    for vr in kpms_results:
        m = _kpms_variant_meta(vr, kpms_selections)
        if m is not None:
            variant_metas.append(m)
    for vr in mapper_results:
        variant_metas.append(_mapper_variant_meta(vr))

    log.info(f"[discovery] evaluating {len(variant_metas)} variants × {len(transitions)} transitions")

    long_rows: list[VariantDiscoveryRow] = []
    for meta in variant_metas:
        cluster_dict = _load_cluster_dict(meta["cluster_dict_path"])
        cluster_summaries = _variant_cluster_summaries(
            cluster_dict, segments, th.second_behavior_threshold
        )
        long_rows.extend(_evaluate_variant(meta, cluster_summaries, transitions, th))

    long_df = pd.DataFrame([asdict(r) for r in long_rows])
    long_df.to_csv(output_dir / "discovery_long.csv", index=False)
    log.info(f"[discovery] wrote discovery_long.csv ({len(long_df)} rows)")

    binary_wide = long_df.pivot(index="transition", columns="variant_id", values="discovered").fillna(0).astype(int)
    counts_wide = long_df.pivot(index="transition", columns="variant_id", values="cluster_count_passing").fillna(0).astype(int)
    binary_wide.to_csv(output_dir / "discovery_binary_wide.csv")
    counts_wide.to_csv(output_dir / "discovery_counts_wide.csv")
    log.info(f"[discovery] wrote wide CSVs ({binary_wide.shape[0]} transitions × {binary_wide.shape[1]} variants)")

    trans_meta = pd.DataFrame([
        {
            "transition": T,
            "occurrence_count": int(transition_counts.get(T, 0)),
            "is_singleton": " <-> " not in T,
            "discovery_rate": float(binary_wide.loc[T].mean()) if T in binary_wide.index else 0.0,
        }
        for T in transitions
    ])
    trans_meta.to_csv(output_dir / "transition_metadata.csv", index=False)
    log.info(f"[discovery] wrote transition_metadata.csv")

    variant_meta_out: dict[str, Any] = {}
    for meta in variant_metas:
        key = meta["variant_id"]
        variant_meta_out[key] = {k: v for k, v in meta.items() if k != "cluster_dict_path"}
    (output_dir / "variant_metadata.yaml").write_text(yaml.safe_dump(variant_meta_out, sort_keys=False))
    log.info(f"[discovery] wrote variant_metadata.yaml ({len(variant_meta_out)} variants)")

    return long_df
