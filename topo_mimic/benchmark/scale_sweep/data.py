"""Scale-sweep data loading with explicit clip-range assertion.

Provides a single `load_scale_sweep_data(cfg)` entry that returns all features needed
downstream: intention, qpos, FK keypoints, rodent.csv segments, and the canonical
clip_id list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from topo_mimic.benchmark.auto_label import BehaviorSegment, load_behavior_segments
from topo_mimic.benchmark.hyperparam_sweep.keypoint_loader import (
    qpos_to_keypoints,
    setup_stac_reference,
)
from topo_mimic.benchmark.scale_sweep.config import ScaleSweepConfig
from topo_mimic.dmwk.utils.dataset_config import get_dataset_preset
from topo_mimic.dmwk.utils.load_utils import get_aggregate_data

log = logging.getLogger(__name__)


@dataclass
class ScaleSweepData:
    clip_ids: list[int]
    intention: np.ndarray                              # (N, T, D_int)
    qpos: np.ndarray                                   # (N, T, D_qpos)
    keypoints: np.ndarray                              # (N, T, K, 3)
    kp_names: list[str]
    segments_by_clip: dict[int, list[BehaviorSegment]]

    @property
    def n_clips(self) -> int:
        return len(self.clip_ids)

    @property
    def n_frames(self) -> int:
        return self.intention.shape[1]


def _resolve_clip_ids(cfg: ScaleSweepConfig) -> list[int]:
    preset = get_dataset_preset(cfg.data.dataset)
    start, end = cfg.data.start_clip, cfg.data.end_clip
    if end > preset.total_clips:
        raise ValueError(
            f"end_clip={end} exceeds dataset total_clips={preset.total_clips} for {cfg.data.dataset!r}"
        )
    all_ids = list(range(start, end))
    if cfg.data.respect_skip_clips and preset.skip_clips:
        skipped = [c for c in all_ids if c in preset.skip_clips]
        if skipped:
            log.info(f"Skipping {len(skipped)} clips from skip_clips preset: {skipped}")
        all_ids = [c for c in all_ids if c not in preset.skip_clips]
    return all_ids


def _filter_to_labeled_clips(
    segments_by_clip: dict[int, list[BehaviorSegment]],
    clip_ids: list[int],
) -> list[int]:
    """Drop clips with no rodent.csv labels (with a logged warning)."""
    missing = [c for c in clip_ids if c not in segments_by_clip or not segments_by_clip[c]]
    if missing:
        log.warning(
            f"rodent.csv missing labels for {len(missing)} clips in configured range: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}. Dropping them from the sweep."
        )
    return [c for c in clip_ids if c in segments_by_clip and segments_by_clip[c]]


def _load_single_key(path: str, group: str, keys: list[str], clip_ids: list[int]) -> np.ndarray:
    arrays = [get_aggregate_data(group, keys, ci, path) for ci in clip_ids]
    return np.stack(arrays, axis=0)


def load_scale_sweep_data(cfg: ScaleSweepConfig) -> ScaleSweepData:
    """Load intention, qpos, FK-derived keypoints, and GT segments for the configured clip range.

    Raises if rodent.csv does not cover every clip in the range (excluding skip_clips).
    """
    clip_ids = _resolve_clip_ids(cfg)
    log.info(f"Loading {len(clip_ids)} clips in range [{cfg.data.start_clip}, {cfg.data.end_clip})")

    segments_by_clip = load_behavior_segments(cfg.data.behavior_labels_csv)
    clip_ids = _filter_to_labeled_clips(segments_by_clip, clip_ids)
    if not clip_ids:
        raise RuntimeError(
            "No labeled clips remain in the configured range; check behavior_labels_csv and start/end_clip."
        )
    log.info(f"After dropping unlabeled clips: {len(clip_ids)} clips will be loaded")

    path = cfg.data.rodent_rollout_dir
    log.info(f"Loading intention from {path}")
    intention = _load_single_key(path, "/intention", [], clip_ids)
    log.info(f"Loading qpos from {path}")
    qpos = _load_single_key(path, "/qpos_rollout", [], clip_ids)

    log.info(f"Computing FK keypoints via stac-mjx ({cfg.data.xml_path})")
    mjx_model, mjx_data, site_ids, kp_names = setup_stac_reference(
        cfg.data.xml_path, cfg.data.stac_data_path
    )
    keypoints = qpos_to_keypoints(qpos, mjx_model, mjx_data, site_ids)

    log.info(
        f"Loaded: intention {intention.shape}, qpos {qpos.shape}, keypoints {keypoints.shape}, "
        f"{len(kp_names)} kp_names"
    )
    if intention.shape[0] != len(clip_ids) or qpos.shape[0] != len(clip_ids):
        raise RuntimeError(
            f"Shape mismatch: {len(clip_ids)} clips requested, "
            f"got intention {intention.shape[0]}, qpos {qpos.shape[0]}"
        )

    return ScaleSweepData(
        clip_ids=clip_ids,
        intention=intention,
        qpos=qpos,
        keypoints=keypoints,
        kp_names=kp_names,
        segments_by_clip=segments_by_clip,
    )


def index_to_clip_id(clip_ids: list[int], i: int) -> int:
    """Map a KPMS/Mapper internal 0..N-1 index back to the external clip_id."""
    return clip_ids[i]


def clip_id_to_index(clip_ids: list[int], clip_id: int) -> int:
    try:
        return clip_ids.index(clip_id)
    except ValueError:
        raise KeyError(f"clip_id {clip_id} not in configured range")
