"""Per-method T extraction for the information-loss experiment.

T_t = globally-unique pre-W2 primitive instance ID at frame t of clip c, or
``-1`` if frame t is not in any primitive of that method.

Every primitive instance gets a unique integer id assigned by a single
deterministic pass; ids do NOT collide across clips. Each method's loader
returns a ``PrimitiveAssignment`` with that array plus light metadata used
later for the joint-distribution heatmaps.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


PrimitiveKind = Literal["dm_cycle", "dm_path", "kpms_sustained", "vae_sustained", "mapper_edge"]


@dataclass(frozen=True)
class PrimitiveAssignment:
    """Per-frame primitive id assignment for one (method, seed, feature-space).

    Attributes:
        method_name: human-readable identifier (e.g., "DMC-Int").
        family: one of {"DMC", "DMP", "KPMS", "VAE", "Mapper"}.
        feature_space: "intention" / "qpos" / "ekp" / None.
        seed: random seed if the method is stochastic, else None.
        T: (n_clips, frames_per_clip) int32 array. -1 = no primitive.
        primitive_id_to_meta: globally-unique pid -> {"clip_id", "n_frames", "type"}.
        n_clips: clip dimension of T.
        frames_per_clip: frame dimension of T.
    """

    method_name: str
    family: str
    feature_space: str | None
    seed: int | None
    T: np.ndarray
    primitive_id_to_meta: dict[int, dict] = field(default_factory=dict)
    n_clips: int = 0
    frames_per_clip: int = 0

    @property
    def coverage(self) -> float:
        return float((self.T >= 0).mean())


def _new_T(n_clips: int, frames_per_clip: int) -> np.ndarray:
    return np.full((n_clips, frames_per_clip), -1, dtype=np.int32)


# ---------------------------------------------------------------------------
# DM cycles and paths
# ---------------------------------------------------------------------------


def load_dm_primitives(
    extraction_dir: str | Path,
    kind: Literal["cycle", "path"],
    feature_space: str,
    n_clips: int,
    frames_per_clip: int,
) -> PrimitiveAssignment:
    """Load DM cycles or paths from ``all_{cycle,path}_dm.pkl``.

    The pickles store ``list[list[np.ndarray[int]]]`` indexed by clip; each
    inner ndarray is the (sorted) original-frame indices of one primitive.

    Args:
        extraction_dir: e.g., ``outputs/extraction/dm_pipeline_intention``.
        kind: ``"cycle"`` -> ``all_cycle_dm.pkl``; ``"path"`` -> ``all_path_dm.pkl``.
        feature_space: ``"intention"`` / ``"qpos"`` / ``"ekp"``.
    """
    ext_dir = Path(extraction_dir)
    fname = "all_cycle_dm.pkl" if kind == "cycle" else "all_path_dm.pkl"
    with open(ext_dir / fname, "rb") as f:
        per_clip = pickle.load(f)

    if len(per_clip) < n_clips:
        raise ValueError(
            f"DM pickle {ext_dir/fname} has {len(per_clip)} clips, expected >= {n_clips}"
        )

    T = _new_T(n_clips, frames_per_clip)
    pid_meta: dict[int, dict] = {}
    next_pid = 0

    for clip_id in range(n_clips):
        primitives = per_clip[clip_id]
        if not primitives:
            continue
        for prim_local_idx, frame_inds in enumerate(primitives):
            if frame_inds is None or len(frame_inds) == 0:
                continue
            pid = next_pid
            next_pid += 1

            frames = np.asarray(frame_inds, dtype=np.int64)
            frames = frames[(frames >= 0) & (frames < frames_per_clip)]
            if frames.size == 0:
                continue

            T[clip_id, frames] = pid
            pid_meta[pid] = {
                "clip_id": int(clip_id),
                "local_idx": int(prim_local_idx),
                "n_frames": int(frames.size),
                "kind": kind,
            }

    family = "DMC" if kind == "cycle" else "DMP"
    method_name = f"{family}-{feature_space.capitalize()}"
    return PrimitiveAssignment(
        method_name=method_name,
        family=family,
        feature_space=feature_space,
        seed=None,
        T=T,
        primitive_id_to_meta=pid_meta,
        n_clips=n_clips,
        frames_per_clip=frames_per_clip,
    )


# ---------------------------------------------------------------------------
# KPMS / VAE / Mapper -- all share the cluster_dict-of-instances structure
# ---------------------------------------------------------------------------


def _load_results_pkl(results_pkl: str | Path) -> dict:
    with open(results_pkl, "rb") as f:
        return pickle.load(f)


def _seed_payload(payload, seed: int) -> dict:
    """Best-effort extractor for the per-seed payload across method dicts."""
    if isinstance(payload, dict):
        if seed in payload:
            return payload[seed]
        if str(seed) in payload:
            return payload[str(seed)]
    raise KeyError(f"Seed {seed} not present in payload (keys={list(payload.keys())[:5]})")


def load_per_frame_labels_primitives(
    results_pkl: str | Path,
    seed: int,
    method_name: str,
    family: str,
    feature_space: str | None,
    n_clips: int,
    frames_per_clip: int,
) -> PrimitiveAssignment:
    """Loader for the per-frame ``labels`` array (KPMS syllables, Mapper
    node IDs, VAE cluster IDs).

    This is the method's natural *type-level* per-frame coarsening, which
    matches the user-facing language ``"syllables / nodes"`` in the paper.
    Unlike ``load_cluster_dict_primitives`` -- which yields instance-level
    pids -- this loader yields ~10-30 unique pids reused across clips, so
    H(T) is bounded by ``log2(n_type)`` and there is no clip-identity leak.
    """
    payload = _load_results_pkl(results_pkl)
    seed_payload = _seed_payload(payload, seed)
    labels = seed_payload.get("labels")
    if labels is None:
        raise KeyError(f"results.pkl at {results_pkl!r} has no 'labels' field")
    if not (isinstance(labels, list) and labels and hasattr(labels[0], "shape")):
        raise TypeError(
            f"results.pkl 'labels' at {results_pkl!r} is not list[ndarray]; got {type(labels)}"
        )

    T = _new_T(n_clips, frames_per_clip)
    pid_meta: dict[int, dict] = {}

    n_payload_clips = len(labels)
    if n_payload_clips > n_clips:
        n_payload_clips = n_clips

    for clip_id in range(n_payload_clips):
        arr = np.asarray(labels[clip_id])
        length = min(arr.size, frames_per_clip)
        if length == 0:
            continue
        row = arr[:length].astype(np.int64)
        out_row = T[clip_id, :length]
        valid = row >= 0
        out_row[valid] = row[valid]

    seen = np.unique(T[T >= 0])
    for pid in seen.tolist():
        pid_meta[int(pid)] = {"type_id": int(pid), "n_frames": int((T == pid).sum())}

    return PrimitiveAssignment(
        method_name=method_name,
        family=family,
        feature_space=feature_space,
        seed=seed,
        T=T,
        primitive_id_to_meta=pid_meta,
        n_clips=n_clips,
        frames_per_clip=frames_per_clip,
    )


def load_cluster_dict_primitives(
    results_pkl: str | Path,
    seed: int,
    method_name: str,
    family: str,
    feature_space: str | None,
    n_clips: int,
    frames_per_clip: int,
    pair_type: bool = False,
) -> PrimitiveAssignment:
    """Loader for any method that produces a ``cluster_dict`` of
    ``{cluster_key: [(clip_id, frame_indices), ...]}`` (KPMS, VAE, Mapper).

    ``cluster_key`` here is a (from-state, to-state) pair — an abstract
    transition CATEGORY produced by ``extract_sustained_transitions`` (top-K
    most frequent pairs). The function offers two id-assignment policies:

    - ``pair_type=False`` (default, "instance" mode): every individual
      ``(clip_id, frame_indices)`` tuple gets its own globally-unique pid.
      Cardinality matches the number of sustained-transition WINDOWS in the
      dataset (~hundreds-thousands).
    - ``pair_type=True``: every window sharing the same ``cluster_key`` gets
      the SAME pid (the pair-type index). Cardinality matches the number of
      distinct (from, to) pair categories (~10 for KPMS / VAE; ~5-25 for
      Mapper). This is the "abstract transition coarsening" level: a frame's
      pid says only which pair-CATEGORY it belongs to, reused across clips.
    """
    payload = _load_results_pkl(results_pkl)
    seed_payload = _seed_payload(payload, seed)
    cluster_dict = seed_payload.get("cluster_dict")
    if cluster_dict is None:
        raise KeyError(f"results.pkl at {results_pkl!r} has no cluster_dict")

    T = _new_T(n_clips, frames_per_clip)
    pid_meta: dict[int, dict] = {}
    next_pid = 0

    cluster_keys = list(cluster_dict.keys())

    for type_idx, cluster_key in enumerate(cluster_keys):
        instances = cluster_dict[cluster_key]
        for entry in instances:
            if not (isinstance(entry, tuple) and len(entry) >= 2):
                continue
            clip_id = int(entry[0])
            frame_inds = np.asarray(entry[1], dtype=np.int64)
            if frame_inds.size == 0 or not (0 <= clip_id < n_clips):
                continue

            frames = frame_inds[(frame_inds >= 0) & (frame_inds < frames_per_clip)]
            if frames.size == 0:
                continue

            if pair_type:
                pid = type_idx
            else:
                pid = next_pid
                next_pid += 1

            T[clip_id, frames] = pid

            if pair_type:
                meta = pid_meta.setdefault(pid, {
                    "cluster_key": str(cluster_key),
                    "n_instances": 0,
                    "n_frames": 0,
                })
                meta["n_instances"] += 1
                meta["n_frames"] += int(frames.size)
            else:
                pid_meta[pid] = {
                    "clip_id": clip_id,
                    "n_frames": int(frames.size),
                    "cluster_key": str(cluster_key),
                }

    return PrimitiveAssignment(
        method_name=method_name,
        family=family,
        feature_space=feature_space,
        seed=seed,
        T=T,
        primitive_id_to_meta=pid_meta,
        n_clips=n_clips,
        frames_per_clip=frames_per_clip,
    )


# Thin wrappers with consistent naming for the orchestrator. Three primitive
# levels are exposed for baseline methods:
#
# - "state"     -- per-frame syllable / node / cluster ID (the method's raw
#                  state-level coarsening output, before any pair construction).
# - "pair_type" -- per-frame (from, to) pair-type CATEGORY id (one id per
#                  distinct pair, reused across all instances and all clips).
#                  This is the abstract transition coarsening for these methods.
# - "instance"  -- per-frame sustained-transition / edge-intersection window
#                  id, globally unique per window.


PrimitiveLevel = Literal["state", "pair_type", "instance"]


def _dispatch_baseline_loader(
    primitive_level: str,
    *,
    results_pkl: str | Path,
    seed: int,
    method_name: str,
    family: str,
    feature_space: str | None,
    n_clips: int,
    frames_per_clip: int,
) -> PrimitiveAssignment:
    if primitive_level == "state":
        return load_per_frame_labels_primitives(
            results_pkl=results_pkl, seed=seed,
            method_name=method_name, family=family,
            feature_space=feature_space,
            n_clips=n_clips, frames_per_clip=frames_per_clip,
        )
    if primitive_level in ("pair_type", "instance"):
        return load_cluster_dict_primitives(
            results_pkl=results_pkl, seed=seed,
            method_name=method_name, family=family,
            feature_space=feature_space,
            n_clips=n_clips, frames_per_clip=frames_per_clip,
            pair_type=(primitive_level == "pair_type"),
        )
    raise ValueError(
        f"Unknown primitive_level={primitive_level!r}; "
        f"expected one of {'state', 'pair_type', 'instance'}"
    )


def load_kpms_primitives(
    results_pkl: str | Path,
    seed: int,
    variant: Literal["intention", "keypoint"],
    n_clips: int,
    frames_per_clip: int,
    primitive_level: PrimitiveLevel = "pair_type",
) -> PrimitiveAssignment:
    feat = "intention" if variant == "intention" else "ekp"
    method_name = "KPMS-Int" if variant == "intention" else "KPMS-KP"
    return _dispatch_baseline_loader(
        primitive_level,
        results_pkl=results_pkl, seed=seed,
        method_name=method_name, family="KPMS",
        feature_space=feat,
        n_clips=n_clips, frames_per_clip=frames_per_clip,
    )


def load_vae_primitives(
    results_pkl: str | Path,
    seed: int,
    n_clips: int,
    frames_per_clip: int,
    primitive_level: PrimitiveLevel = "pair_type",
) -> PrimitiveAssignment:
    return _dispatch_baseline_loader(
        primitive_level,
        results_pkl=results_pkl, seed=seed,
        method_name="VAE", family="VAE",
        feature_space="qpos",
        n_clips=n_clips, frames_per_clip=frames_per_clip,
    )


def load_mapper_primitives(
    results_pkl: str | Path,
    seed: int,
    feature_space: Literal["intention", "qpos", "ekp"],
    n_clips: int,
    frames_per_clip: int,
    primitive_level: PrimitiveLevel = "pair_type",
) -> PrimitiveAssignment:
    short = {"intention": "Int", "qpos": "Qpos", "ekp": "EKP"}[feature_space]
    return _dispatch_baseline_loader(
        primitive_level,
        results_pkl=results_pkl, seed=seed,
        method_name=f"Mapper-{short}", family="Mapper",
        feature_space=feature_space,
        n_clips=n_clips, frames_per_clip=frames_per_clip,
    )


# ---------------------------------------------------------------------------
# Sanity-check helpers
# ---------------------------------------------------------------------------


def summarize_assignment(assignment: PrimitiveAssignment) -> str:
    T = assignment.T
    unique_pids = np.unique(T[T >= 0])
    bg_fraction = float((T == -1).mean())
    return (
        f"method={assignment.method_name}\n"
        f"  feature_space:    {assignment.feature_space}\n"
        f"  seed:             {assignment.seed}\n"
        f"  shape:            {T.shape}\n"
        f"  n_primitives:     {unique_pids.size}\n"
        f"  background_frac:  {bg_fraction:.3f}\n"
        f"  coverage:         {assignment.coverage:.3f}"
    )
