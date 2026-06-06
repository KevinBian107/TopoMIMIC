"""Compute the variance-frame tradeoff for DM-paths and DM-cycles.

For every primitive (path or cycle) over all clips and all extraction spaces we
measure the whole-body qpos variance it captures (sum of per-joint variances on
the root-stripped qpos restricted to the primitive's frames) versus the number
of frames it uses. A matched null (random contiguous windows of the same length
from the same clip) isolates whether the primitive concentrates on behaviorally
rich, high-variance sub-trajectories rather than arbitrary frames.
"""
from __future__ import annotations

import os
import pickle as pkl

import h5py
import numpy as np
import pandas as pd

from .config import SpaceSpec, VFTConfig


def load_qpos(cfg: VFTConfig) -> np.ndarray:
    with h5py.File(cfg.qpos_h5, "r") as f:
        key = cfg.qpos_key if cfg.qpos_key in f else "qpos"
        return f[key][:]  # (n_clips, n_frames, D)


def _load_primitives(extraction_dir: str, kind: str) -> list:
    fname = "all_cycle_dm.pkl" if kind == "cycle" else "all_path_dm.pkl"
    with open(os.path.join(extraction_dir, fname), "rb") as f:
        return pkl.load(f)


def _whole_body_variance(qpos_clip: np.ndarray, frames: np.ndarray, skip: int) -> float:
    """Sum of per-joint variances of the root-stripped qpos over `frames`."""
    sub = qpos_clip[frames, skip:]          # (n, D-skip)
    return float(np.var(sub, axis=0).sum())  # trace of covariance


def _random_window_variance(qpos_clip: np.ndarray, n: int, skip: int,
                            n_random: int, rng: np.random.Generator) -> float:
    """Mean whole-body variance over `n_random` random contiguous windows of
    length n drawn from the clip (the matched same-length sub-trajectory null)."""
    T = qpos_clip.shape[0]
    if n >= T:
        return _whole_body_variance(qpos_clip, np.arange(T), skip)
    starts = rng.integers(0, T - n + 1, size=n_random)
    vals = [_whole_body_variance(qpos_clip, np.arange(s, s + n), skip) for s in starts]
    return float(np.mean(vals))


def _records_for_space(space: SpaceSpec, kind: str, qpos: np.ndarray,
                       cfg: VFTConfig, rng: np.random.Generator) -> list[dict]:
    prims = _load_primitives(space.extraction_dir, kind)
    n_clips, T = qpos.shape[0], qpos.shape[1]
    offset = space.takens_offset
    out: list[dict] = []
    for clip_id, clip_prims in enumerate(prims):
        if clip_id >= n_clips:
            break
        qpos_clip = qpos[clip_id]
        for prim in clip_prims:
            # Takens index -> clip frame, then keep in-range
            frames = np.asarray(prim, dtype=int) + offset
            frames = frames[(frames >= 0) & (frames < T)]
            n = len(frames)
            if n < cfg.min_frames:
                continue
            v_prim = _whole_body_variance(qpos_clip, frames, cfg.skip_root_dims)
            v_rand = _random_window_variance(qpos_clip, n, cfg.skip_root_dims,
                                             cfg.n_random, rng)
            out.append({
                "space": space.name,
                "type": "DM-path" if kind == "path" else "DM-cycle",
                "clip": clip_id,
                "n_frames": n,
                "var_captured": v_prim,
                "var_random": v_rand,
                "efficiency": (v_prim / v_rand) if v_rand > 0 else np.nan,
            })
    return out


def compute_tradeoff(cfg: VFTConfig) -> pd.DataFrame:
    qpos = load_qpos(cfg)
    rng = np.random.default_rng(cfg.rng_seed)
    records: list[dict] = []
    for space in cfg.spaces:
        for kind in ("path", "cycle"):
            recs = _records_for_space(space, kind, qpos, cfg, rng)
            records.extend(recs)
            print(f"  [{space.name}/{kind}] {len(recs)} primitives")
    return pd.DataFrame.from_records(records)
