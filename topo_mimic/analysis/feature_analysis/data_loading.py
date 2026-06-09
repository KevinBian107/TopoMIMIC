"""Data loading utilities for analysis."""

import os
import pickle as pkl

import h5py
import numpy as np
import pandas as pd
import yaml

from topo_mimic.extraction.utils.load_utils import load_rollout_from_source


def load_experiment_results(experiment_dir: str) -> dict:
    results_path = os.path.join(experiment_dir, "all_results.pkl")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results not found: {results_path}")

    with open(results_path, "rb") as f:
        return pkl.load(f)


def load_method_results(method_dir: str) -> dict:
    results_path = os.path.join(method_dir, "results.pkl")
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            return pkl.load(f)
    return {}


def load_cluster_dict(method_dir: str, seed: int = 0) -> dict:
    """
    Load cluster_dict for a specific seed.

    Args:
        method_dir: Path to method output directory
        seed: Random seed to load

    Returns:
        Cluster dict mapping cluster_id to list of (clip_id, frames)
    """
    results = load_method_results(method_dir)
    if seed in results:
        result = results[seed]
        return result.get("cluster_dict", {})
    return {}


# --- Takens-embedding -> frame-space alignment ----------------------------
# DM primitives store Takens-NODE indices. Node t is the embedding vector
# [x(t), x(t+tau), x(t+2tau)]; its dynamic footprint spans clip frames
# [t, t+(m-1)*tau], centered at t + (m-1)*tau/2. To compare a DM primitive against
# frame-space ground truth (behavior/direction segments, per-frame GT), shift its
# frames to that window center. Non-DM methods (KPMS/VAE/Mapper/full/unif) already
# live in frame space -> offset 0. m=3 for every space, so the center offset == tau.
_DM_SPACE_OFFSET = {"intention": 6, "qpos": 9, "ekp": 8}  # (m-1)*tau/2, m=3


def center_offset_for_space(feature_space: str | None) -> int:
    """Takens-window-center offset for a DM feature space (0 if unknown/non-DM)."""
    return _DM_SPACE_OFFSET.get(feature_space, 0)


def center_offset_for_method(method_name: str) -> int:
    """Frames to add to a DM primitive's node index to reach the Takens-window
    center (aligning it with frame-space GT). 0 for non-DM (frame-space) methods."""
    name = (method_name or "").lower()
    if "dmc" not in name and "dmp" not in name:
        return 0
    if "int" in name:
        return _DM_SPACE_OFFSET["intention"]
    if "qpos" in name:
        return _DM_SPACE_OFFSET["qpos"]
    if "ekp" in name:
        return _DM_SPACE_OFFSET["ekp"]
    return 0


def recenter_cluster_dict(cluster_dict: dict, method_name: str) -> dict:
    """Return a copy of ``cluster_dict`` whose DM primitive frames are shifted to
    the Takens-window center so they align with frame-space ground truth. No-op for
    non-DM methods. Clustering is unaffected: this only adjusts the frames handed to
    downstream GT lookup / sequence extraction / rendering, not the stored result."""
    off = center_offset_for_method(method_name)
    if off == 0 or not cluster_dict:
        return cluster_dict
    return {
        cid: [(int(c), np.asarray(f).flatten() + off) for c, f in members]
        for cid, members in cluster_dict.items()
    }


_DIRECTION_TOKENS = {"Left": "left", "Right": "right", "Straight": "straight"}


def _extract_direction_from_label(label: str) -> str | None:
    """Directional auto-labels embed the turn as the trailing token
    (e.g. "Immobile -> Turn Left" or "Immobile Left"). Return "left"/"right"
    /"straight" or None if no direction token is present.
    """
    if not label:
        return None
    tokens = label.strip().split()
    if tokens and tokens[-1] in _DIRECTION_TOKENS:
        return _DIRECTION_TOKENS[tokens[-1]]
    return None


def load_labels_yaml(
    method_dir: str,
    seed: int = 0,
) -> tuple[dict[int, str], dict[int, str]]:
    """Load cluster labels and turn directions.

    Turn directions come from either (a) a legacy ``turn_directions`` YAML
    block, or (b) parsed from the trailing Left/Right/Straight token on the
    new directional cluster labels (``"A -> B Direction"``).
    """
    labels_path = os.path.join(method_dir, f"labels_seed_{seed}.yaml")
    if not os.path.exists(labels_path):
        return {}, {}

    with open(labels_path, "r") as f:
        data = yaml.safe_load(f)

    labels = data.get("labels", {}) or {}
    turn_directions = data.get("turn_directions", {}) or {}

    na_values = {"n/a", "na", "N/A", "NA", ""}
    labels_out = {int(k): v for k, v in labels.items() if v and v.strip().lower() not in na_values}

    turns_out: dict[int, str] = {int(k): v for k, v in turn_directions.items() if v}
    # Back-fill from the label string for the new directional auto-labels.
    for cid, lbl in labels_out.items():
        if cid in turns_out:
            continue
        extracted = _extract_direction_from_label(lbl)
        if extracted is not None:
            turns_out[cid] = extracted

    return labels_out, turns_out


def load_qpos(clip_id: int, rodent_rollout_dir: str) -> np.ndarray:
    roll = load_rollout_from_source(int(clip_id), rodent_rollout_dir)
    return np.asarray(roll["qpos"])


def load_intention(clip_id: int, rodent_rollout_dir: str) -> np.ndarray:
    if rodent_rollout_dir.endswith('.h5'):
        with h5py.File(rodent_rollout_dir, 'r') as f:
            return np.array(f['intention'][clip_id])
    else:
        path = os.path.join(rodent_rollout_dir, f"clip_{clip_id}.h5")
        with h5py.File(path, 'r') as f:
            return np.array(f['intention'])
