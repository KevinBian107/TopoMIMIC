"""Data loading utilities for analysis."""

import os
import pickle as pkl

import h5py
import numpy as np
import pandas as pd
import yaml

from vnl_tda.utils.load_utils import load_rollout_from_source


QPOS_BODY_NAMES = [
    "world", "floor", "walker", "torso", "vertebra_1", "vertebra_2", "vertebra_3",
    "vertebra_4", "vertebra_5", "vertebra_6", "pelvis", "upper_leg_L", "lower_leg_L",
    "foot_L", "toe_L", "upper_leg_R", "lower_leg_R", "foot_R", "toe_R",
    "vertebra_C1", "vertebra_C2", "vertebra_C3", "vertebra_C4", "vertebra_C5",
    "vertebra_C6", "vertebra_C7", "vertebra_C8", "vertebra_C9", "vertebra_C10",
    "vertebra_C11", "vertebra_C12", "vertebra_C13", "vertebra_C14", "vertebra_C15",
    "vertebra_C16", "vertebra_C17", "vertebra_C18", "vertebra_C19", "vertebra_C20",
    "vertebra_C21", "vertebra_C22", "vertebra_C23", "vertebra_C24", "vertebra_C25",
    "vertebra_C26", "vertebra_C27", "vertebra_C28", "vertebra_C29", "vertebra_C30",
    "vertebra_cervical_5", "vertebra_cervical_4", "vertebra_cervical_3",
    "vertebra_cervical_2", "vertebra_cervical_1", "vertebra_axis", "vertebra_atlant",
    "skull", "jaw", "scapula_L", "upper_arm_L", "lower_arm_L", "hand_L", "finger_L",
    "scapula_R", "upper_arm_R", "lower_arm_R", "hand_R", "finger_R",
]


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


def load_labels_yaml(method_dir: str, seed: int = 0) -> dict[int, str]:
    labels_path = os.path.join(method_dir, f"labels_seed_{seed}.yaml")
    if not os.path.exists(labels_path):
        return {}

    with open(labels_path, "r") as f:
        data = yaml.safe_load(f)

    labels = data.get("labels", {})
    return {int(k): v for k, v in labels.items() if v}


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


def load_behavior_labels_legacy(behavior_csv: str) -> dict[str, list[int]]:
    """Legacy behavior labels loader for clip-level labels."""
    df = pd.read_csv(behavior_csv)
    return {
        "walk": df[df["walk"] == 1]["clip_idx"].tolist(),
        "rear": df[df["rear"] == 1]["clip_idx"].tolist(),
        "groom": df[(df["groom_down"] == 1) | (df["groom_up"] == 1)]["clip_idx"].tolist(),
    }


def get_body_part_indices(body_parts: list[str]) -> tuple[list[int], list[str]]:
    """Get indices and valid names for body parts."""
    body_indices = []
    valid_body_parts = []
    for bp in body_parts:
        if bp in QPOS_BODY_NAMES:
            body_indices.append(QPOS_BODY_NAMES.index(bp))
            valid_body_parts.append(bp)
    return body_indices, valid_body_parts
