"""Data loading utilities for analysis."""

import os
import pickle as pkl

import h5py
import numpy as np
import pandas as pd
import yaml

from topo_mimic.dmwk.utils.load_utils import load_rollout_from_source


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


def load_labels_yaml(
    method_dir: str,
    seed: int = 0,
) -> tuple[dict[int, str], dict[int, str]]:
    """
    Load cluster labels and turn directions from YAML file.

    Args:
        method_dir: Path to method output directory
        seed: Random seed

    Returns:
        Tuple of (labels, turn_directions) where both are {cluster_id: value}
    """
    labels_path = os.path.join(method_dir, f"labels_seed_{seed}.yaml")
    if not os.path.exists(labels_path):
        return {}, {}

    with open(labels_path, "r") as f:
        data = yaml.safe_load(f)

    labels = data.get("labels", {})
    turn_directions = data.get("turn_directions", {})

    na_values = {"n/a", "na", "N/A", "NA", ""}
    labels_out = {int(k): v for k, v in labels.items() if v and v.strip().lower() not in na_values}
    turns_out = {int(k): v for k, v in turn_directions.items() if v}

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
