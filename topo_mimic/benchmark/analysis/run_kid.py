"""
KID analysis runner: per-cluster KID against its auto-assigned transition's
ground-truth VAE.

Given a method's cluster_dict + auto-labels from ``labels_seed_{seed}.yaml``
and the raw GT frames from ``data/rodent_ar.h5``, this runner:

  1. Fits one RNN-VAE per canonical transition label (with caching).
  2. For every cluster with an auto-assigned label T:
       - Encodes the cluster's member frames via VAE_T -> ``mu_fake``.
       - Encodes the ground-truth T-transition frames via VAE_T ->
         ``mu_real``.
       - Computes KID between the two sets and records mean + std.
  3. Writes ``{seed_output}/kid/kid_table.csv`` with columns
     ``cluster_id, label, n_member_frames, n_gt_frames, kid_mean, kid_std``.

Skipped if the method has no auto-labels yaml (you must run the labeling
pipeline first).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from topo_mimic.benchmark.analysis.kid import compute_kid
from topo_mimic.benchmark.analysis.vae_per_transition import (
    PerTransitionVAECache,
    encode_frame_indices,
    extract_frames_for_label,
    fit_transition_vaes,
    resolve_vae_for_label,
    encode_windows,
)
from topo_mimic.benchmark.auto_label import load_behavior_segments
from topo_mimic.benchmark.hyperparam_sweep.custom_vae.config import VAEHyperparams


def _load_labels_yaml(labels_path: str) -> dict[int, str]:
    with open(labels_path, "r") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("labels", {}) or {}
    return {int(k): str(v) for k, v in raw.items() if v}


def _load_intention_or_qpos(h5_path: str, key: str = "qpos_rollout") -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if key in f:
            return np.array(f[key])
        if key == "qpos_rollout" and "qpos" in f:
            return np.array(f["qpos"])
        raise KeyError(f"HDF5 key {key!r} not found in {h5_path}; available: {list(f.keys())}")


def run_kid_for_method_seed(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    method_dir: str,
    seed: int,
    behavior_labels_csv: str,
    raw_h5_path: str,
    output_dir: str,
    hyperparams: VAEHyperparams | None = None,
    vae_cache_dir: str | None = None,
    kid_seed: int = 42,
    device: str | None = None,
) -> str | None:
    """
    Run KID for one (method, seed) output. Returns the path to kid_table.csv
    on success, or None on skip.
    """
    labels_path = os.path.join(method_dir, f"labels_seed_{seed}.yaml")
    if not os.path.exists(labels_path):
        print(f"[KID] skipping {method_dir}/seed_{seed}: no {labels_path}")
        return None

    cluster_labels = _load_labels_yaml(labels_path)
    if not cluster_labels:
        print(f"[KID] skipping {method_dir}/seed_{seed}: labels yaml empty")
        return None

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "kid_table.csv")
    if os.path.exists(out_csv):
        print(f"[KID] output already exists: {out_csv}; skipping (delete to rerun)")
        return out_csv

    segments = load_behavior_segments(behavior_labels_csv)
    if not segments:
        print(f"[KID] no GT segments loaded from {behavior_labels_csv}; aborting")
        return None

    raw_data = _load_intention_or_qpos(raw_h5_path, key="qpos_rollout")

    if hyperparams is None:
        hyperparams = VAEHyperparams()

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    labels_used = sorted({lbl for lbl in cluster_labels.values() if lbl})
    if not labels_used:
        print(f"[KID] no valid cluster labels; aborting")
        return None

    print(f"[KID] fitting VAEs for {len(labels_used)} unique labels")
    cache = fit_transition_vaes(
        labels_to_fit=labels_used,
        segments_by_clip=segments,
        raw_data=raw_data,
        hyperparams=hyperparams,
        device=dev,
        seed=kid_seed,
        cache_dir=vae_cache_dir,
    )

    rows = []
    for cluster_id, members in sorted(cluster_dict.items()):
        label = cluster_labels.get(cluster_id, "")
        if not label:
            rows.append((cluster_id, "", 0, 0, float("nan"), float("nan"), "no_label"))
            continue

        vae = resolve_vae_for_label(cache, label)
        if vae is None:
            rows.append((cluster_id, label, 0, 0, float("nan"), float("nan"), "no_vae"))
            continue

        mu_fake_list: list[np.ndarray] = []
        n_member_frames = 0
        for clip_id, frame_indices in members:
            if clip_id < 0 or clip_id >= raw_data.shape[0]:
                continue
            z = encode_frame_indices(
                vae=vae,
                clip_id=int(clip_id),
                frame_indices=frame_indices,
                raw_data=raw_data,
                time_window=hyperparams.time_window,
                device=dev,
            )
            if z.shape[0] > 0:
                mu_fake_list.append(z)
                n_member_frames += int(len(frame_indices))
        if not mu_fake_list:
            rows.append((cluster_id, label, 0, 0, float("nan"), float("nan"), "no_member_windows"))
            continue
        mu_fake = np.concatenate(mu_fake_list, axis=0)

        gt_windows = extract_frames_for_label(
            label=label,
            segments_by_clip=segments,
            raw_data=raw_data,
            time_window=hyperparams.time_window,
        )
        if gt_windows.shape[0] == 0:
            rows.append((cluster_id, label, n_member_frames, 0, float("nan"), float("nan"), "no_gt_windows"))
            continue
        mu_real = encode_windows(vae, gt_windows, device=dev)

        kid_mean, kid_std = compute_kid(mu_real, mu_fake, seed=kid_seed)
        rows.append((cluster_id, label, n_member_frames, int(gt_windows.shape[0] * hyperparams.time_window), kid_mean, kid_std, "ok"))

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "label", "n_member_frames", "n_gt_frames", "kid_mean", "kid_std", "status"])
        for row in rows:
            writer.writerow(row)

    print(f"[KID] wrote {len(rows)} rows to {out_csv}")
    return out_csv
