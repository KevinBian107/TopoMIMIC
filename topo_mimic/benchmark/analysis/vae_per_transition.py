"""
Per-transition VAE wrapper for KID evaluation.

Ports the moseq_jax KID pattern (``run_inception_distance.py``) into TopoVNL
but fits **one VAE per canonical transition label** instead of one shared
VAE. This uses the existing in-repo PyTorch RNN-VAE
(``topo_mimic/benchmark/hyperparam_sweep/custom_vae/``), not the JAX/Flax
version, so no new framework dependency.

Per-transition VAE means: training data for transition T is the set of raw
frames in ground-truth clips where the animal is labelled as one of the two
behaviors in T. For rare transitions with few frames, we fall back to a
shared-VAE trained on all transitions' frames and log the fallback.

Feature space: yaw-stripped qpos (matching the moseq_jax reference). The
caller is expected to load the clip-level qpos array; this module handles
GT-frame-window extraction, training, encoding, and checkpoint I/O.
"""

from __future__ import annotations

import os
import pickle as pkl
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from topo_mimic.benchmark.auto_label import BehaviorSegment
from topo_mimic.benchmark.hyperparam_sweep.custom_vae.config import VAEHyperparams
from topo_mimic.benchmark.hyperparam_sweep.custom_vae.model import RNNVAE
from topo_mimic.benchmark.hyperparam_sweep.custom_vae.trainer import VAETrainer


DEFAULT_MIN_FRAMES_FOR_VAE = 600


@dataclass
class TransitionVAE:
    """Fitted VAE for a single transition label, with metadata for caching."""
    label: str
    model: RNNVAE
    hyperparams: VAEHyperparams
    final_epoch: int
    n_train_frames: int
    used_fallback: bool = False
    fallback_reason: str = ""
    device: str = "cpu"


@dataclass
class PerTransitionVAECache:
    """In-memory cache of per-transition VAEs + a shared-VAE fallback.

    Lookup order for a label:
      1. vaes[label] if present.
      2. shared_vae if the label's own VAE failed to fit.
    """
    vaes: dict[str, TransitionVAE] = field(default_factory=dict)
    shared_vae: TransitionVAE | None = None
    input_dim: int = 0


def _parse_transition_label(label: str) -> tuple[str, str | None]:
    """Split ``"A <-> B"`` into (A, B); singleton returns (label, None)."""
    if "<->" in label:
        a, b = [s.strip() for s in label.split("<->", 1)]
        return a, b
    return label, None


def extract_frames_for_label(
    label: str,
    segments_by_clip: dict[int, list[BehaviorSegment]],
    raw_data: np.ndarray,
    time_window: int,
) -> np.ndarray:
    """
    Collect all raw frames belonging to transition ``label`` across clips.

    For a transition label "A <-> B", we include frames where the behavior is
    either A or B, within any clip where both A and B appear contiguously. For
    a singleton label "A", we include frames where the behavior is A.

    Returns a stack of (n_windows, time_window, n_dim) windows. If the total
    qualifying frames are fewer than ``time_window``, returns an empty array.
    """
    a, b = _parse_transition_label(label)
    labels_of_interest = {a} if b is None else {a, b}

    n_clips, n_frames, n_dim = raw_data.shape
    per_clip_frames: list[np.ndarray] = []

    for clip_id, segs in segments_by_clip.items():
        if clip_id < 0 or clip_id >= n_clips:
            continue
        relevant_segs = [s for s in segs if s.behavior in labels_of_interest]
        if not relevant_segs:
            continue
        if b is not None:
            behaviors_present = {s.behavior for s in relevant_segs}
            if not labels_of_interest.issubset(behaviors_present):
                continue
        mask = np.zeros(n_frames, dtype=bool)
        for s in relevant_segs:
            start = max(0, s.start)
            end = min(n_frames, s.end)
            if end > start:
                mask[start:end] = True
        if mask.sum() < time_window:
            continue
        clip_frames = raw_data[clip_id, mask, :]
        per_clip_frames.append(clip_frames)

    if not per_clip_frames:
        return np.empty((0, time_window, n_dim), dtype=np.float32)

    windows: list[np.ndarray] = []
    for frames in per_clip_frames:
        n_valid = frames.shape[0]
        for start in range(0, n_valid - time_window + 1, time_window):
            windows.append(frames[start:start + time_window])
    if not windows:
        return np.empty((0, time_window, n_dim), dtype=np.float32)
    return np.stack(windows, axis=0).astype(np.float32)


def _fit_vae_on_windows(
    windows: np.ndarray,
    hyperparams: VAEHyperparams,
    device: str,
    seed: int,
) -> tuple[RNNVAE, int]:
    """Train an RNN-VAE on a (n_windows, time_window, n_dim) stack."""
    n_windows, time_window, n_dim = windows.shape
    model = RNNVAE(
        input_dim=n_dim,
        hidden_dim=hyperparams.hidden_dim,
        latent_dim=hyperparams.latent_dim,
        rnn_layers=hyperparams.rnn_layers,
        rnn_type=hyperparams.rnn_type,
    )
    trainer = VAETrainer(model=model, hyperparams=hyperparams, device=device)

    data_for_trainer = windows.transpose(0, 1, 2).astype(np.float32)
    if data_for_trainer.ndim == 3:
        data_for_trainer = data_for_trainer.reshape(1, -1, n_dim)

    result = trainer.train(
        train_data=data_for_trainer,
        modality_name="per_transition",
        val_ratio=0.1,
        seed=seed,
    )
    return result.model, result.final_epoch


def fit_transition_vaes(
    labels_to_fit: list[str],
    segments_by_clip: dict[int, list[BehaviorSegment]],
    raw_data: np.ndarray,
    hyperparams: VAEHyperparams,
    device: str = "cpu",
    min_frames_for_vae: int = DEFAULT_MIN_FRAMES_FOR_VAE,
    seed: int = 0,
    cache_dir: str | None = None,
) -> PerTransitionVAECache:
    """
    Fit a VAE per transition label. Labels whose training set falls below
    ``min_frames_for_vae`` are marked as fallback; the caller gets a shared
    VAE on all labelled frames to use for those.

    If ``cache_dir`` is given, checkpoints are saved to
    ``{cache_dir}/{sanitized_label}.pt``. On next call with the same label,
    the checkpoint is loaded and training is skipped.
    """
    cache = PerTransitionVAECache(input_dim=raw_data.shape[-1])
    time_window = hyperparams.time_window

    for label in labels_to_fit:
        windows = extract_frames_for_label(label, segments_by_clip, raw_data, time_window=time_window)
        n_frames = int(windows.size // max(1, windows.shape[-1]))

        safe_label = label.replace(" ", "_").replace("<->", "to").replace("/", "_")
        ckpt_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            ckpt_path = os.path.join(cache_dir, f"{safe_label}.pt")

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"[VAE cache] loading {label!r} from {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = RNNVAE(
                input_dim=raw_data.shape[-1],
                hidden_dim=hyperparams.hidden_dim,
                latent_dim=hyperparams.latent_dim,
                rnn_layers=hyperparams.rnn_layers,
                rnn_type=hyperparams.rnn_type,
            )
            model.load_state_dict(state["state_dict"])
            model.to(device)
            cache.vaes[label] = TransitionVAE(
                label=label,
                model=model,
                hyperparams=hyperparams,
                final_epoch=state.get("final_epoch", -1),
                n_train_frames=state.get("n_train_frames", n_frames),
                used_fallback=state.get("used_fallback", False),
                fallback_reason=state.get("fallback_reason", ""),
                device=device,
            )
            continue

        if windows.shape[0] * time_window < min_frames_for_vae:
            print(f"[VAE per-transition] label {label!r}: only {windows.shape[0]} windows ({windows.shape[0] * time_window} frames); will fall back to shared VAE")
            cache.vaes[label] = TransitionVAE(
                label=label,
                model=None,  # placeholder; shared_vae used at encode time
                hyperparams=hyperparams,
                final_epoch=-1,
                n_train_frames=int(windows.shape[0] * time_window),
                used_fallback=True,
                fallback_reason="insufficient frames",
                device=device,
            )
            continue

        print(f"[VAE per-transition] training {label!r} on {windows.shape[0]} windows ({windows.shape[0] * time_window} frames)")
        try:
            model, final_epoch = _fit_vae_on_windows(windows, hyperparams, device, seed)
        except Exception as exc:
            print(f"[VAE per-transition] training {label!r} failed: {exc}; marking for fallback")
            cache.vaes[label] = TransitionVAE(
                label=label,
                model=None,
                hyperparams=hyperparams,
                final_epoch=-1,
                n_train_frames=int(windows.shape[0] * time_window),
                used_fallback=True,
                fallback_reason=f"training error: {exc}",
                device=device,
            )
            continue

        cache.vaes[label] = TransitionVAE(
            label=label,
            model=model,
            hyperparams=hyperparams,
            final_epoch=final_epoch,
            n_train_frames=int(windows.shape[0] * time_window),
            used_fallback=False,
            fallback_reason="",
            device=device,
        )
        if ckpt_path:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "final_epoch": final_epoch,
                    "n_train_frames": int(windows.shape[0] * time_window),
                    "used_fallback": False,
                    "fallback_reason": "",
                    "label": label,
                    "hyperparams": vars(hyperparams),
                },
                ckpt_path,
            )
            print(f"[VAE cache] saved {label!r} -> {ckpt_path}")

    fallback_needed = any(v.used_fallback for v in cache.vaes.values())
    if fallback_needed:
        print("[VAE per-transition] fitting shared-VAE fallback on all labelled frames")
        all_labels = list({s.behavior for segs in segments_by_clip.values() for s in segs})
        combined_windows_list = []
        for lab in all_labels:
            w = extract_frames_for_label(lab, segments_by_clip, raw_data, time_window=time_window)
            if w.shape[0] > 0:
                combined_windows_list.append(w)
        if combined_windows_list:
            combined = np.concatenate(combined_windows_list, axis=0)
            try:
                model, final_epoch = _fit_vae_on_windows(combined, hyperparams, device, seed)
                cache.shared_vae = TransitionVAE(
                    label="__shared__",
                    model=model,
                    hyperparams=hyperparams,
                    final_epoch=final_epoch,
                    n_train_frames=int(combined.shape[0] * time_window),
                    used_fallback=False,
                    fallback_reason="",
                    device=device,
                )
            except Exception as exc:
                print(f"[VAE per-transition] shared-VAE fallback training failed: {exc}")

    return cache


def encode_windows(
    vae: RNNVAE,
    windows: np.ndarray,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Encode (n_windows, time_window, n_dim) to (n_windows, latent_dim) via the
    VAE encoder's mean (deterministic).
    """
    if windows.shape[0] == 0:
        return np.empty((0, vae.fc_mu.out_features), dtype=np.float32)
    vae.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, windows.shape[0], batch_size):
            batch = torch.from_numpy(windows[start:start + batch_size]).float().to(device)
            z = vae.encode_sequence(batch)
            outs.append(z.cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def encode_frame_indices(
    vae: RNNVAE,
    clip_id: int,
    frame_indices: np.ndarray,
    raw_data: np.ndarray,
    time_window: int,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Encode a cluster member by extracting ``time_window``-sized windows from
    the member's frames and running them through the VAE encoder.

    If the frame indices are non-contiguous, we slice out the member frames
    first, then window the resulting contiguous array. Members shorter than
    ``time_window`` return an empty array.
    """
    frame_indices = np.asarray(frame_indices, dtype=int)
    if frame_indices.size < time_window:
        return np.empty((0, vae.fc_mu.out_features), dtype=np.float32)
    member_frames = raw_data[clip_id, frame_indices, :]
    n_valid = member_frames.shape[0]
    windows = []
    for start in range(0, n_valid - time_window + 1, time_window):
        windows.append(member_frames[start:start + time_window])
    if not windows:
        return np.empty((0, vae.fc_mu.out_features), dtype=np.float32)
    windows_stack = np.stack(windows, axis=0).astype(np.float32)
    return encode_windows(vae, windows_stack, device=device, batch_size=batch_size)


def resolve_vae_for_label(cache: PerTransitionVAECache, label: str) -> RNNVAE | None:
    """Return the RNNVAE to use for encoding samples at this label. Falls back
    to shared_vae if the per-label model is missing or marked fallback."""
    tv = cache.vaes.get(label)
    if tv is not None and tv.model is not None:
        return tv.model
    if cache.shared_vae is not None:
        return cache.shared_vae.model
    return None
