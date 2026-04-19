"""KPMS reconstruction: real keypoint signal vs model-reconstructed signal.

For each KPMS variant (best-per-kappa selection), load the saved KPMS checkpoint,
reconstruct the keypoint trajectories via `estimate_coordinates`, and render
per-clip overlay PNGs:
  - Top: 6 representative keypoints' z-coordinate, original (gray) vs reconstructed (red)
  - Bottom: per-frame L2 reconstruction error across all keypoints

Output: `traces/{variant_id}/clip_{cid}_recon.png`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_RECON_KEYPOINTS = ["Snout", "SpineF", "SpineM", "SpineL", "TailBase", "HipL"]


def _stack_data_array(d: dict, key: str) -> np.ndarray:
    val = d[key]
    if isinstance(val, dict):
        keys = sorted(val.keys())
        return np.stack([np.array(val[k]) for k in keys], axis=0)
    return np.array(val)


def _kpms_reconstruct(model_ckpt: dict) -> np.ndarray:
    """Reconstruct keypoint coordinates from a fitted KPMS checkpoint.

    Returns Y_recon of shape (n_recordings, n_padded_frames, n_keypoints, 3).
    Caller is responsible for trimming the time axis to the unpadded length.
    """
    from jax_moseq.models.keypoint_slds import estimate_coordinates

    states = model_ckpt["states"]
    params = model_ckpt["params"]
    return np.array(estimate_coordinates(
        x=states["x"], v=states["v"], h=states["h"], Cd=params["Cd"],
    ))


def load_kpms_model_and_reconstruct(
    project_dir: str,
    model_name: str,
    coordinates: dict,
    confidences: dict,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the KPMS checkpoint and run reconstruction.

    Returns (Y_orig, Y_recon, recording_names_sorted), each Y of shape
    (n_recordings, n_unpadded_frames_min, n_keypoints, 3). KPMS pads internally;
    here we trim to the smallest end-frame across recordings (matches in our
    use case where every clip has the same length, e.g. 500).
    """
    import keypoint_moseq as kpms

    config = kpms.load_config(project_dir)
    data, metadata = kpms.format_data(coordinates, confidences, **config)
    model_ckpt, _, _, _ = kpms.load_checkpoint(project_dir, model_name)

    Y_padded = _stack_data_array(data, "Y")
    Y_recon_padded = _kpms_reconstruct(model_ckpt)

    # metadata = (rec_names_array, recording_intervals_array of shape (N, 2))
    intervals = np.asarray(metadata[1])
    n_unpadded = int(intervals[:, 1].min())  # uniform clip length in our case

    # KPMS prepends ar-padding (`nlags` frames) to make the AR transitions valid;
    # the original frames live at indices [start, end) per the intervals.
    starts = intervals[:, 0]
    ends = intervals[:, 1]
    if not np.all(starts == starts[0]) or not np.all(ends == ends[0]):
        raise RuntimeError(f"Heterogeneous clip intervals not supported: {intervals}")
    s, e = int(starts[0]), int(ends[0])
    Y = Y_padded[:, s:e]
    Y_recon = Y_recon_padded[:, s:e]

    rec_names = sorted(coordinates.keys())
    return Y, Y_recon, rec_names


def save_kpms_reconstruction_png(
    Y_clip: np.ndarray,
    Y_recon_clip: np.ndarray,
    kp_names: list[str],
    out_path: Path,
    title: str,
    target_kp_names: list[str] | None = None,
) -> None:
    """Render the original vs reconstructed signal for one clip.

    Args:
        Y_clip: (T, K, 3) original keypoints for the clip.
        Y_recon_clip: (T, K, 3) reconstructed keypoints for the clip.
        kp_names: list of K bodypart names.
        out_path: PNG path.
        title: figure title.
        target_kp_names: subset to plot (defaults to DEFAULT_RECON_KEYPOINTS, then
            falls back to first 6 keypoints if no name match found).
    """
    n_frames = Y_clip.shape[0]
    if target_kp_names is None:
        target_kp_names = DEFAULT_RECON_KEYPOINTS
    target_idxs: list[tuple[int, str]] = []
    for name in target_kp_names:
        if name in kp_names:
            target_idxs.append((kp_names.index(name), name))
    if not target_idxs:
        n_show = min(6, len(kp_names))
        target_idxs = [(i, kp_names[i]) for i in range(n_show)]

    err = np.linalg.norm(Y_clip - Y_recon_clip, axis=2).mean(axis=1)

    n_kp = len(target_idxs)
    n_cols = 3
    n_rows = 2 + (n_kp + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(12, 1.6 * n_rows + 0.8))
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=[1.2] * (n_rows - 1) + [0.7])

    for k, (kidx, kname) in enumerate(target_idxs):
        r, c = divmod(k, n_cols)
        ax = fig.add_subplot(gs[r, c])
        ax.plot(np.arange(n_frames), Y_clip[:, kidx, 2], color="gray", lw=1.0,
                label="real (z)")
        ax.plot(np.arange(n_frames), Y_recon_clip[:, kidx, 2], color="firebrick",
                lw=1.0, alpha=0.8, label="reconstructed (z)")
        ax.set_title(kname, fontsize=8)
        ax.set_xticks([])
        if k == 0:
            ax.legend(loc="best", fontsize=7, frameon=False)

    ax_err = fig.add_subplot(gs[-1, :])
    ax_err.plot(np.arange(n_frames), err, color="black", lw=1.0)
    ax_err.fill_between(np.arange(n_frames), err, 0, color="black", alpha=0.15)
    ax_err.set_xlim(0, n_frames)
    ax_err.set_ylabel("mean L2\nerror", fontsize=8)
    ax_err.set_xlabel("frame")
    ax_err.set_title("per-frame mean L2 keypoint reconstruction error",
                     fontsize=8)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def generate_kpms_reconstruction_pngs_for_variant(
    variant_id: str,
    project_dir: str,
    model_name: str,
    keypoints_array: np.ndarray,
    clip_ids: list[int],
    exemplar_clips: list[int],
    kp_names: list[str],
    out_dir: Path,
    prefix: str = "recording",
) -> list[Path]:
    """Run reconstruction once for the variant, then save one PNG per exemplar clip."""
    from topo_mimic.benchmark.hyperparam_sweep.keypoint_loader import (
        prepare_keypoints_for_kpms,
    )

    coordinates, confidences = prepare_keypoints_for_kpms(keypoints_array, prefix=prefix)
    Y, Y_recon, rec_names = load_kpms_model_and_reconstruct(
        project_dir=project_dir,
        model_name=model_name,
        coordinates=coordinates,
        confidences=confidences,
    )

    cid_to_idx = {cid: i for i, cid in enumerate(clip_ids)}
    written: list[Path] = []
    for cid in exemplar_clips:
        if cid not in cid_to_idx:
            continue
        i = cid_to_idx[cid]
        if i >= Y.shape[0]:
            continue
        png_path = out_dir / variant_id / f"clip_{cid}_recon.png"
        if png_path.exists():
            written.append(png_path)
            continue
        save_kpms_reconstruction_png(
            Y_clip=Y[i],
            Y_recon_clip=Y_recon[i],
            kp_names=kp_names,
            out_path=png_path,
            title=f"{variant_id} | clip_{cid} | KPMS reconstruction",
        )
        written.append(png_path)
    return written
