"""Shared preprocessing + resample path for BOTH cluster and real-GT windows.

Copy-adapted from the deleted classifier pipeline; kept minimal so there is
exactly ONE code path producing the (K, D) windows that feed C2ST.
"""

from __future__ import annotations

import numpy as np


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - w * x)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return np.stack(
        [r00, r01, r02, r10, r11, r12, r20, r21, r22], axis=-1
    ).reshape(q.shape[:-1] + (3, 3))


def handle_rotation(data: np.ndarray) -> np.ndarray:
    quat = data[..., 3:7]
    R = quat_to_rotmat(quat)
    heading = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    cos_h = np.cos(-heading)
    sin_h = np.sin(-heading)
    R_yaw_inv = np.zeros(R.shape[:-2] + (3, 3), dtype=data.dtype)
    R_yaw_inv[..., 0, 0] = cos_h
    R_yaw_inv[..., 0, 1] = -sin_h
    R_yaw_inv[..., 1, 0] = sin_h
    R_yaw_inv[..., 1, 1] = cos_h
    R_yaw_inv[..., 2, 2] = 1.0
    R_local = np.einsum("...ij,...jk->...ik", R_yaw_inv, R)
    col1 = R_local[..., :, 0]
    col2 = R_local[..., :, 1]
    rot_6d = np.concatenate([col1, col2], axis=-1)
    return np.concatenate([data[..., :3], rot_6d, data[..., 7:]], axis=-1)


def preprocess_qpos(
    data: np.ndarray,
    skip_root_dims: int = 7,
) -> np.ndarray:
    """Strip the full root state to match DTW / KID preprocessing.

    Input qpos is (..., T, 74) with layout xyz(3) + quat(4) + joints(67).
    Dropping the first 7 dims removes xyz and the full root quaternion,
    leaving 67 joint angles — the same feature space DTW uses
    (``metrics.run_dtw_analysis`` calls ``data[..., skip_root_dims:]`` with
    ``skip_root_dims=7``) and the same space the KID pipeline in
    track-mjx/moseq_jax operates on. Keeping z or the root orientation
    lets the classifier distinguish cluster vs real on absolute pose
    rather than behavioral dynamics, which inflates C2ST accuracy.
    """
    return data[..., int(skip_root_dims):]


def resample_to_K(arr: np.ndarray, K: int) -> np.ndarray:
    """Linear resample a (T, D) array to (K, D). T==K: identity copy. T==1: broadcast."""
    if arr.ndim != 2:
        raise ValueError(f"expected (T, D) array, got shape {arr.shape}")
    T, D = arr.shape
    if T == K:
        return arr.astype(np.float32, copy=True)
    if T == 1:
        return np.broadcast_to(arr, (K, D)).astype(np.float32, copy=True)
    src_idx = np.linspace(0, T - 1, K, dtype=np.float64)
    lo = np.floor(src_idx).astype(np.int64)
    hi = np.clip(lo + 1, 0, T - 1)
    frac = (src_idx - lo).astype(np.float32)
    out = (1.0 - frac[:, None]) * arr[lo] + frac[:, None] * arr[hi]
    return out.astype(np.float32, copy=True)


def extract_and_resample(
    qpos_preproc: np.ndarray,  # (N_clips, T, D)
    clip_id: int,
    start: int,
    end_exclusive: int,
    K: int,
) -> np.ndarray:
    """Single code path used by BOTH cluster-window and real-GT-window
    extraction: slice [start:end_exclusive] from clip_id, resample to K.

    Returns (K, D). Out-of-bounds or empty slabs return zeros.
    """
    T = qpos_preproc.shape[1]
    lo = max(0, int(start))
    hi = min(T, int(end_exclusive))
    if lo >= hi:
        return np.zeros((K, qpos_preproc.shape[-1]), dtype=np.float32)
    slab = qpos_preproc[int(clip_id), lo:hi]
    return resample_to_K(slab, K)
