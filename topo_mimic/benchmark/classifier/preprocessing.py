"""qpos preprocessing copied from track-mjx KID pipeline.

Source: `track-mjx/moseq_jax/experiments/run_inception_distance.py` (lines 81-163).
Pipeline:
    qpos (T, 74)
    -> handle_rotation: root quaternion -> yaw-removed 6D rotation  (74 -> 76)
    -> exclude_xy:     drop global x,y translation                  (76 -> 74)
    -> normalize_joints: z-score joint dims (joint_start=7) using train-split stats
"""

from __future__ import annotations

import numpy as np


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
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
    """Replace root quaternion with yaw-removed 6D rotation representation.

    Indices 3-6 (MuJoCo w,x,y,z quaternion) -> 6D rotation. Shape 74 -> 76.
    """
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
    exclude_xy: bool = True,
    do_handle_rotation: bool = True,
) -> np.ndarray:
    """handle_rotation (if set) then drop xy. 74 -> 76 -> 74."""
    if do_handle_rotation:
        data = handle_rotation(data)
    if exclude_xy:
        data = data[..., 2:]
    return data


def get_joint_start_index(exclude_xy: bool, handle_rotation_flag: bool) -> int:
    pos_dims = 1 if exclude_xy else 3
    rot_dims = 6 if handle_rotation_flag else 4
    return pos_dims + rot_dims


def compute_joint_normalization(
    data: np.ndarray,
    joint_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std for joint dimensions, computed on TRAIN-SPLIT only."""
    joints = data[..., joint_start:]
    flat = joints.reshape(-1, joints.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_joints(
    data: np.ndarray,
    joint_start: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    result = data.astype(np.float32, copy=True)
    result[..., joint_start:] = (data[..., joint_start:] - mean) / std
    return result
