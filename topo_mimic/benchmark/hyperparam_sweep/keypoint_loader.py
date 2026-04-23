"""Convert qpos to keypoints using mujoco/mjx directly."""

from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import jax
    import jax.numpy as jp
    import mujoco
    import mujoco.mjx as mjx
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

DEFAULT_KP_NAMES = [
    "Snout", "EarL", "EarR", "SpineF", "SpineM", "SpineL", "TailBase",
    "ShoulderL", "ElbowL", "WristL", "HandL",
    "ShoulderR", "ElbowR", "WristR", "HandR",
    "HipL", "KneeL", "AnkleL", "FootL",
    "HipR", "KneeR", "AnkleR", "FootR",
]


def setup_mujoco_model(
    xml_path: str,
    kp_names: list[str] | None = None,
) -> tuple[Any, Any, list[int], list[str]]:
    if not HAS_MUJOCO:
        raise ImportError("mujoco not installed")

    if kp_names is None:
        kp_names = DEFAULT_KP_NAMES

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    site_ids = []
    valid_kp_names = []
    for name in kp_names:
        try:
            site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                site_ids.append(site_id)
                valid_kp_names.append(name)
        except Exception:
            pass

    if not site_ids:
        print(f"Warning: No sites found matching keypoint names. Using all sites.")
        site_ids = list(range(mj_model.nsite))
        valid_kp_names = [mj_model.site(i).name for i in site_ids]

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    return mjx_model, mjx_data, site_ids, valid_kp_names


def qpos_to_keypoints_mujoco(
    qpos_data: np.ndarray,
    mjx_model: Any,
    mjx_data: Any,
    site_ids: list[int],
    clip_batch_size: int = 32,
) -> np.ndarray:
    if not HAS_MUJOCO:
        raise ImportError("mujoco not installed")

    site_ids_array = jp.array(site_ids)

    @jax.jit
    def forward_kinematics(qpos_single):
        data = mjx_data.replace(qpos=qpos_single)
        data = mjx.forward(mjx_model, data)
        markers = data.site_xpos[site_ids_array]
        return markers

    # Vectorize over frames within a single clip, then loop clips in Python
    # batches to keep peak memory bounded — the full (n_clips, n_frames)
    # vmap materialized ~400GB on CPU for 829×500 qpos on the rodent model.
    process_sequence = jax.jit(jax.vmap(forward_kinematics))

    n_clips = qpos_data.shape[0]
    chunks: list[np.ndarray] = []
    for start in range(0, n_clips, clip_batch_size):
        stop = min(start + clip_batch_size, n_clips)
        batch = jp.asarray(qpos_data[start:stop])
        batch_out = jax.vmap(process_sequence)(batch)
        chunks.append(np.asarray(batch_out))
    return np.concatenate(chunks, axis=0)


def setup_stac_reference(
    xml_path: str,
    data_path: str,
) -> tuple[Any, Any, list[int], list[str]]:
    try:
        import stac_mjx
        from stac_mjx import io
        from stac_mjx.stac import Stac

        cfg, d = io.load_stac_data(data_path)
        kp_names = d.kp_names
        stac = Stac(xml_path, cfg, kp_names)

        with h5py.File(data_path, 'r') as f:
            offsets = f['offsets'][:]

        mjx_model = mjx.put_model(stac._mj_model)
        mjx_model = stac_mjx.utils.set_site_pos(
            mjx_model,
            jp.reshape(offsets, (-1, 3)),
            stac._body_site_idxs,
        )
        mjx_data = mjx.make_data(mjx_model)

        return mjx_model, mjx_data, list(stac._body_site_idxs), kp_names

    except Exception as e:
        print(f"stac-mjx initialization failed: {e}")
        print("Falling back to direct mujoco approach...")

        kp_names = None
        if Path(data_path).exists():
            try:
                with h5py.File(data_path, 'r') as f:
                    if 'kp_names' in f:
                        kp_names = [n.decode() if isinstance(n, bytes) else n for n in f['kp_names'][:]]
            except Exception:
                pass

        return setup_mujoco_model(xml_path, kp_names)


def qpos_to_keypoints(
    qpos_data: np.ndarray,
    mjx_model: Any,
    mjx_data: Any,
    site_ids: list[int],
) -> np.ndarray:
    return qpos_to_keypoints_mujoco(qpos_data, mjx_model, mjx_data, site_ids)


def prepare_kpms_coordinates(
    data: np.ndarray,
    prefix: str = "recording",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n_clips = data.shape[0]
    n_frames = data.shape[1]

    if data.ndim == 3:
        n_keypoints = data.shape[2]
    else:
        n_keypoints = data.shape[2]

    coordinates = {}
    confidences = {}

    for i in range(n_clips):
        key = f"{prefix}_{i:04d}"
        coords = np.array(data[i]).astype(np.float64)
        coordinates[key] = coords
        if coords.ndim == 3:
            confidences[key] = np.ones((n_frames, coords.shape[1]), dtype=np.float64)
        else:
            confidences[key] = np.ones((n_frames, n_keypoints), dtype=np.float64)

    return coordinates, confidences


def prepare_intention_for_kpms(
    intention_data: np.ndarray,
    prefix: str = "recording",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n_clips, n_frames, n_dim = intention_data.shape

    coordinates = {}
    confidences = {}

    for i in range(n_clips):
        key = f"{prefix}_{i:04d}"
        coords = intention_data[i][:, :, np.newaxis].astype(np.float64)
        coordinates[key] = coords
        confidences[key] = np.ones((n_frames, n_dim), dtype=np.float64)

    return coordinates, confidences


def prepare_keypoints_for_kpms(
    keypoint_data: np.ndarray,
    prefix: str = "recording",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n_clips, n_frames, n_keypoints, _ = keypoint_data.shape

    coordinates = {}
    confidences = {}

    for i in range(n_clips):
        key = f"{prefix}_{i:04d}"
        coords = np.array(keypoint_data[i]).astype(np.float64)
        coordinates[key] = coords
        confidences[key] = np.ones((n_frames, n_keypoints), dtype=np.float64)

    return coordinates, confidences
