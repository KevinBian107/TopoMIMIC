"""Feature-space transforms for multi-space DM pipeline."""

import os
import numpy as np


def strip_root_dims(data: np.ndarray, skip_dims: int = 7) -> np.ndarray:
    return data[:, :, skip_dims:]


def render_keypoint_video(
    keypoints: np.ndarray,
    kp_names: list[str],
    output_path: str,
    clip_idx: int = 0,
    fps: int = 30,
    figsize: tuple[int, int] = (10, 8),
    max_frames: int = 100,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    kp_clip = keypoints[clip_idx]
    n_frames, n_kp, _ = kp_clip.shape

    # Subsample frames if too many (for faster rendering)
    if n_frames > max_frames:
        frame_indices = np.linspace(0, n_frames - 1, max_frames, dtype=int)
        kp_clip = kp_clip[frame_indices]
        n_frames = max_frames

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    all_x = kp_clip[:, :, 0].flatten()
    all_y = kp_clip[:, :, 1].flatten()
    all_z = kp_clip[:, :, 2].flatten()
    margin = 0.1
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    z_range = all_z.max() - all_z.min()

    skeleton_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 6), (6, 7), (7, 8), (8, 9),
        (0, 11), (11, 12), (12, 13), (13, 14),
        (0, 16), (16, 17), (17, 18), (18, 19),
    ]
    valid_connections = [(i, j) for i, j in skeleton_connections if i < n_kp and j < n_kp]

    scatter = ax.scatter([], [], [], s=50, c='blue')
    lines = [ax.plot([], [], [], 'b-', linewidth=2)[0] for _ in valid_connections]
    title = ax.set_title('')

    def init():
        ax.set_xlim(all_x.min() - margin * x_range, all_x.max() + margin * x_range)
        ax.set_ylim(all_y.min() - margin * y_range, all_y.max() + margin * y_range)
        ax.set_zlim(all_z.min() - margin * z_range, all_z.max() + margin * z_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return [scatter] + lines + [title]

    def update(frame):
        pts = kp_clip[frame]
        scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

        for line, (i, j) in zip(lines, valid_connections):
            line.set_data([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]])
            line.set_3d_properties([pts[i, 2], pts[j, 2]])

        title.set_text(f'Frame {frame}/{n_frames}')
        return [scatter] + lines + [title]

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False, interval=1000/fps)

    # Use .gif extension with pillow writer (no ffmpeg required)
    output_path = output_path.replace('.mp4', '.gif')
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved keypoint animation to {output_path}")


def compute_egocentric_keypoints(
    qpos_data: np.ndarray,
    xml_path: str,
    stac_data_path: str,
    render_output_dir: str | None = None,
) -> np.ndarray:
    from topo_mimic.benchmark.hyperparam_sweep.keypoint_loader import (
        setup_stac_reference,
        qpos_to_keypoints,
    )

    mjx_model, mjx_data, site_ids, kp_names = setup_stac_reference(
        xml_path, stac_data_path
    )
    print(f"FK setup: {len(site_ids)} keypoints ({kp_names[:3]}...)")

    keypoints = qpos_to_keypoints(qpos_data, mjx_model, mjx_data, site_ids)
    print(f"FK output shape: {keypoints.shape}")

    if render_output_dir is not None:
        render_keypoint_video(
            keypoints, kp_names,
            os.path.join(render_output_dir, "keypoints_world.mp4"),
            clip_idx=0, fps=30,
        )

    root_pos = qpos_data[:, :, :3]
    root_quat = qpos_data[:, :, 3:7]

    keypoints_centered = keypoints - root_pos[:, :, np.newaxis, :]

    heading = _heading_from_quat(root_quat)
    keypoints_ego = _rotate_xy_by_heading(keypoints_centered, -heading)

    if render_output_dir is not None:
        render_keypoint_video(
            keypoints_ego, kp_names,
            os.path.join(render_output_dir, "keypoints_egocentric.mp4"),
            clip_idx=0, fps=30,
        )

    keypoints_ego = keypoints_ego[:, :, 1:, :]

    n_clips, n_frames, n_kp, n_dim = keypoints_ego.shape
    print(f"Egocentric keypoints: {n_kp} keypoints x {n_dim} dims = {n_kp * n_dim} features")
    return keypoints_ego.reshape(n_clips, n_frames, n_kp * n_dim)


def _heading_from_quat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    heading = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return heading


def _rotate_xy_by_heading(
    points: np.ndarray,
    heading: np.ndarray,
) -> np.ndarray:
    cos_h = np.cos(heading)[..., np.newaxis]
    sin_h = np.sin(heading)[..., np.newaxis]

    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    x_rot = cos_h * x - sin_h * y
    y_rot = sin_h * x + cos_h * y

    return np.stack([x_rot, y_rot, z], axis=-1)


def apply_dm_key_transform(
    data: np.ndarray,
    dm_key: str,
    xml_path: str = "topo_mimic/dmwk/xml/rodent.xml",
    stac_data_path: str = "data/rodent_reference_clips.h5",
    render_output_dir: str | None = None,
    dataset: str = "rodent",
) -> np.ndarray:
    if dm_key == "intention":
        return data
    elif dm_key == "qpos":
        from topo_mimic.dmwk.utils.dataset_config import get_dataset_preset
        preset = get_dataset_preset(dataset)
        if preset.root_strip_dims == 0:
            return data
        return strip_root_dims(data, skip_dims=preset.root_strip_dims)
    elif dm_key == "ekp":
        return compute_egocentric_keypoints(data, xml_path, stac_data_path, render_output_dir)
    else:
        raise ValueError(f"Unknown dm_key: {dm_key}. Expected 'intention', 'qpos', or 'ekp'.")


def get_hdf5_key_for_dm_key(dm_key: str) -> str:
    if dm_key == "intention":
        return "intention"
    elif dm_key in ("qpos", "ekp"):
        return "qpos_rollout"
    else:
        raise ValueError(f"Unknown dm_key: {dm_key}")
