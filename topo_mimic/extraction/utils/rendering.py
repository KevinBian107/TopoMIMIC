"""Mujoco rendering help"""

import os

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import HTML

import mujoco
from pathlib import Path
import matplotlib.animation as animation


def render_from_saved_rollout(
    rollout: dict,
    camera_name: str = "close_profile",
) -> list[np.ndarray]:
    """
    Render a rollout from saved qposes.

    Args:
        rollout: Dictionary containing trajectory data (must have 'qpos' key)
        camera_name: Camera to use for rendering

    Returns:
        List of rendered frames as numpy arrays
    """
    qposes = rollout["qpos"]
    
    xml_path = str(
        (Path(__file__).parent / ".." / "xml" / "rodent.xml").resolve()
    )
    
    spec = mujoco.MjSpec()
    spec = spec.from_file(xml_path)

    for geom in spec.geoms:
        if geom.size is not None:
            geom.size *= 0.95
        if geom.pos is not None:
            geom.pos *= 0.95

    mj_model = spec.compile()

    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6
    
    mj_data = mujoco.MjData(mj_model)
    
    mujoco.mj_kinematics(mj_model, mj_data)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    
    frames = []
    print("MuJoCo Rendering...")
    for qpos in tqdm(qposes):
        mj_data.qpos = qpos
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=camera_name)
        pixels = renderer.render()
        frames.append(pixels)

    return frames


def display_video(
    frames: list[np.ndarray],
    framerate: int = 30,
) -> HTML:
    """
    Display a video from a list of frames in a Jupyter notebook.

    Args:
        frames: List of frames, each of shape (height, width, 3)
        framerate: Framerate of the video

    Returns:
        HTML object containing the video
    """
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.close("all")  # Figure auto-closing upon backend switching is deprecated.
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(
        fig=fig, func=update, frames=frames, interval=interval, blit=True, repeat=False
    )
    return HTML(anim.to_html5_video())


def display_loops_grid(
    loops_dict: dict[int, dict],
    framerate: int = 20,
    loops_per_row: int = 5,
) -> HTML:
    """
    Display multiple loops side by side in a grid layout.

    Args:
        loops_dict: Dictionary mapping loop_id to dict with 'frames' key
        framerate: Framerate of the video
        loops_per_row: Number of loops to display per row

    Returns:
        HTML object containing the grid video
    """
    num_loops = len(loops_dict)
    num_rows = int(np.ceil(num_loops / loops_per_row))
    num_cols = min(num_loops, loops_per_row)
    
    loop_ids = list(loops_dict.keys())
    max_frames = max(len(loops_dict[lid]['frames']) for lid in loop_ids)
    
    first_frame = loops_dict[loop_ids[0]]['frames'][0]
    height, width, _ = first_frame.shape
    
    dpi = 70
    fig_width = (width * num_cols) / dpi
    fig_height = (height * num_rows) / dpi
    
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), dpi=dpi)
    plt.close("all")
    matplotlib.use(orig_backend)
    
    if num_loops == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    images = []
    for idx, loop_id in enumerate(loop_ids):
        row = idx // loops_per_row
        col = idx % loops_per_row
        ax = axes[row, col]
        
        ax.set_axis_off()
        ax.set_aspect("equal")
        
        frames = loops_dict[loop_id]['frames']
        im = ax.imshow(frames[0])
        ax.set_title(f"Loop {loop_id} ({len(frames)} frames)", fontsize=8, pad=2)
        images.append((im, frames, loop_id))
    
    for row in range(num_rows):
        for col in range(num_cols):
            idx = row * loops_per_row + col
            if idx >= num_loops:
                axes[row, col].axis('off')
    
    plt.tight_layout()
    
    def update(frame_num):
        artists = []
        for im, frames, loop_id in images:
            if frame_num < len(frames):
                im.set_data(frames[frame_num])
            else:
                im.set_data(frames[-1])
            artists.append(im)
        return artists
    
    interval = 1000 / framerate
    anim = animation.FuncAnimation(
        fig=fig, func=update, frames=max_frames, interval=interval, blit=True, repeat=True
    )
    return HTML(anim.to_html5_video())