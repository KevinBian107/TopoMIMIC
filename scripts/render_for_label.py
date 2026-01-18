"""Render clips for labeling purposes.

Renders individual clips from consolidated HDF5 format to MP4 files.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import numpy as np
import yaml
from dataclasses import dataclass
from tqdm import tqdm
import imageio_ffmpeg

from topo_mimic.dmwk.utils.load_utils import (
    is_consolidated_format,
    get_num_clips,
    load_rollout_from_source,
)
from topo_mimic.dmwk.utils.rendering import render_from_saved_rollout


@dataclass
class RenderConfig:
    data_path: str = "data/latest_rodent.h5"
    output_dir: str = "out/rendered_for_label"
    start_clip: int = 642
    end_clip: int = 842
    camera_name: str = "close_profile"
    framerate: int = 30
    width: int = 640
    height: int = 480


def load_config(config_path: str) -> RenderConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return RenderConfig(**cfg)


def save_video(
    frames: list[np.ndarray],
    output_path: str,
    framerate: int = 30,
) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        output_path,
        (w, h),
        fps=framerate,
        codec="h264",
        macro_block_size=1,
    )
    writer.send(None)
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        writer.send(frame)
    writer.close()


def render_single_clip(
    data_path: str,
    clip_idx: int,
    output_path: str,
    camera_name: str = "close_profile",
    framerate: int = 30,
) -> None:
    rollout = load_rollout_from_source(clip_idx, data_path)
    frames = render_from_saved_rollout(rollout, camera_name=camera_name)
    save_video(frames, output_path, framerate=framerate)


def main(config_path: str = "configs/render/render_for_label.yaml"):
    cfg = load_config(config_path)

    if not is_consolidated_format(cfg.data_path):
        print(f"Warning: {cfg.data_path} is not a consolidated HDF5 file.")
        print("This script is designed for consolidated format but will attempt to proceed.")

    total_clips = get_num_clips(cfg.data_path)
    print(f"Data source: {cfg.data_path}")
    print(f"Total clips available: {total_clips}")

    start = max(0, cfg.start_clip)
    end = min(total_clips, cfg.end_clip)
    num_clips = end - start

    if num_clips <= 0:
        print(f"No clips to render (start={start}, end={end})")
        return

    print(f"Rendering clips {start} to {end - 1} ({num_clips} clips)")
    print(f"Output directory: {cfg.output_dir}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    for clip_idx in tqdm(range(start, end), desc="Rendering clips"):
        output_path = os.path.join(cfg.output_dir, f"clip_{clip_idx}.mp4")
        if os.path.exists(output_path):
            continue
        try:
            render_single_clip(
                cfg.data_path,
                clip_idx,
                output_path,
                camera_name=cfg.camera_name,
                framerate=cfg.framerate,
            )
        except Exception as e:
            print(f"Error rendering clip {clip_idx}: {e}")
            continue

    print(f"Rendering complete. Videos saved to {cfg.output_dir}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/render/render_for_label.yaml"
    main(config_path)
