"""Render side-by-side TRACK (generated) vs STAC (reference) clips for labeling.

Left panel: TRACK = policy rollout (qpos_rollout).
Right panel: STAC = reference motion capture (qpos_ref).
Also writes per-clip qpos CSVs for both panels.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import numpy as np
import h5py
import yaml
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

from topo_mimic.extraction.utils.load_utils import is_consolidated_format
from topo_mimic.extraction.utils.rendering import render_from_saved_rollout


GENERATED_KEY = "qpos_rollout"
REFERENCE_KEY = "qpos_ref"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


@dataclass
class RenderConfig:
    data_path: str = "data/rodent_label.h5"
    output_dir: str = "outputs/labeling/rendered_for_label"
    start_clip: int = 0
    end_clip: int = 642
    camera_name: str = "close_profile"
    framerate: int = 30
    width: int = 640
    height: int = 480
    annotate_panels: bool = True
    force: bool = False


def load_config(config_path: str) -> RenderConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return RenderConfig(**cfg)


def load_qpos_pair(data_path: str, clip_idx: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(data_path, "r") as f:
        if GENERATED_KEY not in f or REFERENCE_KEY not in f:
            raise KeyError(
                f"{data_path} must contain both '{GENERATED_KEY}' and '{REFERENCE_KEY}'; "
                f"found keys: {list(f.keys())}"
            )
        gen = np.array(f[GENERATED_KEY][clip_idx])
        ref = np.array(f[REFERENCE_KEY][clip_idx])
    return gen, ref


def _load_font(size: int = 22) -> ImageFont.ImageFont:
    if os.path.exists(FONT_PATH):
        return ImageFont.truetype(FONT_PATH, size)
    return ImageFont.load_default()


def annotate_frame(frame: np.ndarray, text: str, font: ImageFont.ImageFont) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if dx == 0 and dy == 0:
                continue
            draw.text((12 + dx, 10 + dy), text, fill="black", font=font)
    draw.text((12, 10), text, fill="white", font=font)
    return np.array(img)


def combine_side_by_side(
    left_frames: list[np.ndarray],
    right_frames: list[np.ndarray],
    left_label: str = "TRACK",
    right_label: str = "STAC",
    annotate: bool = True,
) -> list[np.ndarray]:
    n = min(len(left_frames), len(right_frames))
    font = _load_font(22) if annotate else None
    combined = []
    for i in range(n):
        lf = left_frames[i]
        rf = right_frames[i]
        if annotate:
            lf = annotate_frame(lf, left_label, font)
            rf = annotate_frame(rf, right_label, font)
        combined.append(np.concatenate([lf, rf], axis=1))
    return combined


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


def save_qpos_csv(qpos: np.ndarray, output_path: str) -> None:
    header = ",".join([f"q{i}" for i in range(qpos.shape[1])])
    np.savetxt(output_path, qpos, delimiter=",", header=header, comments="", fmt="%.8f")


def render_clip_pair(
    data_path: str,
    clip_idx: int,
    video_path: str,
    track_csv_path: str,
    stac_csv_path: str,
    camera_name: str,
    framerate: int,
    annotate: bool,
) -> None:
    track_qpos, stac_qpos = load_qpos_pair(data_path, clip_idx)

    save_qpos_csv(track_qpos, track_csv_path)
    save_qpos_csv(stac_qpos, stac_csv_path)

    track_frames = render_from_saved_rollout({"qpos": track_qpos}, camera_name=camera_name)
    stac_frames = render_from_saved_rollout({"qpos": stac_qpos}, camera_name=camera_name)

    combined = combine_side_by_side(track_frames, stac_frames, annotate=annotate)
    save_video(combined, video_path, framerate=framerate)


VIDEOS_SUBDIR = "videos"
CSVS_SUBDIR = "csvs"


def _outputs_for_clip(output_dir: str, clip_idx: int) -> tuple[str, str, str]:
    video_path = os.path.join(output_dir, VIDEOS_SUBDIR, f"clip_{clip_idx}.mp4")
    track_csv_path = os.path.join(output_dir, CSVS_SUBDIR, f"clip_{clip_idx}_track_qpos.csv")
    stac_csv_path = os.path.join(output_dir, CSVS_SUBDIR, f"clip_{clip_idx}_stac_qpos.csv")
    return video_path, track_csv_path, stac_csv_path


def main(config_path: str = "configs/labeling/render_for_label.yaml"):
    cfg = load_config(config_path)

    if not is_consolidated_format(cfg.data_path):
        print(f"Warning: {cfg.data_path} is not a consolidated HDF5 file.")
        print("This script expects a consolidated H5 with 'qpos_rollout' and 'qpos_ref'.")

    with h5py.File(cfg.data_path, "r") as f:
        total_clips = f[GENERATED_KEY].shape[0]

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

    os.makedirs(os.path.join(cfg.output_dir, VIDEOS_SUBDIR), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, CSVS_SUBDIR), exist_ok=True)

    for clip_idx in tqdm(range(start, end), desc="Rendering clips"):
        video_path, track_csv_path, stac_csv_path = _outputs_for_clip(cfg.output_dir, clip_idx)

        all_exist = (
            os.path.exists(video_path)
            and os.path.exists(track_csv_path)
            and os.path.exists(stac_csv_path)
        )
        if all_exist and not cfg.force:
            continue

        try:
            render_clip_pair(
                data_path=cfg.data_path,
                clip_idx=clip_idx,
                video_path=video_path,
                track_csv_path=track_csv_path,
                stac_csv_path=stac_csv_path,
                camera_name=cfg.camera_name,
                framerate=cfg.framerate,
                annotate=cfg.annotate_panels,
            )
        except Exception as e:
            print(f"Error rendering clip {clip_idx}: {e}")
            continue

    print(f"Rendering complete. Outputs written to {cfg.output_dir}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/labeling/render_for_label.yaml"
    main(config_path)
