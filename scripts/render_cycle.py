import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import numpy as np
import pickle as pkl
import yaml
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
import h5py

from topo_mimic.dmwk.utils.load_utils import load_from_h5py, load_rollout_from_source
from topo_mimic.dmwk.utils.rendering import render_from_saved_rollout


@dataclass
class RenderCycleConfig:
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    output_dir: str = "outputs/out_render_cycle"
    dm_results_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM/dm_results_m3_tau6_p0.75_f500"
    num_clips_to_render: int = 10
    min_cycle_len: int = 30
    max_cycle_len: int = 500
    target_size: tuple[int, int] = (320, 240)
    framerate: int = 30
    gutter: int = 10
    pad_x: int = 24
    pad_y: int = 24


def load_config(config_path: str) -> RenderCycleConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "target_size" in cfg:
        cfg["target_size"] = tuple(cfg["target_size"])

    return RenderCycleConfig(**cfg)


def load_cycles(dm_results_dir: str, num_clips: int) -> list[list[np.ndarray]]:
    pkl_path = os.path.join(dm_results_dir, "all_cycle_dm.pkl")
    with open(pkl_path, "rb") as f:
        all_cycle_dm = pkl.load(f)
    return all_cycle_dm[:num_clips]


def get_cycle_stats(
    all_cycles: list[list[np.ndarray]],
    min_len: int,
    max_len: int,
) -> list[dict]:
    """
    For each clip, find the max, min, and median length cycles.

    Returns:
        List of dicts with keys: clip_id, max_cycle, min_cycle, median_cycle, all_lengths
    """
    stats = []
    for clip_id, cycles in enumerate(all_cycles):
        valid_cycles = [
            c for c in cycles
            if min_len <= len(c) <= max_len
        ]

        if not valid_cycles:
            stats.append({
                "clip_id": clip_id,
                "max_cycle": None,
                "min_cycle": None,
                "median_cycle": None,
                "all_lengths": [],
            })
            continue

        lengths = [len(c) for c in valid_cycles]
        sorted_by_len = sorted(zip(lengths, valid_cycles), key=lambda x: x[0])

        min_cycle = sorted_by_len[0][1]
        max_cycle = sorted_by_len[-1][1]
        median_idx = len(sorted_by_len) // 2
        median_cycle = sorted_by_len[median_idx][1]

        stats.append({
            "clip_id": clip_id,
            "max_cycle": np.asarray(max_cycle, dtype=int),
            "min_cycle": np.asarray(min_cycle, dtype=int),
            "median_cycle": np.asarray(median_cycle, dtype=int),
            "all_lengths": lengths,
        })

    return stats


def load_clip_frames(rodent_rollout_dir: str, clip_id: int) -> list[np.ndarray]:
    rollout = load_rollout_from_source(clip_id, rodent_rollout_dir)
    frames = render_from_saved_rollout(rollout)
    return [np.asarray(fr) for fr in frames]


def ensure_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def render_cycle_grid(
    cycle_stats: list[dict],
    rodent_rollout_dir: str,
    output_path: str,
    cfg: RenderCycleConfig,
) -> str:
    """
    Render a grid with rows = clips, cols = [min, median, max] cycles.
    """
    valid_stats = [s for s in cycle_stats if s["max_cycle"] is not None]
    if not valid_stats:
        raise ValueError("No valid cycles to render")

    n_clips = min(cfg.num_clips_to_render, len(valid_stats))
    valid_stats = valid_stats[:n_clips]

    n_rows = n_clips
    n_cols = 3
    col_names = ["Min Cycle", "Median Cycle", "Max Cycle"]

    Wc, Hc = cfg.target_size

    try:
        font_label = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_title = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font_label = ImageFont.load_default()
        font_title = ImageFont.load_default()

    label_h = 24
    title_h = 34
    row_label_w = 80

    W = cfg.pad_x * 2 + row_label_w + n_cols * Wc + (n_cols - 1) * cfg.gutter
    H = cfg.pad_y * 2 + title_h + n_rows * (Hc + label_h) + (n_rows - 1) * cfg.gutter

    all_frames = []
    max_T = 0

    for stat in valid_stats:
        clip_id = stat["clip_id"]
        full_frames = load_clip_frames(rodent_rollout_dir, clip_id)

        clip_data = {
            "clip_id": clip_id,
            "cycles": {},
        }

        for cycle_type, cycle in [
            ("min", stat["min_cycle"]),
            ("median", stat["median_cycle"]),
            ("max", stat["max_cycle"]),
        ]:
            if cycle is not None:
                idx = np.clip(cycle, 0, len(full_frames) - 1)
                frames = [full_frames[i] for i in idx]
                clip_data["cycles"][cycle_type] = {
                    "frames": frames,
                    "length": len(cycle),
                }
                max_T = max(max_T, len(frames))

        all_frames.append(clip_data)

    if max_T == 0:
        raise ValueError("No frames to render")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        output_path, (W, H), fps=cfg.framerate, codec="h264", macro_block_size=1
    )
    writer.send(None)

    def draw_centered(draw, text, x_center, y_center, font, fill=(0, 0, 0)):
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        draw.text((x_center - tw / 2, y_center - th / 2), text, font=font, fill=fill)

    for t in range(max_T):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        draw_centered(
            draw,
            "Cycle Length Comparison (Min / Median / Max)",
            W // 2,
            cfg.pad_y + title_h // 2,
            font_title,
        )

        y0 = cfg.pad_y + title_h
        x0 = cfg.pad_x + row_label_w

        for col, col_name in enumerate(col_names):
            x = x0 + col * (Wc + cfg.gutter) + Wc // 2
            draw_centered(draw, col_name, x, y0 + label_h // 2, font_label)

        for row, clip_data in enumerate(all_frames):
            clip_id = clip_data["clip_id"]
            y = y0 + label_h + row * (Hc + label_h + cfg.gutter)

            draw_centered(
                draw,
                f"Clip {clip_id}",
                cfg.pad_x + row_label_w // 2,
                y + Hc // 2,
                font_label,
            )

            for col, cycle_type in enumerate(["min", "median", "max"]):
                x = x0 + col * (Wc + cfg.gutter)

                if cycle_type in clip_data["cycles"]:
                    cycle_info = clip_data["cycles"][cycle_type]
                    frames = cycle_info["frames"]
                    length = cycle_info["length"]

                    frame_idx = t % len(frames)
                    fr = ensure_rgb_uint8(frames[frame_idx])
                    img = Image.fromarray(fr).resize((Wc, Hc))
                    canvas.paste(img, (x, y))

                    len_text = f"L={length}"
                    draw.text((x + 5, y + 5), len_text, font=font_label, fill=(255, 255, 0))

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
    return output_path


def main(config_path: str = "configs/render/render_cycle.yaml"):
    cfg = load_config(config_path)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=== Loading cycles ===")
    all_cycles = load_cycles(cfg.dm_results_dir, num_clips=500)
    print(f"Loaded cycles for {len(all_cycles)} clips")

    print("\n=== Computing cycle stats ===")
    cycle_stats = get_cycle_stats(all_cycles, cfg.min_cycle_len, cfg.max_cycle_len)

    valid_count = sum(1 for s in cycle_stats if s["max_cycle"] is not None)
    print(f"Found {valid_count} clips with valid cycles")

    print("\n=== Rendering grid ===")
    output_path = os.path.join(cfg.output_dir, "cycle_comparison.mp4")
    render_cycle_grid(cycle_stats, cfg.rodent_rollout_dir, output_path, cfg)
    print(f"Video saved to: {output_path}")

    print("\n=== Summary ===")
    for stat in cycle_stats[:cfg.num_clips_to_render]:
        if stat["max_cycle"] is not None:
            print(
                f"Clip {stat['clip_id']:3d}: "
                f"min={len(stat['min_cycle']):3d}, "
                f"median={len(stat['median_cycle']):3d}, "
                f"max={len(stat['max_cycle']):3d}, "
                f"n_cycles={len(stat['all_lengths'])}"
            )


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/render/render_cycle.yaml"
    main(config_path)
