import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import pickle as pkl
import yaml
from dataclasses import dataclass
from tqdm import tqdm

from vnl_tda.pipelines.tda.skeletonization.rendering import render_clip_grid


@dataclass
class RenderConfig:
    dm_results_dir: str = ""
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    output_dir: str = "out_no_cycle_clips"
    num_clips: int = 500
    grid_cols: int = 5
    grid_rows: int = 4
    fps: int = 30
    target_size: tuple[int, int] = (320, 240)


def load_config(config_path: str) -> RenderConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    target_size = cfg.get("target_size", [320, 240])
    if isinstance(target_size, list):
        target_size = tuple(target_size)

    return RenderConfig(
        dm_results_dir=cfg.get("dm_results_dir", ""),
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", RenderConfig.rodent_rollout_dir),
        output_dir=cfg.get("output_dir", RenderConfig.output_dir),
        num_clips=cfg.get("num_clips", RenderConfig.num_clips),
        grid_cols=cfg.get("grid_cols", RenderConfig.grid_cols),
        grid_rows=cfg.get("grid_rows", RenderConfig.grid_rows),
        fps=cfg.get("fps", RenderConfig.fps),
        target_size=target_size,
    )


def find_no_cycle_clips(
    dm_results_dir: str,
    num_clips: int,
) -> list[int]:
    cycles_path = os.path.join(dm_results_dir, "all_cycle_dm.pkl")
    if not os.path.exists(cycles_path):
        raise FileNotFoundError(f"Cycles file not found: {cycles_path}")

    with open(cycles_path, "rb") as f:
        all_cycle_dm = pkl.load(f)

    no_cycle_clips = []
    for clip_id in range(min(num_clips, len(all_cycle_dm))):
        cycles = all_cycle_dm[clip_id]
        if len(cycles) == 0:
            no_cycle_clips.append(clip_id)

    return no_cycle_clips


def main(config_path: str = "configs/render_no_cycle_clips.yaml"):
    cfg = load_config(config_path)

    if not cfg.dm_results_dir:
        raise ValueError("dm_results_dir must be specified in config")

    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"=== Finding clips with no cycles ===")
    print(f"DM results dir: {cfg.dm_results_dir}")

    no_cycle_clips = find_no_cycle_clips(cfg.dm_results_dir, cfg.num_clips)

    print(f"Found {len(no_cycle_clips)} clips with no cycles out of {cfg.num_clips}")
    print(f"Clip IDs: {no_cycle_clips}")

    if len(no_cycle_clips) == 0:
        print("No clips without cycles found. Nothing to render.")
        return

    with open(os.path.join(cfg.output_dir, "no_cycle_clip_ids.txt"), "w") as f:
        f.write(f"# Clips with no cycles from {cfg.dm_results_dir}\n")
        f.write(f"# Total: {len(no_cycle_clips)} / {cfg.num_clips}\n")
        for cid in no_cycle_clips:
            f.write(f"{cid}\n")

    print(f"\n=== Rendering clips ===")

    clips_per_video = cfg.grid_rows * cfg.grid_cols
    n_videos = (len(no_cycle_clips) + clips_per_video - 1) // clips_per_video

    for vid_idx in range(n_videos):
        start_idx = vid_idx * clips_per_video
        end_idx = min(start_idx + clips_per_video, len(no_cycle_clips))
        batch_clips = no_cycle_clips[start_idx:end_idx]

        out_path = os.path.join(cfg.output_dir, f"no_cycle_clips_batch_{vid_idx}.mp4")

        print(f"Rendering batch {vid_idx + 1}/{n_videos}: clips {batch_clips}")

        render_clip_grid(
            rodent_rollout_dir=cfg.rodent_rollout_dir,
            clip_ids=batch_clips,
            rows=cfg.grid_rows,
            cols=cfg.grid_cols,
            out_path=out_path,
            fps=cfg.fps,
            target_size=cfg.target_size,
            title=f"No-Cycle Clips (batch {vid_idx + 1}/{n_videos})",
            per_tile_labels=True,
            label_fmt="clip_{id}",
        )

        print(f"Saved: {out_path}")

    print(f"\n=== Done ===")
    print(f"Results saved to: {cfg.output_dir}")
    print(f"Total no-cycle clips: {len(no_cycle_clips)}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/extract_filled_space.yaml"
    main(config_path)
