"""
Varying K Experiment

Tests DM-G behavior on moving vs stationary clips across different k values.
Runs DM-G separately on each clip and compares cycle detection.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import imageio_ffmpeg

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from topo_mimic.extraction.utils.load_utils import (
    get_clip_indices,
    load_rollout_from_source,
)
from topo_mimic.extraction.skeletonization.skeletonization import extract_skeleton_pcd_dm
from topo_mimic.extraction.utils.rendering import render_from_saved_rollout


@dataclass
class DataConfig:
    rodent_rollout_dir: str = ""
    split: str = "train"
    intention_key: str = "intention"
    qpos_key: str = "qpos_rollout"
    skip_clips: list[int] = field(default_factory=list)


@dataclass
class ClipSelectionConfig:
    min_displacement: float = 0.1
    seed: int = 42
    moving_clip_id: int | None = None
    stationary_clip_id: int | None = None


@dataclass
class DMConfig:
    m: int = 3
    tau: int = 6
    k_values: list[int] = field(default_factory=lambda: [3, 5, 7, 10, 15, 20])
    pers_delta: float = 1.0
    epsilon: float = 0.99


@dataclass
class RenderingConfig:
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)
    camera_name: str = "close_profile"


@dataclass
class VaryingKConfig:
    output_dir: str = "outputs/varying_k"
    data: DataConfig = field(default_factory=DataConfig)
    clip_selection: ClipSelectionConfig = field(default_factory=ClipSelectionConfig)
    dm: DMConfig = field(default_factory=DMConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)


def load_config(config_path: str) -> VaryingKConfig:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    config = VaryingKConfig(
        output_dir=cfg_dict.get("output_dir", "outputs/varying_k"),
        data=DataConfig(**cfg_dict.get("data", {})),
        clip_selection=ClipSelectionConfig(**cfg_dict.get("clip_selection", {})),
        dm=DMConfig(**cfg_dict.get("dm", {})),
        rendering=RenderingConfig(**{
            k: tuple(v) if k == "target_size" else v
            for k, v in cfg_dict.get("rendering", {}).items()
        }),
    )
    return config


def load_intention_data(config: DataConfig, clip_indices: list[int]) -> np.ndarray:
    with h5py.File(config.rodent_rollout_dir, "r") as f:
        intention = np.array(f[config.intention_key])
    return intention[clip_indices]


def load_qpos_data(config: DataConfig, clip_indices: list[int]) -> np.ndarray:
    with h5py.File(config.rodent_rollout_dir, "r") as f:
        qpos = np.array(f[config.qpos_key])
    return qpos[clip_indices]


def classify_clips(
    qpos: np.ndarray,
    clip_indices: list[int],
    min_displacement: float,
) -> tuple[list[int], list[int]]:
    moving = []
    stationary = []

    for i, clip_id in enumerate(clip_indices):
        root_xy = qpos[i, :, :2]
        displacement = np.linalg.norm(root_xy[-1] - root_xy[0])
        total_path = np.sum(np.linalg.norm(np.diff(root_xy, axis=0), axis=1))

        if displacement > min_displacement or total_path > min_displacement * 3:
            moving.append(clip_id)
        else:
            stationary.append(clip_id)

    return moving, stationary


def takens_embedding(X: np.ndarray, m: int, tau: int) -> np.ndarray:
    n_frames, D = X.shape
    start = (m - 1) * tau
    idxs = np.arange(start, n_frames)
    pts = np.stack(
        [X[[j - a * tau for a in range(m)]].reshape(-1) for j in idxs], axis=0
    )
    return pts


def run_dm_single_clip(
    intention: np.ndarray,
    k: int,
    config: DMConfig,
    output_dir: str,
) -> tuple[list[np.ndarray], int]:
    sys.path.append(str(PROJECT_ROOT / "PCD-Graph-Recon-DM"))

    original_dir = os.getcwd()

    pts = takens_embedding(intention, m=config.m, tau=config.tau)

    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)

    try:
        cycle_inds, G_sm = extract_skeleton_pcd_dm(
            pts,
            k=k,
            pers_delta=config.pers_delta,
            epsilon=config.epsilon,
            output_dir=output_dir,
            cleanup=False,
        )
    finally:
        os.chdir(original_dir)

    cycle_inds = [np.array(inds, dtype=int) for inds in cycle_inds]
    n_edges = G_sm.number_of_edges()

    return cycle_inds, n_edges


def get_longest_cycle_length(cycles: list[np.ndarray]) -> int:
    if not cycles:
        return 0
    valid = [c for c in cycles if len(c) >= 3]
    if not valid:
        return 0
    return max(len(c) for c in valid)


def get_longest_cycle_frames(
    cycles: list[np.ndarray],
    dm_config: DMConfig,
) -> list[int]:
    if not cycles:
        return []
    valid = [c for c in cycles if len(c) >= 3]
    if not valid:
        return []
    longest = max(valid, key=len)
    offset = (dm_config.m - 1) * dm_config.tau
    return [int(idx) + offset for idx in longest]


def render_varying_k_comparison(
    rodent_rollout_dir: str,
    moving_clip_id: int,
    stationary_clip_id: int,
    k_values: list[int],
    moving_results: dict[int, list[np.ndarray]],
    stationary_results: dict[int, list[np.ndarray]],
    dm_config: DMConfig,
    output_path: str,
    render_config: RenderingConfig,
) -> str:
    """
    Render comparison video with:
    - Row 1: Moving clip original + longest cycle frames for each k
    - Row 2: Stationary clip original + longest cycle frames for each k
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    moving_rollout = load_rollout_from_source(moving_clip_id, rodent_rollout_dir)
    moving_frames = [np.asarray(f, dtype=np.uint8) for f in
                     render_from_saved_rollout(moving_rollout, camera_name=render_config.camera_name)]

    stationary_rollout = load_rollout_from_source(stationary_clip_id, rodent_rollout_dir)
    stationary_frames = [np.asarray(f, dtype=np.uint8) for f in
                         render_from_saved_rollout(stationary_rollout, camera_name=render_config.camera_name)]

    moving_cycle_frames = {}
    stationary_cycle_frames = {}
    for k in k_values:
        moving_cycle_frames[k] = set(get_longest_cycle_frames(moving_results[k], dm_config))
        stationary_cycle_frames[k] = set(get_longest_cycle_frames(stationary_results[k], dm_config))

    n_cols = 1 + len(k_values)
    n_rows = 2
    Wc, Hc = render_config.target_size
    gutter = 10
    label_h = 40
    title_h = 50

    W = n_cols * Wc + (n_cols + 1) * gutter
    H = title_h + n_rows * (Hc + label_h) + (n_rows + 1) * gutter

    W = W + (W % 2)
    H = H + (H % 2)

    max_frames = max(len(moving_frames), len(stationary_frames))

    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        font_label = ImageFont.truetype("DejaVuSans.ttf", 12)
    except:
        font_title = ImageFont.load_default()
        font_label = font_title

    writer = imageio_ffmpeg.write_frames(
        output_path, (W, H), fps=render_config.framerate, codec="h264", macro_block_size=1
    )
    writer.send(None)

    last_cycle_frame_moving = {k: None for k in k_values}
    last_cycle_frame_stationary = {k: None for k in k_values}

    for k in k_values:
        if moving_cycle_frames[k]:
            last_cycle_frame_moving[k] = min(moving_cycle_frames[k])
        if stationary_cycle_frames[k]:
            last_cycle_frame_stationary[k] = min(stationary_cycle_frames[k])

    for t in tqdm(range(max_frames), desc="Rendering comparison"):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        title = "DM-G Varying K: Moving vs Stationary"
        tw = draw.textlength(title, font=font_title)
        draw.text((W // 2 - tw // 2, 10), title, font=font_title, fill=(0, 0, 0))

        row_data = [
            (f"Moving (Clip {moving_clip_id})", moving_frames, moving_cycle_frames, last_cycle_frame_moving),
            (f"Stationary (Clip {stationary_clip_id})", stationary_frames, stationary_cycle_frames, last_cycle_frame_stationary),
        ]

        for row_idx, (row_label, frames, cycle_frames_dict, last_cycle_dict) in enumerate(row_data):
            y = title_h + row_idx * (Hc + label_h + gutter) + gutter

            x = gutter
            label = f"{row_label}"
            tw = draw.textlength(label, font=font_label)
            draw.text((x + Wc // 2 - tw // 2, y), label, font=font_label, fill=(0, 0, 0))

            frame_idx = t % len(frames)
            img = Image.fromarray(frames[frame_idx]).resize((Wc, Hc), Image.LANCZOS)
            canvas.paste(img, (x, y + label_h))

            for k_idx, k in enumerate(k_values):
                x = gutter + (1 + k_idx) * (Wc + gutter)
                cycle_len = get_longest_cycle_length(moving_results[k] if row_idx == 0 else stationary_results[k])
                label = f"k={k} (L={cycle_len})"
                tw = draw.textlength(label, font=font_label)
                draw.text((x + Wc // 2 - tw // 2, y), label, font=font_label, fill=(0, 0, 0))

                cycle_set = cycle_frames_dict[k]
                frame_idx = t % len(frames)

                if frame_idx in cycle_set:
                    last_cycle_dict[k] = frame_idx

                display_frame = last_cycle_dict[k]
                if display_frame is not None and display_frame < len(frames):
                    img = Image.fromarray(frames[display_frame]).resize((Wc, Hc), Image.LANCZOS)
                    canvas.paste(img, (x, y + label_h))
                    if frame_idx in cycle_set:
                        draw.rectangle(
                            [x, y + label_h, x + Wc - 1, y + label_h + Hc - 1],
                            outline=(255, 0, 0), width=3
                        )
                else:
                    draw.rectangle(
                        [x, y + label_h, x + Wc - 1, y + label_h + Hc - 1],
                        fill=(220, 220, 220), outline=(180, 180, 180)
                    )
                    if cycle_len == 0:
                        draw.text((x + Wc // 2 - 30, y + label_h + Hc // 2 - 6),
                                  "No cycle", font=font_label, fill=(120, 120, 120))

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_k_vs_cycle_length(
    k_values: list[int],
    moving_results: dict[int, list[np.ndarray]],
    stationary_results: dict[int, list[np.ndarray]],
    moving_clip_id: int,
    stationary_clip_id: int,
    output_path: str,
) -> None:
    moving_lengths = [get_longest_cycle_length(moving_results[k]) for k in k_values]
    stationary_lengths = [get_longest_cycle_length(stationary_results[k]) for k in k_values]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(k_values, moving_lengths, 'o-', linewidth=2, markersize=8,
            label=f"Moving (Clip {moving_clip_id})", color='tab:blue')
    ax.plot(k_values, stationary_lengths, 's--', linewidth=2, markersize=8,
            label=f"Stationary (Clip {stationary_clip_id})", color='tab:orange')

    ax.set_xlabel("k (number of neighbors)", fontsize=12)
    ax.set_ylabel("Longest Cycle Length", fontsize=12)
    ax.set_title("DM-G Cycle Detection vs k-NN Parameter", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax.set_xticks(k_values)

    for i, (k, m_len, s_len) in enumerate(zip(k_values, moving_lengths, stationary_lengths)):
        ax.annotate(f"{m_len}", (k, m_len), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=9, color='tab:blue')
        ax.annotate(f"{s_len}", (k, s_len), textcoords="offset points",
                    xytext=(0, -12), ha='center', fontsize=9, color='tab:orange')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_experiment(config: VaryingKConfig) -> dict:
    os.makedirs(config.output_dir, exist_ok=True)

    print("=== Stage 1: Loading data ===")
    clip_indices = get_clip_indices(config.data.split)
    skip_set = set(config.data.skip_clips)
    clip_indices = [c for c in clip_indices if c not in skip_set]
    print(f"Using {len(clip_indices)} clips")

    intention = load_intention_data(config.data, clip_indices)
    qpos = load_qpos_data(config.data, clip_indices)
    print(f"Loaded intention: {intention.shape}, qpos: {qpos.shape}")

    print("\n=== Stage 2: Selecting clips ===")
    np.random.seed(config.clip_selection.seed)

    if config.clip_selection.moving_clip_id is not None and config.clip_selection.stationary_clip_id is not None:
        moving_clip_id = config.clip_selection.moving_clip_id
        stationary_clip_id = config.clip_selection.stationary_clip_id
        print(f"Using specified clips: moving={moving_clip_id}, stationary={stationary_clip_id}")
    else:
        moving_clips, stationary_clips = classify_clips(
            qpos, clip_indices, config.clip_selection.min_displacement
        )
        print(f"Found {len(moving_clips)} moving clips, {len(stationary_clips)} stationary clips")

        if not moving_clips or not stationary_clips:
            print("Error: Need both moving and stationary clips")
            return {"status": "error", "message": "insufficient clips"}

        moving_clip_id = np.random.choice(moving_clips)
        stationary_clip_id = np.random.choice(stationary_clips)
        print(f"Selected: moving={moving_clip_id}, stationary={stationary_clip_id}")

    moving_idx = clip_indices.index(moving_clip_id)
    stationary_idx = clip_indices.index(stationary_clip_id)
    moving_intention = intention[moving_idx]
    stationary_intention = intention[stationary_idx]

    print("\n=== Stage 3: Running DM-G with varying k ===")
    moving_results = {}
    stationary_results = {}

    for k in config.dm.k_values:
        print(f"\n  k = {k}")

        print(f"    Running on moving clip {moving_clip_id}...")
        moving_dir = os.path.join(config.output_dir, f"dm_moving_k{k}")
        moving_cycles, moving_edges = run_dm_single_clip(
            moving_intention, k, config.dm, moving_dir
        )
        moving_results[k] = moving_cycles
        moving_len = get_longest_cycle_length(moving_cycles)
        print(f"    Moving: {len(moving_cycles)} cycles, longest={moving_len}, edges={moving_edges}")

        print(f"    Running on stationary clip {stationary_clip_id}...")
        stationary_dir = os.path.join(config.output_dir, f"dm_stationary_k{k}")
        stationary_cycles, stationary_edges = run_dm_single_clip(
            stationary_intention, k, config.dm, stationary_dir
        )
        stationary_results[k] = stationary_cycles
        stationary_len = get_longest_cycle_length(stationary_cycles)
        print(f"    Stationary: {len(stationary_cycles)} cycles, longest={stationary_len}, edges={stationary_edges}")

    print("\n=== Stage 4: Plotting k vs cycle length ===")
    plot_k_vs_cycle_length(
        config.dm.k_values,
        moving_results,
        stationary_results,
        moving_clip_id,
        stationary_clip_id,
        os.path.join(config.output_dir, "k_vs_cycle_length.png"),
    )

    print("\n=== Stage 5: Rendering comparison video ===")
    render_varying_k_comparison(
        config.data.rodent_rollout_dir,
        moving_clip_id,
        stationary_clip_id,
        config.dm.k_values,
        moving_results,
        stationary_results,
        config.dm,
        os.path.join(config.output_dir, "varying_k_comparison.mp4"),
        config.rendering,
    )

    print("\n=== Stage 6: Summary ===")
    summary = {
        "moving_clip_id": int(moving_clip_id),
        "stationary_clip_id": int(stationary_clip_id),
        "k_values": config.dm.k_values,
        "moving_cycle_lengths": {k: get_longest_cycle_length(moving_results[k]) for k in config.dm.k_values},
        "stationary_cycle_lengths": {k: get_longest_cycle_length(stationary_results[k]) for k in config.dm.k_values},
    }

    with open(os.path.join(config.output_dir, "summary.yaml"), "w") as f:
        yaml.dump(summary, f)

    print("\nResults:")
    print(f"  Moving clip {moving_clip_id}:")
    for k in config.dm.k_values:
        print(f"    k={k}: longest cycle = {summary['moving_cycle_lengths'][k]}")
    print(f"  Stationary clip {stationary_clip_id}:")
    for k in config.dm.k_values:
        print(f"    k={k}: longest cycle = {summary['stationary_cycle_lengths'][k]}")

    return summary


def main(config_path: str):
    config = load_config(config_path)
    run_experiment(config)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/scripts/varying_k.yaml"
    main(config_path)
