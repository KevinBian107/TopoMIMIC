"""
Batch Incompressibility Experiment

Tests the DM-G incompressibility theory across multiple trials:
1. Moving vs Moving (k=2): Bar graph comparing high-motion vs low-motion clips
2. Stationary vs Moving (n=2,3,4,5,6): Line plot showing compressibility

Theory (from topomimic.pdf):
- DM-G detects changes through incompressibility
- Steady-state behavior (H_p = 0) collapses - compressible
- Behavioral transitions create H_1 != 0 (loops) - incompressible
- Weak Morse Inequalities (m_p >= beta_p) guarantee critical cell retention

Usage:
    python demos/incompressibility/batch_incompressibility.py [config_path]
    Default config: configs/scripts/batch_incompressibility.yaml

Outputs:
    - moving_vs_moving_bar.pdf: Bar graph for k=2 moving comparison
    - stationary_vs_moving_line.pdf: Line plot for varying n
    - trials/: Per-trial intermediate results and renderings
    - results.yaml: Aggregated statistics
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import functools
import multiprocessing as mp
import pickle as pkl
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
import h5py
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
from tqdm import tqdm
from scipy.spatial.distance import cdist

from topo_mimic.dmwk.utils.load_utils import (
    get_aggregate_data,
    get_clip_indices,
    load_rollout_from_source,
)
from topo_mimic.dmwk.utils.rendering import render_from_saved_rollout
from topo_mimic.dmwk.skeletonization.skeletonization import extract_skeleton_pcd_dm


def plot_trial_intention_with_cycles(
    intention_list: list[np.ndarray],
    clip_ids: list[int],
    clip_labels: list[str],
    cycles: list[np.ndarray],
    clip_ranges: list[tuple[int, int]],
    dm_config,
    output_path: str,
) -> None:
    """
    Plot mean intention for each clip with its cycle frames marked.
    Static plot saved as PNG for each trial.
    """
    if not cycles:
        return

    longest_cycle = max(cycles, key=len) if cycles else []
    offset = (dm_config.m - 1) * dm_config.tau

    cycle_frames_by_clip = {}
    for i, clip_id in enumerate(clip_ids):
        start, end = clip_ranges[i]
        frames = set()
        for global_idx in longest_cycle:
            if start <= global_idx < end:
                local_frame = global_idx - start + offset
                if local_frame < len(intention_list[i]):
                    frames.add(local_frame)
        cycle_frames_by_clip[clip_id] = frames

    n_clips = len(clip_ids)
    fig, axes = plt.subplots(n_clips, 1, figsize=(12, 3 * n_clips), sharex=True)
    if n_clips == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_clips))

    for i, (clip_id, intention, label) in enumerate(zip(clip_ids, intention_list, clip_labels)):
        ax = axes[i]
        n_frames = len(intention)
        frames = np.arange(n_frames)
        mean_intention = intention.mean(axis=1)

        ax.plot(frames, mean_intention, color=colors[i], linewidth=2,
                label=f"{label} mean intention", zorder=1)

        cycle_frames = cycle_frames_by_clip[clip_id]
        cycle_mask = np.array([f in cycle_frames for f in frames])
        cycle_frames_arr = frames[cycle_mask]
        cycle_values = mean_intention[cycle_mask]

        if len(cycle_frames_arr) > 0:
            ax.scatter(cycle_frames_arr, cycle_values, color="red", s=60,
                       marker="^", label=f"Cycle frames ({len(cycle_frames_arr)})", zorder=3)

        ax.set_ylabel("Mean Intention", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        n_total = len(cycle_frames)
        stats_text = f"Frames in cycle: {n_total}"
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    axes[-1].set_xlabel("Frame Index", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


@dataclass
class DataConfig:
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar.h5"
    split: str = "train"
    intention_key: str = "intention"
    qpos_key: str = "qpos_rollout"
    skip_clips: list[int] = field(default_factory=list)


@dataclass
class ClipSelectionConfig:
    min_displacement: float = 0.5
    max_intention_variance: float = 0.1
    base_seed: int = 42
    stationary_percentile: float = 20.0


@dataclass
class DMConfig:
    m: int = 3
    tau: int = 6
    pers_delta: float = 0.9
    k_neighbors: int = 7
    epsilon: float = 0.99


@dataclass
class ExperimentConfig:
    n_trials: int = 10
    n_values: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])


@dataclass
class RenderingConfig:
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)
    camera_name: str = "close_profile"
    save_videos: bool = True


@dataclass
class BatchIncompressibilityConfig:
    output_dir: str = "outputs/batch_incompressibility"
    data: DataConfig = None
    clip_selection: ClipSelectionConfig = None
    dm: DMConfig = None
    experiment: ExperimentConfig = None
    rendering: RenderingConfig = None

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.clip_selection is None:
            self.clip_selection = ClipSelectionConfig()
        if self.dm is None:
            self.dm = DMConfig()
        if self.experiment is None:
            self.experiment = ExperimentConfig()
        if self.rendering is None:
            self.rendering = RenderingConfig()


def load_config(config_path: str) -> BatchIncompressibilityConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = DataConfig(**cfg.get("data", {}))
    clip_selection_cfg = ClipSelectionConfig(**cfg.get("clip_selection", {}))
    dm_cfg = DMConfig(**cfg.get("dm", {}))
    experiment_cfg = ExperimentConfig(**cfg.get("experiment", {}))

    render_dict = cfg.get("rendering", {})
    if "target_size" in render_dict:
        render_dict["target_size"] = tuple(render_dict["target_size"])
    render_cfg = RenderingConfig(**render_dict)

    return BatchIncompressibilityConfig(
        output_dir=cfg.get("output_dir", "outputs/batch_incompressibility"),
        data=data_cfg,
        clip_selection=clip_selection_cfg,
        dm=dm_cfg,
        experiment=experiment_cfg,
        rendering=render_cfg,
    )


def load_intention_data(
    config: DataConfig,
    clip_indices: list[int],
) -> np.ndarray:
    work_fn = functools.partial(
        get_aggregate_data,
        f"/{config.intention_key}",
        [],
        path=config.rodent_rollout_dir,
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        data_list = list(
            tqdm(
                pool.imap(work_fn, clip_indices),
                total=len(clip_indices),
                desc="Loading intention",
            )
        )

    first_clip = data_list[0]
    if first_clip.ndim == 2:
        n_frames, n_dim = first_clip.shape
    else:
        total_size = first_clip.size
        n_dim = total_size // 500
        n_frames = 500

    data = np.array(
        [clip.reshape(-1, n_dim) if clip.ndim != 2 else clip for clip in data_list]
    )
    min_frames = min(d.shape[0] for d in data)
    data = data[:, :min_frames, :]
    return data


def load_qpos_data(
    config: DataConfig,
    clip_indices: list[int],
) -> np.ndarray:
    work_fn = functools.partial(
        get_aggregate_data,
        f"/{config.qpos_key}",
        [],
        path=config.rodent_rollout_dir,
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        data_list = list(
            tqdm(
                pool.imap(work_fn, clip_indices),
                total=len(clip_indices),
                desc="Loading qpos",
            )
        )

    first_clip = data_list[0]
    if first_clip.ndim == 2:
        n_frames, n_dim = first_clip.shape
    else:
        total_size = first_clip.size
        n_dim = total_size // 500
        n_frames = 500

    data = np.array(
        [clip.reshape(-1, n_dim) if clip.ndim != 2 else clip for clip in data_list]
    )
    min_frames = min(d.shape[0] for d in data)
    data = data[:, :min_frames, :]
    return data


def compute_intention_variance(intention: np.ndarray) -> float:
    return np.var(intention, axis=0).sum()


def classify_clips(
    intention: np.ndarray,
    qpos: np.ndarray,
    clip_indices: list[int],
    min_displacement: float,
    max_intention_variance: float,
    stationary_percentile: float = 20.0,
) -> tuple[list[int], list[int], dict[int, float]]:
    """
    Classify clips into moving and stationary categories.

    Uses a two-stage approach:
    1. Physical movement classification based on root displacement
    2. For stationary: clips with low physical movement AND low intention variance
       If no clips meet the strict criteria, use bottom percentile of intention variance

    Returns:
        moving_clips: List of clip IDs that are moving
        stationary_clips: List of clip IDs that are stationary
        intention_variances: Dict mapping clip_id -> intention variance
    """
    intention_variances = {}
    physical_movement = {}

    for i, clip_id in enumerate(clip_indices):
        root_xy = qpos[i, :, :2]
        displacement = np.linalg.norm(root_xy[-1] - root_xy[0])
        total_path = np.sum(np.linalg.norm(np.diff(root_xy, axis=0), axis=1))

        intent_var = compute_intention_variance(intention[i])
        intention_variances[clip_id] = intent_var
        physical_movement[clip_id] = (displacement, total_path)

    all_variances = list(intention_variances.values())
    all_displacements = [physical_movement[c][0] for c in clip_indices]

    print(f"\n  Intention variance: min={min(all_variances):.4f}, "
          f"max={max(all_variances):.4f}, median={np.median(all_variances):.4f}")
    print(f"  Root displacement: min={min(all_displacements):.4f}, "
          f"max={max(all_displacements):.4f}, median={np.median(all_displacements):.4f}")

    moving_clips = []
    non_moving_clips = []

    for clip_id in clip_indices:
        displacement, total_path = physical_movement[clip_id]
        is_physically_moving = displacement > min_displacement or total_path > min_displacement * 3

        if is_physically_moving:
            moving_clips.append(clip_id)
        else:
            non_moving_clips.append(clip_id)

    print(f"  Physically moving: {len(moving_clips)}, non-moving: {len(non_moving_clips)}")

    stationary_clips = []
    if non_moving_clips:
        for clip_id in non_moving_clips:
            if intention_variances[clip_id] < max_intention_variance:
                stationary_clips.append(clip_id)

        if not stationary_clips:
            print(f"  No clips below intention variance threshold {max_intention_variance}")
            non_moving_variances = [(clip_id, intention_variances[clip_id]) for clip_id in non_moving_clips]
            non_moving_variances.sort(key=lambda x: x[1])
            n_select = max(1, int(len(non_moving_clips) * stationary_percentile / 100))
            stationary_clips = [c[0] for c in non_moving_variances[:n_select]]
            print(f"  Using bottom {stationary_percentile}% ({n_select} clips) by intention variance")
    else:
        print(f"  No physically non-moving clips found!")
        print(f"  Falling back: selecting bottom {stationary_percentile}% by intention variance from all clips")
        all_variances_sorted = sorted(intention_variances.items(), key=lambda x: x[1])
        n_select = max(1, int(len(clip_indices) * stationary_percentile / 100))
        stationary_clips = [c[0] for c in all_variances_sorted[:n_select]]
        moving_clips = [c for c in moving_clips if c not in stationary_clips]

    print(f"  Final: {len(moving_clips)} moving, {len(stationary_clips)} stationary")

    return moving_clips, stationary_clips, intention_variances


def takens_embedding(X: np.ndarray, m: int, tau: int) -> np.ndarray:
    n_frames, D = X.shape
    start = (m - 1) * tau
    idxs = np.arange(start, n_frames)
    pts = np.stack(
        [X[[j - a * tau for a in range(m)]].reshape(-1) for j in idxs], axis=0
    )
    return pts


def run_dm_analysis_combined(
    intention_list: list[np.ndarray],
    clip_ids: list[int],
    config: DMConfig,
    output_dir: str,
) -> tuple[list[np.ndarray], any, np.ndarray, list[tuple[int, int]]]:
    """
    Run DM-G on combined point cloud from multiple clips.

    Returns:
        cycles: List of cycle index arrays (indices into combined pts)
        G_sm: The DM graph
        pts: Combined Takens-embedded points
        clip_ranges: List of (start_idx, end_idx) for each clip in pts
    """
    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")

    original_dir = os.getcwd()

    pts_list = []
    clip_ranges = []
    offset = 0

    for intention in intention_list:
        pts = takens_embedding(intention, m=config.m, tau=config.tau)
        pts_list.append(pts)
        clip_ranges.append((offset, offset + len(pts)))
        offset += len(pts)

    combined_pts = np.vstack(pts_list)

    dm_output_dir = os.path.join(output_dir, "dm_combined")
    os.makedirs(dm_output_dir, exist_ok=True)
    dm_output_dir = os.path.abspath(dm_output_dir)

    try:
        cycle_inds, G_sm = extract_skeleton_pcd_dm(
            combined_pts,
            k=config.k_neighbors,
            pers_delta=config.pers_delta,
            epsilon=config.epsilon,
            output_dir=dm_output_dir,
            cleanup=False,
        )
    finally:
        os.chdir(original_dir)

    cycle_inds = [np.array(inds, dtype=int) for inds in cycle_inds]

    return cycle_inds, G_sm, combined_pts, clip_ranges


def count_frames_in_longest_cycle(
    cycles: list[np.ndarray],
    clip_ranges: list[tuple[int, int]],
    dm_config: DMConfig,
    clip_index: int,
) -> int:
    """
    Count how many frames from a specific clip appear in the longest cycle.
    """
    if not cycles:
        return 0

    valid_cycles = [c for c in cycles if len(c) >= 3]
    if not valid_cycles:
        return 0

    longest_cycle = max(valid_cycles, key=len)
    start, end = clip_ranges[clip_index]
    offset = (dm_config.m - 1) * dm_config.tau

    count = 0
    for global_idx in longest_cycle:
        if start <= global_idx < end:
            count += 1

    return count


def get_cycle_frames_for_clip(
    cycles: list[np.ndarray],
    clip_ranges: list[tuple[int, int]],
    dm_config: DMConfig,
    clip_index: int,
    n_original_frames: int,
) -> set[int]:
    """
    Get the set of original frame indices for a clip that appear in the longest cycle.
    """
    if not cycles:
        return set()

    valid_cycles = [c for c in cycles if len(c) >= 3]
    if not valid_cycles:
        return set()

    longest_cycle = max(valid_cycles, key=len)
    start, end = clip_ranges[clip_index]
    offset = (dm_config.m - 1) * dm_config.tau

    frames = set()
    for global_idx in longest_cycle:
        if start <= global_idx < end:
            local_frame = global_idx - start + offset
            if local_frame < n_original_frames:
                frames.add(local_frame)

    return frames


def render_trial_video(
    rodent_rollout_dir: str,
    clip_ids: list[int],
    intention_list: list[np.ndarray],
    cycles: list[np.ndarray],
    clip_ranges: list[tuple[int, int]],
    clip_labels: list[str],
    output_path: str,
    config: RenderingConfig,
    dm_config: DMConfig,
) -> str:
    """
    Render combined video with intention graph and video panels.
    """
    if not cycles:
        print("No cycles found, skipping video rendering")
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_frames = {}
    for clip_id in clip_ids:
        rollout = load_rollout_from_source(clip_id, rodent_rollout_dir)
        frames = render_from_saved_rollout(rollout, camera_name=config.camera_name)
        all_frames[clip_id] = [np.asarray(f, dtype=np.uint8) for f in frames]

    offset = (dm_config.m - 1) * dm_config.tau

    valid_cycles = [c for c in cycles if len(c) >= 3]
    if not valid_cycles:
        return None

    valid_cycles.sort(key=len, reverse=True)
    longest_cycle = valid_cycles[0]

    cycle_frames_by_clip = {}
    for i, clip_id in enumerate(clip_ids):
        cycle_frames_by_clip[clip_id] = get_cycle_frames_for_clip(
            cycles, clip_ranges, dm_config, i, len(all_frames[clip_id])
        )

    n_clips = len(clip_ids)
    Wc, Hc = 400, 300
    gutter = 15
    label_h = 35
    title_h = 50
    graph_height = 140 * n_clips + 30

    W = n_clips * 2 * Wc + (n_clips * 2 + 1) * gutter + 150
    H = title_h + graph_height + 50 + Hc + label_h + 60
    W = W + (W % 2)
    H = H + (H % 2)

    max_clip_len = max(len(f) for f in all_frames.values())

    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        font_label = ImageFont.truetype("DejaVuSans.ttf", 12)
        font_axis = ImageFont.truetype("DejaVuSans.ttf", 10)
    except:
        font_title = ImageFont.load_default()
        font_label = font_title
        font_axis = font_title

    writer = imageio_ffmpeg.write_frames(
        output_path, (W, H), fps=config.framerate, codec="h264", macro_block_size=1
    )
    writer.send(None)

    colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75)]
    last_cycle_frame = {clip_id: None for clip_id in clip_ids}
    for clip_id in clip_ids:
        if cycle_frames_by_clip[clip_id]:
            last_cycle_frame[clip_id] = min(cycle_frames_by_clip[clip_id])

    for t in tqdm(range(max_clip_len), desc="Rendering video"):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        title = f"DM-G Analysis: {len(longest_cycle)} points in longest cycle"
        tw = draw.textlength(title, font=font_title)
        draw.text((W // 2 - tw // 2, 10), title, font=font_title, fill=(0, 0, 0))

        graph_top = title_h
        graph_left = 100
        graph_right = W - 40
        graph_width = graph_right - graph_left
        panel_height = 120

        for i, (clip_id, intention, label) in enumerate(zip(clip_ids, intention_list, clip_labels)):
            panel_top = graph_top + i * (panel_height + 20) + 10
            panel_bottom = panel_top + panel_height

            draw.rectangle([graph_left, panel_top, graph_right, panel_bottom],
                           outline=(180, 180, 180), width=1)

            mean_int = intention.mean(axis=1)
            n_frames = len(mean_int)
            y_min, y_max = mean_int.min(), mean_int.max()
            y_range = y_max - y_min if y_max > y_min else 1

            points = []
            for f in range(n_frames):
                px = graph_left + (f / n_frames) * graph_width
                py = panel_bottom - ((mean_int[f] - y_min) / y_range) * panel_height * 0.85 - 8
                points.append((px, py))

            for j in range(len(points) - 1):
                draw.line([points[j], points[j + 1]], fill=colors[i % len(colors)], width=2)

            cycle_frames = cycle_frames_by_clip[clip_id]
            for f in cycle_frames:
                if f < n_frames:
                    px = graph_left + (f / n_frames) * graph_width
                    py = panel_bottom - ((mean_int[f] - y_min) / y_range) * panel_height * 0.85 - 8
                    draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 0, 0))

            draw.text((graph_left - 90, panel_top + panel_height // 2 - 8),
                      label, font=font_label, fill=colors[i % len(colors)])

        n_frames_ref = len(intention_list[0])
        time_x = graph_left + (t / n_frames_ref) * graph_width
        draw.line([(time_x, graph_top + 10), (time_x, graph_top + graph_height - 20)],
                  fill=(255, 0, 0), width=2)

        video_y = graph_top + graph_height + 30
        orig_start_x = 50

        for i, (clip_id, label) in enumerate(zip(clip_ids, clip_labels)):
            x = orig_start_x + i * (Wc + gutter)
            n_in_cycle = len(cycle_frames_by_clip[clip_id])
            lbl = f"Original - {label} ({n_in_cycle} pts)"
            tw = draw.textlength(lbl, font=font_label)
            draw.text((x + Wc // 2 - tw // 2, video_y), lbl, font=font_label, fill=(0, 0, 0))

            frames = all_frames[clip_id]
            frame_idx = t % len(frames)
            img = Image.fromarray(frames[frame_idx]).resize((Wc, Hc), Image.LANCZOS)
            canvas.paste(img, (x, video_y + label_h))

            if frame_idx in cycle_frames_by_clip[clip_id]:
                draw.rectangle(
                    [x, video_y + label_h, x + Wc - 1, video_y + label_h + Hc - 1],
                    outline=(255, 0, 0), width=3
                )

        cycle_start_x = orig_start_x + n_clips * (Wc + gutter) + 30

        for i, (clip_id, label) in enumerate(zip(clip_ids, clip_labels)):
            x = cycle_start_x + i * (Wc + gutter)
            n_in_cycle = len(cycle_frames_by_clip[clip_id])
            lbl = f"Cycle - {label}"
            tw = draw.textlength(lbl, font=font_label)
            draw.text((x + Wc // 2 - tw // 2, video_y), lbl, font=font_label, fill=(0, 0, 0))

            frames = all_frames[clip_id]
            frame_idx = t % len(frames)

            if frame_idx in cycle_frames_by_clip[clip_id]:
                last_cycle_frame[clip_id] = frame_idx

            display_frame = last_cycle_frame[clip_id]
            if display_frame is not None:
                img = Image.fromarray(frames[display_frame]).resize((Wc, Hc), Image.LANCZOS)
                canvas.paste(img, (x, video_y + label_h))
                if frame_idx in cycle_frames_by_clip[clip_id]:
                    draw.rectangle(
                        [x, video_y + label_h, x + Wc - 1, video_y + label_h + Hc - 1],
                        outline=(255, 0, 0), width=3
                    )
            else:
                draw.rectangle(
                    [x, video_y + label_h, x + Wc - 1, video_y + label_h + Hc - 1],
                    fill=(220, 220, 220), outline=(180, 180, 180)
                )

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
    print(f"Saved: {output_path}")
    return output_path


def run_moving_vs_moving_trial(
    intention: np.ndarray,
    qpos: np.ndarray,
    clip_indices: list[int],
    moving_clips: list[int],
    intention_variances: dict[int, float],
    trial_seed: int,
    trial_output_dir: str,
    config: BatchIncompressibilityConfig,
) -> dict:
    """
    Run a single trial of moving vs moving (k=2).
    """
    np.random.seed(trial_seed)

    selected_clips = np.random.choice(moving_clips, size=2, replace=False)
    clip_id_1, clip_id_2 = selected_clips

    var_1 = intention_variances[clip_id_1]
    var_2 = intention_variances[clip_id_2]

    if var_1 >= var_2:
        high_motion_clip = clip_id_1
        low_motion_clip = clip_id_2
    else:
        high_motion_clip = clip_id_2
        low_motion_clip = clip_id_1

    clip_ids = [high_motion_clip, low_motion_clip]

    idx_high = clip_indices.index(high_motion_clip)
    idx_low = clip_indices.index(low_motion_clip)
    intention_list = [intention[idx_high], intention[idx_low]]

    os.makedirs(trial_output_dir, exist_ok=True)

    cycles, G_sm, combined_pts, clip_ranges = run_dm_analysis_combined(
        intention_list, clip_ids, config.dm, trial_output_dir
    )

    high_frames = count_frames_in_longest_cycle(cycles, clip_ranges, config.dm, 0)
    low_frames = count_frames_in_longest_cycle(cycles, clip_ranges, config.dm, 1)

    labels = [
        f"High Motion (C{high_motion_clip})",
        f"Low Motion (C{low_motion_clip})"
    ]

    plot_trial_intention_with_cycles(
        intention_list,
        clip_ids,
        labels,
        cycles,
        clip_ranges,
        config.dm,
        os.path.join(trial_output_dir, "intention_with_cycles.png"),
    )

    if config.rendering.save_videos:
        render_trial_video(
            config.data.rodent_rollout_dir,
            clip_ids,
            intention_list,
            cycles,
            clip_ranges,
            labels,
            os.path.join(trial_output_dir, "trial_video.mp4"),
            config.rendering,
            config.dm,
        )

    result = {
        "high_motion_clip": int(high_motion_clip),
        "low_motion_clip": int(low_motion_clip),
        "high_motion_variance": float(intention_variances[high_motion_clip]),
        "low_motion_variance": float(intention_variances[low_motion_clip]),
        "high_motion_frames": int(high_frames),
        "low_motion_frames": int(low_frames),
        "n_cycles": len(cycles),
        "longest_cycle_length": max(len(c) for c in cycles) if cycles else 0,
    }

    with open(os.path.join(trial_output_dir, "result.yaml"), "w") as f:
        yaml.dump(result, f)

    return result


def run_stationary_vs_moving_trial(
    intention: np.ndarray,
    qpos: np.ndarray,
    clip_indices: list[int],
    moving_clips: list[int],
    stationary_clips: list[int],
    intention_variances: dict[int, float],
    n_trajectories: int,
    trial_seed: int,
    trial_output_dir: str,
    config: BatchIncompressibilityConfig,
) -> dict:
    """
    Run a single trial of stationary vs moving with n trajectories.
    1 stationary + (n-1) moving clips.
    """
    np.random.seed(trial_seed)

    n_moving = n_trajectories - 1
    if n_moving > len(moving_clips):
        raise ValueError(f"Need {n_moving} moving clips but only have {len(moving_clips)}")

    stationary_clip = np.random.choice(stationary_clips)
    selected_moving = np.random.choice(moving_clips, size=n_moving, replace=False)

    clip_ids = [stationary_clip] + list(selected_moving)

    intention_list = []
    for clip_id in clip_ids:
        idx = clip_indices.index(clip_id)
        intention_list.append(intention[idx])

    os.makedirs(trial_output_dir, exist_ok=True)

    cycles, G_sm, combined_pts, clip_ranges = run_dm_analysis_combined(
        intention_list, clip_ids, config.dm, trial_output_dir
    )

    stationary_frames = count_frames_in_longest_cycle(cycles, clip_ranges, config.dm, 0)

    moving_frames_list = []
    for i in range(1, len(clip_ids)):
        frames = count_frames_in_longest_cycle(cycles, clip_ranges, config.dm, i)
        moving_frames_list.append(frames)
    moving_frames_total = sum(moving_frames_list)

    labels = [f"Stationary (C{stationary_clip})"]
    for clip_id in selected_moving:
        labels.append(f"Moving (C{clip_id})")

    plot_trial_intention_with_cycles(
        intention_list,
        clip_ids,
        labels,
        cycles,
        clip_ranges,
        config.dm,
        os.path.join(trial_output_dir, "intention_with_cycles.png"),
    )

    if config.rendering.save_videos:
        render_trial_video(
            config.data.rodent_rollout_dir,
            clip_ids,
            intention_list,
            cycles,
            clip_ranges,
            labels,
            os.path.join(trial_output_dir, "trial_video.mp4"),
            config.rendering,
            config.dm,
        )

    result = {
        "n_trajectories": n_trajectories,
        "stationary_clip": int(stationary_clip),
        "moving_clips": [int(c) for c in selected_moving],
        "stationary_frames": int(stationary_frames),
        "moving_frames_list": [int(f) for f in moving_frames_list],
        "moving_frames_total": int(moving_frames_total),
        "n_cycles": len(cycles),
        "longest_cycle_length": max(len(c) for c in cycles) if cycles else 0,
    }

    with open(os.path.join(trial_output_dir, "result.yaml"), "w") as f:
        yaml.dump(result, f)

    return result


def plot_moving_vs_moving_bar(
    results: list[dict],
    output_path: str,
) -> None:
    high_frames = [r["high_motion_frames"] for r in results]
    low_frames = [r["low_motion_frames"] for r in results]

    high_mean = np.mean(high_frames)
    high_sem = np.std(high_frames) / np.sqrt(len(high_frames))
    low_mean = np.mean(low_frames)
    low_sem = np.std(low_frames) / np.sqrt(len(low_frames))

    with plt.rc_context({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7, 'axes.linewidth': 0.6,
        'axes.labelsize': 8, 'axes.titlesize': 9,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
    }):
        fig, ax = plt.subplots(figsize=(3.0, 3.5))

        x = np.array([0, 1])
        colors = ['#4878CF', '#E8A838']

        ax.bar(x, [high_mean, low_mean], 0.55,
               color=colors, edgecolor='none', linewidth=0, zorder=2)
        ax.errorbar(x, [high_mean, low_mean], yerr=[high_sem, low_sem],
                    fmt='none', ecolor='black', elinewidth=0.8, capsize=3, capthick=0.8, zorder=3)

        rng = np.random.default_rng(42)
        for i, data in enumerate([high_frames, low_frames]):
            jitter = rng.uniform(-0.1, 0.1, len(data))
            ax.scatter(np.full(len(data), x[i]) + jitter, data,
                       c='black', s=10, alpha=0.45, zorder=4, linewidths=0)

        ax.set_ylabel("Frames in longest DM-loop")
        ax.set_xticks(x)
        ax.set_xticklabels(["High motion", "Low motion"])
        ax.set_title("Moving vs moving (k = 2)", fontweight='bold')

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(direction='out', length=3, width=0.6)

        for i, (mean, sem) in enumerate(zip([high_mean, low_mean], [high_sem, low_sem])):
            ax.text(x[i], mean + sem + 1.5, f'{mean:.1f}', ha='center', va='bottom', fontsize=7)

        ax.set_ylim(0, max(high_mean, low_mean) * 1.35)

        ax.text(0.95, 0.95, f'n = {len(results)} trials', transform=ax.transAxes,
                ha='right', va='top', fontsize=6.5, fontstyle='italic', color='#555555')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
    print(f"Saved: {output_path}")


def plot_stationary_vs_moving_line(
    results_by_n: dict[int, list[dict]],
    output_path: str,
) -> None:
    with plt.rc_context({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7, 'axes.linewidth': 0.6,
        'axes.labelsize': 8, 'axes.titlesize': 9,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
    }):
        fig, ax = plt.subplots(figsize=(3.8, 3.5))

        n_values = sorted(results_by_n.keys())

        stat_m, stat_s, mov_m, mov_s = [], [], [], []
        for n in n_values:
            results = results_by_n[n]
            sf = [r["stationary_frames"] for r in results]
            mf = [r["moving_frames_total"] for r in results]
            stat_m.append(np.mean(sf)); stat_s.append(np.std(sf) / np.sqrt(len(sf)))
            mov_m.append(np.mean(mf)); mov_s.append(np.std(mf) / np.sqrt(len(mf)))

        stat_m, stat_s = np.array(stat_m), np.array(stat_s)
        mov_m, mov_s = np.array(mov_m), np.array(mov_s)

        ax.fill_between(n_values, mov_m - mov_s, mov_m + mov_s,
                         color='#4878CF', alpha=0.15, linewidth=0)
        ax.plot(n_values, mov_m, 's-', color='#4878CF', linewidth=1.2,
                markersize=5, markeredgecolor='white', markeredgewidth=0.5,
                label='Moving (sum)', zorder=3)

        ax.fill_between(n_values, stat_m - stat_s, stat_m + stat_s,
                         color='#C44E52', alpha=0.15, linewidth=0)
        ax.plot(n_values, stat_m, 'o-', color='#C44E52', linewidth=1.2,
                markersize=5, markeredgecolor='white', markeredgewidth=0.5,
                label='Stationary', zorder=3)

        ax.set_xlabel("Number of trajectories (n)")
        ax.set_ylabel("Frames in longest DM-loop")
        ax.set_title("Stationary vs moving", fontweight='bold')

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(direction='out', length=3, width=0.6)
        ax.set_xticks(n_values)

        ax.legend(loc='upper left', frameon=True, framealpha=0.9,
                  edgecolor='#cccccc', fontsize=6.5, handlelength=1.5)

        n_trials = len(results_by_n[n_values[0]])
        ax.text(0.95, 0.95, f'n = {n_trials} trials per point', transform=ax.transAxes,
                ha='right', va='top', fontsize=6.5, fontstyle='italic', color='#555555')

        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
    print(f"Saved: {output_path}")


def load_cached_trial_result(trial_output_dir: str) -> dict | None:
    """Load cached trial result if it exists."""
    result_path = os.path.join(trial_output_dir, "result.yaml")
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            return yaml.safe_load(f)
    return None


def run_experiment(config: BatchIncompressibilityConfig) -> dict:
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 70)
    print("Batch Incompressibility Experiment")
    print("=" * 70)

    all_results_path = os.path.join(config.output_dir, "all_results.pkl")
    if os.path.exists(all_results_path):
        print(f"\n=== Found existing results at {all_results_path} ===")
        print("Loading cached results and regenerating plots...")

        with open(all_results_path, "rb") as f:
            cached_data = pkl.load(f)

        moving_vs_moving_results = cached_data["moving_vs_moving"]
        stationary_vs_moving_results = cached_data["stationary_vs_moving"]

        print(f"  Loaded {len(moving_vs_moving_results)} moving vs moving trials")
        print(f"  Loaded {len(stationary_vs_moving_results)} n-values for stationary vs moving")

        print("\n=== Regenerating summary plots ===")

        plot_moving_vs_moving_bar(
            moving_vs_moving_results,
            os.path.join(config.output_dir, "moving_vs_moving_bar.pdf"),
        )

        plot_stationary_vs_moving_line(
            stationary_vs_moving_results,
            os.path.join(config.output_dir, "stationary_vs_moving_line.pdf"),
        )

        print("\nPlots regenerated from cached results.")
        print(f"To recompute, delete: {all_results_path}")
        return {"status": "loaded_from_cache"}

    print("\n=== Stage 1: Loading data ===")
    clip_indices = get_clip_indices(config.data.split)
    skip_set = set(config.data.skip_clips)
    clip_indices = [c for c in clip_indices if c not in skip_set]
    print(f"Using {len(clip_indices)} clips (skipping {len(skip_set)} clips)")

    intention = load_intention_data(config.data, clip_indices)
    qpos = load_qpos_data(config.data, clip_indices)
    print(f"Loaded intention: {intention.shape}, qpos: {qpos.shape}")

    print("\n=== Stage 2: Classifying clips ===")
    moving_clips, stationary_clips, intention_variances = classify_clips(
        intention, qpos, clip_indices,
        config.clip_selection.min_displacement,
        config.clip_selection.max_intention_variance,
        config.clip_selection.stationary_percentile,
    )

    if len(moving_clips) < 2:
        print("Error: Need at least 2 moving clips")
        return {"status": "error", "message": "insufficient moving clips"}

    if len(stationary_clips) < 1:
        print("Error: Need at least 1 stationary clip")
        return {"status": "error", "message": "insufficient stationary clips"}

    max_n = max(config.experiment.n_values)
    if len(moving_clips) < max_n - 1:
        print(f"Warning: Need {max_n - 1} moving clips for n={max_n}, but only have {len(moving_clips)}")
        config.experiment.n_values = [n for n in config.experiment.n_values if n - 1 <= len(moving_clips)]
        print(f"Adjusted n_values to: {config.experiment.n_values}")

    print("\n=== Stage 3: Running Moving vs Moving trials (k=2) ===")
    moving_vs_moving_results = []
    trials_dir = os.path.join(config.output_dir, "trials", "moving_vs_moving")

    n_cached_mm = 0
    for trial_idx in range(config.experiment.n_trials):
        trial_seed = config.clip_selection.base_seed + trial_idx
        trial_output_dir = os.path.join(trials_dir, f"trial_{trial_idx:02d}")

        cached_result = load_cached_trial_result(trial_output_dir)
        if cached_result is not None:
            moving_vs_moving_results.append(cached_result)
            n_cached_mm += 1
            continue

        print(f"\n  Trial {trial_idx + 1}/{config.experiment.n_trials}")
        result = run_moving_vs_moving_trial(
            intention, qpos, clip_indices, moving_clips, intention_variances,
            trial_seed, trial_output_dir, config
        )
        moving_vs_moving_results.append(result)
        print(f"    High motion: {result['high_motion_frames']} frames, "
              f"Low motion: {result['low_motion_frames']} frames")

    if n_cached_mm > 0:
        print(f"\n  Loaded {n_cached_mm}/{config.experiment.n_trials} trials from cache")

    print("\n=== Stage 4: Running Stationary vs Moving trials ===")
    stationary_vs_moving_results = {}
    trials_dir = os.path.join(config.output_dir, "trials", "stationary_vs_moving")

    for n in config.experiment.n_values:
        print(f"\n  n = {n} (1 stationary + {n-1} moving)")
        stationary_vs_moving_results[n] = []
        n_cached_sm = 0

        for trial_idx in range(config.experiment.n_trials):
            trial_seed = config.clip_selection.base_seed + trial_idx + n * 1000
            trial_output_dir = os.path.join(trials_dir, f"n_{n}", f"trial_{trial_idx:02d}")

            cached_result = load_cached_trial_result(trial_output_dir)
            if cached_result is not None:
                stationary_vs_moving_results[n].append(cached_result)
                n_cached_sm += 1
                continue

            print(f"    Trial {trial_idx + 1}/{config.experiment.n_trials}")
            result = run_stationary_vs_moving_trial(
                intention, qpos, clip_indices, moving_clips, stationary_clips,
                intention_variances, n, trial_seed, trial_output_dir, config
            )
            stationary_vs_moving_results[n].append(result)
            print(f"      Stationary: {result['stationary_frames']} frames, "
                  f"Moving total: {result['moving_frames_total']} frames")

        if n_cached_sm > 0:
            print(f"    Loaded {n_cached_sm}/{config.experiment.n_trials} trials from cache")

    print("\n=== Stage 5: Creating summary plots ===")

    plot_moving_vs_moving_bar(
        moving_vs_moving_results,
        os.path.join(config.output_dir, "moving_vs_moving_bar.pdf"),
    )

    plot_stationary_vs_moving_line(
        stationary_vs_moving_results,
        os.path.join(config.output_dir, "stationary_vs_moving_line.pdf"),
    )

    print("\n=== Stage 6: Saving results ===")

    summary = {
        "config": {
            "n_trials": config.experiment.n_trials,
            "n_values": config.experiment.n_values,
            "n_moving_clips": len(moving_clips),
            "n_stationary_clips": len(stationary_clips),
        },
        "moving_vs_moving": {
            "high_motion_mean": float(np.mean([r["high_motion_frames"] for r in moving_vs_moving_results])),
            "high_motion_std": float(np.std([r["high_motion_frames"] for r in moving_vs_moving_results])),
            "low_motion_mean": float(np.mean([r["low_motion_frames"] for r in moving_vs_moving_results])),
            "low_motion_std": float(np.std([r["low_motion_frames"] for r in moving_vs_moving_results])),
        },
        "stationary_vs_moving": {},
    }

    for n in config.experiment.n_values:
        results = stationary_vs_moving_results[n]
        summary["stationary_vs_moving"][f"n_{n}"] = {
            "stationary_mean": float(np.mean([r["stationary_frames"] for r in results])),
            "stationary_std": float(np.std([r["stationary_frames"] for r in results])),
            "moving_total_mean": float(np.mean([r["moving_frames_total"] for r in results])),
            "moving_total_std": float(np.std([r["moving_frames_total"] for r in results])),
        }

    with open(os.path.join(config.output_dir, "results.yaml"), "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    with open(os.path.join(config.output_dir, "all_results.pkl"), "wb") as f:
        pkl.dump({
            "moving_vs_moving": moving_vs_moving_results,
            "stationary_vs_moving": stationary_vs_moving_results,
        }, f)

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    print("\nMoving vs Moving (k=2):")
    print(f"  High motion: {summary['moving_vs_moving']['high_motion_mean']:.1f} ± "
          f"{summary['moving_vs_moving']['high_motion_std']:.1f} frames")
    print(f"  Low motion:  {summary['moving_vs_moving']['low_motion_mean']:.1f} ± "
          f"{summary['moving_vs_moving']['low_motion_std']:.1f} frames")

    print("\nStationary vs Moving:")
    for n in config.experiment.n_values:
        s = summary["stationary_vs_moving"][f"n_{n}"]
        print(f"  n={n}: Stationary = {s['stationary_mean']:.1f} ± {s['stationary_std']:.1f}, "
              f"Moving = {s['moving_total_mean']:.1f} ± {s['moving_total_std']:.1f}")

    print(f"\nAll outputs saved to: {config.output_dir}")
    print("=" * 70)

    return summary


def main(config_path: str = "configs/scripts/batch_incompressibility.yaml"):
    if os.path.exists(config_path):
        cfg = load_config(config_path)
    else:
        print(f"Config file not found: {config_path}")
        print("Using default configuration")
        cfg = BatchIncompressibilityConfig()

    run_experiment(cfg)


if __name__ == "__main__":
    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else "configs/scripts/batch_incompressibility.yaml"
    )
    main(config_path)
