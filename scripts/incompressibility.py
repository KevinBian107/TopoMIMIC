"""
Incompressibility experiment: tests the theory that DM-G preserves divergent
paths in intention space.

Finds trajectories with similar past but divergent future, applies DM-G,
and renders all detected cycles to verify the topological filtering behavior.

Theory (from proof.pdf and insight.md):
- DM-G acts as a topological high-pass filter
- Steady-state behavior (H_p = 0) collapses - predictable manifolds are deleted
- Transitions create homological residues (H_1 != 0) that cannot be paired away
- At divergence points, multiple future continuations coexist, creating loops
- The Weak Morse Inequalities (m_p >= beta_p) guarantee critical cell retention

Usage:
    python scripts/incompressibility.py [config_path]

    Default config: configs/scripts/incompressibility.yaml

Outputs:
    - intention_divergence.png: Visualizes divergence in intention space
    - root_xyz_3d.png: 3D trajectory of root position
    - key_joints_2d.png: 2D panels showing key joint progression
    - side_by_side.mp4: Video of divergent trajectories side by side
    - cycles_clip_*.mp4: Videos showing DM-G cycles for each trajectory
    - dm_results.pkl: Serialized cycle and graph data
    - summary.yaml: Summary statistics
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
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
from tqdm import tqdm
from scipy.spatial.distance import cdist

from topo_mimic.dmwk.utils.load_utils import (
    get_aggregate_data,
    get_clip_indices,
    load_rollout_from_source,
    is_consolidated_format,
)
from topo_mimic.dmwk.utils.rendering import render_from_saved_rollout
from topo_mimic.dmwk.skeletonization.skeletonization import extract_skeleton_pcd_dm


@dataclass
class DataConfig:
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar_latest.h5"
    split: str = "train"
    intention_key: str = "intention"
    qpos_key: str = "qpos_rollout"
    skip_clips: list[int] = field(default_factory=list)


@dataclass
class DivergenceConfig:
    k_trajectories: int = 2
    past_frames: int = 100
    future_frames: int = 200
    similarity_threshold: float = 0.8
    divergence_threshold: float = 2.0
    n_candidates: int = 50
    seed: int = 42
    min_displacement: float = 0.05
    require_movement: bool = True
    moving_vs_stationary: bool = False
    max_intention_variance: float = 0.1


@dataclass
class DMConfig:
    m: int = 3
    tau: int = 6
    pers_delta: float = 1.0
    k_neighbors: int = 3
    epsilon: float = 0.99


@dataclass
class JointConfig:
    name: str
    indices: list[int]


@dataclass
class VisualizationConfig:
    key_joints: list[JointConfig] = field(default_factory=list)


@dataclass
class RenderingConfig:
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)
    camera_name: str = "close_profile"


@dataclass
class IncompressibilityConfig:
    output_dir: str = "out/incompressibility"
    data: DataConfig = None
    divergence: DivergenceConfig = None
    dm: DMConfig = None
    visualization: VisualizationConfig = None
    rendering: RenderingConfig = None

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.divergence is None:
            self.divergence = DivergenceConfig()
        if self.dm is None:
            self.dm = DMConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
        if self.rendering is None:
            self.rendering = RenderingConfig()


def load_config(config_path: str) -> IncompressibilityConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = DataConfig(**cfg.get("data", {}))
    divergence_cfg = DivergenceConfig(**cfg.get("divergence", {}))
    dm_cfg = DMConfig(**cfg.get("dm", {}))

    vis_dict = cfg.get("visualization", {})
    key_joints = []
    for j in vis_dict.get("key_joints", []):
        key_joints.append(JointConfig(name=j["name"], indices=j["indices"]))
    vis_cfg = VisualizationConfig(key_joints=key_joints)

    render_dict = cfg.get("rendering", {})
    if "target_size" in render_dict:
        render_dict["target_size"] = tuple(render_dict["target_size"])
    render_cfg = RenderingConfig(**render_dict)

    return IncompressibilityConfig(
        output_dir=cfg.get("output_dir", "out/incompressibility"),
        data=data_cfg,
        divergence=divergence_cfg,
        dm=dm_cfg,
        visualization=vis_cfg,
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


def filter_moving_clips(
    qpos: np.ndarray,
    clip_indices: list[int],
    min_displacement: float = 0.05,
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Filter clips that have significant root movement in xy plane.

    Returns:
        Tuple of (filtered_qpos, filtered_indices, moving_mask_indices)
    """
    n_clips = qpos.shape[0]
    moving_mask = []

    for i in range(n_clips):
        root_xy = qpos[i, :, :2]
        displacement = np.linalg.norm(root_xy[-1] - root_xy[0])
        total_path = np.sum(np.linalg.norm(np.diff(root_xy, axis=0), axis=1))

        if displacement > min_displacement or total_path > min_displacement * 3:
            moving_mask.append(i)

    filtered_qpos = qpos[moving_mask]
    filtered_indices = [clip_indices[i] for i in moving_mask]

    return filtered_qpos, filtered_indices, moving_mask


def find_moving_vs_stationary_pairs(
    intention: np.ndarray,
    qpos: np.ndarray,
    config: DivergenceConfig,
    clip_indices: list[int],
) -> list[tuple[int, ...]]:
    """
    Find groups with 1 stationary clip and (k-1) moving clips.
    All clips should have similar past to the stationary clip.
    Stationary requires both low physical movement AND low intention variance.
    """
    np.random.seed(config.seed)
    n_clips = intention.shape[0]
    n_frames = intention.shape[1]
    k = config.k_trajectories

    moving_mask = []
    stationary_mask = []

    for i in range(n_clips):
        root_xy = qpos[i, :, :2]
        displacement = np.linalg.norm(root_xy[-1] - root_xy[0])
        total_path = np.sum(np.linalg.norm(np.diff(root_xy, axis=0), axis=1))

        intent_var = np.var(intention[i], axis=0).mean()

        is_physically_moving = displacement > config.min_displacement or total_path > config.min_displacement * 3
        is_intention_stable = intent_var < config.max_intention_variance

        if is_physically_moving:
            moving_mask.append(i)
        elif is_intention_stable:
            stationary_mask.append(i)

    print(f"Found {len(moving_mask)} moving clips, {len(stationary_mask)} stationary clips (intention var < {config.max_intention_variance})")

    if len(moving_mask) == 0 or len(stationary_mask) == 0:
        print("Warning: Need both moving and stationary clips")
        return []

    if len(moving_mask) < k - 1:
        print(f"Warning: Need at least {k-1} moving clips, only have {len(moving_mask)}")
        return []

    past_end = config.past_frames
    future_start = past_end
    future_end = min(past_end + config.future_frames, n_frames)

    past_data = intention[:, :past_end, :].reshape(n_clips, -1)
    future_data = intention[:, future_start:future_end, :].reshape(n_clips, -1)

    past_norm = past_data / (np.linalg.norm(past_data, axis=1, keepdims=True) + 1e-8)
    future_norm = future_data / (np.linalg.norm(future_data, axis=1, keepdims=True) + 1e-8)

    past_sim = past_norm @ past_norm.T
    future_dist = cdist(future_norm, future_norm, metric="euclidean")

    groups = []
    for j in stationary_mask:
        stat_clip_id = clip_indices[j]

        moving_candidates = []
        for i in moving_mask:
            if (
                past_sim[i, j] > config.similarity_threshold
                and future_dist[i, j] > config.divergence_threshold
            ):
                moving_candidates.append((clip_indices[i], i, past_sim[i, j], future_dist[i, j]))

        moving_candidates.sort(key=lambda x: (-x[2], x[3]))

        if len(moving_candidates) >= k - 1:
            selected_moving = [mc[0] for mc in moving_candidates[:k - 1]]
            group = tuple(selected_moving + [stat_clip_id])
            avg_sim = np.mean([mc[2] for mc in moving_candidates[:k - 1]])
            groups.append((group, avg_sim))

    groups.sort(key=lambda x: -x[1])
    print(f"Found {len(groups)} valid groups with 1 stationary + {k-1} moving clips")

    if len(groups) == 0:
        print("Warning: No groups found, trying with relaxed constraints...")
        for j in stationary_mask:
            stat_clip_id = clip_indices[j]
            moving_candidates = []
            for i in moving_mask:
                moving_candidates.append((clip_indices[i], i, past_sim[i, j], future_dist[i, j]))
            moving_candidates.sort(key=lambda x: (-x[2], x[3]))
            if len(moving_candidates) >= k - 1:
                selected_moving = [mc[0] for mc in moving_candidates[:k - 1]]
                group = tuple(selected_moving + [stat_clip_id])
                groups.append((group, 0))
                break

    return [g[0] for g in groups[:1]] if groups else []


def find_divergent_trajectories(
    intention: np.ndarray,
    config: DivergenceConfig,
    clip_indices: list[int],
) -> list[tuple[int, ...]]:
    np.random.seed(config.seed)
    n_clips = intention.shape[0]
    n_frames = intention.shape[1]

    past_end = config.past_frames
    future_start = past_end
    future_end = min(past_end + config.future_frames, n_frames)

    past_data = intention[:, :past_end, :].reshape(n_clips, -1)
    future_data = intention[:, future_start:future_end, :].reshape(n_clips, -1)

    past_norm = past_data / (np.linalg.norm(past_data, axis=1, keepdims=True) + 1e-8)
    future_norm = future_data / (
        np.linalg.norm(future_data, axis=1, keepdims=True) + 1e-8
    )

    past_sim = past_norm @ past_norm.T
    future_dist = cdist(future_norm, future_norm, metric="euclidean")

    candidate_pairs = []
    for i in range(n_clips):
        for j in range(i + 1, n_clips):
            if (
                past_sim[i, j] > config.similarity_threshold
                and future_dist[i, j] > config.divergence_threshold
            ):
                candidate_pairs.append(
                    (clip_indices[i], clip_indices[j], past_sim[i, j], future_dist[i, j])
                )

    candidate_pairs.sort(key=lambda x: (-x[2], x[3]))
    print(f"Found {len(candidate_pairs)} candidate divergent pairs")

    if len(candidate_pairs) == 0:
        print("Warning: No divergent pairs found with current thresholds")
        print("Falling back to pairs with highest divergence score...")
        all_pairs = []
        for i in range(n_clips):
            for j in range(i + 1, n_clips):
                score = past_sim[i, j] - future_dist[i, j] * 0.1
                all_pairs.append((clip_indices[i], clip_indices[j], past_sim[i, j], future_dist[i, j], score))
        all_pairs.sort(key=lambda x: (x[2], -x[3]), reverse=True)
        candidate_pairs = [(p[0], p[1], p[2], p[3]) for p in all_pairs[: config.n_candidates]]

    groups = []
    k = config.k_trajectories
    if k == 2:
        for pair in candidate_pairs[: config.n_candidates]:
            groups.append((pair[0], pair[1]))
    else:
        used = set()
        for pair in candidate_pairs:
            if pair[0] not in used and pair[1] not in used:
                group = [pair[0], pair[1]]
                used.add(pair[0])
                used.add(pair[1])
                for other_pair in candidate_pairs:
                    if len(group) >= k:
                        break
                    for idx in [other_pair[0], other_pair[1]]:
                        if idx not in used and len(group) < k:
                            group.append(idx)
                            used.add(idx)
                if len(group) == k:
                    groups.append(tuple(group))
                if len(groups) >= config.n_candidates // k:
                    break

    return groups[:1] if groups else []


def plot_root_xyz_3d(
    qpos_list: list[np.ndarray],
    clip_ids: list[int],
    output_path: str,
) -> None:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10(np.linspace(0, 1, len(qpos_list)))

    for i, (qpos, clip_id) in enumerate(zip(qpos_list, clip_ids)):
        x = qpos[:, 0]
        y = qpos[:, 1]
        z = qpos[:, 2]
        ax.plot(x, y, z, color=colors[i], label=f"Clip {clip_id}", linewidth=2)
        ax.scatter(x[0], y[0], z[0], color=colors[i], s=100, marker="o", edgecolors="black")
        ax.scatter(x[-1], y[-1], z[-1], color=colors[i], s=100, marker="s", edgecolors="black")

    ax.set_xlabel("X (forward)", fontsize=12)
    ax.set_ylabel("Y (lateral)", fontsize=12)
    ax.set_zlabel("Z (vertical)", fontsize=12)
    ax.set_title("Root Position Trajectories (3D)", fontsize=14)
    ax.legend()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_key_joints_2d(
    qpos_list: list[np.ndarray],
    clip_ids: list[int],
    joint_configs: list[JointConfig],
    output_path: str,
) -> None:
    if not joint_configs:
        joint_configs = [
            JointConfig(name="root_x", indices=[0]),
            JointConfig(name="root_y", indices=[1]),
            JointConfig(name="root_z", indices=[2]),
        ]

    n_joints = len(joint_configs)
    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 3 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(qpos_list)))

    for ax_idx, joint_cfg in enumerate(joint_configs):
        ax = axes[ax_idx]
        for i, (qpos, clip_id) in enumerate(zip(qpos_list, clip_ids)):
            indices = joint_cfg.indices
            valid_indices = [idx for idx in indices if idx < qpos.shape[1]]
            if not valid_indices:
                continue
            joint_values = qpos[:, valid_indices].mean(axis=1)
            frames = np.arange(len(joint_values))
            ax.plot(frames, joint_values, color=colors[i], label=f"Clip {clip_id}", linewidth=2)

        ax.set_ylabel(joint_cfg.name, fontsize=11)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame", fontsize=12)
    fig.suptitle("Key Joint Progression Over Time", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def render_side_by_side_videos(
    rodent_rollout_dir: str,
    clip_ids: list[int],
    output_path: str,
    config: RenderingConfig,
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_frames = []
    for clip_id in clip_ids:
        rollout = load_rollout_from_source(clip_id, rodent_rollout_dir)
        frames = render_from_saved_rollout(rollout, camera_name=config.camera_name)
        all_frames.append([np.asarray(f, dtype=np.uint8) for f in frames])

    max_len = max(len(f) for f in all_frames)
    for i, frames in enumerate(all_frames):
        while len(frames) < max_len:
            frames.append(frames[-1])

    n_clips = len(clip_ids)
    Wc, Hc = config.target_size
    gutter = 10
    label_h = 30
    W = n_clips * Wc + (n_clips - 1) * gutter + 40
    H = Hc + label_h + 40

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    writer = imageio_ffmpeg.write_frames(
        output_path, (W, H), fps=config.framerate, codec="h264", macro_block_size=1
    )
    writer.send(None)

    for t in tqdm(range(max_len), desc="Rendering side-by-side"):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        for i, clip_id in enumerate(clip_ids):
            x = 20 + i * (Wc + gutter)
            y = 20

            label = f"Clip {clip_id}"
            tw = draw.textlength(label, font=font)
            draw.text((x + Wc // 2 - tw // 2, y), label, font=font, fill=(0, 0, 0))

            frame = all_frames[i][t]
            img = Image.fromarray(frame).resize((Wc, Hc))
            canvas.paste(img, (x, y + label_h))

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
    print(f"Saved: {output_path}")
    return output_path


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
    print(f"Combined point cloud: {combined_pts.shape} from {len(clip_ids)} clips")

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
    print(f"Combined DM-G: {len(cycle_inds)} cycles, {G_sm.number_of_edges()} edges")

    for i, (clip_id, (start, end)) in enumerate(zip(clip_ids, clip_ranges)):
        print(f"  Clip {clip_id}: points [{start}, {end})")

    return cycle_inds, G_sm, combined_pts, clip_ranges


def render_combined_cycles(
    rodent_rollout_dir: str,
    clip_ids: list[int],
    cycles: list[np.ndarray],
    clip_ranges: list[tuple[int, int]],
    output_path: str,
    config: RenderingConfig,
    dm_config: DMConfig,
) -> str:
    """
    Render original clips synced with their cycle frame contributions.

    Layout: [Original Clip 1] | [Original Clip 2] | [Longest Cycle (frozen)] | [Clip 1 in cycle] | [Clip 2 in cycle]

    Original clips play at normal speed. Per-clip cycle panels show frames
    frozen (last seen cycle frame stays visible).
    """
    if not cycles:
        print("No cycles found in combined analysis")
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
        print("No valid cycles after filtering")
        return None

    valid_cycles.sort(key=len, reverse=True)
    longest_cycle = valid_cycles[0]
    print(f"Longest cycle has {len(longest_cycle)} points")

    cycle_frames_by_clip = {clip_id: set() for clip_id in clip_ids}

    for global_idx in longest_cycle:
        for clip_id, (start, end) in zip(clip_ids, clip_ranges):
            if start <= global_idx < end:
                local_frame = global_idx - start + offset
                local_frame = min(local_frame, len(all_frames[clip_id]) - 1)
                cycle_frames_by_clip[clip_id].add(local_frame)
                break

    for clip_id in clip_ids:
        n_frames = len(cycle_frames_by_clip[clip_id])
        print(f"  Clip {clip_id}: {n_frames} frames in longest cycle")

    mid_idx = len(longest_cycle) // 2
    frozen_global_idx = longest_cycle[mid_idx]
    frozen_clip_id = None
    frozen_frame = None
    for clip_id, (start, end) in zip(clip_ids, clip_ranges):
        if start <= frozen_global_idx < end:
            frozen_clip_id = clip_id
            local_frame = frozen_global_idx - start + offset
            local_frame = min(local_frame, len(all_frames[clip_id]) - 1)
            frozen_frame = all_frames[clip_id][local_frame]
            break

    last_cycle_frame = {clip_id: None for clip_id in clip_ids}
    for clip_id in clip_ids:
        sorted_frames = sorted(cycle_frames_by_clip[clip_id])
        if sorted_frames:
            last_cycle_frame[clip_id] = sorted_frames[0]

    n_clips = len(clip_ids)
    n_panels = n_clips + 1 + n_clips
    Wc, Hc = config.target_size
    gutter = 10
    label_h = 30
    sep_w = 6

    W = n_panels * Wc + (n_panels - 1) * gutter + sep_w + 40
    H = Hc + label_h + 40

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    max_clip_len = max(len(f) for f in all_frames.values())

    writer = imageio_ffmpeg.write_frames(
        output_path, (W, H), fps=config.framerate, codec="h264", macro_block_size=1
    )
    writer.send(None)

    for t in tqdm(range(max_clip_len), desc="Rendering synced cycles"):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        y = 20

        for i, clip_id in enumerate(clip_ids):
            x = 20 + i * (Wc + gutter)
            label = f"Original Clip {clip_id}"
            tw = draw.textlength(label, font=font)
            draw.text((x + Wc // 2 - tw // 2, y), label, font=font, fill=(0, 0, 0))

            frames = all_frames[clip_id]
            frame_idx = t % len(frames)
            img = Image.fromarray(frames[frame_idx]).resize((Wc, Hc))
            canvas.paste(img, (x, y + label_h))

            if frame_idx in cycle_frames_by_clip[clip_id]:
                draw.rectangle(
                    [x, y + label_h, x + Wc - 1, y + label_h + Hc - 1],
                    outline="red", width=3
                )

        sep_x = 20 + n_clips * (Wc + gutter)
        draw.rectangle([sep_x, y, sep_x + sep_w - 1, y + label_h + Hc], fill=(0, 0, 0))

        cycle_x = sep_x + sep_w + gutter
        label = f"Longest Cycle (L={len(longest_cycle)})"
        tw = draw.textlength(label, font=font)
        draw.text((cycle_x + Wc // 2 - tw // 2, y), label, font=font, fill=(0, 0, 0))

        if frozen_frame is not None:
            img = Image.fromarray(frozen_frame).resize((Wc, Hc))
            canvas.paste(img, (cycle_x, y + label_h))
            draw.text((cycle_x + 5, y + label_h + 5), f"C{frozen_clip_id}", font=font, fill=(255, 255, 0))

        for i, clip_id in enumerate(clip_ids):
            x = cycle_x + Wc + gutter + i * (Wc + gutter)
            n_in_cycle = len(cycle_frames_by_clip[clip_id])
            label = f"Clip {clip_id} in Cycle ({n_in_cycle})"
            tw = draw.textlength(label, font=font)
            draw.text((x + Wc // 2 - tw // 2, y), label, font=font, fill=(0, 0, 0))

            frames = all_frames[clip_id]
            frame_idx = t % len(frames)

            if frame_idx in cycle_frames_by_clip[clip_id]:
                last_cycle_frame[clip_id] = frame_idx

            display_frame = last_cycle_frame[clip_id]
            if display_frame is not None:
                img = Image.fromarray(frames[display_frame]).resize((Wc, Hc))
                canvas.paste(img, (x, y + label_h))
                if frame_idx in cycle_frames_by_clip[clip_id]:
                    draw.rectangle(
                        [x, y + label_h, x + Wc - 1, y + label_h + Hc - 1],
                        outline="red", width=3
                    )
            else:
                draw.rectangle(
                    [x, y + label_h, x + Wc - 1, y + label_h + Hc - 1],
                    fill=(220, 220, 220)
                )

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
    print(f"Saved: {output_path}")
    return output_path


def render_with_intention_graph(
    rodent_rollout_dir: str,
    clip_ids: list[int],
    intention_list: list[np.ndarray],
    cycles: list[np.ndarray],
    clip_ranges: list[tuple[int, int]],
    past_frames: int,
    output_path: str,
    config: RenderingConfig,
    dm_config: DMConfig,
) -> str:
    """
    Combined rendering: intention graphs on top with moving time bar,
    video panels below (frozen, no flashing).

    Layout:
        Top: Stacked intention graphs (one per clip) with vertical time bar
        Bottom: [Original Clip 1] | [Original Clip 2] | [Clip 1 cycle] | [Clip 2 cycle]
    """
    if not cycles:
        print("No cycles found")
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
        print("No valid cycles")
        return None

    valid_cycles.sort(key=len, reverse=True)
    longest_cycle = valid_cycles[0]

    cycle_frames_by_clip = {clip_id: set() for clip_id in clip_ids}
    for global_idx in longest_cycle:
        for clip_id, (start, end) in zip(clip_ids, clip_ranges):
            if start <= global_idx < end:
                local_frame = global_idx - start + offset
                local_frame = min(local_frame, len(all_frames[clip_id]) - 1)
                cycle_frames_by_clip[clip_id].add(local_frame)
                break

    last_cycle_frame = {clip_id: None for clip_id in clip_ids}
    for clip_id in clip_ids:
        sorted_frames = sorted(cycle_frames_by_clip[clip_id])
        if sorted_frames:
            last_cycle_frame[clip_id] = sorted_frames[0]

    n_clips = len(clip_ids)
    Wc, Hc = 480, 360
    gutter = 20
    label_h = 35
    title_h = 50
    legend_h = 40
    section_label_h = 30

    graph_panel_height = 160
    graph_height = graph_panel_height * n_clips + 30

    video_section_width = n_clips * 2 * Wc + (n_clips * 2 + 1) * gutter + 100

    W = max(video_section_width, 1200)
    H = title_h + graph_height + legend_h + section_label_h + Hc + label_h + 60

    W = W + (W % 2)
    H = H + (H % 2)

    max_clip_len = max(len(f) for f in all_frames.values())

    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        font_section = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        font_label = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_axis = ImageFont.truetype("DejaVuSans.ttf", 12)
        font_legend = ImageFont.truetype("DejaVuSans.ttf", 11)
    except:
        font_title = ImageFont.load_default()
        font_section = font_title
        font_label = font_title
        font_axis = font_title
        font_legend = font_title

    writer = imageio_ffmpeg.write_frames(
        output_path, (W, H), fps=config.framerate, codec="h264", macro_block_size=1
    )
    writer.send(None)

    colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)]

    for t in tqdm(range(max_clip_len), desc="Rendering combined"):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        title = "DM-G Incompressibility: Intention Divergence Analysis"
        tw = draw.textlength(title, font=font_title)
        draw.text((W // 2 - tw // 2, 12), title, font=font_title, fill=(0, 0, 0))

        graph_top = title_h
        graph_left = 100
        graph_right = W - 40
        graph_width = graph_right - graph_left

        draw.text((graph_left - 90, graph_top + (graph_height - 30) // 2 - 8),
                  "Mean", font=font_axis, fill=(80, 80, 80))
        draw.text((graph_left - 90, graph_top + (graph_height - 30) // 2 + 8),
                  "Intention", font=font_axis, fill=(80, 80, 80))

        for i, (clip_id, intention) in enumerate(zip(clip_ids, intention_list)):
            panel_top = graph_top + i * graph_panel_height + 10
            panel_bottom = panel_top + graph_panel_height - 20
            panel_height = panel_bottom - panel_top

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
                    draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(255, 0, 0), outline=(180, 0, 0))

            past_x = graph_left + (past_frames / n_frames) * graph_width
            draw.line([(past_x, panel_top), (past_x, panel_bottom)],
                      fill=(0, 180, 0), width=2)

            draw.text((graph_left - 70, panel_top + panel_height // 2 - 8),
                      f"Clip {clip_id}", font=font_label, fill=colors[i % len(colors)])

            draw.text((graph_right + 5, panel_top + 2),
                      f"{y_max:.2f}", font=font_axis, fill=(120, 120, 120))
            draw.text((graph_right + 5, panel_bottom - 12),
                      f"{y_min:.2f}", font=font_axis, fill=(120, 120, 120))

        n_frames_ref = len(intention_list[0])
        time_x = graph_left + (t / n_frames_ref) * graph_width
        draw.line([(time_x, graph_top + 10), (time_x, graph_top + graph_height - 20)],
                  fill=(255, 0, 0), width=3)

        frame_label = f"t={t}"
        draw.text((time_x - 15, graph_top + graph_height - 15),
                  frame_label, font=font_axis, fill=(255, 0, 0))

        axis_y = graph_top + graph_height - 5
        draw.text((graph_left, axis_y), "0", font=font_axis, fill=(120, 120, 120))
        draw.text((graph_left + graph_width // 2 - 20, axis_y),
                  f"{n_frames_ref // 2}", font=font_axis, fill=(120, 120, 120))
        draw.text((graph_right - 30, axis_y),
                  f"{n_frames_ref}", font=font_axis, fill=(120, 120, 120))
        draw.text((graph_left + graph_width // 2 - 30, axis_y + 15),
                  "Frame", font=font_axis, fill=(80, 80, 80))

        legend_y = graph_top + graph_height + 10
        legend_items = [
            ((255, 0, 0), "ellipse", "Cycle frames (DM-G H1)"),
            ((0, 180, 0), "line", "Past/Future boundary"),
            ((255, 0, 0), "vline", "Current time"),
        ]

        legend_x = graph_left
        for color, shape, text in legend_items:
            if shape == "ellipse":
                draw.ellipse([legend_x, legend_y + 4, legend_x + 10, legend_y + 14], fill=color)
            elif shape == "vline":
                draw.line([(legend_x + 5, legend_y + 2), (legend_x + 5, legend_y + 18)], fill=color, width=3)
            else:
                draw.line([(legend_x, legend_y + 10), (legend_x + 12, legend_y + 10)], fill=color, width=2)

            draw.text((legend_x + 18, legend_y + 3), text, font=font_legend, fill=(60, 60, 60))
            legend_x += draw.textlength(text, font=font_legend) + 60

        video_y = legend_y + legend_h
        section_text = "Video Panels"
        tw = draw.textlength(section_text, font=font_section)
        draw.text((W // 2 - tw // 2, video_y), section_text, font=font_section, fill=(60, 60, 60))
        video_y += section_label_h

        orig_start_x = 50
        for i, clip_id in enumerate(clip_ids):
            x = orig_start_x + i * (Wc + gutter)
            label = f"Original - Clip {clip_id}"
            tw = draw.textlength(label, font=font_label)
            draw.text((x + Wc // 2 - tw // 2, video_y), label, font=font_label, fill=(0, 0, 0))

            frames = all_frames[clip_id]
            frame_idx = t % len(frames)
            img = Image.fromarray(frames[frame_idx]).resize((Wc, Hc), Image.LANCZOS)
            canvas.paste(img, (x, video_y + label_h))

            if frame_idx in cycle_frames_by_clip[clip_id]:
                draw.rectangle(
                    [x, video_y + label_h, x + Wc - 1, video_y + label_h + Hc - 1],
                    outline=(255, 0, 0), width=4
                )

        cycle_start_x = orig_start_x + n_clips * (Wc + gutter) + 40
        draw.line([(cycle_start_x - 20, video_y), (cycle_start_x - 20, video_y + label_h + Hc)],
                  fill=(200, 200, 200), width=2)

        for i, clip_id in enumerate(clip_ids):
            x = cycle_start_x + i * (Wc + gutter)
            n_in_cycle = len(cycle_frames_by_clip[clip_id])
            label = f"Cycle Frames - Clip {clip_id} ({n_in_cycle} pts)"
            tw = draw.textlength(label, font=font_label)
            draw.text((x + Wc // 2 - tw // 2, video_y), label, font=font_label, fill=(0, 0, 0))

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
                        outline=(255, 0, 0), width=4
                    )
                draw.text((x + 5, video_y + label_h + Hc - 20),
                          f"f={display_frame}", font=font_axis, fill=(255, 255, 0))
            else:
                draw.rectangle(
                    [x, video_y + label_h, x + Wc - 1, video_y + label_h + Hc - 1],
                    fill=(220, 220, 220), outline=(180, 180, 180)
                )
                draw.text((x + Wc // 2 - 40, video_y + label_h + Hc // 2 - 8),
                          "Waiting...", font=font_label, fill=(120, 120, 120))

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
    print(f"Saved: {output_path}")
    return output_path


def render_cycles(
    rodent_rollout_dir: str,
    clip_id: int,
    cycles: list[np.ndarray],
    output_path: str,
    config: RenderingConfig,
    dm_config: DMConfig,
) -> str:
    if not cycles:
        print(f"No cycles found for clip {clip_id}")
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rollout = load_rollout_from_source(clip_id, rodent_rollout_dir)
    full_frames = render_from_saved_rollout(rollout, camera_name=config.camera_name)
    full_frames = [np.asarray(f, dtype=np.uint8) for f in full_frames]

    offset = (dm_config.m - 1) * dm_config.tau

    valid_cycles = []
    for cycle in cycles:
        adjusted = cycle + offset
        adjusted = adjusted[adjusted < len(full_frames)]
        if len(adjusted) >= 3:
            valid_cycles.append(adjusted)

    if not valid_cycles:
        print(f"No valid cycles after offset adjustment for clip {clip_id}")
        return None

    valid_cycles.sort(key=len, reverse=True)
    top_cycles = valid_cycles[:5]

    n_cycles = len(top_cycles)
    Wc, Hc = config.target_size
    gutter = 10
    label_h = 30
    W = (1 + n_cycles) * Wc + n_cycles * gutter + 40
    H = Hc + label_h + 40

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    max_len = max(len(full_frames), max(len(c) for c in top_cycles))

    writer = imageio_ffmpeg.write_frames(
        output_path, (W, H), fps=config.framerate, codec="h264", macro_block_size=1
    )
    writer.send(None)

    for t in tqdm(range(max_len), desc=f"Rendering cycles for clip {clip_id}"):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        x = 20
        y = 20
        label = f"Original (Clip {clip_id})"
        tw = draw.textlength(label, font=font)
        draw.text((x + Wc // 2 - tw // 2, y), label, font=font, fill=(0, 0, 0))
        frame_idx = t % len(full_frames)
        img = Image.fromarray(full_frames[frame_idx]).resize((Wc, Hc))
        canvas.paste(img, (x, y + label_h))

        for i, cycle in enumerate(top_cycles):
            x = 20 + (1 + i) * (Wc + gutter)
            label = f"Cycle {i} (L={len(cycle)})"
            tw = draw.textlength(label, font=font)
            draw.text((x + Wc // 2 - tw // 2, y), label, font=font, fill=(0, 0, 0))

            cycle_frame_idx = cycle[t % len(cycle)]
            img = Image.fromarray(full_frames[cycle_frame_idx]).resize((Wc, Hc))
            canvas.paste(img, (x, y + label_h))

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_intention_divergence(
    intention_list: list[np.ndarray],
    clip_ids: list[int],
    past_frames: int,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(intention_list)))

    ax = axes[0, 0]
    for i, (intention, clip_id) in enumerate(zip(intention_list, clip_ids)):
        mean_int = intention.mean(axis=1)
        frames = np.arange(len(mean_int))
        ax.plot(frames, mean_int, color=colors[i], label=f"Clip {clip_id}", linewidth=2)
    ax.axvline(x=past_frames, color="red", linestyle="--", label="Past/Future boundary")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean Intention")
    ax.set_title("Mean Intention Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for i, (intention, clip_id) in enumerate(zip(intention_list, clip_ids)):
        norm = np.linalg.norm(intention, axis=1)
        frames = np.arange(len(norm))
        ax.plot(frames, norm, color=colors[i], label=f"Clip {clip_id}", linewidth=2)
    ax.axvline(x=past_frames, color="red", linestyle="--", label="Past/Future boundary")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Intention Norm")
    ax.set_title("Intention Magnitude Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if len(intention_list) >= 2:
        int1, int2 = intention_list[0], intention_list[1]
        min_len = min(len(int1), len(int2))
        diff = np.linalg.norm(int1[:min_len] - int2[:min_len], axis=1)
        frames = np.arange(min_len)
        ax.plot(frames, diff, color="purple", linewidth=2)
        ax.axvline(x=past_frames, color="red", linestyle="--", label="Past/Future boundary")
        ax.fill_between(frames[:past_frames], 0, diff[:past_frames], alpha=0.3, color="green", label="Similar past")
        ax.fill_between(frames[past_frames:], 0, diff[past_frames:], alpha=0.3, color="orange", label="Divergent future")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Pairwise Distance")
    ax.set_title("Intention Distance Between Trajectories")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    from sklearn.decomposition import PCA
    all_int = np.vstack(intention_list)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_int)

    offset = 0
    for i, (intention, clip_id) in enumerate(zip(intention_list, clip_ids)):
        n = len(intention)
        proj = projected[offset : offset + n]
        ax.plot(proj[:, 0], proj[:, 1], color=colors[i], label=f"Clip {clip_id}", linewidth=1, alpha=0.7)
        ax.scatter(proj[0, 0], proj[0, 1], color=colors[i], s=100, marker="o", edgecolors="black", zorder=5)
        ax.scatter(proj[-1, 0], proj[-1, 1], color=colors[i], s=100, marker="s", edgecolors="black", zorder=5)
        ax.scatter(proj[past_frames, 0], proj[past_frames, 1], color=colors[i], s=80, marker="^", edgecolors="red", zorder=5)
        offset += n
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Intention PCA Projection")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_intention_with_cycles(
    intention_list: list[np.ndarray],
    clip_ids: list[int],
    cycles: list[np.ndarray],
    clip_ranges: list[tuple[int, int]],
    past_frames: int,
    dm_offset: int,
    output_path: str,
) -> None:
    """
    Plot mean intention for each clip with its cycle frames marked.

    Stacked panels (one per clip), each showing mean intention over time
    with that clip's cycle frames highlighted.
    """
    if not cycles:
        print("No cycles to plot")
        return

    longest_cycle = max(cycles, key=len)

    cycle_frames_by_clip = {clip_id: set() for clip_id in clip_ids}
    for global_idx in longest_cycle:
        for clip_id, (start, end) in zip(clip_ids, clip_ranges):
            if start <= global_idx < end:
                local_frame = global_idx - start + dm_offset
                cycle_frames_by_clip[clip_id].add(local_frame)
                break

    n_clips = len(clip_ids)
    fig, axes = plt.subplots(n_clips, 1, figsize=(14, 4 * n_clips), sharex=True)
    if n_clips == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_clips))

    for i, (clip_id, intention) in enumerate(zip(clip_ids, intention_list)):
        ax = axes[i]
        n_frames = len(intention)
        frames = np.arange(n_frames)

        mean_intention = intention.mean(axis=1)

        ax.plot(frames, mean_intention, color=colors[i], linewidth=2,
                label=f"Clip {clip_id} mean intention", zorder=1)

        cycle_frames = cycle_frames_by_clip[clip_id]
        cycle_mask = np.array([f in cycle_frames and f < n_frames for f in frames])
        cycle_frames_arr = frames[cycle_mask]
        cycle_values = mean_intention[cycle_mask]

        if len(cycle_frames_arr) > 0:
            ax.scatter(cycle_frames_arr, cycle_values, color="red", s=60,
                       marker="^", label=f"Cycle frames ({len(cycle_frames_arr)})", zorder=3)

            for f in cycle_frames_arr:
                ax.axvline(x=f, color="red", alpha=0.2, linewidth=1, zorder=0)

        ax.axvline(x=past_frames, color="green", linestyle="--", linewidth=2,
                   label=f"Past/Future boundary")

        ax.axvspan(0, past_frames, alpha=0.1, color="green")
        ax.axvspan(past_frames, n_frames, alpha=0.1, color="orange")

        ax.set_ylabel("Mean Intention", fontsize=12)
        ax.set_title(f"Clip {clip_id}", fontsize=13)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        n_past = sum(1 for f in cycle_frames if f < past_frames)
        n_future = sum(1 for f in cycle_frames if f >= past_frames and f < n_frames)
        total = n_past + n_future
        if total > 0:
            stats_text = f"Past: {n_past} ({100*n_past/total:.0f}%)  |  Future: {n_future} ({100*n_future/total:.0f}%)"
        else:
            stats_text = "No cycle frames"
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    axes[-1].set_xlabel("Frame Index", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_sink_analysis(
    combined_pts: np.ndarray,
    clip_ids: list[int],
    clip_ranges: list[tuple[int, int]],
    cycles: list[np.ndarray],
    output_path: str,
) -> None:
    """
    Visualize the 'sink' effect: project combined point cloud to 2D,
    color by clip, highlight cycle vertices.
    """
    from sklearn.decomposition import PCA

    if not cycles:
        print("No cycles to analyze")
        return

    pca = PCA(n_components=2)
    projected = pca.fit_transform(combined_pts)

    longest_cycle = max(cycles, key=len)
    cycle_vertex_set = set(longest_cycle)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    colors_clips = plt.cm.tab10(np.linspace(0, 1, len(clip_ids)))

    for i, (clip_id, (start, end)) in enumerate(zip(clip_ids, clip_ranges)):
        clip_pts = projected[start:end]
        ax.scatter(clip_pts[:, 0], clip_pts[:, 1], c=[colors_clips[i]],
                   alpha=0.3, s=10, label=f"Clip {clip_id}")
        ax.scatter(clip_pts[0, 0], clip_pts[0, 1], c=[colors_clips[i]],
                   s=100, marker="o", edgecolors="black", zorder=5)
        ax.scatter(clip_pts[-1, 0], clip_pts[-1, 1], c=[colors_clips[i]],
                   s=100, marker="s", edgecolors="black", zorder=5)

    cycle_pts = projected[list(longest_cycle)]
    ax.scatter(cycle_pts[:, 0], cycle_pts[:, 1], c="red", s=50,
               marker="x", linewidths=2, label=f"Cycle vertices (n={len(longest_cycle)})", zorder=10)

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title("Combined Point Cloud with Cycle Vertices", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cycle_by_clip = {clip_id: [] for clip_id in clip_ids}
    for idx in longest_cycle:
        for clip_id, (start, end) in zip(clip_ids, clip_ranges):
            if start <= idx < end:
                cycle_by_clip[clip_id].append(idx)
                break

    clip_labels = [f"Clip {cid}" for cid in clip_ids]
    cycle_counts = [len(cycle_by_clip[cid]) for cid in clip_ids]
    total_counts = [end - start for start, end in clip_ranges]

    x = np.arange(len(clip_ids))
    width = 0.35

    bars1 = ax.bar(x - width/2, total_counts, width, label="Total points", alpha=0.7)
    bars2 = ax.bar(x + width/2, cycle_counts, width, label="Cycle vertices", color="red", alpha=0.7)

    ax.set_xlabel("Clip", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Cycle Vertex Distribution by Clip", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(clip_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bar, count, total in zip(bars2, cycle_counts, total_counts):
        pct = 100 * count / total if total > 0 else 0
        ax.annotate(f"{pct:.1f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_experiment(config: IncompressibilityConfig) -> dict:
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("Incompressibility Experiment")
    print("=" * 60)

    print("\n=== Stage 1: Loading data ===")
    clip_indices = get_clip_indices(config.data.split)
    skip_set = set(config.data.skip_clips)
    clip_indices = [c for c in clip_indices if c not in skip_set]
    print(f"Using {len(clip_indices)} clips (skipping {len(skip_set)} clips)")
    intention = load_intention_data(config.data, clip_indices)
    qpos = load_qpos_data(config.data, clip_indices)
    print(f"Loaded intention: {intention.shape}, qpos: {qpos.shape}")

    if config.divergence.moving_vs_stationary:
        print("\n=== Stage 2: Finding moving vs stationary pairs ===")
        divergent_groups = find_moving_vs_stationary_pairs(
            intention, qpos, config.divergence, clip_indices
        )
    else:
        if config.divergence.require_movement:
            print("\n=== Stage 1b: Filtering for moving clips ===")
            qpos, clip_indices, moving_mask = filter_moving_clips(
                qpos, clip_indices, config.divergence.min_displacement
            )
            intention = intention[moving_mask]
            print(f"Filtered to {len(clip_indices)} moving clips")
            print(f"After filter - intention: {intention.shape}, qpos: {qpos.shape}")

        print("\n=== Stage 2: Finding divergent trajectories ===")
        divergent_groups = find_divergent_trajectories(
            intention, config.divergence, clip_indices
        )

    if not divergent_groups:
        print("No divergent trajectory groups found. Exiting.")
        return {"status": "no_divergent_pairs"}

    selected_group = divergent_groups[0]
    print(f"Selected divergent group: {selected_group}")

    group_indices = [clip_indices.index(cid) for cid in selected_group]
    intention_list = [intention[idx] for idx in group_indices]
    qpos_list = [qpos[idx] for idx in group_indices]

    print("\n=== Stage 3: Visualizing divergence ===")

    plot_intention_divergence(
        intention_list,
        list(selected_group),
        config.divergence.past_frames,
        os.path.join(config.output_dir, "intention_divergence.png"),
    )

    plot_root_xyz_3d(
        qpos_list,
        list(selected_group),
        os.path.join(config.output_dir, "root_xyz_3d.png"),
    )

    plot_key_joints_2d(
        qpos_list,
        list(selected_group),
        config.visualization.key_joints,
        os.path.join(config.output_dir, "key_joints_2d.png"),
    )

    print("\n=== Stage 4: Rendering side-by-side videos ===")
    render_side_by_side_videos(
        config.data.rodent_rollout_dir,
        list(selected_group),
        os.path.join(config.output_dir, "side_by_side.mp4"),
        config.rendering,
    )

    print("\n=== Stage 5: Running DM-G analysis on combined point cloud ===")
    cycles, G_sm, combined_pts, clip_ranges = run_dm_analysis_combined(
        intention_list,
        list(selected_group),
        config.dm,
        config.output_dir,
    )

    with open(os.path.join(config.output_dir, "dm_results.pkl"), "wb") as f:
        pkl.dump({
            "cycles": cycles,
            "graph": G_sm,
            "combined_pts": combined_pts,
            "clip_ranges": clip_ranges,
            "clip_ids": list(selected_group),
        }, f)

    print("\n=== Stage 5b: Plotting intention with cycle frames ===")
    dm_offset = (config.dm.m - 1) * config.dm.tau
    plot_intention_with_cycles(
        intention_list,
        list(selected_group),
        cycles,
        clip_ranges,
        config.divergence.past_frames,
        dm_offset,
        os.path.join(config.output_dir, "intention_with_cycles.png"),
    )

    print("\n=== Stage 5c: Sink analysis (PCA projection) ===")
    plot_sink_analysis(
        combined_pts,
        list(selected_group),
        clip_ranges,
        cycles,
        os.path.join(config.output_dir, "sink_analysis.png"),
    )

    print("\n=== Stage 6: Rendering cycles ===")
    render_combined_cycles(
        config.data.rodent_rollout_dir,
        list(selected_group),
        cycles,
        clip_ranges,
        os.path.join(config.output_dir, "combined_cycles.mp4"),
        config.rendering,
        config.dm,
    )

    print("\n=== Stage 6b: Rendering combined graph + video ===")
    render_with_intention_graph(
        config.data.rodent_rollout_dir,
        list(selected_group),
        intention_list,
        cycles,
        clip_ranges,
        config.divergence.past_frames,
        os.path.join(config.output_dir, "combined_graph_video.mp4"),
        config.rendering,
        config.dm,
    )

    print("\n=== Stage 7: Summary ===")

    cycle_clip_membership = []
    for cycle in cycles:
        clips_in_cycle = set()
        for idx in cycle:
            for clip_id, (start, end) in zip(selected_group, clip_ranges):
                if start <= idx < end:
                    clips_in_cycle.add(clip_id)
                    break
        cycle_clip_membership.append(list(clips_in_cycle))

    summary = {
        "divergent_group": list(selected_group),
        "n_cycles": len(cycles),
        "n_edges": G_sm.number_of_edges(),
        "n_combined_points": len(combined_pts),
        "clip_ranges": {str(cid): list(r) for cid, r in zip(selected_group, clip_ranges)},
        "cycle_lengths": [len(c) for c in cycles],
        "cycle_clip_membership": cycle_clip_membership,
    }

    with open(os.path.join(config.output_dir, "summary.yaml"), "w") as f:
        yaml.dump(summary, f)

    print("\nResults:")
    print(f"  Combined point cloud: {len(combined_pts)} points")
    print(f"  DM-G: {len(cycles)} cycles, {G_sm.number_of_edges()} edges")
    for i, (cycle, clips) in enumerate(zip(cycles[:5], cycle_clip_membership[:5])):
        print(f"  Cycle {i}: {len(cycle)} points, spans clips {clips}")

    print(f"\nAll outputs saved to: {config.output_dir}")
    print("=" * 60)

    return summary


def main(config_path: str = "configs/scripts/incompressibility.yaml"):
    if os.path.exists(config_path):
        cfg = load_config(config_path)
    else:
        cfg = IncompressibilityConfig()

    run_experiment(cfg)


if __name__ == "__main__":
    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else "configs/scripts/incompressibility.yaml"
    )
    main(config_path)
