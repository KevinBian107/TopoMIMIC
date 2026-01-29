"""DTW visualization: warping path, aligned trajectories, and video comparison."""

import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

from .ground_truth import find_all_transitions, TransitionSegment
from .behavior_parsing import parse_cluster_label, parse_turn_direction, TurnDirection
from .metrics import extract_sequences_from_cluster, extract_sequences_for_gt_label


JOINT_NAMES = [
    "vertebra_6_twist",      # 0
    "hip_L_supinate",        # 1
    "hip_L_abduct",          # 2
    "hip_L_extend",          # 3
    "knee_L",                # 4
    "ankle_L",               # 5
    "toe_L",                 # 6
    "hip_R_supinate",        # 7
    "hip_R_abduct",          # 8
    "hip_R_extend",          # 9
    "knee_R",                # 10
    "ankle_R",               # 11
    "toe_R",                 # 12
    "vertebra_C21_bend",     # 13  (tail)
    "vertebra_C19_extend",   # 14
    "vertebra_C17_bend",     # 15
    "vertebra_C15_extend",   # 16
    "vertebra_C13_bend",     # 17
    "vertebra_C11_extend",   # 18
    "vertebra_C9_bend",      # 19
    "vertebra_C7_extend",    # 20
    "vertebra_C6_extend",    # 21
    "vertebra_C6_bend",      # 22
    "vertebra_C5_extend",    # 23
    "vertebra_C5_bend",      # 24
    "vertebra_C4_extend",    # 25
    "vertebra_C4_bend",      # 26
    "vertebra_C3_extend",    # 27
    "vertebra_C3_bend",      # 28
    "vertebra_C2_extend",    # 29
    "vertebra_C2_bend",      # 30
    "vertebra_C1_extend",    # 31
    "vertebra_C1_bend",      # 32
    "vertebra_5_bend",       # 33  (lumbar)
    "vertebra_4_extend",     # 34
    "vertebra_3_twist",      # 35
    "vertebra_2_bend",       # 36
    "vertebra_1_extend",     # 37
    "vertebra_atlant_extend",# 38  (neck)
    "atlas",                 # 39
    "mandible",              # 40
    "vertebra_axis_twist",   # 41
    "vertebra_cervical_1",   # 42
    "vertebra_cervical_2",   # 43
    "vertebra_cervical_3",   # 44
    "vertebra_cervical_4",   # 45
    "vertebra_cervical_5",   # 46
    "scapula_L_supinate",    # 47
    "scapula_L_abduct",      # 48
    "scapula_L_extend",      # 49
    "shoulder_L",            # 50
    "shoulder_sup_L",        # 51
    "elbow_L",               # 52
    "wrist_L",               # 53
    "finger_L",              # 54
    "scapula_R_supinate",    # 55
    "scapula_R_abduct",      # 56
    "scapula_R_extend",      # 57
    "shoulder_R",            # 58
    "shoulder_sup_R",        # 59
    "elbow_R",               # 60
    "wrist_R",               # 61
    "finger_R",              # 62
]

KEY_JOINTS = {
    "Hip L Extend": 3,
    "Hip R Extend": 9,
    "Knee L": 4,
    "Knee R": 10,
    "Spine Twist": 35,
    "Spine Bend": 36,
    "Elbow L": 52,
    "Elbow R": 60,
}



@dataclass
class DTWVisualizationResult:
    label: str
    turn_direction: str | None
    cluster_ids: list[int]
    dtw_distance: float
    cost_matrix_path: str
    trajectory_path: str
    video_path: str


def compute_dtw_with_path(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
) -> tuple[float, np.ndarray, list[tuple[int, int]]]:
    """
    Compute DTW distance and return the warping path.

    Args:
        seq_a: (T1, D) first sequence
        seq_b: (T2, D) second sequence

    Returns:
        (dtw_distance, cost_matrix, warping_path)
        warping_path is list of (i, j) tuples from (0,0) to (n-1, m-1)
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float('inf'), np.array([]), []

    cost = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost[i, j] = np.linalg.norm(seq_a[i] - seq_b[j])

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = cost[i - 1, j - 1] + min(
                dtw[i - 1, j],
                dtw[i, j - 1],
                dtw[i - 1, j - 1],
            )

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [
            (dtw[i - 1, j - 1], i - 1, j - 1),
            (dtw[i - 1, j], i - 1, j),
            (dtw[i, j - 1], i, j - 1),
        ]
        _, i, j = min(candidates, key=lambda x: x[0])

    path.reverse()
    normalized_dist = dtw[n, m] / (n + m)

    return normalized_dist, cost, path


def plot_cost_matrix_with_path(
    cost_matrix: np.ndarray,
    warping_path: list[tuple[int, int]],
    output_path: str,
    title: str = "DTW Cost Matrix with Warping Path",
    discovered_label: str = "Discovered",
    gt_label: str = "GT",
) -> None:
    """Plot the cost matrix heatmap with the optimal warping path overlaid."""
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = LinearSegmentedColormap.from_list("cost", ["#2ECC71", "#F1C40F", "#E74C3C"], N=256)
    im = ax.imshow(cost_matrix, cmap=cmap, aspect='auto', origin='lower')

    if warping_path:
        path_i = [p[0] for p in warping_path]
        path_j = [p[1] for p in warping_path]
        ax.plot(path_j, path_i, 'b-', linewidth=2.5, label='Warping Path')
        ax.plot(path_j[0], path_i[0], 'go', markersize=10, label='Start')
        ax.plot(path_j[-1], path_i[-1], 'ro', markersize=10, label='End')

    ax.set_xlabel(f"{gt_label} frames", fontsize=11)
    ax.set_ylabel(f"{discovered_label} frames", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Frame Distance (L2)", fontsize=10)

    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_aligned_trajectories(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    warping_path: list[tuple[int, int]],
    output_path: str,
    joint_indices: list[int] | None = None,
    joint_names: list[str] | None = None,
    title: str = "Aligned Joint Trajectories",
    discovered_label: str = "Discovered",
    gt_label: str = "GT",
) -> None:
    """
    Plot joint angle trajectories with alignment lines.

    Args:
        seq_a: (T1, D) discovered sequence
        seq_b: (T2, D) GT sequence
        warping_path: List of (i, j) alignment pairs
        output_path: Path to save the plot
        joint_indices: Which joint dimensions to plot (default: key behavioral joints)
        joint_names: Names for the joints
        title: Plot title
        discovered_label: Label for discovered sequence
        gt_label: Label for GT sequence
    """
    if joint_indices is None and joint_names is None:
        joint_names = list(KEY_JOINTS.keys())
        joint_indices = list(KEY_JOINTS.values())
        valid = [(name, idx) for name, idx in zip(joint_names, joint_indices) if idx < seq_a.shape[1]]
        joint_names = [v[0] for v in valid]
        joint_indices = [v[1] for v in valid]

    if joint_indices is None:
        joint_indices = list(range(min(6, seq_a.shape[1])))

    if joint_names is None:
        joint_names = [JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"Joint {i}" for i in joint_indices]

    n_joints = len(joint_indices)
    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 2.5 * n_joints), sharex=False)
    if n_joints == 1:
        axes = [axes]

    path_subsample = warping_path[::max(1, len(warping_path) // 20)]

    for ax_idx, (joint_idx, joint_name) in enumerate(zip(joint_indices, joint_names)):
        ax = axes[ax_idx]

        t_a = np.arange(len(seq_a))
        t_b = np.arange(len(seq_b)) + len(seq_a) + 5

        ax.plot(t_a, seq_a[:, joint_idx], 'b-', linewidth=1.5, label=discovered_label)
        ax.plot(t_b, seq_b[:, joint_idx], 'r-', linewidth=1.5, label=gt_label)

        for i, j in path_subsample:
            ax.plot([t_a[i], t_b[j]], [seq_a[i, joint_idx], seq_b[j, joint_idx]],
                    'gray', alpha=0.3, linewidth=0.5)

        ax.set_ylabel(joint_name, fontsize=10)
        ax.axvline(x=len(seq_a) + 2.5, color='black', linestyle='--', alpha=0.5)

        if ax_idx == 0:
            ax.legend(loc='upper right')

    axes[-1].set_xlabel("Frame index (left: discovered, right: GT)", fontsize=11)
    fig.suptitle(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def render_dtw_comparison_video(
    seq_a_frames: list[np.ndarray],
    seq_b_frames: list[np.ndarray],
    warping_path: list[tuple[int, int]],
    output_path: str,
    fps: int = 30,
    target_size: tuple[int, int] = (320, 240),
    title: str = "DTW Alignment Comparison",
    left_label: str = "Discovered",
    right_label: str = "GT",
) -> None:
    """
    Render side-by-side video of discovered vs GT sequences, playing at aligned time.

    Args:
        seq_a_frames: List of rendered frames for discovered sequence
        seq_b_frames: List of rendered frames for GT sequence
        warping_path: List of (i, j) alignment pairs
        output_path: Output video path
        fps: Frames per second
        target_size: (width, height) per tile
        title: Video title
        left_label: Label for left (discovered) video
        right_label: Label for right (GT) video
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    Wc, Hc = target_size
    try:
        font_label = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_title = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font_label = ImageFont.load_default()
        font_title = ImageFont.load_default()

    pad_x, pad_y = 20, 20
    gutter = 20
    label_h = 28
    title_h = 32

    W = pad_x * 2 + Wc * 2 + gutter
    H = pad_y * 2 + title_h + label_h + Hc

    writer = imageio_ffmpeg.write_frames(
        output_path, (W, H), fps=fps, codec="h264", macro_block_size=1
    )
    writer.send(None)

    def draw_centered(draw, text, x_center, y_center, font, fill=(0, 0, 0)):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x_center - tw / 2, y_center - th / 2), text, font=font, fill=fill)

    def ensure_rgb_uint8(frame):
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        return arr

    for t, (i, j) in enumerate(warping_path):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        draw_centered(draw, title, W // 2, pad_y + title_h // 2, font_title)

        y0 = pad_y + title_h

        x_left = pad_x
        draw_centered(draw, left_label, x_left + Wc // 2, y0 + label_h // 2, font_label)
        fr_left = ensure_rgb_uint8(seq_a_frames[i])
        img_left = Image.fromarray(fr_left).resize((Wc, Hc))
        canvas.paste(img_left, (x_left, y0 + label_h))

        x_right = pad_x + Wc + gutter
        draw_centered(draw, right_label, x_right + Wc // 2, y0 + label_h // 2, font_label)
        fr_right = ensure_rgb_uint8(seq_b_frames[j])
        img_right = Image.fromarray(fr_right).resize((Wc, Hc))
        canvas.paste(img_right, (x_right, y0 + label_h))

        frame_info = f"Frame {t+1}/{len(warping_path)} | Disc:{i} GT:{j}"
        draw.text((pad_x, H - 18), frame_info, font=font_label, fill=(100, 100, 100))

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()


def render_frames_for_sequence(
    clip_id: int,
    frame_indices: np.ndarray,
    rodent_rollout_dir: str,
    camera_name: str = "close_profile",
) -> list[np.ndarray]:
    """Render frames for a sequence of frame indices from a clip."""
    from topo_mimic.dmwk.utils.load_utils import load_rollout_from_source
    from topo_mimic.dmwk.utils.rendering import render_from_saved_rollout

    rollout = load_rollout_from_source(clip_id, rodent_rollout_dir)
    all_frames = render_from_saved_rollout(rollout, camera_name=camera_name)

    frames = []
    for idx in frame_indices:
        if 0 <= idx < len(all_frames):
            frames.append(all_frames[idx])
        else:
            frames.append(all_frames[-1] if all_frames else np.zeros((240, 320, 3), dtype=np.uint8))

    return frames


def run_dtw_visualization(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    turn_directions: dict[int, str] | None,
    behavior_labels_csv: str,
    data: np.ndarray,
    rodent_rollout_dir: str,
    output_dir: str,
    method_name: str = "",
    seed: int = 0,
    short_threshold: int = 10,
    merge_rear: bool = True,
    skip_root_dims: int = 7,
    min_seq_length: int = 3,
    camera_name: str = "close_profile",
    fps: int = 30,
    target_size: tuple[int, int] = (320, 240),
) -> list[DTWVisualizationResult]:
    """
    Generate DTW visualizations for each discovered cluster.

    For each cluster:
    1. Pick one discovered sequence and one GT sequence
    2. Compute DTW with warping path
    3. Generate cost matrix plot
    4. Generate aligned trajectory plot
    5. Render side-by-side video

    Args:
        cluster_dict: Mapping cluster_id -> list of (clip_id, frames)
        cluster_labels: Mapping cluster_id -> label string
        turn_directions: Mapping cluster_id -> turn direction
        behavior_labels_csv: Path to behavior labels CSV
        data: (num_clips, num_frames, dim) feature array
        rodent_rollout_dir: Path to rollout data for rendering
        output_dir: Directory to save outputs
        method_name: Name for output files
        seed: Seed for output files
        short_threshold: Max frames for intermediate behavior bridging
        merge_rear: Merge rear categories
        skip_root_dims: Number of root dimensions to skip
        min_seq_length: Minimum sequence length
        camera_name: Camera for rendering
        fps: Video framerate
        target_size: Video tile size

    Returns:
        List of DTWVisualizationResult
    """
    from collections import defaultdict
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    if turn_directions is None:
        turn_directions = {}

    data_for_dtw = data
    if skip_root_dims > 0:
        data_for_dtw = data[..., skip_root_dims:]

    print("    Loading GT transitions...")
    all_transitions = find_all_transitions(
        behavior_labels_csv,
        short_threshold=short_threshold,
        merge_rear=merge_rear,
    )

    print("    Grouping discovered clusters by (label, turn_direction)...")
    grouped_clusters = defaultdict(list)

    for cluster_id, label in cluster_labels.items():
        if not label or label.startswith("C"):
            continue

        parsed = parse_cluster_label(label, merge_rear=merge_rear)
        if parsed is None or parsed.comparison_type != "transition":
            continue

        if len(parsed.primitives) < 2:
            continue

        target_behaviors = tuple(sorted(parsed.primitives[:2]))
        turn_dir_str = turn_directions.get(cluster_id)

        key = (target_behaviors, turn_dir_str)
        grouped_clusters[key].append(cluster_id)

    results = []
    prefix = f"{method_name}_seed{seed}" if method_name else f"seed{seed}"

    print(f"\n    Generating DTW visualizations for {len(grouped_clusters)} groups...")

    for (target_behaviors, turn_dir_str), cluster_ids in tqdm(
        grouped_clusters.items(), desc="    DTW visualization"
    ):
        turn_dir = parse_turn_direction(turn_dir_str)
        label_str = f"{target_behaviors[0]}↔{target_behaviors[1]}"
        safe_label = label_str.replace("↔", "_").replace(" ", "_")
        turn_suffix = f"_{turn_dir_str}" if turn_dir_str else ""

        discovered_entries = []
        for cid in cluster_ids:
            entries = cluster_dict.get(cid, [])
            discovered_entries.extend(entries)

        if not discovered_entries:
            continue

        disc_clip_id, disc_frame_indices = discovered_entries[0]
        disc_clip_id = int(disc_clip_id)
        disc_frames = np.asarray(disc_frame_indices).flatten()
        disc_frames = np.sort(disc_frames)
        valid_disc = disc_frames[(disc_frames >= 0) & (disc_frames < len(data[disc_clip_id]))]

        if len(valid_disc) < min_seq_length:
            continue

        disc_seq = data_for_dtw[disc_clip_id, valid_disc]

        gt_transitions = [
            t for t in all_transitions
            if t.transition_key == target_behaviors
        ]

        if turn_dir not in (None, TurnDirection.STRAIGHT, TurnDirection.NON_DETERMINISTIC):
            gt_transitions = [
                t for t in gt_transitions
                if t.from_turn == turn_dir or t.to_turn == turn_dir
            ]

        if not gt_transitions:
            continue

        gt_t = gt_transitions[0]
        gt_frames = np.array(gt_t.all_frames)
        gt_frames = np.sort(gt_frames)
        valid_gt = gt_frames[(gt_frames >= 0) & (gt_frames < len(data[gt_t.clip_id]))]

        if len(valid_gt) < min_seq_length:
            continue

        gt_seq = data_for_dtw[gt_t.clip_id, valid_gt]

        dtw_dist, cost_matrix, warping_path = compute_dtw_with_path(disc_seq, gt_seq)

        cost_path = os.path.join(output_dir, f"{prefix}_{safe_label}{turn_suffix}_cost_matrix.pdf")
        plot_cost_matrix_with_path(
            cost_matrix,
            warping_path,
            cost_path,
            title=f"DTW Cost Matrix: {label_str}{' [' + turn_dir_str + ']' if turn_dir_str else ''}\nDTW = {dtw_dist:.4f}",
            discovered_label=f"Discovered (C{cluster_ids[0]})",
            gt_label=f"GT (clip {gt_t.clip_id})",
        )

        traj_path = os.path.join(output_dir, f"{prefix}_{safe_label}{turn_suffix}_trajectories.pdf")
        plot_aligned_trajectories(
            disc_seq,
            gt_seq,
            warping_path,
            traj_path,
            title=f"Aligned Trajectories: {label_str}{' [' + turn_dir_str + ']' if turn_dir_str else ''}\nDTW = {dtw_dist:.4f}",
            discovered_label=f"Discovered (C{cluster_ids[0]})",
            gt_label=f"GT (clip {gt_t.clip_id})",
        )

        print(f"      Rendering video for {label_str}{turn_suffix}...")
        try:
            disc_rendered = render_frames_for_sequence(
                disc_clip_id, valid_disc, rodent_rollout_dir, camera_name
            )
            gt_rendered = render_frames_for_sequence(
                gt_t.clip_id, valid_gt, rodent_rollout_dir, camera_name
            )

            video_path = os.path.join(output_dir, f"{prefix}_{safe_label}{turn_suffix}_comparison.mp4")
            render_dtw_comparison_video(
                disc_rendered,
                gt_rendered,
                warping_path,
                video_path,
                fps=fps,
                target_size=target_size,
                title=f"{label_str} | DTW={dtw_dist:.3f}",
                left_label=f"Discovered (C{cluster_ids[0]})",
                right_label=f"GT (clip {gt_t.clip_id})",
            )
        except Exception as e:
            print(f"        Video rendering failed: {e}")
            video_path = ""

        results.append(DTWVisualizationResult(
            label=label_str,
            turn_direction=turn_dir_str,
            cluster_ids=cluster_ids,
            dtw_distance=dtw_dist,
            cost_matrix_path=cost_path,
            trajectory_path=traj_path,
            video_path=video_path,
        ))

    print(f"\n    Generated {len(results)} DTW visualizations")
    return results
