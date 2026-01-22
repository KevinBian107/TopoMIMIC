"""MMD-based metrics for cluster-GT evaluation."""

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .ground_truth import find_all_transitions, TransitionSegment
from .behavior_parsing import parse_cluster_label


@dataclass
class MMDMetrics:
    cluster_id: int
    cluster_label: str
    gt_behavior: str
    mmd_distance: float
    n_cluster_samples: int
    n_gt_samples: int


def _uniform_downsample(frames: np.ndarray, target_len: int) -> np.ndarray:
    if len(frames) < target_len:
        return None
    indices = np.linspace(0, len(frames) - 1, target_len, dtype=int)
    return frames[indices]


def _interpolate_frames(frames: np.ndarray, target_len: int) -> np.ndarray:
    original_len = len(frames)
    original_indices = np.linspace(0, original_len - 1, target_len)
    interpolated = np.interp(original_indices, np.arange(original_len), frames)
    return np.round(interpolated).astype(int)


def _normalize_frames(frames: np.ndarray, target_len: int) -> np.ndarray:
    if len(frames) < target_len:
        return _interpolate_frames(frames, target_len)
    elif len(frames) > target_len:
        return _uniform_downsample(frames, target_len)
    return frames


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
    """
    Compute Maximum Mean Discrepancy between two sets of samples.

    Uses RBF kernel: k(x,y) = exp(-gamma * ||x-y||^2)
    """
    if len(X) == 0 or len(Y) == 0:
        return float('inf')

    if gamma is None:
        combined = np.vstack([X, Y])
        pairwise_sq = np.sum((combined[:, None] - combined[None, :]) ** 2, axis=-1)
        median_sq = np.median(pairwise_sq[pairwise_sq > 0])
        gamma = 1.0 / median_sq if median_sq > 0 else 1.0

    def rbf_kernel(A, B):
        sq_dist = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-gamma * sq_dist)

    K_XX = rbf_kernel(X, X)
    K_YY = rbf_kernel(Y, Y)
    K_XY = rbf_kernel(X, Y)

    n, m = len(X), len(Y)

    sum_XX = (np.sum(K_XX) - n) / (n * (n - 1)) if n > 1 else 0
    sum_YY = (np.sum(K_YY) - m) / (m * (m - 1)) if m > 1 else 0
    sum_XY = np.sum(K_XY) / (n * m)

    mmd_sq = sum_XX + sum_YY - 2 * sum_XY
    return float(np.sqrt(max(0, mmd_sq)))


def extract_cluster_trajectories(
    cluster_dict: dict,
    cluster_id: int,
    data: np.ndarray,
    target_len: int,
) -> list[np.ndarray]:
    """Extract normalized trajectories for a cluster."""
    trajectories = []
    entries = cluster_dict.get(cluster_id, [])

    for clip_id, frame_indices in entries:
        clip_id = int(clip_id)
        if clip_id >= len(data):
            continue

        frames = np.asarray(frame_indices).flatten()
        frames = _normalize_frames(frames, target_len)
        if frames is None:
            continue

        valid_frames = frames[frames < len(data[clip_id])]
        if len(valid_frames) < target_len:
            continue

        traj = data[clip_id, valid_frames]
        trajectories.append(traj)

    return trajectories


def extract_gt_trajectories(
    transitions: list[TransitionSegment],
    data: np.ndarray,
    target_len: int,
) -> list[np.ndarray]:
    """Extract normalized trajectories for GT transitions."""
    trajectories = []

    for t in transitions:
        if t.clip_id >= len(data):
            continue

        frames = np.array(t.all_frames)
        frames = _normalize_frames(frames, target_len)
        if frames is None:
            continue

        valid_frames = frames[frames < len(data[t.clip_id])]
        if len(valid_frames) < target_len:
            continue

        traj = data[t.clip_id, valid_frames]
        trajectories.append(traj)

    return trajectories


def compute_mmd_for_cluster(
    cluster_dict: dict,
    cluster_id: int,
    cluster_label: str,
    gt_transitions: list[TransitionSegment],
    data: np.ndarray,
    target_len: int,
) -> MMDMetrics | None:
    """Compute MMD distance between cluster and GT pose distributions."""
    parsed = parse_cluster_label(cluster_label, merge_rear=True)
    if parsed is None:
        return None
    if parsed.comparison_type != "transition":
        return None
    if len(parsed.primitives) < 2:
        return None

    primitives = parsed.primitives[:2]
    gt_behavior = f"{primitives[0]} <-> {primitives[1]}"

    target_pair = tuple(sorted(primitives))
    matching_gt = [
        t for t in gt_transitions
        if t.transition_key == target_pair
    ]

    cluster_trajs = extract_cluster_trajectories(
        cluster_dict, cluster_id, data, target_len
    )
    gt_trajs = extract_gt_trajectories(matching_gt, data, target_len)

    if not cluster_trajs or not gt_trajs:
        return None

    cluster_poses = np.vstack([t.reshape(-1, t.shape[-1]) for t in cluster_trajs])
    gt_poses = np.vstack([t.reshape(-1, t.shape[-1]) for t in gt_trajs])

    mmd = compute_mmd(cluster_poses, gt_poses)

    return MMDMetrics(
        cluster_id=cluster_id,
        cluster_label=cluster_label,
        gt_behavior=gt_behavior,
        mmd_distance=mmd,
        n_cluster_samples=len(cluster_poses),
        n_gt_samples=len(gt_poses),
    )


def compute_all_mmd_metrics(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    behavior_labels_csv: str,
    data: np.ndarray,
    target_len: int = 70,
    short_threshold: int = 10,
    merge_rear: bool = True,
) -> list[MMDMetrics]:
    """Compute MMD metrics for all labeled transition clusters."""
    gt_transitions = find_all_transitions(
        behavior_labels_csv,
        short_threshold=short_threshold,
        merge_rear=merge_rear,
    )

    metrics_list = []
    for cluster_id, label in tqdm(cluster_labels.items(), desc="Computing MMD metrics"):
        m = compute_mmd_for_cluster(
            cluster_dict=cluster_dict,
            cluster_id=cluster_id,
            cluster_label=label,
            gt_transitions=gt_transitions,
            data=data,
            target_len=target_len,
        )
        if m is not None:
            metrics_list.append(m)

    return metrics_list


def run_mmd_analysis(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    behavior_labels_csv: str,
    data: np.ndarray,
    output_dir: str,
    method_name: str = "",
    seed: int = 0,
    target_len: int = 70,
    short_threshold: int = 10,
    merge_rear: bool = True,
) -> list[MMDMetrics]:
    """Run MMD metrics analysis."""
    import os
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    print("    Computing MMD set-to-set metrics...")
    mmd_metrics = compute_all_mmd_metrics(
        cluster_dict=cluster_dict,
        cluster_labels=cluster_labels,
        behavior_labels_csv=behavior_labels_csv,
        data=data,
        target_len=target_len,
        short_threshold=short_threshold,
        merge_rear=merge_rear,
    )

    if mmd_metrics:
        mean_mmd = np.mean([m.mmd_distance for m in mmd_metrics])
        print(f"      MMD Metrics: {len(mmd_metrics)} clusters, Mean MMD: {mean_mmd:.4f}")

        rows = []
        for m in mmd_metrics:
            rows.append({
                "cluster_id": m.cluster_id,
                "label": m.cluster_label,
                "gt_behavior": m.gt_behavior,
                "mmd_distance": m.mmd_distance,
                "n_cluster_samples": m.n_cluster_samples,
                "n_gt_samples": m.n_gt_samples,
            })
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, f"{method_name}_seed{seed}_mmd_metrics.csv")
        df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"      Saved to {csv_path}")

    return mmd_metrics


def plot_mmd_comparison_table(
    all_mmd_metrics: dict[str, list[MMDMetrics]],
    output_path: str,
    title: str = "MMD Distance by Method and Transition Type",
    figsize: tuple[float, float] = None,
) -> None:
    """
    Plot MMD comparison table with methods as columns and transitions as rows.

    Args:
        all_mmd_metrics: Dict mapping method_name -> list of MMDMetrics
        output_path: Path to save the figure
        title: Plot title
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if not all_mmd_metrics:
        return

    gt_behaviors = set()
    for metrics_list in all_mmd_metrics.values():
        for m in metrics_list:
            gt_behaviors.add(m.gt_behavior)
    gt_behaviors = sorted(gt_behaviors)

    if not gt_behaviors:
        return

    method_names = list(all_mmd_metrics.keys())
    n_methods = len(method_names)
    n_behaviors = len(gt_behaviors)

    data = np.full((n_behaviors, n_methods), np.nan)
    for j, method in enumerate(method_names):
        metrics_list = all_mmd_metrics[method]
        behavior_to_mmd = {}
        for m in metrics_list:
            if m.gt_behavior in behavior_to_mmd:
                behavior_to_mmd[m.gt_behavior] = min(behavior_to_mmd[m.gt_behavior], m.mmd_distance)
            else:
                behavior_to_mmd[m.gt_behavior] = m.mmd_distance
        for i, behavior in enumerate(gt_behaviors):
            if behavior in behavior_to_mmd:
                data[i, j] = behavior_to_mmd[behavior]

    if figsize is None:
        figsize = (max(8, n_methods * 2 + 2), n_behaviors * 0.6 + 2)

    fig, ax = plt.subplots(figsize=figsize)

    cmap = LinearSegmentedColormap.from_list(
        "mmd", ["#2ecc71", "#f1c40f", "#e74c3c"], N=256
    )
    cmap.set_bad(color='lightgray')

    vmax = np.nanmax(data) if not np.all(np.isnan(data)) else 1.0
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(n_methods))
    ax.set_yticks(np.arange(n_behaviors))
    ax.set_xticklabels(method_names, fontsize=11, fontweight='bold')
    ax.set_yticklabels(gt_behaviors, fontsize=10)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(n_behaviors):
        for j in range(n_methods):
            value = data[i, j]
            if np.isnan(value):
                text = "—"
                text_color = "gray"
            else:
                text = f"{value:.3f}"
                text_color = "white" if value > vmax * 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    color=text_color, fontsize=10, fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Transition Type", fontsize=12)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("MMD Distance (lower = better)", rotation=-90, va="bottom", fontsize=11)

    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(n_methods + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_behaviors + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved MMD comparison table to {output_path}")
