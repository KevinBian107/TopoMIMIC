"""Core logic for automatic label assignment: GT extraction, W2 computation, recommendations."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import ot
    HAS_OT = True
except ImportError:
    HAS_OT = False

from scripts.benchmark.analysis.ground_truth import (
    find_all_transitions,
    find_pure_behavior_segments,
    TransitionSegment,
    BehaviorSegment,
)


@dataclass
class GTStatistics:
    count: int
    avg_original_frames: float
    std_original_frames: float
    min_frames: int
    max_frames: int
    n_upsampled: int
    n_downsampled: int


@dataclass
class GTLibrary:
    windows: dict[str, list[np.ndarray]] = field(default_factory=dict)
    statistics: dict[str, GTStatistics] = field(default_factory=dict)
    target_len: int = 100


@dataclass
class LabelRecommendation:
    cluster_id: int
    labels: list[tuple[str, float]]


def _interpolate_frames(frames: np.ndarray, target_len: int) -> np.ndarray:
    """Upsample frames via linear interpolation."""
    original_len = len(frames)
    original_indices = np.linspace(0, original_len - 1, target_len)
    interpolated = np.interp(original_indices, np.arange(original_len), frames)
    return np.round(interpolated).astype(int)


def _uniform_downsample(frames: np.ndarray, target_len: int) -> np.ndarray:
    """Downsample frames by uniform selection."""
    indices = np.linspace(0, len(frames) - 1, target_len, dtype=int)
    return frames[indices]


def _normalize_frames(frames: np.ndarray, target_len: int) -> np.ndarray:
    """Normalize frames to target length via up/downsampling."""
    if len(frames) < target_len:
        return _interpolate_frames(frames, target_len)
    elif len(frames) > target_len:
        return _uniform_downsample(frames, target_len)
    return frames


def extract_gt_transition_windows(
    behavior_labels_csv: str,
    intention_data: np.ndarray,
    target_len: int = 100,
    short_threshold: int = 10,
    max_gap: int = 10,
    merge_rear: bool = True,
    include_non_transitions: list[str] | None = None,
    non_transition_min_length: int = 30,
) -> GTLibrary:
    """
    Extract GT transition windows normalized to target_len.

    Args:
        behavior_labels_csv: Path to behavior_labels.csv
        intention_data: Full intention array (n_clips, n_frames, latent_dim)
        target_len: Normalize all transitions to this length
        short_threshold: Max frames for intermediate behavior bridging
        max_gap: Max gap between consecutive segments
        merge_rear: Merge Supported/Unsupported Rear into Rear
        include_non_transitions: List of non-transition behaviors to include (e.g., ["Grooming"])
        non_transition_min_length: Minimum length for non-transition segments

    Returns:
        GTLibrary with windows and statistics per transition type
    """
    transitions = find_all_transitions(
        behavior_labels_csv,
        short_threshold=short_threshold,
        max_gap=max_gap,
        merge_rear=merge_rear,
    )

    by_type: dict[str, list[TransitionSegment]] = defaultdict(list)
    for t in transitions:
        key = " <-> ".join(sorted([t.from_behavior, t.to_behavior]))
        by_type[key].append(t)

    gt_library = GTLibrary(target_len=target_len)

    for trans_type, trans_list in by_type.items():
        windows = []
        original_lengths = []
        n_upsampled = 0
        n_downsampled = 0

        for t in trans_list:
            frames = np.array(t.all_frames)
            original_len = len(frames)
            original_lengths.append(original_len)

            if original_len < target_len:
                sampled_frames = _interpolate_frames(frames, target_len)
                n_upsampled += 1
            elif original_len > target_len:
                sampled_frames = _uniform_downsample(frames, target_len)
                n_downsampled += 1
            else:
                sampled_frames = frames

            if t.clip_id < len(intention_data):
                valid_frames = sampled_frames[sampled_frames < len(intention_data[t.clip_id])]
                if len(valid_frames) == target_len:
                    intention_window = intention_data[t.clip_id, valid_frames]
                    windows.append(intention_window)

        if windows:
            gt_library.windows[trans_type] = windows
            gt_library.statistics[trans_type] = GTStatistics(
                count=len(trans_list),
                avg_original_frames=float(np.mean(original_lengths)),
                std_original_frames=float(np.std(original_lengths)),
                min_frames=int(np.min(original_lengths)),
                max_frames=int(np.max(original_lengths)),
                n_upsampled=n_upsampled,
                n_downsampled=n_downsampled,
            )

    if include_non_transitions:
        pure_segments = find_pure_behavior_segments(
            behavior_labels_csv,
            target_behaviors=include_non_transitions,
            min_length=non_transition_min_length,
            merge_rear=merge_rear,
        )

        by_behavior: dict[str, list[BehaviorSegment]] = defaultdict(list)
        for seg in pure_segments:
            by_behavior[seg.behavior].append(seg)

        for behavior, seg_list in by_behavior.items():
            windows = []
            original_lengths = []
            n_upsampled = 0
            n_downsampled = 0

            for seg in seg_list:
                frames = np.arange(seg.start_frame, seg.end_frame + 1)
                original_len = len(frames)
                original_lengths.append(original_len)

                if original_len < target_len:
                    sampled_frames = _interpolate_frames(frames, target_len)
                    n_upsampled += 1
                elif original_len > target_len:
                    sampled_frames = _uniform_downsample(frames, target_len)
                    n_downsampled += 1
                else:
                    sampled_frames = frames

                if seg.clip_id < len(intention_data):
                    valid_frames = sampled_frames[sampled_frames < len(intention_data[seg.clip_id])]
                    if len(valid_frames) == target_len:
                        intention_window = intention_data[seg.clip_id, valid_frames]
                        windows.append(intention_window)

            if windows:
                gt_library.windows[behavior] = windows
                gt_library.statistics[behavior] = GTStatistics(
                    count=len(seg_list),
                    avg_original_frames=float(np.mean(original_lengths)),
                    std_original_frames=float(np.std(original_lengths)),
                    min_frames=int(np.min(original_lengths)),
                    max_frames=int(np.max(original_lengths)),
                    n_upsampled=n_upsampled,
                    n_downsampled=n_downsampled,
                )

    return gt_library


def _compute_w2(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute W2 distance between two trajectories."""
    if not HAS_OT:
        raise ImportError("POT library not installed")

    n = len(X)
    m = len(Y)
    w = np.full(n, 1.0 / n)
    v = np.full(m, 1.0 / m)

    C = ot.dist(X, Y)
    return float(ot.emd2(w, v, C))


def compute_cluster_to_gt_distances(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    gt_library: GTLibrary,
    intention_data: np.ndarray,
    max_cycles_per_cluster: int = 20,
    max_gt_samples: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Compute average W2 distance from each cluster to each GT type.

    Args:
        cluster_dict: {cluster_id: [(clip_id, frames), ...]}
        gt_library: GTLibrary with transition windows
        intention_data: Full intention array
        max_cycles_per_cluster: Max cycles to sample per cluster
        max_gt_samples: Max GT samples per type to compare
        seed: Random seed for sampling

    Returns:
        distances: (n_clusters, n_gt_types) distance matrix
        gt_type_names: List of GT type names
        cluster_ids: List of cluster IDs
    """
    rng = np.random.default_rng(seed)
    cluster_ids = sorted(cluster_dict.keys())
    gt_type_names = sorted(gt_library.windows.keys())
    target_len = gt_library.target_len

    n_clusters = len(cluster_ids)
    n_gt_types = len(gt_type_names)

    distances = np.zeros((n_clusters, n_gt_types))

    for ci, cid in enumerate(tqdm(cluster_ids, desc="Computing cluster-to-GT distances")):
        cycles = list(cluster_dict[cid])
        if len(cycles) > max_cycles_per_cluster:
            indices = rng.choice(len(cycles), max_cycles_per_cluster, replace=False)
            cycles = [cycles[i] for i in indices]

        cycle_windows = []
        for clip_id, frames in cycles:
            if clip_id >= len(intention_data):
                continue
            frames = np.asarray(frames)
            if len(frames) != target_len:
                frames = _normalize_frames(frames, target_len)
            valid_frames = frames[frames < len(intention_data[clip_id])]
            if len(valid_frames) == target_len:
                window = intention_data[clip_id, valid_frames]
                cycle_windows.append(window)

        if not cycle_windows:
            distances[ci, :] = np.inf
            continue

        for gi, gt_type in enumerate(gt_type_names):
            gt_samples = list(gt_library.windows[gt_type])
            if len(gt_samples) > max_gt_samples:
                indices = rng.choice(len(gt_samples), max_gt_samples, replace=False)
                gt_samples = [gt_samples[i] for i in indices]

            all_distances = []
            for cycle_win in cycle_windows:
                for gt_win in gt_samples:
                    d = _compute_w2(cycle_win, gt_win)
                    all_distances.append(d)

            distances[ci, gi] = np.mean(all_distances) if all_distances else np.inf

    return distances, gt_type_names, cluster_ids


def generate_label_recommendations(
    distances: np.ndarray,
    gt_type_names: list[str],
    cluster_ids: list[int],
    top_k: int = 5,
) -> dict[int, LabelRecommendation]:
    """
    Generate top-K label recommendations for each cluster.

    Args:
        distances: (n_clusters, n_gt_types) distance matrix
        gt_type_names: List of GT type names
        cluster_ids: List of cluster IDs
        top_k: Number of recommendations per cluster

    Returns:
        recommendations: {cluster_id: LabelRecommendation}
    """
    recommendations = {}

    for ci, cid in enumerate(cluster_ids):
        cluster_distances = distances[ci]
        sorted_indices = np.argsort(cluster_distances)

        top_labels = []
        for idx in sorted_indices[:top_k]:
            label = gt_type_names[idx]
            dist = cluster_distances[idx]
            top_labels.append((label, float(dist)))

        recommendations[cid] = LabelRecommendation(
            cluster_id=cid,
            labels=top_labels,
        )

    return recommendations


def print_gt_statistics(gt_library: GTLibrary) -> None:
    """Print GT extraction statistics table."""
    print("\n=== GT Transition Statistics ===\n")
    print(f"{'Transition Type':<30} {'Count':>6} {'Avg':>7} {'Std':>6} {'Min':>5} {'Max':>5} {'Up':>4} {'Down':>5}")
    print("-" * 85)

    for trans_type in sorted(gt_library.statistics.keys()):
        stats = gt_library.statistics[trans_type]
        print(
            f"{trans_type:<30} {stats.count:>6} {stats.avg_original_frames:>7.1f} "
            f"{stats.std_original_frames:>6.1f} {stats.min_frames:>5} {stats.max_frames:>5} "
            f"{stats.n_upsampled:>4} {stats.n_downsampled:>5}"
        )

    print(f"\nTarget length: {gt_library.target_len} frames")


def print_recommendations(
    recommendations: dict[int, LabelRecommendation],
    method_name: str = "",
    seed: int = 0,
) -> None:
    """Print label recommendations."""
    header = f"=== Label Recommendations"
    if method_name:
        header += f" ({method_name}"
        if seed is not None:
            header += f" seed {seed}"
        header += ")"
    header += " ==="
    print(f"\n{header}\n")

    for cid in sorted(recommendations.keys()):
        rec = recommendations[cid]
        print(f"Cluster {cid}:")
        for i, (label, dist) in enumerate(rec.labels):
            marker = " <- RECOMMENDED" if i == 0 else ""
            print(f"  {i+1}. {label:<30} (dist: {dist:.4f}){marker}")
        print()


def save_recommendations_csv(
    recommendations: dict[int, LabelRecommendation],
    output_path: str,
) -> None:
    """Save recommendations to CSV file."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "rank", "label", "distance"])

        for cid in sorted(recommendations.keys()):
            rec = recommendations[cid]
            for rank, (label, dist) in enumerate(rec.labels, 1):
                writer.writerow([cid, rank, label, f"{dist:.6f}"])

    print(f"Saved recommendations to: {output_path}")


def save_gt_statistics_csv(gt_library: GTLibrary, output_path: str) -> None:
    """Save GT statistics to CSV file."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "transition_type", "count", "avg_frames", "std_frames",
            "min_frames", "max_frames", "n_upsampled", "n_downsampled"
        ])

        for trans_type in sorted(gt_library.statistics.keys()):
            stats = gt_library.statistics[trans_type]
            writer.writerow([
                trans_type,
                stats.count,
                f"{stats.avg_original_frames:.2f}",
                f"{stats.std_original_frames:.2f}",
                stats.min_frames,
                stats.max_frames,
                stats.n_upsampled,
                stats.n_downsampled,
            ])

    print(f"Saved GT statistics to: {output_path}")
