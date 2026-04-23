"""Metrics for cluster evaluation: DTW sequence comparison."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .ground_truth import (
    find_all_transitions,
    find_pure_behavior_segments,
    TransitionSegment,
    BehaviorSegment,
)
from .behavior_parsing import (
    parse_cluster_label,
    TurnDirection,
    parse_turn_direction,
)


# =============================================================================
# DTW (Dynamic Time Warping) Functions
# =============================================================================

def compute_dtw(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping distance between two sequences.

    Args:
        seq_a: (T1, D) first sequence
        seq_b: (T2, D) second sequence

    Returns:
        DTW distance (lower = more similar movement pattern)
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float('inf')

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq_a[i - 1] - seq_b[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    return dtw_matrix[n, m] / (n + m)


def _resample_to_K(arr: np.ndarray, K: int) -> np.ndarray:
    """Linear resample (T, D) -> (K, D)."""
    if arr.ndim != 2:
        raise ValueError(f"expected (T, D), got {arr.shape}")
    T = arr.shape[0]
    if T == K:
        return arr.astype(np.float32, copy=True)
    if T == 1:
        return np.broadcast_to(arr, (K, arr.shape[1])).astype(np.float32, copy=True)
    src_idx = np.linspace(0, T - 1, K, dtype=np.float64)
    lo = np.floor(src_idx).astype(np.int64)
    hi = np.clip(lo + 1, 0, T - 1)
    frac = (src_idx - lo).astype(np.float32)
    out = (1.0 - frac[:, None]) * arr[lo] + frac[:, None] * arr[hi]
    return out.astype(np.float32, copy=True)


def extract_sequences_from_cluster(
    cluster_entries: list,
    data: np.ndarray,
    min_seq_length: int = 3,
    resample_K: int | None = None,
) -> list[np.ndarray]:
    """
    Extract sequences from cluster entries.

    Args:
        cluster_entries: List of (clip_id, frame_indices) tuples
        data: (num_clips, num_frames, dim) feature array
        min_seq_length: Minimum sequence length to include

    Returns:
        List of (T, D) sequence arrays
    """
    sequences = []

    for clip_id, frame_indices in cluster_entries:
        clip_id = int(clip_id)
        if clip_id >= len(data):
            continue

        frames = np.asarray(frame_indices).flatten()
        frames = np.sort(frames)
        valid_frames = frames[(frames >= 0) & (frames < len(data[clip_id]))]

        if len(valid_frames) < min_seq_length:
            continue

        seq = data[clip_id, valid_frames]
        if resample_K is not None and seq.shape[0] != resample_K:
            seq = _resample_to_K(seq, resample_K)
        sequences.append(seq)

    return sequences


def extract_sequences_from_transitions(
    transitions: list[TransitionSegment],
    data: np.ndarray,
    min_seq_length: int = 3,
) -> dict[str, list[np.ndarray]]:
    """
    Extract sequences for each GT transition type.

    Args:
        transitions: List of TransitionSegment
        data: (num_clips, num_frames, dim) feature array
        min_seq_length: Minimum sequence length

    Returns:
        Dict mapping transition label -> list of (T, D) sequences
    """
    sequences_by_label = defaultdict(list)

    for t in transitions:
        label = f"{t.transition_key[0]}↔{t.transition_key[1]}"

        if t.clip_id >= len(data):
            continue

        frames = np.array(t.all_frames)
        valid_frames = frames[(frames >= 0) & (frames < len(data[t.clip_id]))]

        if len(valid_frames) < min_seq_length:
            continue

        valid_frames = np.sort(valid_frames)
        seq = data[t.clip_id, valid_frames]
        sequences_by_label[label].append(seq)

    return dict(sequences_by_label)


def extract_sequences_for_gt_label(
    transitions: list[TransitionSegment],
    data: np.ndarray,
    target_behaviors: tuple[str, str],
    turn_direction: TurnDirection | None = None,
    min_seq_length: int = 3,
    max_seqs: int = 50,
    directed: bool = True,
    resample_K: int | None = None,
) -> list[np.ndarray]:
    """
    Extract sequences for a specific GT transition type with optional turn filtering.

    Args:
        transitions: List of TransitionSegment
        data: (num_clips, num_frames, dim) feature array
        target_behaviors: (from, to) tuple — e.g. ("Immobile", "Walk").
            When ``directed=True``, matches only that direction in GT.
            When ``directed=False``, matches either order (undirected).
        turn_direction: If LEFT/RIGHT, filter to matching turns. If STRAIGHT or
                        NON_DETERMINISTIC or None, include all GT sequences.
        min_seq_length: Minimum sequence length
        max_seqs: Maximum sequences to return
        directed: When True, match GT direction exactly; when False, fall back
                  to the undirected (sorted) key.

    Returns:
        List of (T, D) sequences matching the criteria
    """
    sequences = []

    filter_by_turn = (
        turn_direction is not None
        and turn_direction != TurnDirection.NON_DETERMINISTIC
    )

    if directed:
        target_key = tuple(target_behaviors)
    else:
        target_key = tuple(sorted(target_behaviors))

    for t in transitions:
        t_key = t.directed_key if directed else t.transition_key
        if t_key != target_key:
            continue

        if filter_by_turn and t.turn != turn_direction:
            continue

        if t.clip_id >= len(data):
            continue

        frames = np.array(t.all_frames)
        valid_frames = frames[(frames >= 0) & (frames < len(data[t.clip_id]))]

        if len(valid_frames) < min_seq_length:
            continue

        valid_frames = np.sort(valid_frames)
        seq = data[t.clip_id, valid_frames]
        if resample_K is not None and seq.shape[0] != resample_K:
            seq = _resample_to_K(seq, resample_K)
        sequences.append(seq)

    if len(sequences) > max_seqs:
        np.random.seed(42)
        indices = np.random.choice(len(sequences), max_seqs, replace=False)
        sequences = [sequences[i] for i in indices]

    return sequences


def extract_sequences_from_behavior_segments(
    segments: list[BehaviorSegment],
    data: np.ndarray,
    min_seq_length: int = 3,
    max_seqs: int = 50,
) -> list[np.ndarray]:
    """
    Extract trajectory sequences from pure behavior segments (frame ranges).

    Args:
        segments: List of BehaviorSegment with clip_id and frame ranges
        data: (num_clips, num_frames, dim) feature array
        min_seq_length: Minimum sequence length to include
        max_seqs: Maximum sequences to return

    Returns:
        List of (T, D) sequence arrays
    """
    sequences = []

    for seg in segments:
        if seg.clip_id >= len(data):
            continue

        frames = np.arange(seg.start_frame, seg.end_frame + 1)
        valid_frames = frames[(frames >= 0) & (frames < len(data[seg.clip_id]))]

        if len(valid_frames) < min_seq_length:
            continue

        seq = data[seg.clip_id, valid_frames]
        sequences.append(seq)

    if len(sequences) > max_seqs:
        np.random.seed(42)
        indices = np.random.choice(len(sequences), max_seqs, replace=False)
        sequences = [sequences[i] for i in indices]

    return sequences


def compute_cluster_dtw_distance(
    seqs_a: list[np.ndarray],
    seqs_b: list[np.ndarray],
    max_comparisons: int = 100,
) -> tuple[float, float]:
    """
    Compute average DTW distance between two sets of sequences.

    Args:
        seqs_a: List of sequences from cluster A
        seqs_b: List of sequences from cluster B
        max_comparisons: Max pairwise comparisons to limit computation

    Returns:
        (mean_dtw, std_dtw)
    """
    if not seqs_a or not seqs_b:
        return float('inf'), 0.0

    n_a, n_b = len(seqs_a), len(seqs_b)
    total_pairs = n_a * n_b

    if total_pairs <= max_comparisons:
        pairs = [(i, j) for i in range(n_a) for j in range(n_b)]
    else:
        np.random.seed(42)
        pairs = []
        for _ in range(max_comparisons):
            i = np.random.randint(n_a)
            j = np.random.randint(n_b)
            pairs.append((i, j))

    dtw_distances = []
    for i, j in pairs:
        d = compute_dtw(seqs_a[i], seqs_b[j])
        if np.isfinite(d):
            dtw_distances.append(d)

    if not dtw_distances:
        return float('inf'), 0.0

    return float(np.mean(dtw_distances)), float(np.std(dtw_distances))


@dataclass
class DTWResult:
    label: str
    turn_direction: str | None
    cluster_ids: list[int]
    mean_dtw: float
    std_dtw: float
    n_discovered_seqs: int
    n_gt_seqs: int


def run_dtw_analysis(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    turn_directions: dict[int, str] | None,
    behavior_labels_csv: str,
    data: np.ndarray,
    output_dir: str,
    method_name: str = "",
    seed: int = 0,
    max_samples: int = 50,
    max_comparisons: int = 100,
    short_threshold: int = 10,
    merge_rear: bool = True,
    skip_root_dims: int = 7,
    min_seq_length: int = 3,
    pure_behavior_labels: list[str] | None = None,
    pure_behavior_min_length: int = 30,
    resample_K: int | None = None,
) -> list[DTWResult]:
    """
    Compare discovered clusters to their corresponding GT using DTW distance.

    Clusters with the same label AND turn direction are combined before
    comparing to the corresponding GT transition type.

    Args:
        cluster_dict: Mapping cluster_id -> list of (clip_id, frames)
        cluster_labels: Mapping cluster_id -> label string (e.g., "Walking Immobility")
        turn_directions: Mapping cluster_id -> turn direction ("left", "right", or None)
        behavior_labels_csv: Path to behavior labels CSV
        data: (num_clips, num_frames, dim) feature array
        output_dir: Directory to save outputs
        method_name: Name for output files
        seed: Seed for output files
        max_samples: Max sequences per cluster
        max_comparisons: Max pairwise DTW comparisons
        short_threshold: Max frames for intermediate behavior bridging
        merge_rear: Merge rear categories
        skip_root_dims: Number of root dimensions to skip
        min_seq_length: Minimum sequence length

    Returns:
        List of DTWResult for each unique (label, turn_direction) combination
    """
    import os
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    if turn_directions is None:
        turn_directions = {}

    if skip_root_dims > 0:
        original_dim = data.shape[-1]
        data = data[..., skip_root_dims:]
        print(f"    Removed first {skip_root_dims} root dims: {original_dim}D -> {data.shape[-1]}D")

    print("    Loading GT transitions...")
    all_transitions = find_all_transitions(
        behavior_labels_csv,
        short_threshold=short_threshold,
        merge_rear=merge_rear,
    )
    print(f"    Found {len(all_transitions)} total GT transitions")

    print("\n    Grouping discovered clusters by (label, turn_direction)...")
    grouped_clusters = defaultdict(list)

    for cluster_id, label in cluster_labels.items():
        if not label or label.startswith("C"):
            continue

        parsed = parse_cluster_label(label, merge_rear=merge_rear)
        if parsed is None or parsed.comparison_type != "transition":
            continue

        if len(parsed.primitives) < 2:
            continue

        # Directional grouping: preserve (from, to) order from the parsed
        # cluster label so "Immobile -> Walk" and "Walk -> Immobile" stay
        # distinct DTW rows.
        target_behaviors = tuple(parsed.primitives[:2])
        turn_dir_str = turn_directions.get(cluster_id)

        key = (target_behaviors, turn_dir_str)
        grouped_clusters[key].append(cluster_id)

    print(f"    Found {len(grouped_clusters)} unique (directed label, turn) groups")

    results = []
    skipped = []

    print("\n    Computing DTW for each group vs corresponding GT...")

    for (target_behaviors, turn_dir_str), cluster_ids in tqdm(
        grouped_clusters.items(), desc="    DTW analysis"
    ):
        turn_dir = parse_turn_direction(turn_dir_str)

        combined_seqs = []
        for cid in cluster_ids:
            entries = cluster_dict.get(cid, [])
            seqs = extract_sequences_from_cluster(entries, data, min_seq_length, resample_K=resample_K)
            combined_seqs.extend(seqs)

        if not combined_seqs:
            label_str = f"{target_behaviors[0]} -> {target_behaviors[1]}"
            skipped.append((cluster_ids, label_str, "no valid sequences"))
            continue

        if len(combined_seqs) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(combined_seqs), max_samples, replace=False)
            combined_seqs = [combined_seqs[i] for i in indices]

        gt_seqs = extract_sequences_for_gt_label(
            all_transitions,
            data,
            target_behaviors,
            turn_direction=turn_dir,
            min_seq_length=min_seq_length,
            max_seqs=max_samples,
            directed=True,
            resample_K=resample_K,
        )

        label_str = f"{target_behaviors[0]} -> {target_behaviors[1]}"
        if not gt_seqs:
            turn_info = f" (turn={turn_dir_str})" if turn_dir_str else ""
            skipped.append((cluster_ids, label_str, f"no GT sequences{turn_info}"))
            continue

        mean_dtw, std_dtw = compute_cluster_dtw_distance(
            combined_seqs,
            gt_seqs,
            max_comparisons=max_comparisons,
        )

        results.append(DTWResult(
            label=label_str,
            turn_direction=turn_dir_str,
            cluster_ids=cluster_ids,
            mean_dtw=mean_dtw,
            std_dtw=std_dtw,
            n_discovered_seqs=len(combined_seqs),
            n_gt_seqs=len(gt_seqs),
        ))

        turn_info = f" [{turn_dir_str}]" if turn_dir_str else ""
        cluster_str = ", ".join(f"C{c}" for c in cluster_ids)
        print(f"      {label_str}{turn_info} ({cluster_str}): "
              f"DTW={mean_dtw:.3f}±{std_dtw:.3f} "
              f"(n={len(combined_seqs)} vs {len(gt_seqs)})")

    if skipped:
        print(f"\n    Skipped {len(skipped)} groups:")
        for cids, lbl, reason in skipped[:5]:
            cluster_str = ", ".join(f"C{c}" for c in cids)
            print(f"      {lbl} ({cluster_str}): {reason}")
        if len(skipped) > 5:
            print(f"      ... and {len(skipped) - 5} more")

    pure_labels = pure_behavior_labels if pure_behavior_labels is not None else ["Grooming"]
    if pure_labels:
        print(f"\n    Computing DTW for pure behaviors: {pure_labels}")

        pure_grouped = defaultdict(list)
        for cluster_id, label in cluster_labels.items():
            if not label or label.startswith("C"):
                continue

            parsed = parse_cluster_label(label, merge_rear=merge_rear)
            if parsed is None or parsed.comparison_type != "single":
                continue

            for target_behavior in pure_labels:
                from .behavior_parsing import normalize_behavior_name, merge_rear_category
                normalized = normalize_behavior_name(target_behavior)
                if merge_rear:
                    normalized = merge_rear_category(normalized)
                if normalized in parsed.primitives:
                    turn_dir_str = turn_directions.get(cluster_id)
                    key = (normalized, turn_dir_str)
                    pure_grouped[key].append(cluster_id)

        gt_segments = find_pure_behavior_segments(
            behavior_labels_csv,
            target_behaviors=pure_labels,
            min_length=pure_behavior_min_length,
            merge_rear=merge_rear,
        )

        if gt_segments:
            gt_seqs = extract_sequences_from_behavior_segments(
                gt_segments, data, min_seq_length=min_seq_length, max_seqs=max_samples,
            )
            print(f"    Found {len(gt_segments)} GT segments -> {len(gt_seqs)} sequences")
        else:
            gt_seqs = []
            print(f"    No GT segments found for {pure_labels}")

        for (behavior_name, turn_dir_str), cluster_ids in pure_grouped.items():
            combined_seqs = []
            for cid in cluster_ids:
                entries = cluster_dict.get(cid, [])
                seqs = extract_sequences_from_cluster(entries, data, min_seq_length, resample_K=resample_K)
                combined_seqs.extend(seqs)

            if not combined_seqs or not gt_seqs:
                cluster_str = ", ".join(f"C{c}" for c in cluster_ids)
                print(f"      {behavior_name} ({cluster_str}): skipped (no sequences)")
                continue

            if len(combined_seqs) > max_samples:
                np.random.seed(42)
                indices = np.random.choice(len(combined_seqs), max_samples, replace=False)
                combined_seqs = [combined_seqs[i] for i in indices]

            mean_dtw, std_dtw = compute_cluster_dtw_distance(
                combined_seqs, gt_seqs, max_comparisons=max_comparisons,
            )

            results.append(DTWResult(
                label=behavior_name,
                turn_direction=turn_dir_str,
                cluster_ids=cluster_ids,
                mean_dtw=mean_dtw,
                std_dtw=std_dtw,
                n_discovered_seqs=len(combined_seqs),
                n_gt_seqs=len(gt_seqs),
            ))

            turn_info = f" [{turn_dir_str}]" if turn_dir_str else ""
            cluster_str = ", ".join(f"C{c}" for c in cluster_ids)
            print(f"      {behavior_name}{turn_info} ({cluster_str}): "
                  f"DTW={mean_dtw:.3f}±{std_dtw:.3f} "
                  f"(n={len(combined_seqs)} vs {len(gt_seqs)})")

    prefix = f"{method_name}_seed{seed}" if method_name else f"seed{seed}"

    if results:
        df = pd.DataFrame([
            {
                "Label": r.label,
                "Turn_Direction": r.turn_direction or "none",
                "Cluster_IDs": ",".join(str(c) for c in r.cluster_ids),
                "N_Clusters": len(r.cluster_ids),
                "Mean_DTW": r.mean_dtw,
                "Std_DTW": r.std_dtw,
                "N_Discovered_Seqs": r.n_discovered_seqs,
                "N_GT_Seqs": r.n_gt_seqs,
            }
            for r in results
        ])
        df = df.sort_values("Mean_DTW", ascending=True)
        csv_path = os.path.join(output_dir, f"{prefix}_dtw_results.csv")
        df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"\n    Saved results to {csv_path}")

        plot_dtw_bar_chart(results, output_dir, prefix)

        mean_all = np.mean([r.mean_dtw for r in results])
        print(f"\n    Summary:")
        print(f"      Groups evaluated: {len(results)}")
        print(f"      Mean DTW: {mean_all:.4f}")
        best = min(results, key=lambda r: r.mean_dtw)
        worst = max(results, key=lambda r: r.mean_dtw)
        print(f"      Best: {best.label} (DTW={best.mean_dtw:.4f})")
        print(f"      Worst: {worst.label} (DTW={worst.mean_dtw:.4f})")

    return results


@dataclass
class InternalConsistencyResult:
    label: str
    turn_direction: str | None
    cluster_ids: list[int]
    mean_intra_dtw: float
    std_intra_dtw: float
    n_sequences: int
    avg_frames: float


def run_internal_consistency_analysis(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    turn_directions: dict[int, str] | None,
    data: np.ndarray,
    output_dir: str,
    method_name: str = "",
    seed: int = 0,
    skip_root_dims: int = 7,
    min_seq_length: int = 3,
    merge_rear: bool = True,
    n_samples: int = 5,
    resample_K: int | None = None,
) -> list[InternalConsistencyResult]:
    """
    Measure internal cluster consistency via pairwise DTW among sampled sequences.

    Args:
        cluster_dict: Mapping cluster_id -> list of (clip_id, frames)
        cluster_labels: Mapping cluster_id -> label string
        turn_directions: Mapping cluster_id -> turn direction string
        data: (num_clips, num_frames, dim) feature array
        output_dir: Directory to save outputs
        method_name: Name for output files
        seed: Seed for output files
        skip_root_dims: Number of root dimensions to skip
        min_seq_length: Minimum sequence length to include
        merge_rear: Merge rear categories
        n_samples: Number of sequences to sample per cluster group

    Returns:
        List of InternalConsistencyResult for each (label, turn_direction) group
    """
    import os
    import pandas as pd
    from itertools import combinations

    os.makedirs(output_dir, exist_ok=True)

    if turn_directions is None:
        turn_directions = {}

    if skip_root_dims > 0:
        data = data[..., skip_root_dims:]

    grouped_clusters: dict[tuple, list[int]] = defaultdict(list)

    for cluster_id, label in cluster_labels.items():
        if not label or label.startswith("C"):
            continue

        parsed = parse_cluster_label(label, merge_rear=merge_rear)
        if parsed is None:
            continue

        if parsed.comparison_type == "transition" and len(parsed.primitives) >= 2:
            # Directional: preserve (from, to) order from the parsed label.
            label_str = f"{parsed.primitives[0]} -> {parsed.primitives[1]}"
        elif parsed.comparison_type == "single":
            behavior = parsed.primitives[0] if parsed.primitives else parsed.compositional[0]
            label_str = behavior
        else:
            continue

        turn_dir_str = turn_directions.get(cluster_id)
        key = (label_str, turn_dir_str)
        grouped_clusters[key].append(cluster_id)

    print(f"    Internal consistency: {len(grouped_clusters)} (label, turn) groups")

    results = []
    rng = np.random.RandomState(42)

    for (label_str, turn_dir_str), cluster_ids in sorted(grouped_clusters.items()):
        combined_seqs = []
        for cid in cluster_ids:
            entries = cluster_dict.get(cid, [])
            seqs = extract_sequences_from_cluster(entries, data, min_seq_length, resample_K=resample_K)
            combined_seqs.extend(seqs)

        if len(combined_seqs) < 2:
            continue

        if len(combined_seqs) > n_samples:
            indices = rng.choice(len(combined_seqs), n_samples, replace=False)
            sampled = [combined_seqs[i] for i in indices]
        else:
            sampled = combined_seqs

        dtw_distances = []
        for (seq_a, seq_b) in combinations(sampled, 2):
            d = compute_dtw(seq_a, seq_b)
            if np.isfinite(d):
                dtw_distances.append(d)

        if not dtw_distances:
            continue

        avg_frames = float(np.mean([len(s) for s in combined_seqs]))

        result = InternalConsistencyResult(
            label=label_str,
            turn_direction=turn_dir_str,
            cluster_ids=cluster_ids,
            mean_intra_dtw=float(np.mean(dtw_distances)),
            std_intra_dtw=float(np.std(dtw_distances)),
            n_sequences=len(combined_seqs),
            avg_frames=avg_frames,
        )
        results.append(result)

        turn_info = f" [{turn_dir_str}]" if turn_dir_str else ""
        cluster_str = ", ".join(f"C{c}" for c in cluster_ids)
        print(f"      {label_str}{turn_info} ({cluster_str}): "
              f"intra-DTW={result.mean_intra_dtw:.3f}±{result.std_intra_dtw:.3f} "
              f"(n={len(combined_seqs)}, sampled={len(sampled)}, "
              f"avg_frames={avg_frames:.1f})")

    prefix = f"{method_name}_seed{seed}" if method_name else f"seed{seed}"

    if results:
        df = pd.DataFrame([
            {
                "Label": r.label,
                "Turn_Direction": r.turn_direction or "none",
                "Cluster_IDs": ",".join(str(c) for c in r.cluster_ids),
                "Mean_Intra_DTW": r.mean_intra_dtw,
                "Std_Intra_DTW": r.std_intra_dtw,
                "N_Sequences": r.n_sequences,
                "Avg_Frames": r.avg_frames,
            }
            for r in results
        ])
        df = df.sort_values("Mean_Intra_DTW", ascending=True)
        csv_path = os.path.join(output_dir, f"{prefix}_consistency.csv")
        df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"\n    Saved consistency results to {csv_path}")

    return results


def compute_dtw_with_warping_stats(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Compute DTW distance and warping path statistics.

    Args:
        seq_a: (T1, D) first sequence
        seq_b: (T2, D) second sequence

    Returns:
        (dtw_distance, non_diagonal_fraction, mean_temporal_deviation, max_run_length)
        - non_diagonal_fraction: fraction of path steps that are horizontal/vertical (0=perfect, 1=max warp)
        - mean_temporal_deviation: mean |i/n - j/m| over path (0=diagonal, higher=more warp)
        - max_run_length: longest consecutive horizontal or vertical stretch
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float('inf'), 1.0, 1.0, 0

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq_a[i - 1] - seq_b[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [
            (dtw_matrix[i - 1, j - 1], i - 1, j - 1),
            (dtw_matrix[i - 1, j], i - 1, j),
            (dtw_matrix[i, j - 1], i, j - 1),
        ]
        _, i, j = min(candidates, key=lambda x: x[0])
    path.reverse()

    normalized_dist = dtw_matrix[n, m] / (n + m)

    n_diag = 0
    deviations = []
    for idx in range(len(path)):
        pi, pj = path[idx]
        deviations.append(abs(pi / max(n - 1, 1) - pj / max(m - 1, 1)))
        if idx > 0:
            di = pi - path[idx - 1][0]
            dj = pj - path[idx - 1][1]
            if di == 1 and dj == 1:
                n_diag += 1

    n_steps = len(path) - 1
    non_diag_frac = 1.0 - (n_diag / max(n_steps, 1))
    mean_dev = float(np.mean(deviations)) if deviations else 0.0

    max_run = 0
    current_run = 0
    current_dir = None
    for idx in range(1, len(path)):
        di = path[idx][0] - path[idx - 1][0]
        dj = path[idx][1] - path[idx - 1][1]
        step_dir = (di, dj)
        if step_dir != (1, 1):
            if step_dir == current_dir:
                current_run += 1
            else:
                current_dir = step_dir
                current_run = 1
            max_run = max(max_run, current_run)
        else:
            current_dir = None
            current_run = 0

    return normalized_dist, non_diag_frac, mean_dev, max_run


def compute_frame_sync_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
) -> float:
    """
    Compute frame-synchronous distance (no warping).

    Linearly resamples both sequences to the same length, then computes
    mean per-frame L2 distance. This is what you'd get if DTW were not
    allowed to rearrange time.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float('inf')

    L = max(n, m)
    idx_a = np.round(np.linspace(0, n - 1, L)).astype(int)
    idx_b = np.round(np.linspace(0, m - 1, L)).astype(int)
    resampled_a = seq_a[idx_a]
    resampled_b = seq_b[idx_b]

    dists = np.linalg.norm(resampled_a - resampled_b, axis=1)
    return float(np.mean(dists))


def compute_dtw_sync_ratio(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute DTW distance, frame-sync distance, and their ratio.

    Returns:
        (dtw_dist, sync_dist, ratio) where ratio = dtw / sync.
        ratio < 1 means DTW improves over frame-sync (warping helps).
        Lower ratio = more temporal rearrangement needed.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float('inf'), float('inf'), 1.0

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq_a[i - 1] - seq_b[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )
    dtw_dist = dtw_matrix[n, m] / (n + m)
    sync_dist = compute_frame_sync_distance(seq_a, seq_b)

    if sync_dist == 0 or not np.isfinite(sync_dist):
        ratio = 1.0
    else:
        ratio = dtw_dist / sync_dist

    return dtw_dist, sync_dist, ratio


@dataclass
class WarpingResult:
    label: str
    turn_direction: str | None
    mean_dtw: float
    mean_non_diag_frac: float
    mean_temporal_dev: float
    mean_max_run: float
    mean_sync_dist: float
    mean_dtw_sync_ratio: float
    n_comparisons: int


def run_warping_analysis(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    turn_directions: dict[int, str] | None,
    behavior_labels_csv: str,
    data: np.ndarray,
    method_name: str = "",
    max_samples: int = 20,
    max_comparisons: int = 50,
    short_threshold: int = 10,
    merge_rear: bool = True,
    skip_root_dims: int = 7,
    min_seq_length: int = 3,
) -> list[WarpingResult]:
    """
    Compute DTW warping statistics for each (label, turn) group vs GT.

    Returns list of WarpingResult with warping heaviness metrics.
    """
    if turn_directions is None:
        turn_directions = {}

    if skip_root_dims > 0:
        data = data[..., skip_root_dims:]

    all_transitions = find_all_transitions(
        behavior_labels_csv,
        short_threshold=short_threshold,
        merge_rear=merge_rear,
    )

    grouped_clusters = defaultdict(list)
    for cluster_id, label in cluster_labels.items():
        if not label or label.startswith("C"):
            continue
        parsed = parse_cluster_label(label, merge_rear=merge_rear)
        if parsed is None or parsed.comparison_type != "transition":
            continue
        if len(parsed.primitives) < 2:
            continue
        target_behaviors = tuple(parsed.primitives[:2])
        turn_dir_str = turn_directions.get(cluster_id)
        key = (target_behaviors, turn_dir_str)
        grouped_clusters[key].append(cluster_id)

    results = []
    for (target_behaviors, turn_dir_str), cluster_ids in grouped_clusters.items():
        turn_dir = parse_turn_direction(turn_dir_str)

        combined_seqs = []
        for cid in cluster_ids:
            entries = cluster_dict.get(cid, [])
            seqs = extract_sequences_from_cluster(entries, data, min_seq_length, resample_K=resample_K)
            combined_seqs.extend(seqs)

        if not combined_seqs:
            continue

        if len(combined_seqs) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(combined_seqs), max_samples, replace=False)
            combined_seqs = [combined_seqs[i] for i in indices]

        gt_seqs = extract_sequences_for_gt_label(
            all_transitions, data, target_behaviors,
            turn_direction=turn_dir, min_seq_length=min_seq_length,
            max_seqs=max_samples,
            directed=True,
            resample_K=resample_K,
        )

        if not gt_seqs:
            continue

        n_a, n_b = len(combined_seqs), len(gt_seqs)
        total_pairs = n_a * n_b
        if total_pairs <= max_comparisons:
            pairs = [(i, j) for i in range(n_a) for j in range(n_b)]
        else:
            np.random.seed(42)
            pairs = [(np.random.randint(n_a), np.random.randint(n_b))
                     for _ in range(max_comparisons)]

        dtw_vals, ndf_vals, dev_vals, run_vals = [], [], [], []
        sync_vals, ratio_vals = [], []
        for i, j in pairs:
            dtw_d, ndf, dev, mr = compute_dtw_with_warping_stats(
                combined_seqs[i], gt_seqs[j]
            )
            _, sync_d, ratio = compute_dtw_sync_ratio(
                combined_seqs[i], gt_seqs[j]
            )
            if np.isfinite(dtw_d):
                dtw_vals.append(dtw_d)
                ndf_vals.append(ndf)
                dev_vals.append(dev)
                run_vals.append(mr)
                sync_vals.append(sync_d)
                ratio_vals.append(ratio)

        if not dtw_vals:
            continue

        label_str = f"{target_behaviors[0]} -> {target_behaviors[1]}"
        results.append(WarpingResult(
            label=label_str,
            turn_direction=turn_dir_str,
            mean_dtw=float(np.mean(dtw_vals)),
            mean_non_diag_frac=float(np.mean(ndf_vals)),
            mean_temporal_dev=float(np.mean(dev_vals)),
            mean_max_run=float(np.mean(run_vals)),
            mean_sync_dist=float(np.mean(sync_vals)),
            mean_dtw_sync_ratio=float(np.mean(ratio_vals)),
            n_comparisons=len(dtw_vals),
        ))

    return results


def plot_dtw_bar_chart(
    results: list[DTWResult],
    output_dir: str,
    prefix: str = "",
) -> None:
    """Plot DTW results as a horizontal bar chart."""
    import os

    if not results:
        return

    sorted_results = sorted(results, key=lambda r: r.mean_dtw)

    labels = []
    for r in sorted_results:
        turn_suffix = f" [{r.turn_direction}]" if r.turn_direction else ""
        n_clusters = len(r.cluster_ids)
        cluster_str = f"(n={n_clusters})" if n_clusters > 1 else f"(C{r.cluster_ids[0]})"
        labels.append(f"{r.label}{turn_suffix} {cluster_str}")

    dtw_values = [r.mean_dtw for r in sorted_results]
    std_values = [r.std_dtw for r in sorted_results]

    n = len(results)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.4 + 1)))

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n))

    y_pos = np.arange(n)
    bars = ax.barh(y_pos, dtw_values, xerr=std_values, color=colors,
                   edgecolor='white', linewidth=1, capsize=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("DTW Distance (lower = more similar to GT)", fontsize=11)
    ax.set_title("Discovered vs Corresponding GT (DTW)\nClusters grouped by label + turn direction",
                 fontsize=12, fontweight='bold')

    for i, (bar, r) in enumerate(zip(bars, sorted_results)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{r.mean_dtw:.3f}", va='center', fontsize=8)

    ax.invert_yaxis()
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{prefix}_dtw_bar_chart.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved DTW bar chart to {save_path}")


