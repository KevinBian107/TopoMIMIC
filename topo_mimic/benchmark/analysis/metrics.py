"""Metrics for cluster evaluation: DTW sequence comparison."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .ground_truth import find_all_transitions, TransitionSegment
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


def extract_sequences_from_cluster(
    cluster_entries: list,
    data: np.ndarray,
    min_seq_length: int = 3,
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
) -> list[np.ndarray]:
    """
    Extract sequences for a specific GT transition type with optional turn filtering.

    Args:
        transitions: List of TransitionSegment
        data: (num_clips, num_frames, dim) feature array
        target_behaviors: Sorted tuple of two behaviors (e.g., ("Immobility", "Walking"))
        turn_direction: If LEFT/RIGHT, filter to matching turns. If STRAIGHT or
                        NON_DETERMINISTIC or None, include all GT sequences.
        min_seq_length: Minimum sequence length
        max_seqs: Maximum sequences to return

    Returns:
        List of (T, D) sequences matching the criteria
    """
    sequences = []

    filter_by_turn = (
        turn_direction is not None
        and turn_direction not in (TurnDirection.STRAIGHT, TurnDirection.NON_DETERMINISTIC)
    )

    for t in transitions:
        if t.transition_key != target_behaviors:
            continue

        if filter_by_turn:
            has_matching_turn = (
                t.from_turn == turn_direction or t.to_turn == turn_direction
            )
            if not has_matching_turn:
                continue

        if t.clip_id >= len(data):
            continue

        frames = np.array(t.all_frames)
        valid_frames = frames[(frames >= 0) & (frames < len(data[t.clip_id]))]

        if len(valid_frames) < min_seq_length:
            continue

        valid_frames = np.sort(valid_frames)
        seq = data[t.clip_id, valid_frames]
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


@dataclass
class PerJointDTWResult:
    label: str
    turn_direction: str | None
    cluster_ids: list[int]
    joint_dtw: dict[str, float]
    joint_std: dict[str, float]
    n_discovered_seqs: int
    n_gt_seqs: int


DEFAULT_ANALYSIS_JOINTS = [
    # Legs
    "hip_L_extend",
    "hip_R_extend",
    "knee_L",
    "knee_R",
    # Arms
    "elbow_L",
    "elbow_R",
    # Head (for orienting)
    "atlas",
    "vertebra_axis_twist",
    # Torso
    "vertebra_6_twist",
    "vertebra_3_twist",
]


def resolve_joint_config(
    joint_names_config: list[str] | None,
    skip_root_dims: int = 7,
) -> tuple[list[int], list[str]]:
    """
    Resolve configured joint names to indices.

    Args:
        joint_names_config: List of joint names from config, or None for defaults
        skip_root_dims: Not used (kept for API compatibility)

    Returns:
        (joint_indices, joint_names) - indices into the data array (after root dims skipped)
    """
    from .dtw_visualization import JOINT_NAMES

    if joint_names_config is None:
        joint_names_config = DEFAULT_ANALYSIS_JOINTS

    joint_indices = []
    joint_names = []

    for name in joint_names_config:
        if name in JOINT_NAMES:
            joint_idx = JOINT_NAMES.index(name)
            joint_indices.append(joint_idx)
            joint_names.append(name)
        else:
            print(f"Warning: Joint '{name}' not found in JOINT_NAMES")

    return joint_indices, joint_names


def compute_per_joint_dtw(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    joint_indices: list[int],
) -> dict[int, float]:
    """
    Compute DTW distance for each joint separately.

    Args:
        seq_a: (T1, D) first sequence
        seq_b: (T2, D) second sequence
        joint_indices: List of joint indices to compute DTW for

    Returns:
        Dict mapping joint_index -> DTW distance
    """
    results = {}
    for joint_idx in joint_indices:
        seq_a_joint = seq_a[:, joint_idx:joint_idx+1]
        seq_b_joint = seq_b[:, joint_idx:joint_idx+1]
        results[joint_idx] = compute_dtw(seq_a_joint, seq_b_joint)
    return results


def compute_cluster_per_joint_dtw(
    seqs_a: list[np.ndarray],
    seqs_b: list[np.ndarray],
    joint_indices: list[int],
    max_comparisons: int = 100,
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Compute average per-joint DTW between two sets of sequences.

    Args:
        seqs_a: List of sequences from cluster A
        seqs_b: List of sequences from cluster B
        joint_indices: List of joint indices to compute DTW for
        max_comparisons: Max pairwise comparisons

    Returns:
        (mean_dtw_per_joint, std_dtw_per_joint)
    """
    if not seqs_a or not seqs_b:
        return (
            {j: float('inf') for j in joint_indices},
            {j: 0.0 for j in joint_indices},
        )

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

    joint_distances = {j: [] for j in joint_indices}
    for i, j in pairs:
        per_joint = compute_per_joint_dtw(seqs_a[i], seqs_b[j], joint_indices)
        for joint_idx, dist in per_joint.items():
            if np.isfinite(dist):
                joint_distances[joint_idx].append(dist)

    mean_dtw = {}
    std_dtw = {}
    for joint_idx in joint_indices:
        dists = joint_distances[joint_idx]
        if dists:
            mean_dtw[joint_idx] = float(np.mean(dists))
            std_dtw[joint_idx] = float(np.std(dists))
        else:
            mean_dtw[joint_idx] = float('inf')
            std_dtw[joint_idx] = 0.0

    return mean_dtw, std_dtw


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

        target_behaviors = tuple(sorted(parsed.primitives[:2]))
        turn_dir_str = turn_directions.get(cluster_id)

        key = (target_behaviors, turn_dir_str)
        grouped_clusters[key].append(cluster_id)

    print(f"    Found {len(grouped_clusters)} unique (label, turn) groups")

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
            seqs = extract_sequences_from_cluster(entries, data, min_seq_length)
            combined_seqs.extend(seqs)

        if not combined_seqs:
            label_str = f"{target_behaviors[0]}↔{target_behaviors[1]}"
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
        )

        label_str = f"{target_behaviors[0]}↔{target_behaviors[1]}"
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


def run_per_joint_dtw_analysis(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    turn_directions: dict[int, str] | None,
    behavior_labels_csv: str,
    data: np.ndarray,
    output_dir: str,
    joint_indices: list[int],
    joint_names: list[str],
    method_name: str = "",
    seed: int = 0,
    max_samples: int = 50,
    max_comparisons: int = 100,
    short_threshold: int = 10,
    merge_rear: bool = True,
    min_seq_length: int = 3,
) -> list[PerJointDTWResult]:
    """
    Compute per-joint DTW for discovered clusters vs corresponding GT.

    Args:
        cluster_dict: Mapping cluster_id -> list of (clip_id, frames)
        cluster_labels: Mapping cluster_id -> label string
        turn_directions: Mapping cluster_id -> turn direction
        behavior_labels_csv: Path to behavior labels CSV
        data: (num_clips, num_frames, dim) feature array (already root-skipped)
        output_dir: Directory to save outputs
        joint_indices: List of joint indices to analyze
        joint_names: Names for the joints
        method_name: Name for output files
        seed: Seed for output files
        max_samples: Max sequences per cluster
        max_comparisons: Max pairwise DTW comparisons
        short_threshold: Max frames for intermediate behavior bridging
        merge_rear: Merge rear categories
        min_seq_length: Minimum sequence length

    Returns:
        List of PerJointDTWResult for each unique (label, turn_direction) combination
    """
    import os
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    if turn_directions is None:
        turn_directions = {}

    print("    Loading GT transitions for per-joint DTW...")
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

    print(f"\n    Computing per-joint DTW ({len(joint_indices)} joints) for {len(grouped_clusters)} groups...")

    for (target_behaviors, turn_dir_str), cluster_ids in tqdm(
        grouped_clusters.items(), desc="    Per-joint DTW"
    ):
        turn_dir = parse_turn_direction(turn_dir_str)

        combined_seqs = []
        for cid in cluster_ids:
            entries = cluster_dict.get(cid, [])
            seqs = extract_sequences_from_cluster(entries, data, min_seq_length)
            combined_seqs.extend(seqs)

        if not combined_seqs:
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
        )

        if not gt_seqs:
            continue

        mean_dtw, std_dtw = compute_cluster_per_joint_dtw(
            combined_seqs,
            gt_seqs,
            joint_indices,
            max_comparisons=max_comparisons,
        )

        label_str = f"{target_behaviors[0]}↔{target_behaviors[1]}"

        joint_dtw_named = {joint_names[i]: mean_dtw[joint_indices[i]] for i in range(len(joint_indices))}
        joint_std_named = {joint_names[i]: std_dtw[joint_indices[i]] for i in range(len(joint_indices))}

        results.append(PerJointDTWResult(
            label=label_str,
            turn_direction=turn_dir_str,
            cluster_ids=cluster_ids,
            joint_dtw=joint_dtw_named,
            joint_std=joint_std_named,
            n_discovered_seqs=len(combined_seqs),
            n_gt_seqs=len(gt_seqs),
        ))

    prefix = f"{method_name}_seed{seed}" if method_name else f"seed{seed}"

    if results:
        rows = []
        for r in results:
            row = {
                "Label": r.label,
                "Turn_Direction": r.turn_direction or "none",
                "Cluster_IDs": ",".join(str(c) for c in r.cluster_ids),
            }
            for jname in joint_names:
                row[f"DTW_{jname}"] = r.joint_dtw.get(jname, float('inf'))
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, f"{prefix}_per_joint_dtw.csv")
        df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"\n    Saved per-joint DTW results to {csv_path}")

        plot_per_joint_dtw_heatmap(results, joint_names, output_dir, prefix)

    return results


def plot_per_joint_dtw_heatmap(
    results: list[PerJointDTWResult],
    joint_names: list[str],
    output_dir: str,
    prefix: str = "",
) -> None:
    """Plot per-joint DTW results as a heatmap (clusters x joints)."""
    import os

    if not results:
        return

    labels = []
    for r in results:
        turn_suffix = f" [{r.turn_direction}]" if r.turn_direction else ""
        labels.append(f"{r.label}{turn_suffix}")

    matrix = np.zeros((len(results), len(joint_names)))
    for i, r in enumerate(results):
        for j, jname in enumerate(joint_names):
            matrix[i, j] = r.joint_dtw.get(jname, float('inf'))

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=np.nanmax(matrix[np.isfinite(matrix)]))

    fig, ax = plt.subplots(figsize=(max(8, len(joint_names) * 0.8), max(4, len(results) * 0.4)))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(np.arange(len(joint_names)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(joint_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(results)):
        for j in range(len(joint_names)):
            val = matrix[i, j]
            text_color = 'white' if val > matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7, color=text_color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("DTW Distance", fontsize=10)

    ax.set_xlabel("Joint", fontsize=11)
    ax.set_ylabel("Cluster", fontsize=11)
    ax.set_title("Per-Joint DTW: Discovered vs GT", fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{prefix}_per_joint_dtw_heatmap.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved per-joint DTW heatmap to {save_path}")


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


