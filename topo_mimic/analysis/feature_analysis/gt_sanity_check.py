"""GT cluster sanity check via DTW.

Compares movement sequences between top GT transition clusters
to verify they represent distinct behavioral states.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from .ground_truth import find_all_transitions, TransitionSegment
from .metrics import compute_dtw, compute_cluster_dtw_distance


@dataclass
class GTDTWResult:
    transition_a: str
    transition_b: str
    mean_dtw: float
    std_dtw: float
    n_seqs_a: int
    n_seqs_b: int


def extract_gt_sequences(
    transitions: list[TransitionSegment],
    data: np.ndarray,
    max_seqs_per_cluster: int = 50,
    min_seq_length: int = 3,
) -> dict[str, list[np.ndarray]]:
    """
    Extract sequences for each GT transition type.

    Args:
        transitions: List of TransitionSegment
        data: (num_clips, num_frames, dim) feature array
        max_seqs_per_cluster: Max sequences per cluster
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

    result = {}
    for label, seqs in sequences_by_label.items():
        if len(seqs) < 2:
            continue

        if len(seqs) > max_seqs_per_cluster:
            np.random.seed(42)
            indices = np.random.choice(len(seqs), max_seqs_per_cluster, replace=False)
            seqs = [seqs[i] for i in indices]

        result[label] = seqs

    return result


def run_gt_dtw_sanity_check(
    behavior_labels_csv: str,
    data: np.ndarray,
    output_dir: str,
    top_k: int = 10,
    max_seqs_per_cluster: int = 50,
    max_comparisons: int = 100,
    short_threshold: int = 10,
    merge_rear: bool = True,
    figsize: tuple[float, float] = (12, 10),
    skip_root_dims: int = 7,
    min_seq_length: int = 3,
) -> list[GTDTWResult]:
    """
    Run GT sanity check comparing transition sequences using DTW.

    DTW compares the actual movement patterns (sequences), not just
    individual frame positions. This captures temporal dynamics.

    Args:
        behavior_labels_csv: Path to behavior labels CSV
        data: (num_clips, num_frames, dim) feature array
        output_dir: Directory to save outputs
        top_k: Number of top GT transition types to compare
        max_seqs_per_cluster: Max sequences per cluster
        max_comparisons: Max pairwise DTW comparisons per cluster pair
        short_threshold: Max frames for intermediate behavior bridging
        merge_rear: Merge rear categories
        figsize: Figure size for plots
        skip_root_dims: Number of root dimensions to skip
        min_seq_length: Minimum sequence length

    Returns:
        List of GTDTWResult for all pairwise comparisons
    """
    import os
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    if skip_root_dims > 0:
        original_dim = data.shape[-1]
        data = data[..., skip_root_dims:]
        print(f"  Removed first {skip_root_dims} root dims: {original_dim}D -> {data.shape[-1]}D")

    print("  Loading GT transitions...")
    all_transitions = find_all_transitions(
        behavior_labels_csv,
        short_threshold=short_threshold,
        merge_rear=merge_rear,
    )

    if not all_transitions:
        print("  No GT transitions found, skipping DTW sanity check")
        return []

    print(f"  Found {len(all_transitions)} total GT transitions")

    print("  Extracting sequences for GT clusters...")
    gt_seqs = extract_gt_sequences(
        all_transitions,
        data,
        max_seqs_per_cluster=max_seqs_per_cluster,
        min_seq_length=min_seq_length,
    )

    seq_counts = {label: len(seqs) for label, seqs in gt_seqs.items()}
    top_clusters = sorted(seq_counts.keys(), key=lambda x: seq_counts[x], reverse=True)[:top_k]

    print(f"  Selected top {len(top_clusters)} GT clusters:")
    for i, label in enumerate(top_clusters):
        seqs = gt_seqs[label]
        avg_len = np.mean([len(s) for s in seqs])
        print(f"    {i+1}. {label}: {len(seqs)} sequences (avg length: {avg_len:.1f})")

    gt_seqs = {label: gt_seqs[label] for label in top_clusters}

    print(f"\n  Computing pairwise DTW distances...")
    n_clusters = len(top_clusters)
    dtw_matrix = np.zeros((n_clusters, n_clusters))
    std_matrix = np.zeros((n_clusters, n_clusters))

    results = []
    pairs = list(combinations(range(n_clusters), 2))

    for i, j in tqdm(pairs, desc="  Pairwise DTW"):
        label_i = top_clusters[i]
        label_j = top_clusters[j]

        mean_dtw, std_dtw = compute_cluster_dtw_distance(
            gt_seqs[label_i],
            gt_seqs[label_j],
            max_comparisons=max_comparisons,
        )

        dtw_matrix[i, j] = mean_dtw
        dtw_matrix[j, i] = mean_dtw
        std_matrix[i, j] = std_dtw
        std_matrix[j, i] = std_dtw

        results.append(GTDTWResult(
            transition_a=label_i,
            transition_b=label_j,
            mean_dtw=mean_dtw,
            std_dtw=std_dtw,
            n_seqs_a=len(gt_seqs[label_i]),
            n_seqs_b=len(gt_seqs[label_j]),
        ))

    plot_gt_dtw_heatmap(
        top_clusters,
        dtw_matrix,
        output_dir,
        figsize=figsize,
    )

    df = pd.DataFrame([
        {
            "Transition_A": r.transition_a,
            "Transition_B": r.transition_b,
            "Mean_DTW": r.mean_dtw,
            "Std_DTW": r.std_dtw,
            "N_Seqs_A": r.n_seqs_a,
            "N_Seqs_B": r.n_seqs_b,
        }
        for r in results
    ])
    df = df.sort_values("Mean_DTW", ascending=True)
    csv_path = os.path.join(output_dir, "gt_dtw_sanity_check_results.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved results to {csv_path}")

    mean_dtw_all = np.mean([r.mean_dtw for r in results])
    min_dtw = min(r.mean_dtw for r in results)
    max_dtw = max(r.mean_dtw for r in results)
    print(f"\n  DTW Summary:")
    print(f"    Mean DTW across all pairs: {mean_dtw_all:.4f}")
    print(f"    Range: [{min_dtw:.4f}, {max_dtw:.4f}]")

    print(f"\n  Most similar GT pairs (lowest DTW):")
    sorted_results = sorted(results, key=lambda r: r.mean_dtw)[:3]
    for r in sorted_results:
        print(f"    {r.transition_a} ↔ {r.transition_b}: {r.mean_dtw:.4f}")

    print(f"\n  Most distinct GT pairs (highest DTW):")
    sorted_results = sorted(results, key=lambda r: r.mean_dtw, reverse=True)[:3]
    for r in sorted_results:
        print(f"    {r.transition_a} ↔ {r.transition_b}: {r.mean_dtw:.4f}")

    return results


def plot_gt_dtw_heatmap(
    labels: list[str],
    dtw_matrix: np.ndarray,
    output_dir: str,
    figsize: tuple[float, float] = (12, 10),
) -> None:
    """Plot GT pairwise DTW distance heatmap."""
    import os

    n = len(labels)
    short_labels = [_shorten_label(l) for l in labels]

    fig, ax = plt.subplots(figsize=figsize)

    mask = np.eye(n, dtype=bool)
    dtw_display = np.ma.masked_where(mask, dtw_matrix)

    cmap = LinearSegmentedColormap.from_list("dtw", ["#2ECC71", "#F1C40F", "#E74C3C"], N=256)
    im = ax.imshow(dtw_display, cmap=cmap, aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)

    vmax = dtw_matrix.max()
    for i in range(n):
        for j in range(n):
            if i != j:
                val = dtw_matrix[i, j]
                color = "white" if val > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=8, fontweight='bold')

    ax.set_title("GT Transition Pairwise DTW Distance\n(lower = more similar movement pattern)",
                 fontsize=13, fontweight='bold', pad=10)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("DTW Distance (normalized)", fontsize=10)

    plt.tight_layout()

    save_path = os.path.join(output_dir, "gt_dtw_sanity_check_heatmap.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved DTW heatmap to {save_path}")

    _plot_dtw_similarity_summary(labels, dtw_matrix, output_dir)


def _shorten_label(label: str) -> str:
    """Shorten transition label for display."""
    abbrevs = {
        "Walking": "Walk",
        "Immobility": "Immob",
        "Orienting": "Orient",
        "Grooming": "Groom",
        "Rear": "Rear",
    }
    parts = label.split("↔")
    if len(parts) == 2:
        a = abbrevs.get(parts[0], parts[0][:5])
        b = abbrevs.get(parts[1], parts[1][:5])
        return f"{a}↔{b}"
    return label[:12]


def _plot_dtw_similarity_summary(
    labels: list[str],
    dtw_matrix: np.ndarray,
    output_dir: str,
) -> None:
    """Plot summary showing average DTW distance per cluster."""
    import os

    n = len(labels)
    avg_dtw = []
    for i in range(n):
        distances = [dtw_matrix[i, j] for j in range(n) if i != j]
        avg_dtw.append(np.mean(distances))

    short_labels = [_shorten_label(l) for l in labels]

    sorted_indices = np.argsort(avg_dtw)

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n))
    bars = ax.barh(range(n), [avg_dtw[i] for i in sorted_indices],
                   color=[colors[np.argsort(sorted_indices)[i]] for i in range(n)],
                   edgecolor='white', linewidth=1.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels([short_labels[i] for i in sorted_indices], fontsize=10)
    ax.set_xlabel("Average DTW Distance to Other GT Clusters", fontsize=11)
    ax.set_title("GT Cluster Distinctiveness (DTW)\n(higher = more unique movement pattern)",
                 fontsize=13, fontweight='bold')

    for i, (bar, idx) in enumerate(zip(bars, sorted_indices)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{avg_dtw[idx]:.3f}", va='center', fontsize=9)

    plt.tight_layout()

    save_path = os.path.join(output_dir, "gt_dtw_distinctiveness_summary.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved DTW distinctiveness summary to {save_path}")
