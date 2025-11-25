'''For TDA with independent density cloud, examining general topology'''


import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN

from vnl_tda.pipelines.tda.topo_space.tda_general import (
    extract_persistent_features,
    compute_persistence_diagrams
)
from vnl_tda.pipelines.tda.topo_space.tda_features import (
    identify_topological_regions
)


def clean_gait_array(arr):
    """Remove NaNs and negative values."""
    arr = np.array(arr)
    arr = arr[~np.isnan(arr)]
    arr = arr[arr >= 0]
    return arr


def plot_distribution_helper(high_values, low_values, key, discrete=False):
    """Plot distribution comparison between high-H1 and low-H1 groups."""
    plt.figure(figsize=(6, 4))

    if discrete:
        high_rounded = np.round(high_values, 2)
        low_rounded = np.round(low_values, 2)

        unique_vals = np.sort(np.unique(np.concatenate([high_rounded, low_rounded])))

        high_counts = [np.sum(high_rounded == val) for val in unique_vals]
        low_counts = [np.sum(low_rounded == val) for val in unique_vals]

        plt.bar(
            unique_vals - 0.0025, high_counts, width=0.005, label="High H1", alpha=0.7
        )
        plt.bar(
            unique_vals + 0.0025, low_counts, width=0.005, label="Low H1", alpha=0.7
        )

        plt.xlabel(f"{key} (rounded)")
        plt.ylabel("Count")
        plt.title(f"Discrete Distribution of {key}")
        plt.legend()

    else:
        # Smooth continuous plot
        from seaborn import kdeplot

        kdeplot(high_values, label="High H1", fill=True)
        kdeplot(low_values, label="Low H1", fill=True)

        plt.xlabel(key)
        plt.ylabel("Density")
        plt.title(f"Distribution of {key}")
        plt.legend()

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    

def find_clip_from_cluster(tda_results, cluster_idx, frames_per_clip=600):
    """
    Given a cluster index, find the corresponding clip number.
    """
    cluster_results = tda_results["cluster_results"]

    if cluster_idx < 0 or cluster_idx >= len(cluster_results):
        print(f"Invalid cluster index: {cluster_idx}")
        return None

    start_frame = cluster_results[cluster_idx]["start_frame"]
    clip_number = start_frame // frames_per_clip

    print(f"Cluster {cluster_idx}:")
    print(f" - Start frame = {start_frame}")
    print(f" - Clip number = {clip_number}")

    return clip_number


def find_clips_with_few_but_persistent_H1(
    tda_results, max_h1_count=3, persistence_threshold=0.2, frames_per_clip=600
):
    """
    Find clips where clusters have few H1 features but at least one highly persistent H1 loop.
    """
    cluster_results = tda_results["cluster_results"]

    matching_clusters = []
    matching_clips = []

    for idx, cluster in enumerate(cluster_results):
        h1_diagram = np.array(cluster["diagrams"].get("H1", []))
        if h1_diagram.shape[0] == 0:
            continue

        # Compute persistence for each H1 feature
        persistences = h1_diagram[:, 1] - h1_diagram[:, 0]

        # Conditions:
        few_h1 = len(h1_diagram) <= max_h1_count
        persistent_loop = np.any(persistences >= persistence_threshold)

        if few_h1 and persistent_loop:
            matching_clusters.append(idx)
            start_frame = cluster["start_frame"]
            clip_idx = start_frame // frames_per_clip
            matching_clips.append(clip_idx)

            print(
                f"Cluster {idx} | Start Frame {start_frame} | Clip {clip_idx} | H1 count: {len(h1_diagram)}, Max persistence: {np.max(persistences):.3f}"
            )

    matching_clips = np.unique(matching_clips)
    return matching_clips


def find_clips_with_many_but_weak_H1(
    tda_results, min_h1_count=10, persistence_threshold=0.2, frames_per_clip=600
):
    """
    Find clips where clusters have many H1 features but all are weak (low persistence).
    """
    cluster_results = tda_results["cluster_results"]

    matching_clusters = []
    matching_clips = []

    for idx, cluster in enumerate(cluster_results):
        h1_diagram = np.array(cluster["diagrams"].get("H1", []))
        if h1_diagram.shape[0] == 0:
            continue

        # Compute persistence for each H1 feature
        persistences = h1_diagram[:, 1] - h1_diagram[:, 0]

        # Conditions:
        many_h1 = len(h1_diagram) >= min_h1_count
        weak_loops = np.all(persistences < persistence_threshold)

        if many_h1 and weak_loops:
            matching_clusters.append(idx)
            start_frame = cluster["start_frame"]
            clip_idx = start_frame // frames_per_clip
            matching_clips.append(clip_idx)

            print(
                f"Cluster {idx} | Start Frame {start_frame} | Clip {clip_idx} | H1 count: {len(h1_diagram)}, Max persistence: {np.max(persistences):.3f}"
            )

    matching_clips = np.unique(matching_clips)
    return matching_clips


def find_high_h1_spike_clusters(
    tda_results, top_percentile=99, total_clips=600, plot=True
):
    """
    From tda_results, find clusters with high H1 feature counts.
    """
    cluster_results = tda_results["cluster_results"]

    h1_counts = np.array([len(r["diagrams"]["H1"]) for r in cluster_results])
    start_frames = np.array([r["start_frame"] for r in cluster_results])

    threshold = np.percentile(h1_counts, top_percentile)

    spike_cluster_indices = np.where(h1_counts >= threshold)[0]
    spike_start_frames = start_frames[spike_cluster_indices]

    print(
        f"Clusters with high H1 counts (>{top_percentile} percentile): {spike_cluster_indices}"
    )
    print(f"Corresponding start frames: {spike_start_frames}")
    print(f"Corresponding clips: {spike_start_frames//total_clips}")

    if plot:
        plt.figure(figsize=(12, 8))
        plt.plot(start_frames, h1_counts, "o-", label="H1 Count")
        plt.axhline(
            threshold,
            color="r",
            linestyle="--",
            label=f"{top_percentile}th Percentile Threshold",
        )
        for frame in spike_start_frames:
            plt.axvline(x=frame, color="r", linestyle="--", alpha=0.5)
        plt.xlabel("Start Frame")
        plt.ylabel("H1 Feature Count")
        plt.title("H1 Feature Count Over Time with Spike Markers")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    return spike_cluster_indices, spike_start_frames


def identify_analyze_high_low_h1_gait(
    tda_results, gait_metrics, percentile_threshold=99.7, total_clips=600
):
    """
    Analyze gait metric differences between high-H1 and low-H1 clips.
    """
    cluster_results = tda_results["cluster_results"]
    start_frames = np.array([r["start_frame"] for r in cluster_results])
    h1_counts = np.array([len(r["diagrams"]["H1"]) for r in cluster_results])

    # clusters to clip indices
    clip_indices_from_clusters = start_frames // total_clips

    # thresholding to find high-H1 clips
    threshold = np.percentile(h1_counts, percentile_threshold)

    high_H1_clip_indices = np.unique(
        [
            clip_idx
            for clip_idx, count in zip(clip_indices_from_clusters, h1_counts)
            if count >= threshold
        ]
    )
    low_H1_clip_indices = np.unique(
        [
            clip_idx
            for clip_idx, count in zip(clip_indices_from_clusters, h1_counts)
            if count < threshold
        ]
    )

    print(f"High-H1 clips: {high_H1_clip_indices}")

    # match gait metrics
    clip_indices_gait = gait_metrics["clip_indices"]  # (N windows,)

    high_mask = np.isin(clip_indices_gait, high_H1_clip_indices)
    low_mask = np.isin(clip_indices_gait, low_H1_clip_indices)

    print(f"Number of high-H1 gait samples: {np.sum(high_mask)}")
    print(f"Number of low-H1 gait samples: {np.sum(low_mask)}")

    # analyze gait distributions
    gait_keys = ["frequencies", "duty_factor", "stance_durations", "swing_durations"]

    for key in gait_keys:
        high_values = gait_metrics[key][high_mask]
        low_values = gait_metrics[key][low_mask]

        if len(high_values) == 0 or len(low_values) == 0:
            print(f"{key}: Not enough data for t-test.")
            continue

        t_stat, p_val = ttest_ind(high_values, low_values, equal_var=False)
        mean_high = np.mean(high_values)
        mean_low = np.mean(low_values)

        print(
            f"{key}: p = {p_val:.4f} (mean_high = {mean_high:.3f}, mean_low = {mean_low:.3f})"
        )

    return high_mask, low_mask


def analyze_h1_distributions(gait_metrics, high_mask, low_mask):
    gait_keys = ["frequencies", "stance_durations", "swing_durations"]

    # Decide which metrics should be treated as discrete
    discrete_keys = ["stance_durations", "swing_durations"]

    for key in gait_keys:
        high_values = clean_gait_array(gait_metrics[key][high_mask])
        low_values = clean_gait_array(gait_metrics[key][low_mask])

        if len(high_values) == 0 or len(low_values) == 0:
            print(f"{key}: Not enough data after cleaning. Skipping.")
            continue

        # Mann-Whitney U test
        t_stat, p_val = ttest_ind(high_values, low_values, equal_var=False)
        mean_high = np.mean(high_values)
        mean_low = np.mean(low_values)

        print(
            f"{key}: p = {p_val:.4f} (mean_high = {mean_high:.3f}, mean_low = {mean_low:.3f})"
        )

        plot_distribution_helper(high_values, low_values, key, discrete=(key in discrete_keys))


def analyze_h1_vs_stance_swing_spearmanr(
    tda_results, gait_metrics, frames_per_clip=600
):
    """
    Analyze how H1 number and H1 persistence relate to stance and swing durations using spearmanr.
    """
    cluster_results = tda_results["cluster_results"]
    h1_counts = np.array([len(r["diagrams"]["H1"]) for r in cluster_results])
    h1_persistences = np.array(
        [
            (
                np.max(
                    np.array(r["diagrams"]["H1"])[:, 1]
                    - np.array(r["diagrams"]["H1"])[:, 0]
                )
                if len(r["diagrams"]["H1"]) > 0
                else 0
            )
            for r in cluster_results
        ]
    )
    clip_indices_from_clusters = np.array(
        [r["start_frame"] // frames_per_clip for r in cluster_results]
    )

    clip_indices_gait = gait_metrics["clip_indices"]
    stance_durations = gait_metrics["stance_durations"]
    swing_durations = gait_metrics["swing_durations"]

    # match gait samples to cluster statistics
    matched_h1_counts = []
    matched_h1_persistences = []
    matched_stance = []
    matched_swing = []

    for gait_clip, stance, swing in zip(
        clip_indices_gait, stance_durations, swing_durations
    ):
        idx = np.where(clip_indices_from_clusters == gait_clip)[0]
        if len(idx) > 0:
            idx = idx[0]
            matched_h1_counts.append(h1_counts[idx])
            matched_h1_persistences.append(h1_persistences[idx])
            matched_stance.append(stance)
            matched_swing.append(swing)

    matched_h1_counts = np.array(matched_h1_counts)
    matched_h1_persistences = np.array(matched_h1_persistences)
    matched_stance = np.array(matched_stance)
    matched_swing = np.array(matched_swing)

    gait_keys = {"stance_durations": matched_stance, "swing_durations": matched_swing}

    for gait_name, gait_array in gait_keys.items():
        corr_count, p_count = spearmanr(matched_h1_counts, gait_array)
        corr_persist, p_persist = spearmanr(matched_h1_persistences, gait_array)

        print(f"--- {gait_name} ---")
        print(
            f" H1 Number vs {gait_name}: Spearman r={corr_count:.3f}, p={p_count:.4f}"
        )
        print(
            f" H1 Max Persistence vs {gait_name}: Spearman r={corr_persist:.3f}, p={p_persist:.4f}"
        )

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(matched_h1_counts, gait_array, alpha=0.5)
        plt.xlabel("Number of H1 Loops")
        plt.ylabel(gait_name)
        plt.title(f"H1 Number vs {gait_name}")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(matched_h1_persistences, gait_array, alpha=0.5)
        plt.xlabel("Max Persistence of H1")
        plt.ylabel(gait_name)
        plt.title(f"H1 Max Persistence vs {gait_name}")
        plt.grid(True)

        plt.tight_layout()
        plt.show()


def cluster_with_dbscan_then_tda(
    embedded, stride=10, max_points=500, eps_factor=0.25, gait_metrics=None
):
    """
    Perform TDA on density-based clusters (via DBSCAN),
    returning results in the same structure as the original sliding cluster method, and using all provided arguments.
    """

    min_samples = 3

    # eps to determine which points are close enough to each other to form a dense region
    # when eps is small, only points that are very close to each other are considered neighbors, many small clusters
    # when eps is big, further apart points are clustered dotegther
    eps = eps_factor * np.std(embedded)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embedded)
    labels = clustering.labels_
    unique_labels = np.unique(labels[labels >= 0])  # ignore noise (-1)
    print(f" Number of unique clusters are: {len(unique_labels)}")

    all_results = []

    for i, cluster_id in enumerate(unique_labels[::stride]):
        true_idx = unique_labels[::stride][i]
        indices = np.where(labels == true_idx)[0]
        cluster_data = embedded[indices]

        print(f"Cluster data shape: {cluster_data.shape}")

        if len(cluster_data) > max_points:
            selected = np.random.choice(len(cluster_data), max_points, replace=False)
            cluster_data = cluster_data[selected]
            indices = indices[selected]

        diagrams = compute_persistence_diagrams(cluster_data, max_dim=2)

        diameter = (
            np.max(np.linalg.norm(cluster_data, axis=1)) if len(cluster_data) > 0 else 0
        )
        threshold = diameter * 0.1
        persistent_features = extract_persistent_features(diagrams, threshold)

        cluster_gait_metrics = None
        tda_regions = None
        if gait_metrics is not None:
            cluster_gait_metrics = {
                key: (
                    value[indices]
                    if isinstance(value, np.ndarray) and len(value) >= np.max(indices)
                    else value
                )
                for key, value in gait_metrics.items()
            }
            try:
                print("Finding topological_regions")
                tda_regions = identify_topological_regions(
                    cluster_data, diagrams, persistent_features, cluster_gait_metrics
                )
            except:
                tda_regions = None

        cluster_result = {
            "cluster_idx": i,
            "start_frame": int(np.min(indices)),
            "end_frame": int(np.max(indices)),
            "diagrams": diagrams,
            "persistent_features": persistent_features,
            "tda_regions": tda_regions,
        }

        all_results.append(cluster_result)

    return all_results