'''Functions for TDA with time delayed intention embedding, capturing time series nature'''

import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from sklearn.cluster import KMeans, AgglomerativeClustering
from persim import wasserstein
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from vnl_tda.pipelines.tda.topo_space.tda_general import (
    compute_persistence_diagrams
)


def analyze_intention_structure_over_time(all_results):
    """
    Analyze how the structure of the intention manifold changes over time,
    including both H1 and H2 features.
    """

    n_clusters = len(all_results["cluster_results"])
    if n_clusters < 2:
        print("Not enough clusters for temporal analysis")
        return

    cluster_indices = [r["cluster_idx"] for r in all_results["cluster_results"]]
    start_frames = [r["start_frame"] for r in all_results["cluster_results"]]

    # Wasserstein distances between H1 diagrams
    cluster_tda_h1_distances = []
    cluster_tda_h2_distances = []

    for i in range(1, n_clusters):
        prev_cluster = all_results["cluster_results"][i - 1]
        curr_cluster = all_results["cluster_results"][i]

        prev_H1 = prev_cluster["diagrams"]["H1"]
        curr_H1 = curr_cluster["diagrams"]["H1"]
        prev_H2 = prev_cluster["diagrams"]["H2"]
        curr_H2 = curr_cluster["diagrams"]["H2"]

        # H1
        if len(prev_H1) > 0 and len(curr_H1) > 0:
            try:
                wass_dist_h1 = wasserstein(prev_H1, curr_H1)
            except:
                wass_dist_h1 = 0.0
        else:
            wass_dist_h1 = 0.0
        cluster_tda_h1_distances.append(wass_dist_h1)

        # H2
        if len(prev_H2) > 0 and len(curr_H2) > 0:
            try:
                wass_dist_h2 = wasserstein(prev_H2, curr_H2)
            except:
                wass_dist_h2 = 0.0
        else:
            wass_dist_h2 = 0.0
        cluster_tda_h2_distances.append(wass_dist_h2)

    # Plot Wasserstein distances
    plt.figure(figsize=(12, 6))
    plt.plot(
        cluster_indices[1:], cluster_tda_h1_distances, "o-", label="H1 Wasserstein"
    )
    plt.plot(
        cluster_indices[1:], cluster_tda_h2_distances, "s-", label="H2 Wasserstein"
    )
    plt.title("Topological Distance Between Consecutive Clusters")
    plt.xlabel("Cluster Index")
    plt.ylabel("Wasserstein Distance")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot counts and mean persistence over time
    plt.figure(figsize=(12, 8))

    h1_counts = [len(r["diagrams"]["H1"]) for r in all_results["cluster_results"]]
    h1_means = [
        r["diagrams"]["H1_stats"]["mean"] for r in all_results["cluster_results"]
    ]
    h2_counts = [len(r["diagrams"]["H2"]) for r in all_results["cluster_results"]]
    h2_means = [
        r["diagrams"]["H2_stats"]["mean"] for r in all_results["cluster_results"]
    ]

    plt.plot(start_frames, h1_counts, "o-", label="H1 Count")
    plt.plot(start_frames, h1_means, "o--", label="H1 Mean Persistence")
    plt.plot(start_frames, h2_counts, "s-", label="H2 Count")
    plt.plot(start_frames, h2_means, "s--", label="H2 Mean Persistence")

    plt.title("Evolution of Topological Features Over Time")
    plt.xlabel("Start Frame")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def compute_tda_per_clip_with_delayed_embedding(
    intentions, m=5, frames_per_clip=600, max_dim=2, max_points=500
):
    """
    Perform TDA for each clip independently, using Takens embeddings.
    """
    n_total, dim = intentions.shape
    n_clips = n_total // frames_per_clip

    tda_results = []

    for clip_idx in range(n_clips):
        print(f"Processing clip {clip_idx+1}/{n_clips}...")

        start = clip_idx * frames_per_clip
        end = (clip_idx + 1) * frames_per_clip
        clip_data = intentions[start:end]

        points = []
        for center in range(m, frames_per_clip - m):
            window = clip_data[center - m : center + m + 1]  # shape (2m+1, D)
            point = window.flatten()  # shape ((2m+1)*D,)
            points.append(point)

        points = np.array(points)  # (frames_per_clip - 2m, (2m+1)*D)

        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]

        diagrams = compute_persistence_diagrams(points, max_dim=max_dim)

        tda_results.append(
            {
                "clip_idx": clip_idx,
                "start_frame": start,
                "end_frame": end,
                "diagrams": diagrams,
            }
        )
    return tda_results


def visualize_wasserstein_distance_matrix(tda_results,
                                          max_clips=None,
                                          diag_type="H1",
                                          skip_clips=[37, 152, 163, 274, 281]):
    """
    Computes and visualizes a Wasserstein distance matrix between clip-level persistence
    diagrams of type `diag_type`, omitting any clips listed in `skip_clips`.
    """
    if max_clips is None:
        max_clips = len(tda_results)

    keep = [i for i in range(max_clips) if i not in skip_clips]
    # print(f"Keeping clips: {keep}")
    
    n = len(keep)
    wass_matrix = np.zeros((n, n))

    for idx_i, i in enumerate(keep):
        diag_i = tda_results[i]["diagrams"][diag_type]
        for idx_j, j in enumerate(keep[idx_i+1:], start=idx_i+1):
            diag_j = tda_results[j]["diagrams"][diag_type]
            dist = wasserstein(diag_i, diag_j)
            wass_matrix[idx_i, idx_j] = wass_matrix[idx_j, idx_i] = dist

    plt.figure(figsize=(10, 10))
    sns.heatmap(
        wass_matrix,
        cmap="viridis",
        # xticklabels=keep,
        # yticklabels=keep,
        square=True,
    )
    # plt.title(f"Wasserstein Distance Between Clip Persistence Diagrams ({diag_type})")
    plt.xlabel("Clip Index")
    plt.ylabel("Clip Index")
    plt.tight_layout()
    plt.show()

    return wass_matrix


def cluster_and_plot_wasserstein_matrix(
    wass_matrix,
    n_clusters=4,
    title="Hierarchical Clustering of Clip Topologies (Wasserstein H1)",
):
    """
    Performs hierarchical clustering and visualizes the clustermap from a Wasserstein distance matrix,
    with clip number labels shown.
    """
    # Agglomerative clustering on precomputed distance
    clip_clusters = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    ).fit_predict(wass_matrix)

    # convert to condensed form for linkage
    wass_condensed = squareform(wass_matrix)
    linkage_matrix = linkage(wass_condensed, method="average")

    clip_labels = np.arange(wass_matrix.shape[0])

    g = sns.clustermap(
        wass_matrix,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        cmap="viridis",
        xticklabels=clip_labels,
        yticklabels=clip_labels,
    )
    plt.title(title, pad=80)  # pad moves the title above the dendrogram
    plt.show()

    return clip_clusters, linkage_matrix


def plot_behavior_clustermap_from_wasserstein(
    wass_matrix,
    all_labels,
    title="Clustering of Clip Topologies by Behavior (Wasserstein H1)",
):
    """
    Plots a hierarchical clustering heatmap of a Wasserstein distance matrix, colored by behavior labels.
    """
    # subset labels to match matrix size
    subset_labels = np.array(all_labels)[: wass_matrix.shape[0]]

    # map labels to a colors
    unique_labels = np.unique(subset_labels)
    palette = sns.color_palette("tab10", len(unique_labels))
    label_to_color = dict(zip(unique_labels, palette))
    row_colors = pd.Series(subset_labels).map(label_to_color).to_numpy()

    # linkage for clustering
    linkage_matrix = linkage(squareform(wass_matrix), method="average")

    g = sns.clustermap(
        wass_matrix,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        row_colors=row_colors,
        col_colors=row_colors,
        cmap="viridis",
        xticklabels=False,
        yticklabels=False,
        figsize=(10, 10),
    )

    handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in label_to_color.items()
    ]
    g.ax_heatmap.legend(
        handles=handles,
        title="Behavior",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )

    plt.title(title, pad=100)
    plt.show()


def delayed_embedding_tda_then_projection(
    tda_results,
    reduction="pca",
    n_components=2,
    n_clusters=None,
    show_variance=True,
    random_state=42,
):
    """
    Summarize and visualize topological signatures using dimensionality reduction.
    """

    def summarize_topology(result):
        stats = result["diagrams"]["H1_stats"]
        return [
            len(result["diagrams"]["H1"]),
            stats["mean"],
            stats["max"],
            stats["median"],
        ]

    assert reduction in ["pca", "tsne", "umap"], "Invalid reduction method"

    summary_features = np.array([summarize_topology(r) for r in tda_results])

    if reduction == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        proj = reducer.fit_transform(summary_features)

        if show_variance:
            explained = reducer.explained_variance_ratio_
            for i, var in enumerate(explained):
                print(f"Variance explained by PC{i+1}: {var:.2%}")

    elif reduction == "tsne":
        reducer = TSNE(n_components=n_components, random_state=random_state)
        proj = reducer.fit_transform(summary_features)

    elif reduction == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        proj = reducer.fit_transform(summary_features)

    # cluster
    if n_clusters is not None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(summary_features)
        colors = labels
        title_suffix = f"(KMeans, k={n_clusters})"
    else:
        colors = "blue"
        title_suffix = ""

    fig = plt.figure(figsize=(8, 6))

    if n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D

        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            proj[:, 0], proj[:, 1], proj[:, 2], c=colors, cmap="tab10", alpha=0.8
        )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
    else:
        plt.scatter(proj[:, 0], proj[:, 1], c=colors, cmap="tab10", alpha=0.8)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)

    plt.title(
        f"Topological Signature Projection via {reduction.upper()} {title_suffix}"
    )
    plt.tight_layout()
    plt.show()