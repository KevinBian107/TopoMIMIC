'''Functions for TDA with time delayed intention embedding, capturing time series nature'''

import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
import ripser

from sklearn.cluster import KMeans, AgglomerativeClustering
from persim import wasserstein
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def compute_persistence_diagrams(
    embedded: np.ndarray,
    max_dim: int = 2,
) -> dict:
    """
    Compute persistence diagrams using Ripser.

    Args:
        embedded: Point cloud array of shape (N, D)
        max_dim: Maximum homology dimension to compute

    Returns:
        Dictionary containing diagrams, H0/H1/H2 arrays, and statistics
    """

    if len(embedded) < 2:
        empty_result = {
            "diagrams": [np.array([]) for _ in range(max_dim + 1)],
            "H0": np.array([]),
            "H1": np.array([]),
            "H2": np.array([]),
            "H1_persistence": np.array([]),
            "H1_sorted": np.array([]),
            "H1_stats": {"mean": 0, "median": 0, "max": 0, "top_5": np.array([])},
            "H2_persistence": np.array([]),
            "H2_sorted": np.array([]),
            "H2_stats": {"mean": 0, "median": 0, "max": 0, "top_5": np.array([])},
        }
        return empty_result

    try:
        diagrams = ripser.ripser(
            embedded,
            maxdim=max_dim,
            thresh=np.inf,
            coeff=2,
            do_cocycles=False,
        )["dgms"]

        H0 = diagrams[0]
        H1 = diagrams[1] if len(diagrams) > 1 else np.array([])
        H2 = diagrams[2] if max_dim >= 2 and len(diagrams) > 2 else np.array([])

        # H1 stats
        H1_persistence = H1[:, 1] - H1[:, 0] if len(H1) > 0 else np.array([])
        H1_sorted_idx = np.argsort(-H1_persistence) if len(H1_persistence) > 0 else []
        H1_sorted = H1[H1_sorted_idx] if len(H1_sorted_idx) > 0 else np.array([])
        H1_sorted_persistence = (
            H1_persistence[H1_sorted_idx] if len(H1_sorted_idx) > 0 else np.array([])
        )
        H1_stats = {
            "mean": np.mean(H1_persistence) if len(H1_persistence) > 0 else 0,
            "median": np.median(H1_persistence) if len(H1_persistence) > 0 else 0,
            "max": np.max(H1_persistence) if len(H1_persistence) > 0 else 0,
            "top_5": (
                H1_sorted_persistence[:5]
                if len(H1_sorted_persistence) >= 5
                else H1_sorted_persistence
            ),
        }

        # H2 stats
        H2_persistence = H2[:, 1] - H2[:, 0] if len(H2) > 0 else np.array([])
        H2_sorted_idx = np.argsort(-H2_persistence) if len(H2_persistence) > 0 else []
        H2_sorted = H2[H2_sorted_idx] if len(H2_sorted_idx) > 0 else np.array([])
        H2_sorted_persistence = (
            H2_persistence[H2_sorted_idx] if len(H2_sorted_idx) > 0 else np.array([])
        )
        H2_stats = {
            "mean": np.mean(H2_persistence) if len(H2_persistence) > 0 else 0,
            "median": np.median(H2_persistence) if len(H2_persistence) > 0 else 0,
            "max": np.max(H2_persistence) if len(H2_persistence) > 0 else 0,
            "top_5": (
                H2_sorted_persistence[:5]
                if len(H2_sorted_persistence) >= 5
                else H2_sorted_persistence
            ),
        }

        return {
            "diagrams": diagrams,
            "H0": H0,
            "H1": H1,
            "H2": H2,
            "H1_persistence": H1_persistence,
            "H1_sorted": H1_sorted,
            "H1_stats": H1_stats,
            "H2_persistence": H2_persistence,
            "H2_sorted": H2_sorted,
            "H2_stats": H2_stats,
        }

    except Exception as e:
        print("Error in compute_persistence_diagrams:", e)
        return {
            "diagrams": [np.array([]) for _ in range(max_dim + 1)],
            "H0": np.array([]),
            "H1": np.array([]),
            "H2": np.array([]),
            "H1_persistence": np.array([]),
            "H1_sorted": np.array([]),
            "H1_stats": {"mean": 0, "median": 0, "max": 0, "top_5": np.array([])},
            "H2_persistence": np.array([]),
            "H2_sorted": np.array([]),
            "H2_stats": {"mean": 0, "median": 0, "max": 0, "top_5": np.array([])},
        }


def compute_tda_per_clip_with_delayed_embedding(
    intentions: np.ndarray,
    m: int = 5,
    frames_per_clip: int = 600,
    max_dim: int = 2,
    max_points: int = 500,
) -> list[dict]:
    """
    Perform TDA for each clip independently using Takens embeddings.

    Args:
        intentions: Intention array of shape (n_total, dim)
        m: Half-window size for Takens embedding
        frames_per_clip: Number of frames per clip
        max_dim: Maximum homology dimension
        max_points: Maximum points for computation

    Returns:
        List of dicts with clip_idx, frame ranges, and diagrams
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


def visualize_wasserstein_distance_matrix(
    tda_results: list[dict],
    max_clips: int | None = None,
    diag_type: str = "H1",
    skip_clips: list[int] = [37, 152, 163, 274, 281],
) -> np.ndarray:
    """
    Compute and visualize Wasserstein distance matrix between clip persistence diagrams.

    Args:
        tda_results: List of TDA result dicts from compute_tda_per_clip_with_delayed_embedding
        max_clips: Maximum number of clips to consider
        diag_type: Diagram type to compare ("H0", "H1", or "H2")
        skip_clips: Clip indices to omit

    Returns:
        Wasserstein distance matrix of shape (n_kept, n_kept)
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
    wass_matrix: np.ndarray,
    n_clusters: int = 4,
    title: str = "Hierarchical Clustering of Clip Topologies (Wasserstein H1)",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform hierarchical clustering and visualize clustermap from Wasserstein distance matrix.

    Args:
        wass_matrix: Square distance matrix
        n_clusters: Number of clusters for agglomerative clustering
        title: Plot title

    Returns:
        Tuple of (cluster_labels, linkage_matrix)
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



def delayed_embedding_tda_then_projection(
    tda_results: list[dict],
    reduction: str = "pca",
    n_components: int = 2,
    n_clusters: int | None = None,
    show_variance: bool = True,
    random_state: int = 42,
) -> None:
    """
    Summarize and visualize topological signatures using dimensionality reduction.

    Args:
        tda_results: List of TDA result dicts
        reduction: Reduction method ("pca", "tsne", or "umap")
        n_components: Number of components for projection (2 or 3)
        n_clusters: Number of KMeans clusters (None for no clustering)
        show_variance: Whether to print variance explained (PCA only)
        random_state: Random seed for reproducibility
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