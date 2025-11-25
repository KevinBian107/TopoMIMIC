'''For TDA features in combination with gait analysis'''

import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

from vnl_tda.pipelines.tda.topo_space.tda_general import (
    subsample_for_tda,
    compute_persistence_diagrams,
    extract_persistent_features,
    compute_topological_features,
    identify_topological_regions,
    cluster_based_on_topology
)

from vnl_tda.pipelines.tda.topo_space.tda_general import (
    visualize_persistence_diagrams,
    visualize_h1_barcodes,
    visualize_h2_barcodes,
)

from vnl_tda.pipelines.tda.topo_space.tda_vis import (
    visualize_topological_correlations,
    visualize_cluster_evolution
)

from vnl_tda.pipelines.tda.topo_space.density_cluster import (
    cluster_with_dbscan_then_tda
)


def intention_tda_general(
    int_rodent,
    jf_rodent,
    stride=10,
    max_points=1000,
    eps_factor=0.25,
    gait_metrics=None,
    embedding_option="combine",
):
    """
    Main entry point: complete function to analyze intention data using TDA and manifold learning.
    """

    int_rodent_reshaped = (
        int_rodent.squeeze() if len(int_rodent.shape) > 2 else int_rodent
    )
    jf_rodent_reshaped = jf_rodent.squeeze() if len(jf_rodent.shape) > 2 else jf_rodent

    all_results = tda_then_cluster(
        int_rodent_reshaped,
        jf_rodent_reshaped,
        stride=stride,
        max_points=max_points,
        eps_factor=eps_factor,
        gait_metrics=gait_metrics,
        embedding_option=embedding_option,
    )

    return all_results


def tda_then_cluster(
    int_rodent,
    jf_rodent,
    stride=10,
    max_points=1000,
    eps_factor=0.25,
    gait_metrics=None,
    embedding_option="combine",
):
    """
    Process intention data directly using TDA with rolling clusters and manifold analysis.

    1. conduct global tda according to max_points & tda is conducted on the UMAP embedding space
    2. conduct clustered tda according to the eps_factor given
    """

    if len(int_rodent.shape) > 2:
        int_rodent = int_rodent.reshape(-1, int_rodent.shape[-1])

    if len(jf_rodent.shape) > 2:
        jf_rodent = jf_rodent.reshape(-1, jf_rodent.shape[-1])

    if embedding_option == "combine":
        print("Using combined embeddings")
        combined_intentions = np.hstack([int_rodent, jf_rodent])

    if embedding_option == "intention":
        print("Using intention embeddings")
        combined_intentions = int_rodent

    if embedding_option == "force":
        print("Using force embeddings")
        combined_intentions = jf_rodent

    reducer = umap.UMAP(
        n_components=3,
        min_dist=0.1,
        n_neighbors=min(30, combined_intentions.shape[0] - 1),
        random_state=42,
    )
    embedded = reducer.fit_transform(combined_intentions)

    all_results = {
        "embedded": embedded,
        "gait_metrics": gait_metrics,
        "cluster_results": [],
        "global_tda": None,
    }

    if embedded.shape[0] > 1:
        # tda upon the UMAP embedded manifold
        global_tda = tda_optimized(
            {"embedded": embedded, "gait_metrics": gait_metrics},
            min(max_points, embedded.shape[0]),
        )
        all_results["global_tda"] = global_tda
    else:
        all_results["global_tda"] = None

    # tda upon the intention/force/combined ambient space
    cluster_tda_results = cluster_with_dbscan_then_tda(
        combined_intentions,
        stride=stride,
        max_points=max_points,
        eps_factor=eps_factor,
        gait_metrics=gait_metrics,
    )
    all_results["cluster_results"] = cluster_tda_results

    if len(cluster_tda_results) > 0:
        try:
            visualize_cluster_evolution(cluster_tda_results)
            visualize_h1_barcodes(cluster_tda_results)
            visualize_h2_barcodes(cluster_tda_results)
        except Exception as e:
            pass

    return all_results


def tda_optimized(results, max_points):
    """
    Given gait metrics, compute an optimized TDA analysis using numerous TDA based features
    """

    embedded = results["embedded"]
    gait_metrics = results.get("gait_metrics", {})

    if len(embedded) < 2:
        print("Not enough points for TDA analysis")
        return None

    min_size = min(len(embedded), len(gait_metrics.get("phases", embedded)))
    embedded = embedded[:min_size]

    for key in gait_metrics:
        if (
            isinstance(gait_metrics[key], np.ndarray)
            and len(gait_metrics[key]) > min_size
        ):
            gait_metrics[key] = gait_metrics[key][:min_size]

    subsampled_embedded = subsample_for_tda(embedded, max_points=max_points)

    diagrams = compute_persistence_diagrams(subsampled_embedded, max_dim=2)

    visualize_persistence_diagrams(diagrams, "Persistence Diagrams (Optimized)")

    diameter = (
        np.max(np.linalg.norm(subsampled_embedded, axis=1))
        if len(subsampled_embedded) > 0
        else 0
    )
    threshold = diameter * 0.2
    persistent_features = extract_persistent_features(diagrams, threshold)

    tda_features = compute_topological_features(subsampled_embedded)

    gait_metrics_subsampled = {
        k: v[: len(subsampled_embedded)] if isinstance(v, np.ndarray) else v
        for k, v in gait_metrics.items()
    }

    tda_results = identify_topological_regions(
        subsampled_embedded, persistent_features, gait_metrics_subsampled
    )

    if tda_results:
        visualize_topological_correlations(
            tda_results, gait_metrics_subsampled, subsampled_embedded
        )

    full_tda_features = {"landscape": tda_features["landscape"]}

    topo_clusters = cluster_based_on_topology(embedded, full_tda_features)

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection="3d")
    ax.scatter(
        embedded[:, 0],
        embedded[:, 1],
        embedded[:, 2] if embedded.shape[1] > 2 else np.zeros(len(embedded)),
        c=topo_clusters,
        cmap="tab10",
        alpha=0.6,
        s=5,
    )
    ax.set_title("Manifold Clusters Based on Topology", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    plt.tight_layout()
    plt.show()

    return {
        "diagrams": diagrams,
        "persistent_features": persistent_features,
        "tda_features": tda_features,
        "tda_results": tda_results,
        "topo_clusters": topo_clusters,
    }