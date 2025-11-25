'''General topological space visualization functions, does not include some graphing rendering functions'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def visualize_cluster_evolution(cluster_results):
    """
    Visualize the evolution of topological features across clusters.
    """

    n_clusters = len(cluster_results)
    if n_clusters == 0:
        return

    cluster_indices = [r["cluster_idx"] for r in cluster_results]
    h1_means = [r["diagrams"]["H1_stats"]["mean"] for r in cluster_results]
    h1_maxes = [r["diagrams"]["H1_stats"]["max"] for r in cluster_results]
    h1_counts = [len(r["diagrams"]["H1"]) for r in cluster_results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(cluster_indices, h1_means, "o-", linewidth=2)
    axes[0, 0].set_title("Mean H1 Persistence Over clusters")
    axes[0, 0].set_xlabel("cluster Index")
    axes[0, 0].set_ylabel("Mean Persistence")
    axes[0, 0].grid(True, linestyle="--", alpha=0.7)

    axes[0, 1].plot(cluster_indices, h1_maxes, "o-", linewidth=2, color="orange")
    axes[0, 1].set_title("Max H1 Persistence Over clusters")
    axes[0, 1].set_xlabel("cluster Index")
    axes[0, 1].set_ylabel("Max Persistence")
    axes[0, 1].grid(True, linestyle="--", alpha=0.7)

    axes[1, 0].plot(cluster_indices, h1_counts, "o-", linewidth=2, color="green")
    axes[1, 0].set_title("Number of H1 Features Over clusters")
    axes[1, 0].set_xlabel("cluster Index")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True, linestyle="--", alpha=0.7)

    try:
        max_features = max(
            [len(r["diagrams"].get("H1_sorted", [])) for r in cluster_results]
        )
        if max_features > 0:
            max_to_show = min(5, max_features)
            persistence_matrix = np.zeros((n_clusters, max_to_show))

            for i, result in enumerate(cluster_results):
                sorted_persistence = result["diagrams"].get("H1_sorted", [])
                for j in range(min(max_to_show, len(sorted_persistence))):
                    if j < len(sorted_persistence) and sorted_persistence.shape[0] > 0:
                        persistence_matrix[i, j] = (
                            sorted_persistence[j, 1] - sorted_persistence[j, 0]
                        )

            im = axes[1, 1].imshow(persistence_matrix, aspect="auto", cmap="viridis")
            axes[1, 1].set_title("Top H1 Persistence Values Over clusters")
            axes[1, 1].set_xlabel("Top Features")
            axes[1, 1].set_ylabel("cluster Index")
            plt.colorbar(im, ax=axes[1, 1], label="Persistence")
    except Exception as e:
        axes[1, 1].text(
            0.5,
            0.5,
            "Error creating persistence heatmap",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )

    plt.tight_layout()
    plt.show()


def visualize_topological_correlations(tda_results, gait_metrics, embedded):
    """
    Visualize potential topological features corrolation with gait.
    """

    if (
        "correlations" not in tda_results
        or len(tda_results["correlations"]) == 0
        or len(embedded) == 0
    ):
        print("No correlations to visualize or empty embedded data")
        return

    correlations = tda_results["correlations"]
    regions = tda_results["regions"]

    fig = plt.figure(figsize=(20, 15))

    ax1 = fig.add_subplot(231, projection="3d")
    ax1.scatter(
        embedded[:, 0],
        embedded[:, 1],
        embedded[:, 2] if embedded.shape[1] > 2 else np.zeros(len(embedded)),
        c="lightgray",
        alpha=0.3,
        s=5,
    )

    colors = plt.cm.tab10.colors
    for i, region in enumerate(regions):
        if i < len(correlations) and np.sum(region) > 0:
            ax1.scatter(
                embedded[region, 0],
                embedded[region, 1],
                (
                    embedded[region, 2]
                    if embedded.shape[1] > 2
                    else np.zeros(np.sum(region))
                ),
                c=[colors[i % len(colors)]],
                alpha=0.6,
                s=10,
                label=f"Loop {i}",
            )

    ax1.set_title("Topological Regions on Manifold", fontsize=14)
    ax1.legend()

    ax2 = fig.add_subplot(232)

    loop_ids = list(correlations.keys())
    gait_params = ["phase", "frequency", "duty_factor"]

    if len(loop_ids) > 0:
        corr_matrix = np.zeros((len(loop_ids), len(gait_params)))

        for i, loop_id in enumerate(loop_ids):
            for j, param in enumerate(gait_params):
                corr_matrix[i, j] = correlations[loop_id][param][0]

        im = ax2.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax2.set_xticks(np.arange(len(gait_params)))
        ax2.set_yticks(np.arange(len(loop_ids)))
        ax2.set_xticklabels(gait_params)
        ax2.set_yticklabels(loop_ids)
        plt.colorbar(im, ax=ax2, label="Correlation")

        for i in range(len(loop_ids)):
            for j in range(len(gait_params)):
                text = ax2.text(
                    j,
                    i,
                    f"{corr_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )

    ax2.set_title("Correlation: Topological Features vs Gait Parameters", fontsize=14)

    ax3 = fig.add_subplot(233)

    if len(loop_ids) > 0:
        persistence_values = [
            correlations[loop_id]["persistence"] for loop_id in loop_ids
        ]
        duty_corr_values = [
            abs(correlations[loop_id]["duty_factor"][0]) for loop_id in loop_ids
        ]
        phase_corr_values = [
            abs(correlations[loop_id]["phase"][0]) for loop_id in loop_ids
        ]
        freq_corr_values = [
            abs(correlations[loop_id]["frequency"][0]) for loop_id in loop_ids
        ]

        ax3.scatter(
            persistence_values, duty_corr_values, label="Duty Factor", alpha=0.7, s=100
        )
        ax3.scatter(
            persistence_values, phase_corr_values, label="Phase", alpha=0.7, s=100
        )
        ax3.scatter(
            persistence_values, freq_corr_values, label="Frequency", alpha=0.7, s=100
        )

        ax3.set_xlabel("Persistence")
        ax3.set_ylabel("Absolute Correlation")
        ax3.set_title("Feature Persistence vs. Parameter Correlation", fontsize=14)
        ax3.legend()
        ax3.grid(True, linestyle="--", alpha=0.7)

    ax4 = fig.add_subplot(234)

    if len(loop_ids) > 0:
        sizes = [correlations[loop_id]["size"] for loop_id in loop_ids]

        bars = ax4.bar(
            loop_ids,
            sizes,
            color=[colors[i % len(colors)] for i in range(len(loop_ids))],
        )

        ax4.set_xlabel("Topological Feature")
        ax4.set_ylabel("Number of Points")
        ax4.set_title("Size of Topological Regions", fontsize=14)

        if len(sizes) > 0:
            max_region_idx = np.argmax(sizes)
            max_region_id = loop_ids[max_region_idx]
            max_region = regions[max_region_idx]

            ax5 = fig.add_subplot(235)

            if (
                len(gait_metrics.get("duty_factor", [])) > 0
                and len(gait_metrics.get("frequencies", [])) > 0
            ):
                ax5.hist(
                    gait_metrics["duty_factor"][max_region],
                    alpha=0.7,
                    label="Duty Factor",
                )
                ax5.hist(
                    gait_metrics["frequencies"][max_region],
                    alpha=0.7,
                    label="Frequency",
                )

                ax5.set_xlabel("Parameter Value")
                ax5.set_ylabel("Count")
                ax5.set_title(f"Parameter Distribution in {max_region_id}", fontsize=14)
                ax5.legend()

            ax6 = fig.add_subplot(236, projection="polar")

            if len(gait_metrics.get("phases", [])) > 0:
                for i, region in enumerate(regions):
                    if np.sum(region) > 5:
                        hist, bins = np.histogram(
                            gait_metrics["phases"][region],
                            bins=16,
                            range=(-np.pi, np.pi),
                        )
                        width = bins[1] - bins[0]
                        ax6.bar(
                            bins[:-1],
                            hist / np.sum(hist),
                            width=width,
                            alpha=0.5,
                            label=f"Loop {i}",
                            color=colors[i % len(colors)],
                        )

                ax6.set_title("Phase Distribution by Topological Region", fontsize=14)
                ax6.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
    

def visualize_tda_cluster_manifolds(all_results):
    """
    Create a grid of manifold plots for selected clusters to see evolution.
    """

    n_clusters = len(all_results["cluster_results"])
    if n_clusters == 0:
        return

    n_to_show = min(9, n_clusters)
    indices_to_show = np.linspace(0, n_clusters - 1, n_to_show, dtype=int)

    rows = int(np.ceil(np.sqrt(n_to_show)))
    cols = int(np.ceil(n_to_show / rows))

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 4), subplot_kw={"projection": "3d"}
    )
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, idx in enumerate(indices_to_show):
        if i < len(axes):
            cluster = all_results["cluster_results"][idx]
            ax = axes[i]

            start_frame = cluster["start_frame"]
            end_frame = cluster["end_frame"]
            embed = all_results["embedded"][start_frame:end_frame]

            if len(embed) > 0:
                ax.scatter(
                    embed[:, 0],
                    embed[:, 1],
                    embed[:, 2] if embed.shape[1] > 2 else np.zeros(len(embed)),
                    alpha=0.7,
                    s=10,
                    c=np.arange(len(embed)),
                    cmap="viridis",
                )

                ax.set_title(f"cluster {idx} (Frames {start_frame}-{end_frame})")
                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
                ax.set_zlabel("UMAP 3")

                if (
                    cluster.get("tda_regions") is not None
                    and "regions" in cluster["tda_regions"]
                ):
                    for j, region in enumerate(cluster["tda_regions"]["regions"]):
                        if np.sum(region) > 5:
                            color = plt.cm.tab10.colors[j % 10]
                            region_pts = np.where(region)[0]
                            if len(region_pts) > 0 and max(region_pts) < len(embed):
                                ax.scatter(
                                    embed[region, 0],
                                    embed[region, 1],
                                    (
                                        embed[region, 2]
                                        if embed.shape[1] > 2
                                        else np.zeros(np.sum(region))
                                    ),
                                    color=color,
                                    alpha=0.9,
                                    s=20,
                                    label=f"Loop {j}",
                                )

                    ax.legend(loc="upper right", fontsize="small")

    for i in range(n_to_show, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_dm_projection(int_rodent, int_fly, n_clusters=4):
    """
    Performs PCA then KMeans clustering
    """
    scaler = StandardScaler()
    int_rodent_scaled = scaler.fit_transform(int_rodent)
    int_fly_scaled = scaler.fit_transform(int_fly)

    pca_rodent = PCA(n_components=2, random_state=42)
    pca_rodent_embedded = pca_rodent.fit_transform(int_rodent_scaled)
    print(
        "Rodent cumulative variance:",
        np.cumsum(pca_rodent.explained_variance_ratio_[:10]),
    )

    pca_fly = PCA(n_components=2, random_state=42)
    pca_fly_embedded = pca_fly.fit_transform(int_fly_scaled)
    print("Fly cumulative variance:", np.cumsum(pca_fly.explained_variance_ratio_[:10]))

    rodent_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(
        pca_rodent_embedded
    )
    fly_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(
        pca_fly_embedded
    )

    explained_var_rodent = pca_rodent.explained_variance_ratio_ * 100
    explained_var_fly = pca_fly.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(
        pca_rodent_embedded[:, 0],
        pca_rodent_embedded[:, 1],
        c=rodent_labels,
        cmap="tab10",
        alpha=0.6,
    )
    axes[0].set_title("Rodent Clusters in PCA Space")
    axes[0].set_xlabel(f"PC1 ({explained_var_rodent[0]:.2f}%)")
    axes[0].set_ylabel(f"PC2 ({explained_var_rodent[1]:.2f}%)")

    axes[1].scatter(
        pca_fly_embedded[:, 0],
        pca_fly_embedded[:, 1],
        c=fly_labels,
        cmap="tab10",
        alpha=0.6,
    )
    axes[1].set_title("Fly Clusters in PCA Space")
    axes[1].set_xlabel(f"PC1 ({explained_var_fly[0]:.2f}%)")
    axes[1].set_ylabel(f"PC2 ({explained_var_fly[1]:.2f}%)")

    plt.tight_layout()
    plt.show()

    combined_data = np.vstack([int_rodent_scaled, int_fly_scaled])
    pca_shared = PCA(n_components=2, random_state=42)
    combined_embedded = pca_shared.fit_transform(combined_data)

    fly_shared = combined_embedded[len(int_rodent_scaled) :]
    rodent_shared = combined_embedded[: len(int_rodent_scaled)]

    explained_var_shared = pca_shared.explained_variance_ratio_ * 100

    plt.figure(figsize=(7, 6))
    plt.scatter(
        fly_shared[:, 0], fly_shared[:, 1], alpha=0.5, label="Fly", color="blue"
    )
    plt.scatter(
        rodent_shared[:, 0],
        rodent_shared[:, 1],
        alpha=0.5,
        label="Rodent",
        color="orange",
    )

    plt.title("Fly vs. Rodent in a Shared PCA Space")
    plt.xlabel(f"PC1 ({explained_var_shared[0]:.2f}%)")
    plt.ylabel(f"PC2 ({explained_var_shared[1]:.2f}%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()