'''General helper functions used in TDA pipelines'''

import numpy as np
import ripser
import matplotlib.pyplot as plt
import gudhi as gd

from persim import plot_diagrams
from math import ceil
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import pearsonr


def subsample_for_tda(embedded, max_points):
    """
    Subsampling the embedding space in case of big dataset
    """

    if len(embedded) <= max_points:
        return embedded

    indices = np.random.choice(len(embedded), max_points, replace=False)
    return embedded[indices]


def landmark_subsample(points, n_landmarks=300):
    """
    Subsampling the embedding space using landmarks, used for RipsComplex
    """

    if len(points) <= n_landmarks:
        return points
    indices = np.random.choice(len(points), n_landmarks, replace=False)
    return points[indices]


def visualize_persistence_diagrams(diagrams, title="Persistence Diagrams"):
    plt.figure(figsize=(12, 8))
    try:
        plot_diagrams(diagrams["diagrams"], show=False)
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        plt.text(0.5, 0.5, "Error plotting diagrams", ha="center", va="center")
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.show()


def manual_plot_barcode(diagram, ax, title=""):
    """
    Plot a TDA persistent barcode diagram from birth-death pairs using matplotlib.
    """

    if len(diagram) == 0:
        ax.text(0.5, 0.5, "No features", ha="center", va="center")
        ax.set_title(title)
        return

    for i, (birth, death) in enumerate(diagram):
        ax.hlines(y=i, xmin=birth, xmax=death, linewidth=2)
        ax.plot([birth, death], [i, i], "o", markersize=4)

    ax.set_ylim(-1, len(diagram))
    ax.set_xlim(np.min(diagram[:, 0]) - 0.1, np.max(diagram[:, 1]) + 0.1)
    ax.set_xlabel("Persistence (Birth → Death)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)


def visualize_h1_barcodes(cluster_results):
    """
    Show barcode plots of H1 persistence diagrams for clusters that contain non-empty H1 diagrams.
    Only clusters with H1 features will be shown.
    """
    filtered_results = [
        (i, r)
        for i, r in enumerate(cluster_results)
        if "H1" in r["diagrams"] and len(r["diagrams"]["H1"]) > 0
    ]

    n_clusters = len(filtered_results)
    if n_clusters == 0:
        print("No clusters with H1 features to visualize.")
        return

    n_cols = min(3, n_clusters)
    n_rows = ceil(n_clusters / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, (idx, result) in enumerate(filtered_results):
        h1 = np.array(result["diagrams"]["H1"])
        ax = axes[i]
        manual_plot_barcode(h1, ax, title=f"H1 Barcode: cluster {idx}")

    for j in range(n_clusters, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_h2_barcodes(cluster_results):
    """
    Show barcode plots of H2 persistence diagrams for clusters that contain non-empty H2 diagrams.
    Only clusters with H2 features will be shown.
    """
    filtered_results = [
        (i, r)
        for i, r in enumerate(cluster_results)
        if "H2" in r["diagrams"] and len(r["diagrams"]["H2"]) > 0
    ]

    n_clusters = len(filtered_results)
    if n_clusters == 0:
        print("No clusters with H2 features to visualize.")
        return

    n_cols = min(3, n_clusters)
    n_rows = ceil(n_clusters / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, (idx, result) in enumerate(filtered_results):
        h2 = np.array(result["diagrams"]["H2"])
        ax = axes[i]
        manual_plot_barcode(h2, ax, title=f"H2 Barcode: cluster {idx}")

    for j in range(n_clusters, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def compute_persistence_diagrams(embedded, max_dim=2):
    """
    Basic persistent calling function
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


def extract_persistent_features(diagrams, threshold=0.1):
    """
    Extract persistent feartures given a diagram and certain threshold
    """
    H1 = diagrams["H1"]
    H1_persistence = diagrams["H1_persistence"]

    persistent_H1_idx = (
        np.where(H1_persistence > threshold)[0]
        if len(H1_persistence) > 0
        else np.array([])
    )
    persistent_H1 = (
        H1[persistent_H1_idx] if len(persistent_H1_idx) > 0 else np.array([])
    )

    H2 = diagrams["H2"]
    persistent_H2 = None

    if (
        H2 is not None
        and isinstance(H2, np.ndarray)
        and H2.ndim == 2
        and H2.shape[1] == 2
    ):
        H2_persistence = H2[:, 1] - H2[:, 0]
        persistent_H2_idx = np.where(H2_persistence > threshold)[0]
        persistent_H2 = (
            H2[persistent_H2_idx] if len(persistent_H2_idx) > 0 else np.empty((0, 2))
        )

    return {"H1": persistent_H1, "H2": persistent_H2, "H1_idx": persistent_H1_idx}


def compute_landscape_features(diagram, resolution=20, num_landscapes=5):
    """
    Compute persistence landscape features manually.
    """

    if len(diagram) == 0:
        return np.zeros((1, resolution))

    persistence_pairs = diagram

    grid = np.linspace(0, 1, resolution)

    landscapes = np.zeros((num_landscapes, resolution))

    for i, x in enumerate(grid):
        # compute all function values at this grid point
        values = []
        for birth, death in persistence_pairs:
            if birth <= x <= death:
                values.append(min(x - birth, death - x))
            else:
                values.append(0)

        values.sort(reverse=True)

        # assign to landscapes
        for k in range(min(num_landscapes, len(values))):
            landscapes[k, i] = values[k]

    landscape_features = landscapes.reshape(1, -1)

    if landscape_features.shape[1] != resolution:
        pad_size = resolution - (landscape_features.shape[1] % resolution)
        landscape_features = np.pad(
            landscape_features, ((0, 0), (0, pad_size)), "constant"
        )
        landscape_features = landscape_features[:, :resolution]

    return landscape_features


def compute_silhouette_features(diagram, resolution=20, weight=lambda x: 1):
    """
    Compute persistence silhouette features manually.
    """

    if len(diagram) == 0:
        return np.zeros((1, resolution))

    persistence_pairs = diagram
    persistences = np.array([death - birth for birth, death in persistence_pairs])

    grid = np.linspace(0, 1, resolution)

    silhouette = np.zeros(resolution)

    for i, x in enumerate(grid):
        # compute the silhouette value at this grid point
        val = 0
        total_weight = 0

        for j, (birth, death) in enumerate(persistence_pairs):
            if birth <= x <= death:
                w = weight(persistences[j])
                val += w * (min(x - birth, death - x) / persistences[j])
                total_weight += w

        if total_weight > 0:
            silhouette[i] = val / total_weight

    return silhouette.reshape(1, -1)


def compute_persistence_image_features(diagram, resolution=(10, 10), spread=0.1):
    """
    Compute persistence image features manually.
    """

    if len(diagram) == 0:
        return np.zeros((1, resolution[0] * resolution[1]))

    persistence_pairs = diagram

    # convert to persistence-birth coordinates
    pers_birth = np.array(
        [(death - birth, birth) for birth, death in persistence_pairs]
    )

    x_grid = np.linspace(0, 1, resolution[0])
    y_grid = np.linspace(0, 1, resolution[1])

    image = np.zeros((resolution[0], resolution[1]))

    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            val = 0

            for pers, birth in pers_birth:
                # gaussian kernel
                val += pers * np.exp(
                    -((x - pers) ** 2 + (y - birth) ** 2) / (2 * spread**2)
                )

            image[i, j] = val

    return image.reshape(1, -1)


def compute_topological_features(
    embedded, max_dim=1, n_landmarks=20, max_edge_length=0.5
):
    embedded_landmarks = landmark_subsample(embedded, n_landmarks=n_landmarks)

    try:
        rips_complex = gd.RipsComplex(
            points=embedded_landmarks, max_edge_length=max_edge_length
        )
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim + 1)
        simplex_tree.compute_persistence()

        persistence = simplex_tree.persistence_intervals_in_dimension
        diagram = [persistence(i) for i in range(max_dim + 1)]

        # implementation of landscape computation
        if len(diagram) > 1 and len(diagram[1]) > 0:
            h1_diagram = diagram[1]
            resolution = 20
            landscape_features = compute_landscape_features(
                h1_diagram, resolution=resolution
            )

            silhouette_features = compute_silhouette_features(
                h1_diagram, resolution=resolution
            )

            pi_features = compute_persistence_image_features(
                h1_diagram, resolution=(10, 10)
            )
        else:
            landscape_features = np.zeros((1, 20))
            silhouette_features = np.zeros((1, 20))
            pi_features = np.zeros((1, 100))

        return {
            "diagram": diagram,
            "landscape": landscape_features,
            "silhouette": silhouette_features,
            "persistence_image": pi_features,
        }
    except Exception as e:
        return {
            "diagram": [np.array([])],
            "landscape": np.zeros((1, 20)),
            "silhouette": np.zeros((1, 20)),
            "persistence_image": np.zeros((1, 100)),
        }


def identify_topological_regions(embedded,persistent_features, gait_metrics):
    """
    Identify potential topological regions.
    """

    phases = gait_metrics.get("phases", np.array([]))
    frequencies = gait_metrics.get("frequencies", np.array([]))
    duty_factor = gait_metrics.get("duty_factor", np.array([]))

    persistent_H1 = persistent_features["H1"]

    if len(persistent_H1) == 0 or len(embedded) == 0:
        return {"regions": [], "correlations": {}}

    regions = []
    correlations = {}

    for i, feature in enumerate(persistent_H1):
        birth, death = feature

        midpoint = (birth + death) / 2
        distances = np.abs(np.linalg.norm(embedded, axis=1) - midpoint)

        tolerance = (death - birth) * 0.2
        near_loop = distances < tolerance

        regions.append(near_loop)

        if np.sum(near_loop) > 5:
            if len(phases) > 0 and len(phases) >= np.sum(near_loop):
                phase_corr, p_phase = pearsonr(phases[near_loop], distances[near_loop])
            else:
                phase_corr, p_phase = 0, 1

            if len(frequencies) > 0 and len(frequencies) >= np.sum(near_loop):
                freq_corr, p_freq = pearsonr(
                    frequencies[near_loop], distances[near_loop]
                )
            else:
                freq_corr, p_freq = 0, 1

            if len(duty_factor) > 0 and len(duty_factor) >= np.sum(near_loop):
                duty_corr, p_duty = pearsonr(
                    duty_factor[near_loop], distances[near_loop]
                )
            else:
                duty_corr, p_duty = 0, 1

            correlations[f"loop_{i}"] = {
                "phase": (phase_corr, p_phase),
                "frequency": (freq_corr, p_freq),
                "duty_factor": (duty_corr, p_duty),
                "size": np.sum(near_loop),
                "birth": birth,
                "death": death,
                "persistence": death - birth,
            }

    return {"regions": regions, "correlations": correlations}


def cluster_based_on_topology(embedded, tda_features):
    """
    Conduct clustering for UMAP using topological features calculated previously
    """

    if len(embedded) < 2:
        return np.zeros(len(embedded), dtype=int)

    landscape_features = tda_features["landscape"]

    if landscape_features.shape[0] == 1:
        point_features = np.tile(landscape_features[0], (len(embedded), 1))
    else:
        point_features = landscape_features

    if point_features.shape[0] != embedded.shape[0]:
        point_features = np.tile(point_features[0], (embedded.shape[0], 1))

    combined_features = np.hstack(
        [
            StandardScaler().fit_transform(embedded),
            StandardScaler().fit_transform(point_features),
        ]
    )

    n_clusters = min(5, len(embedded) // 100)
    n_clusters = max(2, n_clusters)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(combined_features)

    return labels