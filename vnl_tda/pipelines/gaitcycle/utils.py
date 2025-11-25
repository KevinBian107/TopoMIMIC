'''Helper function for some gait cycle + intention analysis'''

import numpy as np
import umap
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import hsv_to_rgb


def create_time_windows(intentions, windows_size=31, stride=1):
    """
    Create Taken's delayed embedding / time windows by stakcing vectors
    """
    n_timesteps, n_features = intentions.shape
    n_windows = (n_timesteps - windows_size) // stride + 1
    windows = np.zeros((n_windows, windows_size * n_features))
    center_frames = np.zeros(n_windows, dtype=int)

    for i in range(n_windows):
        start_idx = i * stride
        # extract window and flatten
        window = intentions[start_idx : start_idx + windows_size]
        windows[i] = window.reshape(-1)
        # calculate center frame
        center_frames[i] = start_idx + windows_size // 2

    return windows, center_frames


def create_color_by_phase(phases):
    """
    Convert phases to colors using HSV colormap with robust error handling
    """

    # normalize phases to range [0, 1] for hue
    phases = np.asarray(phases)
    phases = np.mod(phases, 2 * np.pi)  # ensure in range [0, 2π]
    hues = phases / (2 * np.pi)  # convert to range [0, 1]

    hsv_colors = np.zeros((len(hues), 3))
    hsv_colors[:, 0] = hues
    hsv_colors[:, 1] = 0.8
    hsv_colors[:, 2] = 0.8

    try:
        rgb_colors = hsv_to_rgb(hsv_colors)
        # all values are within 0-1 range
        rgb_colors = np.clip(rgb_colors, 0, 1)
        return rgb_colors
    except:
        # fall back to a simple colormap
        print("Error converting HSV to RGB, using fallback colormap")
        return plt.cm.hsv(hues)


def apply_umap_to_intention_clusters(clusters, n_components=3):
    """
    Apply UMAP to intention space clusters
    """
    scaler = StandardScaler()
    clusters_scaled = scaler.fit_transform(clusters)
    n_neighbors = min(20, max(2, len(clusters) // 5))
    umap_reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )

    embedded = umap_reducer.fit_transform(clusters_scaled)
    return embedded


def sample_diverse_windows(windows, n_samples=10000, n_windows=10):
    """
    Randomly sample diverse windows based on their Eucledian PCA clusters.
    """
    
    print(
        f"Sampling {n_samples} diverse clusters from {len(windows)} total clusters..."
    )

    if len(windows) <= n_samples:
        print(f"Only {len(windows)} clusters available, using all of them")
        return windows, np.arange(len(windows))

    print("Applying PCA for clustering then sample")
    pca = PCA(n_components=min(50, windows.shape[1]))
    reduced_windows = pca.fit_transform(windows)

    print(f"Clustering data into {n_windows} groups...")
    kmeans = KMeans(n_clusters=n_windows, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(reduced_windows)

    # sample from each windows proportionally
    sampled_indices = []
    for cluster_idx in range(n_windows):
        cluster_indices = np.where(cluster_labels == cluster_idx)[0]

        # samples for this cluster (proportional to cluster size)
        n_cluster_samples = int(n_samples * len(cluster_indices) / len(windows))
        n_cluster_samples = max(1, n_cluster_samples)

        # sample indices from this cluster
        if n_cluster_samples >= len(cluster_indices):
            # iIf we need more samples than available, sample with replacement
            cluster_sampled = np.random.choice(
                cluster_indices, size=n_cluster_samples, replace=True
            )
        else:
            # sample without replacement
            cluster_sampled = np.random.choice(
                cluster_indices, size=n_cluster_samples, replace=False
            )

        sampled_indices.extend(cluster_sampled)

    if len(sampled_indices) > n_samples:
        # too many samples, randomly subsample
        sampled_indices = np.random.choice(
            sampled_indices, size=n_samples, replace=False
        )
    elif len(sampled_indices) < n_samples:
        # too few samples, add more from random clusters
        remaining = n_samples - len(sampled_indices)
        remaining_indices = list(set(range(len(windows))) - set(sampled_indices))

        if len(remaining_indices) >= remaining:
            # have enough, sample without replacement
            additional = np.random.choice(
                remaining_indices, size=remaining, replace=False
            )
        else:
            # sample with replacement from all indices
            additional = np.random.choice(
                range(len(windows)), size=remaining, replace=True
            )

        sampled_indices.extend(additional)

    sampled_indices = np.array(sampled_indices)
    np.random.shuffle(sampled_indices)

    return windows[sampled_indices], sampled_indices


def cluster_gait_distribution(embedded, gait_metrics, n_clusters=4):
    """
    Analyze how different gait patterns are distributed on the manifold
    """
    min_size = min(len(embedded), len(gait_metrics["phases"]))
    embedded = embedded[:min_size]

    phases = gait_metrics["phases"][:min_size]
    frequencies = gait_metrics["frequencies"][:min_size]
    feet_in_stance = gait_metrics["feet_in_stance"][:min_size]
    duty_factor = gait_metrics["duty_factor"][:min_size]

    features = np.column_stack([frequencies, duty_factor, feet_in_stance])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    n_clusters = min(n_clusters, len(embedded) // 10)
    n_clusters = max(2, n_clusters)  # At least 2 clusters

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)

    return cluster_labels


def visualize_gait_manifold(embedded, gait_metrics):
    """
    Create visualizations for gait manifold
    """
    phases = gait_metrics["phases"]
    frequencies = gait_metrics["frequencies"]
    feet_in_stance = gait_metrics["feet_in_stance"]

    if len(phases) != len(embedded):
        print(
            f"Warning: Number of gait metrics ({len(phases)}) doesn't match embedding points ({len(embedded)})"
        )
        print("Adjusting data to match sizes...")

        min_size = min(len(embedded), len(phases))
        embedded = embedded[:min_size]
        phases = phases[:min_size]
        frequencies = frequencies[:min_size]
        feet_in_stance = feet_in_stance[:min_size]

    fig = plt.figure(figsize=(20, 16))

    # calculate density for better visualization
    try:
        density = gaussian_kde(embedded.T)(embedded.T)
    except:
        print("Error calculating density, using uniform density")
        density = np.ones(len(embedded))

    idx = density.argsort()
    x, y, z = embedded[idx, 0], embedded[idx, 1], embedded[idx, 2]
    phases_sorted = phases[idx]

    frequencies = np.array(frequencies, dtype=float)
    frequencies = np.nan_to_num(frequencies, nan=1.0)  # replace NaNs with 1.0
    frequencies = np.clip(frequencies, 0.1, 10.0)  # reasonable range for frequencies
    frequencies_sorted = frequencies[idx]

    feet_in_stance = np.round(feet_in_stance).astype(int)
    feet_in_stance = np.clip(feet_in_stance, 0, 6)  # limit to 0-6 range for feet
    feet_sorted = feet_in_stance[idx]

    # colored by frequency
    ax1 = fig.add_subplot(221, projection="3d")
    scatter1 = ax1.scatter(
        x,
        y,
        z,
        c=frequencies_sorted,
        cmap="viridis",
        alpha=0.6,
        s=5,
        vmin=min(0.5, frequencies.min()),
        vmax=max(5.0, frequencies.max()),
    )
    ax1.set_title("Manifold Colored by Mean Frequency of Walking", fontsize=14)
    fig.colorbar(scatter1, ax=ax1, label="Frequency (Hz)")

    # colored by phase
    ax2 = fig.add_subplot(222, projection="3d")
    try:
        phase_colors = create_color_by_phase(phases_sorted)
        ax2.scatter(x, y, z, c=phase_colors, alpha=0.6, s=5)
    except:
        print("Error with phase colors, using viridis colormap instead")
        ax2.scatter(
            x,
            y,
            z,
            c=phases_sorted,
            cmap="viridis",
            alpha=0.6,
            s=5,
            vmin=-np.pi,
            vmax=np.pi,
        )

    ax2.set_title("Manifold Colored by Global Phase", fontsize=14)

    # feet in stance
    ax3 = fig.add_subplot(223, projection="3d")
    scatter4 = ax3.scatter(
        x, y, z, c=feet_sorted, cmap="viridis", alpha=0.6, s=5, vmin=0, vmax=6
    )
    ax3.set_title("Manifold Colored by Number of Feet in Stance", fontsize=14)
    fig.colorbar(scatter4, ax=ax3, label="Feet in stance")
    
    # raw space
    ax4 = fig.add_subplot(224, projection="3d")
    ax4.scatter(
        x, y, z, alpha=0.6, s=5, vmin=0, vmax=6
    )
    ax4.set_title("Intention Manifold Using UMAP", fontsize=14)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.view_init(elev=20, azim=-60)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        ax.grid(False)

    plt.tight_layout()
    plt.show()