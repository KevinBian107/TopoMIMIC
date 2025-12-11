"""
Clustering methods for benchmarking.
Each method returns cluster labels given input data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Any

from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@dataclass
class ClusterResult:
    labels: np.ndarray
    n_clusters: int
    method_name: str
    params: dict
    embedding: np.ndarray | None = None


class BaseClusteringMethod(ABC):
    """Base class for clustering methods."""

    def __init__(self, name: str, params: dict = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def fit_predict(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Fit and predict cluster labels.

        Args:
            X: Input data (n_samples, n_features)
            n_clusters: Number of clusters

        Returns:
            Cluster labels array
        """
        pass

    def get_embedding(self, X: np.ndarray) -> np.ndarray | None:
        """Return embedding if method produces one, else None."""
        return None


class VanillaKMeans(BaseClusteringMethod):
    """Standard KMeans on raw features."""

    def __init__(self, params: dict = None):
        super().__init__("vanilla_kmeans", params)

    def fit_predict(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        return kmeans.fit_predict(X)


class PCAKMeans(BaseClusteringMethod):
    """PCA dimensionality reduction followed by KMeans."""

    def __init__(self, params: dict = None):
        super().__init__("pca_kmeans", params)
        self.n_components = params.get("n_components", 10) if params else 10
        self._embedding = None

    def fit_predict(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        n_components = min(self.n_components, X.shape[1], X.shape[0])
        pca = PCA(n_components=n_components, random_state=0)
        X_pca = pca.fit_transform(X)
        self._embedding = X_pca

        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        return kmeans.fit_predict(X_pca)

    def get_embedding(self, X: np.ndarray) -> np.ndarray | None:
        return self._embedding


class TSNEKMeans(BaseClusteringMethod):
    """t-SNE dimensionality reduction followed by KMeans."""

    def __init__(self, params: dict = None):
        super().__init__("tsne_kmeans", params)
        self.n_components = params.get("n_components", 3) if params else 3
        self.perplexity = params.get("perplexity", 30) if params else 30
        self._embedding = None

    def fit_predict(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        perplexity = min(self.perplexity, X.shape[0] - 1)
        tsne = TSNE(
            n_components=self.n_components,
            perplexity=perplexity,
            random_state=0,
        )
        X_tsne = tsne.fit_transform(X)
        self._embedding = X_tsne

        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        return kmeans.fit_predict(X_tsne)

    def get_embedding(self, X: np.ndarray) -> np.ndarray | None:
        return self._embedding


class UMAPKMeans(BaseClusteringMethod):
    """UMAP dimensionality reduction followed by KMeans."""

    def __init__(self, params: dict = None):
        super().__init__("umap_kmeans", params)
        self.n_components = params.get("n_components", 3) if params else 3
        self.n_neighbors = params.get("n_neighbors", 15) if params else 15
        self.min_dist = params.get("min_dist", 0.1) if params else 0.1
        self._embedding = None

    def fit_predict(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        import umap

        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=min(self.n_neighbors, X.shape[0] - 1),
            min_dist=self.min_dist,
            random_state=0,
        )
        X_umap = reducer.fit_transform(X)
        self._embedding = X_umap

        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        return kmeans.fit_predict(X_umap)

    def get_embedding(self, X: np.ndarray) -> np.ndarray | None:
        return self._embedding


class DBSCANClustering(BaseClusteringMethod):
    """
    DBSCAN clustering.
    Note: DBSCAN doesn't use n_clusters directly - it discovers clusters.
    We sweep eps values and pick the one closest to target n_clusters.
    Auto-estimates eps range based on data if not provided.
    """

    def __init__(self, params: dict = None):
        super().__init__("dbscan", params)
        self.eps_range = params.get("eps_range", None) if params else None
        self.min_samples = params.get("min_samples", 5) if params else 5
        self._best_eps = None

    def fit_predict(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        from sklearn.neighbors import NearestNeighbors

        if self.eps_range is None:
            nn = NearestNeighbors(n_neighbors=self.min_samples)
            nn.fit(X)
            distances, _ = nn.kneighbors(X)
            k_distances = np.sort(distances[:, -1])
            p10, p50, p90 = np.percentile(k_distances, [10, 50, 90])
            self.eps_range = np.linspace(p10, p90, 10).tolist()

        best_labels = None
        best_diff = float('inf')

        for eps in self.eps_range:
            dbscan = DBSCAN(eps=eps, min_samples=self.min_samples)
            labels = dbscan.fit_predict(X)
            n_found = len(set(labels)) - (1 if -1 in labels else 0)

            diff = abs(n_found - n_clusters)
            if diff < best_diff and n_found > 1:
                best_diff = diff
                best_labels = labels
                self._best_eps = eps

        if best_labels is None:
            best_labels = np.zeros(X.shape[0], dtype=int)

        return best_labels


class SpectralClusteringMethod(BaseClusteringMethod):
    """
    Spectral Clustering using graph Laplacian eigenvectors.
    Most similar to DM approach as it uses graph structure.
    """

    def __init__(self, params: dict = None):
        super().__init__("spectral", params)
        self.n_neighbors = params.get("n_neighbors", 10) if params else 10
        self.affinity = params.get("affinity", "nearest_neighbors") if params else "nearest_neighbors"

    def fit_predict(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        n_neighbors = min(self.n_neighbors, X.shape[0] - 1)

        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity=self.affinity,
            n_neighbors=n_neighbors,
            random_state=0,
            assign_labels='kmeans',
        )
        return spectral.fit_predict(X)


def get_clustering_method(name: str, params: dict = None) -> BaseClusteringMethod:
    """Factory function to get clustering method by name."""
    methods = {
        "vanilla_kmeans": VanillaKMeans,
        "pca_kmeans": PCAKMeans,
        "tsne_kmeans": TSNEKMeans,
        "umap_kmeans": UMAPKMeans,
        "dbscan": DBSCANClustering,
        "spectral": SpectralClusteringMethod,
    }

    if name not in methods:
        raise ValueError(f"Unknown clustering method: {name}. Available: {list(methods.keys())}")

    return methods[name](params)
