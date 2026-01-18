"""
Unified method implementations for behavioral clustering and segmentation.

Combines clustering methods (KMeans variants, DBSCAN, Spectral) and
state-space methods (AR-HMM, SLDS) into a single interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import ot
    HAS_OT = True
except ImportError:
    HAS_OT = False

try:
    import keypoint_moseq as kpms
    HAS_KPMS = True
except ImportError:
    HAS_KPMS = False


class MethodType(Enum):
    CLUSTERING = "clustering"
    STATE_SPACE = "state_space"


@dataclass
class MethodConfig:
    name: str
    method_type: MethodType
    enabled: bool = True
    params: dict = field(default_factory=dict)


@dataclass
class ClusteringResult:
    labels: np.ndarray
    n_clusters: int
    method_name: str
    seed: int
    extra: dict = field(default_factory=dict)


@dataclass
class KPMSResult:
    labels: list[np.ndarray]
    n_states: int
    method_name: str
    model: Any = None
    results: dict = None
    reconstructed: np.ndarray = None


class BaseMethod(ABC):
    def __init__(self, name: str, params: dict = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def method_type(self) -> MethodType:
        pass


class VanillaKMeans(BaseMethod):
    def __init__(self, params: dict = None):
        super().__init__("vanilla_kmeans", params)

    @property
    def method_type(self) -> MethodType:
        return MethodType.CLUSTERING

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
        return km.fit_predict(X)


class PCAKMeans(BaseMethod):
    def __init__(self, params: dict = None):
        super().__init__("pca_kmeans", params)
        self.n_components = self.params.get("n_components", 10)

    @property
    def method_type(self) -> MethodType:
        return MethodType.CLUSTERING

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        n_components = min(self.n_components, X.shape[1], X.shape[0])
        pca = PCA(n_components=n_components, random_state=seed)
        X_pca = pca.fit_transform(X)
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
        return km.fit_predict(X_pca)


class TSNEKMeans(BaseMethod):
    def __init__(self, params: dict = None):
        super().__init__("tsne_kmeans", params)
        self.n_components = self.params.get("n_components", 3)
        self.perplexity = self.params.get("perplexity", 30)

    @property
    def method_type(self) -> MethodType:
        return MethodType.CLUSTERING

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        perplexity = min(self.perplexity, X.shape[0] // 4)
        perplexity = max(5, perplexity)
        tsne = TSNE(
            n_components=self.n_components,
            perplexity=perplexity,
            random_state=seed,
        )
        X_tsne = tsne.fit_transform(X)
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
        return km.fit_predict(X_tsne)


class UMAPKMeans(BaseMethod):
    def __init__(self, params: dict = None):
        super().__init__("umap_kmeans", params)
        self.n_components = self.params.get("n_components", 3)
        self.n_neighbors = self.params.get("n_neighbors", 15)
        self.min_dist = self.params.get("min_dist", 0.1)

    @property
    def method_type(self) -> MethodType:
        return MethodType.CLUSTERING

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        if not HAS_UMAP:
            raise ImportError("umap-learn not installed")

        n_neighbors = min(self.n_neighbors, X.shape[0] - 1)
        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=n_neighbors,
            min_dist=self.min_dist,
            random_state=seed,
        )
        X_umap = reducer.fit_transform(X)
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
        return km.fit_predict(X_umap)


class W2KMeans(BaseMethod):
    """
    Wasserstein-2 based clustering using MDS embedding + KMeans.
    Requires precomputed W2 distance matrix passed via params.
    """
    def __init__(self, params: dict = None):
        super().__init__("w2_kmeans", params)
        self.mds_components = self.params.get("mds_components", 3)

    @property
    def method_type(self) -> MethodType:
        return MethodType.CLUSTERING

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        distance_matrix = self.params.get("distance_matrix")
        if distance_matrix is None:
            raise ValueError("W2KMeans requires 'distance_matrix' in params")

        mds = MDS(
            n_components=self.mds_components,
            dissimilarity='precomputed',
            random_state=seed,
            normalized_stress='auto',
        )
        X_mds = mds.fit_transform(distance_matrix)
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
        return km.fit_predict(X_mds)


class W2KMeansBaseline(BaseMethod):
    """
    Baseline: W2 clustering on all frames (not topological cycles).

    Same algorithm as W2KMeans but uses windows from all frames
    instead of topologically-discovered cycles. This serves as a
    baseline to evaluate whether topology-based cycle detection
    improves clustering quality.
    """
    def __init__(self, params: dict = None):
        super().__init__("w2_kmeans_baseline", params)
        self.mds_components = self.params.get("mds_components", 3)

    @property
    def method_type(self) -> MethodType:
        return MethodType.CLUSTERING

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        distance_matrix = self.params.get("distance_matrix")
        if distance_matrix is None:
            raise ValueError("W2KMeansBaseline requires 'distance_matrix' in params")

        mds = MDS(
            n_components=self.mds_components,
            dissimilarity='precomputed',
            random_state=seed,
            normalized_stress='auto',
        )
        X_mds = mds.fit_transform(distance_matrix)
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
        return km.fit_predict(X_mds)


class GWKMeans(BaseMethod):
    """
    Gromov-Wasserstein based clustering using MDS embedding + KMeans.
    Requires precomputed GW distance matrix passed via params.
    """
    def __init__(self, params: dict = None):
        super().__init__("gw_kmeans", params)
        self.mds_components = self.params.get("mds_components", 3)

    @property
    def method_type(self) -> MethodType:
        return MethodType.CLUSTERING

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        distance_matrix = self.params.get("distance_matrix")
        if distance_matrix is None:
            raise ValueError("GWKMeans requires 'distance_matrix' in params")

        mds = MDS(
            n_components=self.mds_components,
            dissimilarity='precomputed',
            random_state=seed,
            normalized_stress='auto',
        )
        X_mds = mds.fit_transform(distance_matrix)
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
        return km.fit_predict(X_mds)


class SpectralMethod(BaseMethod):
    def __init__(self, params: dict = None):
        super().__init__("spectral", params)
        self.affinity = self.params.get("affinity", "nearest_neighbors")
        self.n_neighbors = self.params.get("n_neighbors", 10)

    @property
    def method_type(self) -> MethodType:
        return MethodType.CLUSTERING

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        n_neighbors = min(self.n_neighbors, X.shape[0] - 1)
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity=self.affinity,
            n_neighbors=n_neighbors,
            random_state=seed,
        )
        return sc.fit_predict(X)


class DBSCANMethod(BaseMethod):
    def __init__(self, params: dict = None):
        super().__init__("dbscan", params)
        self.eps = self.params.get("eps", 0.5)
        self.min_samples = self.params.get("min_samples", 5)

    @property
    def method_type(self) -> MethodType:
        return MethodType.CLUSTERING

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        return db.fit_predict(X)


class ARHMMMethod(BaseMethod):
    """AR-HMM from keypoint-moseq."""
    def __init__(self, params: dict = None):
        super().__init__("arhmm", params)
        self.latent_dim = self.params.get("latent_dim", 10)
        self.kappa = self.params.get("kappa", 1e5)
        self.ar_iters = self.params.get("ar_iters", 50)

    @property
    def method_type(self) -> MethodType:
        return MethodType.STATE_SPACE

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        raise NotImplementedError("Use fit() for state-space methods")

    def fit(
        self,
        data: np.ndarray,
        n_states: int,
        project_dir: str,
        seed: int = 0,
    ) -> KPMSResult:
        if not HAS_KPMS:
            raise ImportError("keypoint-moseq not installed")

        import os
        os.makedirs(project_dir, exist_ok=True)

        n_clips, n_frames, n_dim = data.shape
        coordinates = {}
        confidences = {}

        for i in range(n_clips):
            coords = data[i][:, :, np.newaxis].astype(np.float64)
            coordinates[f"recording_{i:04d}"] = coords
            confidences[f"recording_{i:04d}"] = np.ones((n_frames, n_dim), dtype=np.float64)

        kpms.setup_project(project_dir, deeplabcut_config=None, overwrite=True)
        config = kpms.load_config(project_dir)
        config['latent_dim'] = self.latent_dim
        config['kappa'] = self.kappa
        kpms.save_config(project_dir, config)

        int_data, int_metadata = kpms.format_data(coordinates, confidences, **config)

        model_name = f"arhmm_states{n_states}"
        model = kpms.init_model(
            int_data,
            int_metadata,
            num_states=n_states,
            **config,
        )
        model = kpms.fit_model(
            model,
            int_data,
            int_metadata,
            ar_only=True,
            num_iters=self.ar_iters,
            **config,
        )

        kpms.save_checkpoint(model, int_data, int_metadata, project_dir, model_name)
        results = kpms.extract_results(model, int_metadata, project_dir, model_name)

        rec_names = sorted(results.keys())
        labels_list = [results[rn]['syllable'] for rn in rec_names]

        return KPMSResult(
            labels=labels_list,
            n_states=n_states,
            method_name=self.name,
            model=model,
            results=results,
        )


class SLDSMethod(BaseMethod):
    """Switching Linear Dynamical System from keypoint-moseq."""
    def __init__(self, params: dict = None):
        super().__init__("slds", params)
        self.latent_dim = self.params.get("latent_dim", 10)
        self.kappa = self.params.get("kappa", 1e5)
        self.ar_iters = self.params.get("ar_iters", 50)
        self.full_iters = self.params.get("full_iters", 300)

    @property
    def method_type(self) -> MethodType:
        return MethodType.STATE_SPACE

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        raise NotImplementedError("Use fit() for state-space methods")

    def fit(
        self,
        data: np.ndarray,
        n_states: int,
        project_dir: str,
        seed: int = 0,
    ) -> KPMSResult:
        if not HAS_KPMS:
            raise ImportError("keypoint-moseq not installed")

        import os
        os.makedirs(project_dir, exist_ok=True)

        n_clips, n_frames, n_dim = data.shape
        coordinates = {}
        confidences = {}

        for i in range(n_clips):
            coords = data[i][:, :, np.newaxis].astype(np.float64)
            coordinates[f"recording_{i:04d}"] = coords
            confidences[f"recording_{i:04d}"] = np.ones((n_frames, n_dim), dtype=np.float64)

        kpms.setup_project(project_dir, deeplabcut_config=None, overwrite=True)
        config = kpms.load_config(project_dir)
        config['latent_dim'] = self.latent_dim
        config['kappa'] = self.kappa
        kpms.save_config(project_dir, config)

        int_data, int_metadata = kpms.format_data(coordinates, confidences, **config)

        model_name = f"slds_states{n_states}"
        model = kpms.init_model(
            int_data,
            int_metadata,
            num_states=n_states,
            **config,
        )

        model = kpms.fit_model(
            model,
            int_data,
            int_metadata,
            ar_only=True,
            num_iters=self.ar_iters,
            **config,
        )

        model = kpms.fit_model(
            model,
            int_data,
            int_metadata,
            ar_only=False,
            num_iters=self.full_iters,
            **config,
        )

        kpms.save_checkpoint(model, int_data, int_metadata, project_dir, model_name)
        results = kpms.extract_results(model, int_metadata, project_dir, model_name)

        rec_names = sorted(results.keys())
        labels_list = [results[rn]['syllable'] for rn in rec_names]

        return KPMSResult(
            labels=labels_list,
            n_states=n_states,
            method_name=self.name,
            model=model,
            results=results,
        )


METHOD_REGISTRY: dict[str, type[BaseMethod]] = {
    "vanilla_kmeans": VanillaKMeans,
    "pca_kmeans": PCAKMeans,
    "tsne_kmeans": TSNEKMeans,
    "umap_kmeans": UMAPKMeans,
    "w2_kmeans": W2KMeans,
    "w2_kmeans_baseline": W2KMeansBaseline,
    "gw_kmeans": GWKMeans,
    "spectral": SpectralMethod,
    "dbscan": DBSCANMethod,
    "arhmm": ARHMMMethod,
    "slds": SLDSMethod,
}


def get_method(name: str, params: dict = None) -> BaseMethod:
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[name](params)


def get_clustering_method(name: str, params: dict = None, seed: int = 0) -> BaseMethod:
    method = get_method(name, params)
    if method.method_type != MethodType.CLUSTERING:
        raise ValueError(f"{name} is not a clustering method")
    return method


def get_kpms_method(name: str, params: dict = None) -> BaseMethod:
    method = get_method(name, params)
    if method.method_type != MethodType.STATE_SPACE:
        raise ValueError(f"{name} is not a state-space method")
    return method


def list_available_methods() -> dict[str, list[str]]:
    clustering = []
    state_space = []

    for name, cls in METHOD_REGISTRY.items():
        instance = cls()
        if instance.method_type == MethodType.CLUSTERING:
            clustering.append(name)
        else:
            state_space.append(name)

    return {
        "clustering": clustering,
        "state_space": state_space,
    }
