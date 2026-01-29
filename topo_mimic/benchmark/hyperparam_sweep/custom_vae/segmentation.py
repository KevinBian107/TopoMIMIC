"""Segmentation methods for VAE latent space."""

from dataclasses import dataclass

import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class SegmentationResult:
    labels_list: list[np.ndarray]
    n_clusters: int
    method: str


def segment_hmm(
    latent_embeddings: np.ndarray,
    n_clusters: int,
    n_iter: int = 100,
    random_state: int = 0,
) -> SegmentationResult:
    if not HAS_HMMLEARN:
        raise ImportError("hmmlearn not installed")

    n_clips, n_frames, latent_dim = latent_embeddings.shape

    all_data = latent_embeddings.reshape(-1, latent_dim)

    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    all_data_norm = (all_data - mean) / std

    lengths = [n_frames] * n_clips

    model = GaussianHMM(
        n_components=n_clusters,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(all_data_norm, lengths)

    labels_list = []
    for clip_idx in range(n_clips):
        clip_data = latent_embeddings[clip_idx]
        clip_data_norm = (clip_data - mean) / std
        labels = model.predict(clip_data_norm)
        labels_list.append(labels)

    return SegmentationResult(
        labels_list=labels_list,
        n_clusters=n_clusters,
        method="hmm",
    )


def segment_kmeans(
    latent_embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
) -> SegmentationResult:
    if not HAS_SKLEARN:
        raise ImportError("sklearn not installed")

    n_clips, n_frames, latent_dim = latent_embeddings.shape

    all_data = latent_embeddings.reshape(-1, latent_dim)

    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    all_data_norm = (all_data - mean) / std

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    all_labels = model.fit_predict(all_data_norm)

    labels_list = []
    for clip_idx in range(n_clips):
        start = clip_idx * n_frames
        end = (clip_idx + 1) * n_frames
        labels_list.append(all_labels[start:end])

    return SegmentationResult(
        labels_list=labels_list,
        n_clusters=n_clusters,
        method="kmeans",
    )


def segment_latent(
    latent_embeddings: np.ndarray,
    n_clusters: int,
    method: str = "hmm",
    random_state: int = 0,
) -> SegmentationResult:
    if method == "hmm":
        return segment_hmm(latent_embeddings, n_clusters, random_state=random_state)
    elif method == "kmeans":
        return segment_kmeans(latent_embeddings, n_clusters, random_state=random_state)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
