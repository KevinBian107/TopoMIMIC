"""Lens (filter) functions for Mapper scale-sweep.

Each lens takes an (n_points, D) point cloud and returns an (n_points,) real-valued
array. Designed to be plug-and-play with `tdamapper.cover.CubicalCover`.

`knn_density` mirrors the RMS k-NN distance used by DM-G, which is the same lens the
existing `MapperMethod` uses. `pca_1d` and `eccentricity` are orthogonal choices.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from topo_mimic.benchmark.mapper.mapper_method import _knn_inverse_density

LENS_REGISTRY: dict[str, Any] = {}


def register_lens(name: str):
    def _decorator(fn):
        LENS_REGISTRY[name] = fn
        return fn
    return _decorator


@register_lens("knn_density")
def knn_density(points: np.ndarray, knn_k: int = 3, **_ignored) -> np.ndarray:
    """RMS k-NN distance per point. Reuses the DM-G inverse-density helper."""
    return _knn_inverse_density(points, k=knn_k)


@register_lens("pca_1d")
def pca_1d(points: np.ndarray, **_ignored) -> np.ndarray:
    """First principal component projection after per-dim z-score standardization.

    Standardization matters on qpos/ekp where root translation (meters) and joint
    angles (radians) have wildly different scales — without it PC1 collapses to the
    largest-variance dim and becomes a single-coordinate projection.
    """
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=1))])
    return pipe.fit_transform(points).ravel()


@register_lens("eccentricity")
def eccentricity(points: np.ndarray, knn_k: int = 3, **_ignored) -> np.ndarray:
    """Max distance from each point to its k nearest neighbors — classic Mapper lens."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(knn_k + 1, len(points)))
    nn.fit(points)
    distances, _ = nn.kneighbors(points)
    return distances[:, 1:].max(axis=1)


def compute_lens(name: str, points: np.ndarray, params: dict) -> np.ndarray:
    if name not in LENS_REGISTRY:
        raise ValueError(f"Unknown lens {name!r}; registered: {sorted(LENS_REGISTRY)}")
    values = LENS_REGISTRY[name](points, **(params or {}))
    if np.any(~np.isfinite(values)):
        raise RuntimeError(f"Lens {name!r} produced non-finite values")
    if values.std() == 0:
        raise RuntimeError(f"Lens {name!r} produced constant values — Mapper cover will degenerate")
    return values
