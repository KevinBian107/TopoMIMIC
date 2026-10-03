"""Lens (filter) functions for Mapper scale-sweep.

Each lens takes an (n_points, D) point cloud and returns an (n_points,) real-valued
array. Designed to be plug-and-play with `tdamapper.cover.CubicalCover`.

`knn_density` mirrors the RMS k-NN distance used by DM-G, which is the same lens the
existing `MapperMethod` uses. `pca_1d` and `eccentricity` are orthogonal choices.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from topo_mimic.analysis.mapper.mapper_method import _fiedler_lens, _knn_inverse_density

LENS_REGISTRY: dict[str, Any] = {}

# "pointwise" lenses are a per-point function of the cloud, so they may be computed
# once on the full cloud and indexed down to a subsample. "graph" lenses depend on
# the neighbourhood graph of whatever set they are given, so they MUST be computed
# on exactly the set Mapper will cover: the Fiedler vector of a big cloud restricted
# to a random subsample is not the Fiedler vector of that subsample.
LENS_SCOPE: dict[str, str] = {}


def register_lens(name: str, scope: str = "pointwise"):
    def _decorator(fn):
        LENS_REGISTRY[name] = fn
        LENS_SCOPE[name] = scope
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


@register_lens("fiedler", scope="graph")
def fiedler(
    points: np.ndarray,
    fiedler_k: int = 15,
    rank_normalize: bool = False,
    random_state: int = 0,
    **_ignored,
) -> np.ndarray:
    """Fiedler vector of the k-NN graph Laplacian — a standard Mapper filter.

    `fiedler_k` is deliberately separate from `knn_k`: a k-NN *connectivity* graph
    needs a far larger k than a k-NN *density estimate*. DM-G's k=3 sits well below
    the k ~ log(n) connectivity threshold and would disconnect the graph.
    """
    return _fiedler_lens(
        points, k=fiedler_k, rank_normalize=rank_normalize, random_state=random_state
    )


def compute_lens(name: str, points: np.ndarray, params: dict) -> np.ndarray:
    if name not in LENS_REGISTRY:
        raise ValueError(f"Unknown lens {name!r}; registered: {sorted(LENS_REGISTRY)}")
    values = LENS_REGISTRY[name](points, **(params or {}))
    if np.any(~np.isfinite(values)):
        raise RuntimeError(f"Lens {name!r} produced non-finite values")
    if values.std() == 0:
        raise RuntimeError(f"Lens {name!r} produced constant values — Mapper cover will degenerate")
    return values
