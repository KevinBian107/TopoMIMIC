"""MMD and KID (Kernel Inception Distance) between cluster and GT samples.

Two sample units are supported. ``compute_mmd_kid`` flattens each (K, D)
trajectory window to one (K*D,)-dim sample, which makes the kernel
frame-synchronous and therefore order-sensitive. ``compute_frame_kid`` treats
every frame as one D-dim sample, comparing the pooled frame distributions,
which is order-invariant. Either way the two samples X and Y are compared via:

  - Polynomial kernel (KID, Bińkowski 2018):
        k(x, y) = ((x · y) / d + 1)^3
    where d = feature dim. No bandwidth to tune; bounded.

  - Gaussian RBF:
        k(x, y) = exp(-||x - y||² / (2 σ²))
    with σ chosen by the median heuristic on the pooled sample X ∪ Y.

Both are reported via the UNBIASED MMD² estimator (diagonals excluded from
Kxx and Kyy). 0 ≡ distributions identical in RKHS; larger values mean
greater distributional distance. The unbiased estimator can go slightly
negative when distributions truly match — a small-sample variance artifact,
not a bug.
"""

from __future__ import annotations

import numpy as np


def _flatten_windows(x: np.ndarray) -> np.ndarray:
    """(N, K, D) or (N, KD) → (N, KD) float32."""
    if x.ndim == 3:
        return x.reshape(x.shape[0], -1).astype(np.float32, copy=False)
    if x.ndim == 2:
        return x.astype(np.float32, copy=False)
    raise ValueError(f"expected (N, K, D) or (N, D), got shape {x.shape}")


def polynomial_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """KID polynomial kernel: ((x·y) / d + 1)^3."""
    d = float(X.shape[1])
    G = X @ Y.T
    return (G / d + 1.0) ** 3


def median_heuristic_sigma(
    pool: np.ndarray,
    max_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """σ = sqrt(median pairwise squared distance / 2) on a subsample."""
    n = pool.shape[0]
    if n > max_samples:
        rng = rng if rng is not None else np.random.default_rng(0)
        idx = rng.choice(n, size=max_samples, replace=False)
        pool = pool[idx]
    sq = np.sum(pool ** 2, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * pool @ pool.T
    D2 = np.maximum(D2, 0.0)
    iu = np.triu_indices_from(D2, k=1)
    med_sq = float(np.median(D2[iu])) if iu[0].size else 1.0
    if med_sq <= 0.0:
        return 1.0
    return float(np.sqrt(med_sq / 2.0))


def rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    sq_x = np.sum(X ** 2, axis=1)
    sq_y = np.sum(Y ** 2, axis=1)
    D2 = sq_x[:, None] + sq_y[None, :] - 2.0 * X @ Y.T
    D2 = np.maximum(D2, 0.0)
    return np.exp(-D2 / (2.0 * sigma * sigma))


def mmd2_unbiased(Kxx: np.ndarray, Kyy: np.ndarray, Kxy: np.ndarray) -> float:
    n = Kxx.shape[0]
    m = Kyy.shape[0]
    if n < 2 or m < 2:
        return float("nan")
    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    sum_xy = Kxy.sum() * 2.0 / (n * m)
    return float(sum_xx + sum_yy - sum_xy)


def compute_mmd_kid(
    cluster_windows: np.ndarray,
    real_windows: np.ndarray,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Compute KID (poly-kernel unbiased MMD²) and RBF-kernel unbiased MMD²
    between two distributions of (K, D) or (K*D,) flattened windows.
    """
    X = _flatten_windows(cluster_windows)
    Y = _flatten_windows(real_windows)
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        return {"kid": float("nan"), "mmd_rbf": float("nan"),
                "n_cluster": int(n), "n_real": int(m), "sigma": float("nan")}

    Kxx = polynomial_kernel(X, X)
    Kyy = polynomial_kernel(Y, Y)
    Kxy = polynomial_kernel(X, Y)
    kid = mmd2_unbiased(Kxx, Kyy, Kxy)

    pool = np.concatenate([X, Y], axis=0)
    sigma = median_heuristic_sigma(pool, rng=rng)
    Kxx_g = rbf_kernel(X, X, sigma)
    Kyy_g = rbf_kernel(Y, Y, sigma)
    Kxy_g = rbf_kernel(X, Y, sigma)
    mmd_rbf = mmd2_unbiased(Kxx_g, Kyy_g, Kxy_g)

    return {
        "kid": kid,
        "mmd_rbf": mmd_rbf,
        "n_cluster": int(n),
        "n_real": int(m),
        "sigma": float(sigma),
    }


def compute_frame_kid(
    cluster_frames: np.ndarray,
    real_frames: np.ndarray,
    rng: np.random.Generator,
    subset_size: int = 1000,
    n_subsets: int = 10,
) -> dict[str, float]:
    """KID and RBF-MMD² between two pooled frame distributions, each (N, D).

    Every frame is one sample, so both estimates are invariant to frame order.
    Following Bińkowski et al. (2018), the unbiased MMD² is averaged over
    ``n_subsets`` random subsets of equal size min(subset_size, N_x, N_y) drawn
    from each side, so the pool sizes of different methods do not enter the score.
    """
    # float64: MMD² is a small difference of large kernel sums, and float32
    # cancellation alone moves it by ~1% at s=1000.
    X = np.asarray(cluster_frames, dtype=np.float64)
    Y = np.asarray(real_frames, dtype=np.float64)
    n, m = X.shape[0], Y.shape[0]
    s = min(subset_size, n, m)
    if s < 2:
        return {"kid": float("nan"), "mmd_rbf": float("nan"),
                "n_cluster": int(n), "n_real": int(m), "sigma": float("nan")}

    sigma = median_heuristic_sigma(np.concatenate([X, Y], axis=0), rng=rng)
    kids, rbfs = [], []
    for _ in range(n_subsets):
        Xs = X[rng.choice(n, size=s, replace=False)]
        Ys = Y[rng.choice(m, size=s, replace=False)]
        kids.append(mmd2_unbiased(
            polynomial_kernel(Xs, Xs), polynomial_kernel(Ys, Ys), polynomial_kernel(Xs, Ys)))
        rbfs.append(mmd2_unbiased(
            rbf_kernel(Xs, Xs, sigma), rbf_kernel(Ys, Ys, sigma), rbf_kernel(Xs, Ys, sigma)))

    return {
        "kid": float(np.mean(kids)),
        "mmd_rbf": float(np.mean(rbfs)),
        "n_cluster": int(n),
        "n_real": int(m),
        "sigma": float(sigma),
    }
