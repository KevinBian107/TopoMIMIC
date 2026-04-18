"""
Kernel Inception Distance (KID) and Frechet Inception Distance (FID) for the
TopoVNL benchmark, ported verbatim from
``track-mjx/moseq_jax/experiments/run_inception_distance.py:370-431``.

The KID kernel is the unbiased polynomial MMD^2:
    k(x, y) = (x . y / d + 1)^degree
with degree=3 by default. Averaged over ``num_subsets`` random subsets of
size ``subset_size``; diagonals zeroed out for the unbiased estimator.

These functions take already-encoded feature vectors ``mu_real`` and
``mu_fake`` of shape ``(n_samples, latent_dim)``. The VAE-based encoding
step lives in ``topo_mimic/benchmark/analysis/vae_per_transition.py``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.linalg


def compute_fid(mu_real: np.ndarray, mu_fake: np.ndarray) -> float:
    """Frechet Inception Distance between two sets of features."""
    mu1 = np.mean(mu_real, axis=0)
    mu2 = np.mean(mu_fake, axis=0)
    sigma1 = np.cov(mu_real, rowvar=False)
    sigma2 = np.cov(mu_fake, rowvar=False)

    diff = mu1 - mu2
    mean_diff_sq = np.dot(diff, diff)

    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = mean_diff_sq + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def compute_kid(
    mu_real: np.ndarray,
    mu_fake: np.ndarray,
    degree: int = 3,
    num_subsets: int = 100,
    subset_size: int | None = None,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Kernel Inception Distance via unbiased polynomial MMD^2.

    Args:
        mu_real: (n_real, latent_dim) encoded GT samples.
        mu_fake: (n_fake, latent_dim) encoded discovered/candidate samples.
        degree: polynomial kernel degree (default 3).
        num_subsets: number of random subsets to average over (default 100).
        subset_size: size of each subset. If None, defaults to
            ``min(n_real, n_fake, 1000)``.
        seed: RNG seed for subset sampling.

    Returns:
        (kid_mean, kid_std): mean and std of the MMD^2 values across subsets.
    """
    n_real = len(mu_real)
    n_fake = len(mu_fake)
    if n_real == 0 or n_fake == 0:
        return float("nan"), float("nan")
    d = mu_real.shape[1]

    if subset_size is None:
        subset_size = min(n_real, n_fake, 1000)

    rng = np.random.default_rng(seed)
    kid_values = []

    for _ in range(num_subsets):
        real_idx = rng.choice(n_real, size=subset_size, replace=(subset_size > n_real))
        fake_idx = rng.choice(n_fake, size=subset_size, replace=(subset_size > n_fake))

        x = mu_real[real_idx]
        y = mu_fake[fake_idx]

        kxx = (x @ x.T / d + 1.0) ** degree
        kyy = (y @ y.T / d + 1.0) ** degree
        kxy = (x @ y.T / d + 1.0) ** degree

        m = subset_size
        np.fill_diagonal(kxx, 0.0)
        np.fill_diagonal(kyy, 0.0)

        mmd2 = (
            np.sum(kxx) / (m * (m - 1))
            + np.sum(kyy) / (m * (m - 1))
            - 2.0 * np.sum(kxy) / (m * m)
        )
        kid_values.append(float(mmd2))

    return float(np.mean(kid_values)), float(np.std(kid_values))
