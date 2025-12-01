"""Functions for pre-processing"""

from typing import Callable, Any
import os, functools, multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def estimate_delay_embedding_params(
    rodent_rollout_dir: str,
    get_aggregate_data_fn: Callable,
    num_clips: int,
    intention_key: str = "/intention",
    clip_indices: list[int] | None = None,
    max_tau: int = 200,
    bins: int = 64,
    m_max: int = 15,
    R_tol: float = 15.0,
    A_tol: float = 2.0,
    fnn_thr: float = 0.05,
    theiler: str | int | None = "auto",
    max_points: int = 20000,
    rng_seed: int = 0,
    processes: int | None = None,
    plot: bool = True,
    figsize: tuple[float, float] = (12, 4.5),
) -> tuple[int, int, dict]:
    """
    Estimate optimal delay embedding parameters (tau, m) via AMI and FNN.

    Args:
        rodent_rollout_dir: Directory containing clip HDF5 files
        get_aggregate_data_fn: Function to load clip data
        num_clips: Number of clips to process
        intention_key: HDF5 key for intention data
        clip_indices: Specific clip indices to use (None for 0..num_clips-1)
        max_tau: Maximum delay to consider for AMI
        bins: Number of bins for entropy estimation
        m_max: Maximum embedding dimension for FNN
        R_tol: Distance ratio tolerance for FNN
        A_tol: Absolute distance tolerance for FNN
        fnn_thr: FNN fraction threshold for m selection
        theiler: Theiler window for FNN ("auto", int, or None)
        max_points: Maximum points for FNN computation
        rng_seed: Random seed for reproducibility
        processes: Number of parallel processes (None for cpu_count)
        plot: Whether to display diagnostic plots
        figsize: Figure size for plots

    Returns:
        Tuple of (tau_star, m_star, diagnostics_dict)
    """

    # ----------------- helpers (scoped) -----------------
    def pc1(x):
        x = np.asarray(x, float)
        x = x[~np.isnan(x).any(axis=1)] if x.ndim == 2 else x
        if x.ndim == 1:
            return x.astype(float)
        xc = x - x.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(xc, full_matrices=False)
        return (xc @ Vt.T[:, 0]).astype(float)

    def delay_embed(x, m, tau):
        x = np.asarray(x, float)
        N = len(x); T = N - (m - 1) * tau
        if T <= 0:
            raise ValueError("Time series too short for m,tau.")
        s = x.strides[0]
        return np.lib.stride_tricks.as_strided(
            x, shape=(T, m), strides=(s, tau * s)
        ).copy()

    def entropy_from_hist(p, eps=1e-12):
        p = p.astype(float)
        Z = max(p.sum(), eps)
        p = p / Z
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    def average_mutual_information_1d(x, max_tau=200, bins=64):
        x = np.asarray(x, float)
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(xmin) or xmin == xmax:
            return None, None
        edges = np.linspace(xmin, xmax, bins + 1)
        hx, _ = np.histogram(x, bins=edges)
        Hx = entropy_from_hist(hx)
        I = np.zeros(max_tau, float)
        for k in range(1, max_tau + 1):
            x0 = x[:-k]; x1 = x[k:]
            hxy, _, _ = np.histogram2d(x0, x1, bins=[edges, edges])
            hy, _ = np.histogram(x1, bins=edges)
            Hxy = entropy_from_hist(hxy.ravel())
            Hy  = entropy_from_hist(hy)
            I[k - 1] = Hx + Hy - Hxy
        taus = np.arange(1, max_tau + 1)
        return taus, I

    def first_local_minimum_idx(y):
        y = np.asarray(y)
        for i in range(1, len(y) - 1):
            if y[i] < y[i - 1] and y[i] < y[i + 1]:
                return i
        return None

    def fnn_fraction_1d(x, tau, m_max=12, R_tol=15.0, A_tol=2.0,
                        theiler=None, max_points=20000, rng_seed=0):
        rs = np.random.RandomState(rng_seed)
        x = np.asarray(x, float)
        sig = np.std(x)
        if not np.isfinite(sig) or sig == 0:
            return None, None

        ms = np.arange(1, m_max + 1)
        fracs = []
        for m in ms:
            try:
                Xm = delay_embed(x, m, tau)    # (T, m)
                Xm1 = delay_embed(x, m + 1, tau)
            except ValueError:
                fracs.append(np.nan); continue

            Tm = Xm.shape[0]
            idx_all = np.arange(Tm)
            if Tm > max_points:
                idx = np.sort(rs.choice(Tm, size=max_points, replace=False))
                Xm_use = Xm[idx]
            else:
                idx = idx_all
                Xm_use = Xm

            # pairwise distances (brute force)
            d2 = np.sum((Xm_use[:, None, :] - Xm_use[None, :, :])**2, axis=2)
            np.fill_diagonal(d2, np.inf)

            # theiler window: mask neighbors within ±theiler in original time
            theiler_ = 0 if theiler is None else theiler
            if theiler_ > 0:
                I = idx[:, None]; J = idx[None, :]
                mask = np.abs(I - J) <= theiler_
                d2[mask] = np.inf

            nn = np.argmin(d2, axis=1)
            dist_m = np.sqrt(d2[np.arange(len(nn)), nn])
            valid0 = np.isfinite(dist_m) & (dist_m > 0)

            # need indices < T_{m+1}
            Tm1 = Xm1.shape[0]
            base = idx[valid0]
            nnix = nn[valid0]
            valid1 = (base < Tm1) & (nnix < Tm1)
            base = base[valid1]; nnix = nnix[valid1]
            dist_m = dist_m[valid0][valid1]
            if len(base) == 0:
                fracs.append(np.nan); continue

            x_new_diff = np.abs(x[base + m * tau] - x[nnix + m * tau])
            Rm = x_new_diff / np.maximum(dist_m, 1e-12)
            crit1 = Rm > R_tol
            crit2 = (x_new_diff / sig) > A_tol
            fnn = np.logical_or(crit1, crit2)
            fracs.append(np.mean(fnn) if len(fnn) else np.nan)

        return ms, np.asarray(fracs, float)

    # ----------------- data load -----------------
    if clip_indices is None:
        clip_indices = range(num_clips)

    work = functools.partial(get_aggregate_data_fn,
                             intention_key, [],
                             path=rodent_rollout_dir)

    if processes is None:
        processes = mp.cpu_count()

    with mp.Pool(processes=processes) as pool:
        clips = list(tqdm(pool.imap(work, clip_indices),
                          total=len(list(clip_indices)),
                          desc="Loading clips"))

    # ----------------- AMI over clips -----------------
    taus_list, I_list, tau_star_each = [], [], []

    for clip in tqdm(clips, desc="AMI per clip"):
        x = pc1(clip)
        if len(x) < 3:
            continue
        taus, I = average_mutual_information_1d(x, max_tau=max_tau, bins=bins)
        if taus is None:
            continue
        taus_list.append(taus)
        I_list.append(I)
        idx = first_local_minimum_idx(I)
        tau_star_each.append(taus[idx] if idx is not None else np.nan)

    tau_star_each = np.array(tau_star_each, float)
    tau_star = int(np.nanmedian(tau_star_each)) if np.any(np.isfinite(tau_star_each)) else 10

    # Aggregate AMI stats
    I_mat = np.vstack([I for I in I_list]) if len(I_list) > 0 else np.empty((0, max_tau))
    I_mean = np.nanmean(I_mat, axis=0) if I_mat.size else np.full(max_tau, np.nan)
    I_se   = (np.nanstd(I_mat, axis=0) / np.sqrt(I_mat.shape[0])) if I_mat.size else np.full(max_tau, np.nan)

    # ----------------- FNN at τ* over clips -----------------
    if theiler == "auto":
        theiler_val = tau_star
    elif isinstance(theiler, int):
        theiler_val = theiler
    else:
        theiler_val = 0

    fnn_curves = []
    ms = np.arange(1, m_max + 1)

    for clip in tqdm(clips, desc=f"FNN per clip (τ={tau_star})"):
        x = pc1(clip)
        ms_, fnn = fnn_fraction_1d(x, tau_star, m_max=m_max, R_tol=R_tol, A_tol=A_tol,
                                   theiler=theiler_val, max_points=max_points, rng_seed=rng_seed)
        if fnn is None:
            continue
        fnn_curves.append(fnn)

    fnn_mat = np.vstack(fnn_curves) if len(fnn_curves) > 0 else np.empty((0, m_max))
    fnn_mean = np.nanmean(fnn_mat, axis=0) if fnn_mat.size else np.full(m_max, np.nan)
    fnn_se   = (np.nanstd(fnn_mat, axis=0) / np.sqrt(fnn_mat.shape[0])) if fnn_mat.size else np.full(m_max, np.nan)

    # m* selection: first m with mean FNN <= threshold; else argmin
    m_star = None
    for k, v in enumerate(fnn_mean, start=1):
        if np.isfinite(v) and v <= fnn_thr:
            m_star = k
            break
    if m_star is None:
        m_star = int(np.nanargmin(fnn_mean) + 1) if np.any(np.isfinite(fnn_mean)) else 3

    # ----------------- compact plotting (AMI + FNN side-by-side) -----------------
    if plot:
        fig = plt.figure(figsize=figsize)
        # two axes side-by-side
        gs = fig.add_gridspec(1, 2, wspace=0.25)

        # AMI
        ax1 = fig.add_subplot(gs[0, 0])
        x_taus = taus_list[0] if len(taus_list) > 0 else np.arange(1, max_tau + 1)
        ax1.plot(x_taus, I_mean, label="mean AMI")
        if np.all(np.isfinite(I_se)):
            ax1.fill_between(x_taus, I_mean - I_se, I_mean + I_se, alpha=0.2, linewidth=0)
        ax1.axvline(tau_star, ls="--", label=f"τ* = {tau_star}")
        ax1.set_xlabel("Delay τ"); ax1.set_ylabel("AMI")
        ax1.set_title("Average Mutual Information")
        ax1.legend()

        # FNN
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(ms, fnn_mean, marker='o', label="mean FNN")
        if np.all(np.isfinite(fnn_se)):
            ax2.fill_between(ms, fnn_mean - fnn_se, fnn_mean + fnn_se, alpha=0.2, linewidth=0)
        ax2.axhline(fnn_thr, ls=":", label=f"threshold = {fnn_thr}")
        ax2.axvline(m_star, ls="--", label=f"m* = {m_star}")
        ax2.set_xlabel("Embedding dimension m"); ax2.set_ylabel("Fraction of false neighbors")
        ax2.set_title(f"FNN at τ = {tau_star} (Theiler={theiler_val})")
        ax2.legend()

        fig.suptitle("Delay-Embedding Parameter Selection (AMI → τ*, FNN → m*)")
        plt.tight_layout()
        plt.show()

    diagnostics = dict(
        taus_list=taus_list,
        I_list=I_list,
        tau_star_each=tau_star_each,
        I_mean=I_mean,
        I_se=I_se,
        fnn_curves=fnn_curves,
        fnn_mean=fnn_mean,
        fnn_se=fnn_se,
        ms=ms,
        theiler_used=theiler_val,
    )
    return tau_star, m_star, diagnostics


def takens_points(
    X: np.ndarray,
    m: int,
    tau: int,
) -> np.ndarray:
    """
    Construct Takens' delay embedding from time series.

    Args:
        X: Time series array of shape (n_frames, D)
        m: Embedding dimension
        tau: Time delay

    Returns:
        Delay embedded array of shape (n_valid_points, m * D)
    """
    n_frames, D = X.shape
    start = (m - 1) * tau
    idxs = np.arange(start, n_frames)
    pts = np.stack([X[[j - a * tau for a in range(m)]].reshape(-1) for j in idxs], axis=0)
    return pts