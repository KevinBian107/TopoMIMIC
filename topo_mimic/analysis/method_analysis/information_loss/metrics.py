"""Mutual-information metrics block.

Estimates I(T;Y), NMI, AMI, H(T), H(Y), H(Y|T) in bits, plus per-clip
bootstrap CIs and a within-clip shuffle baseline ("permutation floor").

Every estimator uses the plug-in / sklearn implementation. The sample is
discrete-discrete (T and Y are both int labels), so no kernel bandwidth or
k-NN-neighbor sensitivity to tune. Bias from finite samples is mild at our
sample size (~290k labeled frames) and is consistent across methods so
cross-method comparison is fair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    mutual_info_score,
    normalized_mutual_info_score,
)


_NATS_TO_BITS = 1.0 / np.log(2.0)


@dataclass
class MIResult:
    """Bundle of MI metrics with bootstrap uncertainty for one method+seed."""

    method_name: str
    family: str
    feature_space: str | None
    seed: int | None

    n_clips: int
    n_frames: int
    n_primitives: int

    # point estimates in bits
    H_T: float
    H_Y: float
    I_TY: float
    H_Y_given_T: float
    NMI: float
    AMI: float
    efficiency: float        # I(T;Y) / H(T), guarded against /0
    I_TY_shuffled: float     # permutation floor

    bootstrap: dict[str, np.ndarray] = field(default_factory=dict)
    ci_lo: dict[str, float] = field(default_factory=dict)
    ci_hi: dict[str, float] = field(default_factory=dict)


def _entropy_bits(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _mi_bits(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    return float(mutual_info_score(a, b) * _NATS_TO_BITS)


def _all_zero_metrics() -> dict[str, float]:
    return {"H_T": 0.0, "H_Y": 0.0, "I_TY": 0.0,
            "H_Y_given_T": 0.0, "NMI": 0.0, "AMI": 0.0,
            "efficiency": 0.0}


def _point_estimates(T_flat: np.ndarray, Y_flat: np.ndarray) -> dict[str, float]:
    H_T = _entropy_bits(T_flat)
    H_Y = _entropy_bits(Y_flat)
    I_TY = _mi_bits(T_flat, Y_flat)
    H_Y_given_T = max(0.0, H_Y - I_TY)
    NMI = float(normalized_mutual_info_score(Y_flat, T_flat, average_method="geometric"))
    AMI = float(adjusted_mutual_info_score(Y_flat, T_flat))
    eff = I_TY / H_T if H_T > 1e-12 else 0.0
    return {"H_T": H_T, "H_Y": H_Y, "I_TY": I_TY,
            "H_Y_given_T": H_Y_given_T, "NMI": NMI, "AMI": AMI,
            "efficiency": eff}


def _flatten_with_mask(
    T: np.ndarray,
    Y: np.ndarray,
    background_token: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten (n_clips, T_frames) to 1D, restrict to frames where Y is labeled.

    Background frames are kept (T == -1 -> "background" bucket) so sparse
    methods aren't unfairly advantaged by conditioning on "in-skeleton".

    Returns:
        T_flat: int array. Background gets a sentinel id ``int(T.max())+1``.
        Y_flat: int array of labeled-frame Y values.
        labeled_clip_mask_flat: bool array, True where Y >= 0.
    """
    labeled = Y >= 0
    Y_flat = Y[labeled].astype(np.int64)
    T_flat = T[labeled].astype(np.int64)
    if T_flat.size > 0:
        bg_id = int(T_flat.max()) + 1
    else:
        bg_id = 0
    T_flat = np.where(T_flat == background_token, bg_id, T_flat)
    return T_flat, Y_flat, labeled


def _shuffle_T_within_clips(T: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(T)
    for c in range(T.shape[0]):
        out[c] = rng.permutation(T[c])
    return out


def compute_mi(
    T: np.ndarray,
    Y: np.ndarray,
    method_name: str,
    family: str,
    feature_space: str | None,
    seed: int | None,
    n_bootstrap: int = 1000,
    rng_seed: int = 0,
    background_token: int = -1,
) -> MIResult:
    """Compute MI metrics + per-clip-bootstrap CIs + shuffle permutation floor.

    Args:
        T: (n_clips, frames_per_clip) int array; ``background_token`` for
           frames not in any primitive.
        Y: (n_clips, frames_per_clip) int array; ``-1`` for unlabeled clips,
           ``>= 0`` for labeled-clip frames.
        method_name, family, feature_space, seed: bookkeeping only.
        n_bootstrap: number of clip-level bootstrap resamples.
        rng_seed: RNG seed. Derived per-method seeds are recommended in main.

    Returns:
        Fully-populated MIResult.
    """
    if T.shape != Y.shape:
        raise ValueError(f"T shape {T.shape} != Y shape {Y.shape}")

    n_clips, frames_per_clip = T.shape
    rng = np.random.default_rng(rng_seed)

    T_flat, Y_flat, labeled_mask = _flatten_with_mask(T, Y, background_token)

    if T_flat.size == 0:
        zeros = _all_zero_metrics()
        return MIResult(
            method_name=method_name, family=family,
            feature_space=feature_space, seed=seed,
            n_clips=n_clips, n_frames=0, n_primitives=0,
            I_TY_shuffled=0.0,
            **zeros,
        )

    point = _point_estimates(T_flat, Y_flat)

    T_shuffled = _shuffle_T_within_clips(T, rng)
    T_sh_flat, Y_sh_flat, _ = _flatten_with_mask(T_shuffled, Y, background_token)
    I_TY_shuffled = _mi_bits(T_sh_flat, Y_sh_flat)

    bootstrap_keys = ["H_T", "I_TY", "NMI", "AMI", "H_Y_given_T", "efficiency"]
    bootstrap = {k: np.empty(n_bootstrap, dtype=np.float64) for k in bootstrap_keys}

    labeled_clip_ids = np.where(labeled_mask.any(axis=1))[0]
    if labeled_clip_ids.size > 0 and n_bootstrap > 0:
        per_clip_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for c in labeled_clip_ids:
            row_mask = labeled_mask[c]
            if not row_mask.any():
                per_clip_pairs.append((np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
                continue
            T_row = T[c][row_mask].astype(np.int64)
            Y_row = Y[c][row_mask].astype(np.int64)
            per_clip_pairs.append((T_row, Y_row))

        T_global_max = int(max((p[0].max() if p[0].size else -1) for p in per_clip_pairs))
        bg_id = T_global_max + 1

        for b in range(n_bootstrap):
            sample = rng.choice(len(per_clip_pairs), size=len(per_clip_pairs), replace=True)
            T_list = []
            Y_list = []
            for idx in sample:
                Tc, Yc = per_clip_pairs[idx]
                if Tc.size == 0:
                    continue
                T_list.append(Tc)
                Y_list.append(Yc)
            if not T_list:
                for k in bootstrap_keys:
                    bootstrap[k][b] = 0.0
                continue
            T_bflat = np.concatenate(T_list)
            Y_bflat = np.concatenate(Y_list)
            T_bflat = np.where(T_bflat == background_token, bg_id, T_bflat)
            pe = _point_estimates(T_bflat, Y_bflat)
            for k in bootstrap_keys:
                bootstrap[k][b] = pe[k]

    if n_bootstrap > 0:
        ci_lo = {k: float(np.quantile(v, 0.025)) for k, v in bootstrap.items()}
        ci_hi = {k: float(np.quantile(v, 0.975)) for k, v in bootstrap.items()}
    else:
        ci_lo = {k: 0.0 for k in bootstrap}
        ci_hi = {k: 0.0 for k in bootstrap}

    n_primitives = int(np.unique(T[T >= 0]).size)

    return MIResult(
        method_name=method_name, family=family,
        feature_space=feature_space, seed=seed,
        n_clips=int(labeled_clip_ids.size), n_frames=int(T_flat.size),
        n_primitives=n_primitives,
        I_TY_shuffled=I_TY_shuffled,
        bootstrap=bootstrap, ci_lo=ci_lo, ci_hi=ci_hi,
        **point,
    )


def joint_distribution(
    T: np.ndarray,
    Y: np.ndarray,
    background_token: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-(T, Y) co-occurrence count table over labeled frames.

    Returns:
        joint: (n_T_unique, n_Y_unique) int64.
        T_values: 1D int array of unique T labels in row order
            (background, if present, is appended last with id = max(T)+1).
        Y_values: 1D int array of unique Y labels in column order.
    """
    T_flat, Y_flat, _ = _flatten_with_mask(T, Y, background_token)
    if T_flat.size == 0:
        return np.zeros((0, 0), dtype=np.int64), np.array([]), np.array([])

    T_values, T_idx = np.unique(T_flat, return_inverse=True)
    Y_values, Y_idx = np.unique(Y_flat, return_inverse=True)
    joint = np.zeros((T_values.size, Y_values.size), dtype=np.int64)
    np.add.at(joint, (T_idx, Y_idx), 1)
    return joint, T_values, Y_values


# ---------------------------------------------------------------------------
# Built-in sanity checks
# ---------------------------------------------------------------------------


def run_sanity_checks(verbose: bool = True) -> None:
    """Confirm the estimator behaves sensibly on synthetic input.

    1. Independent T, Y -> I(T;Y) ~ 0.
    2. T = Y -> I(T;Y) = H(Y); H(Y|T) = 0.
    3. H(T) <= log2(|unique T|) at machine tolerance.

    Raises AssertionError on failure.
    """
    rng = np.random.default_rng(7)

    Y = rng.integers(0, 5, size=(50, 100))
    T_indep = rng.integers(0, 12, size=Y.shape)
    res = compute_mi(T_indep, Y, "indep", "TEST", None, None,
                     n_bootstrap=10, rng_seed=0, background_token=-99)
    assert res.I_TY < 0.05 * res.H_Y + 0.05, f"independent MI too high: {res.I_TY}"

    T_perfect = Y.copy()
    res2 = compute_mi(T_perfect, Y, "perfect", "TEST", None, None,
                      n_bootstrap=10, rng_seed=0, background_token=-99)
    assert abs(res2.I_TY - res2.H_Y) < 1e-6, f"perfect MI != H_Y: {res2.I_TY} vs {res2.H_Y}"
    assert res2.H_Y_given_T < 1e-6, f"H_Y|T not zero: {res2.H_Y_given_T}"

    Y_uniq = np.unique(T_perfect.reshape(-1))
    H_T = _entropy_bits(T_perfect.reshape(-1))
    assert H_T <= np.log2(Y_uniq.size) + 1e-9, "H_T exceeds log2(|alphabet|)"

    if verbose:
        print("metrics.run_sanity_checks: PASSED")


if __name__ == "__main__":
    run_sanity_checks()
