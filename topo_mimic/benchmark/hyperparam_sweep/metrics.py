"""Unsupervised evaluation metrics for KPMS hyperparameter sweep."""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import keypoint_moseq as kpms
    from jax_moseq.models.keypoint_slds import estimate_coordinates
    HAS_KPMS = True
except ImportError:
    HAS_KPMS = False

try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class EMLResult:
    scores: np.ndarray
    std_errs: np.ndarray
    mean_score: float
    mean_std_err: float


def compute_eml(
    project_dir: str,
    model_names: list[str],
) -> EMLResult:
    if not HAS_KPMS:
        raise ImportError("keypoint_moseq not installed")

    if len(model_names) < 2:
        print(f"Warning: EML requires at least 2 models, got {len(model_names)}. Skipping.")
        return EMLResult(
            scores=np.array([np.nan]),
            std_errs=np.array([np.nan]),
            mean_score=np.nan,
            mean_std_err=np.nan,
        )

    print(f"Computing EML for {len(model_names)} models in {project_dir}")
    scores, std_errs = kpms.expected_marginal_likelihoods(project_dir, model_names)
    print(f"EML scores: {scores}, std_errs: {std_errs}")
    return EMLResult(
        scores=np.array(scores),
        std_errs=np.array(std_errs),
        mean_score=float(np.nanmean(scores)),
        mean_std_err=float(np.nanmean(std_errs)),
    )


def _stack_data_array(data_dict: dict, key: str) -> np.ndarray:
    """Stack data from dict format to array format."""
    val = data_dict[key]
    if isinstance(val, dict):
        keys = sorted(val.keys())
        return np.stack([np.array(val[k]) for k in keys], axis=0)
    else:
        return np.array(val)


def compute_reconstruction_mse(
    model: dict,
    data: dict,
    metadata: tuple,
) -> float:
    if not HAS_KPMS:
        raise ImportError("keypoint_moseq not installed")

    Y = _stack_data_array(data, "Y")
    Y_original = Y.copy()
    mask = _stack_data_array(data, "mask") if "mask" in data else np.ones(Y.shape[:2], dtype=bool)
    conf = _stack_data_array(data, "conf") if "conf" in data else np.ones(Y.shape[:3])
    Y_reconstructed = estimate_coordinates(model, Y, mask, conf)
    Y_reconstructed = np.array(Y_reconstructed)

    mse = np.mean((Y_original - Y_reconstructed) ** 2)
    return float(mse)


@dataclass
class DurationStats:
    mean: float
    std: float
    median: float
    min: int
    max: int
    total_segments: int


def compute_duration_stats(results: dict) -> DurationStats:
    durations = []
    for rec_name, rec_results in results.items():
        syllables = np.array(rec_results["syllable"])
        if len(syllables) == 0:
            continue

        current_syll = syllables[0]
        current_len = 1

        for i in range(1, len(syllables)):
            if syllables[i] == current_syll:
                current_len += 1
            else:
                durations.append(current_len)
                current_syll = syllables[i]
                current_len = 1
        durations.append(current_len)

    if not durations:
        return DurationStats(
            mean=0.0, std=0.0, median=0.0, min=0, max=0, total_segments=0
        )

    durations = np.array(durations)
    return DurationStats(
        mean=float(np.mean(durations)),
        std=float(np.std(durations)),
        median=float(np.median(durations)),
        min=int(np.min(durations)),
        max=int(np.max(durations)),
        total_segments=len(durations),
    )


def compute_transition_entropy(model: dict) -> float:
    if "params" not in model:
        return 0.0

    params = model["params"]

    if "betas" in params:
        trans_probs = np.array(params["betas"])
    elif "trans_probs" in params:
        trans_probs = np.array(params["trans_probs"])
    else:
        return 0.0

    if trans_probs.ndim == 1:
        probs = trans_probs / trans_probs.sum()
        probs = probs[probs > 1e-10]
        entropy = -np.sum(probs * np.log(probs))
    else:
        entropies = []
        for row in trans_probs:
            row_sum = row.sum()
            if row_sum > 1e-10:
                probs = row / row_sum
                probs = probs[probs > 1e-10]
                entropies.append(-np.sum(probs * np.log(probs)))
        entropy = np.mean(entropies) if entropies else 0.0

    return float(entropy)


def compute_active_syllables(results: dict) -> int:
    all_syllables = set()
    for rec_name, rec_results in results.items():
        syllables = np.array(rec_results["syllable"])
        all_syllables.update(syllables.tolist())
    return len(all_syllables)


def compute_syllable_usage_ratio(results: dict, num_states: int) -> float:
    active = compute_active_syllables(results)
    return active / num_states if num_states > 0 else 0.0


@dataclass
class SyllableConsistency:
    ari: float
    nmi: float


def compute_syllable_consistency(
    int_results: dict,
    kp_results: dict,
) -> SyllableConsistency:
    if not HAS_SKLEARN:
        raise ImportError("sklearn not installed")

    int_syllables = []
    kp_syllables = []

    common_keys = set(int_results.keys()) & set(kp_results.keys())

    for key in sorted(common_keys):
        int_s = np.array(int_results[key]["syllable"])
        kp_s = np.array(kp_results[key]["syllable"])

        min_len = min(len(int_s), len(kp_s))
        int_syllables.extend(int_s[:min_len].tolist())
        kp_syllables.extend(kp_s[:min_len].tolist())

    if not int_syllables:
        return SyllableConsistency(ari=0.0, nmi=0.0)

    ari = adjusted_rand_score(int_syllables, kp_syllables)
    nmi = normalized_mutual_info_score(int_syllables, kp_syllables)

    return SyllableConsistency(ari=float(ari), nmi=float(nmi))


def compute_duration_kl(
    int_results: dict,
    kp_results: dict,
    n_bins: int = 50,
    max_duration: int = 200,
) -> float:
    def get_durations(results: dict) -> list[int]:
        durations = []
        for rec_name, rec_results in results.items():
            syllables = np.array(rec_results["syllable"])
            if len(syllables) == 0:
                continue
            current_syll = syllables[0]
            current_len = 1
            for i in range(1, len(syllables)):
                if syllables[i] == current_syll:
                    current_len += 1
                else:
                    durations.append(current_len)
                    current_syll = syllables[i]
                    current_len = 1
            durations.append(current_len)
        return durations

    int_durations = get_durations(int_results)
    kp_durations = get_durations(kp_results)

    if not int_durations or not kp_durations:
        return float("inf")

    bins = np.linspace(0, max_duration, n_bins + 1)
    int_hist, _ = np.histogram(int_durations, bins=bins, density=True)
    kp_hist, _ = np.histogram(kp_durations, bins=bins, density=True)

    eps = 1e-10
    int_hist = int_hist + eps
    kp_hist = kp_hist + eps
    int_hist = int_hist / int_hist.sum()
    kp_hist = kp_hist / kp_hist.sum()

    kl_div = np.sum(int_hist * np.log(int_hist / kp_hist))
    return float(kl_div)


def collect_all_metrics(
    int_model: dict,
    kp_model: dict,
    int_results: dict,
    kp_results: dict,
    int_data: dict,
    kp_data: dict,
    int_metadata: tuple,
    kp_metadata: tuple,
    num_states: int,
) -> dict[str, Any]:
    metrics = {}

    int_duration = compute_duration_stats(int_results)
    metrics["intention_fit/duration_mean"] = int_duration.mean
    metrics["intention_fit/duration_std"] = int_duration.std
    metrics["intention_fit/duration_median"] = int_duration.median
    metrics["intention_fit/n_segments"] = int_duration.total_segments

    kp_duration = compute_duration_stats(kp_results)
    metrics["keypoint_fit/duration_mean"] = kp_duration.mean
    metrics["keypoint_fit/duration_std"] = kp_duration.std
    metrics["keypoint_fit/duration_median"] = kp_duration.median
    metrics["keypoint_fit/n_segments"] = kp_duration.total_segments

    metrics["intention_fit/transition_entropy"] = compute_transition_entropy(int_model)
    metrics["keypoint_fit/transition_entropy"] = compute_transition_entropy(kp_model)

    metrics["intention_fit/n_syllables_active"] = compute_active_syllables(int_results)
    metrics["keypoint_fit/n_syllables_active"] = compute_active_syllables(kp_results)

    metrics["intention_fit/syllable_usage_ratio"] = compute_syllable_usage_ratio(
        int_results, num_states
    )
    metrics["keypoint_fit/syllable_usage_ratio"] = compute_syllable_usage_ratio(
        kp_results, num_states
    )

    consistency = compute_syllable_consistency(int_results, kp_results)
    metrics["dual_fit/syllable_ari"] = consistency.ari
    metrics["dual_fit/syllable_nmi"] = consistency.nmi

    metrics["dual_fit/duration_kl"] = compute_duration_kl(int_results, kp_results)

    return metrics
