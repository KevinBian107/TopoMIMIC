"""KPMS Hyperparameter Sweep System with WandB integration and dual-fitting."""

from topo_mimic.benchmark.hyperparam_sweep.config import (
    SweepConfig,
    KPMSHyperparams,
    DataConfig,
    KeypointConfig,
    WandBConfig,
    EvaluationConfig,
    load_config,
)
from topo_mimic.benchmark.hyperparam_sweep.dual_fit import (
    DualFitResult,
    fit_kpms_model,
    fit_kpms_intention,
    fit_kpms_keypoints,
    run_dual_fit,
)
from topo_mimic.benchmark.hyperparam_sweep.metrics import (
    compute_eml,
    compute_reconstruction_mse,
    compute_duration_stats,
    compute_transition_entropy,
    compute_active_syllables,
    compute_syllable_consistency,
    compute_duration_kl,
)
from topo_mimic.benchmark.hyperparam_sweep.keypoint_loader import (
    setup_stac_reference,
    qpos_to_keypoints,
    prepare_kpms_coordinates,
)
from topo_mimic.benchmark.hyperparam_sweep.wandb_logger import (
    init_sweep_run,
    log_metrics,
    log_plots,
    log_videos,
    finish_run,
)
from topo_mimic.benchmark.hyperparam_sweep.visualization import (
    plot_duration_distribution,
    plot_syllable_frequencies,
    plot_transition_matrix,
    plot_eml_ensemble,
    plot_syllable_consistency_confusion,
    render_dual_syllable_video,
)

__all__ = [
    "SweepConfig",
    "KPMSHyperparams",
    "DataConfig",
    "KeypointConfig",
    "WandBConfig",
    "EvaluationConfig",
    "load_config",
    "DualFitResult",
    "fit_kpms_model",
    "fit_kpms_intention",
    "fit_kpms_keypoints",
    "run_dual_fit",
    "compute_eml",
    "compute_reconstruction_mse",
    "compute_duration_stats",
    "compute_transition_entropy",
    "compute_active_syllables",
    "compute_syllable_consistency",
    "compute_duration_kl",
    "setup_stac_reference",
    "qpos_to_keypoints",
    "prepare_kpms_coordinates",
    "init_sweep_run",
    "log_metrics",
    "log_plots",
    "log_videos",
    "finish_run",
    "plot_duration_distribution",
    "plot_syllable_frequencies",
    "plot_transition_matrix",
    "plot_eml_ensemble",
    "plot_syllable_consistency_confusion",
    "render_dual_syllable_video",
]
