"""Hyperparameter sweep systems with WandB integration and dual-fitting."""

from topo_mimic.hyperparam_sweep.config import (
    SweepConfig,
    DataConfig,
    KeypointConfig,
    WandBConfig,
    EvaluationConfig,
    RenderingConfig,
    load_config,
)
from topo_mimic.hyperparam_sweep.kpms import (
    KPMSHyperparams,
    hyperparams_to_dict,
    KPMSFitResult,
    DualFitResult,
    fit_kpms_model,
    fit_kpms_intention,
    fit_kpms_keypoints,
    run_dual_fit,
    run_ensemble_dual_fit,
)
from topo_mimic.hyperparam_sweep.custom_vae import (
    VAEHyperparams,
    VAESweepConfig,
    vae_hyperparams_to_dict,
    load_vae_config,
    RNNVAE,
    WindowedDataset,
    VAETrainResult,
    VAETrainer,
    SegmentationResult,
    segment_hmm,
    segment_kmeans,
    segment_latent,
    VAEFitResult,
    fit_vae,
)
from topo_mimic.hyperparam_sweep.metrics import (
    compute_eml,
    compute_reconstruction_mse,
    compute_duration_stats,
    compute_transition_entropy,
    compute_active_syllables,
    compute_syllable_consistency,
    compute_duration_kl,
)
from topo_mimic.hyperparam_sweep.keypoint_loader import (
    setup_stac_reference,
    qpos_to_keypoints,
    prepare_kpms_coordinates,
)
from topo_mimic.hyperparam_sweep.wandb_logger import (
    init_sweep_run,
    log_metrics,
    log_plots,
    log_videos,
    log_training_curves,
    finish_run,
    WandBLogger,
)
from topo_mimic.hyperparam_sweep.visualization import (
    plot_duration_distribution,
    plot_syllable_frequencies,
    plot_transition_matrix,
    plot_eml_ensemble,
    plot_syllable_consistency_confusion,
    render_dual_syllable_video,
)

__all__ = [
    # Shared configs
    "SweepConfig",
    "DataConfig",
    "KeypointConfig",
    "WandBConfig",
    "EvaluationConfig",
    "RenderingConfig",
    "load_config",
    # KPMS
    "KPMSHyperparams",
    "hyperparams_to_dict",
    "KPMSFitResult",
    "DualFitResult",
    "fit_kpms_model",
    "fit_kpms_intention",
    "fit_kpms_keypoints",
    "run_dual_fit",
    "run_ensemble_dual_fit",
    # VAE
    "VAEHyperparams",
    "VAESweepConfig",
    "vae_hyperparams_to_dict",
    "load_vae_config",
    "RNNVAE",
    "WindowedDataset",
    "VAETrainResult",
    "VAETrainer",
    "SegmentationResult",
    "segment_hmm",
    "segment_kmeans",
    "segment_latent",
    "VAEFitResult",
    "fit_vae",
    # Metrics
    "compute_eml",
    "compute_reconstruction_mse",
    "compute_duration_stats",
    "compute_transition_entropy",
    "compute_active_syllables",
    "compute_syllable_consistency",
    "compute_duration_kl",
    # Keypoint loader
    "setup_stac_reference",
    "qpos_to_keypoints",
    "prepare_kpms_coordinates",
    # WandB
    "init_sweep_run",
    "log_metrics",
    "log_plots",
    "log_videos",
    "log_training_curves",
    "finish_run",
    "WandBLogger",
    # Visualization
    "plot_duration_distribution",
    "plot_syllable_frequencies",
    "plot_transition_matrix",
    "plot_eml_ensemble",
    "plot_syllable_consistency_confusion",
    "render_dual_syllable_video",
]
