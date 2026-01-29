"""Custom VAE-based behavior segmentation for hyperparameter sweep."""

from topo_mimic.benchmark.hyperparam_sweep.custom_vae.config import (
    VAEHyperparams,
    VAESweepConfig,
    vae_hyperparams_to_dict,
    load_vae_config,
)
from topo_mimic.benchmark.hyperparam_sweep.custom_vae.model import RNNVAE
from topo_mimic.benchmark.hyperparam_sweep.custom_vae.dataset import WindowedDataset
from topo_mimic.benchmark.hyperparam_sweep.custom_vae.trainer import (
    VAETrainResult,
    VAETrainer,
)
from topo_mimic.benchmark.hyperparam_sweep.custom_vae.segmentation import (
    SegmentationResult,
    segment_hmm,
    segment_kmeans,
    segment_latent,
)
from topo_mimic.benchmark.hyperparam_sweep.custom_vae.dual_fit import (
    VAEFitResult,
    fit_vae,
)

__all__ = [
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
]
