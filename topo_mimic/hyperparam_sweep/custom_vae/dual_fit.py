"""VAE fitting for intention modality."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from topo_mimic.hyperparam_sweep.custom_vae.config import VAEHyperparams
from topo_mimic.hyperparam_sweep.custom_vae.model import RNNVAE
from topo_mimic.hyperparam_sweep.custom_vae.trainer import VAETrainer, VAETrainResult
from topo_mimic.hyperparam_sweep.custom_vae.segmentation import (
    SegmentationResult,
    segment_latent,
)


@dataclass
class VAEFitResult:
    model: RNNVAE
    train_result: VAETrainResult
    segmentation: SegmentationResult
    labels_list: list[np.ndarray]
    latent_embeddings: np.ndarray
    results: dict


def _labels_list_to_results_dict(labels_list: list[np.ndarray]) -> dict:
    results = {}
    for i, labels in enumerate(labels_list):
        key = f"recording_{i:04d}"
        results[key] = {"syllable": labels}
    return results


def fit_vae(
    data: np.ndarray,
    hyperparams: VAEHyperparams,
    wandb_logger: Any | None = None,
    modality_name: str = "intention",
    seed: int = 0,
) -> VAEFitResult:
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    n_clips, n_frames, input_dim = data.shape

    model = RNNVAE(
        input_dim=input_dim,
        latent_dim=hyperparams.latent_dim,
        hidden_dim=hyperparams.hidden_dim,
        rnn_layers=hyperparams.rnn_layers,
        rnn_type=hyperparams.rnn_type,
    )

    trainer = VAETrainer(model, hyperparams)

    print(f"Training VAE for {modality_name}...")
    train_result = trainer.train(
        train_data=data,
        wandb_logger=wandb_logger,
        modality_name=modality_name,
        seed=seed,
    )

    print(f"Encoding per-frame latent embeddings for {modality_name}...")
    latent_embeddings = trainer.encode_per_frame(data)

    print(f"Segmenting {modality_name} latent space using {hyperparams.segmentation_algo}...")
    segmentation = segment_latent(
        latent_embeddings=latent_embeddings,
        n_clusters=hyperparams.n_clusters,
        method=hyperparams.segmentation_algo,
        random_state=seed,
    )

    results = _labels_list_to_results_dict(segmentation.labels_list)

    return VAEFitResult(
        model=model,
        train_result=train_result,
        segmentation=segmentation,
        labels_list=segmentation.labels_list,
        latent_embeddings=latent_embeddings,
        results=results,
    )
