"""VAE-specific hyperparameter configuration."""

from dataclasses import dataclass, field
from typing import Any

import yaml

from topo_mimic.hyperparam_sweep.config import (
    DataConfig,
    KeypointConfig,
    WandBConfig,
    RenderingConfig,
)


@dataclass
class VAEHyperparams:
    latent_dim: int = 30
    time_window: int = 30
    hidden_dim: int = 256
    rnn_layers: int = 1
    rnn_type: str = "GRU"
    beta: float = 1.0
    n_clusters: int = 15
    segmentation_algo: str = "hmm"
    learning_rate: float = 5e-4
    batch_size: int = 256
    max_epochs: int = 200


@dataclass
class VAESweepConfig:
    output_dir: str = "outputs/vae_hyperparam_sweep"
    hyperparams: VAEHyperparams = None
    data: DataConfig = None
    keypoint: KeypointConfig = None
    wandb: WandBConfig = None
    rendering: RenderingConfig = None

    def __post_init__(self):
        if self.hyperparams is None:
            self.hyperparams = VAEHyperparams()
        if self.data is None:
            self.data = DataConfig()
        if self.keypoint is None:
            self.keypoint = KeypointConfig()
        if self.wandb is None:
            self.wandb = WandBConfig()
        if self.rendering is None:
            self.rendering = RenderingConfig()


def vae_hyperparams_to_dict(hp: VAEHyperparams) -> dict[str, Any]:
    return {
        "latent_dim": hp.latent_dim,
        "time_window": hp.time_window,
        "hidden_dim": hp.hidden_dim,
        "rnn_layers": hp.rnn_layers,
        "rnn_type": hp.rnn_type,
        "beta": hp.beta,
        "n_clusters": hp.n_clusters,
        "segmentation_algo": hp.segmentation_algo,
        "learning_rate": hp.learning_rate,
        "batch_size": hp.batch_size,
        "max_epochs": hp.max_epochs,
    }


def load_vae_config(config_path: str) -> VAESweepConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    hyperparams_dict = cfg.get("hyperparams", {})
    hyperparams = VAEHyperparams(**hyperparams_dict)

    data_dict = cfg.get("data", {})
    if "train_clips" in data_dict:
        data_dict["train_clips"] = tuple(data_dict["train_clips"])
    if "test_clips" in data_dict:
        data_dict["test_clips"] = tuple(data_dict["test_clips"])
    data = DataConfig(**data_dict)

    keypoint_dict = cfg.get("keypoint", {})
    keypoint = KeypointConfig(**keypoint_dict)

    wandb_dict = cfg.get("wandb", {})
    wandb_cfg = WandBConfig(**wandb_dict)

    rendering_dict = cfg.get("rendering", {})
    if "target_size" in rendering_dict:
        rendering_dict["target_size"] = tuple(rendering_dict["target_size"])
    rendering = RenderingConfig(**rendering_dict)

    return VAESweepConfig(
        output_dir=cfg.get("output_dir", VAESweepConfig.output_dir),
        hyperparams=hyperparams,
        data=data,
        keypoint=keypoint,
        wandb=wandb_cfg,
        rendering=rendering,
    )
