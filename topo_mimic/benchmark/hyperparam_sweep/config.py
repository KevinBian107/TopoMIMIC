"""Shared configuration dataclasses for hyperparameter sweep."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from topo_mimic.benchmark.hyperparam_sweep.kpms.config import (
    KPMSHyperparams,
    hyperparams_to_dict,
)


@dataclass
class DataConfig:
    rodent_rollout_dir: str = "data/rodent_ar.h5"
    train_clips: tuple[int, int] = (0, 642)
    test_clips: tuple[int, int] = (642, 742)


@dataclass
class KeypointConfig:
    xml_path: str = "ncssm/inference/xml/single_rodent.xml"
    stac_data_path: str = "data/RodentReferenceClip.h5"
    kp_names: list[str] = field(default_factory=lambda: [
        "Snout", "EarL", "EarR", "SpineF", "SpineM", "SpineL", "TailBase",
        "ShoulderL", "ElbowL", "WristL", "HandL",
        "ShoulderR", "ElbowR", "WristR", "HandR",
        "HipL", "KneeL", "AnkleL", "FootL",
        "HipR", "KneeR", "AnkleR", "FootR",
    ])


@dataclass
class WandBConfig:
    project: str = "topomimic-benchmark-sweep"
    entity: str | None = None
    enabled: bool = True
    n_video_samples: int = 5


@dataclass
class EvaluationConfig:
    n_models_per_setting: int = 5
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])


@dataclass
class RenderingConfig:
    enabled: bool = True
    fps: int = 30
    target_size: tuple[int, int] = (320, 240)
    camera_name: str = "close_profile"


@dataclass
class SweepConfig:
    output_dir: str = "out/hyperparam_sweep"
    hyperparams: KPMSHyperparams = None
    data: DataConfig = None
    keypoint: KeypointConfig = None
    wandb: WandBConfig = None
    evaluation: EvaluationConfig = None
    rendering: RenderingConfig = None

    def __post_init__(self):
        if self.hyperparams is None:
            self.hyperparams = KPMSHyperparams()
        if self.data is None:
            self.data = DataConfig()
        if self.keypoint is None:
            self.keypoint = KeypointConfig()
        if self.wandb is None:
            self.wandb = WandBConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.rendering is None:
            self.rendering = RenderingConfig()


def load_config(config_path: str) -> SweepConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    hyperparams_dict = cfg.get("hyperparams", {})
    hyperparams = KPMSHyperparams(**hyperparams_dict)

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

    eval_dict = cfg.get("evaluation", {})
    evaluation = EvaluationConfig(**eval_dict)

    rendering_dict = cfg.get("rendering", {})
    if "target_size" in rendering_dict:
        rendering_dict["target_size"] = tuple(rendering_dict["target_size"])
    rendering = RenderingConfig(**rendering_dict)

    return SweepConfig(
        output_dir=cfg.get("output_dir", SweepConfig.output_dir),
        hyperparams=hyperparams,
        data=data,
        keypoint=keypoint,
        wandb=wandb_cfg,
        evaluation=evaluation,
        rendering=rendering,
    )
