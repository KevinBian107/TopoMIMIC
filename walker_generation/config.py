"""Configuration dataclasses for walker generation pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TransitionSpec:
    src: str
    dst: str
    n_clips: int


@dataclass
class CheckpointConfig:
    step1_path: str = "data/checkpoints/walker_multi_behavior/260215_202424"
    step2_path: str = "data/checkpoints/walker_imitation/260224_030923"
    step1_step: int | None = None

    def __post_init__(self):
        self.step1_path = str(Path(self.step1_path).resolve())
        self.step2_path = str(Path(self.step2_path).resolve())


@dataclass
class ScheduleConfig:
    clip_length: int = 500
    rollout_frames: int = 2000
    warmup_frames: int = 200
    warmup_transition_frames: int = 50
    mode_duration_min: int = 250
    mode_duration_extra_max: int = 100
    transition_mean: int = 100
    transition_std: int = 20


@dataclass
class RolloutConfig:
    batch_size: int = 200
    extraction_batch_size: int = 100
    seed: int = 0


@dataclass
class OutputConfig:
    h5_path: str = "data/walker.h5"
    csv_path: str = "walker_behavior_labels.csv"

    def __post_init__(self):
        self.h5_path = str(Path(self.h5_path).resolve())
        self.csv_path = str(Path(self.csv_path).resolve())


@dataclass
class GenerationConfig:
    transitions: list[TransitionSpec] = field(default_factory=list)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(yaml_path: str) -> GenerationConfig:
    """Load GenerationConfig from a YAML file."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    transitions = [TransitionSpec(**t) for t in raw.get("transitions", [])]
    checkpoints = CheckpointConfig(**raw.get("checkpoints", {}))
    schedule = ScheduleConfig(**raw.get("schedule", {}))
    rollout = RolloutConfig(**raw.get("rollout", {}))
    output = OutputConfig(**raw.get("output", {}))

    return GenerationConfig(
        transitions=transitions,
        checkpoints=checkpoints,
        schedule=schedule,
        rollout=rollout,
        output=output,
    )
