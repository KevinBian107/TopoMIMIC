"""Configuration dataclasses for walker generation pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LongSequenceConfig:
    n_clips: int = 20
    clip_length: int = 5000
    n_transitions: int = 8
    source_pool: list[str] = field(
        default_factory=lambda: ["walk_slow", "run", "walk_fast", "walk_backward"]
    )
    target_pool: list[str] = field(
        default_factory=lambda: ["hop", "knee_down", "tiptoe", "stand"]
    )
    transition_length: int = 100
    variable_transition_length: bool = False
    transition_length_min: int = 50
    transition_length_max: int = 150


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
    demo_clip_video: str | None = None

    def __post_init__(self):
        self.h5_path = str(Path(self.h5_path).resolve())
        self.csv_path = str(Path(self.csv_path).resolve())
        if self.demo_clip_video is not None:
            self.demo_clip_video = str(Path(self.demo_clip_video).resolve())


@dataclass
class GenerationConfig:
    long_sequence: LongSequenceConfig = field(default_factory=LongSequenceConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(yaml_path: str) -> GenerationConfig:
    """Load GenerationConfig from a YAML file."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    ls_dict = raw.get("long_sequence", {})
    long_sequence = LongSequenceConfig(**ls_dict)

    checkpoints = CheckpointConfig(**raw.get("checkpoints", {}))
    schedule = ScheduleConfig(**raw.get("schedule", {}))
    rollout = RolloutConfig(**raw.get("rollout", {}))
    output = OutputConfig(**raw.get("output", {}))

    return GenerationConfig(
        long_sequence=long_sequence,
        checkpoints=checkpoints,
        schedule=schedule,
        rollout=rollout,
        output=output,
    )
