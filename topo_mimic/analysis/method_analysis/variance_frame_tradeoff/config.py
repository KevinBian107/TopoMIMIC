"""Config for the variance-frame tradeoff experiment."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SpaceSpec:
    name: str                 # intention | qpos | ekp
    extraction_dir: str       # outputs/extraction/dm_pipeline_<name>
    m: int                    # Takens embedding dimension
    tau: int                  # Takens delay

    @property
    def takens_offset(self) -> int:
        # stored primitive indices are Takens indices; +offset maps to clip frames
        return (self.m - 1) * self.tau


@dataclass
class VFTConfig:
    output_dir: str = "outputs/analysis/method_analysis/variance_frame_tradeoff"
    qpos_h5: str = "data/rodent_ar.h5"
    qpos_key: str = "qpos_rollout"     # (n_clips, n_frames, 74)
    skip_root_dims: int = 7            # drop global root pos(3)+quat(4); 74 -> 67
    n_random: int = 20                 # random windows per primitive for the null
    rng_seed: int = 0
    min_frames: int = 3                # need >=3 frames for a meaningful variance
    spaces: list[SpaceSpec] = field(default_factory=list)


def load_config(path: str | Path) -> VFTConfig:
    cfg = yaml.safe_load(open(path))
    spaces = [SpaceSpec(**s) for s in cfg.get("spaces", [])]
    return VFTConfig(
        output_dir=cfg.get("output_dir", VFTConfig.output_dir),
        qpos_h5=cfg.get("qpos_h5", VFTConfig.qpos_h5),
        qpos_key=cfg.get("qpos_key", VFTConfig.qpos_key),
        skip_root_dims=int(cfg.get("skip_root_dims", 7)),
        n_random=int(cfg.get("n_random", 20)),
        rng_seed=int(cfg.get("rng_seed", 0)),
        min_frames=int(cfg.get("min_frames", 3)),
        spaces=spaces,
    )
