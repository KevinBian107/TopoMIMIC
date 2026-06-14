"""Config for the trajectory-to-skeleton reduction summary."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[4]


@dataclass
class ReductionConfig:
    spaces: list[str] = field(default_factory=lambda: ["intention", "qpos", "ekp"])

    # Production extraction outputs (the cached skeletons) and the configs that
    # produced them; the latter only supply the Takens (m, tau) used to recover
    # the raw frame count n = |X| + (m-1) * tau from the point-cloud size |X|.
    extraction_out: str = "outputs/extraction/dm_pipeline_{space}"
    extraction_cfg: str = "configs/extraction/dm_pipeline_{space}.yaml"

    # Fixed cluster-window length L every primitive is resampled to (Step 3);
    # reported in the appendix so the table documents the full size budget.
    target_len: int = 100

    out_csv: str = "outputs/analysis/method_analysis/skeleton_reduction/reduction.csv"
    out_tex: str = "paper/tables/appendix_skeleton_reduction.tex"


def load_config(path: str | Path) -> ReductionConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return ReductionConfig(**raw)
