"""Config for the k-sweep driver."""
from __future__ import annotations
import sys

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[4]


@dataclass
class KSweepConfig:
    # k values to sweep; `reuse_k` is the production value whose extraction
    # already exists and is reused verbatim (also a fidelity anchor).
    k_values: list[int] = field(default_factory=lambda: [3, 5, 8])
    reuse_k: int = 3
    spaces: list[str] = field(default_factory=lambda: ["intention", "qpos", "ekp"])

    sandbox_dir: str = "outputs/analysis/method_analysis/k_sweep"

    # Production YAMLs that are loaded and overridden (never rewritten).
    prod_extraction_cfg: str = "configs/extraction/dm_pipeline_{space}.yaml"
    prod_experiment_cfg: str = "configs/experiment/transitions.yaml"
    prod_analysis_cfg: str = "configs/analysis/feature_analysis/analysis_transitions.yaml"
    prod_extraction_out: str = "outputs/extraction/dm_pipeline_{space}"

    # The DM methods that depend on k; everything else (k-invariant baselines)
    # is disabled for the sweep.
    dm_methods: list[str] = field(default_factory=lambda: [
        "w2_dmc_int_kmeans", "w2_dmc_qpos_kmeans", "w2_dmc_ekp_kmeans",
        "w2_dmp_int_kmeans", "w2_dmp_qpos_kmeans", "w2_dmp_ekp_kmeans",
    ])

    run_c2st: bool = False          # slow, outside the requested metric set
    disable_rendering: bool = True  # videos don't affect primitives/metrics
    python_bin: str = field(default_factory=lambda: sys.executable)


def load_config(path: str | Path) -> KSweepConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return KSweepConfig(**raw)
