"""
Atomic block runner: forces only one analysis block to run.

Each per-block entry point under ``topo_mimic.analysis.feature_analysis.blocks``
calls ``run_single_block(config_path, flag_name)``. The function loads the
full analysis YAML, disables every ``run_*`` flag except the requested one,
then delegates to the monolithic ``run_analysis`` so logic is shared.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

from .config import load_config
from .main_transitions import run_analysis


_RUN_FLAGS = (
    "run_gt_sanity_check",
    "run_mds_analysis",
    "run_transition_comparison",
    "run_transition_ranking",
    "run_transition_graph",
    "run_dtw_metrics",
    "run_dtw_visualization",
    "run_c2st_analysis",
    "run_purity_analysis",
    "run_mmd_analysis",
)


def run_single_block(config_path: str | Path, active_flag: str) -> None:
    if active_flag not in _RUN_FLAGS:
        raise ValueError(
            f"unknown block flag {active_flag!r}; expected one of {_RUN_FLAGS}"
        )
    cfg = load_config(config_path)
    for name in _RUN_FLAGS:
        if dataclasses.fields(cfg):
            setattr(cfg, name, name == active_flag)
    run_analysis(cfg)


def cli(active_flag: str) -> None:
    default = "configs/analysis/feature_analysis/analysis_transitions.yaml"
    config_path = sys.argv[1] if len(sys.argv) > 1 else default
    run_single_block(config_path, active_flag)
