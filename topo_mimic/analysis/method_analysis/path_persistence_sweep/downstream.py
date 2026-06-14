"""Downstream confirmation: re-cluster (DTW/MMD/coverage) the DMP methods at a
few path-persistence deltas. Reuses the cached cycle/graph outputs and only the
regenerated all_path_dm per delta, then runs the EXACT production experiment +
analysis (DMP methods only) via the k-sweep override-driver utilities."""
from __future__ import annotations

import os
import pickle as pkl
from pathlib import Path

from topo_mimic.analysis.method_analysis.k_sweep.driver import _dump_yaml, _load_yaml, _run

from .config import REPO, PPSweepConfig
from .regen import regenerate_all_path_dm

PROD_EXP = "configs/experiment/transitions.yaml"
PROD_ANA = "configs/analysis/feature_analysis/analysis_transitions.yaml"


def _sandbox_extraction(cfg: PPSweepConfig, mult: float, space: str) -> Path:
    """Production extraction with all_path_dm regenerated at this space's delta."""
    src = REPO / cfg.prod_extraction_out.format(space=space)
    dst = REPO / cfg.sandbox_dir / f"delta_m{mult}" / "extraction" / f"dm_pipeline_{space}"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("all_cycle_dm.pkl", "all_G_sm.pkl", "clips"):
        link = dst / name
        if not link.exists():
            os.symlink(src / name, link)
    delta = round(mult * cfg.construct_delta[space], 6)
    apd = regenerate_all_path_dm(src, delta, cfg.segment_gap_limit, cfg.min_primitive_frames)
    with open(dst / "all_path_dm.pkl", "wb") as f:
        pkl.dump(apd, f)
    return dst


def run_downstream(cfg: PPSweepConfig) -> None:
    for mult in cfg.downstream_multipliers:
        dd = REPO / cfg.sandbox_dir / f"delta_m{mult}"
        ext = {sp: _sandbox_extraction(cfg, mult, sp) for sp in cfg.spaces}

        if not (dd / "experiment" / "all_results.pkl").exists():
            exp = _load_yaml(REPO / PROD_EXP)
            exp["output_dir"] = str(dd / "experiment")
            for sp in cfg.spaces:
                exp["dm_sources"][sp]["results_dir"] = str(ext[sp])
            for m in exp["methods"]:
                m["enabled"] = m["name"] in cfg.dm_path_methods
            exp.setdefault("rendering", {})["enabled"] = False
            exp.setdefault("cluster_merging", {})["enabled"] = False
            _dump_yaml(exp, dd / "_configs" / "experiment.yaml")
            _run([cfg.python_bin, "-m", "topo_mimic.experiment.transitions",
                  str(dd / "_configs" / "experiment.yaml")], dd / "_logs" / "experiment.log")

        if not (dd / "analysis" / "dtw_comparison_table.csv").exists():
            ana = _load_yaml(REPO / PROD_ANA)
            ana["experiment_dir"] = str(dd / "experiment")
            ana["output_dir"] = str(dd / "analysis")
            ana["methods"] = list(cfg.dm_path_methods)
            ana["run_c2st_analysis"] = False
            _dump_yaml(ana, dd / "_configs" / "analysis.yaml")
            _run([cfg.python_bin, "-m",
                  "topo_mimic.analysis.feature_analysis.main_transitions",
                  str(dd / "_configs" / "analysis.yaml")], dd / "_logs" / "analysis.log")
