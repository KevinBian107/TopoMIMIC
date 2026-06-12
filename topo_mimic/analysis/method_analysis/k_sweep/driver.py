"""Orchestrate extraction -> experiment -> analysis per k, by overriding only
`k_neighbors` and path keys in copies of the production YAMLs and subprocessing
the unchanged production modules. The main pipeline is never modified."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import yaml

from .config import REPO, KSweepConfig


def _load_yaml(p: Path) -> dict:
    with open(p) as f:
        return yaml.safe_load(f)


def _dump_yaml(obj: dict, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _run(cmd: list[str], log: Path) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"  $ {' '.join(cmd)}")
    t0 = time.time()
    with open(log, "w") as lf:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=lf,
                              stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"command failed (rc={proc.returncode}); see {log}")
    print(f"  done in {dt:.1f}s -> {log}")
    return dt


def _k_dir(cfg: KSweepConfig, k: int) -> Path:
    return REPO / cfg.sandbox_dir / f"k_{k}"


def extraction_dir(cfg: KSweepConfig, k: int, space: str) -> Path:
    if k == cfg.reuse_k:
        return REPO / cfg.prod_extraction_out.format(space=space)
    return _k_dir(cfg, k) / "extraction" / f"dm_pipeline_{space}"


# ── stages ──────────────────────────────────────────────────────────────────

def run_extraction(cfg: KSweepConfig, k: int, space: str, force: bool = False) -> Path:
    out = extraction_dir(cfg, k, space)
    if k == cfg.reuse_k:
        print(f"[k={k} {space}] reusing production extraction: {out}")
        return out
    if (out / "all_cycle_dm.pkl").exists() and not force:
        print(f"[k={k} {space}] extraction cached, skip")
        return out
    prod = _load_yaml(REPO / cfg.prod_extraction_cfg.format(space=space))
    prod["output_dir"] = str(out)
    prod["dm"]["k_neighbors"] = k
    if cfg.disable_rendering:
        prod.setdefault("rendering", {})["enabled"] = False
    tmp = _k_dir(cfg, k) / "_configs" / f"extraction_{space}.yaml"
    _dump_yaml(prod, tmp)
    _run([cfg.python_bin, "-m", "topo_mimic.extraction.dm_pipeline", str(tmp)],
         _k_dir(cfg, k) / "_logs" / f"extraction_{space}.log")
    return out


def run_experiment(cfg: KSweepConfig, k: int, force: bool = False) -> Path:
    out = _k_dir(cfg, k) / "experiment"
    if (out / "all_results.pkl").exists() and not force:
        print(f"[k={k}] experiment cached, skip")
        return out
    prod = _load_yaml(REPO / cfg.prod_experiment_cfg)
    prod["output_dir"] = str(out)
    for space in cfg.spaces:
        prod["dm_sources"][space]["results_dir"] = str(extraction_dir(cfg, k, space))
    for m in prod["methods"]:
        m["enabled"] = m["name"] in cfg.dm_methods
    if cfg.disable_rendering:
        prod.setdefault("rendering", {})["enabled"] = False
        # Cluster-merging is a post-hoc similar-pair video analysis that needs
        # tslearn (absent in this venv) and does not affect cluster labels or
        # any k-sweep metric; disable the whole stage.
        prod.setdefault("cluster_merging", {})["enabled"] = False
    tmp = _k_dir(cfg, k) / "_configs" / "experiment.yaml"
    _dump_yaml(prod, tmp)
    _run([cfg.python_bin, "-m", "topo_mimic.experiment.transitions", str(tmp)],
         _k_dir(cfg, k) / "_logs" / "experiment.log")
    return out


def run_analysis(cfg: KSweepConfig, k: int, force: bool = False) -> Path:
    out = _k_dir(cfg, k) / "analysis"
    if (out / "dtw_comparison_table.csv").exists() and not force:
        print(f"[k={k}] analysis cached, skip")
        return out
    prod = _load_yaml(REPO / cfg.prod_analysis_cfg)
    prod["experiment_dir"] = str(_k_dir(cfg, k) / "experiment")
    prod["output_dir"] = str(out)
    prod["methods"] = list(cfg.dm_methods)
    prod["run_c2st_analysis"] = cfg.run_c2st
    tmp = _k_dir(cfg, k) / "_configs" / "analysis.yaml"
    _dump_yaml(prod, tmp)
    _run([cfg.python_bin, "-m",
          "topo_mimic.analysis.feature_analysis.main_transitions", str(tmp)],
         _k_dir(cfg, k) / "_logs" / "analysis.log")
    return out
