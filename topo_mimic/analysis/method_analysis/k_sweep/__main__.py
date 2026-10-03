"""CLI driver for the k-sweep.

Examples:
  # timed probe: just extract k=5 intention
  python -m topo_mimic.analysis.method_analysis.k_sweep CONFIG \
         --stages extraction --k 5 --spaces intention
  # full sweep
  python -m topo_mimic.analysis.method_analysis.k_sweep CONFIG \
         --stages extraction experiment analysis structural
"""
from __future__ import annotations

import argparse

from .aggregate import write_downstream
from .config import load_config
from .driver import run_analysis, run_experiment, run_extraction
from .structural import write_structural

DEFAULT_CFG = "configs/analysis/method_analysis/k_sweep/k_sweep.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=DEFAULT_CFG)
    ap.add_argument("--stages", nargs="+",
                    default=["extraction", "experiment", "analysis", "structural", "aggregate"],
                    choices=["extraction", "experiment", "analysis", "structural", "aggregate"])
    ap.add_argument("--k", nargs="+", type=int, default=None,
                    help="subset of k values (default: all in config)")
    ap.add_argument("--spaces", nargs="+", default=None,
                    help="subset of feature spaces (default: all in config)")
    ap.add_argument("--force", action="store_true", help="ignore cached outputs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ks = args.k or cfg.k_values
    spaces = args.spaces or cfg.spaces

    if "extraction" in args.stages:
        for k in ks:
            for space in spaces:
                run_extraction(cfg, k, space, force=args.force)
    if "experiment" in args.stages:
        for k in ks:
            run_experiment(cfg, k, force=args.force)
    if "analysis" in args.stages:
        for k in ks:
            run_analysis(cfg, k, force=args.force)
    if "structural" in args.stages:
        write_structural(cfg)
    if "aggregate" in args.stages:
        write_downstream(cfg)


if __name__ == "__main__":
    main()
