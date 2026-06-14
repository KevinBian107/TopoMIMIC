"""CLI for the path-persistence sweep.

  python -m topo_mimic.analysis.method_analysis.path_persistence_sweep CONFIG \
         --stages ami
"""
from __future__ import annotations

import argparse

from .aggregate import write_downstream_vs_delta
from .ami import run_ami
from .config import load_config
from .downstream import run_downstream

DEFAULT_CFG = "configs/analysis/method_analysis/path_persistence_sweep/path_persistence_sweep.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=DEFAULT_CFG)
    ap.add_argument("--stages", nargs="+", default=["ami"],
                    choices=["ami", "downstream", "aggregate"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    if "ami" in args.stages:
        run_ami(cfg)
    if "downstream" in args.stages:
        run_downstream(cfg)
    if "aggregate" in args.stages:
        write_downstream_vs_delta(cfg)


if __name__ == "__main__":
    main()
