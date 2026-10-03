"""CLI for the trajectory-to-skeleton reduction summary.

  python -m topo_mimic.analysis.method_analysis.skeleton_reduction [CONFIG]

Reads the cached DM-skeletons and writes outputs/.../reduction.csv and
paper/bian_NeurReps_2026/tables/appendix_skeleton_reduction.tex.
"""
from __future__ import annotations

import argparse

from .aggregate import run
from .config import load_config

DEFAULT_CFG = "configs/analysis/method_analysis/skeleton_reduction/skeleton_reduction.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=DEFAULT_CFG)
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
