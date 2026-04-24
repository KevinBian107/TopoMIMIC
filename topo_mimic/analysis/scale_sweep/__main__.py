"""CLI entrypoint for scale-sweep.

Usage:
    python -m topo_mimic.analysis.scale_sweep configs/benchmark/scale_sweep.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def _setup_jax_env() -> None:
    """Set JAX/XLA env vars BEFORE any jax import so KPMS works on shared GPUs.

    Default to non-preallocating allocator with a modest fraction. Without these,
    JAX preallocates 75% of GPU memory at import time, which fails with cuSolver
    internal error when the GPU is partly occupied by other processes.

    User overrides via env: setting XLA_PYTHON_CLIENT_PREALLOCATE / MEM_FRACTION
    or JAX_PLATFORMS=cpu before invoking is honored.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")


_setup_jax_env()

from topo_mimic.analysis.scale_sweep.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Scale-sweep: KPMS kappa × num_states + Mapper lens × n_intervals × feature_space")
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument("--force", action="store_true", help="Ignore skip-if-exists and rerun every phase")
    parser.add_argument("--max-fits", type=int, default=None, help="Cap KPMS fits (smoke testing)")
    parser.add_argument("--phases", type=str, default="0,1,2,3,4,5", help="Comma-separated phase ids to run")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config)
    phases = [int(p) for p in args.phases.split(",") if p.strip()]

    from topo_mimic.analysis.scale_sweep.run import run_scale_sweep
    return run_scale_sweep(cfg, phases=phases, force=args.force, max_kpms_fits=args.max_fits)


if __name__ == "__main__":
    sys.exit(main())
