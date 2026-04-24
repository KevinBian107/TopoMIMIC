#!/bin/bash
# Scale-hyperparameter sweep for KPMS and Mapper baselines (Appendix fairness check).
set -e

PYTHON="${PYTHON:-/home/jovyan/vast/kaiwen/track-mjx/.venv/bin/python3}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================="
echo "Scale Sweep (KPMS + Mapper)"
echo "=============================="

$PYTHON -m topo_mimic.analysis.scale_sweep configs/analysis/scale_sweep/scale_sweep.yaml "$@"

echo ""
echo "Scale sweep complete."
