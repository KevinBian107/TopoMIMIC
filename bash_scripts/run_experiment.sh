#!/bin/bash
# Benchmark clustering experiment (all methods, all feature spaces).
set -e

PYTHON="${PYTHON:-/home/jovyan/vast/kaiwen/track-mjx/.venv/bin/python3}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================="
echo "Benchmark Experiment"
echo "=============================="

$PYTHON -m topo_mimic.experiment.transitions configs/experiment/transitions.yaml

echo ""
echo "Experiment complete."
