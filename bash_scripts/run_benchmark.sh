#!/bin/bash
# Run benchmark experiment
set -e

PYTHON="/home/jovyan/vast/kaiwen/track-mjx/.venv/bin/python3"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================="
echo "Benchmark Experiment"
echo "=============================="

$PYTHON -m topo_mimic.benchmark.experiment configs/benchmark/experiment.yaml

echo ""
echo "Benchmark complete."
