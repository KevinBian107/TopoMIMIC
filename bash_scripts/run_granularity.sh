#!/bin/bash
# Run granularity analysis
set -e

PYTHON="/home/jovyan/vast/kaiwen/track-mjx/.venv/bin/python3"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================="
echo "Granularity Analysis"
echo "=============================="

$PYTHON -m topo_mimic.granularity.main configs/dmwk/granularity.yaml

echo ""
echo "Granularity analysis complete."
