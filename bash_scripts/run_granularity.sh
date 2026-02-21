#!/bin/bash
# Run granularity analysis for all feature spaces
set -e

PYTHON="/home/jovyan/vast/kaiwen/track-mjx/.venv/bin/python3"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================="
echo "Granularity — All Feature Spaces"
echo "=============================="

echo ""
echo "--- Intention ---"
$PYTHON -m topo_mimic.granularity.main configs/dmwk/granularity_intention.yaml

echo ""
echo "--- Qpos ---"
$PYTHON -m topo_mimic.granularity.main configs/dmwk/granularity_qpos.yaml

echo ""
echo "--- EKP ---"
$PYTHON -m topo_mimic.granularity.main configs/dmwk/granularity_ekp.yaml

echo ""
echo "All granularity analyses complete."
