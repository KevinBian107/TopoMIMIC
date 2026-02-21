#!/bin/bash
# Run DM pipeline for all feature spaces (intention, qpos, ekp)
set -e

PYTHON="/home/jovyan/vast/kaiwen/track-mjx/.venv/bin/python3"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================="
echo "DM Pipeline — All Feature Spaces"
echo "=============================="

echo ""
echo "--- Intention ---"
$PYTHON -m topo_mimic.dmwk.dm_pipeline configs/dmwk/dm_pipeline_intention.yaml

echo ""
echo "--- Qpos ---"
$PYTHON -m topo_mimic.dmwk.dm_pipeline configs/dmwk/dm_pipeline_qpos.yaml

echo ""
echo "--- EKP ---"
$PYTHON -m topo_mimic.dmwk.dm_pipeline configs/dmwk/dm_pipeline_ekp.yaml

echo ""
echo "All DM pipelines complete."
