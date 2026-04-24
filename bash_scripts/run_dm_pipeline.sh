#!/bin/bash
# Run DM pipeline for all feature spaces + benchmark experiment
set -e

PYTHON="/root/miniforge3/envs/track_mjx/bin/python3"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================="
echo "DM Pipeline — All Feature Spaces"
echo "=============================="

for space in intention qpos ekp; do
  pkl="outputs/dm_pipeline_${space}/all_cycle_dm.pkl"
  echo ""
  echo "--- ${space} ---"
  if [ -f "$pkl" ]; then
    echo "Results exist ($pkl), skipping."
  else
    $PYTHON -m topo_mimic.extraction.dm_pipeline "configs/dmwk/dm_pipeline_${space}.yaml"
  fi
done

echo ""
echo "=============================="
echo "Benchmark Experiment"
echo "=============================="

$PYTHON -m topo_mimic.experiment.transitions configs/benchmark/experiment_transitions.yaml

echo ""
echo "DM pipeline + benchmark complete."
