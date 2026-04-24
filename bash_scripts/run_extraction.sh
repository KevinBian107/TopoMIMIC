#!/bin/bash
# DM extraction for all three feature spaces (intention, qpos, ekp).
# Skips any feature space whose all_cycle_dm.pkl already exists.
set -e

PYTHON="${PYTHON:-/home/jovyan/vast/kaiwen/track-mjx/.venv/bin/python3}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================="
echo "DM Extraction - All Feature Spaces"
echo "=============================="

for space in intention qpos ekp; do
  pkl="outputs/extraction/dm_pipeline_${space}/all_cycle_dm.pkl"
  echo ""
  echo "--- ${space} ---"
  if [ -f "$pkl" ]; then
    echo "Results exist ($pkl), skipping."
  else
    $PYTHON -m topo_mimic.extraction.dm_pipeline "configs/extraction/dm_pipeline_${space}.yaml"
  fi
done

echo ""
echo "Extraction complete."
