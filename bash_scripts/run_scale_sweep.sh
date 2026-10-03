#!/bin/bash
# Scale-hyperparameter sweep for KPMS and Mapper baselines (Appendix fairness check).
set -e

PYTHON="${PYTHON:-python}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================="
echo "Scale Sweep (KPMS + Mapper)"
echo "=============================="

$PYTHON -m topo_mimic.analysis.scale_sweep configs/analysis/scale_sweep/scale_sweep.yaml "$@"
# Table 3: score every swept variant at the directed, direction-turn resolution
$PYTHON -m topo_mimic.analysis.scale_sweep.rescore

echo ""
echo "Scale sweep complete."
