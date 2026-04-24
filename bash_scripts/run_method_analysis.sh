#!/bin/bash
# Method-analysis pipeline: run each method-property study.
# Each analysis reads its own config under configs/analysis/method_analysis/
# and writes to outputs/analysis/method_analysis/.
set -e

PYTHON="${PYTHON:-/home/jovyan/vast/kaiwen/track-mjx/.venv/bin/python3}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================="
echo "Method Analysis - Property Studies"
echo "=============================="

$PYTHON -m topo_mimic.analysis.method_analysis.property.cycle_asymmetry \
        configs/analysis/method_analysis/property/cycle_asymmetry.yaml
$PYTHON -m topo_mimic.analysis.method_analysis.property.transition_topology \
        configs/analysis/method_analysis/property/transition_topology.yaml
$PYTHON -m topo_mimic.analysis.method_analysis.property.theta_filmstrip \
        configs/analysis/method_analysis/property/theta_filmstrip.yaml
$PYTHON -m topo_mimic.analysis.method_analysis.property.stationary \
        configs/analysis/method_analysis/property/stationary.yaml
$PYTHON -m topo_mimic.analysis.method_analysis.property.rear_cycles \
        configs/analysis/method_analysis/property/rear_cycles.yaml

echo ""
echo "=============================="
echo "Method Analysis - Incompressibility"
echo "=============================="

$PYTHON -m topo_mimic.analysis.method_analysis.incompressibility.run \
        configs/analysis/method_analysis/incompressibility/incompressibility.yaml
$PYTHON -m topo_mimic.analysis.method_analysis.incompressibility.batch \
        configs/analysis/method_analysis/incompressibility/batch.yaml
$PYTHON -m topo_mimic.analysis.method_analysis.incompressibility.varying_k \
        configs/analysis/method_analysis/incompressibility/varying_k.yaml

echo ""
echo "=============================="
echo "Method Analysis - Global Manifold"
echo "=============================="

$PYTHON -m topo_mimic.analysis.method_analysis.global_manifold.run \
        configs/analysis/method_analysis/global_manifold/rodent.yaml

echo ""
echo "=============================="
echo "Method Analysis - Behavioral Asymmetry (Corollary 1)"
echo "=============================="

$PYTHON -m topo_mimic.analysis.method_analysis.behavioral_asymmetry.run \
        configs/analysis/method_analysis/behavioral_asymmetry.yaml

echo ""
echo "=============================="
echo "Method Analysis - Granularity (all feature spaces)"
echo "=============================="

for space in intention qpos ekp; do
  echo ""
  echo "--- ${space} ---"
  $PYTHON -m topo_mimic.analysis.method_analysis.granularity.main \
          "configs/analysis/method_analysis/granularity/granularity_${space}.yaml"
done

echo ""
echo "Method analysis complete."
