#!/bin/bash
# Appendix sensitivity analyses: MDS embedding dimension (Table 15), skeleton
# reduction (Table 4), k-neighbour sweep (Table 5) and path-selection persistence
# (Tables 6-7). Run after run_extraction.sh, run_experiment.sh and
# run_feature_analysis.sh. Serial on purpose: concurrent JAX processes contend for
# GPU memory.
set -e
PYTHON="${PYTHON:-python}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================="
echo "MDS embedding-dimension sweep (Table 15)"
echo "=============================="
for d in 3 5 10 20 40 60; do
  echo "--- mds=${d} ---"
  $PYTHON -m topo_mimic.experiment.transitions configs/experiment/mds_sweep/mds${d}.yaml
  $PYTHON -m topo_mimic.analysis.feature_analysis.main_transitions \
          configs/analysis/feature_analysis/mds_sweep/mds${d}.yaml
done

echo "=============================="
echo "Trajectory-to-skeleton reduction (Table 4)"
echo "=============================="
$PYTHON -m topo_mimic.analysis.method_analysis.skeleton_reduction

echo "=============================="
echo "k-neighbour sweep (Table 5)"
echo "=============================="
$PYTHON -m topo_mimic.analysis.method_analysis.k_sweep

echo "=============================="
echo "Path-selection persistence sweep (Tables 6-7)"
echo "=============================="
$PYTHON -m topo_mimic.analysis.method_analysis.path_persistence_sweep --stages ami downstream aggregate

echo ""
echo "Sensitivity analyses complete."
