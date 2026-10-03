#!/bin/bash
# Feature-analysis pipeline: run each atomic block in sequence.
# Each block reads the same analysis_transitions.yaml config and writes its
# own subdirectory under outputs/analysis/feature_analysis/.
#
# To run only one block, call its module directly, e.g.:
#   python -m topo_mimic.analysis.feature_analysis.blocks.dtw_metrics \
#          configs/analysis/feature_analysis/analysis_transitions.yaml
set -e

PYTHON="${PYTHON:-python}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

CFG="configs/analysis/feature_analysis/analysis_transitions.yaml"

echo "=============================="
echo "Feature Analysis - Atomic Blocks"
echo "=============================="

for block in gt_sanity transition_ranking mds transition_comparison \
             transition_graph dtw_metrics dtw_visualization mmd c2st \
             purity classification; do
  echo ""
  echo "--- block: ${block} ---"
  $PYTHON -m topo_mimic.analysis.feature_analysis.blocks.${block} "$CFG"
done

echo ""
echo "Feature analysis complete."
