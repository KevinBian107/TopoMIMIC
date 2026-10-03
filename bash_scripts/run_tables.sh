#!/bin/bash
# Generate paper tables (results + Figure 4 summary + Appendix DTW + warping).
# Pass "paper", "warping", or "results" to generate just that family.
set -e

PYTHON="${PYTHON:-/home/jovyan/vast/kaiwen/track-mjx/.venv/bin/python3}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

SUB="${1:-all}"

echo "=============================="
echo "Table Generation: ${SUB}"
echo "=============================="

# Auxiliary tables (paper_tables.py PDFs, results_table.py, warping_tables.py
# scale-sweep variants) under outputs/analysis/tables/.
$PYTHON -m topo_mimic.analysis.tables "$SUB"

# Paper-ready .tex tables under paper/tables/. Reads from
# outputs/analysis/feature_analysis/*.csv, including the new
# sync_ratio_comparison_table.csv and non_diag_comparison_table.csv produced
# by the dtw_metrics feature_analysis block.
if [ "$SUB" = "all" ] || [ "$SUB" = "paper" ]; then
  $PYTHON -m topo_mimic.figure_generation.build_tables
fi

echo ""
echo "Tables written to outputs/analysis/tables/ and paper/tables/."
