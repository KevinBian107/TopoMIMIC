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

$PYTHON -m topo_mimic.analysis.tables "$SUB"

echo ""
echo "Tables written to outputs/analysis/tables/."
