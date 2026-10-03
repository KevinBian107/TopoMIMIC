#!/bin/bash
# Full rodent pipeline end-to-end.
# Phases are independent; comment any out to skip.
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

bash bash_scripts/run_extraction.sh
bash bash_scripts/run_experiment.sh
bash bash_scripts/run_feature_analysis.sh
bash bash_scripts/run_method_analysis.sh
bash bash_scripts/run_scale_sweep.sh
bash bash_scripts/run_sensitivity.sh
bash bash_scripts/run_tables.sh

echo ""
echo "All pipelines complete."
