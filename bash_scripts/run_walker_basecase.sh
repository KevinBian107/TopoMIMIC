#!/bin/bash
# Walker base-case: DM extraction (intention + qpos) → benchmark → analysis
set -e

PYTHON="/root/miniforge3/envs/track_mjx/bin/python3"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================="
echo "Walker Base-Case Pipeline"
echo "=============================="

# Step 1: DM extraction
for space in intention qpos; do
  pkl="outputs/dm_pipeline_walker_short_${space}/all_cycle_dm.pkl"
  echo ""
  echo "--- DM extraction: ${space} ---"
  if [ -f "$pkl" ]; then
    echo "Results exist ($pkl), skipping."
  else
    $PYTHON -m topo_mimic.dmwk.dm_pipeline "configs/dmwk/dm_pipeline_walker_short_${space}.yaml"
  fi
done

# Step 2: Benchmark experiment
echo ""
echo "=============================="
echo "Benchmark Experiment"
echo "=============================="

rm -f outputs/benchmark_experiment_transitions_walker/w2_distance_matrix*.pkl
$PYTHON -m topo_mimic.benchmark.experiment_transitions configs/benchmark/experiment_transitions_walker.yaml

# Step 3: Analysis
echo ""
echo "=============================="
echo "Analysis"
echo "=============================="

$PYTHON -m topo_mimic.benchmark.analysis.main_transitions configs/benchmark/analysis_transitions_walker.yaml

echo ""
echo "Walker base-case pipeline complete."
echo "Results: outputs/benchmark_analysis_walker/"
