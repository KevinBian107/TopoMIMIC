#!/bin/bash
# Paper figures that are rendered from pipeline outputs. Cluster-example and
# lexicon panels render MuJoCo scenes and need a working GL backend (MUJOCO_GL=egl
# on a GPU node, or osmesa). Writes into paper/ unless OUT_DIR is set.
set -e
PYTHON="${PYTHON:-python}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
OUT_DIR="${OUT_DIR:-paper/bian_NeurReps_2026/appendix_figures}"
mkdir -p "$OUT_DIR"

$PYTHON -m topo_mimic.figure_generation.build_appendix_stationary      # Figure 8
$PYTHON -m topo_mimic.figure_generation.build_appendix_asymmetry       # Figure 9
$PYTHON -m topo_mimic.figure_generation.build_lexicon "$OUT_DIR/appendix_behavior_lexicon_rat.png"   # Figure 5
for m in w2_dmp_qpos_kmeans:dmp_qpos w2_dmc_int_kmeans:dmc_int kpms_keypoint:kpms_kp mapper_qpos:mapper_qpos; do
  $PYTHON -m topo_mimic.figure_generation.build_cluster_examples "${m%%:*}" "$OUT_DIR/appendix_examples_${m##*:}.png"   # Figures 10-13
done
echo "Figures written to $OUT_DIR"
