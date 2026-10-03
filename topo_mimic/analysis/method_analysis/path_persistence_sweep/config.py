"""Config for the path-persistence sweep."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[4]


@dataclass
class PPSweepConfig:
    # delta_path = multiplier x (per-space construction delta), so the grid auto-
    # scales to each space's inverse-density units (Table tab:dm-hp).
    delta_multipliers: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 4.0])
    construct_delta: dict = field(default_factory=lambda: {
        "intention": 1.0, "qpos": 0.2, "ekp": 0.1})
    spaces: list[str] = field(default_factory=lambda: ["intention", "qpos", "ekp"])

    prod_extraction_out: str = "outputs/extraction/dm_pipeline_{space}"
    sandbox_dir: str = "outputs/analysis/method_analysis/path_persistence_sweep"
    # delta_path = 0 row. The delta_path > 0 sandboxes cluster at MDS dimension 3, so
    # the baseline must too: the k=3 sandbox is the same skeleton at the same MDS
    # dimension, whereas production feature_analysis uses the per-variant dimensions.
    baseline_analysis_dir: str = "outputs/analysis/method_analysis/k_sweep/k_3/analysis"

    # GT / info-theoretic settings (mirror configs/.../information_loss.yaml).
    labels_csv: str = "behavior_labels/rodent_labels.csv"
    n_clips: int = 842
    frames_per_clip: int = 500
    y_granularity: str = "directed"
    min_class_count: int = 5
    n_bootstrap: int = 100

    # path-segmentation params (production dm_pipeline_*.yaml).
    segment_gap_limit: int = 50
    min_primitive_frames: int = 30

    # downstream confirmation runs at a subset of multipliers.
    downstream_multipliers: list[float] = field(default_factory=lambda: [0.0, 1.0, 4.0])
    dm_path_methods: list[str] = field(default_factory=lambda: [
        "w2_dmp_int_kmeans", "w2_dmp_qpos_kmeans", "w2_dmp_ekp_kmeans"])
    python_bin: str = "/home/jovyan/vast/kaiwen/track-mjx/.venv/bin/python3"

    def deltas_for(self, space: str) -> list[float]:
        d = self.construct_delta[space]
        return [round(m * d, 6) for m in self.delta_multipliers]


def load_config(path: str | Path) -> PPSweepConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return PPSweepConfig(**raw)
