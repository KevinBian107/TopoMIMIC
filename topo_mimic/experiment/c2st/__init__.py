"""Classifier 2-Sample Test (C2ST) per-cluster evaluation.

Replaces the KID + pretrained-transition-classifier pipelines.

For each benchmark cluster labeled with a directional transition
(auto_label_clusters_directional output), we:

1. Extract the cluster's member windows (qpos with the first 7 root dims
   stripped — same as DTW / KID preprocessing, i.e. pure joint angles —
   resampled to K=100 frames).
2. Extract real GT windows of the same DIRECTIONAL transition label,
   excluding clips that appear in the cluster's own members (clip-level
   leakage control).
3. Downsample/upsample real GT to match the cluster size (balanced pool).
4. Train three tiny binary classifiers (LogReg, RandomForest, MLP) to
   predict `is_from_cluster` via 5-fold stratified CV. Report mean held-out
   accuracy per model.

Aggregation: pool across 3 seeds, collapse to UNDIRECTED (A↔B) rows,
combine feature-space variants into method families (DMC, DMP, W2-Unif,
KPMS, VAE, Mapper). One output table per model type (3 total).

Interpretation:
  0.50 == indistinguishable → cluster windows look like real L
  1.00 == trivially separable → cluster is NOT like real L
"""

TRANSITION_LABELS = [
    "Turn -> Immobile",    "Immobile -> Turn",
    "Immobile -> Rear",    "Rear -> Immobile",
    "Rear -> Turn",        "Turn -> Rear",
    "Turn -> Walk",        "Walk -> Turn",
    "Walk -> Immobile",    "Immobile -> Walk",
    "Rear -> Walk",        "Walk -> Rear",
]

# Undirected canonical pairs (sorted, alphabetical within pair).
UNDIRECTED_PAIRS = [
    "Immobile ↔ Turn",
    "Immobile ↔ Rear",
    "Rear ↔ Turn",
    "Turn ↔ Walk",
    "Immobile ↔ Walk",
    "Rear ↔ Walk",
]

FAMILY_MEMBERS: dict[str, list[str]] = {
    "DMC (Int+qpos+EKP)":    ["w2_dmc_int_kmeans", "w2_dmc_qpos_kmeans", "w2_dmc_ekp_kmeans"],
    "DMP (Int+qpos+EKP)":    ["w2_dmp_int_kmeans", "w2_dmp_qpos_kmeans", "w2_dmp_ekp_kmeans"],
    "W₂-Unif":               ["w2_unif_kmeans"],
    "KPMS (Int+KP)":         ["kpms_intention", "kpms_keypoint"],
    "VAE":                   ["custom_vae"],
    "Mapper (Int+qpos+EKP)": ["mapper_int", "mapper_qpos", "mapper_ekp"],
}


def undirected_key(directed: str) -> str | None:
    """Map 'A -> B' to canonical 'A ↔ B' (sorted alphabetically)."""
    if " -> " not in directed:
        return None
    a, b = directed.split(" -> ", 1)
    a, b = sorted([a.strip(), b.strip()])
    key = f"{a} ↔ {b}"
    return key if key in UNDIRECTED_PAIRS else None


__all__ = [
    "TRANSITION_LABELS",
    "UNDIRECTED_PAIRS",
    "FAMILY_MEMBERS",
    "undirected_key",
]
