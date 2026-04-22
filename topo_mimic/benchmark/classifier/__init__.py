"""Transition classifier: 8-class MLP over canonical directional transitions.

Replaces the KID pipeline. Provides:
  - `preprocessing`: qpos → yaw-stripped 6D rotation + z-scored joints (same as KID)
  - `labels`: TOP_8_TRANSITIONS + frame-resampling to a fixed window
  - `model`: Flax MLP
  - `data`: window extraction from GT + resampled clip windows
  - `train`: 5-fold CV trainer with confusion-matrix reporting
  - `cluster_eval`: per-cluster match rate / per-cluster confusion against the
    classifier, run as an analysis stage after the benchmark.
"""

TOP_8_TRANSITIONS = [
    "Turn -> Immobile",
    "Immobile -> Turn",
    "Immobile -> Rear",
    "Rear -> Immobile",
    "Rear -> Turn",
    "Turn -> Walk",
    "Walk -> Turn",
    "Walk -> Immobile",
]

LABEL_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(TOP_8_TRANSITIONS)}
IDX_TO_LABEL: dict[int, str] = {i: t for i, t in enumerate(TOP_8_TRANSITIONS)}

__all__ = [
    "TOP_8_TRANSITIONS",
    "LABEL_TO_IDX",
    "IDX_TO_LABEL",
]
