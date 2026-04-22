"""Transition classifier: N-class MLP over canonical directional transitions.

Replaces the KID pipeline. Vocabulary is the full set of directed transitions
observed in `rodent_labels.csv` across {Immobile, Turn, Rear, Walk} (4 behaviors
-> 12 directed pairs). Sorted here by decreasing GT instance count so minority
classes are the rightmost columns in the confusion matrix.
"""

TRANSITION_LABELS = [
    "Turn -> Immobile",    # 669
    "Immobile -> Turn",    # 605
    "Immobile -> Rear",    # 155
    "Rear -> Immobile",    # 153
    "Rear -> Turn",        # 151
    "Turn -> Walk",        # 113
    "Walk -> Turn",        # 100
    "Walk -> Immobile",    # 64
    "Turn -> Rear",        # 46
    "Immobile -> Walk",    # 44
    "Rear -> Walk",        # 12
    "Walk -> Rear",        # 5
]

# Backwards-compat alias — older call sites used TOP_8_TRANSITIONS.
TOP_8_TRANSITIONS = TRANSITION_LABELS

LABEL_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(TRANSITION_LABELS)}
IDX_TO_LABEL: dict[int, str] = {i: t for i, t in enumerate(TRANSITION_LABELS)}

__all__ = [
    "TRANSITION_LABELS",
    "TOP_8_TRANSITIONS",
    "LABEL_TO_IDX",
    "IDX_TO_LABEL",
]
