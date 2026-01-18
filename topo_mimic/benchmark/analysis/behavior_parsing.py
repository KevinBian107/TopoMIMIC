"""Behavior parsing and normalization utilities."""

from dataclasses import dataclass
from enum import Enum


class TurnDirection(Enum):
    LEFT = "left"
    RIGHT = "right"
    STRAIGHT = "straight"

    def __str__(self) -> str:
        return self.value


BEHAVIOR_NAME_MAP = {
    "walking": "Walking",
    "walk": "Walking",
    "immobility": "Immobility",
    "immobile": "Immobility",
    "orienting": "Orienting",
    "orient": "Orienting",
    "grooming": "Grooming",
    "groom": "Grooming",
    "supported_rear": "Supported Rear",
    "supportedrear": "Supported Rear",
    "srear": "Supported Rear",
    "unsupported_rear": "Unsupported Rear",
    "unsupportedrear": "Unsupported Rear",
    "urear": "Unsupported Rear",
    "rear": "Rear",
    "left_turn": "Left Turn",
    "leftturn": "Left Turn",
    "lturn": "Left Turn",
    "right_turn": "Right Turn",
    "rightturn": "Right Turn",
    "rturn": "Right Turn",
}

PRIMITIVE_BEHAVIORS = {
    "Walking", "Immobility", "Orienting", "Grooming",
    "Supported Rear", "Unsupported Rear", "Rear",
}

COMPOSITIONAL_BEHAVIORS = {
    "Left Turn", "Right Turn",
}

VALID_BEHAVIORS = PRIMITIVE_BEHAVIORS | COMPOSITIONAL_BEHAVIORS

REAR_BEHAVIORS = {"Supported Rear", "Unsupported Rear"}


def normalize_behavior_name(name: str) -> str:
    if name in VALID_BEHAVIORS:
        return name
    name_lower = name.lower().strip().replace(" ", "_")
    return BEHAVIOR_NAME_MAP.get(name_lower, name)


def merge_rear_category(behavior: str) -> str:
    if behavior in REAR_BEHAVIORS:
        return "Rear"
    return behavior


@dataclass
class ParsedLabel:
    original: str
    primitives: list[str]
    compositional: list[str]
    comparison_type: str  # "transition", "intersection", or "single"
    from_turn: TurnDirection = TurnDirection.STRAIGHT
    to_turn: TurnDirection = TurnDirection.STRAIGHT

    @property
    def gt_key(self) -> tuple:
        if self.comparison_type == "transition":
            return ("transition", tuple(sorted(self.primitives[:2])))
        elif self.comparison_type == "intersection":
            return ("intersection", tuple(sorted(self.primitives + self.compositional)))
        else:
            return ("single", tuple(self.primitives + self.compositional))

    @property
    def gt_key_with_turns(self) -> tuple:
        if self.comparison_type == "transition":
            return (
                "transition",
                tuple(sorted(self.primitives[:2])),
                self.from_turn,
                self.to_turn,
            )
        elif self.comparison_type == "intersection":
            return (
                "intersection",
                tuple(sorted(self.primitives + self.compositional)),
                self.from_turn,
                self.to_turn,
            )
        else:
            return ("single", tuple(self.primitives + self.compositional))


def parse_cluster_label(label: str, merge_rear: bool = True) -> ParsedLabel | None:
    if not label or not label.strip():
        return None

    parts = label.strip().split()
    primitives = []
    compositional = []

    for part in parts:
        normalized = normalize_behavior_name(part)
        if merge_rear:
            normalized = merge_rear_category(normalized)
        if normalized in PRIMITIVE_BEHAVIORS:
            primitives.append(normalized)
        elif normalized in COMPOSITIONAL_BEHAVIORS:
            compositional.append(normalized)

    if not primitives and not compositional:
        return None

    if len(primitives) >= 2:
        comparison_type = "transition"
    elif len(primitives) == 1 and len(compositional) >= 1:
        comparison_type = "intersection"
    elif len(primitives) == 1:
        comparison_type = "single"
    elif len(compositional) >= 1:
        comparison_type = "single"
    else:
        return None

    from_turn = TurnDirection.STRAIGHT
    to_turn = TurnDirection.STRAIGHT
    if compositional:
        if "Left Turn" in compositional:
            from_turn = TurnDirection.LEFT
            to_turn = TurnDirection.LEFT
        elif "Right Turn" in compositional:
            from_turn = TurnDirection.RIGHT
            to_turn = TurnDirection.RIGHT

    return ParsedLabel(
        original=label,
        primitives=primitives,
        compositional=compositional,
        comparison_type=comparison_type,
        from_turn=from_turn,
        to_turn=to_turn,
    )


def turn_direction_to_string(turn: TurnDirection) -> str:
    if turn == TurnDirection.LEFT:
        return "L"
    elif turn == TurnDirection.RIGHT:
        return "R"
    else:
        return "S"


def format_transition_with_turns(
    from_behavior: str,
    to_behavior: str,
    from_turn: TurnDirection,
    to_turn: TurnDirection,
) -> str:
    from_str = turn_direction_to_string(from_turn)
    to_str = turn_direction_to_string(to_turn)
    return f"{from_str}-{from_behavior} -> {to_str}-{to_behavior}"
