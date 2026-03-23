"""Ground truth extraction with short behavior bridging and turn direction."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .behavior_parsing import (
    PRIMITIVE_BEHAVIORS,
    COMPOSITIONAL_BEHAVIORS,
    TurnDirection,
    normalize_behavior_name,
    merge_rear_category,
    is_forbidden_transition,
)


@dataclass
class BehaviorSegment:
    clip_id: int
    behavior: str
    start_frame: int
    end_frame: int
    turn_direction: TurnDirection = TurnDirection.STRAIGHT

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class TransitionSegment:
    clip_id: int
    from_behavior: str
    to_behavior: str
    from_start: int
    from_end: int
    to_start: int
    to_end: int
    from_turn: TurnDirection = TurnDirection.STRAIGHT
    to_turn: TurnDirection = TurnDirection.STRAIGHT

    @property
    def all_frames(self) -> list[int]:
        frames = list(range(self.from_start, self.from_end + 1))
        frames.extend(range(self.to_start, self.to_end + 1))
        return frames

    @property
    def transition_key(self) -> tuple[str, str]:
        return tuple(sorted([self.from_behavior, self.to_behavior]))

    @property
    def transition_key_with_turns(self) -> tuple:
        key = self.transition_key
        if key[0] == self.from_behavior:
            return (key[0], key[1], self.from_turn, self.to_turn)
        else:
            return (key[0], key[1], self.to_turn, self.from_turn)


def load_behavior_segments(
    csv_path: str,
    merge_rear: bool = True,
) -> tuple[list[BehaviorSegment], dict[int, list[tuple[int, int, str]]]]:
    """
    Load behavior segments and turn overlaps from CSV.

    Returns:
        Tuple of (primitive_segments, turn_overlaps_by_clip)
        - primitive_segments: List of BehaviorSegment for primitive behaviors
        - turn_overlaps_by_clip: Dict mapping clip_id to list of (start, end, turn_type)
    """
    df = pd.read_csv(csv_path)

    primitive_segments = []
    turn_overlaps_by_clip = defaultdict(list)

    for _, row in df.iterrows():
        start_frame = int(row["Start_Frame"])
        end_frame = int(row["End_Frame"])
        if start_frame == end_frame == 0:
            continue

        video = str(row["Video"])
        clip_id = int(video.replace("clip_", "").replace(".mp4", ""))
        behavior = normalize_behavior_name(str(row["Behavior"]))

        if behavior in COMPOSITIONAL_BEHAVIORS:
            turn_overlaps_by_clip[clip_id].append((start_frame, end_frame, behavior))
        elif behavior in PRIMITIVE_BEHAVIORS:
            if merge_rear:
                behavior = merge_rear_category(behavior)
            primitive_segments.append(BehaviorSegment(
                clip_id=clip_id,
                behavior=behavior,
                start_frame=start_frame,
                end_frame=end_frame,
            ))

    return primitive_segments, dict(turn_overlaps_by_clip)


def get_turn_direction_for_segment(
    segment: BehaviorSegment,
    turn_overlaps: list[tuple[int, int, str]],
    transition_frame: int,
) -> TurnDirection:
    """
    Determine turn direction for a segment based on overlapping turn labels.

    If multiple turns overlap, uses the one closest to the transition_frame.

    Args:
        segment: The behavior segment to check
        turn_overlaps: List of (start, end, turn_type) for the clip
        transition_frame: The frame where the transition occurs (end of 'from' segment)

    Returns:
        TurnDirection (LEFT, RIGHT, or STRAIGHT)
    """
    overlapping_turns = []

    for turn_start, turn_end, turn_type in turn_overlaps:
        overlap_start = max(segment.start_frame, turn_start)
        overlap_end = min(segment.end_frame, turn_end)

        if overlap_start <= overlap_end:
            closest_frame = min(max(turn_start, transition_frame), turn_end)
            distance = abs(closest_frame - transition_frame)
            overlapping_turns.append((distance, turn_end, turn_type))

    if not overlapping_turns:
        return TurnDirection.STRAIGHT

    overlapping_turns.sort(key=lambda x: (x[0], -x[1]))
    _, _, turn_type = overlapping_turns[0]

    if turn_type == "Left Turn":
        return TurnDirection.LEFT
    elif turn_type == "Right Turn":
        return TurnDirection.RIGHT
    else:
        return TurnDirection.STRAIGHT


def _merge_short_intermediate_behaviors(
    segments: list[BehaviorSegment],
    short_threshold: int,
) -> list[BehaviorSegment]:
    """
    Merge segments when short intermediate behaviors separate same-type segments.

    Example: rear -> walk -> orienting(short) -> walk becomes rear -> walk
    The short orienting is bridged and the two walk segments are merged.
    """
    if len(segments) < 3:
        return segments

    merged = []
    i = 0

    while i < len(segments):
        current = segments[i]

        if i + 2 < len(segments):
            middle = segments[i + 1]
            after = segments[i + 2]

            if (current.behavior == after.behavior and
                middle.behavior != current.behavior and
                middle.length <= short_threshold):
                new_segment = BehaviorSegment(
                    clip_id=current.clip_id,
                    behavior=current.behavior,
                    start_frame=current.start_frame,
                    end_frame=after.end_frame,
                    turn_direction=current.turn_direction,
                )
                segments = segments[:i] + [new_segment] + segments[i+3:]
                continue

        merged.append(current)
        i += 1

    if len(merged) < len(segments):
        return _merge_short_intermediate_behaviors(merged, short_threshold)

    return merged


def _group_segments_by_clip(
    segments: list[BehaviorSegment],
) -> dict[int, list[BehaviorSegment]]:
    """Group segments by clip_id and sort by start_frame."""
    by_clip = defaultdict(list)
    for seg in segments:
        by_clip[seg.clip_id].append(seg)

    for clip_id in by_clip:
        by_clip[clip_id].sort(key=lambda x: x.start_frame)

    return dict(by_clip)


def find_gt_transition_frames(
    csv_path: str,
    primitives: list[str],
    short_threshold: int = 10,
    max_gap: int = 10,
    balance_segments: bool = True,
    merge_rear: bool = True,
    include_turn_direction: bool = True,
) -> list[TransitionSegment]:
    """
    Find GT transition segments with short behavior bridging and turn directions.

    Args:
        csv_path: Path to behavior_labels.csv
        primitives: List of two primitive behaviors (e.g., ["Walking", "Rear"])
        short_threshold: Max frames for intermediate behavior to be bridged
        max_gap: Maximum allowed gap between consecutive segments
        balance_segments: If True, truncate longer segment to match shorter
        merge_rear: If True, merge Supported/Unsupported Rear into Rear
        include_turn_direction: If True, compute turn directions for each segment

    Returns:
        List of TransitionSegment objects
    """
    if len(primitives) < 2:
        return []

    b1 = normalize_behavior_name(primitives[0])
    b2 = normalize_behavior_name(primitives[1])
    if merge_rear:
        b1 = merge_rear_category(b1)
        b2 = merge_rear_category(b2)
    target_pair = tuple(sorted([b1, b2]))

    all_segments, turn_overlaps_by_clip = load_behavior_segments(csv_path, merge_rear)
    segments_by_clip = _group_segments_by_clip(all_segments)

    transitions = []

    for clip_id, clip_segments in segments_by_clip.items():
        merged_segments = _merge_short_intermediate_behaviors(
            clip_segments, short_threshold
        )

        turn_overlaps = turn_overlaps_by_clip.get(clip_id, [])

        for i in range(len(merged_segments) - 1):
            seg_a = merged_segments[i]
            seg_b = merged_segments[i + 1]

            pair = tuple(sorted([seg_a.behavior, seg_b.behavior]))
            if pair != target_pair:
                continue

            if is_forbidden_transition(seg_a.behavior, seg_b.behavior):
                continue

            gap = seg_b.start_frame - seg_a.end_frame
            if gap > max_gap:
                continue

            transition_frame = seg_a.end_frame

            if include_turn_direction:
                from_turn = get_turn_direction_for_segment(
                    seg_a, turn_overlaps, transition_frame
                )
                to_turn = get_turn_direction_for_segment(
                    seg_b, turn_overlaps, transition_frame
                )
            else:
                from_turn = TurnDirection.STRAIGHT
                to_turn = TurnDirection.STRAIGHT

            len_a = seg_a.end_frame - seg_a.start_frame + 1
            len_b = seg_b.end_frame - seg_b.start_frame + 1

            if balance_segments:
                min_len = min(len_a, len_b)
                a_start = seg_a.end_frame - min_len + 1
                a_end = seg_a.end_frame
                b_start = seg_b.start_frame
                b_end = seg_b.start_frame + min_len - 1
            else:
                a_start = seg_a.start_frame
                a_end = seg_a.end_frame
                b_start = seg_b.start_frame
                b_end = seg_b.end_frame

            transitions.append(TransitionSegment(
                clip_id=clip_id,
                from_behavior=seg_a.behavior,
                to_behavior=seg_b.behavior,
                from_start=a_start,
                from_end=a_end,
                to_start=b_start,
                to_end=b_end,
                from_turn=from_turn,
                to_turn=to_turn,
            ))

    return transitions


def find_all_transitions(
    csv_path: str,
    short_threshold: int = 10,
    max_gap: int = 10,
    merge_rear: bool = True,
) -> list[TransitionSegment]:
    """
    Find all GT transitions in the dataset (not filtered by behavior pair).

    Returns:
        List of all TransitionSegment objects found
    """
    all_segments, turn_overlaps_by_clip = load_behavior_segments(csv_path, merge_rear)
    segments_by_clip = _group_segments_by_clip(all_segments)

    transitions = []

    for clip_id, clip_segments in segments_by_clip.items():
        merged_segments = _merge_short_intermediate_behaviors(
            clip_segments, short_threshold
        )

        turn_overlaps = turn_overlaps_by_clip.get(clip_id, [])

        for i in range(len(merged_segments) - 1):
            seg_a = merged_segments[i]
            seg_b = merged_segments[i + 1]

            if seg_a.behavior == seg_b.behavior:
                continue

            if is_forbidden_transition(seg_a.behavior, seg_b.behavior):
                continue

            gap = seg_b.start_frame - seg_a.end_frame
            if gap > max_gap:
                continue

            transition_frame = seg_a.end_frame

            from_turn = get_turn_direction_for_segment(
                seg_a, turn_overlaps, transition_frame
            )
            to_turn = get_turn_direction_for_segment(
                seg_b, turn_overlaps, transition_frame
            )

            transitions.append(TransitionSegment(
                clip_id=clip_id,
                from_behavior=seg_a.behavior,
                to_behavior=seg_b.behavior,
                from_start=seg_a.start_frame,
                from_end=seg_a.end_frame,
                to_start=seg_b.start_frame,
                to_end=seg_b.end_frame,
                from_turn=from_turn,
                to_turn=to_turn,
            ))

    return transitions


def load_behavior_segments_multilabel(
    csv_path: str,
    merge_rear: bool = True,
) -> dict[int, dict[int, set[str]]]:
    """
    Load behavior segments as frame-level multilabel data (legacy format).

    Returns:
        Dict mapping clip_id -> frame -> set of behavior labels
    """
    df = pd.read_csv(csv_path)
    clip_behaviors = defaultdict(lambda: defaultdict(set))

    for _, row in df.iterrows():
        start_frame = int(row["Start_Frame"])
        end_frame = int(row["End_Frame"])
        if start_frame == end_frame == 0:
            continue

        video = str(row["Video"])
        clip_id = int(video.replace("clip_", "").replace(".mp4", ""))
        behavior = normalize_behavior_name(str(row["Behavior"]))

        if merge_rear:
            behavior = merge_rear_category(behavior)

        for frame in range(start_frame, min(end_frame + 1, 500)):
            clip_behaviors[clip_id][frame].add(behavior)

    return {k: dict(v) for k, v in clip_behaviors.items()}


def find_gt_intersection_frames(
    clip_behaviors: dict[int, dict[int, set[str]]],
    behaviors: list[str],
    merge_rear: bool = True,
) -> list[tuple[int, int]]:
    """Find frames where all specified behaviors are labeled simultaneously."""
    if not behaviors:
        return []

    target_set = set()
    for b in behaviors:
        normalized = normalize_behavior_name(b)
        if merge_rear:
            normalized = merge_rear_category(normalized)
        target_set.add(normalized)

    intersections = []

    for clip_id, frame_map in clip_behaviors.items():
        for frame, frame_behaviors in frame_map.items():
            if target_set <= frame_behaviors:
                intersections.append((clip_id, frame))

    return intersections


def find_pure_behavior_segments(
    csv_path: str,
    target_behaviors: list[str],
    min_length: int = 30,
    merge_rear: bool = True,
) -> list[BehaviorSegment]:
    """
    Find segments of pure behaviors (not transitions).

    This is useful for non-transition behaviors like grooming that involve
    movement but are not behavioral transitions.

    Args:
        csv_path: Path to behavior_labels.csv
        target_behaviors: List of behavior names to extract (e.g., ["Grooming"])
        min_length: Minimum segment length in frames
        merge_rear: Whether to merge rear categories

    Returns:
        List of BehaviorSegment for the target behaviors
    """
    segments, _ = load_behavior_segments(csv_path, merge_rear=merge_rear)

    target_set = set()
    for b in target_behaviors:
        normalized = normalize_behavior_name(b)
        if merge_rear:
            normalized = merge_rear_category(normalized)
        target_set.add(normalized)

    pure_segments = []
    for seg in segments:
        if seg.behavior in target_set and seg.length >= min_length:
            pure_segments.append(seg)

    return pure_segments
