"""Analyze behavior transitions from labeled clips.

Identifies the most common behavior transitions, excluding Left Turn and Right Turn.
Outputs transition statistics and optionally renders example clips for top transitions.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

EXCLUDED_BEHAVIORS = {"Left Turn", "Right Turn"}


@dataclass
class BehaviorSegment:
    video: str
    behavior: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float


@dataclass
class Transition:
    from_behavior: str
    to_behavior: str
    video: str
    from_end_frame: int
    to_start_frame: int


@dataclass
class TransitionStats:
    transition_counts: Counter
    transitions_by_type: dict[tuple[str, str], list[Transition]]
    total_transitions: int
    unique_behaviors: set[str]


@dataclass
class CooccurrenceStats:
    turn_cooccurrence: dict[str, Counter]  # {"Left Turn": Counter, "Right Turn": Counter}
    turn_examples: dict[str, dict[str, list[tuple[str, int, int]]]]  # turn -> behavior -> [(video, start, end)]


def normalize_pair(a: str, b: str) -> tuple[str, str]:
    """Return canonical ordering for bidirectional transition."""
    return tuple(sorted([a, b]))


@dataclass
class AnalysisConfig:
    labels_path: str = "behavior_labels.csv"
    data_path: str = "data/latest_rodent.h5"
    output_dir: str = "out/transition_analysis"
    excluded_behaviors: set[str] = field(default_factory=lambda: EXCLUDED_BEHAVIORS.copy())
    min_gap_frames: int = 0
    max_gap_frames: int = 50
    top_k: int = 10


def load_config(config_path: str) -> AnalysisConfig:
    if not HAS_YAML:
        print("Warning: yaml not available, using defaults")
        return AnalysisConfig()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if "excluded_behaviors" in cfg:
        cfg["excluded_behaviors"] = set(cfg["excluded_behaviors"])
    return AnalysisConfig(**cfg)


def load_behavior_labels(labels_path: str) -> list[BehaviorSegment]:
    segments = []
    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            segments.append(BehaviorSegment(
                video=row["Video"],
                behavior=row["Behavior"],
                start_frame=int(row["Start_Frame"]),
                end_frame=int(row["End_Frame"]),
                start_time=float(row["Start_Time"]),
                end_time=float(row["End_Time"]),
            ))
    return segments


def filter_segments(
    segments: list[BehaviorSegment],
    excluded: set[str],
) -> list[BehaviorSegment]:
    filtered = []
    for seg in segments:
        if seg.behavior not in excluded:
            if seg.start_frame != seg.end_frame:
                filtered.append(seg)
    return filtered


def group_by_video(segments: list[BehaviorSegment]) -> dict[str, list[BehaviorSegment]]:
    groups = defaultdict(list)
    for seg in segments:
        groups[seg.video].append(seg)
    for video in groups:
        groups[video].sort(key=lambda x: x.start_frame)
    return groups


def merge_overlapping_segments(
    segments: list[BehaviorSegment],
) -> list[BehaviorSegment]:
    if not segments:
        return []

    merged = []
    current = segments[0]

    for seg in segments[1:]:
        if seg.behavior == current.behavior and seg.start_frame <= current.end_frame + 1:
            current = BehaviorSegment(
                video=current.video,
                behavior=current.behavior,
                start_frame=current.start_frame,
                end_frame=max(current.end_frame, seg.end_frame),
                start_time=current.start_time,
                end_time=max(current.end_time, seg.end_time),
            )
        else:
            merged.append(current)
            current = seg

    merged.append(current)
    return merged


def find_dominant_behavior_at_each_frame(
    segments: list[BehaviorSegment],
    max_frame: int,
) -> list[Optional[str]]:
    frame_behaviors: list[Optional[str]] = [None] * (max_frame + 1)
    for seg in segments:
        for f in range(seg.start_frame, min(seg.end_frame + 1, max_frame + 1)):
            frame_behaviors[f] = seg.behavior
    return frame_behaviors


def extract_transitions_from_video(
    segments: list[BehaviorSegment],
    video: str,
    min_gap: int,
    max_gap: int,
) -> list[Transition]:
    if len(segments) < 2:
        return []

    merged = merge_overlapping_segments(segments)
    if len(merged) < 2:
        return []

    transitions = []
    for i in range(len(merged) - 1):
        seg_a = merged[i]
        seg_b = merged[i + 1]

        if seg_a.behavior == seg_b.behavior:
            continue

        gap = seg_b.start_frame - seg_a.end_frame
        if min_gap <= gap <= max_gap:
            transitions.append(Transition(
                from_behavior=seg_a.behavior,
                to_behavior=seg_b.behavior,
                video=video,
                from_end_frame=seg_a.end_frame,
                to_start_frame=seg_b.start_frame,
            ))

    return transitions


def segments_overlap(seg_a: BehaviorSegment, seg_b: BehaviorSegment) -> bool:
    return seg_a.start_frame <= seg_b.end_frame and seg_b.start_frame <= seg_a.end_frame


def get_overlap_frames(seg_a: BehaviorSegment, seg_b: BehaviorSegment) -> tuple[int, int]:
    start = max(seg_a.start_frame, seg_b.start_frame)
    end = min(seg_a.end_frame, seg_b.end_frame)
    return start, end


def analyze_turn_cooccurrence(
    segments: list[BehaviorSegment],
) -> CooccurrenceStats:
    turn_behaviors = {"Left Turn", "Right Turn"}
    grouped = group_by_video(segments)

    cooccurrence: dict[str, Counter] = {
        "Left Turn": Counter(),
        "Right Turn": Counter(),
    }
    examples: dict[str, dict[str, list[tuple[str, int, int]]]] = {
        "Left Turn": defaultdict(list),
        "Right Turn": defaultdict(list),
    }

    for video, video_segments in grouped.items():
        turn_segs = [s for s in video_segments if s.behavior in turn_behaviors]
        other_segs = [s for s in video_segments if s.behavior not in turn_behaviors]

        for turn_seg in turn_segs:
            for other_seg in other_segs:
                if segments_overlap(turn_seg, other_seg):
                    start, end = get_overlap_frames(turn_seg, other_seg)
                    overlap_frames = end - start + 1
                    if overlap_frames > 0:
                        cooccurrence[turn_seg.behavior][other_seg.behavior] += overlap_frames
                        examples[turn_seg.behavior][other_seg.behavior].append(
                            (video, start, end)
                        )

    return CooccurrenceStats(
        turn_cooccurrence=cooccurrence,
        turn_examples={k: dict(v) for k, v in examples.items()},
    )


def analyze_transitions(
    segments: list[BehaviorSegment],
    cfg: AnalysisConfig,
) -> TransitionStats:
    filtered = filter_segments(segments, cfg.excluded_behaviors)
    grouped = group_by_video(filtered)

    all_transitions = []
    for video, video_segments in grouped.items():
        transitions = extract_transitions_from_video(
            video_segments, video, cfg.min_gap_frames, cfg.max_gap_frames
        )
        all_transitions.extend(transitions)

    transition_counts = Counter()
    transitions_by_type: dict[tuple[str, str], list[Transition]] = defaultdict(list)

    for t in all_transitions:
        key = normalize_pair(t.from_behavior, t.to_behavior)
        transition_counts[key] += 1
        transitions_by_type[key].append(t)

    unique_behaviors = set()
    for seg in filtered:
        unique_behaviors.add(seg.behavior)

    return TransitionStats(
        transition_counts=transition_counts,
        transitions_by_type=dict(transitions_by_type),
        total_transitions=len(all_transitions),
        unique_behaviors=unique_behaviors,
    )


def print_transition_matrix(stats: TransitionStats) -> None:
    behaviors = sorted(stats.unique_behaviors)
    print("\n=== Transition Matrix ===")
    header = "From \\ To".ljust(20) + "".join(b[:8].ljust(10) for b in behaviors)
    print(header)
    print("-" * len(header))

    for from_b in behaviors:
        row = from_b[:18].ljust(20)
        for to_b in behaviors:
            count = stats.transition_counts.get((from_b, to_b), 0)
            row += str(count).ljust(10)
        print(row)


def print_top_transitions(stats: TransitionStats, top_k: int) -> None:
    print(f"\n=== Top {top_k} Transitions (excluding turns, bidirectional) ===")
    for i, ((b1, b2), count) in enumerate(stats.transition_counts.most_common(top_k)):
        print(f"{i + 1:2d}. {b1:20s} <-> {b2:20s}: {count:3d} occurrences")
        examples = stats.transitions_by_type[(b1, b2)][:3]
        for ex in examples:
            print(f"      {ex.video} frames {ex.from_end_frame}-{ex.to_start_frame}")


def print_turn_cooccurrence(stats: CooccurrenceStats) -> None:
    print("\n=== Turn Co-occurrence Analysis ===")
    print("(Counts = overlapping frames between turn and other behavior)\n")

    for turn_type in ["Left Turn", "Right Turn"]:
        cooc = stats.turn_cooccurrence[turn_type]
        if not cooc:
            print(f"{turn_type}: No co-occurrences found")
            continue

        total_frames = sum(cooc.values())
        print(f"{turn_type} (total {total_frames} overlapping frames):")

        for behavior, frame_count in cooc.most_common():
            pct = 100 * frame_count / total_frames if total_frames > 0 else 0
            print(f"  {behavior:20s}: {frame_count:4d} frames ({pct:5.1f}%)")
            examples = stats.turn_examples[turn_type].get(behavior, [])[:2]
            for video, start, end in examples:
                print(f"      {video} frames {start}-{end}")
        print()


def save_results(stats: TransitionStats, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    transitions_path = os.path.join(output_dir, "transitions.csv")
    with open(transitions_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Behavior1", "Behavior2", "Count"])
        for (b1, b2), count in stats.transition_counts.most_common():
            writer.writerow([b1, b2, count])

    examples_path = os.path.join(output_dir, "transition_examples.csv")
    with open(examples_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["From", "To", "Video", "FromEndFrame", "ToStartFrame"])
        for (from_b, to_b), transitions in stats.transitions_by_type.items():
            for t in transitions:
                writer.writerow([from_b, to_b, t.video, t.from_end_frame, t.to_start_frame])

    print(f"\nResults saved to {output_dir}/")


def main(config_path: Optional[str] = None):
    if config_path and os.path.exists(config_path):
        cfg = load_config(config_path)
    else:
        cfg = AnalysisConfig()

    print("=== Behavior Transition Analysis ===")
    print(f"Labels: {cfg.labels_path}")
    print(f"Excluded behaviors: {cfg.excluded_behaviors}")
    print(f"Gap constraints: {cfg.min_gap_frames}-{cfg.max_gap_frames} frames")

    segments = load_behavior_labels(cfg.labels_path)
    print(f"\nLoaded {len(segments)} behavior segments")

    stats = analyze_transitions(segments, cfg)
    cooc_stats = analyze_turn_cooccurrence(segments)

    print(f"\n=== Summary ===")
    print(f"Total transitions found: {stats.total_transitions}")
    print(f"Unique behaviors (excl. turns): {sorted(stats.unique_behaviors)}")
    print(f"Unique transition types: {len(stats.transition_counts)}")

    print_top_transitions(stats, cfg.top_k)
    print_transition_matrix(stats)
    print_turn_cooccurrence(cooc_stats)

    save_results(stats, cfg.output_dir)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
