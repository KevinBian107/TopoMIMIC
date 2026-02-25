"""Augment behavior_labels.csv with synthetic "Step" labels.

For every Immobility -> Walk transition, creates a Step segment spanning
the last N immobility frames + first M walk frames around the boundary.
Optionally renders a grid video of step examples.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import csv
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from scripts.analyze_transitions import (
    BehaviorSegment,
    filter_segments,
    group_by_video,
    load_behavior_labels,
    merge_overlapping_segments,
)


@dataclass
class RenderingConfig:
    enabled: bool = True
    rodent_rollout_dir: str = "data/rodent_ar.h5"
    max_samples: int = 10
    output_path: str = "outputs/step_augmentation/step_grid.mp4"
    framerate: int = 50
    target_size: list[int] = field(default_factory=lambda: [320, 240])


@dataclass
class AugmentConfig:
    behavior_labels_csv: str = "behavior_labels.csv"
    n_immobility_frames: int = 20
    n_walk_frames: int = 10
    backup: bool = True
    rendering: RenderingConfig = field(default_factory=RenderingConfig)

    def __post_init__(self):
        if isinstance(self.rendering, dict):
            self.rendering = RenderingConfig(**self.rendering)


def load_config(config_path: str) -> AugmentConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return AugmentConfig(**cfg)


def parse_clip_id(video_name: str) -> int | None:
    m = re.search(r"clip_(\d+)", video_name)
    return int(m.group(1)) if m else None


def find_immobility_walk_transitions(
    segments: list[BehaviorSegment],
) -> list[tuple[BehaviorSegment, BehaviorSegment]]:
    excluded = {"Left Turn", "Right Turn"}
    filtered = filter_segments(segments, excluded)
    grouped = group_by_video(filtered)

    transitions = []
    for video, video_segments in grouped.items():
        merged = merge_overlapping_segments(video_segments)
        for i in range(len(merged) - 1):
            seg_a = merged[i]
            seg_b = merged[i + 1]
            if seg_a.behavior == "Immobility" and seg_b.behavior == "Walking":
                transitions.append((seg_a, seg_b))
    return transitions


def create_step_segments(
    transitions: list[tuple[BehaviorSegment, BehaviorSegment]],
    n_immobility: int,
    n_walk: int,
    fps: float = 30.0,
) -> list[BehaviorSegment]:
    steps = []
    for immob_seg, walk_seg in transitions:
        walk_start = walk_seg.start_frame
        step_start = max(immob_seg.start_frame, walk_start - n_immobility)
        step_end = min(walk_seg.end_frame, walk_start + n_walk)

        if step_end <= step_start:
            continue

        steps.append(BehaviorSegment(
            video=immob_seg.video,
            behavior="Step",
            start_frame=step_start,
            end_frame=step_end,
            start_time=step_start / fps,
            end_time=step_end / fps,
        ))
    return steps


def write_csv(
    original_segments: list[BehaviorSegment],
    step_segments: list[BehaviorSegment],
    output_path: str,
) -> None:
    all_segments = original_segments + step_segments
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Video", "Behavior", "Start_Time", "End_Time", "Start_Frame", "End_Frame"])
        for seg in all_segments:
            writer.writerow([
                seg.video,
                seg.behavior,
                seg.start_time,
                seg.end_time,
                seg.start_frame,
                seg.end_frame,
            ])


def render_step_grid(
    step_segments: list[BehaviorSegment],
    cfg: RenderingConfig,
) -> None:
    import numpy as np
    from topo_mimic.dmwk.skeletonization.postprocess import chunk_loops_by_clustered_vpath
    from topo_mimic.dmwk.skeletonization.rendering import render_grid_clusters_side_by_side

    samples = step_segments[:cfg.max_samples]
    cluster_dict: dict[int, list[tuple[int, "np.ndarray"]]] = {0: []}
    for seg in samples:
        clip_id = parse_clip_id(seg.video)
        if clip_id is None:
            continue
        frames = np.arange(seg.start_frame, seg.end_frame + 1)
        cluster_dict[0].append((clip_id, frames))

    if not cluster_dict[0]:
        print("No valid step segments to render.")
        return

    print(f"Rendering {len(cluster_dict[0])} step examples...")
    cluster_chunks, cluster_clip_ids = chunk_loops_by_clustered_vpath(
        cluster_dict,
        cfg.rodent_rollout_dir,
        pad=0,
        return_clip_ids=True,
    )

    render_grid_clusters_side_by_side(
        cluster_chunks,
        out_path=cfg.output_path,
        framerate=cfg.framerate,
        target_size=tuple(cfg.target_size),
        cluster_labels={0: "Step"},
        cluster_clip_ids=cluster_clip_ids,
    )
    print(f"Grid video saved to {cfg.output_path}")


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    print("=== Step Label Augmentation ===")
    print(f"CSV: {cfg.behavior_labels_csv}")
    print(f"Immobility frames before walk onset: {cfg.n_immobility_frames}")
    print(f"Walk frames after onset: {cfg.n_walk_frames}")

    segments = load_behavior_labels(cfg.behavior_labels_csv)
    print(f"Loaded {len(segments)} behavior segments")

    transitions = find_immobility_walk_transitions(segments)
    print(f"Found {len(transitions)} Immobility -> Walk transitions")

    step_segments = create_step_segments(
        transitions,
        n_immobility=cfg.n_immobility_frames,
        n_walk=cfg.n_walk_frames,
    )
    print(f"Created {len(step_segments)} Step segments")

    if not step_segments:
        print("No step segments created. Exiting.")
        return

    for seg in step_segments[:5]:
        print(f"  {seg.video}: frames {seg.start_frame}-{seg.end_frame}")
    if len(step_segments) > 5:
        print(f"  ... and {len(step_segments) - 5} more")

    if cfg.backup:
        backup_path = Path(cfg.behavior_labels_csv).with_name(
            Path(cfg.behavior_labels_csv).stem + "_original" + Path(cfg.behavior_labels_csv).suffix
        )
        shutil.copy2(cfg.behavior_labels_csv, backup_path)
        print(f"Backed up original to {backup_path}")

    write_csv(segments, step_segments, cfg.behavior_labels_csv)
    print(f"Written augmented CSV to {cfg.behavior_labels_csv}")

    if cfg.rendering.enabled:
        render_step_grid(step_segments, cfg.rendering)

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.augment_step_label <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
