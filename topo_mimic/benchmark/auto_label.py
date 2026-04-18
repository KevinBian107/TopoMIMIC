from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class BehaviorSegment:
    clip: int
    behavior: str
    start: int  # inclusive
    end: int  # exclusive


def load_behavior_segments(csv_path: str) -> dict[int, list[BehaviorSegment]]:
    segments: dict[int, list[BehaviorSegment]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            seg = BehaviorSegment(
                clip=int(row["Clip"]),
                behavior=row["Behavior"],
                start=int(row["Start_Frame"]),
                end=int(row["End_Frame"]),
            )
            segments.setdefault(seg.clip, []).append(seg)
    for segs in segments.values():
        segs.sort(key=lambda s: s.start)
    return segments


def label_sample(
    clip_id: int,
    frame_indices: np.ndarray,
    clip_segments: list[BehaviorSegment],
) -> str:
    if not clip_segments or len(frame_indices) == 0:
        return ""

    fmin, fmax = int(frame_indices.min()), int(frame_indices.max())

    overlapping: list[BehaviorSegment] = []
    for seg in clip_segments:
        if seg.start <= fmax and seg.end > fmin:
            overlapping.append(seg)

    behaviors_in_order: list[str] = []
    for seg in overlapping:
        if not behaviors_in_order or behaviors_in_order[-1] != seg.behavior:
            behaviors_in_order.append(seg.behavior)

    if len(behaviors_in_order) >= 2:
        return f"{behaviors_in_order[0]} {behaviors_in_order[-1]}"

    if len(behaviors_in_order) == 1:
        has_gap = _has_transition_frames(fmin, fmax, clip_segments)
        if has_gap:
            the_seg = overlapping[0]
            adjacent = _find_adjacent_behavior(the_seg, clip_segments)
            if adjacent and adjacent != the_seg.behavior:
                if fmin < the_seg.start:
                    return f"{adjacent} {the_seg.behavior}"
                else:
                    return f"{the_seg.behavior} {adjacent}"
        return behaviors_in_order[0]

    before = [s for s in clip_segments if s.end <= fmin]
    after = [s for s in clip_segments if s.start > fmax]
    b_name = before[-1].behavior if before else "unknown"
    a_name = after[0].behavior if after else "unknown"
    if b_name != a_name:
        return f"{b_name} {a_name}"
    return b_name


def _has_transition_frames(
    fmin: int,
    fmax: int,
    clip_segments: list[BehaviorSegment],
) -> bool:
    covered = 0
    for seg in clip_segments:
        ov_start = max(fmin, seg.start)
        ov_end = min(fmax + 1, seg.end)
        if ov_end > ov_start:
            covered += ov_end - ov_start
    window_len = fmax - fmin + 1
    return covered < window_len


def _find_adjacent_behavior(
    seg: BehaviorSegment,
    all_segments: list[BehaviorSegment],
) -> str | None:
    idx = all_segments.index(seg)
    if idx + 1 < len(all_segments):
        return all_segments[idx + 1].behavior
    if idx - 1 >= 0:
        return all_segments[idx - 1].behavior
    return None


def auto_label_clusters(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    segments: dict[int, list[BehaviorSegment]],
) -> tuple[dict[int, str], dict[int, list[str]]]:
    cluster_labels: dict[int, str] = {}
    sample_labels: dict[int, list[str]] = {}

    for cluster_id, samples in sorted(cluster_dict.items()):
        labels = []
        for abs_clip_id, frames in samples:
            clip_segs = segments.get(abs_clip_id, [])
            lbl = label_sample(abs_clip_id, frames, clip_segs)
            labels.append(lbl)

        sample_labels[cluster_id] = labels
        if labels:
            majority = Counter(labels).most_common(1)[0][0]
            cluster_labels[cluster_id] = majority
        else:
            cluster_labels[cluster_id] = ""

    return cluster_labels, sample_labels


def generate_auto_labels_yaml(
    method_name: str,
    n_clusters: int,
    seed: int,
    output_dir: str,
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    behavior_labels_csv: str,
    experiment_id: str | None = None,
) -> str:
    if experiment_id is None:
        from datetime import datetime
        experiment_id = f"experiment_{datetime.now().strftime('%Y-%m-%d')}"

    segments = load_behavior_segments(behavior_labels_csv)
    cluster_labels, sample_labels = auto_label_clusters(cluster_dict, segments)

    all_labels = {i: cluster_labels.get(i, "") for i in range(n_clusters)}
    labels_content = {
        "experiment_id": experiment_id,
        "method": method_name,
        "n_clusters": n_clusters,
        "seed": seed,
        "labels": all_labels,
    }
    labels_path = Path(output_dir) / f"labels_seed_{seed}.yaml"
    with open(labels_path, "w") as f:
        f.write("# AUTO-GENERATED from ground truth behavior labels\n\n")
        yaml.dump(labels_content, f, default_flow_style=False, sort_keys=False)

    sample_detail: dict = {}
    for cluster_id in sorted(cluster_dict.keys()):
        samples = cluster_dict[cluster_id]
        s_labels = sample_labels.get(cluster_id, [])
        entries = []
        for i, (abs_clip_id, frames) in enumerate(samples):
            lbl = s_labels[i] if i < len(s_labels) else ""
            entries.append({
                "clip_id": int(abs_clip_id),
                "frame_start": int(frames.min()),
                "frame_end": int(frames.max()),
                "label": lbl,
            })
        sample_detail[int(cluster_id)] = {
            "majority_label": cluster_labels.get(cluster_id, ""),
            "n_samples": len(samples),
            "samples": entries,
        }

    sample_content = {
        "experiment_id": experiment_id,
        "method": method_name,
        "seed": seed,
        "clusters": sample_detail,
    }
    sample_path = Path(output_dir) / f"sample_labels_seed_{seed}.yaml"
    with open(sample_path, "w") as f:
        yaml.dump(sample_content, f, default_flow_style=False, sort_keys=False)

    return str(labels_path)
