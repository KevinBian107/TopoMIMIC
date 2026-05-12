#!/usr/bin/env python3
"""
GT Transition Filmstrip Grid
=============================
Visualizes each directed GT transition type (e.g., "Rear -> Walking" !=
"Walking -> Rear") as a chronophotography filmstrip, then assembles them
into a single grid figure.

Usage:
    python demos/demo_transition_filmstrip.py
    python demos/demo_transition_filmstrip.py --config configs/demos/demo_transition_filmstrip.yaml
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import math
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, "/home/jovyan/vast/kaiwen/TopoVNL")

from topo_mimic.analysis.method_analysis.property.theta_filmstrip import (
    FilmstripConfig,
    FilmstripResult,
    compute_filmstrip_camera,
    render_fixed_camera_frames,
    _composite_frames_opaque,
)
from topo_mimic.analysis.feature_analysis.ground_truth import (
    find_all_transitions,
    TransitionSegment,
)
from topo_mimic.extraction.utils.load_utils import load_rollout_from_source


@dataclass
class DataConfig:
    rollout_path: str = "data/rodent_ar.h5"
    xml_path: str = "topo_mimic/extraction/xml/rodent.xml"


@dataclass
class Config:
    dataset: str = "rodent"
    output_dir: str = "outputs/demos/transition_filmstrip"
    behavior_labels_csv: str = "behavior_labels.csv"
    data: DataConfig = field(default_factory=DataConfig)
    filmstrip: FilmstripConfig = field(default_factory=FilmstripConfig)
    grid_columns: int = 4
    examples_per_type: int = 1
    short_threshold: int = 10
    max_gap: int = 10
    merge_rear: bool = True
    min_transition_frames: int = 20
    max_context_frames: int = 120
    output_format: str = "pdf"


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    data_cfg = DataConfig(**cfg_dict.get("data", {}))
    film_cfg = FilmstripConfig(**cfg_dict.get("filmstrip", {}))

    return Config(
        dataset=cfg_dict.get("dataset", "rodent"),
        output_dir=cfg_dict.get("output_dir", "outputs/demos/transition_filmstrip"),
        behavior_labels_csv=cfg_dict.get("behavior_labels_csv", "behavior_labels.csv"),
        data=data_cfg,
        filmstrip=film_cfg,
        grid_columns=cfg_dict.get("grid_columns", 4),
        examples_per_type=cfg_dict.get("examples_per_type", 1),
        short_threshold=cfg_dict.get("short_threshold", 10),
        max_gap=cfg_dict.get("max_gap", 10),
        merge_rear=cfg_dict.get("merge_rear", True),
        min_transition_frames=cfg_dict.get("min_transition_frames", 20),
        max_context_frames=cfg_dict.get("max_context_frames", 120),
        output_format=cfg_dict.get("output_format", "pdf"),
    )


def render_transition_filmstrip(
    qposes: np.ndarray,
    from_start: int,
    to_end: int,
    filmstrip_cfg: FilmstripConfig,
    clip_id: int,
    xml_path: str,
) -> FilmstripResult:
    """Render a chronophotography filmstrip for a transition frame range."""
    n_qpos = qposes.shape[0]
    frame_indices_all = [fi for fi in range(from_start, to_end + 1) if fi < n_qpos]

    if len(frame_indices_all) == 0:
        blank = np.full(
            (filmstrip_cfg.render_height, filmstrip_cfg.render_width, 3),
            255, dtype=np.uint8,
        )
        return FilmstripResult(
            clip_id=clip_id,
            path_index=0,
            composite_image=blank,
            n_poses_rendered=0,
            frame_indices=[],
        )

    n_poses = min(filmstrip_cfg.n_poses, len(frame_indices_all))
    sample_idx = np.linspace(0, len(frame_indices_all) - 1, n_poses).round().astype(int)
    frame_indices = [frame_indices_all[i] for i in sample_idx]

    lookat, distance, auto_azimuth = compute_filmstrip_camera(
        qposes,
        frame_indices,
        filmstrip_cfg.camera_elevation,
        filmstrip_cfg.camera_azimuth,
        filmstrip_cfg.camera_distance_margin,
    )
    elevation = filmstrip_cfg.camera_elevation

    frames = render_fixed_camera_frames(
        qposes,
        frame_indices,
        lookat,
        distance,
        elevation,
        auto_azimuth,
        filmstrip_cfg.render_height,
        filmstrip_cfg.render_width,
        xml_path,
    )

    if not frames:
        blank = np.full(
            (filmstrip_cfg.render_height, filmstrip_cfg.render_width, 3),
            255, dtype=np.uint8,
        )
        return FilmstripResult(
            clip_id=clip_id,
            path_index=0,
            composite_image=blank,
            n_poses_rendered=0,
            frame_indices=frame_indices,
        )

    composite = _composite_frames_opaque(frames, temporal_tint=True)

    return FilmstripResult(
        clip_id=clip_id,
        path_index=0,
        composite_image=composite,
        n_poses_rendered=len(frames),
        frame_indices=frame_indices,
    )


def create_transition_grid(
    filmstrips_by_type: dict[str, tuple[TransitionSegment, FilmstripResult]],
    transition_counts: dict[str, int],
    output_path: str,
    grid_columns: int = 4,
    dpi: int = 300,
) -> None:
    """Assemble one filmstrip per transition type into a Nature-style grid."""
    sorted_keys = sorted(filmstrips_by_type.keys())
    n_total = len(sorted_keys)
    n_cols = min(grid_columns, n_total)
    n_rows = (n_total + n_cols - 1) // n_cols

    cell_w, cell_h = 4.0, 2.8
    fig_w = n_cols * cell_w
    fig_h = n_rows * cell_h

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_axis_off()

    for idx, key in enumerate(sorted_keys):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        seg, fr = filmstrips_by_type[key]
        ax.imshow(fr.composite_image)
        count = transition_counts[key]
        parts = key.split(" -> ")
        if len(parts) == 2:
            label = f"{parts[0]} $\\rightarrow$ {parts[1]}"
        else:
            label = key
        ax.set_title(
            f"{label}  ($n$={count})",
            fontsize=16, fontfamily="sans-serif", fontweight="bold", pad=4,
        )

    fig.subplots_adjust(
        left=0.005, right=0.995, top=0.97, bottom=0.005,
        wspace=0.005, hspace=0.06,
    )
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved grid: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="GT Transition Filmstrip Grid")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    default_config_path = (
        Path(__file__).parent.parent / "configs" / "demos" / "demo_transition_filmstrip.yaml"
    )

    if args.config:
        cfg = load_config(args.config)
    elif default_config_path.exists():
        print(f"Loading config from {default_config_path}")
        cfg = load_config(str(default_config_path))
    else:
        cfg = Config()

    project_root = Path(__file__).parent.parent.resolve()

    if os.path.isabs(cfg.output_dir):
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = project_root / cfg.output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.isabs(cfg.behavior_labels_csv):
        cfg.behavior_labels_csv = str(project_root / cfg.behavior_labels_csv)
    if not os.path.isabs(cfg.data.rollout_path):
        cfg.data.rollout_path = str(project_root / cfg.data.rollout_path)
    if not os.path.isabs(cfg.data.xml_path):
        cfg.data.xml_path = str(project_root / cfg.data.xml_path)

    print("=" * 70)
    print(f"GT Transition Filmstrip Grid ({cfg.dataset})")
    print("=" * 70)
    print(f"  Output dir: {output_dir}")
    print(f"  CSV: {cfg.behavior_labels_csv}")
    print(f"  Min transition frames: {cfg.min_transition_frames}")
    print(f"  Filmstrip n_poses: {cfg.filmstrip.n_poses}")

    print("\nFinding all GT transitions...")
    all_transitions = find_all_transitions(
        cfg.behavior_labels_csv,
        short_threshold=cfg.short_threshold,
        max_gap=cfg.max_gap,
        merge_rear=cfg.merge_rear,
    )
    print(f"  Found {len(all_transitions)} total transition instances")

    grouped: dict[str, list[TransitionSegment]] = defaultdict(list)
    for seg in all_transitions:
        directed_key = f"{seg.from_behavior} -> {seg.to_behavior}"
        total_frames = (seg.from_end - seg.from_start + 1) + (seg.to_end - seg.to_start + 1)
        if total_frames >= cfg.min_transition_frames:
            grouped[directed_key].append(seg)

    print(f"  {len(grouped)} directed transition types (after min_frames filter)")
    for key in sorted(grouped.keys()):
        print(f"    {key}: {len(grouped[key])} instances")

    for key in grouped:
        grouped[key].sort(
            key=lambda s: abs(
                (s.from_end - s.from_start + 1) + (s.to_end - s.to_start + 1) - 80
            ),
        )

    qpos_cache: dict[int, np.ndarray] = {}
    filmstrips_by_type: dict[str, tuple[TransitionSegment, FilmstripResult]] = {}
    transition_counts: dict[str, int] = {k: len(v) for k, v in grouped.items()}

    sorted_keys = sorted(grouped.keys())
    for type_idx, key in enumerate(sorted_keys):
        seg = grouped[key][0]
        print(f"\n[{type_idx + 1}/{len(sorted_keys)}] {key} "
              f"(best of {len(grouped[key])})")

        if seg.clip_id not in qpos_cache:
            try:
                ro = load_rollout_from_source(seg.clip_id, cfg.data.rollout_path)
                qpos_cache[seg.clip_id] = ro["qpos"]
            except Exception as e:
                print(f"  ERROR loading clip {seg.clip_id}: {e}")
                continue

        qposes = qpos_cache[seg.clip_id]

        boundary = (seg.from_end + seg.to_start) // 2
        half_ctx = cfg.max_context_frames // 2
        render_start = max(seg.from_start, boundary - half_ctx)
        render_end = min(seg.to_end, boundary + half_ctx)
        if render_start >= render_end:
            render_start = seg.from_start
            render_end = seg.to_end
        print(f"  Clip {seg.clip_id} F{render_start}-{render_end} "
              f"(boundary ~{boundary})")

        fr = render_transition_filmstrip(
            qposes,
            render_start,
            render_end,
            cfg.filmstrip,
            seg.clip_id,
            cfg.data.xml_path,
        )
        filmstrips_by_type[key] = (seg, fr)
        print(f"  Rendered {fr.n_poses_rendered} poses")

    ext = cfg.output_format
    print(f"\nAssembling full grid...")
    grid_path = str(output_dir / f"grid.{ext}")
    create_transition_grid(
        filmstrips_by_type,
        transition_counts,
        grid_path,
        grid_columns=cfg.grid_columns,
    )

    print(f"\nOutput saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
