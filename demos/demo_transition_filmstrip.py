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
    python demos/demo_transition_filmstrip.py --config configs/demos/demo_transition_filmstrip_walker.yaml
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import math
import argparse
import yaml
import h5py
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, "/home/jovyan/vast/kaiwen/TopoVNL")

from demos.demo_theta_filmstrip import (
    FilmstripConfig,
    FilmstripResult,
    compute_filmstrip_camera,
    render_fixed_camera_frames,
    _composite_frames_opaque,
)
from topo_mimic.benchmark.analysis.ground_truth import (
    find_all_transitions,
    TransitionSegment,
    TurnDirection,
)
from topo_mimic.dmwk.utils.load_utils import load_rollout_from_source


@dataclass
class DataConfig:
    rollout_path: str = "data/rodent_ar.h5"
    xml_path: str = "topo_mimic/dmwk/xml/rodent.xml"


@dataclass
class Config:
    dataset: str = "rodent"
    output_dir: str = "outputs/transition_filmstrip"
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
        output_dir=cfg_dict.get("output_dir", "outputs/transition_filmstrip"),
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


def find_walker_transitions(
    csv_path: str,
    n_frames: int = 500,
) -> list[TransitionSegment]:
    """Load walker transitions from the walker-format CSV (Clip, Behavior, Start_Frame, End_Frame)."""
    df = pd.read_csv(csv_path)
    clips_grouped = df.groupby("Clip")

    transitions = []
    for clip_id, group in clips_grouped:
        rows = group.sort_values("Start_Frame").reset_index(drop=True)
        if len(rows) < 2:
            continue

        for i in range(len(rows) - 1):
            from_row = rows.iloc[i]
            to_row = rows.iloc[i + 1]

            from_end = int(from_row["End_Frame"])
            if from_end == -1:
                from_end = n_frames - 1
            to_end = int(to_row["End_Frame"])
            if to_end == -1:
                to_end = n_frames - 1

            transitions.append(TransitionSegment(
                clip_id=int(clip_id),
                from_behavior=str(from_row["Behavior"]),
                to_behavior=str(to_row["Behavior"]),
                from_start=int(from_row["Start_Frame"]),
                from_end=from_end,
                to_start=int(to_row["Start_Frame"]),
                to_end=to_end,
                from_turn=TurnDirection.STRAIGHT,
                to_turn=TurnDirection.STRAIGHT,
            ))

    return transitions


def load_qpos_walker(clip_id: int, h5_path: str) -> np.ndarray:
    """Load qpos for a walker clip from the unified HDF5."""
    with h5py.File(h5_path, "r") as f:
        return f["qpos_rollout"][clip_id]


def compute_filmstrip_camera_walker(
    qposes: np.ndarray,
    frame_indices: list[int],
    distance_margin: float,
    render_width: int = 960,
    render_height: int = 720,
) -> tuple[np.ndarray, float, float]:
    """Camera for 2D planar walker. qpos layout: [rootz, rootx, rooty, ...].

    World coords: x = qpos[1], y = 0, z = qpos[0].
    Camera looks from -Y (azimuth=90) for a side view.
    Uses separate horizontal/vertical extent to respect render aspect ratio.
    """
    world_positions = np.zeros((len(frame_indices), 3))
    for i, fi in enumerate(frame_indices):
        world_positions[i, 0] = qposes[fi, 1]  # rootx -> world x
        world_positions[i, 2] = qposes[fi, 0]  # rootz -> world z

    bbox_min = world_positions.min(axis=0)
    bbox_max = world_positions.max(axis=0)

    walker_height = 1.6
    bbox_min[2] -= 0.1
    bbox_max[2] = bbox_min[2] + walker_height

    center = (bbox_min + bbox_max) / 2.0

    fovy_rad = math.radians(45.0)
    distance = (walker_height / 2.0) / math.tan(fovy_rad / 2.0) * distance_margin

    return center, distance, 90.0


def render_walker_strip(
    qposes: np.ndarray,
    frame_indices: list[int],
    filmstrip_cfg: FilmstripConfig,
    xml_path: str,
) -> np.ndarray:
    """Render each walker pose individually centered, crop, tint, and tile horizontally."""
    from demos.demo_theta_filmstrip import _load_mujoco_model, _ensure_rgb_uint8
    import mujoco

    view_height = 2.0
    fixed_lookat_z = 0.65
    fovy_rad = math.radians(45.0)
    distance = (view_height / 2.0) / math.tan(fovy_rad / 2.0) * 1.05

    mj_model, mj_data = _load_mujoco_model(xml_path)
    floor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    rh, rw = 720, 400
    rgb_renderer = mujoco.Renderer(mj_model, height=rh, width=rw)
    seg_renderer = mujoco.Renderer(mj_model, height=rh, width=rw)
    seg_renderer.enable_segmentation_rendering()

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = distance
    camera.elevation = -10.0
    camera.azimuth = 90.0

    n_poses = len(frame_indices)
    tint_start = np.array([0.68, 0.80, 0.88])
    tint_end = np.array([0.22, 0.38, 0.50])

    cropped_poses = []
    for pose_i, fi in enumerate(frame_indices):
        world_x = qposes[fi, 1]
        camera.lookat[:] = [world_x, 0.0, fixed_lookat_z]

        mj_data.qpos[:] = qposes[fi]
        mujoco.mj_forward(mj_model, mj_data)

        rgb_renderer.update_scene(mj_data, camera=camera)
        rgb = rgb_renderer.render().copy()

        seg_renderer.update_scene(mj_data, camera=camera)
        seg = seg_renderer.render()
        geom_ids = seg[:, :, 0].astype(np.int32)
        body_mask = (geom_ids >= 0) & (geom_ids != floor_id)

        rows_with_body = np.any(body_mask, axis=1)
        cols_with_body = np.any(body_mask, axis=0)
        if not np.any(rows_with_body):
            cropped_poses.append((np.full((100, 60, 3), 255, dtype=np.uint8),
                                  np.zeros((100, 60), dtype=bool)))
            continue

        r_min, r_max = np.where(rows_with_body)[0][[0, -1]]
        c_min, c_max = np.where(cols_with_body)[0][[0, -1]]
        pad = 4
        r_min = max(0, r_min - pad)
        r_max = min(rh - 1, r_max + pad)
        c_min = max(0, c_min - pad)
        c_max = min(rw - 1, c_max + pad)

        crop_rgb = rgb[r_min:r_max+1, c_min:c_max+1]
        crop_mask = body_mask[r_min:r_max+1, c_min:c_max+1]
        cropped_poses.append((crop_rgb, crop_mask))

    rgb_renderer.close()
    seg_renderer.close()

    max_h = max(c[0].shape[0] for c in cropped_poses)
    for i, (crop_rgb, crop_mask) in enumerate(cropped_poses):
        h, w = crop_rgb.shape[:2]
        if h < max_h:
            pad_top = (max_h - h) // 2
            pad_bot = max_h - h - pad_top
            crop_rgb = np.pad(crop_rgb, ((pad_top, pad_bot), (0, 0), (0, 0)),
                              mode='constant', constant_values=255)
            crop_mask = np.pad(crop_mask, ((pad_top, pad_bot), (0, 0)),
                               mode='constant', constant_values=False)
            cropped_poses[i] = (crop_rgb, crop_mask)

    total_w = sum(c[0].shape[1] for c in cropped_poses)
    avg_w = total_w // n_poses
    gap = max(avg_w // 5, 6)
    strip_w = total_w + gap * (n_poses - 1)
    strip_h = max_h

    composite = np.full((strip_h, strip_w, 3), 255, dtype=np.uint8)
    x_cursor = 0
    for pose_i, (crop_rgb, crop_mask) in enumerate(cropped_poses):
        h, w = crop_rgb.shape[:2]
        t = pose_i / max(n_poses - 1, 1)
        tint = tint_start * (1 - t) + tint_end * t

        body_pixels = crop_rgb[crop_mask].astype(np.float32) / 255.0
        luminance = 1.0 - np.mean(body_pixels, axis=-1, keepdims=True)
        tint_rgb = tint[None, :]
        colored = luminance * tint_rgb + (1.0 - luminance) * 1.0
        tinted_crop = np.full_like(crop_rgb, 255)
        tinted_crop[crop_mask] = np.clip(colored * 255.0, 0, 255).astype(np.uint8)

        composite[:h, x_cursor:x_cursor+w] = tinted_crop
        x_cursor += w + gap

    out_h, out_w = filmstrip_cfg.render_height, filmstrip_cfg.render_width
    from PIL import Image
    pil_img = Image.fromarray(composite)
    scale = min(out_w / composite.shape[1], out_h / composite.shape[0])
    new_w = int(composite.shape[1] * scale)
    new_h = int(composite.shape[0] * scale)
    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

    final = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
    y_off = (out_h - new_h) // 2
    x_off = (out_w - new_w) // 2
    final[y_off:y_off+new_h, x_off:x_off+new_w] = np.array(pil_img)

    return final


def render_transition_filmstrip(
    qposes: np.ndarray,
    from_start: int,
    to_end: int,
    filmstrip_cfg: FilmstripConfig,
    clip_id: int,
    xml_path: str,
    is_walker: bool = False,
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

    if is_walker:
        composite = render_walker_strip(
            qposes,
            frame_indices,
            filmstrip_cfg,
            xml_path,
        )
    else:
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

    n_rendered = len(frame_indices) if is_walker else len(frames)
    return FilmstripResult(
        clip_id=clip_id,
        path_index=0,
        composite_image=composite,
        n_poses_rendered=n_rendered,
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
            fontsize=9, fontfamily="sans-serif", pad=2,
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

    is_walker = cfg.dataset == "walker"

    print("=" * 70)
    print(f"GT Transition Filmstrip Grid ({cfg.dataset})")
    print("=" * 70)
    print(f"  Output dir: {output_dir}")
    print(f"  CSV: {cfg.behavior_labels_csv}")
    print(f"  Min transition frames: {cfg.min_transition_frames}")
    print(f"  Filmstrip n_poses: {cfg.filmstrip.n_poses}")

    print("\nFinding all GT transitions...")
    if is_walker:
        all_transitions = find_walker_transitions(cfg.behavior_labels_csv)
    else:
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
                if is_walker:
                    qpos_cache[seg.clip_id] = load_qpos_walker(
                        seg.clip_id, cfg.data.rollout_path,
                    )
                else:
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
            is_walker=is_walker,
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
