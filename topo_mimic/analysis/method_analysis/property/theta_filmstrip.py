#!/usr/bin/env python3
"""
Theta-Graph Filmstrip Demo
============================
Discovers clips with theta-graph topology (two edge-disjoint paths between
density minima), checks that both paths share the same start/end behaviors
in intention space, and produces a paper-ready figure: cosine similarity
matrix on the left, chronophotography filmstrips on the right (one per path).

Usage:
    python demos/demo_theta_filmstrip.py
    python demos/demo_theta_filmstrip.py --config configs/demos/demo_theta_filmstrip.yaml
    python demos/demo_theta_filmstrip.py --top-k 3
    python demos/demo_theta_filmstrip.py --clip-ids 42 99
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import argparse
import math
import shutil
import yaml
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from topo_mimic.paths import DMPCD_ROOT
sys.path.append(str(DMPCD_ROOT))

import mujoco

from topo_mimic.analysis.method_analysis.property.cycle_asymmetry import (
    load_single_clip,
    run_dm_extraction,
    order_cycle_vertices,
    DataConfig,
    FeatureSpaceConfig,
    DMConfig,
    VisualizationConfig,
    DMResult,
)
from topo_mimic.analysis.method_analysis.property.transition_topology import (
    discover_rear_walk_rear_clips,
    find_dm_minima,
    analyze_edge_connectivity,
    compute_endpoint_similarity,
    ClipDiscoveryConfig,
    ConnectivityResult,
    EndpointSimilarityResult,
    RearWalkRearHit,
)
from topo_mimic.extraction.skeletonization.remap import (
    parse_swr_remap,
    remap_cycle_to_original,
    split_cycle_into_segments,
)
from topo_mimic.extraction.utils.load_utils import get_aggregate_data, load_rollout_from_source
from topo_mimic.extraction.skeletonization.rendering import _ensure_rgb_uint8


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


PATH_COLORS = ["#e6194b", "#4363d8", "#3cb44b", "#f58231"]
SEGMENT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990", "#dcbeff", "#9A6324",
]


@dataclass
class FilmstripConfig:
    n_poses: int = 5
    camera_elevation: float = -55.0
    camera_azimuth: float = 90.0
    camera_distance_margin: float = 2.5
    render_height: int = 720
    render_width: int = 1280
    similarity_threshold: float = 0.5
    gap_threshold: int = 10
    min_segment_length: int = 30


@dataclass
class Config:
    output_dir: str = "outputs/analysis/method_analysis/theta_filmstrip"
    data: DataConfig = field(default_factory=DataConfig)
    feature_spaces: list[FeatureSpaceConfig] = field(default_factory=list)
    clip_ids: list[int] = field(default_factory=list)
    dm: DMConfig = field(default_factory=DMConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    clip_discovery: ClipDiscoveryConfig = field(default_factory=ClipDiscoveryConfig)
    filmstrip: FilmstripConfig = field(default_factory=FilmstripConfig)


@dataclass
class FilmstripResult:
    clip_id: int
    path_index: int
    composite_image: np.ndarray
    n_poses_rendered: int
    frame_indices: list[int]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    data_cfg = DataConfig(**cfg_dict.get("data", {}))

    feature_spaces = []
    for fs_dict in cfg_dict.get("feature_spaces", []):
        feature_spaces.append(FeatureSpaceConfig(**fs_dict))

    dm_cfg = DMConfig(**cfg_dict.get("dm", {}))
    vis_cfg = VisualizationConfig(**cfg_dict.get("visualization", {}))
    disc_cfg = ClipDiscoveryConfig(**cfg_dict.get("clip_discovery", {}))
    film_cfg = FilmstripConfig(**cfg_dict.get("filmstrip", {}))

    return Config(
        output_dir=cfg_dict.get("output_dir", "outputs/analysis/method_analysis/theta_filmstrip"),
        data=data_cfg,
        feature_spaces=feature_spaces,
        clip_ids=cfg_dict.get("clip_ids", []),
        dm=dm_cfg,
        visualization=vis_cfg,
        clip_discovery=disc_cfg,
        filmstrip=film_cfg,
    )


# ---------------------------------------------------------------------------
# Function 1: check_theta_same_endpoints
# ---------------------------------------------------------------------------


def check_theta_same_endpoints(
    segments: list[np.ndarray],
    intention_data: np.ndarray,
    takens_m: int,
    takens_tau: int,
    similarity_threshold: float,
) -> tuple[bool, float]:
    """Check that cycle segments share the same start/end behaviors.

    1 segment:  cos(seg.start, seg.end) >= threshold
                (the single loop returns to where it started)
    2 segments: cos(seg1.start, seg2.end) >= threshold  AND
                cos(seg1.end,   seg2.start) >= threshold
                (one goes A->B, the other B->A in time)
    """
    if len(segments) not in (1, 2):
        return False, 0.0

    offset = (takens_m - 1) * takens_tau
    n_frames = intention_data.shape[0]

    def _vec(seg_idx: int, pos: int) -> np.ndarray | None:
        fi = int(segments[seg_idx][pos]) + offset
        if fi >= n_frames:
            return None
        return intention_data[fi].reshape(1, -1)

    if len(segments) == 1:
        if len(segments[0]) < 2:
            return False, 0.0
        v_start, v_end = _vec(0, 0), _vec(0, -1)
        if v_start is None or v_end is None:
            return False, 0.0
        endpoint_sim = float(cosine_similarity(v_start, v_end)[0, 0])
    else:
        if len(segments[0]) < 2 or len(segments[1]) < 2:
            return False, 0.0
        s1s, s1e = _vec(0, 0), _vec(0, -1)
        s2s, s2e = _vec(1, 0), _vec(1, -1)
        if any(v is None for v in [s1s, s1e, s2s, s2e]):
            return False, 0.0
        sim_cross1 = float(cosine_similarity(s1s, s2e)[0, 0])
        sim_cross2 = float(cosine_similarity(s1e, s2s)[0, 0])
        endpoint_sim = min(sim_cross1, sim_cross2)

    return endpoint_sim >= similarity_threshold, endpoint_sim


# ---------------------------------------------------------------------------
# Function 2: compute_filmstrip_camera
# ---------------------------------------------------------------------------


def compute_filmstrip_camera(
    qposes: np.ndarray,
    frame_indices: list[int],
    elevation: float,
    azimuth: float,
    distance_margin: float,
    vertical_padding: float = 0.12,
) -> tuple[np.ndarray, float, float]:
    """Compute a fixed camera position that frames all poses in the filmstrip.

    Returns (lookat, distance, azimuth) where azimuth is auto-computed
    perpendicular to the trajectory direction for best side-view.
    """
    root_positions = qposes[frame_indices, 0:3]

    bbox_min = root_positions.min(axis=0)
    bbox_max = root_positions.max(axis=0)

    bbox_max[2] += vertical_padding
    bbox_min[2] -= vertical_padding * 0.25

    center = (bbox_min + bbox_max) / 2.0

    extent = np.linalg.norm(bbox_max - bbox_min)
    fovy_rad = math.radians(45.0)
    distance = max((extent / 2.0) / math.tan(fovy_rad / 2.0) * distance_margin, 0.5)

    if len(root_positions) >= 2:
        xy_diff = root_positions[-1, :2] - root_positions[0, :2]
        traj_angle = math.degrees(math.atan2(xy_diff[1], xy_diff[0]))
        auto_azimuth = traj_angle + 90.0
    else:
        auto_azimuth = azimuth

    return center, distance, auto_azimuth


# ---------------------------------------------------------------------------
# Function 3: render_fixed_camera_frames
# ---------------------------------------------------------------------------


def _load_mujoco_model(
    xml_path: str,
    offscreen_width: int = 1920,
    offscreen_height: int = 1080,
) -> tuple:
    spec = mujoco.MjSpec()
    spec = spec.from_file(xml_path)
    for geom in spec.geoms:
        if geom.size is not None:
            geom.size *= 0.95
        if geom.pos is not None:
            geom.pos *= 0.95
    vis_global = getattr(spec.visual, "global_", None) or getattr(spec.visual, "global")
    vis_global.offwidth = offscreen_width
    vis_global.offheight = offscreen_height
    mj_model = spec.compile()
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_kinematics(mj_model, mj_data)
    return mj_model, mj_data


def _render_poses_individually(
    qposes: np.ndarray,
    frame_indices: list[int],
    xml_path: str,
    camera_elevation: float,
    camera_azimuth: float,
    camera_distance: float,
    camera_lookat_fn,
    render_h: int = 720,
    render_w: int = 480,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Render each pose individually centered, return list of (cropped_rgb, body_mask)."""
    import mujoco

    mj_model, mj_data = _load_mujoco_model(xml_path)
    floor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    rgb_renderer = mujoco.Renderer(mj_model, height=render_h, width=render_w)
    seg_renderer = mujoco.Renderer(mj_model, height=render_h, width=render_w)
    seg_renderer.enable_segmentation_rendering()

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = camera_distance
    camera.elevation = camera_elevation
    camera.azimuth = camera_azimuth

    cropped_poses = []
    for fi in frame_indices:
        camera.lookat[:] = camera_lookat_fn(qposes[fi])

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
            cropped_poses.append((
                np.full((100, 60, 3), 255, dtype=np.uint8),
                np.zeros((100, 60), dtype=bool),
            ))
            continue

        r_min, r_max = np.where(rows_with_body)[0][[0, -1]]
        c_min, c_max = np.where(cols_with_body)[0][[0, -1]]
        pad = 6
        r_min = max(0, r_min - pad)
        r_max = min(render_h - 1, r_max + pad)
        c_min = max(0, c_min - pad)
        c_max = min(render_w - 1, c_max + pad)

        crop_rgb = rgb[r_min:r_max + 1, c_min:c_max + 1]
        crop_mask = body_mask[r_min:r_max + 1, c_min:c_max + 1]
        cropped_poses.append((crop_rgb, crop_mask))

    rgb_renderer.close()
    seg_renderer.close()
    return cropped_poses


def _tile_cropped_poses(
    cropped_poses: list[tuple[np.ndarray, np.ndarray]],
    gap_fraction: float = 0.3,
) -> np.ndarray:
    """Tint and tile cropped poses into a tight strip with no padding."""
    n_poses = len(cropped_poses)
    tint_start = np.array([0.68, 0.80, 0.88])
    tint_end = np.array([0.22, 0.38, 0.50])

    max_h = max(c[0].shape[0] for c in cropped_poses)
    normalized = []
    for crop_rgb, crop_mask in cropped_poses:
        h, w = crop_rgb.shape[:2]
        if h < max_h:
            pad_top = (max_h - h) // 2
            pad_bot = max_h - h - pad_top
            crop_rgb = np.pad(crop_rgb, ((pad_top, pad_bot), (0, 0), (0, 0)),
                              mode='constant', constant_values=255)
            crop_mask = np.pad(crop_mask, ((pad_top, pad_bot), (0, 0)),
                               mode='constant', constant_values=False)
        normalized.append((crop_rgb, crop_mask))

    avg_w = sum(c[0].shape[1] for c in normalized) // n_poses
    gap = max(int(avg_w * gap_fraction), 4)
    total_w = sum(c[0].shape[1] for c in normalized) + gap * (n_poses - 1)

    composite = np.full((max_h, total_w, 3), 255, dtype=np.uint8)
    x_cursor = 0
    for pose_i, (crop_rgb, crop_mask) in enumerate(normalized):
        h, w = crop_rgb.shape[:2]
        t = pose_i / max(n_poses - 1, 1)
        tint = tint_start * (1 - t) + tint_end * t

        body_pixels = crop_rgb[crop_mask].astype(np.float32) / 255.0
        luminance = 1.0 - np.mean(body_pixels, axis=-1, keepdims=True)
        colored = luminance * tint[None, :] + (1.0 - luminance) * 1.0
        tinted_crop = np.full_like(crop_rgb, 255)
        tinted_crop[crop_mask] = np.clip(colored * 255.0, 0, 255).astype(np.uint8)

        composite[:h, x_cursor:x_cursor + w] = tinted_crop
        x_cursor += w + gap

    return composite[:, :x_cursor - gap]


def render_fixed_camera_frames(
    qposes: np.ndarray,
    frame_indices: list[int],
    camera_lookat: np.ndarray,
    camera_distance: float,
    camera_elevation: float,
    camera_azimuth: float,
    render_height: int,
    render_width: int,
    xml_path: str,
) -> list[np.ndarray]:
    """Render frames on a white background using segmentation masking."""
    mj_model, mj_data = _load_mujoco_model(xml_path)

    floor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    rgb_renderer = mujoco.Renderer(mj_model, height=render_height, width=render_width)
    seg_renderer = mujoco.Renderer(mj_model, height=render_height, width=render_width)
    seg_renderer.enable_segmentation_rendering()

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = camera_lookat
    camera.distance = camera_distance
    camera.elevation = camera_elevation
    camera.azimuth = camera_azimuth

    frames = []
    for fi in frame_indices:
        mj_data.qpos[:] = qposes[fi]
        mujoco.mj_forward(mj_model, mj_data)

        rgb_renderer.update_scene(mj_data, camera=camera)
        rgb = rgb_renderer.render().copy()

        seg_renderer.update_scene(mj_data, camera=camera)
        seg = seg_renderer.render()
        geom_ids = seg[:, :, 0].astype(np.int32)

        body_mask = (geom_ids >= 0) & (geom_ids != floor_id)

        white = np.full_like(rgb, 255)
        white[body_mask] = rgb[body_mask]
        frames.append(_ensure_rgb_uint8(white))

    rgb_renderer.close()
    seg_renderer.close()
    return frames


# ---------------------------------------------------------------------------
# Helpers: segment selection + opaque compositing
# ---------------------------------------------------------------------------


def _select_longest_continuous_segment(
    indices: np.ndarray,
    gap_threshold: int,
) -> np.ndarray:
    """Split sorted indices by gap_threshold, return the longest continuous segment."""
    if len(indices) < 2:
        return indices
    sorted_idx = np.sort(indices)
    diffs = np.diff(sorted_idx)
    split_points = np.where(diffs > gap_threshold)[0] + 1
    segments = np.split(sorted_idx, split_points)
    segments = [s for s in segments if len(s) > 0]
    if not segments:
        return indices
    return max(segments, key=len)


def _composite_frames_opaque(
    frames: list[np.ndarray],
    temporal_tint: bool = False,
    tint_start: tuple[float, float, float] = (0.68, 0.80, 0.88),
    tint_end: tuple[float, float, float] = (0.22, 0.38, 0.50),
) -> np.ndarray:
    """Painter-style compositing with optional temporal color gradient.

    When temporal_tint=True, each pose is tinted along a light-to-dark
    color ramp (Nature chronophotography style). Earlier poses are lighter,
    later poses are darker, making the time direction visible even when
    poses overlap.
    """
    if not frames:
        return np.full((100, 100, 3), 255, dtype=np.uint8)

    ref = frames[0]
    border = np.concatenate([
        ref[:5, :5].reshape(-1, 3),
        ref[:5, -5:].reshape(-1, 3),
        ref[-5:, :5].reshape(-1, 3),
        ref[-5:, -5:].reshape(-1, 3),
    ])
    bg_color = np.median(border, axis=0).astype(np.float32)

    h, w = ref.shape[:2]
    composite = np.full((h, w, 3), 255, dtype=np.uint8)

    n = len(frames)
    for i, frame in enumerate(frames):
        diff = np.max(np.abs(frame.astype(np.float32) - bg_color), axis=-1)
        mask = diff > 30

        if temporal_tint and n > 1:
            t = i / (n - 1)
            tint = np.array([
                tint_start[c] * (1 - t) + tint_end[c] * t for c in range(3)
            ], dtype=np.float32)
            body_pixels = frame[mask].astype(np.float32) / 255.0
            luminance = 1.0 - np.mean(body_pixels, axis=-1, keepdims=True)
            tint_rgb = tint[None, :]
            colored = luminance * tint_rgb + (1.0 - luminance) * 1.0
            composite[mask] = np.clip(colored * 255.0, 0, 255).astype(np.uint8)
        else:
            composite[mask] = frame[mask]

    return composite


# ---------------------------------------------------------------------------
# Function 4: create_filmstrip_composite
# ---------------------------------------------------------------------------


def create_filmstrip_composite(
    qposes: np.ndarray,
    takens_indices: np.ndarray,
    takens_m: int,
    takens_tau: int,
    filmstrip_cfg: FilmstripConfig,
    path_index: int,
    clip_id: int,
    xml_path: str,
) -> FilmstripResult:
    """Create a chronophotography filmstrip composite for one path."""
    continuous = _select_longest_continuous_segment(
        np.asarray(takens_indices), filmstrip_cfg.gap_threshold,
    )

    offset = (takens_m - 1) * takens_tau
    frame_indices_all = [int(idx) + offset for idx in continuous]
    n_qpos = qposes.shape[0]
    frame_indices_all = [fi for fi in frame_indices_all if fi < n_qpos]

    if len(frame_indices_all) == 0:
        blank = np.full((filmstrip_cfg.render_height, filmstrip_cfg.render_width, 3), 255, dtype=np.uint8)
        return FilmstripResult(
            clip_id=clip_id, path_index=path_index,
            composite_image=blank, n_poses_rendered=0, frame_indices=[],
        )

    n_poses = min(filmstrip_cfg.n_poses, len(frame_indices_all))
    sample_idx = np.linspace(0, len(frame_indices_all) - 1, n_poses).round().astype(int)
    frame_indices = [frame_indices_all[i] for i in sample_idx]

    lookat, distance, auto_azimuth = compute_filmstrip_camera(
        qposes, frame_indices,
        filmstrip_cfg.camera_elevation,
        filmstrip_cfg.camera_azimuth,
        filmstrip_cfg.camera_distance_margin,
    )

    frames = render_fixed_camera_frames(
        qposes, frame_indices,
        lookat, distance,
        filmstrip_cfg.camera_elevation,
        auto_azimuth,
        filmstrip_cfg.render_height,
        filmstrip_cfg.render_width,
        xml_path,
    )

    if not frames:
        blank = np.full((filmstrip_cfg.render_height, filmstrip_cfg.render_width, 3), 255, dtype=np.uint8)
        return FilmstripResult(
            clip_id=clip_id, path_index=path_index,
            composite_image=blank, n_poses_rendered=0, frame_indices=frame_indices,
        )

    composite = _composite_frames_opaque(frames, temporal_tint=True)

    return FilmstripResult(
        clip_id=clip_id,
        path_index=path_index,
        composite_image=composite,
        n_poses_rendered=len(frames),
        frame_indices=frame_indices,
    )


# ---------------------------------------------------------------------------
# 3D PCA drawing helpers
# ---------------------------------------------------------------------------


def _draw_3d_paths_ax(
    ax: plt.Axes,
    pts_3d: np.ndarray,
    paths_remapped: list[np.ndarray],
    min1_idx: int | None,
    min2_idx: int | None,
    elev: float,
    azim: float,
) -> None:
    temporal_colors = np.arange(len(pts_3d)) / max(len(pts_3d) - 1, 1)
    ax.scatter(
        pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2],
        c=temporal_colors, cmap="viridis", s=4, alpha=0.3, edgecolors="none",
    )
    for i, path_idx in enumerate(paths_remapped):
        valid = [idx for idx in path_idx if idx < len(pts_3d)]
        if len(valid) > 1:
            p = pts_3d[valid]
            color = PATH_COLORS[i % len(PATH_COLORS)]
            ax.plot(p[:, 0], p[:, 1], p[:, 2], c=color, linewidth=2.5, alpha=0.9,
                    label=f"Path {i+1} (L={len(valid)})")
    for min_idx in [min1_idx, min2_idx]:
        if min_idx is not None and min_idx < len(pts_3d):
            p = pts_3d[min_idx]
            ax.scatter([p[0]], [p[1]], [p[2]], c="gold", s=200, marker="*", zorder=200)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def _draw_3d_cycle_ax(
    ax: plt.Axes,
    pts_3d: np.ndarray,
    cycle_indices: np.ndarray | None,
    segments: list[np.ndarray],
    takens_points: np.ndarray,
    min1_idx: int | None,
    min2_idx: int | None,
    elev: float,
    azim: float,
) -> None:
    ax.scatter(
        pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2],
        c="gray", s=3, alpha=0.15, edgecolors="none",
    )
    if cycle_indices is not None and len(cycle_indices) > 1:
        ordered = order_cycle_vertices(takens_points, cycle_indices)
        idx_to_seg: dict[int, int] = {}
        for si, seg in enumerate(segments):
            for idx in seg:
                idx_to_seg[int(idx)] = si
        for i in range(len(ordered)):
            j = (i + 1) % len(ordered)
            idx_i, idx_j = int(ordered[i]), int(ordered[j])
            if idx_i >= len(pts_3d) or idx_j >= len(pts_3d):
                continue
            p0, p1 = pts_3d[idx_i], pts_3d[idx_j]
            seg_i = idx_to_seg.get(idx_i, -1)
            seg_j = idx_to_seg.get(idx_j, -1)
            if seg_i == seg_j and seg_i >= 0:
                color = SEGMENT_COLORS[seg_i % len(SEGMENT_COLORS)]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                        c=color, linewidth=2.5, alpha=0.9)
            else:
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                        c="#888888", linewidth=1.0, alpha=0.35)
        for si, seg in enumerate(segments):
            color = SEGMENT_COLORS[si % len(SEGMENT_COLORS)]
            valid = [int(i) for i in seg if int(i) < len(pts_3d)]
            if valid:
                sp = pts_3d[valid]
                ax.scatter(sp[:, 0], sp[:, 1], sp[:, 2], c=color, s=15, alpha=0.8,
                           edgecolors="none", label=f"Seg {si+1}")
    for min_idx in [min1_idx, min2_idx]:
        if min_idx is not None and min_idx < len(pts_3d):
            p = pts_3d[min_idx]
            ax.scatter([p[0]], [p[1]], [p[2]], c="gold", s=200, marker="*", zorder=200)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


# ---------------------------------------------------------------------------
# Function 6: create_combined_figure
# ---------------------------------------------------------------------------


def create_combined_figure(
    endpoint_result: EndpointSimilarityResult,
    path_filmstrips: list[FilmstripResult],
    cycle_filmstrips: list[FilmstripResult],
    clip_id: int,
    cross_similarity: float,
    output_path: str,
    pts_3d: np.ndarray,
    paths_remapped: list[np.ndarray],
    cycle_indices: np.ndarray | None,
    segments: list[np.ndarray],
    takens_points: np.ndarray,
    min1_idx: int | None = None,
    min2_idx: int | None = None,
    dpi: int = 150,
) -> None:
    """Create a paper-ready 2-row figure.

    Layout (2 rows x 4 cols):
        Row 0: [Sim Matrix] [PCA Paths top-down] [Path 1 filmstrip] [Path 2 filmstrip]
        Row 1: [   ...     ] [PCA Cycle top-down] [Seg 1 filmstrip]  [Seg 2 filmstrip]
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(30, 10))
    gs = GridSpec(
        2, 5, figure=fig,
        width_ratios=[1, 0.15, 1, 2, 2], wspace=0.12, hspace=0.20,
    )

    ax_heatmap = fig.add_subplot(gs[:, 0])
    cos_sim = endpoint_result.similarity_matrix_cosine
    labels = endpoint_result.endpoint_labels
    n = len(labels)
    im = ax_heatmap.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_heatmap.set_xticks(range(n))
    ax_heatmap.set_yticks(range(n))
    ax_heatmap.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_heatmap.set_yticklabels(labels, fontsize=8)
    ax_heatmap.set_title("Cosine Similarity", fontsize=11)
    for i in range(n):
        for j in range(n):
            val = cos_sim[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax_heatmap.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=color)
    fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)

    # Row 0 col 2: PCA paths (top-down)
    ax_pca_path = fig.add_subplot(gs[0, 2], projection="3d")
    _draw_3d_paths_ax(ax_pca_path, pts_3d, paths_remapped, min1_idx, min2_idx, 90, 0)
    ax_pca_path.set_title("Paths — Top-Down", fontsize=9)
    ax_pca_path.legend(loc="upper left", fontsize=6, framealpha=0.8)

    # Row 0 cols 3-4: path filmstrips
    for pi, fr in enumerate(path_filmstrips[:2]):
        ax = fig.add_subplot(gs[0, 3 + pi])
        ax.imshow(fr.composite_image)
        ax.set_axis_off()
        fi_label = f"frames {fr.frame_indices[0]}–{fr.frame_indices[-1]}" if fr.frame_indices else ""
        ax.set_title(f"Path {pi+1} ({fr.n_poses_rendered} poses, {fi_label})", fontsize=9)

    # Row 1 col 2: PCA cycle (top-down)
    ax_pca_cycle = fig.add_subplot(gs[1, 2], projection="3d")
    _draw_3d_cycle_ax(
        ax_pca_cycle, pts_3d, cycle_indices, segments, takens_points,
        min1_idx, min2_idx, 90, 0,
    )
    n_s = len(segments)
    ax_pca_cycle.set_title(f"Cycle ({n_s} segs) — Top-Down", fontsize=9)
    if n_s > 0:
        ax_pca_cycle.legend(loc="upper left", fontsize=6, framealpha=0.8)

    # Row 1 cols 3-4: cycle segment filmstrips
    if len(cycle_filmstrips) == 1:
        ax = fig.add_subplot(gs[1, 3:])
        fr = cycle_filmstrips[0]
        ax.imshow(fr.composite_image)
        ax.set_axis_off()
        fi_label = f"frames {fr.frame_indices[0]}–{fr.frame_indices[-1]}" if fr.frame_indices else ""
        ax.set_title(f"Cycle ({fr.n_poses_rendered} poses, {fi_label})", fontsize=9)
    else:
        for si, fr in enumerate(cycle_filmstrips[:2]):
            ax = fig.add_subplot(gs[1, 3 + si])
            ax.imshow(fr.composite_image)
            ax.set_axis_off()
            fi_label = f"frames {fr.frame_indices[0]}–{fr.frame_indices[-1]}" if fr.frame_indices else ""
            ax.set_title(f"Seg {si+1} ({fr.n_poses_rendered} poses, {fi_label})", fontsize=9)

    fig.suptitle(
        f"Clip {clip_id} — Theta-graph filmstrip | endpoint-sim = {cross_similarity:.3f}",
        fontsize=13, fontweight="bold",
    )
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved combined figure: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Theta-Graph Filmstrip Demo")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--clip-ids", type=int, nargs="+", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-discover", action="store_true")
    args = parser.parse_args()

    default_config_path = (
        Path(__file__).parent.parent / "configs" / "demos" / "demo_theta_filmstrip.yaml"
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

    if not os.path.isabs(cfg.data.rodent_rollout_dir):
        cfg.data.rodent_rollout_dir = str(project_root / cfg.data.rodent_rollout_dir)
    if not os.path.isabs(cfg.data.xml_path):
        cfg.data.xml_path = str(project_root / cfg.data.xml_path)
    if not os.path.isabs(cfg.data.stac_data_path):
        cfg.data.stac_data_path = str(project_root / cfg.data.stac_data_path)

    if args.top_k is not None:
        cfg.clip_discovery.top_k = args.top_k
    if args.no_discover:
        cfg.clip_discovery.enabled = False

    print("=" * 70)
    print("Theta-Graph Filmstrip Demo")
    print("=" * 70)
    print(f"  Output dir: {output_dir}")
    print(f"  Feature spaces: {[fs.dm_key for fs in cfg.feature_spaces]}")
    print(f"  DM params: k={cfg.dm.k_neighbors}, epsilon={cfg.dm.epsilon}")
    print(f"  Filmstrip: n_poses={cfg.filmstrip.n_poses}, "
          f"threshold={cfg.filmstrip.similarity_threshold}, "
          f"gap_threshold={cfg.filmstrip.gap_threshold}")

    qualified_dir = output_dir / "qualified"
    qualified_dir.mkdir(parents=True, exist_ok=True)

    discovery_hits: list[RearWalkRearHit] = []
    if args.clip_ids:
        clip_ids = args.clip_ids
        print(f"  Clip IDs (CLI override): {clip_ids}")
    elif cfg.clip_discovery.enabled:
        print(f"\n  Auto-discovering rear-walk-rear clips "
              f"(top_k={cfg.clip_discovery.top_k})...")
        discovery_hits = discover_rear_walk_rear_clips(
            cfg.data.rodent_rollout_dir, cfg.clip_discovery
        )
        clip_ids = [h.clip_id for h in discovery_hits]
        print(f"  Discovered clip IDs: {clip_ids}")
    else:
        clip_ids = cfg.clip_ids
        print(f"  Clip IDs (config): {clip_ids}")

    all_results = []
    theta_count = 0
    total_combos = len(clip_ids) * len(cfg.feature_spaces)
    combo_idx = 0

    for clip_id in clip_ids:
        print(f"\n{'='*70}")
        print(f"Clip {clip_id}")
        print(f"{'='*70}")

        for fs_cfg in cfg.feature_spaces:
            combo_idx += 1
            print(
                f"\n  [{combo_idx}/{total_combos}] "
                f"{fs_cfg.dm_key} (m={fs_cfg.takens_m}, tau={fs_cfg.takens_tau}, "
                f"pers_delta={fs_cfg.pers_delta})"
            )

            clip_output_dir = str(output_dir / f"clip_{clip_id}" / fs_cfg.dm_key)
            os.makedirs(clip_output_dir, exist_ok=True)

            try:
                clip_data, var_explained = load_single_clip(
                    clip_id, fs_cfg.dm_key, cfg.data, fs_cfg.pca_dims
                )
                print(f"    Clip data shape: {clip_data.shape}")

                dm_result = run_dm_extraction(
                    clip_data, fs_cfg, cfg.dm, clip_output_dir
                )

                minima_result = find_dm_minima(dm_result)

                remap_to_orig = parse_swr_remap(dm_result.dm_output_dir)
                n_takens = len(dm_result.takens_points)

                conn_result = analyze_edge_connectivity(
                    dm_result.G_dm, minima_result, remap_to_orig, n_takens
                )

                if conn_result is None or conn_result.edge_connectivity < 2:
                    print("    Skipping: not a theta-graph")
                    all_results.append({
                        "clip_id": clip_id, "dm_key": fs_cfg.dm_key,
                        "topology_type": conn_result.topology_type if conn_result else "unknown",
                        "edge_connectivity": conn_result.edge_connectivity if conn_result else 0,
                        "qualified": False, "reason": "not theta",
                    })
                    continue

                paths_2 = [p for p in conn_result.paths_remapped[:2] if len(p) >= 2]
                if len(paths_2) != 2:
                    print(f"    Skipping: need exactly 2 valid paths, got {len(paths_2)}")
                    all_results.append({
                        "clip_id": clip_id, "dm_key": fs_cfg.dm_key,
                        "topology_type": "theta",
                        "edge_connectivity": conn_result.edge_connectivity,
                        "qualified": False, "reason": "not 2 paths",
                    })
                    continue

                cycle_indices = None
                segments = []
                if dm_result.cycles:
                    cycle = dm_result.cycles[0]
                    valid = remap_cycle_to_original(cycle, remap_to_orig, n_takens)
                    if len(valid) > 1:
                        cycle_indices = order_cycle_vertices(dm_result.takens_points, valid)
                        raw_segments = split_cycle_into_segments(
                            cycle_indices,
                            gap_limit=cfg.visualization.segment_gap_limit,
                        )
                        segments = [s for s in raw_segments if len(s) >= cfg.filmstrip.min_segment_length]

                if len(segments) not in (1, 2):
                    print(f"    Skipping: need 1 or 2 cycle segments, got {len(segments)}")
                    all_results.append({
                        "clip_id": clip_id, "dm_key": fs_cfg.dm_key,
                        "topology_type": "theta",
                        "edge_connectivity": conn_result.edge_connectivity,
                        "qualified": False, "reason": f"{len(segments)} cycle segs",
                    })
                    continue

                intention_raw = get_aggregate_data(
                    "/intention", [], clip_id, cfg.data.rodent_rollout_dir
                )

                is_same, endpoint_sim = check_theta_same_endpoints(
                    segments, intention_raw,
                    fs_cfg.takens_m, fs_cfg.takens_tau,
                    cfg.filmstrip.similarity_threshold,
                )
                print(f"    Same-endpoint check (cycle segs): endpoint_sim={endpoint_sim:.3f}, "
                      f"qualified={is_same}")

                if not is_same:
                    print("    Skipping: segment endpoint similarity below threshold")
                    all_results.append({
                        "clip_id": clip_id, "dm_key": fs_cfg.dm_key,
                        "topology_type": "theta",
                        "edge_connectivity": conn_result.edge_connectivity,
                        "endpoint_similarity": round(endpoint_sim, 4),
                        "qualified": False, "reason": "low endpoint-sim",
                    })
                    continue

                endpoint_result = compute_endpoint_similarity(
                    segments, intention_raw,
                    fs_cfg.takens_m, fs_cfg.takens_tau,
                )

                if endpoint_result is None:
                    print("    Skipping: could not compute endpoint similarity")
                    continue

                pca_3d = PCA(n_components=3)
                pts_3d = pca_3d.fit_transform(dm_result.takens_points)

                min1_idx = None
                min2_idx = None
                if minima_result.min1_remapped is not None and minima_result.min1_remapped in remap_to_orig:
                    orig = remap_to_orig[minima_result.min1_remapped]
                    if orig < n_takens:
                        min1_idx = orig
                if minima_result.min2_remapped is not None and minima_result.min2_remapped in remap_to_orig:
                    orig = remap_to_orig[minima_result.min2_remapped]
                    if orig < n_takens:
                        min2_idx = orig

                ro = load_rollout_from_source(clip_id, cfg.data.rodent_rollout_dir)
                qposes = ro["qpos"]
                print(f"    Loaded qposes: {qposes.shape}")

                path_filmstrips = []
                for pi, path_remapped in enumerate(paths_2):
                    print(f"    Rendering filmstrip for path {pi+1} "
                          f"({len(path_remapped)} Takens indices)...")
                    fr = create_filmstrip_composite(
                        qposes, path_remapped,
                        fs_cfg.takens_m, fs_cfg.takens_tau,
                        cfg.filmstrip, pi, clip_id,
                        cfg.data.xml_path,
                    )
                    path_filmstrips.append(fr)
                    print(f"    Path {pi+1}: rendered {fr.n_poses_rendered} poses")

                cycle_filmstrips = []
                for si, seg in enumerate(segments[:2]):
                    print(f"    Rendering filmstrip for cycle seg {si+1} "
                          f"({len(seg)} Takens indices)...")
                    fr = create_filmstrip_composite(
                        qposes, seg,
                        fs_cfg.takens_m, fs_cfg.takens_tau,
                        cfg.filmstrip, si, clip_id,
                        cfg.data.xml_path,
                    )
                    cycle_filmstrips.append(fr)
                    print(f"    Seg {si+1}: rendered {fr.n_poses_rendered} poses")

                if not path_filmstrips and not cycle_filmstrips:
                    print("    Skipping: no valid filmstrips produced")
                    continue

                fig_path = os.path.join(clip_output_dir, "filmstrip_combined.png")
                create_combined_figure(
                    endpoint_result, path_filmstrips, cycle_filmstrips,
                    clip_id, endpoint_sim,
                    fig_path,
                    pts_3d=pts_3d,
                    paths_remapped=paths_2,
                    cycle_indices=cycle_indices,
                    segments=segments,
                    takens_points=dm_result.takens_points,
                    min1_idx=min1_idx,
                    min2_idx=min2_idx,
                    dpi=cfg.visualization.dpi,
                )

                qualified_copy = str(
                    qualified_dir / f"clip_{clip_id}_{fs_cfg.dm_key}_filmstrip.png"
                )
                shutil.copy2(fig_path, qualified_copy)
                print(f"    Copied to qualified: {qualified_copy}")

                theta_count += 1
                all_results.append({
                    "clip_id": clip_id, "dm_key": fs_cfg.dm_key,
                    "topology_type": "theta",
                    "edge_connectivity": conn_result.edge_connectivity,
                    "endpoint_similarity": round(endpoint_sim, 4),
                    "qualified": True,
                    "n_paths": len(path_filmstrips),
                    "path_poses": [fr.n_poses_rendered for fr in path_filmstrips],
                    "n_cycle_segs": len(cycle_filmstrips),
                    "cycle_seg_poses": [fr.n_poses_rendered for fr in cycle_filmstrips],
                    "output": fig_path,
                })

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "clip_id": clip_id, "dm_key": fs_cfg.dm_key,
                    "topology_type": "error", "error": str(e),
                    "qualified": False,
                })

    # Summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    header = f"{'Clip':>5} {'Feature':>10} {'Topology':>10} {'EdgeConn':>9} {'EndptSim':>9} {'Qualified':>10}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in all_results:
        es = r.get("endpoint_similarity", 0.0)
        print(
            f"{r['clip_id']:>5} {r['dm_key']:>10} {r['topology_type']:>10} "
            f"{r.get('edge_connectivity', 0):>9} {es:>9.3f} "
            f"{'YES' if r.get('qualified') else 'no':>10}"
        )
    print(sep)
    print(f"\nTheta-graph filmstrips produced: {theta_count}/{total_combos}")

    summary_path = str(output_dir / "summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump({"results": all_results}, f, default_flow_style=False, sort_keys=False)
    print(f"Saved summary YAML: {summary_path}")

    print(f"\nOutput saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
