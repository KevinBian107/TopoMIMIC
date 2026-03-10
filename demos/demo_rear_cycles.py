#!/usr/bin/env python3
"""
Rear-Cycle Grid Demo
=====================
Discovers N clips with X -> middle_behavior -> X pattern (z low->high->low),
runs DM extraction, and produces four grid outputs:

  1. cycle_rotating_grid.mp4  — rotating 3D PCA + synced video for DM cycles
  2. cycle_filmstrip_grid.png — chronophotography filmstrip for DM cycle segments
  3. path_rotating_grid.mp4   — rotating 3D PCA + synced video for DM paths
  4. path_filmstrip_grid.png  — chronophotography filmstrip for DM path segments

Each clip gets a base color. Multiple segments from the same clip use graded
shades of that color. A combined PCA panel on the left shows all clips overlaid.

Usage:
    python demos/demo_rear_cycles.py
    python demos/demo_rear_cycles.py --config configs/demos/demo_rear_cycles.yaml
    python demos/demo_rear_cycles.py --clip-ids 42 99 103
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import argparse
import io
import yaml
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "PCD-Graph-Recon-DM"))

from demos.demo_cycle_asymmetry import (
    load_single_clip,
    run_dm_extraction,
    order_cycle_vertices,
    DataConfig,
    FeatureSpaceConfig,
    DMConfig,
    VisualizationConfig,
    DMResult,
)
from demos.demo_transition_topology import (
    discover_rear_walk_rear_clips,
    find_dm_minima,
    analyze_edge_connectivity,
    ClipDiscoveryConfig,
)
from demos.demo_theta_filmstrip import (
    FilmstripConfig,
    create_filmstrip_composite,
    FilmstripResult,
)
from topo_mimic.dmwk.skeletonization.remap import (
    parse_swr_remap,
    remap_cycle_to_original,
    split_cycle_into_segments,
)
from topo_mimic.dmwk.utils.load_utils import load_rollout_from_source
from topo_mimic.dmwk.skeletonization.rendering import _ensure_rgb_uint8


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class DiscoveryConfig:
    n_clips: int = 6
    middle_behavior: str = "rear"
    z_high: float = 0.06
    z_low: float = 0.02
    min_peak_frames: int = 15
    min_valley_frames: int = 10
    smooth_window: int = 5
    max_xy_displacement: float = 0.03
    max_paths: int = 3
    clip_ids: list[int] = field(default_factory=list)


@dataclass
class Config:
    output_dir: str = "outputs/rear_cycles"
    data: DataConfig = field(default_factory=DataConfig)
    feature_space: FeatureSpaceConfig = field(default_factory=FeatureSpaceConfig)
    dm: DMConfig = field(default_factory=DMConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    filmstrip: FilmstripConfig = field(default_factory=FilmstripConfig)


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(
        output_dir=raw.get("output_dir", "outputs/rear_cycles"),
        data=DataConfig(**raw.get("data", {})),
        feature_space=FeatureSpaceConfig(**raw.get("feature_space", {})),
        dm=DMConfig(**raw.get("dm", {})),
        discovery=DiscoveryConfig(**raw.get("discovery", {})),
        visualization=VisualizationConfig(**raw.get("visualization", {})),
        filmstrip=FilmstripConfig(**raw.get("filmstrip", {})),
    )


# ── Per-clip DM extraction result ──────────────────────────────────────────


@dataclass
class ClipAnalysis:
    clip_id: int
    dm_result: DMResult
    pts_3d: np.ndarray
    cycle_segments: list[np.ndarray]
    path_segments: list[list[np.ndarray]]
    qposes: np.ndarray
    remap_to_orig: dict
    n_takens: int


# ── Discovery ───────────────────────────────────────────────────────────────


def discover_clips(cfg: Config) -> list[int]:
    import h5py

    if cfg.discovery.clip_ids:
        print(f"Using provided clip IDs: {cfg.discovery.clip_ids}")
        return cfg.discovery.clip_ids

    # Request more candidates so we can filter by XY displacement
    n_candidates = cfg.discovery.n_clips * 10
    disc_cfg = ClipDiscoveryConfig(
        enabled=True,
        top_k=n_candidates,
        z_high=cfg.discovery.z_high,
        z_low=cfg.discovery.z_low,
        min_peak_frames=cfg.discovery.min_peak_frames,
        min_valley_frames=cfg.discovery.min_valley_frames,
        smooth_window=cfg.discovery.smooth_window,
    )
    print(f"Discovering clips with "
          f"stationary->{cfg.discovery.middle_behavior}->stationary pattern "
          f"(max_xy_disp={cfg.discovery.max_xy_displacement})...")
    hits = discover_rear_walk_rear_clips(cfg.data.rodent_rollout_dir, disc_cfg)

    # Filter by XY displacement: keep only near-stationary clips
    max_disp = cfg.discovery.max_xy_displacement
    with h5py.File(cfg.data.rodent_rollout_dir, "r") as f:
        key = "qpos_rollout" if "qpos_rollout" in f else "qpos"
        filtered = []
        for h in hits:
            xy = f[key][h.clip_id, :, :2]
            disp = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1)).mean()
            if disp <= max_disp:
                filtered.append((h, disp))
        filtered.sort(key=lambda x: x[0].score, reverse=True)

    selected = filtered[:cfg.discovery.n_clips]
    clip_ids = [h.clip_id for h, _ in selected]
    print(f"  {len(hits)} rear candidates -> {len(filtered)} stationary -> "
          f"top {len(selected)}")
    print(f"Discovered: {clip_ids}")
    for h, disp in selected:
        print(f"  Clip {h.clip_id}: score={h.score:.4f}, "
              f"peak_z=({h.peak1_max_z:.4f}, {h.peak2_max_z:.4f}), "
              f"xy_disp={disp:.5f}")
    return clip_ids


# ── Per-clip analysis ───────────────────────────────────────────────────────


def _dm_cache_exists(clip_output: str) -> bool:
    import pickle as pkl
    return os.path.exists(os.path.join(clip_output, "dm_result.pkl"))


def _save_dm_cache(clip_output: str, dm_result: DMResult):
    import pickle as pkl
    with open(os.path.join(clip_output, "dm_result.pkl"), "wb") as f:
        pkl.dump(dm_result, f)


def _load_dm_cache(clip_output: str) -> DMResult:
    import pickle as pkl
    with open(os.path.join(clip_output, "dm_result.pkl"), "rb") as f:
        return pkl.load(f)


def analyze_clip(
    clip_id: int,
    cfg: Config,
    output_dir: str,
) -> ClipAnalysis | None:
    fs = cfg.feature_space
    clip_output = os.path.join(output_dir, f"clip_{clip_id}")
    os.makedirs(clip_output, exist_ok=True)

    try:
        if _dm_cache_exists(clip_output):
            print(f"  Loading cached DM result from {clip_output}")
            dm_result = _load_dm_cache(clip_output)
        else:
            clip_data, _ = load_single_clip(clip_id, fs.dm_key, cfg.data, fs.pca_dims)
            print(f"  Data: {clip_data.shape}")
            dm_result = run_dm_extraction(clip_data, fs, cfg.dm, clip_output)
            _save_dm_cache(clip_output, dm_result)
        if not dm_result.cycles:
            print(f"  No cycles found, skipping")
            return None

        remap_to_orig = parse_swr_remap(dm_result.dm_output_dir)
        n_takens = len(dm_result.takens_points)

        pca_3d = PCA(n_components=3)
        pts_3d = pca_3d.fit_transform(dm_result.takens_points)

        # Extract segments from the longest cycle
        cycle_segments = []
        cycle = dm_result.cycles[0]
        valid = remap_cycle_to_original(cycle, remap_to_orig, n_takens)
        if len(valid) > 1:
            ordered = order_cycle_vertices(dm_result.takens_points, valid)
            raw_segs = split_cycle_into_segments(
                ordered, gap_limit=cfg.visualization.segment_gap_limit,
            )
            cycle_segments = [
                s for s in raw_segs
                if len(s) >= cfg.filmstrip.min_segment_length
            ]

        # Extract top N path segments
        max_paths = cfg.discovery.max_paths
        path_segments = []
        try:
            minima = find_dm_minima(dm_result)
            conn = analyze_edge_connectivity(
                dm_result.G_dm, minima, remap_to_orig, n_takens,
            )
            if conn is not None and conn.paths_remapped:
                valid_paths = [p for p in conn.paths_remapped
                               if len(p) >= cfg.filmstrip.min_segment_length]
                valid_paths.sort(key=len, reverse=True)
                for p in valid_paths[:max_paths]:
                    path_segments.append([np.array(p)])
        except Exception as e:
            print(f"  Path extraction failed: {e}")

        ro = load_rollout_from_source(clip_id, cfg.data.rodent_rollout_dir)
        qposes = ro["qpos"]

        print(f"  DM: {dm_result.n_cycles} total cycles, "
              f"longest -> {len(cycle_segments)} segments; "
              f"{len(path_segments)} paths")

        return ClipAnalysis(
            clip_id=clip_id,
            dm_result=dm_result,
            pts_3d=pts_3d,
            cycle_segments=cycle_segments,
            path_segments=path_segments,
            qposes=qposes,
            remap_to_orig=remap_to_orig,
            n_takens=n_takens,
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


# ── Rotating PCA frame renderer ─────────────────────────────────────────────


def _render_combined_pca_frame(
    all_pts_3d: list[np.ndarray],
    all_segment_traces: list[list[list[int]]],
    azimuth: float,
    progress: float,
    figsize: tuple[float, float] = (3.0, 3.0),
    dpi: int = 80,
) -> np.ndarray:
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    for pts_3d in all_pts_3d:
        ax.scatter(
            pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2],
            c="gray", s=0.5, alpha=0.08, edgecolors="none",
        )

    for ci, (pts_3d, seg_traces) in enumerate(zip(all_pts_3d, all_segment_traces)):
        base_color = CLIP_BASE_COLORS[ci % len(CLIP_BASE_COLORS)]
        n_segs = len(seg_traces)
        for si, trace in enumerate(seg_traces):
            if not trace:
                continue
            color = _grade_color(base_color, n_segs, si)
            n_show = max(1, int(len(trace) * progress))
            show = trace[:n_show]
            valid = [i for i in show if i < len(pts_3d)]
            if len(valid) > 1:
                p = pts_3d[valid]
                ax.plot(p[:, 0], p[:, 1], p[:, 2], c=color, linewidth=1.5, alpha=0.9)
                ax.scatter([p[-1, 0]], [p[-1, 1]], [p[-1, 2]],
                           c="gold", s=30, marker="o", zorder=200)

    ax.view_init(elev=25, azim=azimuth)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)

    from PIL import Image
    img = np.array(Image.open(buf))[:, :, :3]
    return img


def _render_pca_frame(
    pts_3d: np.ndarray,
    segment_traces: list[list[int]],
    azimuth: float,
    progress: float,
    base_color: str = "#e6194b",
    figsize: tuple[float, float] = (3.0, 3.0),
    dpi: int = 80,
) -> np.ndarray:
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2],
        c="gray", s=1, alpha=0.15, edgecolors="none",
    )

    n_segs = len(segment_traces)
    for si, trace in enumerate(segment_traces):
        if not trace:
            continue
        color = _grade_color(base_color, n_segs, si)
        n_show = max(1, int(len(trace) * progress))
        show = trace[:n_show]
        valid = [i for i in show if i < len(pts_3d)]
        if len(valid) > 1:
            p = pts_3d[valid]
            ax.plot(p[:, 0], p[:, 1], p[:, 2], c=color, linewidth=1.5, alpha=0.9)
            ax.scatter([p[-1, 0]], [p[-1, 1]], [p[-1, 2]],
                       c="gold", s=40, marker="o", zorder=200)

    ax.view_init(elev=25, azim=azimuth)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)

    from PIL import Image
    img = np.array(Image.open(buf))[:, :, :3]
    return img


def _render_rodent_frame(
    qposes: np.ndarray,
    frame_idx: int,
    xml_path: str,
    mj_model=None,
    mj_data=None,
    renderer=None,
) -> tuple[np.ndarray, object, object, object]:
    import mujoco

    if mj_model is None:
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(mj_model)
        renderer = mujoco.Renderer(mj_model, height=240, width=320)

    fi = min(frame_idx, len(qposes) - 1)
    mj_data.qpos[:] = qposes[fi]
    mujoco.mj_forward(mj_model, mj_data)
    renderer.update_scene(mj_data, camera="close_profile")
    frame = renderer.render().copy()
    return _ensure_rgb_uint8(frame), mj_model, mj_data, renderer


# ── Grid rotating video ─────────────────────────────────────────────────────

CLIP_BASE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]


def _grade_color(hex_color: str, n_shades: int, idx: int) -> str:
    from matplotlib.colors import to_rgb, to_hex
    r, g, b = to_rgb(hex_color)
    if n_shades <= 1:
        return hex_color
    # Scale from 40% (dark) to 130% (light) of base, clamped to [0,1]
    scale = 0.4 + 0.9 * idx / (n_shades - 1)
    return to_hex((
        min(r * scale, 1.0),
        min(g * scale, 1.0),
        min(b * scale, 1.0),
    ))


def _get_segment_traces(
    a: ClipAnalysis,
    segment_type: str,
) -> list[list[int]]:
    traces = []
    if segment_type == "cycle":
        for seg in a.cycle_segments:
            traces.append(sorted(int(i) for i in seg))
    else:
        for path_segs in a.path_segments:
            for seg in path_segs:
                traces.append(sorted(int(i) for i in seg))
    return traces


@dataclass
class _SegmentEntry:
    analysis: ClipAnalysis
    clip_idx: int
    seg_idx: int
    n_segs: int
    trace: list[int]
    color: str


def _build_segment_entries(
    analyses: list[ClipAnalysis],
    segment_type: str,
    close_cycles: bool = True,
) -> list[_SegmentEntry]:
    entries = []
    for ai, a in enumerate(analyses):
        traces = _get_segment_traces(a, segment_type)
        base_color = CLIP_BASE_COLORS[ai % len(CLIP_BASE_COLORS)]
        n_segs = len(traces)
        for si, trace in enumerate(traces):
            if close_cycles and segment_type == "cycle" and len(trace) > 1:
                trace = trace + [trace[0]]
            color = _grade_color(base_color, n_segs, si)
            entries.append(_SegmentEntry(
                analysis=a, clip_idx=ai, seg_idx=si,
                n_segs=n_segs, trace=trace, color=color,
            ))
    return entries


def render_rotating_grid_video(
    analyses: list[ClipAnalysis],
    segment_type: str,
    cfg: Config,
    output_path: str,
):
    import imageio
    from PIL import Image

    n_frames = cfg.visualization.gif_n_frames
    fps = cfg.visualization.gif_fps

    entries = _build_segment_entries(analyses, segment_type)
    if not entries:
        print(f"  No valid {segment_type} segments to render")
        return

    # Group entries by clip — one grid cell per clip, all segments drawn together
    from collections import OrderedDict
    clip_groups: OrderedDict[int, list[_SegmentEntry]] = OrderedDict()
    for e in entries:
        clip_groups.setdefault(e.clip_idx, []).append(e)

    clip_list = list(clip_groups.values())
    n_clips = len(clip_list)
    n_cols = min(n_clips, 3)
    n_rows = (n_clips + n_cols - 1) // n_cols

    pca_size = (240, 240)
    vid_size = (320, 240)
    cell_w = pca_size[0] + vid_size[0]
    cell_h = max(pca_size[1], vid_size[1])

    combined_pca_w = pca_size[0]
    combined_pca_h = pca_size[1]

    grid_w = combined_pca_w + n_cols * cell_w
    grid_h = n_rows * cell_h
    grid_w = grid_w + (grid_w % 2)
    grid_h = grid_h + (grid_h % 2)

    # Build combined PCA data
    combined_pts_3d = [grp[0].analysis.pts_3d for grp in clip_list]
    combined_seg_traces = [[e.trace for e in grp] for grp in clip_list]

    import mujoco
    xml_path = cfg.data.xml_path
    renderers = []
    for grp in clip_list:
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(mj_model)
        renderer = mujoco.Renderer(mj_model, height=vid_size[1], width=vid_size[0])
        renderers.append((mj_model, mj_data, renderer))

    total_segs = sum(len(grp) for grp in clip_list)
    print(f"  Rendering {segment_type} rotating grid: "
          f"{n_clips} clips ({total_segs} segments), {n_frames} frames, "
          f"{n_rows}x{n_cols} grid + combined PCA")

    m = cfg.feature_space.takens_m
    tau = cfg.feature_space.takens_tau
    offset = (m - 1) * tau

    frames_out = []
    for fi in range(n_frames):
        progress = fi / max(n_frames - 1, 1)
        azimuth = (fi / n_frames) * 360.0

        canvas = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

        combined_img = _render_combined_pca_frame(
            combined_pts_3d, combined_seg_traces, azimuth, progress,
            figsize=(combined_pca_w / 80, combined_pca_h / 80), dpi=80,
        )
        combined_img = np.array(Image.fromarray(combined_img).resize(
            (combined_pca_w, combined_pca_h), Image.LANCZOS))
        pca_y0 = (grid_h - combined_pca_h) // 2
        canvas[pca_y0:pca_y0 + combined_pca_h, :combined_pca_w] = combined_img

        x_offset = combined_pca_w

        for ci, grp in enumerate(clip_list):
            row = ci // n_cols
            col = ci % n_cols
            y0 = row * cell_h
            x0 = x_offset + col * cell_w

            a = grp[0].analysis
            seg_traces = [e.trace for e in grp]
            base_color = CLIP_BASE_COLORS[grp[0].clip_idx % len(CLIP_BASE_COLORS)]

            pca_img = _render_pca_frame(
                a.pts_3d, seg_traces, azimuth, progress,
                base_color=base_color,
                figsize=(pca_size[0] / 80, pca_size[1] / 80), dpi=80,
            )
            pca_img = np.array(Image.fromarray(pca_img).resize(
                (pca_size[0], pca_size[1]), Image.LANCZOS))

            # Use combined indices across all segments for the MuJoCo frame
            all_idx = sorted({i for t in seg_traces for i in t})
            n_show = max(1, int(len(all_idx) * progress))
            current_takens_idx = all_idx[min(n_show - 1, len(all_idx) - 1)]
            frame_idx = min(current_takens_idx + offset, len(a.qposes) - 1)

            mj_model, mj_data, renderer = renderers[ci]
            mj_data.qpos[:] = a.qposes[frame_idx]
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera="close_profile")
            vid_img = _ensure_rgb_uint8(renderer.render().copy())

            canvas[y0:y0 + pca_size[1], x0:x0 + pca_size[0]] = pca_img
            canvas[y0:y0 + vid_size[1], x0 + pca_size[0]:x0 + cell_w] = vid_img

        frames_out.append(canvas)

        if fi % 20 == 0:
            print(f"    Frame {fi}/{n_frames}")

    for mj_model, mj_data, renderer in renderers:
        renderer.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264",
        quality=None, pixelformat="yuv420p",
        output_params=["-crf", "23"],
    )
    for frame in frames_out:
        writer.append_data(frame)
    writer.close()
    print(f"  Saved: {output_path}")


# ── Grid filmstrip figure ───────────────────────────────────────────────────


def _render_combined_pca_static(
    all_pts_3d: list[np.ndarray],
    all_seg_traces: list[list[list[int]]],
    clip_ids: list[int],
    figsize: tuple[float, float] = (4.0, 4.0),
    dpi: int = 100,
    azimuth: float = 45.0,
) -> np.ndarray:
    import matplotlib.lines as mlines
    from PIL import Image as PILImage

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    for pts_3d in all_pts_3d:
        ax.scatter(
            pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2],
            c="gray", s=0.5, alpha=0.08, edgecolors="none",
        )

    legend_handles = []
    for ci, (pts_3d, seg_traces) in enumerate(zip(all_pts_3d, all_seg_traces)):
        base_color = CLIP_BASE_COLORS[ci % len(CLIP_BASE_COLORS)]
        n_segs = len(seg_traces)
        for si, trace in enumerate(seg_traces):
            color = _grade_color(base_color, n_segs, si)
            valid = [i for i in trace if i < len(pts_3d)]
            if len(valid) > 1:
                p = pts_3d[valid]
                ax.plot(p[:, 0], p[:, 1], p[:, 2], c=color, linewidth=1.5, alpha=0.9)
        legend_handles.append(
            mlines.Line2D([], [], color=base_color, linewidth=2,
                          label=f"Clip {clip_ids[ci]} ({n_segs} seg)")
        )

    ax.view_init(elev=25, azim=azimuth)
    ax.set_axis_off()
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7,
              framealpha=0.8, handlelength=1.5)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    img = np.array(PILImage.open(buf))[:, :, :3]
    return img


def render_filmstrip_grid(
    analyses: list[ClipAnalysis],
    segment_type: str,
    cfg: Config,
    output_path: str,
):
    from PIL import Image as PILImage

    entries = _build_segment_entries(analyses, segment_type)

    filmstrip_entries: list[tuple[_SegmentEntry, FilmstripResult]] = []
    for e in entries:
        if segment_type == "cycle":
            segs = e.analysis.cycle_segments
        else:
            segs = []
            for ps in e.analysis.path_segments:
                segs.extend(ps)
        if e.seg_idx >= len(segs):
            continue
        seg = segs[e.seg_idx]
        fr = create_filmstrip_composite(
            e.analysis.qposes, seg,
            cfg.feature_space.takens_m,
            cfg.feature_space.takens_tau,
            cfg.filmstrip, e.seg_idx, e.analysis.clip_id,
            cfg.data.xml_path,
        )
        if fr.n_poses_rendered > 0:
            filmstrip_entries.append((e, fr))

    if not filmstrip_entries:
        print(f"  No valid {segment_type} filmstrips")
        return

    n_entries = len(filmstrip_entries)
    grid_cols = min(n_entries, 3)
    grid_rows = (n_entries + grid_cols - 1) // grid_cols

    sample_img = filmstrip_entries[0][1].composite_image
    fs_h, fs_w = sample_img.shape[:2]
    fs_aspect = fs_w / fs_h

    cell_w_inch = 5.0
    cell_h_inch = cell_w_inch / fs_aspect
    pca_col_inch = max(4.0, grid_rows * cell_h_inch * 0.6)
    pca_col_inch = min(pca_col_inch, grid_rows * cell_h_inch)

    fig_w = pca_col_inch + grid_cols * cell_w_inch
    fig_h = max(grid_rows * cell_h_inch, pca_col_inch)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=cfg.visualization.dpi)

    # Left column: combined PCA (square, centered vertically)
    pca_frac = pca_col_inch / fig_w
    pca_side = min(pca_col_inch, fig_h)
    pca_h_frac = pca_side / fig_h
    pca_w_frac = pca_side / fig_w
    pca_x = (pca_frac - pca_w_frac) / 2
    pca_y = (1.0 - pca_h_frac) / 2
    pca_ax = fig.add_axes([max(pca_x, 0.0), max(pca_y, 0.0),
                           min(pca_w_frac, pca_frac), min(pca_h_frac, 1.0)])

    # Build per-clip grouped traces for combined PCA
    clip_pts: dict[int, np.ndarray] = {}
    clip_traces: dict[int, list[list[int]]] = {}
    for e, _ in filmstrip_entries:
        clip_pts[e.clip_idx] = e.analysis.pts_3d
        clip_traces.setdefault(e.clip_idx, []).append(e.trace)
    clip_ids_ordered = sorted(clip_pts)
    combined_pts_3d = [clip_pts[k] for k in clip_ids_ordered]
    combined_seg_traces = [clip_traces[k] for k in clip_ids_ordered]
    clip_ids = [analyses[k].clip_id for k in clip_ids_ordered]

    pca_img = _render_combined_pca_static(
        combined_pts_3d, combined_seg_traces, clip_ids,
        figsize=(pca_side, pca_side), dpi=cfg.visualization.dpi,
    )
    pca_ax.imshow(pca_img, aspect="equal")
    pca_ax.set_axis_off()

    # Right grid: filmstrip cells
    grid_left = pca_frac
    grid_w_frac = 1.0 - pca_frac
    cell_w_frac = grid_w_frac / grid_cols
    cell_h_frac = 1.0 / grid_rows if grid_rows > 0 else 1.0

    for ei, (e, fr) in enumerate(filmstrip_entries):
        row = ei // grid_cols
        col = ei % grid_cols

        x0 = grid_left + col * cell_w_frac
        y0 = 1.0 - (row + 1) * cell_h_frac
        ax = fig.add_axes([x0, y0, cell_w_frac, cell_h_frac])
        ax.imshow(fr.composite_image, aspect="auto")

        for spine in ax.spines.values():
            spine.set_edgecolor(e.color)
            spine.set_linewidth(3)
            spine.set_visible(True)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        fi_label = ""
        if fr.frame_indices:
            fi_label = f"f{fr.frame_indices[0]}-{fr.frame_indices[-1]}"
        ax.set_title(
            f"Clip {e.analysis.clip_id} seg{e.seg_idx + 1} "
            f"({fr.n_poses_rendered}p, {fi_label})",
            fontsize=7, color=e.color, fontweight="bold",
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=cfg.visualization.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Rear-Cycle Grid Demo")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--clip-ids", type=int, nargs="+", default=None)
    args = parser.parse_args()

    default_cfg_path = (
        Path(__file__).parent.parent / "configs" / "demos" / "demo_rear_cycles.yaml"
    )
    if args.config:
        cfg = load_config(args.config)
    elif default_cfg_path.exists():
        print(f"Loading config: {default_cfg_path}")
        cfg = load_config(str(default_cfg_path))
    else:
        cfg = Config()

    project_root = Path(__file__).parent.parent.resolve()
    if not os.path.isabs(cfg.output_dir):
        cfg.output_dir = str(project_root / cfg.output_dir)
    if not os.path.isabs(cfg.data.rodent_rollout_dir):
        cfg.data.rodent_rollout_dir = str(project_root / cfg.data.rodent_rollout_dir)
    if not os.path.isabs(cfg.data.xml_path):
        cfg.data.xml_path = str(project_root / cfg.data.xml_path)
    if not os.path.isabs(cfg.data.stac_data_path):
        cfg.data.stac_data_path = str(project_root / cfg.data.stac_data_path)

    if args.clip_ids:
        cfg.discovery.clip_ids = args.clip_ids

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print("Rear-Cycle Grid Demo")
    print("=" * 60)
    print(f"  Output: {cfg.output_dir}")
    print(f"  Feature: {cfg.feature_space.dm_key} "
          f"(m={cfg.feature_space.takens_m}, tau={cfg.feature_space.takens_tau})")
    print(f"  DM: k={cfg.dm.k_neighbors}, eps={cfg.dm.epsilon}")

    # Stage 1: Discover clips
    print("\n=== Stage 1: Clip discovery ===")
    clip_ids = discover_clips(cfg)

    # Stage 2: Run DM extraction on each clip
    print("\n=== Stage 2: DM extraction ===")
    analyses: list[ClipAnalysis] = []
    for clip_id in clip_ids:
        print(f"\nClip {clip_id}:")
        result = analyze_clip(clip_id, cfg, cfg.output_dir)
        if result is not None:
            analyses.append(result)

    if not analyses:
        print("\nNo clips produced valid analyses. Exiting.")
        return

    print(f"\n{len(analyses)} clips with valid DM results")

    # Stage 3: Cycle rotating grid video
    print("\n=== Stage 3: Cycle rotating grid ===")
    cycle_clips = [a for a in analyses if a.cycle_segments]
    if cycle_clips:
        render_rotating_grid_video(
            cycle_clips, "cycle", cfg,
            os.path.join(cfg.output_dir, "cycle_rotating_grid.mp4"),
        )
    else:
        print("  No clips with cycle segments")

    # Stage 4: Cycle filmstrip grid
    print("\n=== Stage 4: Cycle filmstrip grid ===")
    if cycle_clips:
        render_filmstrip_grid(
            cycle_clips, "cycle", cfg,
            os.path.join(cfg.output_dir, "cycle_filmstrip_grid.png"),
        )

    # Stage 5: Path rotating grid video
    print("\n=== Stage 5: Path rotating grid ===")
    path_clips = [a for a in analyses if a.path_segments]
    if path_clips:
        render_rotating_grid_video(
            path_clips, "path", cfg,
            os.path.join(cfg.output_dir, "path_rotating_grid.mp4"),
        )
    else:
        print("  No clips with path segments")

    # Stage 6: Path filmstrip grid
    print("\n=== Stage 6: Path filmstrip grid ===")
    if path_clips:
        render_filmstrip_grid(
            path_clips, "path", cfg,
            os.path.join(cfg.output_dir, "path_filmstrip_grid.png"),
        )

    print("\n" + "=" * 60)
    print("Rear-Cycle Grid Demo Complete!")
    print(f"  Output: {cfg.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
