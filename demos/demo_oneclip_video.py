#!/usr/bin/env python3
"""Animated single-clip video demo (companion to ``demo_oneclip.png``).

Direct-to-transition video. Strictly mirrors the skeleton-extraction logic
of ``demo_transitions_projections_single_clip.py``: one minimum per unique
behaviour cluster, then DM-paths between every minimum pair.

For each DM-path, a tinted filmstrip grows on the right side: poses are
revealed left-to-right, with the per-pose tint linearly interpolated
between the source-cluster colour and the destination-cluster colour
(same colour scheme as the still-image demo). Simultaneously the path
line grows on the PCA + t-SNE projection panels (left side).

Aspect ratios are preserved -- the matplotlib projection figure is
rendered at exact pixel dimensions and dropped into the canvas without
stretching, and the filmstrip is letterboxed into its row.

Usage
-----
    python -m demos.demo_oneclip_video \
        --config configs/demos/demo_oneclip_video.yaml
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import imageio.v2 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "PCD-Graph-Recon-DM"))

from demos.demo_transitions_projections_single_clip import (  # noqa: E402
    DMPipelineConfig,
    SkeletonAnalysisConfig,
    SkeletonResult,
    _build_skel_color_map,
    _run_dm_pipeline,
    extract_skeleton_for_clip,
)


_PRIMARY_BEHAVIORS = {"Walk", "Immobile", "Turn", "Rear"}
_DIRECTIONS = {"Left", "Right", "Straight"}


def _build_canonical_frame_labels(
    clip_id: int, clip_length: int,
    clip_behaviors: dict[int, dict[int, set[str]]],
) -> np.ndarray:
    """Per-frame canonical label combining primary behaviour + direction.

    Resolves the multi-label ambiguity in the rodent CSV: a frame tagged
    {Turn, Left} is a left turn, not "Left". Output labels are the
    canonical cluster names used elsewhere in the project (Walking,
    Immobile, Rear, "Left Turn", "Right Turn", Turn).
    """
    labels = np.full(clip_length, "transition", dtype=object)
    frame_map = clip_behaviors.get(clip_id, {})
    for frame, behaviors in frame_map.items():
        if frame >= clip_length or not behaviors:
            continue
        primaries = [b for b in behaviors if b in _PRIMARY_BEHAVIORS]
        directions = [b for b in behaviors if b in _DIRECTIONS]
        if "Turn" in primaries:
            d = next((x for x in directions if x in ("Left", "Right")), None)
            labels[frame] = f"{d} Turn" if d else "Turn"
        elif "Walk" in primaries:
            labels[frame] = "Walking"
        elif primaries:
            labels[frame] = sorted(primaries)[0]
        # else keep "transition" -- a direction-only frame is ambiguous.
    return labels
from demos.demo_transitions_projections_single_clip import (  # noqa: E402
    Config as SingleClipConfig,
)
from demos.skeleton_filmstrip import (  # noqa: E402
    _get_camera_params,
    _render_poses_individually,
)
from topo_mimic.analysis.feature_analysis.ground_truth import (  # noqa: E402
    load_behavior_segments_multilabel,
)
from topo_mimic.extraction.utils.feature_transforms import (  # noqa: E402
    get_hdf5_key_for_dm_key,
)


# --- Config ---------------------------------------------------------------


@dataclass
class VideoDemoConfig:
    dataset: str = "rodent"
    data_path: str = "data/rodent_ar.h5"
    labels_csv: str = "behavior_labels/rodent_labels.csv"
    xml_path: str = "topo_mimic/extraction/xml/rodent.xml"

    dm_key: str = "intention"
    clip_id: int = 650

    # DM-G parameters mirror the single-clip yaml.
    pers_delta: float = 1.0
    k_neighbors: int = 3

    # Skeleton extraction (mirrors run_skeleton_analysis behaviour).
    minima_frame_radius: int = 12
    k_paths: int = 1

    # Video composition.
    fps: int = 12             # slow framerate so the rodent's motion reads
    n_video_frames: int = 80  # rodent poses rendered per path (1 per frame)
    hold_frames: int = 18     # ~1.5s hold at the end
    canvas_size: tuple[int, int] = (1400, 1000)
    tsne_panel_px: tuple[int, int] = (820, 820)  # square; aspect preserved
    filmstrip_margin_x: int = 20
    filmstrip_label_h: int = 24
    row_body_w: int = 360         # snug width of each rodent row
    tsne_perplexity: int = 30

    output_dir: str = "outputs/demos/rodent_oneclip_video"
    out_filename: str = "demo_oneclip_video.mp4"


def load_config(path: str | None) -> VideoDemoConfig:
    cfg = VideoDemoConfig()
    if not path or not os.path.exists(path):
        return cfg
    with open(path) as f:
        blob = yaml.safe_load(f) or {}
    for k, v in blob.items():
        if hasattr(cfg, k):
            if k in ("canvas_size", "tsne_panel_px") and isinstance(v, list):
                v = tuple(v)
            setattr(cfg, k, v)
    return cfg


# --- DM extraction on a single clip --------------------------------------


def run_single_clip_dm(
    cfg: VideoDemoConfig, data_clip: np.ndarray,
) -> tuple[object, int, int, int, str]:
    sc_cfg = SingleClipConfig(
        dataset=cfg.dataset,
        data_path=cfg.data_path,
        labels_csv=cfg.labels_csv,
        dm_key=cfg.dm_key,
        output_dir=cfg.output_dir,
        dm_pipeline=DMPipelineConfig(
            enabled=True,
            dm_key=cfg.dm_key,
            min_segment_len=5,
            takens={"auto_estimate": True},
            dm={"pers_delta": cfg.pers_delta,
                "k_neighbors": cfg.k_neighbors,
                "use_dm_path": True},
        ),
    )
    one_clip_data = data_clip[None, ...]
    _, all_G_sm, _, m, tau = _run_dm_pipeline(
        one_clip_data, [cfg.clip_id], sc_cfg,
    )
    G_sm = all_G_sm[0]
    clip_dir = os.path.join(cfg.output_dir, "dm", "clips", "clip_0")
    feat_path = os.path.join(clip_dir, "features.txt")
    n_takens = sum(1 for _ in open(feat_path))
    return G_sm, n_takens, m, tau, clip_dir


# --- Pose rendering helpers ----------------------------------------------


def render_path_animation_poses(
    qpos_clip: np.ndarray,
    path_frames: list[int],
    xml_path: str,
    dataset: str,
    n_video_frames: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[int]]:
    """Render one MuJoCo pose per video frame.

    Resamples ``path_frames`` (the original path-frame indices) to length
    ``n_video_frames`` and renders each one. Returns (per-frame
    cropped+masked poses, per-frame source path-frame indices).
    """
    if len(path_frames) >= 2:
        positions = np.linspace(0, len(path_frames) - 1, n_video_frames)
        sampled = [int(path_frames[int(round(p))]) for p in positions]
    else:
        sampled = [int(path_frames[0])] * n_video_frames
    elev, azim, dist, lookat_fn, rh, rw, fixed_crop = _get_camera_params(dataset)
    cropped = _render_poses_individually(
        qpos_clip, sampled, xml_path,
        camera_elevation=elev, camera_azimuth=azim,
        camera_distance=dist, camera_lookat_fn=lookat_fn,
        render_h=rh, render_w=rw, fixed_crop=fixed_crop,
    )
    return cropped, sampled


def normalize_crops(
    crops: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    if not crops:
        return crops
    max_h = max(c[0].shape[0] for c in crops)
    max_w = max(c[0].shape[1] for c in crops)
    out = []
    for crop_rgb, crop_mask in crops:
        h, w = crop_rgb.shape[:2]
        pt = (max_h - h) // 2; pb = max_h - h - pt
        pl = (max_w - w) // 2; pr = max_w - w - pl
        rgb = np.pad(crop_rgb, ((pt, pb), (pl, pr), (0, 0)),
                     mode="constant", constant_values=255)
        mask = np.pad(crop_mask, ((pt, pb), (pl, pr)),
                      mode="constant", constant_values=False)
        out.append((rgb, mask))
    return out


def apply_tint(crop_rgb: np.ndarray, crop_mask: np.ndarray,
               tint: np.ndarray) -> np.ndarray:
    if not np.any(crop_mask):
        return crop_rgb.copy()
    body = crop_rgb[crop_mask].astype(np.float32) / 255.0
    luminance = 1.0 - np.mean(body, axis=-1, keepdims=True)
    coloured = luminance * tint[None, :] + (1.0 - luminance)
    out = np.full_like(crop_rgb, 255)
    out[crop_mask] = np.clip(coloured * 255.0, 0, 255).astype(np.uint8)
    return out




def _behavior_tint_pair(src_color: tuple, dst_color: tuple,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Same start/end tint logic as skeleton_filmstrip._transition_tints."""
    return (
        np.clip(np.array(src_color[:3]) * 0.5 + 0.5, 0, 1),
        np.clip(np.array(dst_color[:3]) * 0.75, 0, 1),
    )


# --- Layout helpers ------------------------------------------------------


def fit_letterbox(img: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    """Fit ``img`` into a (box_w, box_h) rectangle preserving aspect."""
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return np.full((box_h, box_w, 3), 255, dtype=np.uint8)
    scale = min(box_w / iw, box_h / ih)
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))
    resized = np.array(Image.fromarray(img).resize((nw, nh), Image.LANCZOS))
    canvas = np.full((box_h, box_w, 3), 255, dtype=np.uint8)
    y0 = (box_h - nh) // 2
    x0 = (box_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return buf.reshape(h, w, 4)[:, :, [1, 2, 3]]


def draw_label_bar(
    width: int, height: int,
    src_label: str, dst_label: str,
    src_color: tuple, dst_color: tuple,
    pair_str: str,
) -> np.ndarray:
    bar = np.full((height, width, 3), 245, dtype=np.uint8)
    img = Image.fromarray(bar)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    sx = 14
    draw.text((sx, 4), pair_str, font=font, fill=(40, 40, 40))
    bx = sx + draw.textlength(pair_str + "  ", font=font)
    sc = tuple(int(c * 255) for c in src_color[:3])
    dc = tuple(int(c * 255) for c in dst_color[:3])
    draw.text((bx, 4), src_label, font=font, fill=sc)
    bx2 = bx + draw.textlength(src_label + "  ", font=font)
    draw.text((bx2, 4), "->", font=font, fill=(0, 0, 0))
    bx3 = bx2 + draw.textlength("->  ", font=font)
    draw.text((bx3, 4), dst_label, font=font, fill=dc)
    return np.array(img)


def draw_header(canvas_w: int, height: int, title: str) -> np.ndarray:
    bar = np.full((height, canvas_w, 3), 255, dtype=np.uint8)
    img = Image.fromarray(bar)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 6), title, font=font, fill=(0, 0, 0))
    return np.array(img)


# --- Projection figure ---------------------------------------------------


def make_tsne_figure(
    tsne_coords: np.ndarray,
    frame_labels: np.ndarray,
    color_map: dict,
    skeleton: SkeletonResult,
    paths_data: list[dict],
    growth_step: float,
    fig_w_px: int,
    fig_h_px: int,
    dpi: int = 100,
) -> plt.Figure:
    """Single square t-SNE panel with a legend explaining every glyph."""
    fig_w_in = fig_w_px / dpi
    fig_h_in = fig_h_px / dpi
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)

    colors = [color_map.get(str(l), (0.5, 0.5, 0.5, 1.0)) for l in frame_labels]
    ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=colors,
               s=14, alpha=0.6, edgecolors="none", rasterized=True)

    for k, (orig, beh) in enumerate(
        zip(skeleton.minima_orig, skeleton.minima_behaviors)
    ):
        if orig < len(tsne_coords):
            ax.scatter(tsne_coords[orig, 0], tsne_coords[orig, 1],
                       marker="*", s=320,
                       c=[color_map.get(beh, (0.2, 0.2, 0.2, 1.0))],
                       edgecolors="black", linewidths=1.0, zorder=6)
            ax.annotate(f"M{k}", (tsne_coords[orig, 0], tsne_coords[orig, 1]),
                        fontsize=11, fontweight="bold",
                        ha="left", va="bottom",
                        xytext=(6, 5), textcoords="offset points")

    from matplotlib.collections import LineCollection

    def _seg_color(idx: int) -> tuple:
        c = color_map.get(str(frame_labels[idx]), (0.5, 0.5, 0.5, 1.0))
        return tuple(c[:3])

    for p in paths_data:
        full = p["full_path_indices"]
        if len(full) < 2:
            continue
        n_proj = max(2, int(round(len(full) * growth_step)))
        seg = [s for s in full[:n_proj] if s < len(tsne_coords)]
        if len(seg) < 2:
            continue
        # Per-segment colour matches the cluster colour of the point the
        # segment passes through (filmstrip-style gradient).
        pts = tsne_coords[seg].reshape(-1, 1, 2)
        segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
        seg_colors = [_seg_color(seg[i]) for i in range(len(seg) - 1)]
        lc = LineCollection(segments, colors=seg_colors,
                            linewidths=2.6, alpha=0.95, zorder=4,
                            capstyle="round", joinstyle="round")
        ax.add_collection(lc)
        ax.scatter(tsne_coords[seg[-1], 0], tsne_coords[seg[-1], 1],
                   marker="o", s=110,
                   c=[_seg_color(seg[-1])],
                   edgecolors="black", linewidths=0.8, zorder=7)

    ax.set_title("t-SNE of intention features",
                 fontsize=14, fontweight="bold", pad=8)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Legend: behaviour swatches + structural glyphs.
    handles = []
    for beh in skeleton.minima_behaviors:
        c = color_map.get(beh, (0.5, 0.5, 0.5))
        handles.append(plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=c, markeredgecolor="none",
            markersize=10, label=beh,
        ))
    handles.append(plt.Line2D(
        [0], [0], marker="*", color="w",
        markerfacecolor="0.6", markeredgecolor="black",
        markersize=14, label="cluster minimum (Mk)",
    ))
    handles.append(plt.Line2D(
        [0], [0], color="0.2", linewidth=2.2,
        label="DM-path (growing, coloured by cluster)",
    ))
    handles.append(plt.Line2D(
        [0], [0], marker="o", color="w",
        markerfacecolor="0.5", markeredgecolor="black",
        markersize=10, label="current path frame",
    ))
    ax.legend(handles=handles, loc="upper right",
              fontsize=9, frameon=True, framealpha=0.92,
              edgecolor="0.7", handletextpad=0.5,
              borderpad=0.5, labelspacing=0.45)

    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.10)
    return fig


# --- Per-frame composition -----------------------------------------------


def compose_canvas(
    tsne_img: np.ndarray,
    rows_data: list[tuple[np.ndarray, dict]],
    cfg: VideoDemoConfig,
) -> np.ndarray:
    """Left = square t-SNE panel; right = stack of compact rodent rows
    (each one ``cfg.row_body_w`` wide). The rodent is letterboxed, no
    artificial translation -- the MuJoCo camera follows the body so the
    actual behaviour (limb / posture) reads in place.
    """
    canvas_w, canvas_h = cfg.canvas_size
    out = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    header_h = 38
    out[:header_h] = draw_header(
        canvas_w, header_h,
        "Phase: Transitions  (rodent traverses each DM-path)",
    )

    # t-SNE on the LEFT; native pixel size preserves aspect.
    th, tw = tsne_img.shape[:2]
    left_w = tw + 2 * cfg.filmstrip_margin_x
    tx = cfg.filmstrip_margin_x
    ty = header_h + (canvas_h - header_h - th) // 2
    out[ty:ty + th, tx:tx + tw] = tsne_img

    # Compact rodent rows on the RIGHT of the t-SNE.
    right_x = left_w
    right_w = canvas_w - right_x
    body_w = min(cfg.row_body_w, right_w - 2 * cfg.filmstrip_margin_x)
    body_x_left = right_x + (right_w - body_w) // 2

    n = max(len(rows_data), 1)
    fs_top = header_h
    fs_total_h = canvas_h - fs_top
    row_h = fs_total_h // n

    for k, (pose_img, meta) in enumerate(rows_data):
        y0 = fs_top + k * row_h
        label_h = cfg.filmstrip_label_h
        body_h = max(20, row_h - label_h - 8)
        label = draw_label_bar(body_w, label_h,
                               meta["src_label"], meta["dst_label"],
                               meta["src_color"], meta["dst_color"],
                               meta["pair_str"])
        out[y0:y0 + label_h, body_x_left:body_x_left + body_w] = label
        body = fit_letterbox(pose_img, body_w, body_h)
        out[y0 + label_h:y0 + label_h + body_h,
            body_x_left:body_x_left + body_w] = body
    return out


# --- Top-level orchestration --------------------------------------------


def run(cfg: VideoDemoConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Loading clip {cfg.clip_id} from {cfg.data_path}...")
    hdf5_key = get_hdf5_key_for_dm_key(cfg.dm_key)
    with h5py.File(cfg.data_path, "r") as h5f:
        data_clip = np.asarray(h5f[hdf5_key][cfg.clip_id])
        qpos_clip = np.asarray(h5f["qpos_rollout"][cfg.clip_id])
    if data_clip.ndim == 1:
        data_clip = data_clip[:, None]
    print(f"  data_clip {data_clip.shape}, qpos_clip {qpos_clip.shape}")

    print("Running DM-G on the single clip...")
    G_sm, n_takens, takens_m, takens_tau, clip_dir = run_single_clip_dm(
        cfg, data_clip,
    )
    takens_offset = (takens_m - 1) * takens_tau
    print(f"  n_takens={n_takens}, m={takens_m}, tau={takens_tau}, "
          f"offset={takens_offset}")

    clip_behaviors = load_behavior_segments_multilabel(
        cfg.labels_csv, merge_rear=True,
    )
    frame_labels = _build_canonical_frame_labels(
        cfg.clip_id, data_clip.shape[0], clip_behaviors,
    )
    n_unique = len(set(str(l) for l in frame_labels) - {"transition"})
    print(f"  unique non-transition behaviours: {n_unique}")
    if n_unique < 2:
        print("ERROR: need at least 2 distinct behaviours; pick a different clip.")
        return

    sk_cfg = SkeletonAnalysisConfig(
        enabled=True,
        n_minima=n_unique,
        k_paths=cfg.k_paths,
        minima_frame_radius=cfg.minima_frame_radius,
        target_size=(640, 480),
    )
    skeleton = extract_skeleton_for_clip(
        G_sm, clip_dir, data_clip, frame_labels, sk_cfg, n_takens,
        n_minima_override=n_unique,
        takens_offset=takens_offset,
    )
    if skeleton is None or len(skeleton.minima_orig) < 2 or not skeleton.paths:
        print("ERROR: skeleton has insufficient minima/paths; "
              "try a different clip.")
        return

    # Strictly one minimum per cluster label: drop duplicates from the
    # extractor's fill-pass, then prune paths whose endpoints were dropped.
    seen: dict[str, int] = {}
    keep: list[int] = []
    for idx, beh in enumerate(skeleton.minima_behaviors):
        if beh != "transition" and beh not in seen:
            seen[beh] = idx
            keep.append(idx)
    if len(keep) < len(skeleton.minima_orig):
        old_to_new = {old: new for new, old in enumerate(keep)}
        skeleton = SkeletonResult(
            minima_orig=[skeleton.minima_orig[i] for i in keep],
            minima_behaviors=[skeleton.minima_behaviors[i] for i in keep],
            paths=[p for p, (pi, pj) in zip(skeleton.paths,
                                             skeleton.path_pair_labels)
                   if pi in old_to_new and pj in old_to_new],
            path_pair_labels=[(old_to_new[pi], old_to_new[pj])
                              for (pi, pj) in skeleton.path_pair_labels
                              if pi in old_to_new and pj in old_to_new],
            purity=1.0,
        )
    if len(skeleton.minima_orig) < 2 or not skeleton.paths:
        print("ERROR: after dedup not enough distinct minima/paths.")
        return
    print(f"  {len(skeleton.minima_orig)} minima "
          f"({skeleton.minima_behaviors}), {len(skeleton.paths)} paths")

    print("Computing t-SNE...")
    perp = min(cfg.tsne_perplexity, max(5, data_clip.shape[0] // 4))
    tsne_coords = TSNE(n_components=2, random_state=0, perplexity=perp,
                       init="pca", learning_rate="auto").fit_transform(data_clip)

    color_map = _build_skel_color_map([str(l) for l in frame_labels])

    print(f"Pre-rendering animated rodent for "
          f"{len(skeleton.paths)} paths "
          f"({cfg.n_video_frames} poses each)...")
    paths_data: list[dict] = []
    for j, path_frames in enumerate(skeleton.paths):
        valid = [int(f) for f in path_frames
                 if 0 <= int(f) < len(qpos_clip)]
        if len(valid) < 2:
            continue
        pi, pj = skeleton.path_pair_labels[j]
        src_lab = skeleton.minima_behaviors[pi]
        dst_lab = skeleton.minima_behaviors[pj]
        src_col = np.array(color_map.get(src_lab, (0.5, 0.5, 0.5))[:3])
        dst_col = np.array(color_map.get(dst_lab, (0.5, 0.5, 0.5))[:3])
        src_tint, dst_tint = _behavior_tint_pair(src_col, dst_col)

        cropped, _sampled = render_path_animation_poses(
            qpos_clip, valid, cfg.xml_path, cfg.dataset, cfg.n_video_frames,
        )
        normalized = normalize_crops(cropped)
        tinted = []
        for i, (rgb, mask) in enumerate(normalized):
            t = i / max(len(normalized) - 1, 1)
            tint = src_tint * (1 - t) + dst_tint * t
            tinted.append(apply_tint(rgb, mask, tint))

        paths_data.append({
            "full_path_indices": valid,
            "anim_poses": tinted,  # one tinted pose per video frame
            "src_label": src_lab, "dst_label": dst_lab,
            "src_color": src_col, "dst_color": dst_col,
            "pi": pi, "pj": pj,
            "pair_str": f"M{pi} -> M{pj}",
        })
    if not paths_data:
        print("ERROR: no DM-paths survived filtering.")
        return

    print(f"Composing {cfg.n_video_frames} video frames + "
          f"{cfg.hold_frames} hold...")
    tsne_w_px, tsne_h_px = cfg.tsne_panel_px
    frames = []
    for t in range(cfg.n_video_frames):
        progress = t / max(cfg.n_video_frames - 1, 1)
        fig = make_tsne_figure(
            tsne_coords, frame_labels, color_map, skeleton,
            paths_data, progress, tsne_w_px, tsne_h_px,
        )
        tsne_img = fig_to_rgb(fig)
        plt.close(fig)

        rows = [(p["anim_poses"][t], p) for p in paths_data]
        frames.append(compose_canvas(tsne_img, rows, cfg))

    last = frames[-1]
    for _ in range(cfg.hold_frames):
        frames.append(last)

    out_path = os.path.join(cfg.output_dir, cfg.out_filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = iio.get_writer(out_path, fps=cfg.fps, codec="libx264",
                            quality=8, macro_block_size=1)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  wrote {out_path}  ({len(frames)} frames @ {cfg.fps} fps)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
