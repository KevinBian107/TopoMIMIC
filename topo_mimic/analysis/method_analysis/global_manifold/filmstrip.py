"""Cross-clip stitched filmstrip renderer for the global-manifold demo.

A single filmstrip is built from a list of (clip_id, frame_idx) tuples, where
the frames come from different clips in the pooled point cloud. Each rendered
pose is annotated with its source clip id so reviewers can see the cross-clip
nature of a global DM path.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from demos.skeleton_filmstrip import (
    _get_camera_params,
    _render_poses_individually,
    _replace_white_bg,
    _tile_cropped_poses,
    _transition_tints,
)

from .common import (
    GlobalManifoldConfig,
    SampledPath,
    SpaceResult,
    _build_color_map,
    load_qpos_for_clips,
    takens_anchor_in_full,
)

MAX_STRIPS_PER_PAIR = 3
BOX_RGB = (242, 242, 242)


@dataclass
class StitchedFilmstrip:
    label_pair: tuple[str, str]
    path_index: int
    composite: np.ndarray                        # (H, W, 3) uint8
    clip_sequence: list[int]                     # source clip per pose


def _pick_evenly_spaced_takens_points(
    takens_idxs: np.ndarray,
    n_poses: int,
) -> np.ndarray:
    if len(takens_idxs) <= n_poses:
        return takens_idxs
    idx = np.linspace(0, len(takens_idxs) - 1, n_poses).round().astype(int)
    return takens_idxs[idx]


def _build_virtual_qposes(
    selected_takens_idxs: np.ndarray,
    res: SpaceResult,
    qpos_by_clip: dict[int, np.ndarray],
) -> tuple[np.ndarray, list[int], list[int]]:
    virtual = []
    source_clips: list[int] = []
    source_frames: list[int] = []
    for t_idx in selected_takens_idxs:
        full_idx = takens_anchor_in_full(res.takens, res.pooled, int(t_idx))
        cid = int(res.pooled.clip_ids[full_idx])
        fr = int(res.pooled.frame_idxs[full_idx])
        qpos = qpos_by_clip[cid]
        fr_clamped = min(fr, qpos.shape[0] - 1)
        virtual.append(qpos[fr_clamped])
        source_clips.append(cid)
        source_frames.append(fr_clamped)
    stacked = np.stack(virtual, axis=0)
    return stacked, source_clips, source_frames


def _render_cropped_for_sample(
    sampled: SampledPath,
    res: SpaceResult,
    qpos_by_clip: dict[int, np.ndarray],
    cfg: GlobalManifoldConfig,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[int]]:
    n_poses = cfg.visualization.filmstrip_n_poses
    selected_t_idxs = _pick_evenly_spaced_takens_points(sampled.takens_idxs, n_poses)

    virtual_qposes, source_clips, _ = _build_virtual_qposes(
        selected_t_idxs, res, qpos_by_clip,
    )

    elev, azim, dist, lookat_fn, rh, rw, fixed_crop = _get_camera_params(cfg.dataset)

    cropped = _render_poses_individually(
        virtual_qposes,
        list(range(len(virtual_qposes))),
        cfg.xml_path,
        camera_elevation=elev,
        camera_azimuth=azim,
        camera_distance=dist,
        camera_lookat_fn=lookat_fn,
        render_h=rh,
        render_w=rw,
        fixed_crop=fixed_crop,
    )
    return cropped, source_clips


def _pad_cropped_uniform(
    cropped: list[tuple[np.ndarray, np.ndarray]],
    target_h: int,
    target_w: int,
    bg: tuple[int, int, int] = BOX_RGB,
) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for rgb, mask in cropped:
        h, w = rgb.shape[:2]
        pad_top = (target_h - h) // 2
        pad_l = (target_w - w) // 2
        rgb_out = np.full((target_h, target_w, 3), bg, dtype=np.uint8)
        rgb_out[pad_top:pad_top + h, pad_l:pad_l + w] = rgb
        mask_out = np.zeros((target_h, target_w), dtype=bool)
        mask_out[pad_top:pad_top + h, pad_l:pad_l + w] = mask
        out.append((rgb_out, mask_out))
    return out


def _assemble_stitched(
    sampled: SampledPath,
    path_index: int,
    cropped_uniform: list[tuple[np.ndarray, np.ndarray]],
    source_clips: list[int],
    tint_start: np.ndarray,
    tint_end: np.ndarray,
) -> StitchedFilmstrip:
    composite = _tile_cropped_poses(
        cropped_uniform, gap_fraction=0.08,
        tint_start=tint_start, tint_end=tint_end,
    )
    composite = _replace_white_bg(composite, BOX_RGB)
    return StitchedFilmstrip(
        label_pair=sampled.label_pair,
        path_index=path_index,
        composite=composite,
        clip_sequence=source_clips,
    )


def render_filmstrip_grid(
    cfg: GlobalManifoldConfig,
    results: list[SpaceResult],
    output_pdf: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    primary = next(
        (r for r in results if r.fs.dm_key == "intention"),
        results[0] if results else None,
    )
    if primary is None:
        print("  [filmstrip] no feature spaces — skipping")
        return

    color_map = _build_color_map(cfg.minima_selection.primitives, cfg.visualization.behavior_colors)

    all_clip_ids = set()
    for pair_paths in list(primary.sampled_paths.values()) + list(primary.sampled_cycles.values()):
        for p in pair_paths[:MAX_STRIPS_PER_PAIR]:
            for t_idx in p.takens_idxs:
                full_idx = takens_anchor_in_full(primary.takens, primary.pooled, int(t_idx))
                all_clip_ids.add(int(primary.pooled.clip_ids[full_idx]))

    print(f"  [filmstrip] loading qpos for {len(all_clip_ids)} source clips")
    qpos_by_clip = load_qpos_for_clips(cfg.data_path, sorted(all_clip_ids))

    raw_by_pair: dict[tuple[str, str], list[tuple[SampledPath, int, list, list[int]]]] = {}
    for (a, b), paths in primary.sampled_paths.items():
        entries = []
        for i, p in enumerate(paths[:MAX_STRIPS_PER_PAIR]):
            try:
                cropped, clips = _render_cropped_for_sample(p, primary, qpos_by_clip, cfg)
                entries.append((p, i, cropped, clips))
            except Exception as e:
                print(f"    [filmstrip] ({a},{b}) path {i}: render failed: {e}")
        raw_by_pair[(a, b)] = entries

    all_hw = [rgb.shape[:2]
              for entries in raw_by_pair.values()
              for (_, _, cropped, _) in entries
              for (rgb, _) in cropped]
    if not all_hw:
        print("  [filmstrip] no renderable poses — skipping")
        return
    target_h = max(h for h, _ in all_hw)
    target_w = max(w for _, w in all_hw)

    per_pair_strips: dict[tuple[str, str], list[StitchedFilmstrip]] = {}
    for (a, b), entries in raw_by_pair.items():
        rgb_a = np.array(mcolors.to_rgb(color_map.get(a, "#888888")))
        rgb_b = np.array(mcolors.to_rgb(color_map.get(b, "#555555")))
        ts, te = _transition_tints(rgb_a, rgb_b)
        strips = []
        for (p, i, cropped, clips) in entries:
            cropped_u = _pad_cropped_uniform(cropped, target_h, target_w)
            strips.append(_assemble_stitched(p, i, cropped_u, clips, ts, te))
        per_pair_strips[(a, b)] = strips

    pairs = [pair for pair, strips in per_pair_strips.items() if strips]
    if not pairs:
        print("  [filmstrip] no renderable paths — skipping")
        return

    n_rows_per_pair = max(len(per_pair_strips[p]) for p in pairs)

    sample_composite = per_pair_strips[pairs[0]][0].composite
    img_h, img_w = sample_composite.shape[:2]
    img_aspect = img_w / img_h

    fig_w = 10.0
    fig_left, fig_right = 0.04, 0.98
    strip_col_frac = 6.0 / 6.5
    strip_axes_w_in = fig_w * (fig_right - fig_left) * strip_col_frac
    fig_h_per_row = strip_axes_w_in / img_aspect

    title_h_ratio = 0.15
    per_sample_h_units = 1.0 + title_h_ratio

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.size": 7,
        "axes.titlesize": 8,
        "figure.dpi": 300,
    }):
        fig_h = (n_rows_per_pair * fig_h_per_row * per_sample_h_units * 1.35 + 0.5) * len(pairs)
        fig = plt.figure(figsize=(fig_w, fig_h))

        gs = fig.add_gridspec(
            len(pairs), 1, hspace=0.6, left=fig_left, right=fig_right, top=0.98, bottom=0.02,
        )

        sample_groups: list[tuple] = []

        for pi, (a, b) in enumerate(pairs):
            strips = per_pair_strips[(a, b)]
            pair_gs = gs[pi].subgridspec(len(strips), 1, hspace=0.55)

            for si, strip in enumerate(strips):
                sample_gs = pair_gs[si].subgridspec(
                    2, 2,
                    width_ratios=[6.0, 0.5],
                    height_ratios=[title_h_ratio, 1.0],
                    hspace=0.02, wspace=0.002,
                )

                title_ax = fig.add_subplot(sample_gs[0, :])
                title_ax.axis("off")
                title_ax.text(
                    0.5, 0.5,
                    f"{a} → {b}   —   sample {si + 1}",
                    fontsize=10, fontweight="bold", va="center", ha="center",
                    color="#222222",
                )

                strip_ax = fig.add_subplot(sample_gs[1, 0])
                strip_ax.imshow(strip.composite)
                strip_ax.set_xticks([])
                strip_ax.set_yticks([])
                for spine in strip_ax.spines.values():
                    spine.set_visible(False)

                text_ax = fig.add_subplot(sample_gs[1, 1])
                text_ax.axis("off")
                clip_str = "\n".join(f"c{c}" for c in strip.clip_sequence)
                text_ax.text(
                    0.0, 0.5, clip_str,
                    fontsize=10, va="center", ha="left",
                    color="#333333", family="monospace",
                )

                sample_groups.append((title_ax, strip_ax, text_ax))

        fig.canvas.draw()
        for title_ax, strip_ax, text_ax in sample_groups:
            bboxes = [title_ax.get_position(), strip_ax.get_position(), text_ax.get_position()]
            x0 = min(bb.x0 for bb in bboxes) - 0.003
            x1 = max(bb.x1 for bb in bboxes) + 0.003
            y0 = min(bb.y0 for bb in bboxes) - 0.003
            y1 = max(bb.y1 for bb in bboxes) + 0.003
            rect = FancyBboxPatch(
                (x0, y0), x1 - x0, y1 - y0,
                boxstyle="round,pad=0.002",
                transform=fig.transFigure,
                linewidth=1.0, edgecolor="#666666", facecolor="#F2F2F2",
                zorder=-10, clip_on=False,
            )
            fig.patches.append(rect)

        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
        fig.savefig(output_pdf, dpi=cfg.visualization.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  [fig] wrote {output_pdf}")
