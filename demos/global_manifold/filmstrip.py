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

from demos.transition_demos.skeleton_filmstrip import (
    _get_camera_params,
    _render_poses_individually,
    _tile_cropped_poses,
)

from .common import (
    GlobalManifoldConfig,
    SampledPath,
    SpaceResult,
    load_qpos_for_clips,
    takens_anchor_in_full,
)


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


def render_stitched_filmstrip(
    sampled: SampledPath,
    path_index: int,
    res: SpaceResult,
    qpos_by_clip: dict[int, np.ndarray],
    cfg: GlobalManifoldConfig,
) -> StitchedFilmstrip:
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
    composite = _tile_cropped_poses(cropped, gap_fraction=0.25)

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
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    primary = next(
        (r for r in results if r.fs.dm_key == "intention"),
        results[0] if results else None,
    )
    if primary is None:
        print("  [filmstrip] no feature spaces — skipping")
        return

    all_clip_ids = set()
    for pair_paths in list(primary.sampled_paths.values()) + list(primary.sampled_cycles.values()):
        for p in pair_paths:
            for t_idx in p.takens_idxs:
                full_idx = takens_anchor_in_full(primary.takens, primary.pooled, int(t_idx))
                all_clip_ids.add(int(primary.pooled.clip_ids[full_idx]))

    print(f"  [filmstrip] loading qpos for {len(all_clip_ids)} source clips")
    qpos_by_clip = load_qpos_for_clips(cfg.data_path, sorted(all_clip_ids))

    per_pair_strips: dict[tuple[str, str], list[StitchedFilmstrip]] = {}
    for (a, b), paths in primary.sampled_paths.items():
        strips = []
        for i, p in enumerate(paths):
            try:
                s = render_stitched_filmstrip(p, i, primary, qpos_by_clip, cfg)
                strips.append(s)
            except Exception as e:
                print(f"    [filmstrip] ({a},{b}) path {i}: render failed: {e}")
        per_pair_strips[(a, b)] = strips

    pairs = [pair for pair, strips in per_pair_strips.items() if strips]
    if not pairs:
        print("  [filmstrip] no renderable paths — skipping")
        return

    n_rows_per_pair = max(len(per_pair_strips[p]) for p in pairs)

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.size": 7,
        "axes.titlesize": 8,
        "figure.dpi": 300,
    }):
        fig_h_per_row = 1.2
        fig_h = (n_rows_per_pair * fig_h_per_row + 0.4) * len(pairs)
        fig_w = 8.5
        fig = plt.figure(figsize=(fig_w, fig_h))

        gs = fig.add_gridspec(
            len(pairs), 1, hspace=0.4, left=0.02, right=0.98, top=0.98, bottom=0.02,
        )

        for pi, (a, b) in enumerate(pairs):
            strips = per_pair_strips[(a, b)]
            outer = gs[pi].subgridspec(len(strips) + 1, 1, hspace=0.1)

            title_ax = fig.add_subplot(outer[0])
            title_ax.axis("off")
            title_ax.text(
                0.0, 0.5,
                f"{a} ↔ {b}   (k={len(strips)})",
                fontsize=9, fontweight="bold", va="center",
            )
            title_ax.add_patch(FancyBboxPatch(
                (0.001, -0.15),
                0.998, len(strips) + 1.1,
                boxstyle="round,pad=0.01", transform=title_ax.transAxes,
                linewidth=0.8, edgecolor="lightgray", facecolor="#FAFAFA",
                zorder=-1, clip_on=False,
            ))

            for si, strip in enumerate(strips):
                ax = fig.add_subplot(outer[si + 1])
                ax.imshow(strip.composite)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel(f"path {si + 1}", fontsize=6, rotation=0, labelpad=28, va="center")
                clip_str = " → ".join(f"c{c}" for c in strip.clip_sequence)
                ax.set_title(clip_str, fontsize=5, pad=1, loc="left", color="gray")
                for spine in ax.spines.values():
                    spine.set_visible(False)

        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
        fig.savefig(output_pdf, dpi=cfg.visualization.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  [fig] wrote {output_pdf}")
