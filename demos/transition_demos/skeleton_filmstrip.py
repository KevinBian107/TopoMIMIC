#!/usr/bin/env python3
"""
Skeleton filmstrip helper for Figure 2B/C.
Called by demo_walker_transitions_path.py and demo_rodent_transitions.py
when skeleton_analysis.enabled = true.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from dataclasses import dataclass


@dataclass
class FilmstripSkeletonConfig:
    n_filmstrip_poses: int = 5
    filmstrip_height: int = 120
    gap_fraction: float = 0.15
    dpi: int = 300


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
    fixed_crop: tuple[int, int, int, int] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from demos.property.demo_theta_filmstrip import _load_mujoco_model
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
        if fi >= qposes.shape[0]:
            cropped_poses.append((
                np.full((100, 60, 3), 255, dtype=np.uint8),
                np.zeros((100, 60), dtype=bool),
            ))
            continue

        camera.lookat[:] = camera_lookat_fn(qposes[fi])
        mj_data.qpos[:] = qposes[fi]
        mujoco.mj_forward(mj_model, mj_data)

        rgb_renderer.update_scene(mj_data, camera=camera)
        rgb = rgb_renderer.render().copy()

        seg_renderer.update_scene(mj_data, camera=camera)
        seg = seg_renderer.render()
        geom_ids = seg[:, :, 0].astype(np.int32)
        body_mask = (geom_ids >= 0) & (geom_ids != floor_id)

        if fixed_crop is not None:
            r_min, r_max, c_min, c_max = fixed_crop
        else:
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

        cropped_poses.append((
            rgb[r_min:r_max + 1, c_min:c_max + 1],
            body_mask[r_min:r_max + 1, c_min:c_max + 1],
        ))

    rgb_renderer.close()
    seg_renderer.close()
    return cropped_poses


def _tile_cropped_poses(
    cropped_poses: list[tuple[np.ndarray, np.ndarray]],
    gap_fraction: float = 0.3,
) -> np.ndarray:
    if not cropped_poses:
        return np.full((100, 100, 3), 255, dtype=np.uint8)

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
    gap = max(int(avg_w * gap_fraction), 2)
    total_w = sum(c[0].shape[1] for c in normalized) + gap * (n_poses - 1)

    composite = np.full((max_h, total_w, 3), 255, dtype=np.uint8)
    x_cursor = 0
    for pose_i, (crop_rgb, crop_mask) in enumerate(normalized):
        h, w = crop_rgb.shape[:2]
        t = pose_i / max(n_poses - 1, 1)
        tint = tint_start * (1 - t) + tint_end * t

        if np.any(crop_mask):
            body_pixels = crop_rgb[crop_mask].astype(np.float32) / 255.0
            luminance = 1.0 - np.mean(body_pixels, axis=-1, keepdims=True)
            colored = luminance * tint[None, :] + (1.0 - luminance) * 1.0
            tinted_crop = np.full_like(crop_rgb, 255)
            tinted_crop[crop_mask] = np.clip(colored * 255.0, 0, 255).astype(np.uint8)
        else:
            tinted_crop = crop_rgb

        composite[:h, x_cursor:x_cursor + w] = tinted_crop
        x_cursor += w + gap

    return composite[:, :max(1, x_cursor - gap)]


def _get_camera_params(dataset: str):
    if dataset in ("walker", "walker_long"):
        fovy_rad = math.radians(45.0)
        view_h = 2.4
        dist = (view_h / 2.0) / math.tan(fovy_rad / 2.0) * 1.15

        def lookat_fn(qpos):
            return [qpos[1], 0.0, 0.65]

        fixed_crop = (60, 660, 60, 340)
        return -10.0, 90.0, dist, lookat_fn, 720, 400, fixed_crop
    else:
        fovy_rad = math.radians(45.0)
        view_h = 0.35
        dist = (view_h / 2.0) / math.tan(fovy_rad / 2.0) * 1.3

        def lookat_fn(qpos):
            return [qpos[0], qpos[1], qpos[2] + 0.02]

        return -35.0, 90.0, dist, lookat_fn, 800, 800, None


def render_filmstrip_for_frames(
    qposes: np.ndarray,
    frame_indices: list[int],
    xml_path: str,
    dataset: str,
    n_poses: int = 5,
    gap_fraction: float = 0.3,
) -> np.ndarray:
    if not frame_indices:
        return np.full((100, 200, 3), 255, dtype=np.uint8)

    if len(frame_indices) > n_poses:
        idx = np.linspace(0, len(frame_indices) - 1, n_poses).round().astype(int)
        frame_indices = [frame_indices[i] for i in idx]
    frame_indices = frame_indices[:n_poses]

    elev, azim, dist, lookat_fn, rh, rw, fixed_crop = _get_camera_params(dataset)

    cropped = _render_poses_individually(
        qposes, frame_indices, xml_path,
        camera_elevation=elev, camera_azimuth=azim,
        camera_distance=dist, camera_lookat_fn=lookat_fn,
        render_h=rh, render_w=rw,
        fixed_crop=fixed_crop,
    )
    return _tile_cropped_poses(cropped, gap_fraction=gap_fraction)


def _resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
    oh, ow = img.shape[:2]
    if oh == 0 or ow == 0:
        return np.full((h, h, 3), 255, dtype=np.uint8)
    scale = h / oh
    nw = max(1, int(ow * scale))
    return np.array(Image.fromarray(img).resize((nw, h), Image.LANCZOS))


def _overlay_skeleton_on_ax(
    ax,
    coords: np.ndarray,
    frame_labels: np.ndarray,
    skeleton,
    color_map: dict,
    title: str,
    show_legend: bool = False,
) -> None:
    colors = [color_map.get(str(l), (0.5, 0.5, 0.5, 1.0)) for l in frame_labels]
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, s=1.5, alpha=0.3, edgecolors="none", rasterized=True,
    )

    for path_frames, (pi, pj) in zip(skeleton.paths, skeleton.path_pair_labels):
        valid = path_frames[path_frames < len(coords)]
        if len(valid) < 2:
            continue
        path_coords = coords[valid]
        ax.plot(
            path_coords[:, 0], path_coords[:, 1],
            color=(0.4, 0.4, 0.4), linestyle=":", linewidth=0.6, alpha=0.5, zorder=3,
        )

    for k, (orig_idx, beh) in enumerate(
        zip(skeleton.minima_orig, skeleton.minima_behaviors)
    ):
        if orig_idx < len(coords):
            ax.scatter(
                coords[orig_idx, 0], coords[orig_idx, 1],
                marker="*", s=60, c=[color_map.get(beh, (0.2, 0.2, 0.2, 1.0))],
                edgecolors="black", linewidths=0.4, zorder=5,
            )

    ax.set_title(title, fontsize=7, pad=3)
    ax.tick_params(axis="both", labelsize=4, length=1.5, pad=1)

    if show_legend:
        unique = sorted(set(str(l) for l in frame_labels if str(l) != "transition"))
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                        markerfacecolor=color_map.get(l, (0.5, 0.5, 0.5)),
                        markersize=3, label=l)
            for l in unique
        ]
        handles.append(plt.Line2D([0], [0], marker="*", color="w",
                                   markerfacecolor="gray", markersize=5,
                                   markeredgecolor="black", label="DM min"))
        ax.legend(handles=handles, fontsize=3.5, loc="best",
                  frameon=True, framealpha=0.8, edgecolor="none",
                  handletextpad=0.2, borderpad=0.3)


def _replace_white_bg(img: np.ndarray, color: tuple[int, int, int],
                      thresh: int = 248) -> np.ndarray:
    out = img.copy()
    white_mask = np.all(img >= thresh, axis=-1)
    out[white_mask] = color
    return out


def plot_skeleton_filmstrip(
    data_clip: np.ndarray,
    frame_labels: np.ndarray,
    skeleton,
    qpos_clip: np.ndarray,
    xml_path: str,
    dataset: str,
    output_path: str,
    clip_id: int,
    filmstrip_cfg: FilmstripSkeletonConfig | None = None,
) -> None:
    from matplotlib.patches import FancyBboxPatch
    from sklearn.decomposition import PCA

    if filmstrip_cfg is None:
        filmstrip_cfg = FilmstripSkeletonConfig()

    n_minima = len(skeleton.minima_orig)
    n_paths = min(len(skeleton.paths), n_minima)
    n_poses = filmstrip_cfg.n_filmstrip_poses

    labels_list = list(frame_labels)
    unique = sorted(set(l for l in labels_list if l != "transition"))
    _PALETTE = [
        (0.12, 0.47, 0.71), (1.0, 0.50, 0.05), (0.17, 0.63, 0.17),
        (0.84, 0.15, 0.16), (0.58, 0.40, 0.74), (0.55, 0.34, 0.29),
        (0.89, 0.47, 0.76), (0.50, 0.50, 0.50), (0.74, 0.74, 0.13),
        (0.09, 0.75, 0.81),
    ]
    color_map = {label: _PALETTE[i % len(_PALETTE)] for i, label in enumerate(unique)}
    color_map["transition"] = (0.7, 0.7, 0.7, 1.0)

    pca_pre = PCA(n_components=min(50, data_clip.shape[1]))
    feats_pca = pca_pre.fit_transform(data_clip)

    try:
        from sklearn.manifold import TSNE
        perp = min(30, data_clip.shape[0] - 1)
        embed_coords = TSNE(n_components=2, random_state=42,
                            perplexity=perp).fit_transform(feats_pca)
        embed_title = "t-SNE"
    except Exception:
        pca2 = PCA(n_components=2)
        embed_coords = pca2.fit_transform(data_clip)
        embed_title = "PCA"

    # --- Render filmstrips ---
    print(f"  Rendering {n_minima} minima filmstrips ({n_poses} poses each)...")
    minima_strips = []
    for k, (orig_idx, beh) in enumerate(
        zip(skeleton.minima_orig, skeleton.minima_behaviors)
    ):
        radius = n_poses * 5
        start = max(0, orig_idx - radius)
        end = min(qpos_clip.shape[0], orig_idx + radius + 1)
        frames = list(range(start, end))
        strip = render_filmstrip_for_frames(
            qpos_clip, frames, xml_path, dataset,
            n_poses=n_poses, gap_fraction=filmstrip_cfg.gap_fraction,
        )
        minima_strips.append((strip, beh, k))

    print(f"  Rendering {n_paths} transition filmstrips ({n_poses} poses each)...")
    path_strips = []
    for pidx in range(n_paths):
        path_frames = skeleton.paths[pidx]
        pi, pj = skeleton.path_pair_labels[pidx]
        valid = [int(f) for f in path_frames if f < qpos_clip.shape[0]]
        strip = render_filmstrip_for_frames(
            qpos_clip, valid, xml_path, dataset,
            n_poses=n_poses, gap_fraction=filmstrip_cfg.gap_fraction,
        )
        src_beh = skeleton.minima_behaviors[pi]
        dst_beh = skeleton.minima_behaviors[pj]
        path_strips.append((strip, f"M{pi}\u2192M{pj}", f"{src_beh}\u2192{dst_beh}"))

    # --- Normalize all strips to same pixel dimensions ---
    all_strips = [s for s, _, _ in minima_strips] + [s for s, _, _ in path_strips]
    if all_strips:
        target_h = max(s.shape[0] for s in all_strips)
        target_w = max(s.shape[1] for s in all_strips)

        def _pad_to_size(img, th, tw):
            oh, ow = img.shape[:2]
            resized = _resize_to_height(img, th)
            rh, rw = resized.shape[:2]
            if rw >= tw:
                return resized
            out = np.full((rh, tw, 3), 255, dtype=np.uint8)
            x_off = (tw - rw) // 2
            out[:, x_off:x_off + rw] = resized
            return out

        minima_strips = [
            (_pad_to_size(s, target_h, target_w), beh, idx)
            for s, beh, idx in minima_strips
        ]
        path_strips = [
            (_pad_to_size(s, target_h, target_w), pair, beh)
            for s, pair, beh in path_strips
        ]

    # --- Layout ---
    n_right_cols = max(n_minima, n_paths, 1)
    fig_w = max(8.5, 2.5 + n_right_cols * 1.6)
    fig_h = 4.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[0.30, 0.70],
        height_ratios=[1, 1],
        wspace=0.06, hspace=0.18,
        figure=fig,
    )

    # Left: t-SNE spanning both rows, forced square
    ax_embed = fig.add_subplot(gs[:, 0])
    _overlay_skeleton_on_ax(
        ax_embed, embed_coords, frame_labels, skeleton, color_map,
        embed_title, show_legend=True,
    )
    ax_embed.set_aspect("equal", adjustable="box")
    ax_embed.margins(0.05)

    box_color = "#e8e8e8"
    box_edge = "#b0b0b0"
    box_rgb = (232, 232, 232)
    filmstrip_axes = []

    # Top right: Density Minima
    gs_minima = gridspec.GridSpecFromSubplotSpec(
        1, n_right_cols, subplot_spec=gs[0, 1], wspace=0.14,
    )
    for k, (strip, beh, idx) in enumerate(minima_strips):
        ax = fig.add_subplot(gs_minima[0, k])
        strip_grey = _replace_white_bg(strip, box_rgb)
        ax.imshow(strip_grey)
        ax.set_axis_off()
        ax.set_title(f"M{idx}: {beh}", fontsize=5, pad=2)
        filmstrip_axes.append(ax)

    for k in range(n_minima, n_right_cols):
        ax = fig.add_subplot(gs_minima[0, k])
        ax.set_axis_off()


    # Bottom right: Transition Paths
    gs_paths = gridspec.GridSpecFromSubplotSpec(
        1, n_right_cols, subplot_spec=gs[1, 1], wspace=0.14,
    )
    for pidx, (strip, pair_label, beh_label) in enumerate(path_strips):
        ax = fig.add_subplot(gs_paths[0, pidx])
        strip_grey = _replace_white_bg(strip, box_rgb)
        ax.imshow(strip_grey)
        ax.set_axis_off()
        ax.set_title(f"{pair_label}\n{beh_label}", fontsize=4, pad=2)
        filmstrip_axes.append(ax)

    for pidx in range(n_paths, n_right_cols):
        ax = fig.add_subplot(gs_paths[0, pidx])
        ax.set_axis_off()


    # Draw layout first so get_tightbbox returns correct positions
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in filmstrip_axes:
        tb = ax.get_tightbbox(renderer)
        if tb is None:
            continue
        bb = tb.transformed(fig.transFigure.inverted())
        pad_box = 0.005
        rect = FancyBboxPatch(
            (bb.x0 - pad_box, bb.y0 - pad_box),
            bb.width + 2 * pad_box,
            bb.height + 2 * pad_box,
            boxstyle="round,pad=0.002,rounding_size=0.008",
            facecolor=box_color, edgecolor=box_edge, linewidth=0.8,
            transform=fig.transFigure, zorder=-1,
        )
        fig.patches.append(rect)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=filmstrip_cfg.dpi, bbox_inches="tight",
                facecolor="white", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved skeleton filmstrip: {output_path}")
