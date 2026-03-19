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


def _behavior_tints(color: tuple) -> tuple[np.ndarray, np.ndarray]:
    c = np.array(color[:3])
    tint_start = np.clip(c * 0.5 + 0.5, 0, 1)
    tint_end = np.clip(c * 0.75, 0, 1)
    return tint_start, tint_end


def _transition_tints(
    src_color: tuple, dst_color: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.clip(np.array(src_color[:3]) * 0.5 + 0.5, 0, 1),
        np.clip(np.array(dst_color[:3]) * 0.75, 0, 1),
    )


def _tile_cropped_poses(
    cropped_poses: list[tuple[np.ndarray, np.ndarray]],
    gap_fraction: float = 0.3,
    tint_start: np.ndarray | None = None,
    tint_end: np.ndarray | None = None,
) -> np.ndarray:
    if not cropped_poses:
        return np.full((100, 100, 3), 255, dtype=np.uint8)

    n_poses = len(cropped_poses)
    if tint_start is None:
        tint_start = np.array([0.68, 0.80, 0.88])
    if tint_end is None:
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
    tint_start: np.ndarray | None = None,
    tint_end: np.ndarray | None = None,
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
    return _tile_cropped_poses(cropped, gap_fraction=gap_fraction,
                               tint_start=tint_start, tint_end=tint_end)


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
        c=colors, s=8, alpha=0.6, edgecolors="white", linewidths=0.1,
        rasterized=True,
    )

    for path_frames, (pi, pj) in zip(skeleton.paths, skeleton.path_pair_labels):
        valid = path_frames[path_frames < len(coords)]
        if len(valid) < 2:
            continue
        path_coords = coords[valid]
        ax.plot(
            path_coords[:, 0], path_coords[:, 1],
            color=(0.4, 0.4, 0.4), linestyle=":", linewidth=0.8, alpha=0.5, zorder=3,
        )

    for k, (orig_idx, beh) in enumerate(
        zip(skeleton.minima_orig, skeleton.minima_behaviors)
    ):
        if orig_idx < len(coords):
            ax.scatter(
                coords[orig_idx, 0], coords[orig_idx, 1],
                marker="*", s=120, c=[color_map.get(beh, (0.2, 0.2, 0.2, 1.0))],
                edgecolors="black", linewidths=0.5, zorder=5,
            )

    ax.set_title(title, fontsize=8, fontweight="bold", pad=4)
    ax.tick_params(axis="both", labelsize=5, length=2, pad=2)

    if show_legend:
        unique = sorted(set(str(l) for l in frame_labels if str(l) != "transition"))
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                        markerfacecolor=color_map.get(l, (0.5, 0.5, 0.5)),
                        markeredgecolor="white", markeredgewidth=0.3,
                        markersize=5, label=l)
            for l in unique
        ]
        handles.append(plt.Line2D([0], [0], marker="*", color="w",
                                   markerfacecolor="gray", markersize=7,
                                   markeredgecolor="black", label="DM min"))
        ax.legend(handles=handles, fontsize=6, loc="best",
                  frameon=True, framealpha=0.85, edgecolor="none",
                  handletextpad=0.3, borderpad=0.4)


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

    # --- Compute all three embeddings ---
    pca_2d = PCA(n_components=2)
    pca_coords = pca_2d.fit_transform(data_clip)
    pca_title = f"PCA ({pca_2d.explained_variance_ratio_[:2].sum():.0%} var)"

    pca_pre = PCA(n_components=min(50, data_clip.shape[1]))
    feats_pca = pca_pre.fit_transform(data_clip)

    try:
        from umap import UMAP
        umap_coords = UMAP(n_components=2, random_state=42).fit_transform(feats_pca)
        umap_title = "UMAP"
    except Exception:
        umap_coords = pca_coords.copy()
        umap_title = "PCA (UMAP failed)"

    try:
        from sklearn.manifold import TSNE
        perp = min(30, data_clip.shape[0] - 1)
        tsne_coords = TSNE(n_components=2, random_state=42,
                           perplexity=perp).fit_transform(feats_pca)
        tsne_title = "t-SNE"
    except Exception:
        tsne_coords = pca_coords.copy()
        tsne_title = "PCA (t-SNE failed)"

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
        beh_color = color_map.get(beh, (0.5, 0.5, 0.5))
        ts, te = _behavior_tints(beh_color)
        strip = render_filmstrip_for_frames(
            qpos_clip, frames, xml_path, dataset,
            n_poses=n_poses, gap_fraction=filmstrip_cfg.gap_fraction,
            tint_start=ts, tint_end=te,
        )
        minima_strips.append((strip, beh, k))

    print(f"  Rendering {n_paths} transition filmstrips ({n_poses} poses each)...")
    path_strips = []
    for pidx in range(n_paths):
        path_frames = skeleton.paths[pidx]
        pi, pj = skeleton.path_pair_labels[pidx]
        valid = [int(f) for f in path_frames if f < qpos_clip.shape[0]]
        src_beh = skeleton.minima_behaviors[pi]
        dst_beh = skeleton.minima_behaviors[pj]
        src_color = color_map.get(src_beh, (0.5, 0.5, 0.5))
        dst_color = color_map.get(dst_beh, (0.5, 0.5, 0.5))
        ts, te = _transition_tints(src_color, dst_color)
        strip = render_filmstrip_for_frames(
            qpos_clip, valid, xml_path, dataset,
            n_poses=n_poses, gap_fraction=filmstrip_cfg.gap_fraction,
            tint_start=ts, tint_end=te,
        )
        path_strips.append((strip, f"M{pi}\u2192M{pj}", f"{src_beh}\u2192{dst_beh}"))

    # --- Normalize all strips to same pixel dimensions ---
    all_strips = [s for s, _, _ in minima_strips] + [s for s, _, _ in path_strips]
    box_rgb = (232, 232, 232)
    if all_strips:
        target_h = max(s.shape[0] for s in all_strips)
        target_w = max(s.shape[1] for s in all_strips)

        def _pad_to_size(img, th, tw):
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

    out_dir = os.path.dirname(output_path) or "."
    out_name = os.path.basename(output_path)
    base_name, ext = os.path.splitext(out_name)
    embed_name = base_name.replace("_filmstrip", "_embeddings") + ext
    embed_path = os.path.join(out_dir, embed_name)
    os.makedirs(out_dir, exist_ok=True)

    _NATURE_RC = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }

    # --- Figure 1: Embeddings (PCA, UMAP, t-SNE) side by side as squares ---
    with plt.rc_context(_NATURE_RC):
        fig_e, axes_e = plt.subplots(1, 3, figsize=(9.6, 3.4))
        fig_e.subplots_adjust(wspace=0.38, left=0.05, right=0.97, top=0.84, bottom=0.06)

        for i, (ax, coords, title) in enumerate([
            (axes_e[0], pca_coords, pca_title),
            (axes_e[1], umap_coords, umap_title),
            (axes_e[2], tsne_coords, tsne_title),
        ]):
            _overlay_skeleton_on_ax(
                ax, coords, frame_labels, skeleton, color_map,
                title, show_legend=(i == 0),
            )
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_box_aspect(1)
            ax.margins(0.05)

        fig_e.suptitle(
            f"Clip {clip_id} \u2014 Skeleton Embeddings (purity={skeleton.purity:.2f})",
            fontsize=9, fontweight="bold",
        )
        fig_e.savefig(embed_path, dpi=filmstrip_cfg.dpi, bbox_inches="tight",
                      facecolor="white", pad_inches=0.03)
        plt.close(fig_e)
    print(f"  Saved embeddings: {embed_path}")

    # --- Figure 2: Filmstrips (minima top, transitions bottom) ---
    if not minima_strips and not path_strips:
        return

    n_cols = max(n_minima, n_paths, 1)
    strip_w_in = 1.8
    fig_w = max(5.0, n_cols * strip_w_in + 0.8)
    fig_h = 4.5

    fig_f = plt.figure(figsize=(fig_w, fig_h))
    fig_f.patch.set_facecolor("white")

    gs_f = gridspec.GridSpec(
        2, n_cols, figure=fig_f,
        wspace=0.45, hspace=0.50,
        left=0.04, right=0.96, top=0.86, bottom=0.04,
    )

    filmstrip_axes = []

    for k, (strip, beh, idx) in enumerate(minima_strips):
        ax = fig_f.add_subplot(gs_f[0, k])
        strip_grey = _replace_white_bg(strip, box_rgb)
        ax.imshow(strip_grey)
        ax.set_axis_off()
        ax.set_title(f"Minimum M{idx}\n{beh}", fontsize=5, pad=3)
        filmstrip_axes.append(ax)

    for k in range(n_minima, n_cols):
        ax = fig_f.add_subplot(gs_f[0, k])
        ax.set_axis_off()

    for pidx, (strip, pair_label, beh_label) in enumerate(path_strips):
        ax = fig_f.add_subplot(gs_f[1, pidx])
        strip_grey = _replace_white_bg(strip, box_rgb)
        ax.imshow(strip_grey)
        ax.set_axis_off()
        ax.set_title(f"{pair_label}\n{beh_label}", fontsize=5, pad=3)
        filmstrip_axes.append(ax)

    for pidx in range(n_paths, n_cols):
        ax = fig_f.add_subplot(gs_f[1, pidx])
        ax.set_axis_off()

    fig_f.canvas.draw()
    renderer_f = fig_f.canvas.get_renderer()

    bboxes = []
    for ax in filmstrip_axes:
        tb = ax.get_tightbbox(renderer_f)
        if tb is not None:
            bboxes.append(tb.transformed(fig_f.transFigure.inverted()))
        else:
            bboxes.append(None)

    valid_heights = [bb.height for bb in bboxes if bb is not None]
    max_box_h = max(valid_heights) if valid_heights else 0.1
    pad_box = 0.008

    for bb in bboxes:
        if bb is None:
            continue
        h_diff = max_box_h - bb.height
        rect = FancyBboxPatch(
            (bb.x0 - pad_box, bb.y0 - pad_box - h_diff / 2),
            bb.width + 2 * pad_box,
            max_box_h + 2 * pad_box,
            boxstyle="round,pad=0.002,rounding_size=0.008",
            facecolor="#e8e8e8", edgecolor="#b0b0b0", linewidth=0.8,
            transform=fig_f.transFigure, zorder=-1,
        )
        fig_f.patches.append(rect)

    fig_f.suptitle(
        f"Clip {clip_id} \u2014 Density Minima & Transition Paths",
        fontsize=9, fontweight="bold",
    )
    fig_f.savefig(output_path, dpi=filmstrip_cfg.dpi, bbox_inches="tight",
                  facecolor="white", pad_inches=0.03)
    plt.close(fig_f)
    print(f"  Saved filmstrip: {output_path}")
