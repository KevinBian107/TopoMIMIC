"""Functions for creating nice video rendering"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

from topo_mimic.dmwk.utils.load_utils import load_rollout_from_source

from topo_mimic.dmwk.utils.rendering import (
    render_from_saved_rollout,
)


def _ensure_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2: arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1: arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0: arr = (arr*255).clip(0,255).astype(np.uint8)
        else: arr = arr.clip(0,255).astype(np.uint8)
    return arr


def _load_full_clip_frames(rodent_rollout_dir, clip_id, camera_name="close_profile"):
    ro = load_rollout_from_source(clip_id, rodent_rollout_dir)
    frames = render_from_saved_rollout(ro, camera_name=camera_name)
    return [_ensure_rgb_uint8(fr) for fr in frames]


def render_grid_clusters_side_by_side(
    cluster_chunks: dict[int, list[list[np.ndarray]]],
    out_path: str = "dm_videos/cluster_side_by_side.mp4",
    framerate: int = 30,
    max_samples_per_cluster: int | None = None,
    target_size: tuple[int, int] = (320, 240),
    pad_bg: tuple[int, int, int] = (255, 255, 255),
    gutter: int = 10,
    pad_x: int = 24,
    pad_y: int = 24,
    title: str | None = None,
    per_tile_labels: bool = True,
    n_cols: int | None = None,
    selected_clusters: list[int] | None = None,
    cluster_labels: dict[int, str] | None = None,
    cluster_clip_ids: dict[int, list[int]] | None = None,
    tile_labels: dict[tuple[int, int], str] | None = None,
) -> None:
    """
    Render cluster samples in a grid layout.

    Args:
        cluster_chunks: Dict mapping cluster_id to list of frame lists
        out_path: Output video path
        framerate: Frames per second
        max_samples_per_cluster: Max samples to show per cluster
        target_size: (width, height) per tile
        pad_bg: Background color RGB
        gutter: Gap between tiles
        pad_x: Horizontal padding
        pad_y: Vertical padding
        title: Optional title string
        per_tile_labels: Whether to show labels
        n_cols: Number of columns (None for auto)
        selected_clusters: Specific clusters to render
        cluster_labels: Dict mapping cluster_id to label string
        cluster_clip_ids: Dict mapping cluster_id to list of clip_ids for each sample
        tile_labels: Dict mapping (cluster_id, col) to per-cell label string (overrides default)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if selected_clusters is None:
        order = sorted(cluster_chunks.keys())
    else:
        order = [c for c in selected_clusters if c in cluster_chunks]
    if not order:
        raise ValueError("No clusters to render.")

    Wc, Hc = target_size
    try:
        font_label = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_title = ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font_label = ImageFont.load_default()
        font_title = ImageFont.load_default()

    def name_for_cluster(cid):
        if isinstance(cluster_labels, dict) and cid in cluster_labels:
            return str(cluster_labels[cid])
        return f"{cid}"

    label_h = 24 if per_tile_labels else 0
    title_h = 34 if title else 0

    samples_by_cluster = {}
    clip_ids_by_cluster = {}
    max_cols = 0
    for cid in order:
        samples = list(cluster_chunks[cid])
        clip_ids = list(cluster_clip_ids[cid]) if cluster_clip_ids and cid in cluster_clip_ids else [None] * len(samples)
        if max_samples_per_cluster is not None:
            samples = samples[:max_samples_per_cluster]
            clip_ids = clip_ids[:max_samples_per_cluster]
        samples_by_cluster[cid] = samples
        clip_ids_by_cluster[cid] = clip_ids
        max_cols = max(max_cols, len(samples))

    if n_cols is None:
        n_cols_eff = max_cols if max_cols > 0 else 1
    else:
        n_cols_eff = max(1, min(n_cols, max_cols if max_cols > 0 else n_cols))

    n_rows = len(order)

    cell_w = Wc
    cell_h = label_h + Hc
    W = pad_x*2 + n_cols_eff*cell_w + (n_cols_eff-1)*gutter
    H = pad_y*2 + title_h + n_rows*cell_h + (n_rows-1)*gutter

    def sample_len(s):
        try:
            return len(s)
        except:
            return 0

    T = 0
    for cid in order:
        for s in samples_by_cluster[cid][:n_cols_eff]:
            T = max(T, sample_len(s))
    if T == 0:
        raise ValueError("No frames found in the selected clusters/samples.")

    writer = imageio_ffmpeg.write_frames(out_path, (W, H), fps=framerate, codec="h264", macro_block_size=1)
    writer.send(None)

    def draw_centered(draw, text, x_center, y_center, font, fill=(0,0,0)):
        tw, th = draw.textbbox((0,0), text, font=font)[2:]
        draw.text((x_center - tw/2, y_center - th/2), text, font=font, fill=fill)

    for t in range(T):
        canvas = Image.new("RGB", (W, H), pad_bg)
        draw = ImageDraw.Draw(canvas)
        if title:
            draw_centered(draw, title, W//2, pad_y + title_h//2, font_title)
        y0 = pad_y + title_h

        for row, cid in enumerate(order):
            cname = name_for_cluster(cid)
            for col in range(n_cols_eff):
                x = pad_x + col*(cell_w + gutter)
                y = y0 + row*(cell_h + gutter)
                samples = samples_by_cluster[cid]
                clip_ids = clip_ids_by_cluster[cid]
                if col >= len(samples):
                    continue
                seq = samples[col]
                if len(seq) == 0:
                    continue
                fr = seq[t % len(seq)]
                img = Image.fromarray(_ensure_rgb_uint8(fr)).resize((Wc, Hc))
                if per_tile_labels:
                    if tile_labels and (cid, col) in tile_labels:
                        label_text = tile_labels[(cid, col)]
                    else:
                        clip_id = clip_ids[col] if col < len(clip_ids) else None
                        if clip_id is not None:
                            label_text = f"{cname} (clip {clip_id})"
                        else:
                            label_text = f"{cname} · Sample {col+1}"
                    draw_centered(draw, label_text, x + cell_w//2, y + label_h//2, font_label)
                canvas.paste(img, (x, y + label_h))
        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
