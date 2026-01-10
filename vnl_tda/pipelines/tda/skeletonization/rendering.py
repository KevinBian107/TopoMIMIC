"""Functions for creating nice video rendering"""

import os, h5py, numpy as np
import networkx as nx
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
import math

from vnl_tda.utils.load_utils import (
    load_from_h5py,
    get_aggregate_data,
    extract_clip_info,
    is_consolidated_format,
    load_rollout_from_source,
)

from vnl_tda.utils.rendering import (
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


def _load_dimo_edges(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 3: continue
            out.append((int(p[0]), int(p[1]), int(p[2])))
    return np.array(out, dtype=int) if out else np.empty((0,3), dtype=int)


def _descending_graph(edges, L=None):
    desc = edges[edges[:,2] == -1] if edges.size else np.empty((0,3), dtype=int)
    G = nx.DiGraph()
    for u,v,_ in desc:
        if L is not None and (u<0 or v<0 or u>=L or v>=L): continue
        G.add_edge(int(u), int(v))
    return G


def _paths_from_sources_to_minima(G, max_paths=None):
    nodes = list(G.nodes())
    minima = [n for n in nodes if G.out_degree(n)==0]
    sources = [n for n in nodes if G.in_degree(n)==0]
    paths = []
    for s in sources:
        v = s
        seen = set([v])
        seq = [v]
        while True:
            outs = list(G.successors(v))
            if len(outs)==0: break
            outs.sort()
            v = outs[0]
            if v in seen: break
            seen.add(v)
            seq.append(v)
        if len(seq)>=2 and seq[-1] in minima:
            paths.append((s, seq[-1], seq))
    paths.sort(key=lambda x: len(x[2]), reverse=True)
    if max_paths is not None:
        paths = paths[:max_paths]
    return paths


def _draw_centered_text(img, text, xy_center, font, fill=(0,0,0)):
    draw = ImageDraw.Draw(img)
    tw, th = draw.textbbox((0,0), text, font=font)[2:]
    draw.text((xy_center[0]-tw/2, xy_center[1]-th/2), text, font=font, fill=fill)
    

def _node_density_from_weighted_knn(G_knn, k_smallest=10):
    """Density from weighted kNN"""
    dens = {}
    for n in G_knn.nodes():
        wts = [d.get("weight", np.nan) for _, _, d in G_knn.edges(n, data=True)]
        wts = np.array([w for w in wts if np.isfinite(w)], dtype=float)
        dens[n] = float(np.partition(wts, min(k_smallest, len(wts))-1)[:min(k_smallest, len(wts))].mean()) if len(wts) else np.nan
    return dens


def _select_densest_nodes(G_knn, q=0.10, k_smallest=10):
    """Select the denses node"""
    nodes = list(G_knn.nodes())
    dens = _node_density_from_weighted_knn(G_knn, k_smallest)
    vals = np.array([dens.get(n, np.nan) for n in nodes], dtype=float)
    med = np.nanmedian(vals); vals = np.where(np.isnan(vals), med, vals)
    thr = np.quantile(vals, q)
    return [n for n, v in zip(nodes, vals) if v <= thr]


def _dense_frame_ranges(dense_nodes, total_frames, m=0, pad=25, min_gap=30):
    """Retrieve all the dense frames"""
    idx = np.clip(np.unique(np.array(dense_nodes, int) + m), 0, total_frames-1)
    if idx.size == 0: return []
    out, s = [], idx[0]
    for i in range(1, len(idx)):
        if idx[i] - idx[i-1] > min_gap:
            out.append((max(0, s - pad), min(total_frames-1, idx[i-1] + pad)))
            s = idx[i]
    out.append((max(0, s - pad), min(total_frames-1, idx[-1] + pad)))
    return out


def _gather_frames(frames, chunks):
    out = []
    for a, b in chunks: out.extend(frames[a:b+1])
    return out


def _resample_frames(frames, L):
    if len(frames) == 0:
        raise ValueError("Empty frame list for resampling.")
    idx = np.linspace(0, len(frames)-1, L).round().astype(int)
    return [frames[i] for i in idx]


def _concat_cluster_frames(cluster_chunks, max_samples_per_cluster=None):
    concatenated = {}
    for cid, vids in sorted(cluster_chunks.items()):
        take = vids if max_samples_per_cluster is None else vids[:max_samples_per_cluster]
        frames = [_ensure_rgb_uint8(frame) for chunk in take for frame in chunk]
        concatenated[cid] = frames
    return concatenated


def render_clip_grid(
    rodent_rollout_dir: str,
    clip_ids: list[int],
    rows: int,
    cols: int,
    out_path: str = "dm_videos/clip_grid.mp4",
    camera_name: str = "close_profile",
    fps: int = 60,
    target_size: tuple[int, int] = (320, 240),
    pad_bg: tuple[int, int, int] = (255, 255, 255),
    gutter: int = 10,
    pad_x: int = 24,
    pad_y: int = 24,
    title: str | None = None,
    per_tile_labels: bool = True,
    label_fmt: str = "Clip {id}",
) -> str:
    """
    Render raw clips side-by-side in a rows x cols grid.

    Args:
        rodent_rollout_dir: Directory containing clip HDF5 files
        clip_ids: List of clip IDs to render
        rows: Number of rows in grid
        cols: Number of columns in grid
        out_path: Output video path
        camera_name: Camera name for rendering
        fps: Frames per second
        target_size: (width, height) per tile
        pad_bg: Background color RGB
        gutter: Gap between tiles in pixels
        pad_x: Horizontal padding
        pad_y: Vertical padding
        title: Optional title string
        per_tile_labels: Whether to show clip labels
        label_fmt: Format string for labels

    Returns:
        Path to output video
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load all clips
    frames_list = []
    lengths = []
    for cid in clip_ids:
        frs = _load_full_clip_frames(rodent_rollout_dir, cid)
        frs = [_ensure_rgb_uint8(fr) for fr in frs]
        frames_list.append(frs)
        lengths.append(len(frs))

    # Grid bookkeeping
    k = len(clip_ids)
    Wc, Hc = target_size
    try:
        font_label = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_title = ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font_label = ImageFont.load_default()
        font_title = ImageFont.load_default()

    label_h = 24 if per_tile_labels else 0
    title_h = 34 if title else 0
    cell_w = Wc
    cell_h = label_h + Hc

    # Canvas size
    W = pad_x*2 + cols*cell_w + (cols-1)*gutter
    H = pad_y*2 + title_h + rows*cell_h + (rows-1)*gutter

    # Timeline length = max clip length
    T = max(lengths) if lengths else 0
    if T == 0:
        raise ValueError("No frames to render (all clips empty?).")

    # Video writer
    writer = imageio_ffmpeg.write_frames(out_path, (W, H), fps=fps, codec="h264", macro_block_size=1)
    writer.send(None)

    # Helper to draw centered text
    def draw_centered(draw, text, x_center, y_center, font, fill=(0,0,0)):
        tw, th = draw.textbbox((0,0), text, font=font)[2:]
        draw.text((x_center - tw/2, y_center - th/2), text, font=font, fill=fill)

    # Render loop
    for t in range(T):
        canvas = Image.new("RGB", (W, H), pad_bg)
        draw = ImageDraw.Draw(canvas)

        # Title
        if title:
            draw_centered(draw, title, W//2, pad_y + title_h//2, font_title, fill=(0,0,0))

        # Grid
        y0 = pad_y + title_h
        for r in range(rows):
            for c in range(cols):
                idx = r*cols + c
                x = pad_x + c*(cell_w + gutter)
                y = y0 + r*(cell_h + gutter)

                # Label band
                if per_tile_labels:
                    # Label text if clip exists, else show a faint placeholder band
                    if idx < k:
                        label = label_fmt.format(id=clip_ids[idx])
                        draw_centered(draw, label, x + cell_w//2, y + label_h//2, font_label, fill=(0,0,0))
                    else:
                        # Blank cell: draw a light band for visual consistency (optional)
                        pass

                # Image area
                if idx < k:
                    seq = frames_list[idx]
                    fr = seq[t] if t < len(seq) else seq[-1]
                    img = Image.fromarray(fr).resize((Wc, Hc))
                    canvas.paste(img, (x, y + label_h))
                else:
                    # Leave blank for cells beyond provided clips
                    pass

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
    return out_path


def render_crossclip_paths_to_minima(
    rodent_rollout_dir: str,
    clip_ids: list[int],
    edge_path: str,
    out_path: str = "dm_videos/crossclip_paths.mp4",
    camera_name: str = "close_profile",
    fps: int = 60,
    target_size: tuple[int, int] = (320, 240),
    pad_bg: tuple[int, int, int] = (255, 255, 255),
    gutter: int = 10,
    pad_x: int = 24,
    pad_y: int = 24,
    title: str | None = None,
    per_tile_labels: bool = True,
    num_paths: int = 4,
) -> str:
    """
    Render descending paths from sources to minima across clips.

    Args:
        rodent_rollout_dir: Directory containing clip HDF5 files
        clip_ids: List of clip IDs
        edge_path: Path to edge file from DM computation
        out_path: Output video path
        camera_name: Camera name for rendering
        fps: Frames per second
        target_size: (width, height) per tile
        pad_bg: Background color RGB
        gutter: Gap between tiles
        pad_x: Horizontal padding
        pad_y: Vertical padding
        title: Optional title string
        per_tile_labels: Whether to show path labels
        num_paths: Number of paths to render

    Returns:
        Path to output video
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    concat = []
    clip_offsets = []
    off = 0
    for cid in clip_ids:
        frs = _load_full_clip_frames(rodent_rollout_dir, cid, camera_name=camera_name)
        concat.extend(frs)
        clip_offsets.append((cid, off, off+len(frs)))
        off += len(frs)
    L = len(concat)
    edges = _load_dimo_edges(edge_path)
    G = _descending_graph(edges, L=L)
    pm = _paths_from_sources_to_minima(G, max_paths=num_paths)
    per_path_frames = []
    labels = []
    for s, m, seq in pm:
        idx = np.asarray(seq, dtype=int)
        idx = idx[(idx>=0)&(idx<L)]
        per_path_frames.append([concat[i] for i in idx])
        labels.append(f"src {s} → min {m} | L={len(idx)}")
    if not per_path_frames or max((len(s) for s in per_path_frames), default=0) == 0:
        raise ValueError("No frames collected for the selected paths.")
    nC = len(per_path_frames)
    Wc, Hc = target_size
    try:
        font_label = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_title = ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font_label = ImageFont.load_default()
        font_title = ImageFont.load_default()
    label_h = 24 if per_tile_labels else 0
    title_h = 34 if title else 0
    cell_w = Wc
    cell_h = label_h + Hc
    W = pad_x*2 + nC*cell_w + (nC-1)*gutter
    H = pad_y*2 + title_h + cell_h
    T = max(len(seq) for seq in per_path_frames)
    writer = imageio_ffmpeg.write_frames(out_path, (W, H), fps=fps, codec="h264", macro_block_size=1)
    writer.send(None)
    def draw_centered(draw, text, x_center, y_center, font, fill=(0,0,0)):
        tw, th = draw.textbbox((0,0), text, font=font)[2:]
        draw.text((x_center - tw/2, y_center - th/2), text, font=font, fill=fill)
    for t in range(T):
        canvas = Image.new("RGB", (W, H), pad_bg)
        draw = ImageDraw.Draw(canvas)
        if title:
            draw_centered(draw, title, W//2, pad_y + title_h//2, font_title, fill=(0,0,0))
        y0 = pad_y + title_h
        for j, seq in enumerate(per_path_frames):
            x = pad_x + j*(cell_w + gutter)
            if per_tile_labels:
                draw_centered(draw, labels[j], x + cell_w//2, y0 + label_h//2, font_label, fill=(0,0,0))
            fr = seq[t] if t < len(seq) else seq[-1] if seq else None
            if fr is not None:
                img = Image.fromarray(fr).resize((Wc, Hc))
                canvas.paste(img, (x, y0 + label_h))
        writer.send(np.asarray(canvas, dtype=np.uint8))
    writer.close()
    return out_path


def render_original_plus_top2_cycles(
    rodent_rollout_dir: str,
    clip_id: int,
    cycles: list[list[int]],
    out_path: str = "dm_videos/cycles_with_original.mp4",
    camera_name: str = "close_profile",
    fps: int = 60,
    target_size: tuple[int, int] = (320, 240),
    title: str | None = None,
    col_labels: bool = True,
    pad_bg: tuple[int, int, int] = (255, 255, 255),
    sep_color: tuple[int, int, int] = (0, 0, 0),
    sep_w: int = 6,
) -> str:
    """
    Render original clip alongside top cycles.

    Args:
        rodent_rollout_dir: Directory containing clip HDF5 files
        clip_id: Clip ID to render
        cycles: List of cycle index lists
        out_path: Output video path
        camera_name: Camera name for rendering
        fps: Frames per second
        target_size: (width, height) per tile
        title: Optional title string
        col_labels: Whether to show column labels
        pad_bg: Background color RGB
        sep_color: Separator line color RGB
        sep_w: Separator width in pixels

    Returns:
        Path to output video
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames_full = _load_full_clip_frames(rodent_rollout_dir, clip_id, camera_name=camera_name)
    T = len(frames_full)

    top2 = sorted([list(map(int, c)) for c in cycles if len(c)], key=len, reverse=True)[:1]
    per_cycle = []
    for cyc in top2:
        idx = np.array(cyc, dtype=int)
        idx = idx[(idx >= 0) & (idx < T)]
        per_cycle.append([frames_full[i] for i in idx])

    # if not per_cycle:
    #     raise ValueError("No valid cycles to render.")

    Wc, Hc = target_size
    nC = len(per_cycle)
    Lmax = max([len(seq) for seq in per_cycle] + [1])
    total_len = max(T, Lmax)

    try:
        font_col = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_title = ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font_col = ImageFont.load_default()
        font_title = ImageFont.load_default()

    pad_x, pad_y = 40, 30
    gutter = 10
    col_h = 40 if col_labels else 0
    title_h = 34 if title else 0

    left_w = Wc
    cycles_w = nC*Wc + max(0, nC-1)*gutter
    W = pad_x*2 + left_w + gutter + sep_w + gutter + cycles_w
    H = pad_y*2 + title_h + col_h + Hc

    writer = imageio_ffmpeg.write_frames(out_path, (W, H), fps=fps, codec='h264', macro_block_size=1)
    writer.send(None)

    for t in range(total_len):
        canvas = Image.new("RGB", (W, H), pad_bg)
        draw = ImageDraw.Draw(canvas)

        if title:
            _draw_centered_text(canvas, title, (W//2, pad_y + title_h//2), font_title, fill=(0,0,0))

        y0 = pad_y + title_h

        x_left = pad_x
        if col_labels:
            _draw_centered_text(canvas, "Original", (x_left + Wc//2, y0 + col_h//2), font_col, fill=(0,0,0))
        fr_left = frames_full[t] if t < T else frames_full[-1]
        img_left = Image.fromarray(fr_left).resize((Wc, Hc))
        canvas.paste(img_left, (x_left, y0 + col_h))

        x_sep = pad_x + left_w + gutter
        draw.rectangle([x_sep, y0, x_sep + sep_w - 1, y0 + col_h + Hc], fill=sep_color)

        x_cycles_start = x_sep + sep_w + gutter
        for j, seq in enumerate(per_cycle):
            x = x_cycles_start + j*(Wc + gutter)
            if col_labels:
                _draw_centered_text(canvas, f"Cycle {j}", (x + Wc//2, y0 + col_h//2), font_col, fill=(0,0,0))
            fr = seq[t] if t < len(seq) else seq[-1] if len(seq) else frames_full[-1]
            img = Image.fromarray(fr).resize((Wc, Hc))
            canvas.paste(img, (x, y0 + col_h))

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
    return out_path


def render_crossclip_cycles_side_by_side(
    rodent_rollout_dir: str,
    clip_ids: list[int],
    cycles: list[list[int]],
    out_path: str = "dm_videos/crossclip_cycles.mp4",
    camera_name: str = "close_profile",
    fps: int = 60,
    target_size: tuple[int, int] = (320, 240),
    pad_bg: tuple[int, int, int] = (255, 255, 255),
    gutter: int = 10,
    pad_x: int = 24,
    pad_y: int = 24,
    title: str | None = None,
    per_tile_labels: bool = True,
    cycle_labels: list[str] | None = None,
) -> str:
    """
    Render cycles side by side across concatenated clips.

    Args:
        rodent_rollout_dir: Directory containing clip HDF5 files
        clip_ids: List of clip IDs
        cycles: List of cycle index lists
        out_path: Output video path
        camera_name: Camera name for rendering
        fps: Frames per second
        target_size: (width, height) per tile
        pad_bg: Background color RGB
        gutter: Gap between tiles
        pad_x: Horizontal padding
        pad_y: Vertical padding
        title: Optional title string
        per_tile_labels: Whether to show labels
        cycle_labels: Custom labels for each cycle

    Returns:
        Path to output video
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    concat = []
    for cid in clip_ids:
        concat.extend(_load_full_clip_frames(rodent_rollout_dir, cid, camera_name=camera_name))
    L = len(concat)

    per_cycle_frames = []
    for cyc in cycles:
        idx = np.asarray(cyc, dtype=int)
        idx = idx[(idx >= 0) & (idx < L)]
        per_cycle_frames.append([concat[i] for i in idx])

    if not per_cycle_frames or max((len(s) for s in per_cycle_frames), default=0) == 0:
        raise ValueError("No frames collected for the provided cycles.")

    nC = len(per_cycle_frames)
    Wc, Hc = target_size
    try:
        font_label = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_title = ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font_label = ImageFont.load_default()
        font_title = ImageFont.load_default()

    label_h = 24 if per_tile_labels else 0
    title_h = 34 if title else 0
    cell_w = Wc
    cell_h = label_h + Hc

    W = pad_x*2 + nC*cell_w + (nC-1)*gutter
    H = pad_y*2 + title_h + cell_h

    T = max(len(seq) for seq in per_cycle_frames)

    writer = imageio_ffmpeg.write_frames(out_path, (W, H), fps=fps, codec="h264", macro_block_size=1)
    writer.send(None)

    def draw_centered(draw, text, x_center, y_center, font, fill=(0,0,0)):
        tw, th = draw.textbbox((0,0), text, font=font)[2:]
        draw.text((x_center - tw/2, y_center - th/2), text, font=font, fill=fill)

    for t in range(T):
        canvas = Image.new("RGB", (W, H), pad_bg)
        draw = ImageDraw.Draw(canvas)

        if title:
            draw_centered(draw, title, W//2, pad_y + title_h//2, font_title, fill=(0,0,0))

        y0 = pad_y + title_h
        for j, seq in enumerate(per_cycle_frames):
            x = pad_x + j*(cell_w + gutter)
            if per_tile_labels:
                label = cycle_labels[j] if (cycle_labels and j < len(cycle_labels)) else f"Cycle {j}"
                draw_centered(draw, label, x + cell_w//2, y0 + label_h//2, font_label, fill=(0,0,0))
            fr = seq[t] if t < len(seq) else seq[-1] if seq else None
            if fr is not None:
                img = Image.fromarray(fr).resize((Wc, Hc))
                canvas.paste(img, (x, y0 + label_h))

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
    return out_path


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
                    clip_id = clip_ids[col] if col < len(clip_ids) else None
                    if clip_id is not None:
                        label_text = f"{cname} (clip {clip_id})"
                    else:
                        label_text = f"{cname} · Sample {col+1}"
                    draw_centered(draw, label_text, x + cell_w//2, y + label_h//2, font_label)
                canvas.paste(img, (x, y + label_h))
        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()