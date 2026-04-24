"""Rendering functions for similar cluster pair verification and stationary analysis."""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
from tqdm import tqdm

from .config import GranularityConfig
from .redundancy import RedundancyResult
from .dm_multi_persistence import DMExtractionResult
from .stationary_analysis import StationaryAnalysisOutput, StationaryClipInfo


def _ensure_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def _load_clip_frames(
    rodent_rollout_dir: str,
    clip_id: int,
    camera_name: str = "close_profile",
) -> list[np.ndarray]:
    """Load rendered frames for a clip."""
    from topo_mimic.extraction.utils.load_utils import load_rollout_from_source
    from topo_mimic.extraction.utils.rendering import render_from_saved_rollout

    ro = load_rollout_from_source(clip_id, rodent_rollout_dir)
    frames = render_from_saved_rollout(ro, camera_name=camera_name)
    return [_ensure_rgb_uint8(fr) for fr in frames]


def _get_fonts():
    """Get fonts for rendering labels."""
    try:
        font_label = ImageFont.truetype("DejaVuSans.ttf", 16)
        font_title = ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font_label = ImageFont.load_default()
        font_title = ImageFont.load_default()
    return font_label, font_title


def _draw_centered_text(draw, text, x_center, y_center, font, fill=(0, 0, 0)):
    """Draw centered text."""
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    draw.text((x_center - tw / 2, y_center - th / 2), text, font=font, fill=fill)


def render_similar_pair(
    cluster_i_frames: list[list[np.ndarray]],
    cluster_j_frames: list[list[np.ndarray]],
    cluster_i_id: int,
    cluster_j_id: int,
    dtw_distance: float,
    out_path: str,
    max_samples: int = 3,
    target_size: tuple[int, int] = (320, 240),
    framerate: int = 30,
    pad_bg: tuple[int, int, int] = (255, 255, 255),
) -> str:
    """
    Render a similar cluster pair side-by-side for visual comparison.

    Args:
        cluster_i_frames: List of frame sequences for cluster i
        cluster_j_frames: List of frame sequences for cluster j
        cluster_i_id: Cluster i ID
        cluster_j_id: Cluster j ID
        dtw_distance: DTW distance between clusters
        out_path: Output video path
        max_samples: Max samples to show per cluster
        target_size: (width, height) per tile
        framerate: Video framerate
        pad_bg: Background color

    Returns:
        Path to output video
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    samples_i = cluster_i_frames[:max_samples]
    samples_j = cluster_j_frames[:max_samples]

    n_samples = max(len(samples_i), len(samples_j))
    if n_samples == 0:
        raise ValueError("No samples to render")

    Wc, Hc = target_size
    font_label, font_title = _get_fonts()

    title_h = 40
    label_h = 28
    sep_w = 6
    gutter = 8
    pad_x, pad_y = 20, 16

    col_w = Wc
    col_h = label_h + Hc

    W = pad_x * 2 + n_samples * col_w + (n_samples - 1) * gutter + sep_w + n_samples * col_w + (n_samples - 1) * gutter
    H = pad_y * 2 + title_h + col_h

    max_len = 0
    for s in samples_i + samples_j:
        if len(s) > max_len:
            max_len = len(s)

    if max_len == 0:
        raise ValueError("All samples have 0 frames")

    writer = imageio_ffmpeg.write_frames(
        out_path, (W, H), fps=framerate, codec="h264", macro_block_size=1
    )
    writer.send(None)

    title = f"Similar Pair: C{cluster_i_id} vs C{cluster_j_id} (DTW={dtw_distance:.4f})"

    for t in range(max_len):
        canvas = Image.new("RGB", (W, H), pad_bg)
        draw = ImageDraw.Draw(canvas)

        _draw_centered_text(draw, title, W // 2, pad_y + title_h // 2, font_title, fill=(0, 0, 0))

        y0 = pad_y + title_h

        for idx, seq in enumerate(samples_i):
            x = pad_x + idx * (col_w + gutter)
            _draw_centered_text(
                draw, f"C{cluster_i_id} #{idx+1}", x + col_w // 2, y0 + label_h // 2, font_label
            )
            if len(seq) > 0:
                fr = seq[t % len(seq)]
                img = Image.fromarray(_ensure_rgb_uint8(fr)).resize((Wc, Hc))
                canvas.paste(img, (x, y0 + label_h))

        x_sep = pad_x + n_samples * col_w + (n_samples - 1) * gutter
        draw.rectangle([x_sep, y0, x_sep + sep_w - 1, y0 + col_h], fill=(100, 100, 100))

        for idx, seq in enumerate(samples_j):
            x = x_sep + sep_w + gutter + idx * (col_w + gutter)
            _draw_centered_text(
                draw, f"C{cluster_j_id} #{idx+1}", x + col_w // 2, y0 + label_h // 2, font_label
            )
            if len(seq) > 0:
                fr = seq[t % len(seq)]
                img = Image.fromarray(_ensure_rgb_uint8(fr)).resize((Wc, Hc))
                canvas.paste(img, (x, y0 + label_h))

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
    return out_path


def render_all_similar_pairs(
    result: RedundancyResult,
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    rodent_rollout_dir: str,
    output_dir: Path,
    config: GranularityConfig,
) -> list[str]:
    """
    Render videos for all top similar cluster pairs.

    Args:
        result: RedundancyResult containing similar pairs
        cluster_dict: Dict mapping cluster_id to list of (clip_id, frame_indices)
        rodent_rollout_dir: Path to rollout data
        output_dir: Output directory for videos
        config: Configuration object

    Returns:
        List of output video paths
    """
    from tqdm import tqdm

    output_dir = Path(output_dir)
    videos_dir = output_dir / "similar_pairs_videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    rendered_paths = []
    max_samples = config.rendering.max_samples

    print(f"\n  Rendering {len(result.top_similar_pairs)} similar pairs...")

    clip_frame_cache = {}

    for pair_idx, (ci, cj, dist) in enumerate(tqdm(result.top_similar_pairs, desc="Rendering pairs")):
        entries_i = cluster_dict.get(ci, [])
        entries_j = cluster_dict.get(cj, [])

        if not entries_i or not entries_j:
            print(f"    Skipping pair C{ci}-C{cj}: empty cluster")
            continue

        def load_cluster_samples(entries, max_n):
            samples = []
            for clip_id, frame_indices in entries[:max_n]:
                clip_id = int(clip_id)
                if clip_id not in clip_frame_cache:
                    try:
                        clip_frame_cache[clip_id] = _load_clip_frames(
                            rodent_rollout_dir, clip_id
                        )
                    except Exception as e:
                        print(f"    Warning: Failed to load clip {clip_id}: {e}")
                        clip_frame_cache[clip_id] = []

                all_frames = clip_frame_cache[clip_id]
                if len(all_frames) == 0:
                    continue

                frames = np.asarray(frame_indices).flatten()
                frames = np.sort(frames)
                valid_frames = frames[(frames >= 0) & (frames < len(all_frames))]

                if len(valid_frames) < 3:
                    continue

                sample_frames = [all_frames[f] for f in valid_frames]
                samples.append(sample_frames)

            return samples

        samples_i = load_cluster_samples(entries_i, max_samples)
        samples_j = load_cluster_samples(entries_j, max_samples)

        if not samples_i or not samples_j:
            print(f"    Skipping pair C{ci}-C{cj}: no valid samples")
            continue

        out_path = str(
            videos_dir / f"pair_{pair_idx:02d}_C{ci}_C{cj}_dtw{dist:.4f}.mp4"
        )

        try:
            render_similar_pair(
                samples_i,
                samples_j,
                ci,
                cj,
                dist,
                out_path,
                max_samples=max_samples,
                target_size=config.rendering.target_size,
                framerate=config.rendering.framerate,
            )
            rendered_paths.append(out_path)
            print(f"    Rendered: {out_path}")
        except Exception as e:
            print(f"    Error rendering pair C{ci}-C{cj}: {e}")

    return rendered_paths


def render_similar_pairs_grid(
    redundancy_results: dict[tuple[float, int], "RedundancyResult"],
    rodent_rollout_dir: str,
    out_path: str,
    config: GranularityConfig,
) -> str:
    """
    Render a grid of similar cluster pairs.
    Rows = persistence values, Cols = number of clusters.
    Each cell shows the top similar pair side-by-side.

    Args:
        redundancy_results: Dict mapping (pers_delta, n_clusters) to RedundancyResult
        rodent_rollout_dir: Path to rollout data
        out_path: Output video path
        config: Configuration object

    Returns:
        Path to output video
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pers_values = sorted(set(k[0] for k in redundancy_results.keys()))
    k_values = sorted(set(k[1] for k in redundancy_results.keys()))

    n_rows = len(pers_values)
    n_cols = len(k_values)

    if n_rows == 0 or n_cols == 0:
        raise ValueError("No redundancy results to render")

    print(f"  Building grid: {n_rows} persistence values x {n_cols} cluster counts")

    all_clips_needed = set()
    for (pers_delta, n_clusters), result in redundancy_results.items():
        if result.top_similar_pairs:
            ci, cj, _ = result.top_similar_pairs[0]
            for cid in [ci, cj]:
                entries = result.cluster_dict.get(cid, [])
                for clip_id, _ in entries[:1]:
                    all_clips_needed.add(int(clip_id))

    print(f"  Loading frames for {len(all_clips_needed)} clips...")
    clip_frame_cache = {}
    for clip_id in tqdm(all_clips_needed, desc="Loading clips"):
        try:
            clip_frame_cache[clip_id] = _load_clip_frames(rodent_rollout_dir, clip_id)
        except Exception as e:
            print(f"    Warning: Failed to load clip {clip_id}: {e}")
            clip_frame_cache[clip_id] = []

    takens_offset = (config.takens.m - 1) * config.takens.tau

    grid_data = []
    grid_labels = []
    max_cycle_len = 1

    for pers_delta in pers_values:
        row_data = []
        row_labels = []

        for n_clusters in k_values:
            result = redundancy_results.get((pers_delta, n_clusters))

            if result is None or not result.top_similar_pairs:
                row_data.append(([], []))
                row_labels.append(None)
                continue

            ci, cj, dtw_dist = result.top_similar_pairs[0]

            def get_first_sample(cluster_id):
                entries = result.cluster_dict.get(cluster_id, [])
                if not entries:
                    return []

                clip_id, frame_indices = entries[0]
                clip_id = int(clip_id)
                all_frames = clip_frame_cache.get(clip_id, [])

                if not all_frames:
                    return []

                frames = np.asarray(frame_indices).flatten()
                frames = np.sort(frames)
                valid_frames = frames[(frames >= 0) & (frames < len(all_frames))]

                if len(valid_frames) < 2:
                    return []

                return [all_frames[f] for f in valid_frames]

            frames_i = get_first_sample(ci)
            frames_j = get_first_sample(cj)

            row_data.append((frames_i, frames_j))
            row_labels.append(f"C{ci}-C{cj}\n{dtw_dist:.3f}")

            max_cycle_len = max(max_cycle_len, len(frames_i), len(frames_j))

        grid_data.append(row_data)
        grid_labels.append(row_labels)

    Wc, Hc = config.rendering.target_size
    cell_w = Wc * 2 + 4
    cell_h = Hc

    font_label, font_title = _get_fonts()

    title_h = 40
    row_label_w = 80
    col_label_h = 40
    cell_label_h = 24
    gutter = 8
    pad_x, pad_y = 16, 12

    W = pad_x * 2 + row_label_w + n_cols * cell_w + (n_cols - 1) * gutter
    H = pad_y * 2 + title_h + col_label_h + n_rows * (cell_label_h + cell_h) + (n_rows - 1) * gutter

    writer = imageio_ffmpeg.write_frames(
        out_path, (W, H), fps=config.rendering.framerate, codec="h264", macro_block_size=1
    )
    writer.send(None)

    title = "Similar Cluster Pairs: Persistence × Cluster Count"

    for t in range(max_cycle_len):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        _draw_centered_text(draw, title, W // 2, pad_y + title_h // 2, font_title, fill=(0, 0, 0))

        y0 = pad_y + title_h

        for col_idx, k in enumerate(k_values):
            x = pad_x + row_label_w + col_idx * (cell_w + gutter) + cell_w // 2
            _draw_centered_text(draw, f"k={k}", x, y0 + col_label_h // 2, font_label)

        for row_idx, pers_delta in enumerate(pers_values):
            y = y0 + col_label_h + row_idx * (cell_label_h + cell_h + gutter)

            _draw_centered_text(
                draw, f"δ={pers_delta:.2f}",
                pad_x + row_label_w // 2, y + (cell_label_h + cell_h) // 2, font_label
            )

            row_data = grid_data[row_idx]
            row_label = grid_labels[row_idx]

            for col_idx, ((frames_i, frames_j), label) in enumerate(zip(row_data, row_label)):
                x = pad_x + row_label_w + col_idx * (cell_w + gutter)

                if label:
                    _draw_centered_text(draw, label, x + cell_w // 2, y + cell_label_h // 2, font_label, fill=(80, 80, 80))

                y_img = y + cell_label_h

                if frames_i:
                    fr = frames_i[t % len(frames_i)]
                    img = Image.fromarray(_ensure_rgb_uint8(fr)).resize((Wc, Hc))
                    canvas.paste(img, (x, y_img))
                else:
                    draw.rectangle([x, y_img, x + Wc - 1, y_img + Hc - 1], fill=(220, 220, 220))

                if frames_j:
                    fr = frames_j[t % len(frames_j)]
                    img = Image.fromarray(_ensure_rgb_uint8(fr)).resize((Wc, Hc))
                    canvas.paste(img, (x + Wc + 4, y_img))
                else:
                    draw.rectangle([x + Wc + 4, y_img, x + cell_w - 1, y_img + Hc - 1], fill=(220, 220, 220))

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
    return out_path


def render_filtered_cycles_grid(
    stationary_clips: list[StationaryClipInfo],
    dm_results: dict[float, DMExtractionResult],
    clip_indices: list[int],
    rodent_rollout_dir: str,
    out_path: str,
    config: GranularityConfig,
    max_samples: int = 5,
) -> str:
    """
    Render cycles that get FILTERED OUT at each persistence transition.
    Rows = persistence transitions, Cols = clips that were lost at that transition.

    Args:
        stationary_clips: List of stationary clip info
        dm_results: Dict mapping pers_delta to DMExtractionResult
        clip_indices: Global clip indices
        rodent_rollout_dir: Path to rollout data
        out_path: Output video path
        config: Configuration object
        max_samples: Max samples (columns) per row

    Returns:
        Path to output video
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pers_values = sorted(dm_results.keys())
    if len(pers_values) < 2:
        raise ValueError("Need at least 2 persistence values for transition analysis")

    takens_offset = (config.takens.m - 1) * config.takens.tau

    stationary_local_indices = {c.clip_idx for c in stationary_clips}

    clips_with_cycles = {}
    for pers_delta in pers_values:
        dm_result = dm_results[pers_delta]
        clips_with_cycles[pers_delta] = set()
        for local_idx in stationary_local_indices:
            if local_idx < len(dm_result.all_cycle_dm):
                cycles = dm_result.all_cycle_dm[local_idx]
                if cycles and len(cycles) > 0:
                    clips_with_cycles[pers_delta].add(local_idx)

    transitions = []
    lost_clips_per_transition = []
    for i in range(len(pers_values) - 1):
        pers_low = pers_values[i]
        pers_high = pers_values[i + 1]
        lost = clips_with_cycles[pers_low] - clips_with_cycles[pers_high]
        transitions.append((pers_low, pers_high))
        lost_clips_per_transition.append(list(lost)[:max_samples])

    n_rows = len(transitions)
    n_cols = max(len(lost) for lost in lost_clips_per_transition) if lost_clips_per_transition else 0

    if n_rows == 0 or n_cols == 0:
        raise ValueError("No filtered cycles to render")

    all_clips_to_load = set()
    for lost_list in lost_clips_per_transition:
        all_clips_to_load.update(lost_list)

    print(f"  Loading frames for {len(all_clips_to_load)} clips with filtered cycles...")
    clip_frame_cache = {}
    for local_idx in tqdm(all_clips_to_load, desc="Loading clips"):
        global_idx = clip_indices[local_idx]
        try:
            clip_frame_cache[local_idx] = _load_clip_frames(rodent_rollout_dir, global_idx)
        except Exception as e:
            print(f"    Warning: Failed to load clip {global_idx}: {e}")
            clip_frame_cache[local_idx] = []

    grid_data = []
    grid_clip_ids = []
    max_cycle_len = 1

    for row_idx, (pers_low, pers_high) in enumerate(transitions):
        dm_result = dm_results[pers_low]
        lost_clips = lost_clips_per_transition[row_idx]
        row_data = []
        row_clip_ids = []

        for local_idx in lost_clips:
            all_frames = clip_frame_cache.get(local_idx, [])
            global_idx = clip_indices[local_idx]

            if local_idx >= len(dm_result.all_cycle_dm) or not all_frames:
                row_data.append([])
                row_clip_ids.append(global_idx)
                continue

            cycles = dm_result.all_cycle_dm[local_idx]
            if not cycles:
                row_data.append([])
                row_clip_ids.append(global_idx)
                continue

            longest_cycle = max(cycles, key=len)
            cycle_frames = np.asarray(longest_cycle, dtype=int) + takens_offset
            valid_frames = cycle_frames[(cycle_frames >= 0) & (cycle_frames < len(all_frames))]

            if len(valid_frames) < 2:
                row_data.append([])
                row_clip_ids.append(global_idx)
                continue

            sample_frames = [all_frames[f] for f in valid_frames]
            row_data.append(sample_frames)
            row_clip_ids.append(global_idx)
            max_cycle_len = max(max_cycle_len, len(sample_frames))

        while len(row_data) < n_cols:
            row_data.append([])
            row_clip_ids.append(None)

        grid_data.append(row_data)
        grid_clip_ids.append(row_clip_ids)

    Wc, Hc = config.rendering.target_size
    font_label, font_title = _get_fonts()

    title_h = 40
    row_label_w = 100
    col_label_h = 28
    gutter = 6
    pad_x, pad_y = 16, 12

    W = pad_x * 2 + row_label_w + n_cols * Wc + (n_cols - 1) * gutter
    H = pad_y * 2 + title_h + col_label_h + n_rows * Hc + (n_rows - 1) * gutter

    writer = imageio_ffmpeg.write_frames(
        out_path, (W, H), fps=config.rendering.framerate, codec="h264", macro_block_size=1
    )
    writer.send(None)

    title = "Cycles Filtered Out at Each Persistence Transition"

    for t in range(max_cycle_len):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        _draw_centered_text(draw, title, W // 2, pad_y + title_h // 2, font_title, fill=(0, 0, 0))

        y0 = pad_y + title_h

        for col_idx in range(n_cols):
            x = pad_x + row_label_w + col_idx * (Wc + gutter) + Wc // 2
            _draw_centered_text(draw, f"Sample {col_idx + 1}", x, y0 + col_label_h // 2, font_label)

        for row_idx, (pers_low, pers_high) in enumerate(transitions):
            y = y0 + col_label_h + row_idx * (Hc + gutter)

            _draw_centered_text(
                draw, f"δ {pers_low:.2f}→{pers_high:.2f}",
                pad_x + row_label_w // 2, y + Hc // 2, font_label
            )

            row_data = grid_data[row_idx]
            row_clips = grid_clip_ids[row_idx]

            for col_idx, (sample_frames, clip_id) in enumerate(zip(row_data, row_clips)):
                x = pad_x + row_label_w + col_idx * (Wc + gutter)

                if sample_frames:
                    fr = sample_frames[t % len(sample_frames)]
                    img = Image.fromarray(_ensure_rgb_uint8(fr)).resize((Wc, Hc))
                    canvas.paste(img, (x, y))
                    if clip_id is not None:
                        draw.text((x + 4, y + 4), f"#{clip_id}", font=font_label, fill=(255, 255, 255))
                else:
                    draw.rectangle([x, y, x + Wc - 1, y + Hc - 1], fill=(220, 220, 220))

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
    return out_path


def render_stationary_analysis(
    stationary_output: StationaryAnalysisOutput,
    dm_results: dict[float, DMExtractionResult],
    clip_indices: list[int],
    rodent_rollout_dir: str,
    output_dir: Path,
    config: GranularityConfig,
) -> list[str]:
    """
    Render videos for stationary analysis visualization.
    Shows cycles that get filtered out at each persistence transition.

    Args:
        stationary_output: Output from stationary analysis
        dm_results: Dict mapping pers_delta to DMExtractionResult
        clip_indices: Global clip indices
        rodent_rollout_dir: Path to rollout data
        output_dir: Output directory for videos
        config: Configuration object

    Returns:
        List of output video paths
    """
    output_dir = Path(output_dir)
    videos_dir = output_dir / "stationary_videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    rendered_paths = []

    if not stationary_output.stationary_clips:
        print("  No stationary clips to render")
        return rendered_paths

    out_path = str(videos_dir / "filtered_cycles_grid.mp4")

    try:
        print(f"  Rendering filtered cycles grid...")
        render_filtered_cycles_grid(
            stationary_output.stationary_clips,
            dm_results,
            clip_indices,
            rodent_rollout_dir,
            out_path,
            config,
            max_samples=config.rendering.max_samples,
        )
        rendered_paths.append(out_path)
        print(f"  Rendered: {out_path}")
    except Exception as e:
        print(f"  Error rendering filtered cycles grid: {e}")

    return rendered_paths
