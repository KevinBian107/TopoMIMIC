#!/usr/bin/env python3
"""
Segment Category Demo
=====================
Tests the behavioral asymmetry hypothesis: when a rodent performs an asymmetric
behavior (e.g., walk->rear->walk back differently), the DM cycle splits into
multiple temporally disjoint segments. Symmetric behaviors produce a single
long segment.

Selects high-movement clips, categorizes them by segment structure, renders
them, and computes qpos similarity metrics comparing bridge endpoints to
cycle starts.

Recomputes segmentation from raw per-clip DM output files (dimo_edge.txt)
rather than loading pre-segmented pkl, so bridge-length segments are preserved.

Usage:
    python demos/demo_segment_categories.py
    python demos/demo_segment_categories.py --config configs/demos/demo_segment_categories.yaml
    python demos/demo_segment_categories.py --skip-rendering
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import argparse
import yaml
import numpy as np
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, "/home/jovyan/vast/kaiwen/TopoVNL")

from topo_mimic.dmwk.utils.load_utils import get_aggregate_data, get_clip_indices
from topo_mimic.dmwk.skeletonization.skeletonization import extract_cycles_from_dm_graph
from topo_mimic.dmwk.skeletonization.remap import (
    parse_swr_remap,
    remap_cycle_to_original,
    split_cycle_into_segments,
)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    rodent_rollout_dir: str = "data/rodent_ar.h5"
    split: str = "train"


@dataclass
class DMConfig:
    results_dir: str = "outputs/dm_pipeline_ar_0.75"
    pers_delta: float = 0.75
    takens_m: int = 3
    takens_tau: int = 6
    segment_gap_limit: int = 50


@dataclass
class SelectionConfig:
    top_k: int = 20
    displacement_dims: int = 2
    percentile_low: float = 50.0


@dataclass
class CategorizationConfig:
    min_segment_len: int = 50
    max_bridge_len: int = 5


@dataclass
class RenderingConfig:
    enabled: bool = True
    fps: int = 30
    target_size: list[int] = field(default_factory=lambda: [320, 240])


@dataclass
class AnalysisConfig:
    skip_root_dims: int = 7
    dpi: int = 150


@dataclass
class Config:
    output_dir: str = "outputs/segment_categories"
    data: DataConfig = field(default_factory=DataConfig)
    dm: DMConfig = field(default_factory=DMConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    categorization: CategorizationConfig = field(default_factory=CategorizationConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    data_cfg = DataConfig(**cfg_dict.get("data", {}))
    dm_cfg = DMConfig(**cfg_dict.get("dm", {}))
    selection_cfg = SelectionConfig(**cfg_dict.get("selection", {}))
    cat_cfg = CategorizationConfig(**cfg_dict.get("categorization", {}))
    render_cfg = RenderingConfig(**cfg_dict.get("rendering", {}))
    analysis_cfg = AnalysisConfig(**cfg_dict.get("analysis", {}))

    return Config(
        output_dir=cfg_dict.get("output_dir", "outputs/segment_categories"),
        data=data_cfg,
        dm=dm_cfg,
        selection=selection_cfg,
        categorization=cat_cfg,
        rendering=render_cfg,
        analysis=analysis_cfg,
    )


# ---------------------------------------------------------------------------
# Stage 1: Clip Selection
# ---------------------------------------------------------------------------


def load_qpos_for_clip(clip_idx: int, rodent_rollout_dir: str) -> np.ndarray:
    return get_aggregate_data("/qpos_rollout", [], clip_idx, rodent_rollout_dir)


def load_all_qpos_bulk(rodent_rollout_dir: str, clip_indices: list[int]) -> np.ndarray:
    """Bulk-load qpos for all clips via single HDF5 reads.

    Returns:
        Array of shape (n_clips, n_frames, 74).
    """
    import h5py
    with h5py.File(rodent_rollout_dir, "r") as f:
        all_qpos = f["qpos_rollout"][clip_indices]
    return np.array(all_qpos)


def compute_displacements_bulk(
    all_qpos: np.ndarray,
    dims: int = 2,
) -> np.ndarray:
    """Compute root displacement for all clips at once.

    Args:
        all_qpos: (n_clips, n_frames, 74)
        dims: Number of root dims (2=XY, 3=XYZ)

    Returns:
        (n_clips,) displacement array.
    """
    root = all_qpos[:, :, :dims]
    diffs = np.diff(root, axis=1)
    return np.sum(np.linalg.norm(diffs, axis=2), axis=1)


def select_top_k_clips(
    clip_indices: list[int],
    all_qpos: np.ndarray,
    displacements: np.ndarray,
    top_k: int,
    percentile_low: float = 50.0,
) -> list[tuple[int, int, float]]:
    """Select top-k clips from those above a displacement percentile.

    Args:
        clip_indices: Global clip indices for the split
        all_qpos: Bulk-loaded qpos array
        displacements: Pre-computed displacement per clip
        top_k: Number of clips to return
        percentile_low: Only consider clips above this displacement percentile

    Returns:
        List of (global_clip_idx, local_idx, displacement) sorted desc.
    """
    threshold = np.percentile(displacements, percentile_low)
    candidates = [
        (clip_indices[i], i, float(displacements[i]))
        for i in range(len(clip_indices))
        if displacements[i] >= threshold
    ]
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Stage 2: Cycle extraction from raw DM files + Segment Categorization
# ---------------------------------------------------------------------------


def load_cycles_from_dm_output(
    clip_dir: str,
    pers_delta: float,
) -> list[np.ndarray]:
    """Load DM graph from per-clip output files and extract cycles.

    Args:
        clip_dir: Path to clip's DM output directory (contains features.txt, etc.)
        pers_delta: Persistence delta (determines subdirectory with dimo_edge.txt)

    Returns:
        List of remapped cycle arrays (original Takens indices), sorted by length desc.
    """
    features_path = os.path.join(clip_dir, "features.txt")
    edge_file = os.path.join(clip_dir, str(pers_delta), "dimo_edge.txt")

    if not os.path.exists(features_path) or not os.path.exists(edge_file):
        return []

    features = np.loadtxt(features_path)
    n_points = len(features)

    G = nx.Graph()
    G.add_nodes_from(range(n_points))
    edges = np.loadtxt(edge_file, dtype=int)
    if edges.size > 0:
        if edges.ndim == 1:
            edges = edges.reshape(1, -1)
        for edge in edges:
            if len(edge) >= 2:
                G.add_edge(int(edge[0]), int(edge[1]))

    cycles = extract_cycles_from_dm_graph(G, features)
    if not cycles:
        return []

    remap = parse_swr_remap(clip_dir)
    remapped = []
    for cycle in cycles:
        orig = remap_cycle_to_original(cycle, remap, n_points)
        if len(orig) > 0:
            remapped.append(orig)

    remapped.sort(key=len, reverse=True)
    return remapped


def segment_longest_cycle(
    remapped_cycles: list[np.ndarray],
    gap_limit: int,
) -> list[np.ndarray]:
    """Split the longest cycle into contiguous segments (no min_segment_len filter).

    Returns:
        List of segment arrays (sorted Takens indices), including bridge-length ones.
    """
    if not remapped_cycles:
        return []

    longest = remapped_cycles[0]
    return split_cycle_into_segments(longest, gap_limit=gap_limit)


@dataclass
class CategorizedClip:
    clip_idx: int
    local_idx: int
    displacement: float
    category: str
    segments: list[np.ndarray]
    major_segment_idx: int | None
    bridge_indices: list[int] | None


def takens_to_frame_indices(takens_indices: np.ndarray, m: int, tau: int) -> np.ndarray:
    offset = (m - 1) * tau
    return takens_indices + offset


def categorize_clip_segments(
    segments_takens: list[np.ndarray],
    m: int,
    tau: int,
    min_segment_len: int,
    max_bridge_len: int,
) -> tuple[str | None, list[np.ndarray], int | None, list[int] | None]:
    """Categorize a clip's segments.

    Returns:
        (category, frame_segments, major_idx, bridge_idxs) or (None, ...) if uncategorizable.
    """
    if not segments_takens:
        return None, [], None, None

    frame_segments = [takens_to_frame_indices(seg, m, tau) for seg in segments_takens]
    seg_lens = [len(seg) for seg in frame_segments]

    long_segments = [i for i, l in enumerate(seg_lens) if l >= min_segment_len]
    bridge_segments = [i for i, l in enumerate(seg_lens) if l <= max_bridge_len]

    # Category 1: multi-segment — >1 segment, ALL segments >= min_segment_len
    if len(long_segments) > 1 and len(long_segments) == len(frame_segments):
        return "multi-segment", frame_segments, None, None

    # Category 2: single-major — exactly 1 long segment, >=1 bridge, nothing in between
    mid_segments = [
        i for i, l in enumerate(seg_lens)
        if l > max_bridge_len and l < min_segment_len
    ]
    if len(long_segments) == 1 and len(bridge_segments) >= 1 and len(mid_segments) == 0:
        return "single-major", frame_segments, long_segments[0], bridge_segments

    return None, frame_segments, None, None


def extract_and_categorize_clips(
    top_clips: list[tuple[int, int, float]],
    dm_cfg: DMConfig,
    cat_cfg: CategorizationConfig,
) -> tuple[list[CategorizedClip], list[CategorizedClip]]:
    """Load raw DM output per clip, segment, and categorize."""
    multi_clips = []
    single_clips = []

    for global_idx, local_idx, displacement in top_clips:
        clip_dir = os.path.join(dm_cfg.results_dir, "clips", f"clip_{local_idx}")
        if not os.path.isdir(clip_dir):
            print(f"    Clip {global_idx}: DM output dir not found, skipping")
            continue

        remapped_cycles = load_cycles_from_dm_output(clip_dir, dm_cfg.pers_delta)
        if not remapped_cycles:
            continue

        segments_takens = segment_longest_cycle(remapped_cycles, dm_cfg.segment_gap_limit)
        if not segments_takens:
            continue

        category, frame_segs, major_idx, bridge_idxs = categorize_clip_segments(
            segments_takens, dm_cfg.takens_m, dm_cfg.takens_tau,
            cat_cfg.min_segment_len, cat_cfg.max_bridge_len,
        )

        seg_lens = [len(s) for s in frame_segs]
        if category is None:
            print(f"    Clip {global_idx}: uncategorized, seg_lens={seg_lens}")
            continue

        clip = CategorizedClip(
            clip_idx=global_idx,
            local_idx=local_idx,
            displacement=displacement,
            category=category,
            segments=frame_segs,
            major_segment_idx=major_idx,
            bridge_indices=bridge_idxs,
        )

        if category == "multi-segment":
            multi_clips.append(clip)
        else:
            single_clips.append(clip)

    return multi_clips, single_clips


# ---------------------------------------------------------------------------
# Stage 3: Rendering
# ---------------------------------------------------------------------------


def render_category_video(
    multi_clips: list[CategorizedClip],
    single_clips: list[CategorizedClip],
    rodent_rollout_dir: str,
    output_path: str,
    fps: int = 30,
    target_size: tuple[int, int] = (320, 240),
) -> None:
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg
    from topo_mimic.dmwk.skeletonization.rendering import (
        _load_full_clip_frames,
        _ensure_rgb_uint8,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_title = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        font_title = ImageFont.load_default()

    Wc, Hc = target_size
    gutter = 8
    label_h = 24
    row_label_h = 28
    pad = 16

    # Row 1 (multi-segment): each clip's segments as panels
    row1_panels = []
    for clip in multi_clips[:5]:
        print(f"    Rendering multi-segment clip {clip.clip_idx}...")
        frames = _load_full_clip_frames(rodent_rollout_dir, clip.clip_idx)
        for si, seg in enumerate(clip.segments):
            seg_frames = [frames[min(int(fi), len(frames) - 1)] for fi in seg]
            row1_panels.append((f"C{clip.clip_idx} S{si + 1}", seg_frames))

    # Row 2 (single-major): major segment only
    row2_panels = []
    for clip in single_clips[:5]:
        print(f"    Rendering single-major clip {clip.clip_idx}...")
        frames = _load_full_clip_frames(rodent_rollout_dir, clip.clip_idx)
        major = clip.segments[clip.major_segment_idx]
        seg_frames = [frames[min(int(fi), len(frames) - 1)] for fi in major]
        row2_panels.append((f"C{clip.clip_idx} major", seg_frames))

    if not row1_panels and not row2_panels:
        print("  No panels to render, skipping video")
        return

    n_cols = max(len(row1_panels), len(row2_panels), 1)
    n_rows = (1 if row1_panels else 0) + (1 if row2_panels else 0)

    cell_w = Wc
    cell_h = label_h + Hc
    W = pad * 2 + n_cols * cell_w + (n_cols - 1) * gutter
    H = pad * 2 + n_rows * (row_label_h + cell_h) + (n_rows - 1) * gutter

    max_t = 0
    for _, frames in row1_panels + row2_panels:
        max_t = max(max_t, len(frames))
    if max_t == 0:
        print("  No frames to render, skipping video")
        return

    writer = imageio_ffmpeg.write_frames(
        output_path, (W, H), fps=fps, codec="h264", macro_block_size=1,
    )
    writer.send(None)

    rows_data = []
    if row1_panels:
        rows_data.append(("Multi-Segment", row1_panels))
    if row2_panels:
        rows_data.append(("Single-Major", row2_panels))

    for t in range(max_t):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        y = pad
        for row_label, panels in rows_data:
            tw, th = draw.textbbox((0, 0), row_label, font=font_title)[2:]
            draw.text(
                (pad, y + (row_label_h - th) // 2),
                row_label, font=font_title, fill=(0, 0, 0),
            )
            y += row_label_h

            for col, (panel_label, panel_frames) in enumerate(panels):
                x = pad + col * (cell_w + gutter)

                tw2, th2 = draw.textbbox((0, 0), panel_label, font=font)[2:]
                draw.text(
                    (x + (cell_w - tw2) // 2, y + (label_h - th2) // 2),
                    panel_label, font=font, fill=(0, 0, 0),
                )

                fi = min(t, len(panel_frames) - 1)
                fr = _ensure_rgb_uint8(panel_frames[fi])
                img = Image.fromarray(fr).resize((Wc, Hc))
                canvas.paste(img, (x, y + label_h))

            y += cell_h + gutter

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
    print(f"  Saved video: {output_path}")


# ---------------------------------------------------------------------------
# Stage 4: QPos Similarity Analysis
# ---------------------------------------------------------------------------


@dataclass
class SimilarityResult:
    euclidean_mean: float
    euclidean_std: float
    cosine_mean: float
    cosine_std: float
    n_pairs: int


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def analyze_single_major_clips(
    single_clips: list[CategorizedClip],
    rodent_rollout_dir: str,
    skip_root_dims: int,
) -> SimilarityResult | None:
    """Category 2: bridge endpoints vs cycle start."""
    euclidean_dists = []
    cosine_sims = []

    for clip in single_clips:
        qpos = load_qpos_for_clip(clip.clip_idx, rodent_rollout_dir)
        major_seg = clip.segments[clip.major_segment_idx]
        first_frame_major = int(major_seg[0])
        qpos_major_start = qpos[min(first_frame_major, len(qpos) - 1), skip_root_dims:]

        for bi in clip.bridge_indices:
            bridge_seg = clip.segments[bi]
            last_frame_bridge = int(bridge_seg[-1])
            qpos_bridge_end = qpos[min(last_frame_bridge, len(qpos) - 1), skip_root_dims:]

            euclidean_dists.append(float(np.linalg.norm(qpos_major_start - qpos_bridge_end)))
            cosine_sims.append(_cosine_similarity(qpos_major_start, qpos_bridge_end))

    if not euclidean_dists:
        return None

    return SimilarityResult(
        euclidean_mean=float(np.mean(euclidean_dists)),
        euclidean_std=float(np.std(euclidean_dists)),
        cosine_mean=float(np.mean(cosine_sims)),
        cosine_std=float(np.std(cosine_sims)),
        n_pairs=len(euclidean_dists),
    )


def analyze_multi_segment_clips(
    multi_clips: list[CategorizedClip],
    rodent_rollout_dir: str,
    skip_root_dims: int,
) -> SimilarityResult | None:
    """Category 1: cross-segment start/end comparisons."""
    euclidean_dists = []
    cosine_sims = []

    for clip in multi_clips:
        qpos = load_qpos_for_clip(clip.clip_idx, rodent_rollout_dir)
        n_segs = len(clip.segments)

        for i in range(n_segs):
            for j in range(n_segs):
                if i == j:
                    continue
                start_i = int(clip.segments[i][0])
                end_j = int(clip.segments[j][-1])

                q_start = qpos[min(start_i, len(qpos) - 1), skip_root_dims:]
                q_end = qpos[min(end_j, len(qpos) - 1), skip_root_dims:]

                euclidean_dists.append(float(np.linalg.norm(q_start - q_end)))
                cosine_sims.append(_cosine_similarity(q_start, q_end))

    if not euclidean_dists:
        return None

    return SimilarityResult(
        euclidean_mean=float(np.mean(euclidean_dists)),
        euclidean_std=float(np.std(euclidean_dists)),
        cosine_mean=float(np.mean(cosine_sims)),
        cosine_std=float(np.std(cosine_sims)),
        n_pairs=len(euclidean_dists),
    )


# ---------------------------------------------------------------------------
# Stage 5: Bar Charts
# ---------------------------------------------------------------------------


def plot_similarity_bars(
    single_result: SimilarityResult | None,
    multi_result: SimilarityResult | None,
    output_dir: str,
    dpi: int = 150,
) -> None:
    labels = []
    euc_means = []
    euc_stds = []
    cos_means = []
    cos_stds = []

    if single_result:
        labels.append(f"Asymmetric\n(bridge<>cycle start)\nn={single_result.n_pairs}")
        euc_means.append(single_result.euclidean_mean)
        euc_stds.append(single_result.euclidean_std)
        cos_means.append(single_result.cosine_mean)
        cos_stds.append(single_result.cosine_std)

    if multi_result:
        labels.append(f"Multi-segment\n(seg start<>seg end)\nn={multi_result.n_pairs}")
        euc_means.append(multi_result.euclidean_mean)
        euc_stds.append(multi_result.euclidean_std)
        cos_means.append(multi_result.cosine_mean)
        cos_stds.append(multi_result.cosine_std)

    if not labels:
        print("  No results to plot")
        return

    x = np.arange(len(labels))
    colors = ["#e6194b", "#3cb44b"][:len(labels)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, euc_means, yerr=euc_stds, capsize=5, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Euclidean Distance (67D qpos)")
    ax.set_title("Bridge vs Cycle Start: Euclidean Distance")
    plt.tight_layout()
    euc_path = os.path.join(output_dir, "euclidean_comparison.png")
    plt.savefig(euc_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {euc_path}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, cos_means, yerr=cos_stds, capsize=5, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Cosine Similarity (67D qpos)")
    ax.set_title("Bridge vs Cycle Start: Cosine Similarity")
    plt.tight_layout()
    cos_path = os.path.join(output_dir, "cosine_comparison.png")
    plt.savefig(cos_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {cos_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Segment Category Demo")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--skip-rendering", action="store_true", help="Skip video rendering")
    args = parser.parse_args()

    default_config_path = (
        Path(__file__).parent.parent / "configs" / "demos" / "demo_segment_categories.yaml"
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
    if not os.path.isabs(cfg.dm.results_dir):
        cfg.dm.results_dir = str(project_root / cfg.dm.results_dir)

    print("=" * 70)
    print("Segment Category Demo")
    print("=" * 70)
    print(f"  Output dir: {output_dir}")
    print(f"  Data: {cfg.data.rodent_rollout_dir}")
    print(f"  DM results: {cfg.dm.results_dir}")
    print(f"  pers_delta: {cfg.dm.pers_delta}")
    print(f"  Top-k: {cfg.selection.top_k}")
    print(f"  Categorization: min_seg={cfg.categorization.min_segment_len}, max_bridge={cfg.categorization.max_bridge_len}")
    print(f"  Skip rendering: {args.skip_rendering}")

    # Stage 1: Clip Selection
    print("\n=== Stage 1: Clip Selection ===")
    clip_indices = get_clip_indices(cfg.data.split)
    print(f"  {cfg.data.split} split: {len(clip_indices)} clips")

    print(f"  Bulk-loading qpos for {len(clip_indices)} clips...")
    all_qpos = load_all_qpos_bulk(cfg.data.rodent_rollout_dir, clip_indices)
    displacements = compute_displacements_bulk(all_qpos, dims=cfg.selection.displacement_dims)
    print(f"  Displacement range: [{displacements.min():.3f}, {displacements.max():.3f}]")
    print(f"  Filtering above {cfg.selection.percentile_low}th percentile, taking top {cfg.selection.top_k}...")

    top_clips = select_top_k_clips(
        clip_indices, all_qpos, displacements,
        cfg.selection.top_k, cfg.selection.percentile_low,
    )
    print(f"  Top {len(top_clips)} by displacement:")
    for global_idx, local_idx, disp in top_clips[:5]:
        print(f"    Clip {global_idx} (local {local_idx}): disp={disp:.2f}")

    # Stage 2: Extract cycles from raw DM files + categorize
    print("\n=== Stage 2: Cycle Extraction & Categorization ===")
    print(f"  Loading per-clip DM outputs from {cfg.dm.results_dir}/clips/")

    multi_clips, single_clips = extract_and_categorize_clips(
        top_clips, cfg.dm, cfg.categorization,
    )

    print(f"\n  Multi-segment clips: {len(multi_clips)}")
    for clip in multi_clips:
        seg_lens = [len(s) for s in clip.segments]
        print(f"    Clip {clip.clip_idx}: {len(clip.segments)} segs, lens={seg_lens}")
    print(f"  Single-major clips: {len(single_clips)}")
    for clip in single_clips:
        seg_lens = [len(s) for s in clip.segments]
        print(f"    Clip {clip.clip_idx}: major_idx={clip.major_segment_idx}, lens={seg_lens}")

    # Stage 3: Rendering
    if not args.skip_rendering and cfg.rendering.enabled:
        print("\n=== Stage 3: Rendering ===")
        video_path = str(output_dir / "categories_side_by_side.mp4")
        render_category_video(
            multi_clips, single_clips,
            cfg.data.rodent_rollout_dir, video_path,
            fps=cfg.rendering.fps,
            target_size=tuple(cfg.rendering.target_size),
        )
    else:
        print("\n=== Stage 3: Rendering (skipped) ===")

    # Stage 4: QPos Similarity Analysis
    print("\n=== Stage 4: QPos Similarity Analysis ===")
    single_result = analyze_single_major_clips(
        single_clips, cfg.data.rodent_rollout_dir, cfg.analysis.skip_root_dims,
    )
    multi_result = analyze_multi_segment_clips(
        multi_clips, cfg.data.rodent_rollout_dir, cfg.analysis.skip_root_dims,
    )

    if single_result:
        print(f"  Single-major (asymmetric): n={single_result.n_pairs}")
        print(f"    Euclidean: {single_result.euclidean_mean:.4f} +/- {single_result.euclidean_std:.4f}")
        print(f"    Cosine:    {single_result.cosine_mean:.4f} +/- {single_result.cosine_std:.4f}")
    else:
        print("  Single-major: no valid pairs")

    if multi_result:
        print(f"  Multi-segment: n={multi_result.n_pairs}")
        print(f"    Euclidean: {multi_result.euclidean_mean:.4f} +/- {multi_result.euclidean_std:.4f}")
        print(f"    Cosine:    {multi_result.cosine_mean:.4f} +/- {multi_result.cosine_std:.4f}")
    else:
        print("  Multi-segment: no valid pairs")

    # Stage 5: Bar Charts
    print("\n=== Stage 5: Bar Charts ===")
    plot_similarity_bars(single_result, multi_result, str(output_dir), cfg.analysis.dpi)

    # Save summary
    summary = {
        "config": {
            "top_k": cfg.selection.top_k,
            "displacement_dims": cfg.selection.displacement_dims,
            "min_segment_len": cfg.categorization.min_segment_len,
            "max_bridge_len": cfg.categorization.max_bridge_len,
            "skip_root_dims": cfg.analysis.skip_root_dims,
            "takens_m": cfg.dm.takens_m,
            "takens_tau": cfg.dm.takens_tau,
            "pers_delta": cfg.dm.pers_delta,
            "segment_gap_limit": cfg.dm.segment_gap_limit,
        },
        "multi_segment_clips": [
            {
                "clip_idx": c.clip_idx,
                "displacement": float(c.displacement),
                "n_segments": len(c.segments),
                "segment_lens": [len(s) for s in c.segments],
            }
            for c in multi_clips
        ],
        "single_major_clips": [
            {
                "clip_idx": c.clip_idx,
                "displacement": float(c.displacement),
                "n_segments": len(c.segments),
                "major_segment_idx": c.major_segment_idx,
                "segment_lens": [len(s) for s in c.segments],
            }
            for c in single_clips
        ],
        "analysis": {},
    }

    if single_result:
        summary["analysis"]["single_major"] = {
            "euclidean_mean": single_result.euclidean_mean,
            "euclidean_std": single_result.euclidean_std,
            "cosine_mean": single_result.cosine_mean,
            "cosine_std": single_result.cosine_std,
            "n_pairs": single_result.n_pairs,
        }
    if multi_result:
        summary["analysis"]["multi_segment"] = {
            "euclidean_mean": multi_result.euclidean_mean,
            "euclidean_std": multi_result.euclidean_std,
            "cosine_mean": multi_result.cosine_mean,
            "cosine_std": multi_result.cosine_std,
            "n_pairs": multi_result.n_pairs,
        }

    summary_path = str(output_dir / "summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    print(f"\n  Saved summary: {summary_path}")

    print("\n" + "=" * 70)
    print("Segment Category Demo Complete!")
    print(f"Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
