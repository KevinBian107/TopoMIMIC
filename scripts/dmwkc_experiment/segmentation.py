"""
Segmentation Pipeline for New Clips.

Given a new clip (by index or frame range):
1. Run DM graph to extract essential keyframes (longest loop)
2. Compare keyframe segment to existing clusters via Wasserstein distance
3. Return top-k cluster matches
4. Render: side-by-side comparison of new clip vs top cluster samples
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import numpy as np
import pickle as pkl
import yaml
import h5py
import ot
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

from vnl_tda.utils.load_utils import load_from_h5py, get_aggregate_data, is_consolidated_format, _get_consolidated_file
from vnl_tda.utils.rendering import render_from_saved_rollout
from vnl_tda.pipelines.tda.skeletonization.skeletonization import extract_skeleton_pcd_dm
from vnl_tda.pipelines.tda.skeletonization.postprocess import (
    _uniform_downsample,
    _expand_frames_nonconsec,
)


@dataclass
class DMConfig:
    pers_delta: float = 0.75
    k_neighbors: int = 5
    epsilon: float = 0.99


@dataclass
class TakensConfig:
    m: int = 3
    tau: int = 6


@dataclass
class ReferenceConfig:
    results_dir: str = "out_subspace_dunn"
    cluster_dict_path: str = "out_subspace_dunn/clustering/cluster_dict.pkl"
    w2_cache_path: str = "out_subspace_dunn/w2_distance_matrix.pkl"
    cluster_labels: dict[int, str] = field(default_factory=dict)


@dataclass
class SegmentationConfig:
    target_len: int = 50
    top_k: int = 3
    min_loop_len: int = 10
    max_loop_len: int = 500


@dataclass
class RenderingConfig:
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)
    camera_name: str = "close_profile"


@dataclass
class PipelineConfig:
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    output_dir: str = "out_segmentation"
    clip_ids: list[int] = field(default_factory=list)
    reference: ReferenceConfig = None
    dm: DMConfig = None
    takens: TakensConfig = None
    segmentation: SegmentationConfig = None
    rendering: RenderingConfig = None

    def __post_init__(self):
        if self.reference is None:
            self.reference = ReferenceConfig()
        if self.dm is None:
            self.dm = DMConfig()
        if self.takens is None:
            self.takens = TakensConfig()
        if self.segmentation is None:
            self.segmentation = SegmentationConfig()
        if self.rendering is None:
            self.rendering = RenderingConfig()


def load_config(config_path: str) -> PipelineConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    reference_dict = cfg.get("reference", {})
    if "cluster_labels" in reference_dict:
        reference_dict["cluster_labels"] = {
            int(k): v for k, v in reference_dict["cluster_labels"].items()
        }
    reference_cfg = ReferenceConfig(**reference_dict)

    dm_cfg = DMConfig(**cfg.get("dm", {}))
    takens_cfg = TakensConfig(**cfg.get("takens", {}))
    seg_cfg = SegmentationConfig(**cfg.get("segmentation", {}))

    rendering_dict = cfg.get("rendering", {})
    if "target_size" in rendering_dict:
        rendering_dict["target_size"] = tuple(rendering_dict["target_size"])
    rendering_cfg = RenderingConfig(**rendering_dict)

    clip_ids = cfg.get("clip_ids", [])

    return PipelineConfig(
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", PipelineConfig.rodent_rollout_dir),
        output_dir=cfg.get("output_dir", PipelineConfig.output_dir),
        clip_ids=clip_ids,
        reference=reference_cfg,
        dm=dm_cfg,
        takens=takens_cfg,
        segmentation=seg_cfg,
        rendering=rendering_cfg,
    )


def takens_points(X: np.ndarray, m: int, tau: int) -> np.ndarray:
    n_frames, D = X.shape
    start = (m - 1) * tau
    idxs = np.arange(start, n_frames)
    pts = np.stack([X[[j - a * tau for a in range(m)]].reshape(-1) for j in idxs], axis=0)
    return pts


def load_clip_data(
    clip_id: int,
    rodent_rollout_dir: str,
) -> dict:
    if is_consolidated_format(rodent_rollout_dir):
        f = _get_consolidated_file(rodent_rollout_dir)
        return {
            'intention': f['intention'][clip_id],
            'qpos': f['qpos_rollout'][clip_id],
        }
    with h5py.File(f"{rodent_rollout_dir}/clip_{clip_id}.h5", "r") as f:
        rollout = load_from_h5py(f)
    return rollout


def load_intention(clip_id: int, rodent_rollout_dir: str) -> np.ndarray:
    rollout = load_clip_data(clip_id, rodent_rollout_dir)
    return np.asarray(rollout["intention"])


def extract_keyframes_dm(
    intention: np.ndarray,
    dm_cfg: DMConfig,
    takens_cfg: TakensConfig,
    output_dir: str | None = None,
    use_longest_loop: bool = True,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Extract keyframes from intention trajectory using DM graph.

    Args:
        intention: (T, D) intention trajectory
        dm_cfg: DM configuration
        takens_cfg: Takens embedding configuration
        output_dir: Optional output directory for DM files
        use_longest_loop: If True, only return keyframes from the longest loop

    Returns:
        Tuple of (cycles, keyframe_indices)
    """
    pts = takens_points(intention, m=takens_cfg.m, tau=takens_cfg.tau)

    cycles, G = extract_skeleton_pcd_dm(
        pts,
        k=dm_cfg.k_neighbors,
        pers_delta=dm_cfg.pers_delta,
        epsilon=dm_cfg.epsilon,
        output_dir=output_dir,
        cleanup=output_dir is None,
    )

    offset = (takens_cfg.m - 1) * takens_cfg.tau

    if use_longest_loop and cycles:
        longest_cycle = max(cycles, key=len)
        keyframes = set()
        for idx in longest_cycle:
            keyframes.add(int(idx) + offset)
        keyframe_indices = np.array(sorted(keyframes), dtype=int)
    else:
        all_keyframes = set()
        for cycle in cycles:
            for idx in cycle:
                all_keyframes.add(int(idx) + offset)
        keyframe_indices = np.array(sorted(all_keyframes), dtype=int)

    keyframe_indices = keyframe_indices[keyframe_indices < len(intention)]

    return cycles, keyframe_indices


def compute_w2_to_cluster(
    query_intention: np.ndarray,
    query_frames: np.ndarray,
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    int_rodent: np.ndarray,
    target_len: int,
) -> dict[int, float]:
    """
    Compute Wasserstein-2 distance from query segment to each cluster prototype.

    Args:
        query_intention: (T, D) intention for the query clip
        query_frames: Frame indices for the query segment
        cluster_dict: Reference cluster dictionary
        int_rodent: (n_clips, T, D) intention array for reference clips
        target_len: Target length for uniform sampling

    Returns:
        Dict mapping cluster_id to mean W2 distance
    """
    query_sampled = _uniform_downsample(query_frames, target_len)
    if query_sampled is None:
        if len(query_frames) >= target_len:
            query_sampled = query_frames[:target_len]
        else:
            query_sampled = query_frames

    X_query = query_intention[query_sampled]
    w_query = np.full(len(query_sampled), 1.0 / len(query_sampled))

    cluster_distances = {}

    for cid, entries in cluster_dict.items():
        distances = []

        for clip_id, frames in entries[:10]:
            sampled = _uniform_downsample(frames, target_len)
            if sampled is None:
                continue

            X_ref = int_rodent[clip_id, sampled]
            w_ref = np.full(target_len, 1.0 / target_len)

            C = ot.dist(X_query, X_ref)
            d = ot.emd2(w_query, w_ref, C)
            distances.append(np.sqrt(d))

        if distances:
            cluster_distances[cid] = np.mean(distances)

    return cluster_distances


def get_top_k_clusters(
    distances: dict[int, float],
    k: int = 3,
) -> list[tuple[int, float]]:
    """Return top-k closest clusters sorted by distance."""
    sorted_clusters = sorted(distances.items(), key=lambda x: x[1])
    return sorted_clusters[:k]


@dataclass
class SegmentationResult:
    clip_id: int
    keyframe_indices: np.ndarray
    cycles: list[np.ndarray]
    cluster_distances: dict[int, float]
    top_k_clusters: list[tuple[int, float]]


def segment_clip(
    clip_id: int,
    cfg: PipelineConfig,
    cluster_dict: dict,
    int_rodent: np.ndarray,
    frame_indices: np.ndarray | None = None,
) -> SegmentationResult:
    """
    Segment a single clip and find matching clusters.

    Args:
        clip_id: Clip index
        cfg: Pipeline configuration
        cluster_dict: Reference cluster dictionary
        int_rodent: Reference intention array
        frame_indices: Optional specific frame indices to analyze

    Returns:
        SegmentationResult with keyframes and cluster matches
    """
    intention = load_intention(clip_id, cfg.rodent_rollout_dir)

    if frame_indices is not None:
        intention_subset = intention[frame_indices]
    else:
        intention_subset = intention
        frame_indices = np.arange(len(intention))

    dm_output_dir = os.path.join(cfg.output_dir, f"dm_clip_{clip_id}")
    dm_output_dir = os.path.abspath(dm_output_dir)
    os.makedirs(dm_output_dir, exist_ok=True)

    cycles, keyframe_indices = extract_keyframes_dm(
        intention_subset,
        cfg.dm,
        cfg.takens,
        output_dir=dm_output_dir,
    )

    if len(keyframe_indices) == 0:
        keyframe_indices = np.linspace(0, len(intention_subset) - 1, cfg.segmentation.target_len).astype(int)

    cluster_distances = compute_w2_to_cluster(
        intention_subset,
        keyframe_indices,
        cluster_dict,
        int_rodent,
        cfg.segmentation.target_len,
    )

    top_k = get_top_k_clusters(cluster_distances, cfg.segmentation.top_k)

    return SegmentationResult(
        clip_id=clip_id,
        keyframe_indices=keyframe_indices,
        cycles=cycles,
        cluster_distances=cluster_distances,
        top_k_clusters=top_k,
    )


def get_cluster_sample_frames(
    cluster_id: int,
    cluster_dict: dict,
    rodent_rollout_dir: str,
) -> list[np.ndarray] | None:
    """Get rendered frames for one sample from a cluster."""
    if cluster_id not in cluster_dict:
        return None

    entries = cluster_dict[cluster_id]
    if not entries:
        return None

    clip_id, frames = entries[0]
    rollout = load_clip_data(clip_id, rodent_rollout_dir)
    all_frames = render_from_saved_rollout(rollout)

    frames = np.asarray(frames, dtype=int)
    frames = frames[frames < len(all_frames)]

    if len(frames) == 0:
        return None

    return [all_frames[i] for i in frames]


def render_comparison_grid(
    results: list["SegmentationResult"],
    cluster_dict: dict,
    rodent_rollout_dir: str,
    output_path: str,
    cluster_labels: dict[int, str] | None = None,
    framerate: int = 30,
    target_size: tuple[int, int] = (320, 240),
) -> str:
    """
    Render grid comparison: each row is one clip.

    Layout per row:
    [New Clip] [New Clip Traversal] | [Top 1 Segmentation] [Top 2] [Top 3]

    Args:
        results: List of SegmentationResult for all clips
        cluster_dict: Reference cluster dictionary
        rodent_rollout_dir: Path to rollout data
        output_path: Output video path
        cluster_labels: Optional cluster label names
        framerate: Video framerate
        target_size: (width, height) per tile
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_rows = len(results)
    n_cluster_cols = 3

    W, H = target_size
    header_h = 30
    title_h = 35
    separator_w = 6
    gutter = 6
    row_gap = 4

    canvas_w = (W * 2) + gutter + separator_w + gutter + (W * n_cluster_cols) + (gutter * (n_cluster_cols - 1))
    canvas_h = header_h + n_rows * (H + title_h + row_gap)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
        font_header = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_header = font

    all_clip_data = []
    for result in results:
        rollout = load_clip_data(result.clip_id, rodent_rollout_dir)
        full_frames = render_from_saved_rollout(rollout)
        keyframes = [full_frames[i] for i in result.keyframe_indices if i < len(full_frames)]
        if not keyframes:
            keyframes = full_frames

        cluster_frames_list = []
        cluster_titles = []
        for cid, dist in result.top_k_clusters:
            label = cluster_labels.get(cid, f"cluster_{cid}") if cluster_labels else f"cluster_{cid}"
            frames = get_cluster_sample_frames(cid, cluster_dict, rodent_rollout_dir)
            if frames is not None:
                cluster_frames_list.append(frames)
                cluster_titles.append(f"{label}\n(W2={dist:.3f})")
            else:
                cluster_frames_list.append([np.zeros((480, 640, 3), dtype=np.uint8)])
                cluster_titles.append(f"{label}\n(no sample)")

        all_clip_data.append({
            "clip_id": result.clip_id,
            "full_frames": full_frames,
            "keyframes": keyframes,
            "keyframe_count": len(result.keyframe_indices),
            "cluster_frames": cluster_frames_list,
            "cluster_titles": cluster_titles,
        })

    all_frame_lists = []
    for data in all_clip_data:
        all_frame_lists.append(data["full_frames"])
        all_frame_lists.append(data["keyframes"])
        all_frame_lists.extend(data["cluster_frames"])
    T = max(len(f) for f in all_frame_lists)

    writer = imageio_ffmpeg.write_frames(
        output_path, (canvas_w, canvas_h), fps=framerate, codec="h264", macro_block_size=1
    )
    writer.send(None)

    def draw_centered_text(draw, text, x_center, y, font, fill=(0, 0, 0)):
        lines = text.split('\n')
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x_center - tw // 2, y + i * (th + 2)), line, font=font, fill=fill)

    for t in range(T):
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Column headers
        x_col1 = W // 2
        x_col2 = W + gutter + W // 2
        sep_x_header = W * 2 + gutter + gutter // 2
        x_seg_start = sep_x_header + separator_w + gutter

        draw_centered_text(draw, "New Clip", x_col1, 5, font_header, fill=(0, 0, 128))
        draw_centered_text(draw, "New Clip Traversal", x_col2, 5, font_header, fill=(128, 0, 128))
        draw_centered_text(draw, "Top 3 Segmentation Matches", x_seg_start + (W * 1.5) + gutter, 5, font_header, fill=(0, 100, 0))

        # Separator line in header
        draw.rectangle([sep_x_header, 0, sep_x_header + separator_w, canvas_h], fill=(50, 50, 50))

        # Each row
        for row_idx, data in enumerate(all_clip_data):
            y_row = header_h + row_idx * (H + title_h + row_gap)

            # New Clip (column 1)
            clip_title = f"Clip {data['clip_id']}"
            draw_centered_text(draw, clip_title, W // 2, y_row + 2, font, fill=(0, 0, 128))
            fr_full = data["full_frames"][t % len(data["full_frames"])]
            img_full = Image.fromarray(np.asarray(fr_full)).resize((W, H))
            canvas.paste(img_full, (0, y_row + title_h))

            # New Clip Traversal (column 2)
            x_col2_start = W + gutter
            trav_title = f"{data['keyframe_count']} keyframes"
            draw_centered_text(draw, trav_title, x_col2_start + W // 2, y_row + 2, font, fill=(128, 0, 128))
            fr_key = data["keyframes"][t % len(data["keyframes"])]
            img_key = Image.fromarray(np.asarray(fr_key)).resize((W, H))
            canvas.paste(img_key, (x_col2_start, y_row + title_h))

            # Top 3 segmentation matches (columns 3-5)
            x_offset = sep_x_header + separator_w + gutter
            for i, (cluster_frames, title) in enumerate(zip(data["cluster_frames"], data["cluster_titles"])):
                x = x_offset + i * (W + gutter)
                draw_centered_text(draw, title, x + W // 2, y_row + 2, font, fill=(0, 100, 0))
                fr_cluster = cluster_frames[t % len(cluster_frames)]
                img_cluster = Image.fromarray(np.asarray(fr_cluster)).resize((W, H))
                canvas.paste(img_cluster, (x, y_row + title_h))

            # Row separator line
            if row_idx < n_rows - 1:
                y_line = y_row + title_h + H + row_gap // 2
                draw.line([(0, y_line), (canvas_w, y_line)], fill=(200, 200, 200), width=1)

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
    return output_path


def run_segmentation(
    clip_ids: list[int],
    cfg: PipelineConfig,
    cluster_dict: dict,
    int_rodent: np.ndarray,
    cluster_labels: dict[int, str] | None = None,
) -> list[SegmentationResult]:
    """
    Run segmentation pipeline on multiple clips.

    Args:
        clip_ids: List of clip indices to segment
        cfg: Pipeline configuration
        cluster_dict: Reference cluster dictionary
        int_rodent: Reference intention array
        cluster_labels: Optional cluster label names

    Returns:
        List of SegmentationResult for each clip
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    results = []

    for clip_id in clip_ids:
        print(f"\n=== Processing Clip {clip_id} ===")

        result = segment_clip(clip_id, cfg, cluster_dict, int_rodent)
        results.append(result)

        print(f"  Longest loop keyframes: {len(result.keyframe_indices)}")
        print(f"  Total cycles found: {len(result.cycles)}")
        print(f"  Top {cfg.segmentation.top_k} cluster matches:")
        for cid, dist in result.top_k_clusters:
            label = cluster_labels.get(cid, f"cluster_{cid}") if cluster_labels else f"cluster_{cid}"
            print(f"    Cluster {cid} ({label}): W2={dist:.4f}")

    print("\n=== Rendering comparison grid ===")
    output_path = os.path.join(cfg.output_dir, "segmentation_grid.mp4")
    render_comparison_grid(
        results=results,
        cluster_dict=cluster_dict,
        rodent_rollout_dir=cfg.rodent_rollout_dir,
        output_path=output_path,
        cluster_labels=cluster_labels,
        framerate=cfg.rendering.framerate,
        target_size=cfg.rendering.target_size,
    )
    print(f"  Saved grid video to {output_path}")

    results_path = os.path.join(cfg.output_dir, "segmentation_results.pkl")
    with open(results_path, "wb") as f:
        pkl.dump(results, f)
    print(f"\nSaved results to {results_path}")

    return results


def main(
    config_path: str = "configs/dmwkc_experiment/segmentation.yaml",
    clip_ids: list[int] | None = None,
):
    """
    Main entry point for segmentation pipeline.

    Args:
        config_path: Path to configuration YAML
        clip_ids: List of clip indices to segment (overrides config if provided)
    """
    base_dir = "/home/jovyan/vast/kaiwen/TopoVNL"

    cfg = load_config(config_path)

    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")

    if not os.path.isabs(cfg.output_dir):
        cfg.output_dir = os.path.join(base_dir, cfg.output_dir)
    if not os.path.isabs(cfg.reference.cluster_dict_path):
        cfg.reference.cluster_dict_path = os.path.join(base_dir, cfg.reference.cluster_dict_path)
    if not os.path.isabs(cfg.reference.results_dir):
        cfg.reference.results_dir = os.path.join(base_dir, cfg.reference.results_dir)

    os.makedirs(cfg.output_dir, exist_ok=True)

    if clip_ids is not None and len(clip_ids) > 0:
        cfg.clip_ids = clip_ids

    if len(cfg.clip_ids) == 0:
        raise ValueError("Must provide clip_ids in config or via --clips argument")

    print("=== Segmentation Pipeline ===")
    print(f"Output directory: {cfg.output_dir}")
    print(f"Clips to segment: {cfg.clip_ids}")

    print("\n=== Loading reference data ===")

    print(f"Loading cluster dict from {cfg.reference.cluster_dict_path}")
    with open(cfg.reference.cluster_dict_path, "rb") as f:
        cluster_dict = pkl.load(f)
    print(f"  Loaded {len(cluster_dict)} clusters")

    cluster_labels = cfg.reference.cluster_labels
    if cluster_labels:
        print(f"  Loaded {len(cluster_labels)} cluster labels from config")
    else:
        print("  No cluster labels provided, using default names")

    print("Loading intention data...")
    import functools
    import multiprocessing as mp
    from tqdm import tqdm

    num_clips = 500
    work_rodent = functools.partial(
        get_aggregate_data, "/intention", [], path=cfg.rodent_rollout_dir
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        int_rodent = list(tqdm(pool.imap(work_rodent, range(num_clips)), total=num_clips))
    int_rodent = np.vstack(int_rodent)
    int_rodent = int_rodent.reshape([num_clips, 499, 16])
    print(f"  Loaded intention: {int_rodent.shape}")

    print("\n=== Running segmentation ===")
    results = run_segmentation(
        cfg.clip_ids,
        cfg,
        cluster_dict,
        int_rodent,
        cluster_labels=cluster_labels,
    )

    print("\n=== Summary ===")
    for result in results:
        print(f"\nClip {result.clip_id}:")
        print(f"  Keyframes: {len(result.keyframe_indices)}")
        print(f"  Top matches:")
        for cid, dist in result.top_k_clusters:
            label = cluster_labels.get(cid, f"cluster_{cid}") if cluster_labels else f"cluster_{cid}"
            print(f"    {label}: W2={dist:.4f}")

    print("\n=== Pipeline complete! ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segment new clips using DM graph")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dmwkc_experiment/segmentation.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--clips",
        type=int,
        nargs="*",
        default=None,
        help="Clip indices to segment (overrides config if provided)",
    )

    args = parser.parse_args()
    main(config_path=args.config, clip_ids=args.clips)
