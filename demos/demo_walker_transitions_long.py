#!/usr/bin/env python3
"""
Walker Transition Demo (Long Sequences)
=========================================
Analysis pipeline for long walker clips with multiple behavior transitions per clip.

Two visualization modes:
  - Per-frame: each frame is an individual point colored by GT behavior label
  - Windowed: sliding windows of N stacked frames, colored by majority behavior

Also runs DM-path and DM-cycle extraction, W2 clustering, and grid rendering
(rows = clips, columns = longest K DM segments).

Usage:
    python demos/demo_walker_transitions_long.py
    python demos/demo_walker_transitions_long.py --config configs/demos/demo_walker_transitions_long.yaml
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import argparse
import pickle as pkl
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from topo_mimic.dmwk.utils.dataset_config import get_dataset_preset
from topo_mimic.dmwk.dm_pipeline import (
    PipelineConfig,
    DataConfig as DMDataConfig,
    TakensConfig,
    DMConfig as DMExtractConfig,
    RenderingConfig as DMRenderingConfig,
    run_dm_extraction,
    get_cycle_statistics,
    get_path_statistics,
    takens_embedding,
    plot_ami_fnn_diagnostics,
)
from topo_mimic.benchmark.experiment import (
    ClusteringConfig,
    prepare_windows_for_clustering,
    build_feature_matrix,
    compute_w2_distance_matrix,
    labels_to_cluster_dict,
)
from topo_mimic.benchmark.methods import get_method
from topo_mimic.dmwk.skeletonization.postprocess import _uniform_downsample


# ── Config dataclasses ──────────────────────────────────────────────────────


@dataclass
class VisualizationConfig:
    subsample_stride: int = 5
    window_size: int = 70
    window_stride: int = 10


@dataclass
class DemoDMConfig:
    dm_key: str = "intention"
    split: str = "all"
    takens: dict = field(default_factory=lambda: {"auto_estimate": True})
    dm: dict = field(default_factory=lambda: {"pers_delta": 0.6, "k_neighbors": 3})


@dataclass
class DemoClusteringConfig:
    n_clusters: int = 10
    target_len: int = 70
    skip_clips: tuple[int, ...] = ()
    min_len: int = 20
    max_len: int = 2000


@dataclass
class DemoRenderingConfig:
    enabled: bool = True
    walker_xml: str = "topo_mimic/dmwk/xml/walker.xml"
    max_clips: int = 10
    max_segments: int = 5
    max_samples: int = 5
    max_gt_types: int | None = None
    max_gt_samples: int | None = None
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)


@dataclass
class DemoConfig:
    dataset: str = "walker_long"
    data_path: str = "data/walker_long.h5"
    labels_csv: str = "walker_long_behavior_labels.csv"
    output_dir: str = "outputs/walker_long_demo"
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    dm_pipeline: DemoDMConfig = field(default_factory=DemoDMConfig)
    clustering: DemoClusteringConfig = field(default_factory=DemoClusteringConfig)
    rendering: DemoRenderingConfig = field(default_factory=DemoRenderingConfig)
    only_analysis: bool = False


def load_demo_config(config_path: str) -> DemoConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    vis_dict = cfg.get("visualization", {})
    vis_cfg = VisualizationConfig(**vis_dict)

    dm_dict = cfg.get("dm_pipeline", {})
    dm_cfg = DemoDMConfig(
        dm_key=dm_dict.get("dm_key", "intention"),
        split=dm_dict.get("split", "all"),
        takens=dm_dict.get("takens", {"auto_estimate": True}),
        dm=dm_dict.get("dm", {"pers_delta": 0.6, "k_neighbors": 3}),
    )

    cl_dict = cfg.get("clustering", {})
    if "skip_clips" in cl_dict:
        cl_dict["skip_clips"] = tuple(cl_dict["skip_clips"])
    cl_cfg = DemoClusteringConfig(**cl_dict)

    rend_dict = cfg.get("rendering", {})
    if "target_size" in rend_dict:
        rend_dict["target_size"] = tuple(rend_dict["target_size"])
    rend_cfg = DemoRenderingConfig(**rend_dict)

    return DemoConfig(
        dataset=cfg.get("dataset", "walker_long"),
        data_path=cfg.get("data_path", "data/walker_long.h5"),
        labels_csv=cfg.get("labels_csv", "walker_long_behavior_labels.csv"),
        output_dir=cfg.get("output_dir", "outputs/walker_long_demo"),
        visualization=vis_cfg,
        dm_pipeline=dm_cfg,
        clustering=cl_cfg,
        rendering=rend_cfg,
        only_analysis=cfg.get("only_analysis", False),
    )


# ── GT label loading ────────────────────────────────────────────────────────


def load_multi_segment_labels(
    labels_csv: str,
) -> dict[int, list[dict]]:
    """Load multi-segment behavior labels from CSV.

    Returns:
        Dict mapping clip_id -> list of segment dicts with
        "behavior", "start_frame", "end_frame" keys.
    """
    df = pd.read_csv(labels_csv)
    result: dict[int, list[dict]] = defaultdict(list)
    for _, row in df.iterrows():
        result[int(row["Clip"])].append({
            "behavior": row["Behavior"],
            "start_frame": int(row["Start_Frame"]),
            "end_frame": int(row["End_Frame"]),
        })
    for clip_id in result:
        result[clip_id].sort(key=lambda s: s["start_frame"])
    return dict(result)


def build_frame_labels(
    clip_segments: list[dict],
    clip_length: int,
) -> np.ndarray:
    """Build per-frame behavior label array from segment list.

    Returns string array of shape (clip_length,). Frames in transition
    gaps are labeled as "transition".
    """
    labels = np.full(clip_length, "transition", dtype=object)
    for seg in clip_segments:
        start = seg["start_frame"]
        end = seg["end_frame"] if seg["end_frame"] >= 0 else clip_length
        end = min(end, clip_length)
        labels[start:end] = seg["behavior"]
    return labels


# ── Plotting utilities ──────────────────────────────────────────────────────


_NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

_TAB10_NO_GRAY = [plt.cm.tab10(i) for i in range(10) if i != 7]


def _build_color_map(labels: list[str], cmap_name: str = "tab10") -> dict[str, tuple]:
    palette = _TAB10_NO_GRAY
    unique = sorted(set(l for l in labels if l != "transition"))
    cmap = {label: palette[i % len(palette)] for i, label in enumerate(unique)}
    cmap["transition"] = (0.7, 0.7, 0.7, 1.0)
    return cmap


def _scatter_plot(
    ax, coords, cat_labels, title, axis_labels, color_map=None,
):
    if color_map is None:
        color_map = _build_color_map(cat_labels)
    colors = [color_map.get(l, (0.5, 0.5, 0.5, 1.0)) for l in cat_labels]

    ax.scatter(
        coords[:, 0], coords[:, 1], c=colors, s=4, alpha=0.5,
        edgecolors="none", rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    unique = sorted(set(cat_labels))
    if len(unique) <= 20:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color_map.get(l, (0.5, 0.5, 0.5, 1.0)),
                       markersize=5, label=l)
            for l in unique
        ]
        ax.legend(handles=handles, fontsize=5, loc="best",
                  ncol=2 if len(unique) > 8 else 1)


def _run_embedding_plots(
    features: np.ndarray,
    labels: list[str],
    output_dir: str,
    prefix: str,
    suptitle: str,
):
    from sklearn.decomposition import PCA

    color_map = _build_color_map(labels)
    n_points = features.shape[0]

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(features)

    with plt.rc_context(_NATURE_RC):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
        fig.subplots_adjust(left=0.06, right=0.97, top=0.85, bottom=0.15, wspace=0.35)

        _scatter_plot(
            axes[0], pca_coords, labels,
            f"PCA (n={n_points})",
            [f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
             f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"],
            color_map,
        )

        try:
            from umap import UMAP
            pca_pre = PCA(n_components=min(50, features.shape[1]))
            feats_pca = pca_pre.fit_transform(features)
            umap_coords = UMAP(n_components=2, random_state=42).fit_transform(feats_pca)
            _scatter_plot(axes[1], umap_coords, labels, "UMAP",
                          ["UMAP-1", "UMAP-2"], color_map)
        except Exception as e:
            print(f"WARNING: UMAP failed: {e}")
            axes[1].set_title("UMAP (failed)")

        try:
            from sklearn.manifold import TSNE
            pca_pre = PCA(n_components=min(50, features.shape[1]))
            feats_pca = pca_pre.fit_transform(features)
            perp = min(30, n_points - 1)
            tsne_coords = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(feats_pca)
            _scatter_plot(axes[2], tsne_coords, labels, "t-SNE",
                          ["t-SNE-1", "t-SNE-2"], color_map)
        except Exception as e:
            print(f"WARNING: t-SNE failed: {e}")
            axes[2].set_title("t-SNE (failed)")

        fig.suptitle(suptitle, fontsize=9, fontweight="bold")
        out_path = os.path.join(output_dir, f"{prefix}_embeddings.png")
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(out_path, facecolor="white")
        plt.close(fig)
    print(f"Saved {suptitle}: {out_path}")


# ── Per-frame visualization ─────────────────────────────────────────────────


def run_per_frame_visualization(
    data: np.ndarray,
    clip_labels: dict[int, list[dict]],
    clip_indices: list[int],
    config: DemoConfig,
):
    stride = config.visualization.subsample_stride
    all_features = []
    all_labels = []

    for local_idx, clip_id in enumerate(clip_indices):
        clip_data = data[local_idx]
        T = clip_data.shape[0]
        segments = clip_labels.get(clip_id, [])
        frame_labels = build_frame_labels(segments, T)

        for fi in range(0, T, stride):
            all_features.append(clip_data[fi])
            all_labels.append(frame_labels[fi])

    features = np.array(all_features)
    print(f"Per-frame visualization: {features.shape[0]} points "
          f"(stride={stride}, {len(clip_indices)} clips)")

    _run_embedding_plots(
        features, all_labels, config.output_dir,
        "perframe", "Per-Frame Behavior Embedding",
    )


# ── Windowed visualization ──────────────────────────────────────────────────


def _majority_label(frame_labels: np.ndarray) -> str:
    unique, counts = np.unique(frame_labels, return_counts=True)
    non_trans = [(u, c) for u, c in zip(unique, counts) if u != "transition"]
    if non_trans:
        return max(non_trans, key=lambda x: x[1])[0]
    return "transition"


def run_windowed_visualization(
    data: np.ndarray,
    clip_labels: dict[int, list[dict]],
    clip_indices: list[int],
    config: DemoConfig,
):
    W = config.visualization.window_size
    stride = config.visualization.window_stride
    all_features = []
    all_labels = []

    for local_idx, clip_id in enumerate(clip_indices):
        clip_data = data[local_idx]
        T = clip_data.shape[0]
        segments = clip_labels.get(clip_id, [])
        frame_labels = build_frame_labels(segments, T)

        for start in range(0, T - W + 1, stride):
            window = clip_data[start:start + W]
            all_features.append(window.flatten())
            all_labels.append(_majority_label(frame_labels[start:start + W]))

    features = np.array(all_features)
    print(f"Windowed visualization: {features.shape[0]} windows "
          f"(W={W}, stride={stride}, {len(clip_indices)} clips)")

    _run_embedding_plots(
        features, all_labels, config.output_dir,
        "windowed", f"Windowed Behavior Embedding (W={W})",
    )


# ── DM pipeline ─────────────────────────────────────────────────────────────


def build_dm_pipeline_config(config: DemoConfig) -> PipelineConfig:
    dm_output = os.path.join(config.output_dir, "dm")

    takens_dict = dict(config.dm_pipeline.takens)
    theiler_val = takens_dict.get("theiler", "auto")
    if theiler_val == "auto":
        pass
    elif theiler_val is not None:
        theiler_val = int(theiler_val)
    takens_dict["theiler"] = theiler_val

    dm_dict = dict(config.dm_pipeline.dm)

    return PipelineConfig(
        output_dir=dm_output,
        data=DMDataConfig(
            dataset=config.dataset,
            dm_key=config.dm_pipeline.dm_key,
            rodent_rollout_dir=config.data_path,
            split=config.dm_pipeline.split,
        ),
        takens=TakensConfig(**takens_dict),
        dm=DMExtractConfig(use_dm_path=True, **dm_dict),
        rendering=DMRenderingConfig(enabled=False),
    )


def run_dm_on_data(
    data: np.ndarray,
    clip_indices: list[int],
    config: DemoConfig,
) -> tuple[list, list, list | None]:
    import yaml as _yaml
    from topo_mimic.dmwk.skeletonization.preprocess import estimate_delay_embedding_params
    from topo_mimic.dmwk.utils.feature_transforms import get_hdf5_key_for_dm_key
    from topo_mimic.dmwk.utils.load_utils import get_aggregate_data
    import functools

    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")

    dm_output = os.path.join(config.output_dir, "dm")
    os.makedirs(dm_output, exist_ok=True)

    pipeline_cfg = build_dm_pipeline_config(config)
    pipeline_cfg.data.rodent_rollout_dir = config.data_path
    pipeline_cfg.output_dir = dm_output

    print(f"Running DM on {data.shape[0]} clips (pre-loaded)")

    if pipeline_cfg.takens.auto_estimate:
        print("Auto-estimating Takens parameters (AMI + FNN)...")
        hdf5_key = get_hdf5_key_for_dm_key(pipeline_cfg.data.dm_key)
        tau_star, m_star, diagnostics = estimate_delay_embedding_params(
            rodent_rollout_dir=pipeline_cfg.data.rodent_rollout_dir,
            get_aggregate_data_fn=get_aggregate_data,
            num_clips=len(clip_indices),
            intention_key=f"/{hdf5_key}",
            clip_indices=clip_indices,
            transform_fn=None,
            max_tau=pipeline_cfg.takens.max_tau,
            bins=pipeline_cfg.takens.bins,
            m_max=pipeline_cfg.takens.m_max,
            R_tol=pipeline_cfg.takens.R_tol,
            A_tol=pipeline_cfg.takens.A_tol,
            fnn_thr=pipeline_cfg.takens.fnn_thr,
            theiler=pipeline_cfg.takens.theiler,
            max_points=pipeline_cfg.takens.max_points,
            plot=False,
        )
        pipeline_cfg.takens.tau = tau_star
        pipeline_cfg.takens.m = m_star
        print(f"Estimated: tau* = {tau_star}, m* = {m_star}")

        diag_path = os.path.join(dm_output, "ami_fnn_diagnostics.pkl")
        with open(diag_path, "wb") as f:
            pkl.dump({"tau_star": tau_star, "m_star": m_star, "diagnostics": diagnostics}, f)

        plot_path = os.path.join(dm_output, "ami_fnn_diagnostics.png")
        plot_ami_fnn_diagnostics(diagnostics, tau_star, m_star, plot_path,
                                 fnn_thr=pipeline_cfg.takens.fnn_thr)
    else:
        print(f"Using fixed Takens: m={pipeline_cfg.takens.m}, tau={pipeline_cfg.takens.tau}")

    print("Running DM extraction...")
    all_cycle_dm, all_G_sm, all_path_dm = run_dm_extraction(data, pipeline_cfg)

    stats = get_cycle_statistics(all_cycle_dm)
    print("\nCycle Statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val:.2f}" if isinstance(val, float) else f"  {key}: {val}")

    stats_path = os.path.join(dm_output, "cycle_stats.yaml")
    with open(stats_path, "w") as f:
        _yaml.dump(stats, f)

    if all_path_dm is not None:
        path_stats = get_path_statistics(all_path_dm)
        print("\nPath Statistics:")
        for key, val in path_stats.items():
            print(f"  {key}: {val:.2f}" if isinstance(val, float) else f"  {key}: {val}")

    return all_cycle_dm, all_G_sm, all_path_dm


# ── Clustering ──────────────────────────────────────────────────────────────


def run_clustering(
    all_dm_results: list,
    data: np.ndarray,
    clip_indices: list[int],
    clustering_cfg: DemoClusteringConfig,
    output_dir: str,
) -> tuple[dict[int, list[tuple[int, np.ndarray]]], np.ndarray, list]:
    exp_clustering = ClusteringConfig(
        min_len=clustering_cfg.min_len,
        max_len=clustering_cfg.max_len,
        target_len=clustering_cfg.target_len,
        skip_clips=clustering_cfg.skip_clips,
    )

    windows, cleaned_loops = prepare_windows_for_clustering(
        all_dm_results, data, exp_clustering, clip_indices
    )
    print(f"Prepared {len(windows)} windows for clustering")

    if len(windows) == 0:
        print("WARNING: No valid windows for clustering")
        return {}, np.array([]), windows

    w2_cache = os.path.join(output_dir, "w2_distance_matrix.pkl")
    if os.path.exists(w2_cache):
        print(f"Loading cached W2 matrix from {w2_cache}")
        with open(w2_cache, "rb") as f:
            w2_data = pkl.load(f)
        Dn = w2_data["Dn"]
    else:
        D, Dn = compute_w2_distance_matrix(
            windows, data, clustering_cfg.target_len
        )
        with open(w2_cache, "wb") as f:
            pkl.dump({"D": D, "Dn": Dn}, f)
        print(f"Cached W2 matrix to {w2_cache}")

    method = get_method("w2_dmc_int_kmeans", {"distance_matrix": Dn})
    X = build_feature_matrix(windows, data)
    labels = method.fit_predict(X, clustering_cfg.n_clusters, seed=0)
    cluster_dict = labels_to_cluster_dict(labels, windows, clip_indices=clip_indices)

    return cluster_dict, labels, windows


# ── DM segment comparison with GT ──────────────────────────────────────────


def build_dm_segment_gt_labels(
    windows: list[tuple[int, np.ndarray]],
    clip_labels: dict[int, list[dict]],
    clip_indices: list[int],
    clip_length: int,
) -> list[str]:
    """Assign GT behavior label to each DM window by majority vote."""
    gt_labels = []
    for local_idx, frames in windows:
        clip_id = clip_indices[local_idx]
        segments = clip_labels.get(clip_id, [])
        frame_labels = build_frame_labels(segments, clip_length)
        window_labels = frame_labels[frames[frames < clip_length]]
        gt_labels.append(_majority_label(window_labels))
    return gt_labels


def run_dm_vs_gt_visualization(
    windows: list[tuple[int, np.ndarray]],
    cluster_labels_arr: np.ndarray,
    data: np.ndarray,
    clip_labels: dict[int, list[dict]],
    clip_indices: list[int],
    clip_length: int,
    config: DemoConfig,
    variant_prefix: str,
    cluster_label_names: dict[int, str] | None = None,
):
    from sklearn.decomposition import PCA

    features = build_feature_matrix(windows, data)
    gt_labels = build_dm_segment_gt_labels(
        windows, clip_labels, clip_indices, clip_length
    )

    dm_labels = []
    for l in cluster_labels_arr:
        if l < 0:
            dm_labels.append("filtered")
        elif cluster_label_names and l in cluster_label_names:
            dm_labels.append(cluster_label_names[l])
        else:
            dm_labels.append(f"cluster_{l}")

    gt_color_map = _build_color_map(gt_labels)
    dm_color_map = _build_color_map(dm_labels)

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(features)

    with plt.rc_context(_NATURE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.4))
        fig.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.15, wspace=0.35)

        axis_labels = [
            f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
            f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        ]
        _scatter_plot(axes[0], pca_coords, gt_labels,
                      f"GT Behavior ({len(windows)} segments)", axis_labels, gt_color_map)
        _scatter_plot(axes[1], pca_coords, dm_labels,
                      f"DM+W2 Clusters", axis_labels, dm_color_map)

        fig.suptitle(f"{variant_prefix.title()} DM Segments: GT vs Discovered",
                     fontsize=9, fontweight="bold")
        out_path = os.path.join(config.output_dir, f"{variant_prefix}_gt_vs_dm_pca.png")
        fig.savefig(out_path, facecolor="white")
        plt.close(fig)
    print(f"Saved {variant_prefix} GT vs DM comparison: {out_path}")


# ── Grid rendering (rows=clips, cols=segments) ─────────────────────────────


def _get_longest_k_segments(
    all_dm_results: list,
    clip_idx: int,
    k: int,
) -> list[np.ndarray]:
    """Get K longest DM segments for a given clip."""
    if clip_idx >= len(all_dm_results) or all_dm_results[clip_idx] is None:
        return []

    clip_segments = all_dm_results[clip_idx]
    if not clip_segments:
        return []

    sorted_segs = sorted(clip_segments, key=lambda s: len(s), reverse=True)
    return [np.array(seg) for seg in sorted_segs[:k]]


def render_clip_segment_grid(
    all_dm_results: list,
    data_path: str,
    xml_path: str,
    clip_indices: list[int],
    max_clips: int,
    max_segments: int,
    output_path: str,
    clip_labels: dict[int, list[dict]] | None = None,
    framerate: int = 30,
    target_size: tuple[int, int] = (320, 240),
):
    """Render grid video: rows = clips, cols = longest K DM segments."""
    import h5py
    import mujoco
    from tqdm import tqdm

    clips_with_segs = []
    for local_idx in range(len(all_dm_results)):
        segs = _get_longest_k_segments(all_dm_results, local_idx, max_segments)
        if segs:
            clips_with_segs.append((local_idx, segs))

    clips_with_segs.sort(key=lambda x: sum(len(s) for s in x[1]), reverse=True)
    clips_with_segs = clips_with_segs[:max_clips]

    if not clips_with_segs:
        print("WARNING: No segments to render")
        return

    print(f"Rendering grid: {len(clips_with_segs)} clips x up to {max_segments} segments")

    with h5py.File(data_path, "r") as f:
        qpos_all = f["qpos_rollout"]

        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(mj_model)
        renderer = mujoco.Renderer(mj_model, height=target_size[1], width=target_size[0])

        grid_chunks: dict[int, list[list[np.ndarray]]] = {}
        grid_labels: dict[int, str] = {}

        for row_idx, (local_idx, segments) in enumerate(tqdm(clips_with_segs, desc="Rendering clips")):
            clip_id = clip_indices[local_idx]
            qpos_clip = qpos_all[clip_id]
            row_chunks = []

            for seg_frames in segments:
                chunk_frames = []
                for fi in sorted(int(f) for f in seg_frames):
                    if fi >= len(qpos_clip):
                        continue
                    mj_data.qpos[:] = qpos_clip[fi]
                    mujoco.mj_forward(mj_model, mj_data)
                    renderer.update_scene(mj_data, camera="side")
                    chunk_frames.append(renderer.render().copy())
                if chunk_frames:
                    row_chunks.append(chunk_frames)

            if row_chunks:
                grid_chunks[row_idx] = row_chunks
                label_parts = [f"Clip {clip_id}"]
                if clip_labels and clip_id in clip_labels:
                    behaviors = [s["behavior"] for s in clip_labels[clip_id]]
                    label_parts.append(" | ".join(behaviors[:4]))
                grid_labels[row_idx] = ", ".join(label_parts)

        renderer.close()

    if grid_chunks:
        from topo_mimic.dmwk.skeletonization.rendering import render_grid_clusters_side_by_side
        render_grid_clusters_side_by_side(
            grid_chunks,
            out_path=output_path,
            framerate=framerate,
            max_samples_per_cluster=max_segments,
            target_size=target_size,
            title="DM Segments per Clip (rows=clips, cols=longest segments)",
            cluster_labels=grid_labels,
        )
        print(f"Saved segment grid video: {output_path}")


# ── Label YAML helpers ──────────────────────────────────────────────────────


def load_labels_yaml(yaml_path: str) -> dict[int, str]:
    with open(yaml_path, "r") as f:
        content = yaml.safe_load(f)
    labels = content.get("labels", {})
    return {int(k): v for k, v in labels.items() if v}


def write_labels_template(
    yaml_path: str,
    n_clusters: int,
    variant_prefix: str,
):
    if os.path.exists(yaml_path):
        print(f"Label YAML already exists: {yaml_path}")
        return
    content = {
        "experiment_id": f"walker_long_{variant_prefix}",
        "method": "w2_dmc_int_kmeans",
        "n_clusters": n_clusters,
        "seed": 0,
        "labels": {i: "" for i in range(n_clusters)},
    }
    with open(yaml_path, "w") as f:
        f.write("# AUTO-GENERATED - Fill in labels after watching videos\n")
        f.write('# Label format: single behavior ("rear"), transition ("walk rear")\n\n')
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)
    print(f"Generated label template: {yaml_path}")


# ── Confusion table ─────────────────────────────────────────────────────────


def print_segment_confusion(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    clip_labels: dict[int, list[dict]],
    clip_length: int,
    cluster_label_names: dict[int, str] | None = None,
):
    behavior_counts: dict[int, dict[str, int]] = {}
    for cid, items in cluster_dict.items():
        behavior_counts[cid] = defaultdict(int)
        for clip_id, frames in items:
            segments = clip_labels.get(clip_id, [])
            frame_labels = build_frame_labels(segments, clip_length)
            valid = frames[frames < clip_length]
            majority = _majority_label(frame_labels[valid])
            behavior_counts[cid][majority] += 1

    all_behaviors = sorted(set(
        b for counts in behavior_counts.values() for b in counts
    ))
    cluster_ids = sorted(behavior_counts.keys())

    print("\n" + "=" * 60)
    print("Cluster vs GT Behavior Confusion")
    print("=" * 60)

    header = ["Cluster".ljust(20)]
    for b in all_behaviors:
        header.append(b[:12].rjust(13))
    print(" | ".join(header))
    print("-" * (21 + 16 * len(all_behaviors)))

    for cid in cluster_ids:
        name = cluster_label_names.get(cid, f"C{cid}") if cluster_label_names else f"C{cid}"
        row = [name.ljust(20)]
        for b in all_behaviors:
            row.append(str(behavior_counts[cid].get(b, 0)).rjust(13))
        print(" | ".join(row))


# ── Main pipeline ───────────────────────────────────────────────────────────


def run_demo(config: DemoConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("Walker Long-Sequence Analysis Demo")
    print("=" * 60)
    print(f"Dataset: {config.dataset}")
    print(f"Data: {config.data_path}")
    print(f"Labels: {config.labels_csv}")
    print(f"Output: {config.output_dir}")
    print(f"Only-analysis: {config.only_analysis}")

    # --- Stage 1: Load data ---
    print("\n=== Stage 1: Loading data ===")
    import functools
    import multiprocessing as mp
    from tqdm import tqdm
    from topo_mimic.dmwk.utils.load_utils import get_aggregate_data, get_clip_indices
    from topo_mimic.dmwk.utils.feature_transforms import get_hdf5_key_for_dm_key

    preset = get_dataset_preset(config.dataset)
    clip_indices = get_clip_indices(
        config.dm_pipeline.split,
        total_clips=preset.total_clips,
        test_size=preset.test_size,
    )

    clip_labels = load_multi_segment_labels(config.labels_csv)
    n_labeled = sum(1 for c in clip_indices if c in clip_labels)
    print(f"Loaded labels for {n_labeled}/{len(clip_indices)} clips")

    hdf5_key = get_hdf5_key_for_dm_key(config.dm_pipeline.dm_key)
    work_fn = functools.partial(
        get_aggregate_data, f"/{hdf5_key}", [], path=config.data_path
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        data_list = list(tqdm(
            pool.imap(work_fn, clip_indices),
            total=len(clip_indices),
            desc=f"Loading {hdf5_key}",
        ))

    data = np.array(data_list)
    if data.ndim == 2:
        n_dim = data.shape[-1]
        data = data.reshape(len(clip_indices), -1, n_dim)
    clip_length = data.shape[1]
    print(f"Loaded data: {data.shape} (clip_length={clip_length})")

    # --- Stage 2: Per-frame visualization ---
    print("\n=== Stage 2: Per-frame visualization ===")
    run_per_frame_visualization(data, clip_labels, clip_indices, config)

    # --- Stage 3: Windowed visualization ---
    print("\n=== Stage 3: Windowed visualization ===")
    run_windowed_visualization(data, clip_labels, clip_indices, config)

    # --- Stage 4+5: DM extraction + clustering ---
    dm_output = os.path.join(config.output_dir, "dm")
    cycles_path = os.path.join(dm_output, "all_cycle_dm.pkl")
    paths_path = os.path.join(dm_output, "all_path_dm.pkl")

    if not config.only_analysis:
        print("\n=== Stage 4: DM extraction ===")
        if os.path.exists(cycles_path):
            print(f"Found cached DM results at {cycles_path}, skipping extraction")
            with open(cycles_path, "rb") as f:
                all_cycle_dm = pkl.load(f)
            all_path_dm = None
            if os.path.exists(paths_path):
                with open(paths_path, "rb") as f:
                    all_path_dm = pkl.load(f)
        else:
            all_cycle_dm, _, all_path_dm = run_dm_on_data(data, clip_indices, config)

        for variant_name, dm_results, prefix in [
            ("Cycle", all_cycle_dm, "cycle"),
            ("Path", all_path_dm, "path"),
        ]:
            if dm_results is None:
                continue

            print(f"\n=== Stage 5: {variant_name} W2 clustering ===")
            variant_output = os.path.join(config.output_dir, prefix)
            os.makedirs(variant_output, exist_ok=True)

            cluster_dict, labels, windows = run_clustering(
                dm_results, data, clip_indices, config.clustering, variant_output,
            )

            with open(os.path.join(variant_output, "cluster_results.pkl"), "wb") as f:
                pkl.dump({
                    "cluster_dict": cluster_dict,
                    "labels": labels,
                    "windows": windows,
                }, f)
            print(f"Saved {variant_name} clustering results")

            if cluster_dict and windows:
                run_dm_vs_gt_visualization(
                    windows, labels, data, clip_labels, clip_indices,
                    clip_length, config, prefix,
                )
                print_segment_confusion(cluster_dict, clip_labels, clip_length)

            n_clusters = len(set(labels[labels >= 0])) if len(labels) > 0 else 0
            if n_clusters > 0:
                yaml_path = os.path.join(config.output_dir, f"{prefix}_labels_seed_0.yaml")
                write_labels_template(yaml_path, n_clusters, prefix)

            # Grid rendering (rows=clips, cols=segments)
            if config.rendering.enabled and dm_results:
                print(f"\n=== {variant_name} Grid Rendering ===")
                grid_path = os.path.join(config.output_dir, f"{prefix}_segment_grid.mp4")
                try:
                    render_clip_segment_grid(
                        dm_results,
                        config.data_path,
                        config.rendering.walker_xml,
                        clip_indices,
                        max_clips=config.rendering.max_clips,
                        max_segments=config.rendering.max_segments,
                        output_path=grid_path,
                        clip_labels=clip_labels,
                        framerate=config.rendering.framerate,
                        target_size=config.rendering.target_size,
                    )
                except Exception as e:
                    print(f"WARNING: {variant_name} grid rendering failed: {e}")
                    import traceback
                    traceback.print_exc()

            # Cluster video rendering
            if config.rendering.enabled and cluster_dict:
                print(f"\n=== {variant_name} Cluster Rendering ===")
                try:
                    from demos.demo_walker_transitions_short import chunk_loops_walker
                    from topo_mimic.dmwk.skeletonization.rendering import render_grid_clusters_side_by_side

                    dm_chunks = chunk_loops_walker(
                        cluster_dict, config.data_path, config.rendering.walker_xml,
                        max_samples=config.rendering.max_samples,
                    )
                    video_path = os.path.join(config.output_dir, f"{prefix}_clusters.mp4")
                    render_grid_clusters_side_by_side(
                        dm_chunks, out_path=video_path,
                        framerate=config.rendering.framerate,
                        max_samples_per_cluster=config.rendering.max_samples,
                        target_size=config.rendering.target_size,
                        title=f"{variant_name} DM+W2 Clusters",
                    )
                except Exception as e:
                    print(f"WARNING: {variant_name} cluster rendering failed: {e}")
                    import traceback
                    traceback.print_exc()

    else:
        # --- Only-analysis mode ---
        print("\n=== Loading cached results (only_analysis=true) ===")
        assert os.path.exists(cycles_path), f"Cached cycles not found: {cycles_path}"
        with open(cycles_path, "rb") as f:
            all_cycle_dm = pkl.load(f)

        all_path_dm = None
        if os.path.exists(paths_path):
            with open(paths_path, "rb") as f:
                all_path_dm = pkl.load(f)

        for variant_name, dm_results, prefix in [
            ("Cycle", all_cycle_dm, "cycle"),
            ("Path", all_path_dm, "path"),
        ]:
            if dm_results is None:
                continue

            variant_output = os.path.join(config.output_dir, prefix)
            results_path = os.path.join(variant_output, "cluster_results.pkl")
            if not os.path.exists(results_path):
                print(f"No cached {variant_name} results at {results_path}, skipping")
                continue

            with open(results_path, "rb") as f:
                cached = pkl.load(f)
            cluster_dict = cached["cluster_dict"]
            labels = cached["labels"]
            windows = cached["windows"]
            print(f"Loaded {variant_name}: {len(cluster_dict)} clusters, {len(windows)} windows")

            cluster_label_names = None
            yaml_path = os.path.join(config.output_dir, f"{prefix}_labels_seed_0.yaml")
            if os.path.exists(yaml_path):
                cluster_label_names = load_labels_yaml(yaml_path)
                if cluster_label_names:
                    print(f"Loaded {variant_name} cluster labels from {yaml_path}")

            if cluster_dict and windows:
                run_dm_vs_gt_visualization(
                    windows, labels, data, clip_labels, clip_indices,
                    clip_length, config, prefix, cluster_label_names,
                )
                print_segment_confusion(
                    cluster_dict, clip_labels, clip_length, cluster_label_names,
                )

            if config.rendering.enabled and dm_results:
                grid_path = os.path.join(config.output_dir, f"{prefix}_segment_grid.mp4")
                try:
                    render_clip_segment_grid(
                        dm_results,
                        config.data_path,
                        config.rendering.walker_xml,
                        clip_indices,
                        max_clips=config.rendering.max_clips,
                        max_segments=config.rendering.max_segments,
                        output_path=grid_path,
                        clip_labels=clip_labels,
                        framerate=config.rendering.framerate,
                        target_size=config.rendering.target_size,
                    )
                except Exception as e:
                    print(f"WARNING: {variant_name} grid rendering failed: {e}")
                    import traceback
                    traceback.print_exc()

    print("\n" + "=" * 60)
    print("Walker Long-Sequence Demo Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Walker long-sequence analysis demo")
    parser.add_argument(
        "--config",
        default="configs/demos/demo_walker_transitions_long.yaml",
        help="Path to demo config YAML",
    )
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = load_demo_config(args.config)
    else:
        print(f"Config not found: {args.config}, using defaults")
        config = DemoConfig()

    run_demo(config)


if __name__ == "__main__":
    main()
