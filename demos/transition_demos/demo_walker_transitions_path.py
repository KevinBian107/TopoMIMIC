#!/usr/bin/env python3
"""
Walker Transition Demo (Path Analysis)
========================================
Analysis pipeline for long walker clips with multiple behavior transitions per clip.

Runs DM all-pairs-path extraction, GT-labeled embeddings, and debug
side-by-side rendering (original clip vs DM segments).

Usage:
    python demos/transition_demos/demo_walker_transitions_path.py
    python demos/transition_demos/demo_walker_transitions_path.py --config configs/demos/transition_demos/demo_walker_transitions_path.yaml
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
import networkx as nx
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from topo_mimic.dmwk.utils.dataset_config import get_dataset_preset
from topo_mimic.dmwk.dm_pipeline import (
    PipelineConfig,
    DataConfig as DMDataConfig,
    TakensConfig,
    DMConfig as DMExtractConfig,
    RenderingConfig as DMRenderingConfig,
    run_dm_extraction,
    get_cycle_statistics,
    plot_ami_fnn_diagnostics,
)
from topo_mimic.dmwk.skeletonization.rendering import _ensure_rgb_uint8


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
    top_k_minima: int = 6
    top_k_paths: int = 1



@dataclass
class DemoRenderingConfig:
    enabled: bool = False
    pca_debug: bool = False
    walker_xml: str = "topo_mimic/dmwk/xml/walker.xml"
    max_clips: int = 10
    max_segments: int = 5
    max_samples: int = 5
    max_gt_types: int | None = None
    max_gt_samples: int | None = None
    max_debug_clips: int = 3
    top_k_debug_paths: int = 3
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)


@dataclass
class TransitionAnalysisConfig:
    enabled: bool = True
    top_k_paths: int = 3
    min_segment_len: int = 10


@dataclass
class DemoConfig:
    dataset: str = "walker_long"
    data_path: str = "data/walker_long.h5"
    labels_csv: str = "walker_long_behavior_labels.csv"
    output_dir: str = "outputs/walker_long_demo"
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    dm_pipeline: DemoDMConfig = field(default_factory=DemoDMConfig)
    rendering: DemoRenderingConfig = field(default_factory=DemoRenderingConfig)
    transition_analysis: TransitionAnalysisConfig = field(default_factory=TransitionAnalysisConfig)


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
        top_k_minima=dm_dict.get("top_k_minima", 6),
        top_k_paths=dm_dict.get("top_k_paths", 1),
    )

    rend_dict = cfg.get("rendering", {})
    if "target_size" in rend_dict:
        rend_dict["target_size"] = tuple(rend_dict["target_size"])
    rend_cfg = DemoRenderingConfig(**rend_dict)

    ta_dict = cfg.get("transition_analysis", {})
    ta_cfg = TransitionAnalysisConfig(**ta_dict)

    return DemoConfig(
        dataset=cfg.get("dataset", "walker_long"),
        data_path=cfg.get("data_path", "data/walker_long.h5"),
        labels_csv=cfg.get("labels_csv", "walker_long_behavior_labels.csv"),
        output_dir=cfg.get("output_dir", "outputs/walker_long_demo"),
        visualization=vis_cfg,
        dm_pipeline=dm_cfg,
        rendering=rend_cfg,
        transition_analysis=ta_cfg,
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


_TAB20_NO_GRAY = [plt.cm.tab20(i) for i in range(20) if i not in (14, 15)]


def _build_color_map(labels: list[str], cmap_name: str = "tab10") -> dict[str, tuple]:
    unique = sorted(set(l for l in labels if l != "transition"))
    palette = _TAB20_NO_GRAY if len(unique) > 9 else _TAB10_NO_GRAY
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
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6))
        fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.32, wspace=0.35)

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

        unique = sorted(set(labels))
        if len(unique) <= 40:
            handles = [
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=color_map.get(l, (0.5, 0.5, 0.5, 1.0)),
                           markersize=5, label=l)
                for l in unique
            ]
            fig.legend(
                handles=handles, fontsize=5,
                loc="lower center", bbox_to_anchor=(0.5, 0.02),
                ncol=min(len(unique), 5), frameon=False,
            )

        out_path = os.path.join(output_dir, f"{prefix}_embeddings.png")
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(out_path, facecolor="white")
        plt.close(fig)
    print(f"Saved {suptitle}: {out_path}")


# ── Auto-labeling from GT ──────────────────────────────────────────────────


def _majority_label(frame_labels: np.ndarray) -> str:
    unique, counts = np.unique(frame_labels, return_counts=True)
    non_trans = [(u, c) for u, c in zip(unique, counts) if u != "transition"]
    if non_trans:
        return max(non_trans, key=lambda x: x[1])[0]
    return "transition"


def _normalize_transition(label: str) -> str:
    """Make transition labels direction-agnostic by sorting behaviors alphabetically."""
    parts = label.split("\u2192")
    if len(parts) > 1:
        unique = []
        for p in parts:
            if not unique or unique[-1] != p:
                unique.append(p)
        return "\u2194".join(sorted(unique)) if len(unique) > 1 else unique[0]
    return label


def _segment_transition_pattern(
    clip_segments: list[dict],
    frame_start: int,
    frame_end: int,
) -> str:
    """Identify the GT behavior transition that a DM segment spans.

    Looks at which GT behavior segments overlap with [frame_start, frame_end],
    returning them in temporal order joined by arrow, e.g. "walk_slow→hop".
    This correctly captures transitions even when the DM segment falls in a
    gap between two GT behavior regions.
    """
    overlapping = []
    for seg in clip_segments:
        seg_start = seg["start_frame"]
        seg_end = seg["end_frame"] if seg["end_frame"] >= 0 else frame_end + 1
        if seg_start < frame_end and seg_end > frame_start:
            overlapping.append((seg_start, seg["behavior"]))

    if not overlapping:
        before = [s for s in clip_segments if s["end_frame"] <= frame_start and s["end_frame"] >= 0]
        after = [s for s in clip_segments if s["start_frame"] >= frame_end]
        parts = []
        if before:
            parts.append(max(before, key=lambda s: s["end_frame"])["behavior"])
        if after:
            parts.append(min(after, key=lambda s: s["start_frame"])["behavior"])
        if parts:
            seen = []
            for p in parts:
                if not seen or seen[-1] != p:
                    seen.append(p)
            return _normalize_transition("\u2192".join(seen))
        return "transition"

    overlapping.sort(key=lambda x: x[0])
    behaviors = []
    for _, beh in overlapping:
        if not behaviors or behaviors[-1] != beh:
            behaviors.append(beh)
    return _normalize_transition("\u2192".join(behaviors))


# ── Per-frame and windowed visualization ─────────────────────────────────────


def run_perframe_visualization(
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
) -> tuple[list, list]:
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
    all_cycle_dm, all_G_sm, _ = run_dm_extraction(data, pipeline_cfg)

    stats = get_cycle_statistics(all_cycle_dm)
    print("\nCycle Statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val:.2f}" if isinstance(val, float) else f"  {key}: {val}")

    stats_path = os.path.join(dm_output, "cycle_stats.yaml")
    with open(stats_path, "w") as f:
        _yaml.dump(stats, f)

    return all_cycle_dm, all_G_sm


# ── All-pairs path extraction ─────────────────────────────────────────────


@dataclass
class AllPairsPathResult:
    minima: list[int]
    paths_remapped: list[np.ndarray]
    pair_labels: list[tuple[int, int]]
    disconnected_pairs: list[tuple[int, int]]


def _find_all_persistent_minima(
    output_dir: str,
    top_k: int = 10,
    min_persistence: float = 0.0,
) -> list[int]:
    from topo_mimic.dmwk.skeletonization.dm_path import (
        _parse_persistence_pairs,
        _parse_sorted_weights,
    )
    pairs = _parse_persistence_pairs(output_dir)
    weights = _parse_sorted_weights(output_dir)

    type0_pairs = [(v0, v1, pers) for v0, v1, t, pers in pairs if t == 0]
    type0_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    seen = set()
    minima = []
    for v0, v1, pers in type0_pairs:
        if abs(pers) < min_persistence:
            break
        w0 = weights.get(v0, float("inf"))
        w1 = weights.get(v1, float("inf"))
        min_v = v0 if w0 < w1 else v1
        if min_v not in seen:
            minima.append(min_v)
            seen.add(min_v)
        if len(minima) >= top_k:
            break
    return minima


def extract_allpairs_paths(
    G_dm: nx.Graph,
    output_dir: str,
    remap_to_orig: dict[int, int],
    n_takens: int,
    min_segment_len: int = 10,
    top_k_minima: int = 6,
    min_persistence: float = 0.0,
) -> AllPairsPathResult:
    from topo_mimic.dmwk.skeletonization.remap import remap_cycle_to_original
    from itertools import combinations

    minima = _find_all_persistent_minima(output_dir, top_k_minima, min_persistence)
    graph_minima = [m for m in minima if m in G_dm]

    all_paths = []
    all_pair_labels = []
    disconnected_pairs: list[tuple[int, int]] = []

    for m_a, m_b in combinations(graph_minima, 2):
        try:
            raw_paths = list(nx.edge_disjoint_paths(G_dm, m_a, m_b))
        except (nx.NetworkXError, nx.NetworkXNoPath, nx.NetworkXUnfeasible):
            disconnected_pairs.append((m_a, m_b))
            continue

        best = None
        for p in raw_paths:
            remapped = remap_cycle_to_original(p, remap_to_orig, n_takens)
            if len(remapped) >= min_segment_len:
                if best is None or len(remapped) > len(best):
                    best = np.array(remapped, dtype=int)

        if best is not None:
            all_paths.append(best)
            all_pair_labels.append((m_a, m_b))
        else:
            disconnected_pairs.append((m_a, m_b))

    return AllPairsPathResult(
        minima=graph_minima,
        paths_remapped=all_paths,
        pair_labels=all_pair_labels,
        disconnected_pairs=disconnected_pairs,
    )


def run_allpairs_path_extraction(
    all_G_sm: list,
    clip_indices: list[int],
    config: DemoConfig,
) -> list[list[np.ndarray]]:
    from topo_mimic.dmwk.skeletonization.remap import parse_swr_remap

    dm_output = os.path.join(config.output_dir, "dm")
    dm_dict = dict(config.dm_pipeline.dm)
    min_seg = dm_dict.get("min_segment_len", 10)

    all_allpairs_dm: list[list[np.ndarray]] = []

    for local_idx, G_sm in enumerate(all_G_sm):
        clip_id = clip_indices[local_idx]
        clip_dir = os.path.join(dm_output, "clips", f"clip_{local_idx}")

        feat_path = os.path.join(clip_dir, "features.txt")
        if not os.path.exists(feat_path):
            all_allpairs_dm.append([])
            continue

        with open(feat_path) as f:
            n_takens = sum(1 for _ in f)

        remap = parse_swr_remap(clip_dir)

        result = extract_allpairs_paths(
            G_sm, clip_dir, remap, n_takens,
            min_segment_len=min_seg,
            top_k_minima=config.dm_pipeline.top_k_minima,
        )

        all_allpairs_dm.append(result.paths_remapped)
        n_pairs = len(result.minima) * (len(result.minima) - 1) // 2
        msg = (
            f"  Clip {clip_id}: {len(result.minima)} minima, "
            f"{len(result.paths_remapped)}/{n_pairs} pairs connected"
        )
        if result.disconnected_pairs:
            msg += f", {len(result.disconnected_pairs)} DISCONNECTED: {result.disconnected_pairs}"
        print(msg)

    return all_allpairs_dm


# ── DM segment GT-labeled embeddings ──────────────────────────────────────


def _label_segment_from_gt(
    local_idx: int,
    frames: np.ndarray,
    clip_indices: list[int],
    clip_labels: dict[int, list[dict]],
    clip_length: int,
) -> str:
    clip_id = clip_indices[local_idx]
    segments = clip_labels.get(clip_id, [])
    valid = frames[frames < clip_length]
    if len(valid) == 0:
        return "unknown"
    return _segment_transition_pattern(segments, int(valid.min()), int(valid.max()) + 1)


def run_dm_perframe_embedding(
    windows: list[tuple[int, np.ndarray]],
    data: np.ndarray,
    clip_indices: list[int],
    clip_labels: dict[int, list[dict]],
    clip_length: int,
    config: DemoConfig,
    variant_prefix: str,
    output_dir: str | None = None,
):
    """Per-frame embedding of DM segments, each frame labeled by GT transition."""
    all_features = []
    all_labels = []

    for local_idx, frames in windows:
        label = _label_segment_from_gt(
            local_idx, frames, clip_indices, clip_labels, clip_length,
        )
        clip_data = data[local_idx]
        for fi in frames:
            fi = int(fi)
            if fi < clip_data.shape[0]:
                all_features.append(clip_data[fi])
                all_labels.append(label)

    if not all_features:
        print(f"  No frames for {variant_prefix} per-frame embedding")
        return

    features = np.array(all_features)
    print(f"{variant_prefix.title()} per-frame embedding: {features.shape[0]} frames, "
          f"{len(set(all_labels))} labels")

    out = output_dir if output_dir is not None else config.output_dir
    _run_embedding_plots(
        features, all_labels, out,
        f"{variant_prefix}_dm_perframe",
        f"{variant_prefix.title()} DM Per-Frame Embedding",
    )




def run_dm_segment_perframe_embedding(
    data: np.ndarray,
    clip_indices: list[int],
    clip_labels: dict[int, list[dict]],
    clip_length: int,
    config: DemoConfig,
    all_dm_segments: list,
    top_k: int = 1,
    variant_name: str = "path",
):
    """Per-frame embedding using top K longest DM segments per clip, labeled by clip transition sequence."""
    all_features = []
    all_labels = []

    for local_idx, clip_id in enumerate(clip_indices):
        segments = clip_labels.get(clip_id, [])
        if len(segments) < 2:
            continue
        clip_data = data[local_idx]
        clip_label = "\u2192".join(seg["behavior"] for seg in segments)

        if local_idx < len(all_dm_segments):
            segs = all_dm_segments[local_idx]
            if segs:
                sorted_segs = sorted(segs, key=len, reverse=True)[:top_k]
                for seg in sorted_segs:
                    for fi in seg:
                        fi = int(fi)
                        if fi < clip_data.shape[0]:
                            all_features.append(clip_data[fi])
                            all_labels.append(clip_label)

    if not all_features:
        print(f"  No {variant_name} frames found")
        return

    features = np.array(all_features)
    print(f"{variant_name.title()} per-frame embedding: {features.shape[0]} frames, "
          f"{len(set(all_labels))} labels")

    _run_embedding_plots(
        features, all_labels, config.output_dir,
        f"dm_{variant_name}_perframe",
        f"DM {variant_name.title()} Per-Frame Embedding (top-{top_k})",
    )


# ── Side-by-side debug rendering ───────────────────────────────────────────


def render_clip_debug_side_by_side(
    all_dm_results: list,
    data_path: str,
    xml_path: str,
    clip_indices: list[int],
    clip_labels: dict[int, list[dict]],
    output_dir: str,
    variant_prefix: str,
    max_clips: int = 10,
    top_k_paths: int = 3,
    framerate: int = 30,
    target_size: tuple[int, int] = (320, 240),
):
    """Render per-clip debug videos: original clip on left, deduplicated sub-segments on right.

    Takes the top K longest paths per clip, splits each into contiguous
    sub-segments, and deduplicates across paths. Each unique sub-segment
    gets one tile with a distinct color. The tile border lights up in its
    color when playback reaches its frame range.
    """
    import h5py
    import mujoco
    import math
    import bisect
    from tqdm import tqdm

    Wc, Hc = target_size
    try:
        font_label = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_title = ImageFont.truetype("DejaVuSans.ttf", 12)
        font_trans = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font_label = ImageFont.load_default()
        font_title = ImageFont.load_default()
        font_trans = ImageFont.load_default()

    clips_rendered = 0
    for local_idx in range(len(all_dm_results)):
        if clips_rendered >= max_clips:
            break
        clip_segments_dm = all_dm_results[local_idx]
        if not clip_segments_dm:
            continue

        clip_id = clip_indices[local_idx]
        top_paths = sorted(clip_segments_dm, key=lambda s: len(s), reverse=True)
        top_paths = [np.array(s, dtype=int) for s in top_paths[:top_k_paths]]
        if not top_paths:
            continue

        unique_segs = _build_unique_subsegments(top_paths)
        n_segs = len(unique_segs)
        if n_segs == 0:
            continue

        gt_segments = clip_labels.get(clip_id, [])

        grid_cols = max(1, int(math.ceil(math.sqrt(n_segs))))
        grid_rows = max(1, int(math.ceil(n_segs / grid_cols)))
        seg_h = max(48, min(80, Hc // grid_rows))
        seg_w = int(seg_h * Wc / Hc)
        gutter = 4
        label_h = 14
        title_h = 20
        right_w = grid_cols * (seg_w + gutter) + 10
        right_panel_h = title_h + grid_rows * (seg_h + label_h + gutter)
        canvas_w = Wc + 10 + right_w
        canvas_h = max(Hc + label_h + 8, right_panel_h)

        out_path = os.path.join(
            output_dir, f"{variant_prefix}_debug_clip{clip_id}.mp4"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with h5py.File(data_path, "r") as f:
            qpos_clip = np.array(f["qpos_rollout"][clip_id])

        clip_len = len(qpos_clip)

        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(mj_model)
        renderer = mujoco.Renderer(mj_model, height=Hc, width=Wc)

        seg_rendered_by_frame: list[list[tuple[int, np.ndarray]]] = []
        seg_sorted_frame_keys: list[list[int]] = []
        seg_renderer = mujoco.Renderer(mj_model, height=seg_h, width=seg_w)
        for useg in unique_segs:
            frame_pairs = []
            for frame_idx in sorted(int(x) for x in useg.frames):
                if frame_idx >= clip_len:
                    continue
                mj_data.qpos[:] = qpos_clip[frame_idx]
                mujoco.mj_forward(mj_model, mj_data)
                seg_renderer.update_scene(mj_data, camera="side")
                frame_pairs.append((frame_idx, seg_renderer.render().copy()))
            seg_rendered_by_frame.append(frame_pairs)
            seg_sorted_frame_keys.append([fp[0] for fp in frame_pairs])
        seg_renderer.close()

        writer = imageio_ffmpeg.write_frames(
            out_path, (canvas_w, canvas_h), fps=framerate,
            codec="h264", macro_block_size=1,
        )
        writer.send(None)

        for fi in tqdm(range(clip_len), desc=f"Clip {clip_id}", leave=False):
            mj_data.qpos[:] = qpos_clip[fi]
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera="side")
            main_frame = renderer.render().copy()

            canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)

            main_img = Image.fromarray(_ensure_rgb_uint8(main_frame))
            canvas.paste(main_img, (0, label_h))

            current_behavior = _get_behavior_at_frame(gt_segments, fi, clip_len)
            if current_behavior and current_behavior != "transition":
                draw.text(
                    (4, 2), current_behavior, font=font_label, fill=(0, 0, 0),
                )

            trans_label = _get_transition_label_at_frame(gt_segments, fi, clip_len)
            if trans_label:
                tw = draw.textbbox((0, 0), trans_label, font=font_trans)[2]
                tx = max(0, (Wc - tw) // 2)
                draw.rectangle(
                    (tx - 4, label_h + 4, tx + tw + 4, label_h + 28),
                    fill=(255, 255, 200, 200),
                )
                draw.text(
                    (tx, label_h + 6), trans_label,
                    font=font_trans, fill=(200, 50, 50),
                )

            progress_y = label_h + Hc + 2
            bar_w = int(Wc * fi / max(clip_len - 1, 1))
            draw.rectangle((0, progress_y, Wc, progress_y + 4), fill=(220, 220, 220))
            draw.rectangle((0, progress_y, bar_w, progress_y + 4), fill=(80, 80, 80))
            for useg in unique_segs:
                sx = int(Wc * useg.key[0] / max(clip_len - 1, 1))
                ex = int(Wc * useg.key[1] / max(clip_len - 1, 1))
                draw.rectangle((sx, progress_y, ex, progress_y + 4),
                               fill=_seg_color(useg.color_idx))

            right_x0 = Wc + 10
            draw.text(
                (right_x0, 2),
                f"{variant_prefix.title()} Sub-Segments ({n_segs})",
                font=font_title, fill=(0, 0, 0),
            )

            for si, useg in enumerate(unique_segs):
                gc = si % grid_cols
                gr = si // grid_cols
                cell_x = right_x0 + gc * (seg_w + gutter)
                cell_y = title_h + gr * (seg_h + label_h + gutter)
                seg_start, seg_end = useg.key
                is_active = seg_start <= fi <= seg_end
                color = _seg_color(useg.color_idx)

                paths_tag = ",".join(str(p) for p in useg.path_indices)
                seg_label = f"P[{paths_tag}]:{seg_start}-{seg_end}"
                draw.text(
                    (cell_x, cell_y), seg_label,
                    font=font_title,
                    fill=color if is_active else (180, 180, 180),
                )

                if seg_rendered_by_frame[si]:
                    keys = seg_sorted_frame_keys[si]
                    idx = bisect.bisect_right(keys, fi) - 1
                    if idx < 0:
                        idx = 0
                    sf = seg_rendered_by_frame[si][idx][1]
                    seg_img = Image.fromarray(_ensure_rgb_uint8(sf))
                    tile_y = cell_y + label_h

                    if is_active:
                        bw = 2
                        draw.rectangle(
                            (cell_x - bw, tile_y - bw,
                             cell_x + seg_w + bw, tile_y + seg_h + bw),
                            outline=color, width=bw,
                        )
                    else:
                        draw.rectangle(
                            (cell_x, tile_y, cell_x + seg_w, tile_y + seg_h),
                            outline=(220, 220, 220), width=1,
                        )

                    canvas.paste(seg_img, (cell_x, tile_y))

            writer.send(np.asarray(canvas, dtype=np.uint8))

        renderer.close()
        writer.close()
        clips_rendered += 1
        print(f"  Saved debug video: {out_path}")

    print(f"Rendered {clips_rendered} {variant_prefix} debug videos")


def _get_behavior_at_frame(
    gt_segments: list[dict], frame: int, clip_length: int,
) -> str | None:
    for seg in gt_segments:
        start = seg["start_frame"]
        end = seg["end_frame"] if seg["end_frame"] >= 0 else clip_length
        if start <= frame < end:
            return seg["behavior"]
    return None


def _get_transition_label_at_frame(
    gt_segments: list[dict], frame: int, clip_length: int,
) -> str | None:
    for i in range(len(gt_segments) - 1):
        end_a = gt_segments[i]["end_frame"]
        if end_a < 0:
            end_a = clip_length
        start_b = gt_segments[i + 1]["start_frame"]
        margin = 15
        if end_a - margin <= frame <= start_b + margin:
            beh_a = gt_segments[i]["behavior"]
            beh_b = gt_segments[i + 1]["behavior"]
            return f"{beh_a} \u2192 {beh_b}"
    return None


_SEG_PALETTE_RGB = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44),
    (214, 39, 40), (148, 103, 189), (140, 86, 75),
    (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
]

_SEG_PALETTE_FLOAT = [
    (r / 255, g / 255, b / 255) for r, g, b in _SEG_PALETTE_RGB
]


def _seg_color(idx: int) -> tuple[int, int, int]:
    return _SEG_PALETTE_RGB[idx % len(_SEG_PALETTE_RGB)]


def _split_contiguous(frames: np.ndarray) -> list[np.ndarray]:
    """Split a sorted array of frame indices into contiguous sub-segments."""
    if len(frames) == 0:
        return []
    sorted_f = np.sort(frames.astype(int))
    breaks = np.where(np.diff(sorted_f) > 1)[0] + 1
    return np.split(sorted_f, breaks)


@dataclass
class UniqueSubsegment:
    key: tuple[int, int]
    frames: np.ndarray
    color_idx: int
    path_indices: list[int]


def _build_unique_subsegments(
    paths: list[np.ndarray],
) -> list[UniqueSubsegment]:
    """Split paths into contiguous sub-segments and deduplicate across paths.

    Sub-segments with the same (start, end) frame range are merged into one
    entry. Each unique sub-segment gets a distinct color index.
    """
    seg_map: dict[tuple[int, int], UniqueSubsegment] = {}
    next_color = 0

    for pi, path in enumerate(paths):
        for subseg in _split_contiguous(np.array(path, dtype=int)):
            key = (int(subseg[0]), int(subseg[-1]))
            if key not in seg_map:
                seg_map[key] = UniqueSubsegment(
                    key=key, frames=subseg, color_idx=next_color, path_indices=[pi],
                )
                next_color += 1
            else:
                if pi not in seg_map[key].path_indices:
                    seg_map[key].path_indices.append(pi)

    result = sorted(seg_map.values(), key=lambda s: s.key[0])
    return result


# ── Degenerate clip rendering ──────────────────────────────────────────────


def render_degenerate_clips(
    all_cycle_dm: list,
    allpairs_path_dm: list | None,
    data_path: str,
    xml_path: str,
    clip_indices: list[int],
    clip_labels: dict[int, list[dict]],
    output_dir: str,
    max_clips: int = 5,
    framerate: int = 30,
    target_size: tuple[int, int] = (320, 240),
):
    """Render clips that have no cycles, no paths, and no allpairs paths."""
    import h5py
    import mujoco
    from tqdm import tqdm

    Wc, Hc = target_size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        font_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        font_big = ImageFont.load_default()

    degenerate = []
    n_clips = len(allpairs_path_dm) if allpairs_path_dm else len(all_cycle_dm)
    for local_idx in range(n_clips):
        has_allpairs = bool(allpairs_path_dm and local_idx < len(allpairs_path_dm) and allpairs_path_dm[local_idx])
        if not has_allpairs:
            degenerate.append(local_idx)

    if not degenerate:
        print("No degenerate clips found (all clips have at least some DM results)")
        return

    degenerate = degenerate[:max_clips]
    print(f"Rendering {len(degenerate)} degenerate clips: "
          f"{[clip_indices[i] for i in degenerate]}")

    os.makedirs(output_dir, exist_ok=True)

    for local_idx in degenerate:
        clip_id = clip_indices[local_idx]
        gt_segments = clip_labels.get(clip_id, [])

        with h5py.File(data_path, "r") as f:
            qpos_clip = np.array(f["qpos_rollout"][clip_id])
        clip_len = len(qpos_clip)

        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(mj_model)
        renderer = mujoco.Renderer(mj_model, height=Hc, width=Wc)

        label_h = 22
        canvas_h = Hc + label_h + 6
        out_path = os.path.join(output_dir, f"degenerate_clip{clip_id}.mp4")

        writer = imageio_ffmpeg.write_frames(
            out_path, (Wc, canvas_h), fps=framerate,
            codec="h264", macro_block_size=1,
        )
        writer.send(None)

        for fi in tqdm(range(clip_len), desc=f"Degen clip {clip_id}", leave=False):
            mj_data.qpos[:] = qpos_clip[fi]
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera="side")
            raw = renderer.render().copy()

            canvas = Image.new("RGB", (Wc, canvas_h), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            canvas.paste(Image.fromarray(_ensure_rgb_uint8(raw)), (0, label_h))

            behavior = _get_behavior_at_frame(gt_segments, fi, clip_len)
            if behavior:
                draw.text((4, 2), behavior, font=font, fill=(0, 0, 0))

            trans_label = _get_transition_label_at_frame(gt_segments, fi, clip_len)
            if trans_label:
                tw = draw.textbbox((0, 0), trans_label, font=font_big)[2]
                tx = max(0, (Wc - tw) // 2)
                draw.rectangle(
                    (tx - 4, label_h + 4, tx + tw + 4, label_h + 26),
                    fill=(255, 255, 200),
                )
                draw.text((tx, label_h + 6), trans_label, font=font_big, fill=(200, 50, 50))

            tag = f"DEGENERATE clip {clip_id} | f{fi}/{clip_len}"
            draw.text((Wc - 200, 2), tag, font=font, fill=(200, 0, 0))

            writer.send(np.asarray(canvas, dtype=np.uint8))

        renderer.close()
        writer.close()
        print(f"  Saved: {out_path}")


# ── Per-clip PCA + DM path debug plots ─────────────────────────────────────


def render_pca_dm_debug(
    data: np.ndarray,
    clip_indices: list[int],
    clip_labels: dict[int, list[dict]],
    all_dm_results: dict[str, list],
    output_dir: str,
):
    """For each clip, save a figure: panel 0 = 2D PCA colored by GT, panels 1..N = each DM path.

    Args:
        data: (n_clips, T, D) feature array.
        clip_indices: global clip IDs.
        clip_labels: GT behavior labels per clip.
        all_dm_results: {"cycle": [...], "allpairs_path": [...]} per-clip segment lists.
        output_dir: folder to save per-clip PNGs.
    """
    from sklearn.decomposition import PCA
    import math

    os.makedirs(output_dir, exist_ok=True)

    for local_idx in range(len(data)):
        clip_id = clip_indices[local_idx]
        clip_data = data[local_idx]
        T, D = clip_data.shape

        all_paths_for_clip: list[np.ndarray] = []
        variant_labels: list[str] = []
        for variant_name, dm_results in all_dm_results.items():
            if dm_results is None or local_idx >= len(dm_results):
                continue
            segs = dm_results[local_idx]
            if segs:
                for s in segs:
                    all_paths_for_clip.append(np.array(s, dtype=int))
                    variant_labels.append(variant_name)

        unique_segs = _build_unique_subsegments(all_paths_for_clip)
        if not unique_segs:
            continue

        pca = PCA(n_components=2)
        coords = pca.fit_transform(clip_data)

        gt_segments = clip_labels.get(clip_id, [])
        frame_labels = build_frame_labels(gt_segments, T)
        color_map = _build_color_map(list(frame_labels))

        n_panels = 1 + len(unique_segs)
        n_cols = min(n_panels, max(4, int(math.ceil(math.sqrt(n_panels)))))
        n_rows = max(1, int(math.ceil(n_panels / n_cols)))

        with plt.rc_context(_NATURE_RC):
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(2.4 * n_cols, 2.4 * n_rows),
                squeeze=False,
            )

            ax0 = axes[0, 0]
            gt_colors = [color_map.get(l, (0.5, 0.5, 0.5, 1.0)) for l in frame_labels]
            ax0.scatter(
                coords[:, 0], coords[:, 1], c=gt_colors, s=3, alpha=0.5,
                edgecolors="none", rasterized=True,
            )
            ev = pca.explained_variance_ratio_
            ax0.set_title(f"GT behaviors (clip {clip_id})")
            ax0.set_xlabel(f"PC1 ({ev[0]:.0%})")
            ax0.set_ylabel(f"PC2 ({ev[1]:.0%})")

            handles = []
            for label, c in color_map.items():
                handles.append(plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=c, markersize=3, label=label))
            ax0.legend(handles=handles, loc="upper right", fontsize=4,
                       framealpha=0.7, handletextpad=0.2, borderpad=0.3)

            for si, useg in enumerate(unique_segs):
                panel_idx = 1 + si
                row = panel_idx // n_cols
                col = panel_idx % n_cols
                if row >= n_rows:
                    break
                ax = axes[row, col]

                ax.scatter(
                    coords[:, 0], coords[:, 1], c=[(0.9, 0.9, 0.9)], s=2,
                    alpha=0.3, edgecolors="none", rasterized=True,
                )

                valid = useg.frames[useg.frames < T]
                if len(valid) > 0:
                    seg_color = _SEG_PALETTE_FLOAT[useg.color_idx % len(_SEG_PALETTE_FLOAT)]
                    ax.scatter(
                        coords[valid, 0], coords[valid, 1],
                        c=[seg_color], s=6, alpha=0.8,
                        edgecolors="none", rasterized=True,
                    )
                    ax.plot(
                        coords[valid, 0], coords[valid, 1],
                        color=seg_color, alpha=0.3, linewidth=0.5,
                    )

                fmin, fmax = (int(valid.min()), int(valid.max())) if len(valid) > 0 else (0, 0)
                gt_label = _segment_transition_pattern(gt_segments, fmin, fmax + 1)
                paths_tag = ",".join(str(p) for p in useg.path_indices)
                ax.set_title(f"P[{paths_tag}] [{fmin}-{fmax}]\n{gt_label}", fontsize=5)
                ax.set_xticks([])
                ax.set_yticks([])

            for idx in range(1 + len(unique_segs), n_rows * n_cols):
                r, c = idx // n_cols, idx % n_cols
                axes[r, c].set_visible(False)

            fig.suptitle(f"Clip {clip_id}: PCA + DM Sub-Segments", fontsize=9, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            out_path = os.path.join(output_dir, f"clip_{clip_id}_pca_dm.png")
            fig.savefig(out_path)
            plt.close(fig)
            print(f"  Saved: {out_path}")

    print(f"PCA+DM debug plots saved to {output_dir}")


# ── Grid rendering ─────────────────────────────────────────────────────────


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
    """Render grid video with each segment as its own cell in a 2D grid.

    Segments are grouped into rows of n_cols each. Each row is a "cluster"
    with multiple samples (columns) for render_grid_clusters_side_by_side.
    """
    import h5py
    import mujoco
    import math
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

    flat_entries: list[tuple[int, int, np.ndarray]] = []
    for local_idx, segments in clips_with_segs:
        clip_id = clip_indices[local_idx]
        for si, seg in enumerate(segments):
            flat_entries.append((clip_id, si, seg))

    n_total = len(flat_entries)
    n_cols = min(n_total, max(3, int(math.ceil(math.sqrt(n_total)))))
    print(f"Rendering segment grid: {n_total} segments in {n_cols} columns")

    with h5py.File(data_path, "r") as f:
        qpos_all = f["qpos_rollout"]

        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(mj_model)
        renderer = mujoco.Renderer(mj_model, height=target_size[1], width=target_size[0])

        rendered_segments: list[tuple[str, list[np.ndarray]]] = []

        for clip_id, seg_idx, seg_frames in tqdm(flat_entries, desc="Rendering segments"):
            qpos_clip = qpos_all[clip_id]
            chunk_frames = []
            for fi in sorted(int(f) for f in seg_frames):
                if fi >= len(qpos_clip):
                    continue
                mj_data.qpos[:] = qpos_clip[fi]
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(mj_data, camera="side")
                chunk_frames.append(renderer.render().copy())

            if chunk_frames:
                label = f"C{clip_id}s{seg_idx}"
                rendered_segments.append((label, chunk_frames))

        renderer.close()

    if not rendered_segments:
        print("WARNING: No frames rendered")
        return

    grid_chunks: dict[int, list[list[np.ndarray]]] = {}
    grid_labels: dict[int, str] = {}
    tile_labels: dict[tuple[int, int], str] = {}

    for i, (label, frames) in enumerate(rendered_segments):
        row = i // n_cols
        col = i % n_cols
        if row not in grid_chunks:
            grid_chunks[row] = []
            grid_labels[row] = ""
        grid_chunks[row].append(frames)
        tile_labels[(row, col)] = label

    from topo_mimic.dmwk.skeletonization.rendering import render_grid_clusters_side_by_side
    render_grid_clusters_side_by_side(
        grid_chunks,
        out_path=output_path,
        framerate=framerate,
        max_samples_per_cluster=n_cols,
        target_size=target_size,
        title="DM Segments",
        cluster_labels=grid_labels,
        tile_labels=tile_labels,
    )
    print(f"Saved segment grid video: {output_path}")


# ── Transition coverage analysis ───────────────────────────────────────────


@dataclass
class GTTransition:
    clip_id: int
    local_idx: int
    src_behavior: str
    dst_behavior: str
    transition_type: str
    src_start: int
    src_end: int
    dst_start: int
    dst_end: int
    gap_start: int
    gap_end: int


def _extract_gt_transitions(
    clip_indices: list[int],
    clip_labels: dict[int, list[dict]],
    clip_length: int,
) -> list[GTTransition]:
    """Extract all GT transitions (consecutive behavior pairs) across clips."""
    transitions: list[GTTransition] = []
    for local_idx, clip_id in enumerate(clip_indices):
        segments = clip_labels.get(clip_id, [])
        if len(segments) < 2:
            continue
        for i in range(len(segments) - 1):
            src = segments[i]
            dst = segments[i + 1]
            src_start = src["start_frame"]
            src_end = src["end_frame"] if src["end_frame"] >= 0 else clip_length
            dst_start = dst["start_frame"]
            dst_end = dst["end_frame"] if dst["end_frame"] >= 0 else clip_length
            pair = tuple(sorted([src["behavior"], dst["behavior"]]))
            transitions.append(GTTransition(
                clip_id=clip_id,
                local_idx=local_idx,
                src_behavior=src["behavior"],
                dst_behavior=dst["behavior"],
                transition_type=f"{pair[0]}\u2194{pair[1]}",
                src_start=src_start,
                src_end=src_end,
                dst_start=dst_start,
                dst_end=min(dst_end, clip_length),
                gap_start=src_end,
                gap_end=dst_start,
            ))
    return transitions


def _check_transition_covered(
    gt: GTTransition,
    dm_segments: list,
    top_k: int,
    clip_length: int,
    clip_labels_for_clip: list[dict],
) -> tuple[bool, float, float]:
    """Check if a GT transition A->B is spatially covered by any DM path.

    A DM path "covers" a transition if it contains frames labeled as both
    the source behavior A and the destination behavior B. This is a spatial
    check — the path visits both behavior regions in the manifold.

    Returns:
        detected: True if any path visits both src and dst behavior regions.
        src_frac: Best fraction of path frames labeled as src behavior.
        dst_frac: Best fraction of path frames labeled as dst behavior.
    """
    if not dm_segments:
        return False, 0.0, 0.0

    frame_labels = build_frame_labels(clip_labels_for_clip, clip_length)

    sorted_paths = sorted(dm_segments, key=lambda s: len(s), reverse=True)
    sorted_paths = [np.array(s, dtype=int) for s in sorted_paths[:top_k]]

    best_score = 0.0
    best_src_frac = 0.0
    best_dst_frac = 0.0

    for path in sorted_paths:
        valid = path[(path >= 0) & (path < clip_length)]
        if len(valid) == 0:
            continue

        path_labels = frame_labels[valid]
        n_src = np.sum(path_labels == gt.src_behavior)
        n_dst = np.sum(path_labels == gt.dst_behavior)
        src_frac = n_src / len(valid)
        dst_frac = n_dst / len(valid)
        score = min(src_frac, dst_frac)

        if score > best_score:
            best_score = score
            best_src_frac = src_frac
            best_dst_frac = dst_frac

    detected = best_src_frac > 0 and best_dst_frac > 0
    return detected, best_src_frac, best_dst_frac


@dataclass
class TransitionCoverageResult:
    gt_transitions: list[GTTransition]
    covered: list[bool]
    src_fracs: list[float]
    dst_fracs: list[float]
    variant_name: str


def compute_transition_coverage(
    gt_transitions: list[GTTransition],
    all_dm_results: list,
    clip_labels: dict[int, list[dict]],
    clip_length: int,
    config: TransitionAnalysisConfig,
    variant_name: str,
) -> TransitionCoverageResult:
    covered = []
    src_fracs = []
    dst_fracs = []
    for gt in gt_transitions:
        segs = all_dm_results[gt.local_idx] if gt.local_idx < len(all_dm_results) else []
        clip_segs = clip_labels.get(gt.clip_id, [])
        is_covered, sf, df = _check_transition_covered(
            gt, segs, config.top_k_paths, clip_length, clip_segs,
        )
        covered.append(is_covered)
        src_fracs.append(sf)
        dst_fracs.append(df)
    return TransitionCoverageResult(
        gt_transitions=gt_transitions,
        covered=covered,
        src_fracs=src_fracs,
        dst_fracs=dst_fracs,
        variant_name=variant_name,
    )


def plot_coverage_matrix(
    results: list[TransitionCoverageResult],
    output_dir: str,
):
    """Plot transition coverage heatmaps: detection rate + mean src/dst coverage fractions."""
    from collections import Counter

    all_types = sorted(set(gt.transition_type for gt in results[0].gt_transitions))
    type_totals = Counter(gt.transition_type for gt in results[0].gt_transitions)

    variant_names = [r.variant_name for r in results]
    n_types = len(all_types)
    n_variants = len(variant_names)

    coverage_rates = np.zeros((n_types, n_variants))
    coverage_counts = np.zeros((n_types, n_variants), dtype=int)
    mean_src_fracs = np.zeros((n_types, n_variants))
    mean_dst_fracs = np.zeros((n_types, n_variants))

    for vi, res in enumerate(results):
        type_covered = Counter()
        type_src_sum: dict[str, float] = defaultdict(float)
        type_dst_sum: dict[str, float] = defaultdict(float)
        for gt, cov, sf, df in zip(res.gt_transitions, res.covered, res.src_fracs, res.dst_fracs):
            if cov:
                type_covered[gt.transition_type] += 1
            type_src_sum[gt.transition_type] += sf
            type_dst_sum[gt.transition_type] += df

        for ti, ttype in enumerate(all_types):
            total = type_totals[ttype]
            coverage_counts[ti, vi] = type_covered[ttype]
            coverage_rates[ti, vi] = type_covered[ttype] / total if total > 0 else 0
            mean_src_fracs[ti, vi] = type_src_sum[ttype] / total if total > 0 else 0
            mean_dst_fracs[ti, vi] = type_dst_sum[ttype] / total if total > 0 else 0

    with plt.rc_context(_NATURE_RC):
        fig, axes = plt.subplots(1, 3, figsize=(10, max(2.5, 0.5 * n_types + 1.5)))
        fig.subplots_adjust(wspace=0.6, left=0.12, right=0.95, top=0.85, bottom=0.15)

        type_labels = [f"{t} (n={type_totals[t]})" for t in all_types]

        for ax_idx, (matrix, title, cmap, fmt) in enumerate([
            (coverage_rates, "Detection Rate\n(path visits both behaviors)", "RdYlGn",
             lambda ti, vi: f"{coverage_counts[ti, vi]}/{type_totals[all_types[ti]]}"),
            (mean_src_fracs, "Mean Source Behavior\nFraction in Path", "YlOrRd",
             lambda ti, vi: f"{mean_src_fracs[ti, vi]:.2f}"),
            (mean_dst_fracs, "Mean Dest Behavior\nFraction in Path", "YlOrRd",
             lambda ti, vi: f"{mean_dst_fracs[ti, vi]:.2f}"),
        ]):
            ax = axes[ax_idx]
            im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(n_variants))
            ax.set_xticklabels(variant_names, rotation=30, ha="right")
            ax.set_yticks(range(n_types))
            ax.set_yticklabels(type_labels if ax_idx == 0 else [""] * n_types)
            ax.set_title(title, fontsize=7)
            for ti in range(n_types):
                for vi in range(n_variants):
                    ax.text(vi, ti, fmt(ti, vi),
                            ha="center", va="center", fontsize=5,
                            color="white" if matrix[ti, vi] < 0.5 else "black")
            fig.colorbar(im, ax=ax, shrink=0.7)

        fig.suptitle("DM Transition Coverage (Spatial)", fontsize=9, fontweight="bold")

        out_path = os.path.join(output_dir, "transition_coverage_matrix.png")
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(out_path, facecolor="white")
        plt.close(fig)
    print(f"Saved coverage matrix: {out_path}")


def plot_detection_curves(
    results: list[TransitionCoverageResult],
    output_dir: str,
):
    """Plot detection curves by sweeping min behavior-fraction threshold.

    Score = min(src_frac, dst_frac) for the best path per transition.
    Sweeping the threshold answers: "if we require at least X% of the path
    to be in each behavior, how many transitions are still detected?"
    """
    from collections import Counter

    thresholds = np.linspace(0.0, 0.5, 51)

    with plt.rc_context(_NATURE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
        fig.subplots_adjust(wspace=0.4, left=0.1, right=0.95, top=0.85, bottom=0.18)

        for res in results:
            n_total = len(res.gt_transitions)
            scores = np.array([min(s, d) for s, d in zip(res.src_fracs, res.dst_fracs)])

            detection_rates = []
            for thr in thresholds:
                n_detected = np.sum(scores >= thr)
                detection_rates.append(n_detected / n_total)

            auc = np.trapz(detection_rates, thresholds)

            axes[0].plot(thresholds, detection_rates,
                         label=f"{res.variant_name} (AUC={auc:.3f})", linewidth=1.2)

            all_types = sorted(set(gt.transition_type for gt in res.gt_transitions))
            type_totals = Counter(gt.transition_type for gt in res.gt_transitions)
            color_map = _build_color_map(all_types)

            for ttype in all_types:
                type_scores = np.array([
                    min(s, d) for gt, s, d in zip(res.gt_transitions, res.src_fracs, res.dst_fracs)
                    if gt.transition_type == ttype
                ])
                total = type_totals[ttype]
                rates = [np.sum(type_scores >= thr) / total for thr in thresholds]
                axes[1].plot(thresholds, rates, linewidth=0.8, alpha=0.7,
                             color=color_map.get(ttype, (0.5, 0.5, 0.5, 1.0)),
                             label=f"{ttype} (n={total})")

        axes[0].set_xlabel("Min Behavior Fraction Threshold")
        axes[0].set_ylabel("Detection Rate")
        axes[0].set_title("Overall: min(src_frac, dst_frac) >= thr")
        axes[0].legend(fontsize=5)
        axes[0].set_xlim(0, 0.5)
        axes[0].set_ylim(0, 1.05)

        axes[1].set_xlabel("Min Behavior Fraction Threshold")
        axes[1].set_ylabel("Detection Rate")
        axes[1].set_title("Per-Transition-Type Curves")
        handles, labels = axes[1].get_legend_handles_labels()
        seen = set()
        unique_h, unique_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                unique_h.append(h)
                unique_l.append(l)
                seen.add(l)
        axes[1].legend(unique_h, unique_l, fontsize=4, loc="lower left")
        axes[1].set_xlim(0, 0.5)
        axes[1].set_ylim(0, 1.05)

        fig.suptitle("DM Spatial Detection Curves", fontsize=9, fontweight="bold")

        out_path = os.path.join(output_dir, "transition_detection_curves.png")
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(out_path, facecolor="white")
        plt.close(fig)
    print(f"Saved detection curves: {out_path}")


def plot_latent_neighborhood_analysis(
    data: np.ndarray,
    gt_transitions: list[GTTransition],
    covered: list[bool],
    src_fracs: list[float],
    dst_fracs: list[float],
    clip_labels: dict[int, list[dict]],
    clip_length: int,
    output_dir: str,
    variant_name: str,
):
    """Analyze latent space properties around each transition to explain coverage.

    For each GT transition, computes:
      - Centroid distance between src and dst behavior neighborhoods
      - Variance within each behavior neighborhood
      - Gap-region density (how many frames fall in the transition gap)
    Then plots these properties stratified by detected vs missed.
    """
    from sklearn.decomposition import PCA

    records: list[dict] = []
    for i, gt in enumerate(gt_transitions):
        clip_data = data[gt.local_idx]
        T = clip_data.shape[0]

        segments = clip_labels.get(gt.clip_id, [])
        src_seg = None
        dst_seg = None
        for s in segments:
            if s["behavior"] == gt.src_behavior and s["end_frame"] == gt.src_end:
                src_seg = s
            if s["behavior"] == gt.dst_behavior and s["start_frame"] == gt.dst_start:
                dst_seg = s

        if src_seg is None or dst_seg is None:
            continue

        src_start = src_seg["start_frame"]
        src_end = min(src_seg["end_frame"] if src_seg["end_frame"] >= 0 else T, T)
        dst_start = dst_seg["start_frame"]
        dst_end = min(dst_seg["end_frame"] if dst_seg["end_frame"] >= 0 else T, T)

        if src_end <= src_start or dst_end <= dst_start:
            continue

        src_feats = clip_data[src_start:src_end]
        dst_feats = clip_data[dst_start:dst_end]

        src_centroid = src_feats.mean(axis=0)
        dst_centroid = dst_feats.mean(axis=0)
        centroid_dist = np.linalg.norm(src_centroid - dst_centroid)

        src_var = np.mean(np.var(src_feats, axis=0))
        dst_var = np.mean(np.var(dst_feats, axis=0))

        gap_start = src_end
        gap_end = min(dst_start, T)
        gap_len = max(gap_end - gap_start, 0)
        if gap_len > 0:
            gap_feats = clip_data[gap_start:gap_end]
            gap_centroid = gap_feats.mean(axis=0)
            gap_to_src = np.linalg.norm(gap_centroid - src_centroid)
            gap_to_dst = np.linalg.norm(gap_centroid - dst_centroid)
        else:
            gap_to_src = 0.0
            gap_to_dst = 0.0

        spatial_score = min(src_fracs[i], dst_fracs[i])
        records.append({
            "transition_type": gt.transition_type,
            "detected": covered[i],
            "spatial_score": spatial_score,
            "src_frac": src_fracs[i],
            "dst_frac": dst_fracs[i],
            "centroid_dist": centroid_dist,
            "src_var": src_var,
            "dst_var": dst_var,
            "gap_len": gap_len,
            "gap_to_src": gap_to_src,
            "gap_to_dst": gap_to_dst,
            "src_behavior": gt.src_behavior,
            "dst_behavior": gt.dst_behavior,
        })

    if not records:
        print(f"  No records for latent neighborhood analysis ({variant_name})")
        return

    detected = [r for r in records if r["detected"]]
    missed = [r for r in records if not r["detected"]]

    metrics = [
        ("centroid_dist", "Centroid Distance (src\u2194dst)"),
        ("src_var", "Source Behavior Variance"),
        ("dst_var", "Dest Behavior Variance"),
        ("gap_len", "Transition Gap (frames)"),
    ]

    with plt.rc_context(_NATURE_RC):
        fig, axes = plt.subplots(2, 3, figsize=(9, 6.5))
        fig.subplots_adjust(hspace=0.55, wspace=0.45, left=0.08, right=0.95,
                            top=0.88, bottom=0.1)

        for mi, (key, label) in enumerate(metrics):
            ax = axes[mi // 3, mi % 3]
            det_vals = [r[key] for r in detected] if detected else []
            mis_vals = [r[key] for r in missed] if missed else []

            parts = []
            labels_bp = []
            positions = []
            if det_vals:
                parts.append(det_vals)
                labels_bp.append(f"Detected\n(n={len(det_vals)})")
                positions.append(0)
            if mis_vals:
                parts.append(mis_vals)
                labels_bp.append(f"Missed\n(n={len(mis_vals)})")
                positions.append(1)

            if parts:
                bp = ax.boxplot(parts, positions=positions, widths=0.5, patch_artist=True)
                colors_bp = [(0.4, 0.76, 0.65), (0.99, 0.55, 0.38)]
                for patch, color in zip(bp["boxes"], colors_bp[:len(parts)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.set_xticks(positions)
                ax.set_xticklabels(labels_bp)
            ax.set_ylabel(label)
            ax.set_title(label, fontsize=6)

        ax_scatter = axes[1, 0]
        for r in records:
            color = (0.4, 0.76, 0.65) if r["detected"] else (0.99, 0.55, 0.38)
            ax_scatter.scatter(r["centroid_dist"], r["spatial_score"],
                               c=[color], s=8, alpha=0.6, edgecolors="none")
        ax_scatter.set_xlabel("Centroid Distance")
        ax_scatter.set_ylabel("Spatial Score\nmin(src_frac, dst_frac)")
        ax_scatter.set_title("Separation vs Spatial Coverage", fontsize=6)

        ax_type = axes[1, 1]
        all_types = sorted(set(r["transition_type"] for r in records))
        type_det_rates = []
        type_mean_dists = []
        for ttype in all_types:
            type_recs = [r for r in records if r["transition_type"] == ttype]
            det_rate = sum(1 for r in type_recs if r["detected"]) / len(type_recs)
            mean_dist = np.mean([r["centroid_dist"] for r in type_recs])
            type_det_rates.append(det_rate)
            type_mean_dists.append(mean_dist)

        color_map = _build_color_map(all_types)
        for ti, ttype in enumerate(all_types):
            ax_type.scatter(type_mean_dists[ti], type_det_rates[ti],
                            c=[color_map.get(ttype, (0.5, 0.5, 0.5, 1.0))],
                            s=40, edgecolors="black", linewidths=0.5, zorder=3)
            ax_type.annotate(ttype.replace("\u2194", "\n\u2194"), (type_mean_dists[ti], type_det_rates[ti]),
                             fontsize=4, ha="center", va="bottom",
                             xytext=(0, 6), textcoords="offset points")
        ax_type.set_xlabel("Mean Centroid Distance")
        ax_type.set_ylabel("Detection Rate")
        ax_type.set_title("Per-Type: Separation vs Detection", fontsize=6)
        ax_type.set_ylim(-0.05, 1.1)

        ax_gap = axes[1, 2]
        for r in records:
            color = (0.4, 0.76, 0.65) if r["detected"] else (0.99, 0.55, 0.38)
            ax_gap.scatter(r["gap_len"], r["centroid_dist"],
                           c=[color], s=8, alpha=0.6, edgecolors="none")
        ax_gap.set_xlabel("Gap Length (frames)")
        ax_gap.set_ylabel("Centroid Distance")
        ax_gap.set_title("Gap Length vs Separation", fontsize=6)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=(0.4, 0.76, 0.65), label=f"Detected ({len(detected)})"),
            Patch(facecolor=(0.99, 0.55, 0.38), label=f"Missed ({len(missed)})"),
        ]
        fig.legend(handles=legend_elements, loc="lower center",
                   bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=6, frameon=False)

        fig.suptitle(f"Latent Neighborhood Analysis ({variant_name})",
                     fontsize=9, fontweight="bold")

        out_path = os.path.join(output_dir, f"latent_neighborhood_{variant_name}.png")
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(out_path, facecolor="white")
        plt.close(fig)
    print(f"Saved latent neighborhood analysis: {out_path}")


def plot_missed_vs_detected_proximity(
    data: np.ndarray,
    gt_transitions: list[GTTransition],
    covered: list[bool],
    clip_labels: dict[int, list[dict]],
    clip_length: int,
    output_dir: str,
    variant_name: str,
):
    """Compare latent locations of missed vs detected transitions.

    For each transition, computes src and dst behavior centroids. Then for
    each missed transition, finds the nearest detected transition of the
    same type and measures how far apart they are in latent space. If
    missed transitions occupy similar regions to detected ones, the DM
    simply failed to find a path despite the structure being there.
    """
    from sklearn.decomposition import PCA

    @dataclass
    class TransitionRecord:
        idx: int
        gt: GTTransition
        detected: bool
        src_centroid: np.ndarray
        dst_centroid: np.ndarray

    records: list[TransitionRecord] = []
    for i, gt in enumerate(gt_transitions):
        clip_data = data[gt.local_idx]
        T = clip_data.shape[0]

        src_start, src_end = gt.src_start, min(gt.src_end, T)
        dst_start, dst_end = gt.dst_start, min(gt.dst_end, T)
        if src_end <= src_start or dst_end <= dst_start:
            continue

        records.append(TransitionRecord(
            idx=i,
            gt=gt,
            detected=covered[i],
            src_centroid=clip_data[src_start:src_end].mean(axis=0),
            dst_centroid=clip_data[dst_start:dst_end].mean(axis=0),
        ))

    if not records:
        return

    detected_recs = [r for r in records if r.detected]
    missed_recs = [r for r in records if not r.detected]

    if not detected_recs and not missed_recs:
        print(f"  Skipping proximity analysis ({variant_name}): no records")
        return

    all_centroids = np.array([
        np.concatenate([r.src_centroid, r.dst_centroid]) for r in records
    ])
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(all_centroids)

    all_types = sorted(set(r.gt.transition_type for r in records))
    color_map = _build_color_map(all_types)

    with plt.rc_context(_NATURE_RC):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        fig.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.22)

        for ri, r in enumerate(records):
            c = color_map.get(r.gt.transition_type, (0.5, 0.5, 0.5, 1.0))
            marker = "o" if r.detected else "X"
            ax.scatter(coords_2d[ri, 0], coords_2d[ri, 1],
                       c=[c], marker=marker,
                       s=15 if r.detected else 25,
                       alpha=0.7, linewidths=0.3, edgecolors="black" if not r.detected else "none")

        ev = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({ev[0]:.0%})")
        ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
        ax.set_title("Transition Centroids (src||dst PCA)", fontsize=8)

        from matplotlib.lines import Line2D
        handles = []
        for ttype in all_types:
            c = color_map.get(ttype, (0.5, 0.5, 0.5, 1.0))
            handles.append(Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=c, markersize=5, label=ttype))
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="gray", markersize=5,
                              markeredgecolor="none", label="Detected"))
        handles.append(Line2D([0], [0], marker="X", color="w",
                              markerfacecolor="gray", markersize=6,
                              markeredgecolor="black", markeredgewidth=0.3,
                              label="Missed"))

        n_det = len(detected_recs)
        n_mis = len(missed_recs)
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, 0.0),
                   ncol=min(len(handles), 4), fontsize=5, frameon=False,
                   title=f"Detected: {n_det}, Missed: {n_mis}", title_fontsize=6)

        out_path = os.path.join(output_dir, f"missed_vs_detected_{variant_name}.png")
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(out_path, facecolor="white")
        plt.close(fig)
    print(f"Saved missed-vs-detected proximity: {out_path}")


def run_transition_analysis(
    data: np.ndarray,
    clip_indices: list[int],
    clip_labels: dict[int, list[dict]],
    clip_length: int,
    all_cycle_dm: list,
    allpairs_path_dm: list | None,
    config: DemoConfig,
):
    ta_cfg = config.transition_analysis
    output_dir = os.path.join(config.output_dir, "transition_analysis")
    os.makedirs(output_dir, exist_ok=True)

    gt_transitions = _extract_gt_transitions(clip_indices, clip_labels, clip_length)
    print(f"Extracted {len(gt_transitions)} GT transitions across "
          f"{len(set(gt.transition_type for gt in gt_transitions))} types")

    coverage_results: list[TransitionCoverageResult] = []

    if allpairs_path_dm is not None:
        ap_cov = compute_transition_coverage(
            gt_transitions, allpairs_path_dm, clip_labels, clip_length, ta_cfg, "allpairs_path",
        )
        coverage_results.append(ap_cov)
        n_det = sum(ap_cov.covered)
        print(f"  AllPairs: {n_det}/{len(gt_transitions)} detected "
              f"({n_det / len(gt_transitions):.1%})")

    plot_coverage_matrix(coverage_results, output_dir)
    plot_detection_curves(coverage_results, output_dir)

    for res in coverage_results:
        plot_latent_neighborhood_analysis(
            data, gt_transitions, res.covered, res.src_fracs, res.dst_fracs,
            clip_labels, clip_length, output_dir, res.variant_name,
        )
        plot_missed_vs_detected_proximity(
            data, gt_transitions, res.covered,
            clip_labels, clip_length, output_dir, res.variant_name,
        )


# ── Main pipeline ───────────────────────────────────────────────────────────


def run_demo(config: DemoConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("Walker Path Analysis Demo")
    print("=" * 60)
    print(f"Dataset: {config.dataset}")
    print(f"Data: {config.data_path}")
    print(f"Labels: {config.labels_csv}")
    print(f"Output: {config.output_dir}")

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

    # --- Per-frame and windowed behavior embeddings ---
    print("\n=== Per-Frame & Windowed Embeddings ===")
    run_perframe_visualization(data, clip_labels, clip_indices, config)
    run_windowed_visualization(data, clip_labels, clip_indices, config)


    # --- Stage 2: DM extraction ---
    dm_output = os.path.join(config.output_dir, "dm")
    cycles_path = os.path.join(dm_output, "all_cycle_dm.pkl")
    graphs_path = os.path.join(dm_output, "all_G_sm.pkl")

    print("\n=== Stage 2: DM extraction ===")
    if os.path.exists(cycles_path):
        print(f"Found cached DM results at {cycles_path}, skipping extraction")
        with open(cycles_path, "rb") as f:
            all_cycle_dm = pkl.load(f)
        all_G_sm = None
        if os.path.exists(graphs_path):
            with open(graphs_path, "rb") as f:
                all_G_sm = pkl.load(f)
    else:
        all_cycle_dm, all_G_sm = run_dm_on_data(data, clip_indices, config)

    # --- All-pairs path extraction ---
    allpairs_path_dm = None
    allpairs_cache = os.path.join(dm_output, "all_allpairs_path_dm.pkl")
    if all_G_sm is not None:
        if os.path.exists(allpairs_cache):
            print(f"Found cached all-pairs paths at {allpairs_cache}")
            with open(allpairs_cache, "rb") as f:
                allpairs_path_dm = pkl.load(f)
        else:
            print("\n=== All-Pairs Path Extraction ===")
            allpairs_path_dm = run_allpairs_path_extraction(
                all_G_sm, clip_indices, config,
            )
            with open(allpairs_cache, "wb") as f:
                pkl.dump(allpairs_path_dm, f)
            print(f"Cached all-pairs paths to {allpairs_cache}")

    print("\n=== DM Segment Per-Frame Embeddings ===")
    if allpairs_path_dm is not None:
        run_dm_segment_perframe_embedding(
            data, clip_indices, clip_labels, clip_length, config,
            all_dm_segments=allpairs_path_dm,
            top_k=config.dm_pipeline.top_k_paths,
            variant_name="allpairs_path",
        )

    for variant_name, dm_results, prefix in [
        ("AllPairsPath", allpairs_path_dm, "allpairs_path"),
    ]:
        if dm_results is None:
            continue

        print(f"\n=== Stage 3: {variant_name} GT-labeled DM segment embeddings ===")

        windows = []
        for local_idx in range(len(data)):
            if local_idx >= len(dm_results):
                continue
            for seg in dm_results[local_idx]:
                if len(seg) >= 10:
                    windows.append((local_idx, np.array(seg, dtype=int)))
        print(f"Prepared {len(windows)} {variant_name.lower()} segments")

        variant_dir = os.path.join(config.output_dir, prefix)
        os.makedirs(variant_dir, exist_ok=True)

        if windows:
            run_dm_perframe_embedding(
                windows, data, clip_indices, clip_labels,
                clip_length, config, prefix,
                output_dir=variant_dir,
            )

        # Grid rendering (rows=clips, cols=segments)
        if config.rendering.enabled and dm_results:
            print(f"\n=== {variant_name} Grid Rendering ===")
            grid_path = os.path.join(variant_dir, "segment_grid.mp4")
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

            print(f"\n=== {variant_name} Debug Side-by-Side ===")
            try:
                render_clip_debug_side_by_side(
                    dm_results,
                    config.data_path,
                    config.rendering.walker_xml,
                    clip_indices,
                    clip_labels,
                    output_dir=variant_dir,
                    variant_prefix=prefix,
                    max_clips=config.rendering.max_debug_clips,
                    top_k_paths=config.rendering.top_k_debug_paths,
                    framerate=config.rendering.framerate,
                    target_size=config.rendering.target_size,
                )
            except Exception as e:
                print(f"WARNING: {variant_name} debug rendering failed: {e}")
                import traceback
                traceback.print_exc()

    # --- PCA + DM debug plots (separate for cycle and path) ---
    if config.rendering.pca_debug and allpairs_path_dm is not None:
        print("\n=== PCA + AllPairs Path Debug Plots ===")
        try:
            render_pca_dm_debug(
                data, clip_indices, clip_labels,
                all_dm_results={"allpairs_path": allpairs_path_dm},
                output_dir=os.path.join(config.output_dir, "pca_allpairs_debug"),
            )
        except Exception as e:
            print(f"WARNING: PCA+allpairs debug plots failed: {e}")
            import traceback
            traceback.print_exc()

    # --- Transition coverage analysis ---
    if config.transition_analysis.enabled:
        print("\n=== Transition Coverage Analysis ===")
        try:
            run_transition_analysis(
                data, clip_indices, clip_labels, clip_length,
                all_cycle_dm, allpairs_path_dm, config,
            )
        except Exception as e:
            print(f"WARNING: Transition analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # --- Degenerate clip rendering ---
    if config.rendering.enabled:
        print("\n=== Degenerate Clip Rendering ===")
        degen_dir = os.path.join(config.output_dir, "degenerate")
        try:
            render_degenerate_clips(
                all_cycle_dm, allpairs_path_dm,
                config.data_path,
                config.rendering.walker_xml,
                clip_indices, clip_labels,
                output_dir=degen_dir,
                max_clips=5,
                framerate=config.rendering.framerate,
                target_size=config.rendering.target_size,
            )
        except Exception as e:
            print(f"WARNING: Degenerate clip rendering failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Walker Path Analysis Demo Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Walker path analysis demo")
    parser.add_argument(
        "--config",
        default="configs/demos/transition_demos/demo_walker_transitions_path.yaml",
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
