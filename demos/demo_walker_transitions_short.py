#!/usr/bin/env python3
"""
Walker Transition Demo (Short Clips)
======================================
End-to-end demo for SHORT transition clips (one transition per clip):
DM extraction -> W2 clustering -> GT comparison -> PCA plots.

Each clip is treated as a single unit; windows are stacked feature vectors.

Usage:
    python demos/demo_walker_transitions_short.py
    python demos/demo_walker_transitions_short.py --config configs/demos/demo_walker_transitions_short.yaml
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
    run_pipeline,
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


@dataclass
class DemoDMConfig:
    dm_key: str = "intention"
    split: str = "all"
    takens: dict = field(default_factory=lambda: {"auto_estimate": True})
    dm: dict = field(default_factory=lambda: {"pers_delta": 1.0, "k_neighbors": 3})


@dataclass
class DemoClusteringConfig:
    n_clusters: int = 10
    target_len: int = 70
    skip_clips: tuple[int, ...] = ()
    min_len: int = 0
    max_len: int = 500


@dataclass
class DemoRenderingConfig:
    enabled: bool = True
    walker_xml: str = "topo_mimic/dmwk/xml/walker.xml"
    max_samples: int = 5
    max_gt_types: int | None = None
    max_gt_samples: int | None = None
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)


@dataclass
class DemoConfig:
    dataset: str = "walker"
    data_path: str = "data/walker.h5"
    labels_csv: str = "walker_behavior_labels.csv"
    output_dir: str = "outputs/walker_demo"
    dm_pipeline: DemoDMConfig = field(default_factory=DemoDMConfig)
    clustering: DemoClusteringConfig = field(default_factory=DemoClusteringConfig)
    rendering: DemoRenderingConfig = field(default_factory=DemoRenderingConfig)
    only_analysis: bool = False


def load_demo_config(config_path: str) -> DemoConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dm_dict = cfg.get("dm_pipeline", {})
    dm_cfg = DemoDMConfig(
        dm_key=dm_dict.get("dm_key", "intention"),
        split=dm_dict.get("split", "all"),
        takens=dm_dict.get("takens", {"auto_estimate": True}),
        dm=dm_dict.get("dm", {"pers_delta": 1.0, "k_neighbors": 3}),
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
        dataset=cfg.get("dataset", "walker"),
        data_path=cfg.get("data_path", "data/walker.h5"),
        labels_csv=cfg.get("labels_csv", "walker_behavior_labels.csv"),
        output_dir=cfg.get("output_dir", "outputs/walker_demo"),
        dm_pipeline=dm_cfg,
        clustering=cl_cfg,
        rendering=rend_cfg,
        only_analysis=cfg.get("only_analysis", False),
    )


def build_gt_cluster_dict(
    labels_csv: str,
    clip_indices: list[int],
) -> dict[str, list[tuple[int, np.ndarray]]]:
    df = pd.read_csv(labels_csv)
    clip_set = set(clip_indices)
    gt_groups: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)

    clips_in_df = sorted(df["Clip"].unique())
    for clip_id in clips_in_df:
        if clip_id not in clip_set:
            continue
        clip_rows = df[df["Clip"] == clip_id].sort_values("Start_Frame")
        if len(clip_rows) < 2:
            continue
        behaviors = list(clip_rows["Behavior"])
        transition_key = f"{behaviors[0]} -> {behaviors[1]}"

        start_frame = int(clip_rows.iloc[0]["End_Frame"])
        end_frame = int(clip_rows.iloc[1]["Start_Frame"])
        if end_frame < 0:
            end_frame = 499
        mid = (start_frame + end_frame) // 2
        window_start = max(0, mid - 35)
        window_end = min(499, mid + 35)
        frames = np.arange(window_start, window_end)

        gt_groups[transition_key].append((clip_id, frames))

    return dict(gt_groups)


def filter_gt_cluster_dict(
    gt_cluster_dict: dict[str, list[tuple[int, np.ndarray]]],
    max_types: int | None = None,
    max_samples: int | None = None,
) -> dict[str, list[tuple[int, np.ndarray]]]:
    sorted_keys = sorted(gt_cluster_dict.keys(), key=lambda k: len(gt_cluster_dict[k]), reverse=True)
    if max_types is not None:
        sorted_keys = sorted_keys[:max_types]
    filtered = {}
    for k in sorted_keys:
        items = gt_cluster_dict[k]
        if max_samples is not None:
            items = items[:max_samples]
        filtered[k] = items
    return filtered


def build_gt_pca_windows(
    gt_cluster_dict: dict[str, list[tuple[int, np.ndarray]]],
    clip_indices: list[int],
    target_len: int,
) -> tuple[list[tuple[int, np.ndarray]], list[str]]:
    abs_to_local = {abs_id: i for i, abs_id in enumerate(clip_indices)}
    all_gt_windows: list[tuple[int, np.ndarray]] = []
    gt_labels: list[str] = []

    for transition_type, items in sorted(gt_cluster_dict.items()):
        for clip_id, frames in items:
            local_idx = abs_to_local.get(clip_id)
            if local_idx is None:
                continue
            resampled = _uniform_downsample(frames, target_len)
            if resampled is None:
                continue
            all_gt_windows.append((local_idx, resampled))
            gt_labels.append(transition_type)

    return all_gt_windows, gt_labels


def _build_dm_cluster_map(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
) -> dict[int, int]:
    dm_map: dict[int, int] = {}
    for cluster_label, items in cluster_dict.items():
        for clip_id, _ in items:
            dm_map[clip_id] = cluster_label
    return dm_map


def build_dm_pipeline_config(
    demo_cfg: DemoConfig,
) -> PipelineConfig:
    dm_output = os.path.join(demo_cfg.output_dir, "dm")

    takens_dict = dict(demo_cfg.dm_pipeline.takens)
    theiler_val = takens_dict.get("theiler", "auto")
    if theiler_val == "auto":
        pass
    elif theiler_val is not None:
        theiler_val = int(theiler_val)
    takens_dict["theiler"] = theiler_val

    dm_dict = dict(demo_cfg.dm_pipeline.dm)

    return PipelineConfig(
        output_dir=dm_output,
        data=DMDataConfig(
            dataset=demo_cfg.dataset,
            dm_key=demo_cfg.dm_pipeline.dm_key,
            rodent_rollout_dir=demo_cfg.data_path,
            split=demo_cfg.dm_pipeline.split,
        ),
        takens=TakensConfig(**takens_dict),
        dm=DMExtractConfig(use_dm_path=True, **dm_dict),
        rendering=DMRenderingConfig(enabled=False),
    )


def run_pipeline_on_data(
    data: np.ndarray,
    clip_indices: list[int],
    config: PipelineConfig,
    output_dir: str,
) -> tuple[list, list, list | None]:
    import pickle as pkl
    import yaml as _yaml
    from topo_mimic.dmwk.skeletonization.preprocess import estimate_delay_embedding_params
    from topo_mimic.dmwk.utils.feature_transforms import get_hdf5_key_for_dm_key
    from topo_mimic.dmwk.utils.load_utils import get_aggregate_data
    import functools

    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")
    os.makedirs(output_dir, exist_ok=True)
    config.output_dir = output_dir

    print(f"Running DM on {data.shape[0]} clips (pre-loaded)")

    if config.takens.auto_estimate:
        print("Auto-estimating Takens parameters (AMI + FNN)...")
        hdf5_key = get_hdf5_key_for_dm_key(config.data.dm_key)
        tau_star, m_star, diagnostics = estimate_delay_embedding_params(
            rodent_rollout_dir=config.data.rodent_rollout_dir,
            get_aggregate_data_fn=get_aggregate_data,
            num_clips=len(clip_indices),
            intention_key=f"/{hdf5_key}",
            clip_indices=clip_indices,
            transform_fn=None,
            max_tau=config.takens.max_tau,
            bins=config.takens.bins,
            m_max=config.takens.m_max,
            R_tol=config.takens.R_tol,
            A_tol=config.takens.A_tol,
            fnn_thr=config.takens.fnn_thr,
            theiler=config.takens.theiler,
            max_points=config.takens.max_points,
            plot=False,
        )
        config.takens.tau = tau_star
        config.takens.m = m_star
        print(f"Estimated: tau* = {tau_star}, m* = {m_star}")

        diag_path = os.path.join(output_dir, "ami_fnn_diagnostics.pkl")
        with open(diag_path, "wb") as f:
            pkl.dump({"tau_star": tau_star, "m_star": m_star, "diagnostics": diagnostics}, f)

        plot_path = os.path.join(output_dir, "ami_fnn_diagnostics.png")
        plot_ami_fnn_diagnostics(diagnostics, tau_star, m_star, plot_path, fnn_thr=config.takens.fnn_thr)
    else:
        print(f"Using fixed Takens parameters (m={config.takens.m}, tau={config.takens.tau})")

    print("Running DM extraction...")
    all_cycle_dm, all_G_sm, all_path_dm = run_dm_extraction(data, config)

    stats = get_cycle_statistics(all_cycle_dm)
    print("\nCycle Statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val:.2f}" if isinstance(val, float) else f"  {key}: {val}")

    stats_path = os.path.join(output_dir, "cycle_stats.yaml")
    with open(stats_path, "w") as f:
        _yaml.dump(stats, f)

    if all_path_dm is not None:
        path_stats = get_path_statistics(all_path_dm)
        print("\nPath Statistics:")
        for key, val in path_stats.items():
            print(f"  {key}: {val:.2f}" if isinstance(val, float) else f"  {key}: {val}")

        path_stats_path = os.path.join(output_dir, "path_stats.yaml")
        with open(path_stats_path, "w") as f:
            _yaml.dump(path_stats, f)

    return all_cycle_dm, all_G_sm, all_path_dm


def run_clustering(
    all_cycle_dm: list,
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
        all_cycle_dm, data, exp_clustering, clip_indices
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


def render_walker_frame(
    qpos: np.ndarray,
    xml_path: str,
    mj_model=None,
    mj_data=None,
    renderer=None,
) -> np.ndarray:
    import mujoco

    if mj_model is None:
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(mj_model)
        renderer = mujoco.Renderer(mj_model, height=240, width=320)

    mj_data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, mj_data)
    renderer.update_scene(mj_data, camera="side")
    return renderer.render().copy(), mj_model, mj_data, renderer


def chunk_loops_walker(
    cluster_dict: dict,
    data_path: str,
    xml_path: str,
    max_samples: int = 5,
    pad: int = 0,
) -> dict[int | str, list[list[np.ndarray]]]:
    import h5py
    import mujoco
    from tqdm import tqdm

    with h5py.File(data_path, "r") as f:
        qpos_all = f["qpos_rollout"]

        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(mj_model)
        renderer = mujoco.Renderer(mj_model, height=240, width=320)

        cluster_chunks = {}
        for cid, items in tqdm(cluster_dict.items(), desc="Rendering clusters"):
            sampled = items[:max_samples] if len(items) > max_samples else items
            cluster_chunks[cid] = []

            for clip_id, frames in sampled:
                qpos_clip = qpos_all[clip_id]
                chunk_frames = []
                for fi in sorted(int(f) for f in frames):
                    if fi >= len(qpos_clip):
                        continue
                    mj_data.qpos[:] = qpos_clip[fi]
                    mujoco.mj_forward(mj_model, mj_data)
                    renderer.update_scene(mj_data, camera="side")
                    chunk_frames.append(renderer.render().copy())
                if chunk_frames:
                    cluster_chunks[cid].append(chunk_frames)

        renderer.close()

    return cluster_chunks


def _split_transition(label: str) -> tuple[str, str]:
    if " -> " in label:
        s, d = label.split(" -> ", 1)
        return s, d
    return label, label


def _prepare_pca_data(
    gt_windows: list[tuple[int, np.ndarray]],
    gt_labels: list[str],
    clip_indices: list[int],
    dm_cluster_map: dict[int, int],
    cluster_labels: dict[int, str] | None = None,
) -> dict:
    src_labels = []
    dst_labels = []
    for gt in gt_labels:
        s, d = _split_transition(gt)
        src_labels.append(s)
        dst_labels.append(d)

    dm_assignments = []
    for local_idx, _ in gt_windows:
        abs_clip_id = clip_indices[local_idx]
        dm_assignments.append(dm_cluster_map.get(abs_clip_id, -1))

    unique_dm = sorted(set(a for a in dm_assignments if a >= 0))
    if cluster_labels:
        label_names = {l: cluster_labels.get(l, f"Cluster {l}") for l in unique_dm}
    else:
        label_names = {l: f"Cluster {l}" for l in unique_dm}

    result = dict(
        gt_labels=gt_labels,
        src_labels=src_labels,
        dst_labels=dst_labels,
        dm_assignments=dm_assignments,
        label_names=label_names,
    )

    if cluster_labels:
        filtered_tag = "Filtered"
        disc_labels = []
        disc_src = []
        disc_dst = []
        for a in dm_assignments:
            if a < 0:
                disc_labels.append(filtered_tag)
                disc_src.append(filtered_tag)
                disc_dst.append(filtered_tag)
            else:
                name = cluster_labels.get(a, f"Cluster {a}")
                disc_labels.append(name)
                s, d = _split_transition(name)
                disc_src.append(s)
                disc_dst.append(d)
        result["disc_labels"] = disc_labels
        result["disc_src_labels"] = disc_src
        result["disc_dst_labels"] = disc_dst

    return result


def _scatter_categorical(ax, coords, cat_labels, cmap_name, title, axis_labels, legend_kw=None, color_map=None):
    unique = sorted(set(cat_labels))
    if color_map is None:
        color_map = _build_unified_color_map(cat_labels, cmap_name=cmap_name)
    colors = [color_map.get(l, (0.5, 0.5, 0.5, 1.0)) for l in cat_labels]

    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=10, alpha=0.7, edgecolors="white", linewidths=0.15, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    if len(unique) <= 20:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color_map.get(l, (0.5, 0.5, 0.5, 1.0)),
                       markersize=6, label=l)
            for l in unique
        ]
        kw = dict(fontsize=6, loc="best", ncol=2 if len(unique) > 8 else 1)
        if legend_kw:
            kw.update(legend_kw)
        ax.legend(handles=handles, **kw)


def _scatter_with_filtered(ax, coords, cat_labels, cmap_name, title, axis_labels, filtered_tag="Filtered", legend_kw=None, color_map=None):
    scatter_kw = dict(s=10, alpha=0.7, edgecolors="white", linewidths=0.15, rasterized=True)

    filtered_mask = np.array([l == filtered_tag for l in cat_labels])
    active_mask = ~filtered_mask

    if filtered_mask.any():
        fc = coords[filtered_mask]
        ax.scatter(fc[:, 0], fc[:, 1], c="#AAAAAA", **scatter_kw)

    active_labels = [l for l, m in zip(cat_labels, active_mask) if m]
    if active_labels:
        unique = sorted(set(active_labels))
        if color_map is None:
            color_map = _build_unified_color_map(active_labels, cmap_name=cmap_name)
        colors = [color_map.get(l, (0.5, 0.5, 0.5, 1.0)) for l in active_labels]
        ac = coords[active_mask]
        ax.scatter(ac[:, 0], ac[:, 1], c=colors, **scatter_kw)
    else:
        unique = []
        color_map = color_map or {}

    n_filt = int(filtered_mask.sum())
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map.get(l, (0.5, 0.5, 0.5, 1.0)),
                   markersize=6, label=l)
        for l in unique
    ] if active_labels else []
    if filtered_mask.any():
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#AAAAAA",
                       markersize=6, label=f"Filtered ({n_filt})")
        )
    if handles and len(handles) <= 20:
        kw = dict(fontsize=6, loc="best", ncol=2 if len(handles) > 8 else 1)
        if legend_kw:
            kw.update(legend_kw)
        ax.legend(handles=handles, **kw)


def _scatter_cluster_labels(ax, coords, dm_assignments, label_names, axis_labels, legend_kw=None):
    dm_arr = np.array(dm_assignments)
    filtered_mask = dm_arr == -1
    clustered_mask = ~filtered_mask

    scatter_kw = dict(s=10, alpha=0.7, edgecolors="white", linewidths=0.15, rasterized=True)

    if filtered_mask.any():
        fc = coords[filtered_mask]
        ax.scatter(fc[:, 0], fc[:, 1], c="#AAAAAA", **scatter_kw)

    if clustered_mask.any():
        cc = coords[clustered_mask]
        cl = dm_arr[clustered_mask]
        unique_labels = sorted(set(cl))
        n = max(len(unique_labels) - 1, 1)
        cmap = plt.cm.tab10
        colors = [cmap(l / n) for l in cl]
        ax.scatter(cc[:, 0], cc[:, 1], c=colors, **scatter_kw)
    else:
        unique_labels = []
        n = 1
        cmap = plt.cm.tab10

    n_filt = int(filtered_mask.sum())
    n_clust = int(clustered_mask.sum())
    ax.set_title(f"DM+W2 Clusters ({n_clust} clustered, {n_filt} filtered)")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cmap(l / n),
                   markersize=6, label=label_names.get(l, f"Cluster {l}"))
        for l in unique_labels
    ]
    if filtered_mask.any():
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#AAAAAA",
                       markersize=6, label=f"Filtered ({n_filt})")
        )
    kw = dict(fontsize=6, loc="best", ncol=2 if len(handles) > 8 else 1)
    if legend_kw:
        kw.update(legend_kw)
    ax.legend(handles=handles, **kw)


_TAB10_NO_GRAY = [plt.cm.tab10(i) for i in range(10) if i != 7]
_TAB20_NO_GRAY = [plt.cm.tab20(i) for i in range(20) if i not in (14, 15)]


def _build_unified_color_map(
    *label_lists: list[str],
    cmap_name: str = "tab10",
    filtered_tag: str = "Filtered",
) -> dict[str, tuple]:
    palette = _TAB20_NO_GRAY if "20" in cmap_name else _TAB10_NO_GRAY
    all_labels = set()
    for labels in label_lists:
        all_labels.update(l for l in labels if l != filtered_tag)
    return {label: palette[i % len(palette)] for i, label in enumerate(sorted(all_labels))}


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



def _plot_embedding_grid(
    coords: np.ndarray,
    info: dict,
    n_win: int,
    axis_labels: list[str],
    suptitle: str,
    output_path: str,
) -> None:
    with plt.rc_context(_NATURE_RC):
        if "disc_labels" in info:
            trans_cmap = _build_unified_color_map(info["gt_labels"], info["disc_labels"], cmap_name="tab20")
            src_cmap = _build_unified_color_map(info["src_labels"], info["disc_src_labels"], cmap_name="tab10")
            dst_cmap = _build_unified_color_map(info["dst_labels"], info["disc_dst_labels"], cmap_name="tab10")

            fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
            fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.08, wspace=0.3, hspace=0.4)
            _scatter_categorical(axes[0, 0], coords, info["gt_labels"], "tab20", f"GT Transition Type (n={n_win})", axis_labels, color_map=trans_cmap)
            _scatter_categorical(axes[0, 1], coords, info["src_labels"], "tab10", "GT Source Behavior", axis_labels, color_map=src_cmap)
            _scatter_categorical(axes[0, 2], coords, info["dst_labels"], "tab10", "GT Destination Behavior", axis_labels, color_map=dst_cmap)
            _scatter_with_filtered(axes[1, 0], coords, info["disc_labels"], "tab20", "Discovered Transition Type", axis_labels, color_map=trans_cmap)
            _scatter_with_filtered(axes[1, 1], coords, info["disc_src_labels"], "tab10", "Discovered Source Behavior", axis_labels, color_map=src_cmap)
            _scatter_with_filtered(axes[1, 2], coords, info["disc_dst_labels"], "tab10", "Discovered Destination Behavior", axis_labels, color_map=dst_cmap)
        else:
            fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4))
            fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.15, wspace=0.35)
            _scatter_categorical(axes[0], coords, info["gt_labels"], "tab20", f"Transition Type (n={n_win})", axis_labels)
            _scatter_categorical(axes[1], coords, info["src_labels"], "tab10", "Source Behavior", axis_labels)
            _scatter_categorical(axes[2], coords, info["dst_labels"], "tab10", "Destination Behavior", axis_labels)
            _scatter_cluster_labels(axes[3], coords, info["dm_assignments"], info["label_names"], axis_labels)

        fig.suptitle(suptitle, fontsize=9, fontweight="bold")
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        fig.savefig(output_path, facecolor="white")
        plt.close(fig)
    print(f"Saved {suptitle} to {output_path}")


def plot_pca_comparison(
    gt_windows: list[tuple[int, np.ndarray]],
    gt_labels: list[str],
    data: np.ndarray,
    clip_indices: list[int],
    dm_cluster_map: dict[int, int],
    output_path: str,
    cluster_labels: dict[int, str] | None = None,
) -> None:
    from sklearn.decomposition import PCA

    features = build_feature_matrix(gt_windows, data)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(features)

    info = _prepare_pca_data(gt_windows, gt_labels, clip_indices, dm_cluster_map, cluster_labels)
    axis_labels = [f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"]
    _plot_embedding_grid(coords, info, len(gt_windows), axis_labels, "PCA of Transition Windows", output_path)


def plot_umap_comparison(
    gt_windows: list[tuple[int, np.ndarray]],
    gt_labels: list[str],
    data: np.ndarray,
    clip_indices: list[int],
    dm_cluster_map: dict[int, int],
    output_path: str,
    cluster_labels: dict[int, str] | None = None,
) -> None:
    from sklearn.decomposition import PCA
    from umap import UMAP

    features = build_feature_matrix(gt_windows, data)
    pca_pre = PCA(n_components=min(50, features.shape[1]))
    features_pca = pca_pre.fit_transform(features)
    coords = UMAP(n_components=2, random_state=42).fit_transform(features_pca)

    info = _prepare_pca_data(gt_windows, gt_labels, clip_indices, dm_cluster_map, cluster_labels)
    axis_labels = ["UMAP-1", "UMAP-2"]
    _plot_embedding_grid(coords, info, len(gt_windows), axis_labels, "UMAP of Transition Windows", output_path)


def plot_tsne_comparison(
    gt_windows: list[tuple[int, np.ndarray]],
    gt_labels: list[str],
    data: np.ndarray,
    clip_indices: list[int],
    dm_cluster_map: dict[int, int],
    output_path: str,
    cluster_labels: dict[int, str] | None = None,
) -> None:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    features = build_feature_matrix(gt_windows, data)
    pca_pre = PCA(n_components=min(50, features.shape[1]))
    features_pca = pca_pre.fit_transform(features)
    coords = TSNE(n_components=2, random_state=42, perplexity=min(30, len(gt_windows) - 1)).fit_transform(features_pca)

    info = _prepare_pca_data(gt_windows, gt_labels, clip_indices, dm_cluster_map, cluster_labels)
    axis_labels = ["t-SNE-1", "t-SNE-2"]
    _plot_embedding_grid(coords, info, len(gt_windows), axis_labels, "t-SNE of Transition Windows", output_path)


def load_labels_yaml(yaml_path: str) -> dict[int, str]:
    with open(yaml_path, "r") as f:
        content = yaml.safe_load(f)
    labels = content.get("labels", {})
    return {int(k): v for k, v in labels.items() if v}


def print_confusion_table(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    gt_cluster_dict: dict[str, list[tuple[int, np.ndarray]]],
    cluster_labels: dict[int, str] | None = None,
) -> None:
    clip_to_gt: dict[int, str] = {}
    for gt_label, items in gt_cluster_dict.items():
        for clip_id, _ in items:
            clip_to_gt[clip_id] = gt_label

    gt_types = sorted(set(clip_to_gt.values()))
    cluster_ids = sorted(cluster_dict.keys())

    matrix: dict[int, dict[str, int]] = {}
    for cid in cluster_ids:
        matrix[cid] = defaultdict(int)
        for clip_id, _ in cluster_dict[cid]:
            gt = clip_to_gt.get(clip_id, "unknown")
            matrix[cid][gt] += 1

    print("\n" + "=" * 60)
    print("Cluster vs GT Transition Type Confusion Matrix")
    print("=" * 60)

    header_parts = ["Cluster".ljust(25)]
    for gt in gt_types:
        short = gt[:15]
        header_parts.append(short.rjust(16))
    print(" | ".join(header_parts))
    print("-" * (26 + 19 * len(gt_types)))

    for cid in cluster_ids:
        name = cluster_labels.get(cid, f"C{cid}") if cluster_labels else f"C{cid}"
        row_parts = [name.ljust(25)]
        for gt in gt_types:
            count = matrix[cid].get(gt, 0)
            row_parts.append(str(count).rjust(16))
        print(" | ".join(row_parts))


def run_demo(config: DemoConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    preset = get_dataset_preset(config.dataset)

    print("=" * 60)
    print("Walker Transition Demo")
    print("=" * 60)
    print(f"Dataset: {config.dataset}")
    print(f"Data: {config.data_path}")
    print(f"Output: {config.output_dir}")
    print(f"Only-analysis: {config.only_analysis}")

    # --- Stage 1: Filter GT first, then load only needed clips ---
    print("\n=== Stage 1: Building GT pool & loading data ===")
    import functools
    import multiprocessing as mp
    from tqdm import tqdm
    from topo_mimic.dmwk.utils.load_utils import get_aggregate_data, get_clip_indices
    from topo_mimic.dmwk.utils.feature_transforms import get_hdf5_key_for_dm_key

    all_clip_indices = get_clip_indices(
        config.dm_pipeline.split,
        total_clips=preset.total_clips,
        test_size=preset.test_size,
    )

    gt_cluster_dict_full = build_gt_cluster_dict(config.labels_csv, all_clip_indices)
    total_gt_clips = sum(len(v) for v in gt_cluster_dict_full.values())
    print(f"GT transition types: {len(gt_cluster_dict_full)} total ({total_gt_clips} clips)")
    for gt_label, items in sorted(gt_cluster_dict_full.items()):
        print(f"  {gt_label}: {len(items)} clips")

    gt_cluster_dict = filter_gt_cluster_dict(
        gt_cluster_dict_full,
        max_types=config.rendering.max_gt_types,
        max_samples=config.rendering.max_gt_samples,
    )

    pool_clip_ids = set()
    for items in gt_cluster_dict.values():
        for clip_id, _ in items:
            pool_clip_ids.add(clip_id)
    clip_indices = sorted(pool_clip_ids)

    filtered_gt_clips = sum(len(v) for v in gt_cluster_dict.values())
    print(f"Filtered to {len(gt_cluster_dict)} GT types, {filtered_gt_clips} clips -> {len(clip_indices)} unique clips to process")

    hdf5_key = get_hdf5_key_for_dm_key(config.dm_pipeline.dm_key)
    work_fn = functools.partial(
        get_aggregate_data, f"/{hdf5_key}", [], path=config.data_path
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        data_list = list(tqdm(pool.imap(work_fn, clip_indices), total=len(clip_indices), desc=f"Loading {hdf5_key}"))

    data = np.array(data_list)
    if data.ndim == 2:
        n_dim = data.shape[-1]
        data = data.reshape(len(clip_indices), -1, n_dim)
    print(f"Loaded data: {data.shape}")

    dm_output = os.path.join(config.output_dir, "dm")
    cycles_path = os.path.join(dm_output, "all_cycle_dm.pkl")
    paths_path = os.path.join(dm_output, "all_path_dm.pkl")

    if not config.only_analysis:
        # --- Stage 2: DM extraction (cycles + paths) ---
        print("\n=== Stage 2: DM extraction ===")
        if os.path.exists(cycles_path):
            print(f"Found cached DM results at {cycles_path}, skipping extraction")
            with open(cycles_path, "rb") as f:
                all_cycle_dm = pkl.load(f)
            all_path_dm = None
            if os.path.exists(paths_path):
                with open(paths_path, "rb") as f:
                    all_path_dm = pkl.load(f)
        else:
            pipeline_cfg = build_dm_pipeline_config(config)
            pipeline_cfg.data.rodent_rollout_dir = config.data_path
            all_cycle_dm, _, all_path_dm = run_pipeline_on_data(data, clip_indices, pipeline_cfg, dm_output)

        # --- Stage 3: W2 clustering (cycles) ---
        print("\n=== Stage 3a: W2 clustering (cycles) ===")
        cluster_dict, labels, windows = run_clustering(
            all_cycle_dm, data, clip_indices, config.clustering, config.output_dir,
        )

        with open(os.path.join(config.output_dir, "cluster_results.pkl"), "wb") as f:
            pkl.dump({
                "cluster_dict": cluster_dict,
                "labels": labels,
                "windows": windows,
            }, f)
        print(f"Saved cycle clustering results")

        # --- Stage 3b: W2 clustering (paths) ---
        path_cluster_dict, path_labels, path_windows = {}, np.array([]), []
        if all_path_dm is not None:
            print("\n=== Stage 3b: W2 clustering (paths) ===")
            path_output = os.path.join(config.output_dir, "paths")
            os.makedirs(path_output, exist_ok=True)
            path_cluster_dict, path_labels, path_windows = run_clustering(
                all_path_dm, data, clip_indices, config.clustering, path_output,
            )

            with open(os.path.join(config.output_dir, "path_cluster_results.pkl"), "wb") as f:
                pkl.dump({
                    "cluster_dict": path_cluster_dict,
                    "labels": path_labels,
                    "windows": path_windows,
                }, f)
            print(f"Saved path clustering results")

    else:
        # --- Only-analysis: load cached ---
        print("\n=== Loading cached results (only_analysis=true) ===")
        assert os.path.exists(cycles_path), f"Cached DM results not found: {cycles_path}"
        with open(cycles_path, "rb") as f:
            all_cycle_dm = pkl.load(f)

        all_path_dm = None
        if os.path.exists(paths_path):
            with open(paths_path, "rb") as f:
                all_path_dm = pkl.load(f)

        results_path = os.path.join(config.output_dir, "cluster_results.pkl")
        assert os.path.exists(results_path), f"Cached cluster results not found: {results_path}"
        with open(results_path, "rb") as f:
            cached = pkl.load(f)
        cluster_dict = cached["cluster_dict"]
        labels = cached["labels"]
        windows = cached["windows"]
        print(f"Loaded cached cycle results: {len(cluster_dict)} clusters, {len(windows)} windows")

        path_cluster_dict, path_labels, path_windows = {}, np.array([]), []
        path_results_path = os.path.join(config.output_dir, "path_cluster_results.pkl")
        if os.path.exists(path_results_path):
            with open(path_results_path, "rb") as f:
                path_cached = pkl.load(f)
            path_cluster_dict = path_cached["cluster_dict"]
            path_labels = path_cached["labels"]
            path_windows = path_cached["windows"]
            print(f"Loaded cached path results: {len(path_cluster_dict)} clusters, {len(path_windows)} windows")

    # --- Load cluster labels if available ---
    cluster_labels = None
    cycle_labels_yaml_path = os.path.join(config.output_dir, "cycle_labels_seed_0.yaml")
    old_labels_yaml_path = os.path.join(config.output_dir, "labels_seed_0.yaml")
    if config.only_analysis:
        for lp in [cycle_labels_yaml_path, old_labels_yaml_path]:
            if os.path.exists(lp):
                cluster_labels = load_labels_yaml(lp)
                if cluster_labels:
                    print(f"Loaded cycle cluster labels from {lp}")
                    break

    path_cluster_labels = None
    path_labels_yaml_path = os.path.join(config.output_dir, "path_labels_seed_0.yaml")
    if config.only_analysis and os.path.exists(path_labels_yaml_path):
        path_cluster_labels = load_labels_yaml(path_labels_yaml_path)
        if path_cluster_labels:
            print(f"Loaded path cluster labels from {path_labels_yaml_path}")

    # --- Stage 4: GT transition groups ---
    print("\n=== Stage 4: GT transition groups ===")
    print(f"GT groups: {len(gt_cluster_dict)} transition types")

    all_gt_clip_ids = set()
    for items in gt_cluster_dict.values():
        for clip_id, _ in items:
            all_gt_clip_ids.add(clip_id)

    # --- Helper: run stages 5-7 for a given DM variant (cycles or paths) ---
    def _run_analysis_stages(
        variant_name: str,
        variant_cluster_dict: dict,
        variant_labels: np.ndarray,
        variant_windows: list,
        variant_cluster_labels: dict[int, str] | None,
        variant_prefix: str,
    ) -> None:
        dm_cluster_map = _build_dm_cluster_map(variant_cluster_dict) if variant_cluster_dict else {}
        dm_clustered_ids = set(dm_cluster_map.keys())
        filtered_ids = all_gt_clip_ids - dm_clustered_ids
        print(f"\n{variant_name} DM coverage: {len(dm_clustered_ids)}/{len(all_gt_clip_ids)} GT clips clustered, {len(filtered_ids)} filtered out")

        # --- Rendering ---
        if config.rendering.enabled and variant_cluster_dict:
            print(f"\n=== Rendering ({variant_name}) ===")
            try:
                from topo_mimic.dmwk.skeletonization.rendering import render_grid_clusters_side_by_side

                print(f"Rendering {variant_name} DM+W2 clusters...")
                dm_chunks = chunk_loops_walker(
                    variant_cluster_dict, config.data_path, config.rendering.walker_xml,
                    max_samples=config.rendering.max_samples,
                )
                dm_video_path = os.path.join(config.output_dir, f"{variant_prefix}_dm_w2_clusters.mp4")
                render_grid_clusters_side_by_side(
                    dm_chunks, out_path=dm_video_path,
                    framerate=config.rendering.framerate,
                    max_samples_per_cluster=config.rendering.max_samples,
                    target_size=config.rendering.target_size,
                    title=f"{variant_name} DM+W2 Clusters",
                    cluster_labels=variant_cluster_labels,
                )

                if variant_prefix == "cycle":
                    print("Rendering GT transition groups...")
                    gt_dict_int = {i: items for i, (_, items) in enumerate(sorted(gt_cluster_dict.items()))}
                    gt_label_map = {i: k for i, k in enumerate(sorted(gt_cluster_dict.keys()))}
                    gt_chunks = chunk_loops_walker(
                        gt_dict_int, config.data_path, config.rendering.walker_xml,
                        max_samples=config.rendering.max_samples,
                    )
                    gt_video_path = os.path.join(config.output_dir, "gt_transitions.mp4")
                    render_grid_clusters_side_by_side(
                        gt_chunks, out_path=gt_video_path,
                        framerate=config.rendering.framerate,
                        max_samples_per_cluster=config.rendering.max_samples,
                        target_size=config.rendering.target_size,
                        title="GT Transition Groups",
                        cluster_labels=gt_label_map,
                    )

                if filtered_ids:
                    print(f"Rendering {len(filtered_ids)} {variant_name} DM-filtered-out clips...")
                    clip_to_gt_frames: dict[int, np.ndarray] = {}
                    for items in gt_cluster_dict.values():
                        for clip_id, frames in items:
                            clip_to_gt_frames[clip_id] = frames
                    filtered_items = [(cid, clip_to_gt_frames[cid]) for cid in sorted(filtered_ids) if cid in clip_to_gt_frames]
                    filtered_render_dict: dict[int, list[tuple[int, np.ndarray]]] = {0: filtered_items}
                    filt_chunks = chunk_loops_walker(
                        filtered_render_dict, config.data_path, config.rendering.walker_xml,
                        max_samples=15,
                    )
                    filt_video_path = os.path.join(config.output_dir, f"{variant_prefix}_dm_filtered_clips.mp4")
                    render_grid_clusters_side_by_side(
                        filt_chunks, out_path=filt_video_path,
                        framerate=config.rendering.framerate,
                        max_samples_per_cluster=15,
                        target_size=config.rendering.target_size,
                        title=f"{variant_name} DM Filtered-Out Clips ({len(filtered_ids)})",
                        cluster_labels={0: "Filtered"},
                        n_cols=5,
                    )
            except Exception as e:
                print(f"WARNING: {variant_name} rendering failed: {e}")
                import traceback
                traceback.print_exc()

        # --- Embedding plots ---
        print(f"\n=== Embedding plots ({variant_name}) ===")
        gt_pca_windows, gt_pca_labels = build_gt_pca_windows(
            gt_cluster_dict, clip_indices, config.clustering.target_len,
        )
        print(f"Built {len(gt_pca_windows)} GT windows for {variant_name} plots ({len(dm_clustered_ids)} with DM clusters, {len(filtered_ids)} filtered)")

        if gt_pca_windows:
            embed_common = dict(
                gt_windows=gt_pca_windows, gt_labels=gt_pca_labels,
                data=data, clip_indices=clip_indices,
                dm_cluster_map=dm_cluster_map,
                cluster_labels=variant_cluster_labels,
            )
            for name, fn in [
                ("PCA", plot_pca_comparison),
                ("UMAP", plot_umap_comparison),
                ("t-SNE", plot_tsne_comparison),
            ]:
                try:
                    fn(
                        output_path=os.path.join(config.output_dir, f"{variant_prefix}_{name.lower().replace('-', '')}_comparison.png"),
                        **embed_common,
                    )
                except Exception as e:
                    print(f"WARNING: {variant_name} {name} plot failed: {e}")
                    import traceback
                    traceback.print_exc()

        # --- Label YAML ---
        variant_labels_yaml = os.path.join(config.output_dir, f"{variant_prefix}_labels_seed_0.yaml")
        if variant_cluster_dict and not os.path.exists(variant_labels_yaml):
            n_clusters = len(set(variant_labels[variant_labels >= 0])) if len(variant_labels) > 0 else 0
            if n_clusters > 0:
                labels_content = {
                    "experiment_id": f"walker_demo_{variant_prefix}",
                    "method": "w2_dmc_int_kmeans",
                    "n_clusters": n_clusters,
                    "seed": 0,
                    "labels": {i: "" for i in range(n_clusters)},
                }
                with open(variant_labels_yaml, "w") as f:
                    f.write("# AUTO-GENERATED - Fill in labels after watching videos\n")
                    f.write("# Label format: single behavior (\"rear\"), transition (\"walk rear\"), compound (\"rear left_turn\")\n\n")
                    yaml.dump(labels_content, f, default_flow_style=False, sort_keys=False)
                print(f"Generated {variant_name} label template: {variant_labels_yaml}")
        elif os.path.exists(variant_labels_yaml):
            print(f"{variant_name} label YAML already exists: {variant_labels_yaml}")

        # --- Confusion table ---
        if variant_cluster_dict and gt_cluster_dict_full:
            print(f"\n{variant_name} Confusion Table:")
            print_confusion_table(variant_cluster_dict, gt_cluster_dict_full, variant_cluster_labels)

    # --- Run analysis for cycles ---
    print("\n" + "=" * 60)
    print("=== Cycle Analysis ===")
    print("=" * 60)
    _run_analysis_stages("Cycle", cluster_dict, labels, windows, cluster_labels, "cycle")

    # --- Run analysis for paths ---
    if path_cluster_dict or (all_path_dm is not None):
        print("\n" + "=" * 60)
        print("=== Path Analysis ===")
        print("=" * 60)
        _run_analysis_stages("Path", path_cluster_dict, path_labels, path_windows, path_cluster_labels, "path")

    print("\n" + "=" * 60)
    print("Walker Demo Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Walker transition demo")
    parser.add_argument(
        "--config",
        default="configs/demos/demo_walker_transitions_short.yaml",
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
