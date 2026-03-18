#!/usr/bin/env python3
"""
Rodent Transition Demo
======================
Visualizes labeled rodent transitions using PCA/UMAP/t-SNE.
Each embedding shows up to 3 rows: GT transitions, DM cycles, DM paths —
all colored by behavior transition labels looked up from the GT CSV.

Usage:
    python demos/demo_rodent_transitions.py
    python demos/demo_rodent_transitions.py --config configs/demos/demo_rodent_transitions.yaml
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import argparse
import functools
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from topo_mimic.dmwk.utils.dataset_config import get_dataset_preset
from topo_mimic.dmwk.utils.load_utils import get_aggregate_data, get_clip_indices
from topo_mimic.dmwk.utils.feature_transforms import get_hdf5_key_for_dm_key
from topo_mimic.dmwk.skeletonization.postprocess import _uniform_downsample
from topo_mimic.benchmark.experiment_transitions import build_feature_matrix
from topo_mimic.benchmark.analysis.ground_truth import (
    find_all_transitions,
    load_behavior_segments_multilabel,
    TransitionSegment,
)


@dataclass
class RenderingConfig:
    enabled: bool = True
    xml_path: str = "topo_mimic/dmwk/xml/rodent.xml"
    max_samples: int = 5
    framerate: int = 30
    target_size: tuple[int, int] = (640, 480)


@dataclass
class DMPipelineConfig:
    enabled: bool = True
    dm_key: str = "intention"
    min_segment_len: int = 5
    takens: dict = field(default_factory=lambda: {"auto_estimate": True})
    dm: dict = field(default_factory=lambda: {
        "pers_delta": 1.0, "k_neighbors": 3, "use_dm_path": True,
    })


@dataclass
class SkeletonAnalysisConfig:
    enabled: bool = False
    n_minima: int = 8
    k_paths: int = 1
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    render_minima: bool = True
    minima_n_frames: int = 5
    minima_frame_radius: int = 10
    frechet_enabled: bool = True
    target_size: tuple[int, int] = (640, 480)


@dataclass
class Config:
    dataset: str = "rodent"
    data_path: str = "data/rodent_ar.h5"
    labels_csv: str = "behavior_labels.csv"
    dm_key: str = "intention"
    output_dir: str = "outputs/rodent_transitions"
    window_size: int = 100
    min_transition_frames: int = 20
    short_threshold: int = 10
    max_gap: int = 10
    top_k: int = 10
    bootstrap_n: int = 1200
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    dm_pipeline: DMPipelineConfig = field(default_factory=DMPipelineConfig)
    skeleton_analysis: SkeletonAnalysisConfig = field(default_factory=SkeletonAnalysisConfig)


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    rend_dict = cfg.pop("rendering", {})
    if "target_size" in rend_dict:
        rend_dict["target_size"] = tuple(rend_dict["target_size"])
    rend_cfg = RenderingConfig(**rend_dict)

    dm_dict = cfg.pop("dm_pipeline", {})
    dm_dict.pop("slide_stride", None)
    dm_cfg = DMPipelineConfig(
        enabled=dm_dict.get("enabled", True),
        dm_key=dm_dict.get("dm_key", "intention"),
        min_segment_len=dm_dict.get("min_segment_len", 5),
        takens=dm_dict.get("takens", {"auto_estimate": True}),
        dm=dm_dict.get("dm", {"pers_delta": 1.0, "k_neighbors": 3, "use_dm_path": True}),
    )
    cfg.pop("slide_stride", None)

    sk_dict = cfg.pop("skeleton_analysis", {})
    if "target_size" in sk_dict:
        sk_dict["target_size"] = tuple(sk_dict["target_size"])
    sk_cfg = SkeletonAnalysisConfig(**sk_dict)

    return Config(**cfg, rendering=rend_cfg, dm_pipeline=dm_cfg, skeleton_analysis=sk_cfg)


NATURE_PALETTE = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC9A6C", "#7E6148", "#B09C85",
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC9A6C", "#7E6148", "#B09C85",
]

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

_PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"


_SELF_TRANSITION_COLOR = "#BFBFBF"

_TURN_MERGE = {"Left Turn": "Turn", "Right Turn": "Turn"}


def _merge_turns(label: str) -> str:
    return _TURN_MERGE.get(label, label)


def _merge_transition_turns(label: str) -> str:
    if " -> " not in label:
        return _merge_turns(label)
    s, d = label.split(" -> ", 1)
    return f"{_merge_turns(s)} -> {_merge_turns(d)}"


def _canonicalize_transition(src: str, dst: str) -> str:
    if src == dst:
        return f"{src} -> {dst}"
    a, b = sorted([src, dst])
    return f"{a} <-> {b}"


def _is_self_transition(label: str) -> bool:
    if " <-> " in label:
        return False
    if " -> " not in label:
        return True
    s, d = label.split(" -> ", 1)
    return s == d


def _build_transition_color_map(labels: list[str]) -> dict[str, str]:
    cross = sorted({l for l in set(labels) if not _is_self_transition(l)})
    cmap: dict[str, str] = {}
    for i, label in enumerate(cross):
        cmap[label] = NATURE_PALETTE[i % len(NATURE_PALETTE)]
    for l in set(labels):
        if _is_self_transition(l):
            cmap[l] = _SELF_TRANSITION_COLOR
    return cmap


def _build_behavior_color_map(labels: list[str]) -> dict[str, str]:
    unique = sorted(set(labels))
    return {
        label: NATURE_PALETTE[i % len(NATURE_PALETTE)]
        for i, label in enumerate(unique)
    }


def _scatter(
    ax: plt.Axes,
    coords: np.ndarray,
    colors: list[str],
    title: str,
    axis_labels: list[str],
) -> None:
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, s=12, alpha=0.8,
        edgecolors="white", linewidths=0.15,
        rasterized=True,
    )
    ax.set_title(title, fontweight="bold", pad=4)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.tick_params(axis="both", which="both", length=2, pad=2)


def _plot_embedding(
    row_data: list[tuple[str, np.ndarray, list[str], list[str], list[str]]],
    axis_labels: list[str],
    suptitle: str,
    output_path: str,
) -> None:
    n_rows = len(row_data)

    all_trans: list[str] = []
    all_behaviors: list[str] = []
    for _, _, tl, sl, dl in row_data:
        all_trans.extend(tl)
        all_behaviors.extend(sl)
        all_behaviors.extend(dl)
    trans_cmap = _build_transition_color_map(all_trans)
    behav_cmap = _build_behavior_color_map(all_behaviors)

    with plt.rc_context(_NATURE_RC):
        legend_height = 1.8
        fig_height = 2.2 * n_rows + legend_height + 0.4
        fig, axes = plt.subplots(
            n_rows, 3,
            figsize=(7.2, fig_height),
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        bottom_frac = legend_height / fig_height
        fig.subplots_adjust(
            left=0.08, right=0.97, top=0.93, bottom=bottom_frac + 0.02,
            wspace=0.3, hspace=0.4,
        )

        for row_i, (row_title, coords, trans_labels, src_labels, dst_labels) in enumerate(row_data):
            n_win = len(trans_labels)

            trans_colors = [trans_cmap.get(l, _SELF_TRANSITION_COLOR) for l in trans_labels]
            src_colors = [behav_cmap.get(l, "#999999") for l in src_labels]
            dst_colors = [behav_cmap.get(l, "#999999") for l in dst_labels]

            _scatter(axes[row_i, 0], coords, trans_colors,
                     f"Transition type (n={n_win})", axis_labels)
            _scatter(axes[row_i, 1], coords, src_colors,
                     "Source behavior", axis_labels)
            _scatter(axes[row_i, 2], coords, dst_colors,
                     "Destination behavior", axis_labels)

            for col_i in range(3):
                panel_idx = row_i * 3 + col_i
                axes[row_i, col_i].text(
                    -0.12, 1.08, _PANEL_LETTERS[panel_idx],
                    transform=axes[row_i, col_i].transAxes,
                    fontsize=10, fontweight="bold", va="top",
                )

            axes[row_i, 0].set_ylabel(
                f"{row_title}\n{axis_labels[1]}",
                fontsize=7, fontweight="bold",
            )

        cross_labels = sorted({l for l in trans_cmap if not _is_self_transition(l)})
        trans_handles = [
            plt.Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=trans_cmap[l], markeredgecolor="white",
                markeredgewidth=0.3, markersize=4,
                label=l, linestyle="None",
            )
            for l in cross_labels
        ]
        trans_handles.append(plt.Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=_SELF_TRANSITION_COLOR, markeredgecolor="white",
            markeredgewidth=0.3, markersize=4,
            label="Self-transition", linestyle="None",
        ))

        behav_labels = sorted(behav_cmap.keys())
        behav_handles = [
            plt.Line2D(
                [0], [0], marker="s", color="none",
                markerfacecolor=behav_cmap[l], markeredgecolor="white",
                markeredgewidth=0.3, markersize=4,
                label=l, linestyle="None",
            )
            for l in behav_labels
        ]

        trans_title = plt.Line2D([], [], linestyle="None", label="")
        behav_title = plt.Line2D([], [], linestyle="None", label="")

        trans_ncol = min(5, len(trans_handles))
        behav_ncol = min(5, len(behav_handles))

        leg_y_trans = bottom_frac * 0.55
        leg_y_behav = bottom_frac * 0.1

        fig.text(
            0.02, leg_y_trans + 0.03, "Transition type:",
            fontsize=6.5, fontweight="bold", va="center",
            transform=fig.transFigure,
        )
        fig.legend(
            handles=trans_handles, loc="lower left",
            ncol=trans_ncol, frameon=False,
            bbox_to_anchor=(0.15, leg_y_trans),
            handletextpad=0.3, columnspacing=0.6,
            borderpad=0.1, fontsize=6,
        )

        fig.text(
            0.02, leg_y_behav + 0.03, "Behavior:",
            fontsize=6.5, fontweight="bold", va="center",
            transform=fig.transFigure,
        )
        fig.legend(
            handles=behav_handles, loc="lower left",
            ncol=behav_ncol, frameon=False,
            bbox_to_anchor=(0.15, leg_y_behav),
            handletextpad=0.3, columnspacing=0.6,
            borderpad=0.1, fontsize=6,
        )

        fig.suptitle(suptitle, fontsize=9, fontweight="bold")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, facecolor="white")
        plt.close(fig)
    print(f"  Saved {output_path}")


def chunk_loops_rodent(
    cluster_dict: dict[int | str, list[tuple[int, np.ndarray]]],
    data_path: str,
    xml_path: str,
    max_samples: int = 5,
) -> dict[int | str, list[list[np.ndarray]]]:
    import h5py
    import mujoco
    from tqdm import tqdm

    width, height = 640, 480

    with h5py.File(data_path, "r") as f:
        qpos_all = f["qpos_rollout"]

        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(mj_model)
        renderer = mujoco.Renderer(mj_model, height=height, width=width)

        cluster_chunks = {}
        for cid, items in tqdm(cluster_dict.items(), desc="Rendering clusters"):
            sampled = items[:max_samples] if len(items) > max_samples else items
            cluster_chunks[cid] = []

            for clip_id, frames in sampled:
                qpos_clip = qpos_all[clip_id]
                chunk_frames = []
                for fi in sorted(int(f_) for f_ in frames):
                    if fi >= len(qpos_clip):
                        continue
                    mj_data.qpos[:] = qpos_clip[fi]
                    mujoco.mj_forward(mj_model, mj_data)
                    renderer.update_scene(mj_data, camera="close_profile")
                    chunk_frames.append(renderer.render().copy())
                if chunk_frames:
                    cluster_chunks[cid].append(chunk_frames)

        renderer.close()

    return cluster_chunks


def _bootstrap_to_target(
    labeled_segments: list[tuple[int, np.ndarray, str, str, str]],
    target_n: int,
    window_size: int,
    seed: int = 42,
) -> tuple[list[tuple[int, np.ndarray]], list[str], list[str], list[str]]:
    if not labeled_segments:
        return [], [], [], []

    rng = np.random.default_rng(seed)

    windows: list[tuple[int, np.ndarray]] = []
    trans_labels: list[str] = []
    src_labels: list[str] = []
    dst_labels: list[str] = []

    for local_idx, seg, trans, src, dst in labeled_segments:
        resampled = _uniform_downsample(seg, window_size)
        if resampled is None:
            continue
        windows.append((local_idx, resampled))
        trans_labels.append(trans)
        src_labels.append(src)
        dst_labels.append(dst)

    n_raw = len(windows)
    if n_raw == 0 or n_raw >= target_n:
        return windows, trans_labels, src_labels, dst_labels

    lengths = np.array([len(seg) for _, seg, _, _, _ in labeled_segments])
    probs = lengths / lengths.sum()

    n_extra = target_n - n_raw
    chosen = rng.choice(len(labeled_segments), size=n_extra, replace=True, p=probs)
    for i in chosen:
        local_idx, seg, trans, src, dst = labeled_segments[i]
        indices = rng.choice(len(seg), size=window_size, replace=True)
        indices.sort()
        resampled = seg[indices]
        windows.append((local_idx, resampled))
        trans_labels.append(trans)
        src_labels.append(src)
        dst_labels.append(dst)

    return windows, trans_labels, src_labels, dst_labels


# --- DM pipeline helpers ---


def _build_dm_pipeline_config(config: Config) -> "PipelineConfig":
    from topo_mimic.dmwk.dm_pipeline import (
        PipelineConfig,
        DataConfig as DMDataConfig,
        TakensConfig,
        DMConfig as DMExtractConfig,
        RenderingConfig as DMRenderingConfig,
    )

    dm_output = os.path.join(config.output_dir, "dm")

    takens_dict = dict(config.dm_pipeline.takens)
    theiler_val = takens_dict.get("theiler", "auto")
    if theiler_val == "auto":
        pass
    elif theiler_val is not None:
        theiler_val = int(theiler_val)
    takens_dict["theiler"] = theiler_val

    dm_dict = dict(config.dm_pipeline.dm)
    use_dm_path = dm_dict.pop("use_dm_path", True)

    return PipelineConfig(
        output_dir=dm_output,
        data=DMDataConfig(
            dataset=config.dataset,
            dm_key=config.dm_pipeline.dm_key,
            rodent_rollout_dir=config.data_path,
            split="test",
        ),
        takens=TakensConfig(**takens_dict),
        dm=DMExtractConfig(use_dm_path=use_dm_path, **dm_dict),
        rendering=DMRenderingConfig(enabled=False),
    )


def _run_dm_pipeline(
    data: np.ndarray,
    clip_indices: list[int],
    config: Config,
) -> tuple[list, list, list | None, int, int]:
    import pickle as pkl
    import yaml as _yaml
    from topo_mimic.dmwk.dm_pipeline import (
        run_dm_extraction,
        get_cycle_statistics,
        get_path_statistics,
        plot_ami_fnn_diagnostics,
    )
    from topo_mimic.dmwk.skeletonization.preprocess import estimate_delay_embedding_params

    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")

    pipeline_cfg = _build_dm_pipeline_config(config)
    output_dir = pipeline_cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

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

        diag_path = os.path.join(output_dir, "ami_fnn_diagnostics.pkl")
        with open(diag_path, "wb") as f:
            pkl.dump({"tau_star": tau_star, "m_star": m_star, "diagnostics": diagnostics}, f)

        plot_path = os.path.join(output_dir, "ami_fnn_diagnostics.png")
        plot_ami_fnn_diagnostics(diagnostics, tau_star, m_star, plot_path, fnn_thr=pipeline_cfg.takens.fnn_thr)
    else:
        print(f"Using fixed Takens parameters (m={pipeline_cfg.takens.m}, tau={pipeline_cfg.takens.tau})")

    print("Running DM extraction...")
    all_cycle_dm, all_G_sm, all_path_dm = run_dm_extraction(data, pipeline_cfg)

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

    m = pipeline_cfg.takens.m
    tau = pipeline_cfg.takens.tau
    return all_cycle_dm, all_G_sm, all_path_dm, m, tau


def _label_dm_segments(
    all_segments: list[list[np.ndarray]],
    clip_indices: list[int],
    clip_behaviors: dict[int, dict[int, set[str]]],
    min_segment_len: int,
    takens_offset: int = 0,
) -> list[tuple[int, np.ndarray, str, str, str]]:
    labeled: list[tuple[int, np.ndarray, str, str, str]] = []

    for local_idx in range(len(all_segments)):
        abs_clip_id = clip_indices[local_idx] if local_idx < len(clip_indices) else local_idx

        for seg in all_segments[local_idx]:
            if len(seg) < min_segment_len:
                continue

            behavior_counts: Counter = Counter()
            for frame in seg:
                frame_int = int(frame) + takens_offset
                behaviors = clip_behaviors.get(abs_clip_id, {}).get(frame_int, set())
                for b in behaviors:
                    behavior_counts[_merge_turns(b)] += 1

            if not behavior_counts:
                continue

            sorted_behaviors = behavior_counts.most_common()
            if len(sorted_behaviors) == 1:
                src = dst = sorted_behaviors[0][0]
            else:
                src = sorted_behaviors[0][0]
                dst = sorted_behaviors[1][0]

            labeled.append((local_idx, seg, _canonicalize_transition(src, dst), src, dst))

    return labeled


# --- Skeleton analysis ---


@dataclass
class SkeletonResult:
    minima_orig: list[int]
    minima_behaviors: list[str]
    paths: list[np.ndarray]
    path_pair_labels: list[tuple[int, int]]
    purity: float


def discrete_frechet_distance(P: np.ndarray, Q: np.ndarray) -> float:
    n, m = len(P), len(Q)
    ca = np.full((n, m), -1.0)
    ca[0, 0] = np.linalg.norm(P[0] - Q[0])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], np.linalg.norm(P[i] - Q[0]))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], np.linalg.norm(P[0] - Q[j]))
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]),
                np.linalg.norm(P[i] - Q[j]),
            )
    return float(ca[n - 1, m - 1])


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


def _build_frame_labels_rodent(
    clip_id: int,
    clip_length: int,
    clip_behaviors: dict[int, dict[int, set[str]]],
) -> np.ndarray:
    labels = np.full(clip_length, "transition", dtype=object)
    frame_map = clip_behaviors.get(clip_id, {})
    for frame, behaviors in frame_map.items():
        if frame < clip_length and behaviors:
            labels[frame] = sorted(behaviors)[0]
    return labels


def extract_skeleton_for_clip(
    G_sm: "nx.Graph",
    clip_dir: str,
    data_clip: np.ndarray,
    frame_labels: np.ndarray,
    config: SkeletonAnalysisConfig,
    n_takens: int,
    n_minima_override: int | None = None,
) -> SkeletonResult | None:
    from topo_mimic.dmwk.skeletonization.remap import (
        parse_swr_remap,
        remap_cycle_to_original,
    )
    from itertools import combinations
    import networkx as nx

    remap = parse_swr_remap(clip_dir)
    n_desired = n_minima_override if n_minima_override is not None else config.n_minima
    candidate_pool = _find_all_persistent_minima(clip_dir, top_k=max(n_desired * 5, 30))
    candidate_pool = [m for m in candidate_pool if m in G_sm]

    if len(candidate_pool) < 2:
        return None

    candidate_orig = []
    candidate_behaviors = []
    for m in candidate_pool:
        orig_frames = remap_cycle_to_original([m], remap, n_takens)
        orig_idx = orig_frames[0] if orig_frames else 0
        orig_idx = min(orig_idx, len(frame_labels) - 1)
        candidate_orig.append(orig_idx)
        candidate_behaviors.append(str(frame_labels[orig_idx]))

    seen_behaviors: set[str] = set()
    selected_indices: list[int] = []
    for idx, beh in enumerate(candidate_behaviors):
        if beh != "transition" and beh not in seen_behaviors:
            seen_behaviors.add(beh)
            selected_indices.append(idx)
    for idx, beh in enumerate(candidate_behaviors):
        if idx not in selected_indices and len(selected_indices) < n_desired:
            selected_indices.append(idx)
    selected_indices = selected_indices[:n_desired]

    graph_minima = [candidate_pool[i] for i in selected_indices]
    minima_orig = [candidate_orig[i] for i in selected_indices]
    minima_behaviors = [candidate_behaviors[i] for i in selected_indices]

    if len(graph_minima) < 2:
        return None

    paths: list[np.ndarray] = []
    path_pair_labels: list[tuple[int, int]] = []

    for i, j in combinations(range(len(graph_minima)), 2):
        m_a, m_b = graph_minima[i], graph_minima[j]
        try:
            raw_paths = list(nx.edge_disjoint_paths(G_sm, m_a, m_b))
        except (nx.NetworkXError, nx.NetworkXNoPath, nx.NetworkXUnfeasible):
            continue

        for k, p in enumerate(raw_paths[:config.k_paths]):
            remapped = remap_cycle_to_original(p, remap, n_takens)
            if len(remapped) >= 3:
                paths.append(np.array(remapped, dtype=int))
                path_pair_labels.append((i, j))

    unique_behaviors = set(minima_behaviors)
    unique_behaviors.discard("transition")
    purity = len(unique_behaviors) / max(len(graph_minima), 1)

    return SkeletonResult(
        minima_orig=minima_orig,
        minima_behaviors=minima_behaviors,
        paths=paths,
        path_pair_labels=path_pair_labels,
        purity=purity,
    )


_SKEL_PALETTE_FLOAT = [
    (r / 255, g / 255, b / 255)
    for r, g, b in [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
        (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    ]
]


def _overlay_skeleton_on_ax(
    ax,
    coords: np.ndarray,
    frame_labels: np.ndarray,
    skeleton: SkeletonResult,
    color_map: dict,
    title: str,
) -> None:
    colors = [color_map.get(str(l), (0.5, 0.5, 0.5, 1.0)) for l in frame_labels]
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, s=2, alpha=0.3, edgecolors="none", rasterized=True,
    )

    for path_frames, (pi, pj) in zip(skeleton.paths, skeleton.path_pair_labels):
        valid = path_frames[path_frames < len(coords)]
        if len(valid) < 2:
            continue
        path_coords = coords[valid]
        ax.plot(
            path_coords[:, 0], path_coords[:, 1],
            color=(0.5, 0.5, 0.5), linestyle=":", linewidth=1.0, alpha=0.6, zorder=3,
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
            ax.annotate(
                f"M{k}:{beh}", (coords[orig_idx, 0], coords[orig_idx, 1]),
                fontsize=5, ha="left", va="bottom",
                xytext=(3, 3), textcoords="offset points",
            )

    ax.set_title(title, fontsize=7)


def _build_skel_color_map(labels: list[str]) -> dict[str, tuple]:
    unique = sorted(set(l for l in labels if l != "transition"))
    cmap = {label: _SKEL_PALETTE_FLOAT[i % len(_SKEL_PALETTE_FLOAT)] for i, label in enumerate(unique)}
    cmap["transition"] = (0.7, 0.7, 0.7, 1.0)
    return cmap


def _render_minima_grid_image(
    skeleton: SkeletonResult,
    qpos_clip: np.ndarray,
    xml_path: str,
    config: SkeletonAnalysisConfig,
) -> np.ndarray:
    import mujoco

    n_minima = len(skeleton.minima_orig)
    n_cols = config.minima_n_frames
    radius = config.minima_frame_radius
    w, h = config.target_size

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=h, width=w)

    grid_img = np.zeros((n_minima * h, n_cols * w, 3), dtype=np.uint8)

    for row, (orig_idx, beh) in enumerate(
        zip(skeleton.minima_orig, skeleton.minima_behaviors)
    ):
        start = max(0, orig_idx - radius)
        end = min(len(qpos_clip), orig_idx + radius + 1)
        neighborhood = list(range(start, end))
        if len(neighborhood) < n_cols:
            sample_frames = neighborhood
        else:
            step = max(1, len(neighborhood) // n_cols)
            sample_frames = neighborhood[::step][:n_cols]

        for col, fi in enumerate(sample_frames):
            if fi >= len(qpos_clip):
                continue
            mj_data.qpos[:] = qpos_clip[fi]
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera="close_profile")
            frame = renderer.render().copy()
            grid_img[row * h:(row + 1) * h, col * w:(col + 1) * w] = frame

    renderer.close()
    return grid_img


def _render_paths_grid_image(
    skeleton: SkeletonResult,
    qpos_clip: np.ndarray,
    xml_path: str,
    config: SkeletonAnalysisConfig,
) -> np.ndarray | None:
    import mujoco

    if not skeleton.paths:
        return None

    n_paths = len(skeleton.paths)
    n_cols = config.minima_n_frames
    w, h = config.target_size

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=h, width=w)

    grid_img = np.zeros((n_paths * h, n_cols * w, 3), dtype=np.uint8)

    for row, path_frames in enumerate(skeleton.paths):
        valid = path_frames[path_frames < len(qpos_clip)]
        if len(valid) == 0:
            continue
        if len(valid) <= n_cols:
            sample_frames = list(valid)
        else:
            step = max(1, len(valid) // n_cols)
            sample_frames = list(valid[::step][:n_cols])

        for col, fi in enumerate(sample_frames):
            mj_data.qpos[:] = qpos_clip[int(fi)]
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera="close_profile")
            frame = renderer.render().copy()
            grid_img[row * h:(row + 1) * h, col * w:(col + 1) * w] = frame

    renderer.close()
    return grid_img


def plot_skeleton_combined(
    data_clip: np.ndarray,
    frame_labels: np.ndarray,
    skeleton: SkeletonResult,
    config: SkeletonAnalysisConfig,
    output_path: str,
    clip_id: int,
    qpos_clip: np.ndarray | None = None,
    xml_path: str | None = None,
) -> np.ndarray:
    from umap import UMAP
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    labels_list = list(frame_labels)
    color_map = _build_skel_color_map(labels_list)
    n_points = data_clip.shape[0]

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(data_clip)

    pca_pre = PCA(n_components=min(50, data_clip.shape[1]))
    feats_pca = pca_pre.fit_transform(data_clip)

    umap_obj = UMAP(
        n_components=2,
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        random_state=42,
    )
    umap_coords = umap_obj.fit_transform(feats_pca)

    perp = min(30, n_points - 1)
    tsne_coords = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(feats_pca)

    has_render = (
        config.render_minima
        and qpos_clip is not None
        and xml_path is not None
    )

    grid_img = None
    paths_img = None
    if has_render:
        grid_img = _render_minima_grid_image(skeleton, qpos_clip, xml_path, config)
        paths_img = _render_paths_grid_image(skeleton, qpos_clip, xml_path, config)

    with plt.rc_context(_NATURE_RC):
        if grid_img is not None:
            n_minima = len(skeleton.minima_orig)
            n_paths = len(skeleton.paths) if paths_img is not None else 0

            if paths_img is not None:
                stacked_img = np.vstack([grid_img, paths_img])
            else:
                stacked_img = grid_img

            render_aspect = stacked_img.shape[1] / stacked_img.shape[0]
            left_h = 12.0
            right_h = left_h
            right_w = right_h * render_aspect
            left_w = left_h / 3.0

            fig = plt.figure(figsize=(left_w + right_w + 0.5, left_h))
            gs = fig.add_gridspec(
                3, 2,
                width_ratios=[left_w, right_w],
                wspace=0.15, hspace=0.3,
            )
        else:
            left_h = 12.0
            fig = plt.figure(figsize=(5, left_h))
            gs = fig.add_gridspec(3, 1, hspace=0.3)

        embeddings = [
            (pca_coords, f"PCA (var={pca.explained_variance_ratio_[:2].sum():.1%})"),
            (umap_coords, "UMAP"),
            (tsne_coords, "t-SNE"),
        ]
        for row, (coords, method_name) in enumerate(embeddings):
            ax = fig.add_subplot(gs[row, 0])
            _overlay_skeleton_on_ax(
                ax, coords, frame_labels, skeleton, color_map, method_name,
            )
            ax.set_aspect("equal", adjustable="datalim")
            if row == 0:
                unique = sorted(set(str(l) for l in frame_labels if str(l) != "transition"))
                if len(unique) <= 20:
                    handles = [
                        plt.Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor=color_map.get(l, (0.5, 0.5, 0.5, 1.0)),
                                   markersize=4, label=l)
                        for l in unique
                    ]
                    handles.append(plt.Line2D([0], [0], marker="*", color="w",
                                              markerfacecolor="gray", markersize=8,
                                              markeredgecolor="black", label="DM minimum"))
                    ax.legend(handles=handles, fontsize=5, loc="best", frameon=False)

        if grid_img is not None:
            ax_render = fig.add_subplot(gs[:, 1])
            ax_render.imshow(stacked_img)
            ax_render.set_xticks([])

            _, h_px = config.target_size
            ytick_positions = []
            ytick_labels_list = []
            for row in range(n_minima):
                ytick_positions.append((row + 0.5) * h_px)
                ytick_labels_list.append(
                    f"M{row}: {skeleton.minima_behaviors[row]} (f{skeleton.minima_orig[row]})"
                )
            for row, (pi, pj) in enumerate(skeleton.path_pair_labels):
                ytick_positions.append((n_minima + row + 0.5) * h_px)
                ytick_labels_list.append(f"M{pi}\u2192M{pj}")

            ax_render.set_yticks(ytick_positions)
            ax_render.set_yticklabels(ytick_labels_list, fontsize=6)

            separator_y = n_minima * h_px - 0.5
            ax_render.axhline(separator_y, color="white", linewidth=2)
            ax_render.set_title("Minima Neighborhoods  |  Paths", fontsize=7)

        fig.suptitle(
            f"Clip {clip_id} \u2014 DM Skeleton (purity={skeleton.purity:.2f})",
            fontsize=9, fontweight="bold",
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, facecolor="white")
        plt.close(fig)

    print(f"  Saved skeleton combined: {output_path}")
    return umap_coords


def plot_path_geometry(
    data_clip: np.ndarray,
    skeleton: SkeletonResult,
    umap_coords: np.ndarray,
    output_path: str,
    clip_id: int,
) -> None:
    n_paths = len(skeleton.paths)
    if n_paths == 0:
        return

    n_plot = min(n_paths, 6)
    with plt.rc_context(_NATURE_RC):
        fig, axes = plt.subplots(1, n_plot + 1, figsize=(3 * (n_plot + 1), 3))
        if n_plot + 1 == 1:
            axes = [axes]

        frechet_dm = []
        frechet_baseline = []
        path_labels = []

        for pidx in range(n_plot):
            path_frames = skeleton.paths[pidx]
            pi, pj = skeleton.path_pair_labels[pidx]
            valid = path_frames[path_frames < len(umap_coords)]
            if len(valid) < 2:
                frechet_dm.append(0)
                frechet_baseline.append(0)
                path_labels.append(f"M{pi}-M{pj}")
                continue

            dm_path_umap = umap_coords[valid]

            temporal_order = np.argsort(valid)
            temporal_umap = umap_coords[valid[temporal_order]]

            m_i_umap = umap_coords[skeleton.minima_orig[pi]]
            m_j_umap = umap_coords[skeleton.minima_orig[pj]]
            t_interp = np.linspace(0, 1, len(valid))[:, None]
            baseline_umap = m_i_umap[None, :] + t_interp * (m_j_umap - m_i_umap)[None, :]

            fd_dm = discrete_frechet_distance(dm_path_umap, temporal_umap)
            fd_bl = discrete_frechet_distance(baseline_umap, temporal_umap)
            frechet_dm.append(fd_dm)
            frechet_baseline.append(fd_bl)
            path_labels.append(f"M{pi}-M{pj}")

            ax = axes[pidx]
            ax.plot(dm_path_umap[:, 0], dm_path_umap[:, 1],
                    "b-", linewidth=1.2, alpha=0.8, label="DM path")
            ax.plot(temporal_umap[:, 0], temporal_umap[:, 1],
                    "r--", linewidth=1.0, alpha=0.6, label="Temporal")
            ax.plot(baseline_umap[:, 0], baseline_umap[:, 1],
                    "k:", linewidth=0.8, alpha=0.4, label="Straight")
            ax.scatter([m_i_umap[0], m_j_umap[0]], [m_i_umap[1], m_j_umap[1]],
                       marker="*", s=80, c="gold", edgecolors="black", zorder=5)
            ax.set_title(f"M{pi}\u2192M{pj}\nFD={fd_dm:.2f} vs BL={fd_bl:.2f}", fontsize=6)
            ax.legend(fontsize=4)

        ax_bar = axes[n_plot]
        x = np.arange(len(path_labels))
        width = 0.35
        ax_bar.bar(x - width / 2, frechet_dm, width, label="DM path", color="steelblue")
        ax_bar.bar(x + width / 2, frechet_baseline, width, label="Straight line", color="gray")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(path_labels, rotation=45, fontsize=5)
        ax_bar.set_ylabel("Fr\u00e9chet distance")
        ax_bar.set_title("DM path vs baseline", fontsize=7)
        ax_bar.legend(fontsize=5)

        fig.suptitle(f"Clip {clip_id} \u2014 Path Geometry Comparison", fontsize=9, fontweight="bold")
        fig.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, facecolor="white")
        plt.close(fig)

    print(f"  Saved path geometry: {output_path}")


def run_skeleton_analysis(
    data: np.ndarray,
    clip_indices: list[int],
    clip_behaviors: dict[int, dict[int, set[str]]],
    all_G_sm: list,
    config: Config,
) -> None:
    import networkx as nx

    sk_cfg = config.skeleton_analysis
    dm_output = os.path.join(config.output_dir, "dm")
    out_dir = os.path.join(config.output_dir, "skeleton_analysis")
    os.makedirs(out_dir, exist_ok=True)
    clip_length = data.shape[1]

    for local_idx, G_sm in enumerate(all_G_sm):
        clip_id = clip_indices[local_idx]
        clip_dir = os.path.join(dm_output, "clips", f"clip_{local_idx}")

        feat_path = os.path.join(clip_dir, "features.txt")
        if not os.path.exists(feat_path):
            continue

        with open(feat_path) as f:
            n_takens = sum(1 for _ in f)

        frame_labels = _build_frame_labels_rodent(clip_id, clip_length, clip_behaviors)
        data_clip = data[local_idx]

        n_behaviors = len(set(frame_labels) - {"transition"})
        if n_behaviors < 2:
            continue

        skeleton = extract_skeleton_for_clip(
            G_sm, clip_dir, data_clip, frame_labels, sk_cfg, n_takens,
            n_minima_override=n_behaviors,
        )
        if skeleton is None:
            print(f"  Clip {clip_id}: insufficient minima, skipping")
            continue

        print(f"  Clip {clip_id}: {len(skeleton.minima_orig)} minima, "
              f"{len(skeleton.paths)} paths, purity={skeleton.purity:.2f}")

        qpos_clip = None
        if sk_cfg.render_minima:
            try:
                import h5py
                with h5py.File(config.data_path, "r") as f:
                    qpos_clip = f["qpos_rollout"][clip_id][:]
            except Exception as e:
                print(f"  WARNING: Could not load qpos for clip {clip_id}: {e}")

        umap_coords = plot_skeleton_combined(
            data_clip, frame_labels, skeleton, sk_cfg,
            os.path.join(out_dir, f"clip_{clip_id}_skeleton.png"),
            clip_id,
            qpos_clip=qpos_clip,
            xml_path=config.rendering.xml_path,
        )

        if sk_cfg.frechet_enabled:
            plot_path_geometry(
                data_clip, skeleton, umap_coords,
                os.path.join(out_dir, f"clip_{clip_id}_path_geometry.png"),
                clip_id,
            )

        if qpos_clip is not None:
            from demos.transition_demos.skeleton_filmstrip import (
                plot_skeleton_filmstrip,
                FilmstripSkeletonConfig,
            )
            filmstrip_dir = os.path.join(config.output_dir, "skeleton_analysis_filmstrip")
            plot_skeleton_filmstrip(
                data_clip, frame_labels, skeleton,
                qpos_clip, config.rendering.xml_path,
                dataset=config.dataset,
                output_path=os.path.join(filmstrip_dir, f"clip_{clip_id}_filmstrip.pdf"),
                clip_id=clip_id,
            )


# --- Main demo ---


def run_demo(config: Config) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    preset = get_dataset_preset(config.dataset)

    print("=" * 60)
    print("Rodent Transition Demo")
    print("=" * 60)
    print(f"Dataset: {config.dataset}")
    print(f"Data: {config.data_path}")
    print(f"Labels: {config.labels_csv}")
    print(f"Output: {config.output_dir}")
    print(f"DM pipeline: {'enabled' if config.dm_pipeline.enabled else 'disabled'}")

    # --- Stage 1: Find all GT transitions ---
    print("\n=== Stage 1: Find GT transitions ===")
    all_transitions = find_all_transitions(
        config.labels_csv,
        short_threshold=config.short_threshold,
        max_gap=config.max_gap,
    )
    print(f"Found {len(all_transitions)} raw transitions")

    test_indices = sorted(get_clip_indices(
        "test",
        total_clips=preset.total_clips,
        test_size=preset.test_size,
    ))
    test_set = set(test_indices)
    transitions = [t for t in all_transitions if t.clip_id in test_set]
    print(f"Filtered to {len(transitions)} transitions in test clips ({min(test_indices)}-{max(test_indices)})")

    transitions = [
        t for t in transitions
        if len(t.all_frames) >= config.min_transition_frames
    ]
    print(f"After min_transition_frames={config.min_transition_frames}: {len(transitions)} transitions")

    # --- Stage 2: Group by directed transition type, pick top K ---
    print("\n=== Stage 2: Group transitions ===")
    type_groups: dict[str, list[TransitionSegment]] = defaultdict(list)
    for t in transitions:
        key = _canonicalize_transition(_merge_turns(t.from_behavior), _merge_turns(t.to_behavior))
        type_groups[key].append(t)

    sorted_types = sorted(type_groups.items(), key=lambda kv: len(kv[1]), reverse=True)

    print(f"\nAll transition types ({len(sorted_types)}):")
    for ttype, items in sorted_types:
        print(f"  {ttype}: {len(items)}")

    top_types = sorted_types[:config.top_k]
    top_type_names = {name for name, _ in top_types}
    transitions = [
        t for t in transitions
        if _canonicalize_transition(_merge_turns(t.from_behavior), _merge_turns(t.to_behavior)) in top_type_names
    ]
    print(f"\nKept top {config.top_k} types -> {len(transitions)} transitions")

    # --- Stage 3: Load data & build GT feature matrix ---
    print("\n=== Stage 3: Load data & build GT feature matrix ===")
    hdf5_key = get_hdf5_key_for_dm_key(config.dm_key)
    work_fn = functools.partial(
        get_aggregate_data, f"/{hdf5_key}", [], path=config.data_path,
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        from tqdm import tqdm
        data_list = list(tqdm(
            pool.imap(work_fn, test_indices),
            total=len(test_indices),
            desc=f"Loading {hdf5_key}",
        ))
    data = np.array(data_list)
    if data.ndim == 2:
        n_dim = data.shape[-1]
        data = data.reshape(len(test_indices), -1, n_dim)
    print(f"Loaded data: {data.shape}")

    abs_to_local = {abs_id: i for i, abs_id in enumerate(test_indices)}
    n_timeframes = data.shape[1]

    gt_labeled: list[tuple[int, np.ndarray, str, str, str]] = []
    for t in transitions:
        local_idx = abs_to_local[t.clip_id]
        frames = np.array([f for f in t.all_frames if f < n_timeframes])
        if len(frames) == 0:
            continue
        src = _merge_turns(t.from_behavior)
        dst = _merge_turns(t.to_behavior)
        gt_labeled.append((local_idx, frames, _canonicalize_transition(src, dst), src, dst))

    print(f"GT raw segments: {len(gt_labeled)}")

    if len(gt_labeled) == 0:
        print("ERROR: No valid GT segments. Exiting.")
        return

    # --- Stage 3b: Run DM extraction (optional) ---
    cycle_labeled: list[tuple[int, np.ndarray, str, str, str]] = []
    path_labeled: list[tuple[int, np.ndarray, str, str, str]] = []

    if config.dm_pipeline.enabled:
        print("\n=== Stage 3b: Run DM extraction ===")
        dm_output = os.path.join(config.output_dir, "dm")
        cycles_cache = os.path.join(dm_output, "all_cycle_dm.pkl")
        paths_cache = os.path.join(dm_output, "all_path_dm.pkl")

        all_G_sm = None
        if os.path.exists(cycles_cache):
            import pickle as pkl
            print(f"Loading cached DM results from {dm_output}")
            with open(cycles_cache, "rb") as f:
                all_cycle_dm = pkl.load(f)
            all_path_dm = None
            if os.path.exists(paths_cache):
                with open(paths_cache, "rb") as f:
                    all_path_dm = pkl.load(f)
            graphs_cache = os.path.join(dm_output, "all_G_sm.pkl")
            if os.path.exists(graphs_cache):
                with open(graphs_cache, "rb") as f:
                    all_G_sm = pkl.load(f)
            print(f"Loaded {len(all_cycle_dm)} cycle sets"
                  + (f", {len(all_path_dm)} path sets" if all_path_dm else "")
                  + (f", {len(all_G_sm)} graphs" if all_G_sm else ""))

            diag_path = os.path.join(dm_output, "ami_fnn_diagnostics.pkl")
            if os.path.exists(diag_path):
                with open(diag_path, "rb") as f:
                    diag = pkl.load(f)
                takens_m = diag["m_star"]
                takens_tau = diag["tau_star"]
            else:
                takens_m = config.dm_pipeline.takens.get("m", 1)
                takens_tau = config.dm_pipeline.takens.get("tau", 1)
            print(f"Takens params: m={takens_m}, tau={takens_tau}")
        else:
            all_cycle_dm, all_G_sm, all_path_dm, takens_m, takens_tau = _run_dm_pipeline(
                data, test_indices, config,
            )

        takens_offset = (takens_m - 1) * takens_tau
        print(f"Takens offset for label lookup: {takens_offset} frames")

        print("\n=== Stage 3c: Label DM segments ===")
        clip_behaviors = load_behavior_segments_multilabel(config.labels_csv)
        min_seg = config.dm_pipeline.min_segment_len

        cycle_labeled = _label_dm_segments(
            all_cycle_dm, test_indices, clip_behaviors, min_seg, takens_offset,
        )
        print(f"DM cycle raw segments with labels: {len(cycle_labeled)}")

        if all_path_dm is not None:
            path_labeled = _label_dm_segments(
                all_path_dm, test_indices, clip_behaviors, min_seg, takens_offset,
            )
            print(f"DM path raw segments with labels: {len(path_labeled)}")

    # --- Skeleton analysis ---
    if config.skeleton_analysis.enabled and all_G_sm is not None:
        print("\n=== Skeleton Analysis ===")
        try:
            clip_behaviors_skel = load_behavior_segments_multilabel(
                config.labels_csv, merge_rear=False,
            )
            run_skeleton_analysis(
                data, test_indices, clip_behaviors_skel, all_G_sm, config,
            )
        except Exception as e:
            print(f"WARNING: Skeleton analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # --- Stage 3d: Bootstrap all groups to same target count ---
    print("\n=== Stage 3d: Bootstrap resampling ===")
    target_n = config.bootstrap_n
    print(f"Raw counts — GT: {len(gt_labeled)}, Cycles: {len(cycle_labeled)}, Paths: {len(path_labeled)}")
    print(f"Bootstrap target: {target_n} points per group")

    gt_windows, gt_transition_labels, gt_src_labels, gt_dst_labels = (
        _bootstrap_to_target(gt_labeled, target_n, config.window_size, seed=42)
    )
    gt_features = build_feature_matrix(gt_windows, data)
    print(f"GT: {len(gt_windows)} windows, feature matrix {gt_features.shape}")

    dm_cycle_features = None
    dm_cycle_trans = dm_cycle_src = dm_cycle_dst = None
    dm_path_features = None
    dm_path_trans = dm_path_src = dm_path_dst = None

    if cycle_labeled:
        cycle_windows, dm_cycle_trans, dm_cycle_src, dm_cycle_dst = (
            _bootstrap_to_target(cycle_labeled, target_n, config.window_size, seed=43)
        )
        dm_cycle_features = build_feature_matrix(cycle_windows, data)
        print(f"DM Cycles: {len(cycle_windows)} windows, feature matrix {dm_cycle_features.shape}")

    if path_labeled:
        path_windows, dm_path_trans, dm_path_src, dm_path_dst = (
            _bootstrap_to_target(path_labeled, target_n, config.window_size, seed=44)
        )
        dm_path_features = build_feature_matrix(path_windows, data)
        print(f"DM Paths: {len(path_windows)} windows, feature matrix {dm_path_features.shape}")

    # --- Stage 4: Embedding plots ---
    print("\n=== Stage 4: Embedding plots ===")
    from sklearn.decomposition import PCA

    pca_2d = PCA(n_components=2)
    gt_pca_coords = pca_2d.fit_transform(gt_features)
    pca_labels = [
        f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})",
        f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})",
    ]

    def _build_row_data(
        gt_coords: np.ndarray,
        dm_cycle_coords: np.ndarray | None,
        dm_path_coords: np.ndarray | None,
    ) -> list[tuple[str, np.ndarray, list[str], list[str], list[str]]]:
        rows = [("GT Transitions", gt_coords, gt_transition_labels, gt_src_labels, gt_dst_labels)]
        if dm_cycle_coords is not None:
            rows.append(("DM Cycles", dm_cycle_coords, dm_cycle_trans, dm_cycle_src, dm_cycle_dst))
        if dm_path_coords is not None:
            rows.append(("DM Paths", dm_path_coords, dm_path_trans, dm_path_src, dm_path_dst))
        return rows

    dm_cycle_pca = pca_2d.transform(dm_cycle_features) if dm_cycle_features is not None else None
    dm_path_pca = pca_2d.transform(dm_path_features) if dm_path_features is not None else None
    _plot_embedding(
        _build_row_data(gt_pca_coords, dm_cycle_pca, dm_path_pca),
        pca_labels, "PCA of Rodent Transition Windows",
        os.path.join(config.output_dir, "pca_comparison.png"),
    )

    pca_pre = PCA(n_components=min(50, gt_features.shape[1]))
    gt_features_pca = pca_pre.fit_transform(gt_features)
    dm_cycle_pca_pre = pca_pre.transform(dm_cycle_features) if dm_cycle_features is not None else None
    dm_path_pca_pre = pca_pre.transform(dm_path_features) if dm_path_features is not None else None

    try:
        from umap import UMAP
        umap_model = UMAP(n_components=2, random_state=42)
        gt_umap = umap_model.fit_transform(gt_features_pca)
        dm_cycle_umap = umap_model.transform(dm_cycle_pca_pre) if dm_cycle_pca_pre is not None else None
        dm_path_umap = umap_model.transform(dm_path_pca_pre) if dm_path_pca_pre is not None else None
        _plot_embedding(
            _build_row_data(gt_umap, dm_cycle_umap, dm_path_umap),
            ["UMAP-1", "UMAP-2"], "UMAP of Rodent Transition Windows",
            os.path.join(config.output_dir, "umap_comparison.png"),
        )
    except Exception as e:
        print(f"  WARNING: UMAP failed: {e}")

    try:
        from sklearn.manifold import TSNE
        all_pca_list = [gt_features_pca]
        splits = [len(gt_features_pca)]
        if dm_cycle_pca_pre is not None:
            all_pca_list.append(dm_cycle_pca_pre)
            splits.append(len(dm_cycle_pca_pre))
        if dm_path_pca_pre is not None:
            all_pca_list.append(dm_path_pca_pre)
            splits.append(len(dm_path_pca_pre))

        combined = np.vstack(all_pca_list)
        perplexity = min(30, len(combined) - 1)
        tsne_coords = TSNE(
            n_components=2, random_state=42, perplexity=perplexity,
        ).fit_transform(combined)

        idx = 0
        gt_tsne = tsne_coords[idx:idx + splits[0]]
        idx += splits[0]
        dm_cycle_tsne = None
        dm_path_tsne = None
        if dm_cycle_pca_pre is not None:
            dm_cycle_tsne = tsne_coords[idx:idx + splits[1]]
            idx += splits[1]
        if dm_path_pca_pre is not None:
            dm_path_tsne = tsne_coords[idx:idx + splits[-1]]

        _plot_embedding(
            _build_row_data(gt_tsne, dm_cycle_tsne, dm_path_tsne),
            ["t-SNE-1", "t-SNE-2"], "t-SNE of Rodent Transition Windows",
            os.path.join(config.output_dir, "tsne_comparison.png"),
        )
    except Exception as e:
        print(f"  WARNING: t-SNE failed: {e}")

    # --- Stage 5: Summary table ---
    print("\n=== Stage 5: Transition summary ===")
    summary_path = os.path.join(config.output_dir, "transition_summary.txt")
    lines = []
    lines.append(f"{'Transition Type':<40} {'Count':>6}")
    lines.append("-" * 48)
    for ttype, items in sorted_types[:config.top_k]:
        lines.append(f"{ttype:<40} {len(items):>6}")
    lines.append("-" * 48)
    lines.append(f"{'Total':<40} {len(transitions):>6}")
    lines.append(f"\nAll types (including non-top-K):")
    for ttype, items in sorted_types:
        lines.append(f"  {ttype:<40} {len(items):>6}")

    summary_text = "\n".join(lines)
    print(summary_text)
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"  Saved {summary_path}")

    # --- Stage 6: Rendering ---
    if config.rendering.enabled:
        print("\n=== Stage 6: Rendering ===")
        try:
            from topo_mimic.dmwk.skeletonization.rendering import (
                render_grid_clusters_side_by_side,
            )

            gt_dict: dict[int, list[tuple[int, np.ndarray]]] = {}
            gt_label_map: dict[int, str] = {}
            for i, (ttype, _) in enumerate(top_types):
                gt_items = []
                for t in type_groups[ttype]:
                    frames = np.array(t.all_frames)
                    resampled = _uniform_downsample(frames, config.window_size)
                    if resampled is not None:
                        gt_items.append((t.clip_id, resampled))
                gt_dict[i] = gt_items
                gt_label_map[i] = ttype

            gt_chunks = chunk_loops_rodent(
                gt_dict, config.data_path, config.rendering.xml_path,
                max_samples=config.rendering.max_samples,
            )
            gt_video_path = os.path.join(config.output_dir, "gt_transitions.mp4")
            render_grid_clusters_side_by_side(
                gt_chunks,
                out_path=gt_video_path,
                framerate=config.rendering.framerate,
                max_samples_per_cluster=config.rendering.max_samples,
                target_size=config.rendering.target_size,
                title="GT Rodent Transitions",
                cluster_labels=gt_label_map,
            )
            print(f"  Saved {gt_video_path}")
        except Exception as e:
            print(f"  WARNING: Rendering failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Rodent Transition Demo Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Rodent transition demo")
    parser.add_argument(
        "--config",
        default="configs/demos/transition_demos/demo_rodent_transitions.yaml",
        help="Path to demo config YAML",
    )
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config not found: {args.config}, using defaults")
        config = Config()

    run_demo(config)


if __name__ == "__main__":
    main()
