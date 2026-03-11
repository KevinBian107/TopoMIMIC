#!/usr/bin/env python3
"""
Transition Topology Demo
=========================
Empirically verifies and visualizes two transition styles in DM cycles:
- Theta-graph: Two edge-disjoint paths between density minima (entering/exiting)
- Single-path: Start/end poses similar in intention space, only one path exists

Computes edge-connectivity, extracts edge-disjoint paths, renders each as a
separate synced GIF, and computes segment endpoint similarity matrices.

Usage:
    python demos/demo_transition_topology.py
    python demos/demo_transition_topology.py --config configs/demos/demo_transition_topology.yaml
    python demos/demo_transition_topology.py --skip-gifs
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import argparse
import yaml
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, "/home/jovyan/vast/kaiwen/TopoVNL")
sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")

from demos.property.demo_cycle_asymmetry import (
    load_single_clip,
    run_dm_extraction,
    order_cycle_vertices,
    DataConfig,
    FeatureSpaceConfig,
    DMConfig,
    VisualizationConfig,
    DMResult,
)
from topo_mimic.dmwk.skeletonization.rendering import (
    _load_full_clip_frames,
    _ensure_rgb_uint8,
)
from topo_mimic.dmwk.skeletonization.remap import (
    parse_swr_remap,
    remap_cycle_to_original,
    split_cycle_into_segments,
)
from topo_mimic.dmwk.utils.feature_transforms import (
    get_hdf5_key_for_dm_key,
)
from topo_mimic.dmwk.utils.load_utils import get_aggregate_data


# ---------------------------------------------------------------------------
# DM-Path extraction (minima finding for edge-connectivity analysis)
# ---------------------------------------------------------------------------


@dataclass
class DMPathResult:
    min1_remapped: int | None
    min2_remapped: int | None


def _parse_sorted_weights(output_dir: str) -> dict[int, float]:
    weights_path = os.path.join(output_dir, "sorted-weights.txt")
    weights = {}
    with open(weights_path) as f:
        for i, line in enumerate(f):
            weights[i] = float(line.strip())
    return weights


def _parse_persistence_pairs(output_dir: str) -> list[tuple[int, int, int, float]]:
    edge_path = os.path.join(output_dir, "edge_for_morse_only.txt")
    pairs = []
    with open(edge_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            v0, v1, pair_type = int(parts[0]), int(parts[1]), int(parts[2])
            pers = float("inf") if parts[3] == "inf" else float(parts[3])
            pairs.append((v0, v1, pair_type, pers))
    return pairs


def _find_persistent_minima(
    pairs: list[tuple[int, int, int, float]],
    weights: dict[int, float],
    top_k: int = 2,
) -> list[int]:
    type0_pairs = [(v0, v1, pers) for v0, v1, t, pers in pairs if t == 0]
    type0_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    seen = set()
    minima = []
    for v0, v1, pers in type0_pairs:
        w0 = weights.get(v0, float("inf"))
        w1 = weights.get(v1, float("inf"))
        min_v = v0 if w0 < w1 else v1
        if min_v not in seen:
            minima.append(min_v)
            seen.add(min_v)
        if len(minima) >= top_k:
            break
    return minima


def find_dm_minima(dm_result: DMResult) -> DMPathResult:
    """Find the two most persistent density minima in the DM graph."""
    output_dir = dm_result.dm_output_dir
    pairs_path = os.path.join(output_dir, "edge_for_morse_only.txt")
    weights_path = os.path.join(output_dir, "sorted-weights.txt")

    if not os.path.exists(pairs_path) or not os.path.exists(weights_path):
        print("    Minima: persistence files not found, skipping")
        return DMPathResult(min1_remapped=None, min2_remapped=None)

    pairs = _parse_persistence_pairs(output_dir)
    weights = _parse_sorted_weights(output_dir)
    minima = _find_persistent_minima(pairs, weights, top_k=2)

    min1 = minima[0] if len(minima) >= 1 else None
    min2 = minima[1] if len(minima) >= 2 else None

    if min1 is not None and min2 is not None:
        print(f"    Minima: {min1}, {min2} (top-2 persistent)")
    else:
        print(f"    Minima: found only {len(minima)}, need 2")

    return DMPathResult(min1_remapped=min1, min2_remapped=min2)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ClipDiscoveryConfig:
    enabled: bool = True
    top_k: int = 10
    z_high: float = 0.06
    z_low: float = 0.02
    min_peak_frames: int = 15
    min_valley_frames: int = 10
    smooth_window: int = 5


@dataclass
class Config:
    output_dir: str = "outputs/transition_topology"
    data: DataConfig = field(default_factory=DataConfig)
    feature_spaces: list[FeatureSpaceConfig] = field(default_factory=list)
    clip_ids: list[int] = field(default_factory=list)
    dm: DMConfig = field(default_factory=DMConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    clip_discovery: ClipDiscoveryConfig = field(default_factory=ClipDiscoveryConfig)
    min_segment_frames: int = 50


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    data_cfg = DataConfig(**cfg_dict.get("data", {}))

    feature_spaces = []
    for fs_dict in cfg_dict.get("feature_spaces", []):
        feature_spaces.append(FeatureSpaceConfig(**fs_dict))

    dm_cfg = DMConfig(**cfg_dict.get("dm", {}))
    vis_cfg = VisualizationConfig(**cfg_dict.get("visualization", {}))
    disc_cfg = ClipDiscoveryConfig(**cfg_dict.get("clip_discovery", {}))

    return Config(
        output_dir=cfg_dict.get("output_dir", "outputs/transition_topology"),
        data=data_cfg,
        feature_spaces=feature_spaces,
        clip_ids=cfg_dict.get("clip_ids", []),
        dm=dm_cfg,
        visualization=vis_cfg,
        clip_discovery=disc_cfg,
        min_segment_frames=cfg_dict.get("min_segment_frames", 50),
    )


# ---------------------------------------------------------------------------
# Stage 1b: Clip Discovery — find rear->walk->rear clips
# ---------------------------------------------------------------------------


@dataclass
class RearWalkRearHit:
    clip_id: int
    score: float
    peak1_frames: tuple[int, int]
    peak2_frames: tuple[int, int]
    valley_frames: tuple[int, int]
    peak1_max_z: float
    peak2_max_z: float
    valley_mean_z: float


def _score_rear_walk_rear(
    z_series: np.ndarray,
    z_high: float,
    z_low: float,
    min_peak_frames: int,
    min_valley_frames: int,
    smooth_window: int,
) -> RearWalkRearHit | None:
    n = len(z_series)
    kernel = np.ones(smooth_window) / smooth_window
    z_smooth = np.convolve(z_series, kernel, mode="same")

    high = z_smooth > z_high
    transitions = np.diff(high.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1
    if high[0]:
        starts = np.insert(starts, 0, 0)
    if high[-1]:
        ends = np.append(ends, n)

    peaks = [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s) >= min_peak_frames]
    if len(peaks) < 2:
        return None

    best: RearWalkRearHit | None = None
    for i in range(len(peaks) - 1):
        _, end1 = peaks[i]
        start2, _ = peaks[i + 1]
        valley_dur = start2 - end1
        if valley_dur < min_valley_frames:
            continue
        valley_z = z_smooth[end1:start2]
        if valley_z.mean() > z_low:
            continue
        p1_max = float(z_smooth[peaks[i][0]:peaks[i][1]].max())
        p2_max = float(z_smooth[peaks[i+1][0]:peaks[i+1][1]].max())
        depth = z_high - valley_z.mean()
        score = p1_max * p2_max * depth * valley_dur
        if best is None or score > best.score:
            best = RearWalkRearHit(
                clip_id=-1,
                score=score,
                peak1_frames=peaks[i],
                peak2_frames=peaks[i + 1],
                valley_frames=(end1, start2),
                peak1_max_z=p1_max,
                peak2_max_z=p2_max,
                valley_mean_z=float(valley_z.mean()),
            )
    return best


def discover_rear_walk_rear_clips(
    rodent_rollout_dir: str,
    disc_cfg: ClipDiscoveryConfig,
) -> list[RearWalkRearHit]:
    import h5py

    with h5py.File(rodent_rollout_dir, "r") as f:
        key = "qpos_rollout" if "qpos_rollout" in f else "qpos"
        root_z = f[key][:, :, 2]

    n_clips = root_z.shape[0]
    hits: list[RearWalkRearHit] = []

    for cid in range(n_clips):
        hit = _score_rear_walk_rear(
            root_z[cid],
            z_high=disc_cfg.z_high,
            z_low=disc_cfg.z_low,
            min_peak_frames=disc_cfg.min_peak_frames,
            min_valley_frames=disc_cfg.min_valley_frames,
            smooth_window=disc_cfg.smooth_window,
        )
        if hit is not None:
            hit.clip_id = cid
            hits.append(hit)

    hits.sort(key=lambda h: h.score, reverse=True)
    top = hits[:disc_cfg.top_k]

    print(f"  Clip discovery: {len(hits)}/{n_clips} clips match rear-walk-rear")
    print(f"  Top {len(top)} clips:")
    for h in top:
        print(
            f"    Clip {h.clip_id:3d}: score={h.score:.4f} | "
            f"rear1={h.peak1_frames} z={h.peak1_max_z:.3f} | "
            f"walk={h.valley_frames} z_avg={h.valley_mean_z:.3f} | "
            f"rear2={h.peak2_frames} z={h.peak2_max_z:.3f}"
        )

    return top


# ---------------------------------------------------------------------------
# Stage 2: Edge-Connectivity Analysis
# ---------------------------------------------------------------------------


@dataclass
class ConnectivityResult:
    edge_connectivity: int
    edge_disjoint_paths: list[list[int]]
    paths_remapped: list[np.ndarray]
    topology_type: str
    min1_remapped: int
    min2_remapped: int


def analyze_edge_connectivity(
    G_dm: nx.Graph,
    minima_result: DMPathResult,
    remap_to_orig: dict[int, int],
    n_takens: int,
) -> ConnectivityResult | None:
    min1 = minima_result.min1_remapped
    min2 = minima_result.min2_remapped

    if min1 is None or min2 is None:
        print("    Connectivity: insufficient minima, skipping")
        return None

    if min1 not in G_dm or min2 not in G_dm:
        print(f"    Connectivity: minima {min1}/{min2} not in graph, skipping")
        return None

    ec = nx.edge_connectivity(G_dm, min1, min2)
    print(f"    Edge connectivity({min1}, {min2}) = {ec}")

    try:
        raw_paths = list(nx.edge_disjoint_paths(G_dm, min1, min2))
    except nx.NetworkXError:
        raw_paths = []

    paths_remapped = []
    for p in raw_paths:
        remapped = remap_cycle_to_original(p, remap_to_orig, n_takens)
        paths_remapped.append(remapped)

    topology_type = "theta" if ec >= 2 else "single-path"
    print(f"    Topology: {topology_type} ({len(raw_paths)} edge-disjoint paths)")
    for i, p in enumerate(raw_paths):
        print(f"      Path {i+1}: {len(p)} DM vertices -> {len(paths_remapped[i])} Takens indices")

    return ConnectivityResult(
        edge_connectivity=ec,
        edge_disjoint_paths=raw_paths,
        paths_remapped=paths_remapped,
        topology_type=topology_type,
        min1_remapped=min1,
        min2_remapped=min2,
    )


# ---------------------------------------------------------------------------
# Stage 3: Segment Endpoint Similarity
# ---------------------------------------------------------------------------


@dataclass
class EndpointSimilarityResult:
    segments: list[np.ndarray]
    n_segments: int
    similarity_matrix_cosine: np.ndarray
    similarity_matrix_l2: np.ndarray
    endpoint_labels: list[str]
    intention_endpoints: np.ndarray


def compute_endpoint_similarity(
    segments: list[np.ndarray],
    intention_data: np.ndarray,
    takens_m: int,
    takens_tau: int,
) -> EndpointSimilarityResult | None:
    if not segments:
        return None

    offset = (takens_m - 1) * takens_tau
    n_frames = intention_data.shape[0]

    endpoints = []
    labels = []
    valid_segments = []

    for i, seg in enumerate(segments):
        if len(seg) < 2:
            continue
        frame_start = int(seg[0]) + offset
        frame_end = int(seg[-1]) + offset
        if frame_start >= n_frames or frame_end >= n_frames:
            continue
        valid_segments.append(seg)
        endpoints.append(intention_data[frame_start])
        labels.append(f"Seg{len(valid_segments)}-start")
        endpoints.append(intention_data[frame_end])
        labels.append(f"Seg{len(valid_segments)}-end")

    if len(endpoints) < 2:
        return None

    endpoint_matrix = np.array(endpoints)
    cos_sim = cosine_similarity(endpoint_matrix)

    diffs = endpoint_matrix[:, np.newaxis, :] - endpoint_matrix[np.newaxis, :, :]
    l2_dist = np.linalg.norm(diffs, axis=-1)

    return EndpointSimilarityResult(
        segments=valid_segments,
        n_segments=len(valid_segments),
        similarity_matrix_cosine=cos_sim,
        similarity_matrix_l2=l2_dist,
        endpoint_labels=labels,
        intention_endpoints=endpoint_matrix,
    )


def plot_endpoint_heatmap(
    result: EndpointSimilarityResult,
    clip_id: int,
    output_path: str,
    dpi: int = 150,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    n = len(result.endpoint_labels)

    im0 = axes[0].imshow(result.similarity_matrix_cosine, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(result.endpoint_labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels(result.endpoint_labels, fontsize=8)
    axes[0].set_title("Cosine Similarity", fontsize=11)
    for i in range(n):
        for j in range(n):
            axes[0].text(j, i, f"{result.similarity_matrix_cosine[i,j]:.2f}",
                         ha="center", va="center", fontsize=7,
                         color="white" if abs(result.similarity_matrix_cosine[i,j]) > 0.5 else "black")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(result.similarity_matrix_l2, cmap="viridis")
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    axes[1].set_xticklabels(result.endpoint_labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticklabels(result.endpoint_labels, fontsize=8)
    axes[1].set_title("L2 Distance", fontsize=11)
    for i in range(n):
        for j in range(n):
            axes[1].text(j, i, f"{result.similarity_matrix_l2[i,j]:.2f}",
                         ha="center", va="center", fontsize=7,
                         color="white" if result.similarity_matrix_l2[i,j] > result.similarity_matrix_l2.max()*0.5 else "black")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"Clip {clip_id} — Segment Endpoint Similarity (intention space)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved endpoint heatmap: {output_path}")


# ---------------------------------------------------------------------------
# Stage 4: Visualization — Path GIFs
# ---------------------------------------------------------------------------


PATH_COLORS = ["#e6194b", "#4363d8", "#3cb44b", "#f58231"]
SEGMENT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990", "#dcbeff", "#9A6324",
]
BRIDGE_COLOR = "#888888"


class TransitionPathAnimation:
    """Two-row animation: paths on top, segmented cycle on bottom.

    Layout:
        Row 0: [PCA 3D — paths]             [Path 1 Rodent] [Path 2 Rodent]
        Row 1: [PCA 3D — segmented cycle]   [Full Rodent]   [Seg 1] [Seg 2] ...

    Both PCA panels rotate in sync. Top paths are progressively traced.
    Bottom cycle is drawn fully with per-segment coloring and a playhead
    synced to the clip timeline. The full rodent panel plays live during
    segments and freezes between them. Segment panels freeze when inactive.
    """

    def __init__(
        self,
        pts_3d: np.ndarray,
        all_path_indices: list[list[int]],
        segments: list[np.ndarray],
        cycle_indices: np.ndarray | None,
        min1_idx: int | None,
        min2_idx: int | None,
        path_colors: list[str],
        topology_label: str,
        rodent_frames: list[np.ndarray],
        takens_points: np.ndarray,
        m: int,
        tau: int,
        endpoint_result: EndpointSimilarityResult | None = None,
        n_frames: int = 120,
        fps: int = 15,
        dpi: int = 150,
    ):
        self.pts_3d = pts_3d
        self.all_path_indices = all_path_indices
        self.segments = segments
        self.cycle_indices = cycle_indices
        self.min1_idx = min1_idx
        self.min2_idx = min2_idx
        self.path_colors = path_colors
        self.topology_label = topology_label
        self.rodent_frames = rodent_frames
        self.endpoint_result = endpoint_result
        self.n_frames = n_frames
        self.fps = fps
        self.dpi = dpi

        self.offset = (m - 1) * tau
        self.n_paths = len(all_path_indices)
        self.n_segs = len(segments)
        n_clip = len(rodent_frames)

        self.per_path_frame_indices = []
        for path_idx in all_path_indices:
            fi = [i + self.offset for i in path_idx]
            fi = [f for f in fi if f < n_clip]
            self.per_path_frame_indices.append(fi)

        self.seg_frame_ranges: list[tuple[int, int]] = []
        for seg in segments:
            f_start = min(int(seg[0]) + self.offset, n_clip - 1)
            f_end = min(int(seg[-1]) + self.offset, n_clip - 1)
            self.seg_frame_ranges.append((f_start, f_end))

        self._clip_len = n_clip

        self._idx_to_seg: dict[int, int] = {}
        for si, seg in enumerate(segments):
            for idx in seg:
                self._idx_to_seg[int(idx)] = si

        self._precompute_cycle_edges(takens_points)
        self._setup_figure()

    def _clip_frame_at(self, progress: float) -> int:
        return min(int(progress * (self._clip_len - 1)), self._clip_len - 1)

    def _active_seg(self, clip_frame: int) -> int | None:
        for si, (fs, fe) in enumerate(self.seg_frame_ranges):
            if fs <= clip_frame <= fe:
                return si
        return None

    def _precompute_cycle_edges(self, takens_points: np.ndarray):
        self._seg_edge_groups: dict[int, list] = {}
        self._bridge_edges: list = []
        self._non_cycle_mask = np.ones(len(self.pts_3d), dtype=bool)

        if self.cycle_indices is None or len(self.cycle_indices) < 2:
            return

        cycle_ordered = order_cycle_vertices(takens_points, self.cycle_indices)
        self._cycle_ordered = cycle_ordered

        idx_to_seg: dict[int, int] = {}
        all_cycle_set: set[int] = set()
        for si, seg in enumerate(self.segments):
            for idx in seg:
                idx_to_seg[int(idx)] = si
                all_cycle_set.add(int(idx))

        for idx in all_cycle_set:
            if idx < len(self._non_cycle_mask):
                self._non_cycle_mask[idx] = False

        n = len(cycle_ordered)
        for i in range(n):
            j = (i + 1) % n
            idx_i, idx_j = int(cycle_ordered[i]), int(cycle_ordered[j])
            seg_i = idx_to_seg.get(idx_i, -1)
            seg_j = idx_to_seg.get(idx_j, -1)
            edge = [self.pts_3d[idx_i].tolist(), self.pts_3d[idx_j].tolist()]
            if seg_i == seg_j and seg_i >= 0:
                self._seg_edge_groups.setdefault(seg_i, []).append(edge)
            else:
                self._bridge_edges.append(edge)

    def _setup_figure(self):
        n_mid = 1 + max(self.n_paths, 1)
        n_bot = 1 + max(self.n_segs, 1)
        width = max(7 + 4 * max(self.n_paths, 1), 7 + 3.5 * max(self.n_segs, 1))

        self.fig = plt.figure(figsize=(width, 16), facecolor="white")
        outer = GridSpec(
            3, 1, figure=self.fig,
            height_ratios=[0.8, 1, 1], hspace=0.3,
        )

        has_heatmap = self.endpoint_result is not None
        top_gs = GridSpecFromSubplotSpec(
            1, 2 if has_heatmap else 1, subplot_spec=outer[0],
            width_ratios=[1, 1] if has_heatmap else [1], wspace=0.15,
        )

        self.ax_full_rodent = self.fig.add_subplot(top_gs[0, 0])
        self.ax_full_rodent.set_axis_off()

        self.ax_heatmap = None
        if has_heatmap:
            self.ax_heatmap = self.fig.add_subplot(top_gs[0, 1])
            self._draw_static_heatmap()

        mid_gs = GridSpecFromSubplotSpec(
            1, n_mid, subplot_spec=outer[1],
            width_ratios=[2] + [1] * max(self.n_paths, 1), wspace=0.08,
        )
        bot_gs = GridSpecFromSubplotSpec(
            1, n_bot, subplot_spec=outer[2],
            width_ratios=[2] + [1] * max(self.n_segs, 1), wspace=0.08,
        )

        self.ax_path_pca = self.fig.add_subplot(mid_gs[0, 0], projection="3d", facecolor="white")
        self.ax_path_imgs = []
        for i in range(self.n_paths):
            ax = self.fig.add_subplot(mid_gs[0, 1 + i])
            ax.set_axis_off()
            self.ax_path_imgs.append(ax)

        self.ax_cycle_pca = self.fig.add_subplot(bot_gs[0, 0], projection="3d", facecolor="white")
        self.ax_seg_imgs = []
        for i in range(self.n_segs):
            ax = self.fig.add_subplot(bot_gs[0, 1 + i])
            ax.set_axis_off()
            self.ax_seg_imgs.append(ax)

    def _draw_static_heatmap(self):
        ax = self.ax_heatmap
        r = self.endpoint_result
        n = len(r.endpoint_labels)

        im = ax.imshow(r.similarity_matrix_cosine, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(r.endpoint_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(r.endpoint_labels, fontsize=7)
        ax.set_title("Endpoint Cosine Similarity", fontsize=10)
        for i in range(n):
            for j in range(n):
                val = r.similarity_matrix_cosine[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(val) > 0.5 else "black")
        self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _draw_path_pca(self, progress: float, azimuth: float):
        ax = self.ax_path_pca
        ax.clear()
        ax.view_init(elev=25, azim=azimuth)

        ax.scatter(
            self.pts_3d[:, 0], self.pts_3d[:, 1], self.pts_3d[:, 2],
            c="gray", s=3, alpha=0.15, edgecolors="none",
        )

        if self.cycle_indices is not None and len(self.cycle_indices) > 1:
            cpts = self.pts_3d[self.cycle_indices]
            ax.plot(cpts[:, 0], cpts[:, 1], cpts[:, 2], c="lightgray", linewidth=1.0, alpha=0.25)

        for pi, path_idx in enumerate(self.all_path_indices):
            color = self.path_colors[pi] if pi < len(self.path_colors) else "gray"
            n_trace = max(1, int(len(path_idx) * progress))
            valid = [i for i in path_idx[:n_trace] if i < len(self.pts_3d)]
            if len(valid) > 1:
                tp = self.pts_3d[valid]
                ax.plot(tp[:, 0], tp[:, 1], tp[:, 2], c=color, linewidth=2.5, alpha=0.9)
                head = tp[-1]
                ax.scatter([head[0]], [head[1]], [head[2]], c="yellow", s=100, edgecolors=color, linewidths=2, zorder=200)

        for min_idx in [self.min1_idx, self.min2_idx]:
            if min_idx is not None and min_idx < len(self.pts_3d):
                p = self.pts_3d[min_idx]
                ax.scatter([p[0]], [p[1]], [p[2]], c="gold", s=200, marker="*", zorder=200)

        ax.set_axis_off()
        ax.set_title(self.topology_label, fontsize=10)

    def _draw_cycle_pca(self, clip_frame: int, active_seg: int | None, azimuth: float, progress: float):
        ax = self.ax_cycle_pca
        ax.clear()
        ax.view_init(elev=25, azim=azimuth)

        nc = self._non_cycle_mask
        ax.scatter(
            self.pts_3d[nc, 0], self.pts_3d[nc, 1], self.pts_3d[nc, 2],
            c="gray", s=3, alpha=0.15, edgecolors="none",
        )

        if self.cycle_indices is not None and len(self._cycle_ordered) > 1:
            n_cycle = len(self._cycle_ordered)
            n_trace = max(1, int(n_cycle * progress))
            traced = self._cycle_ordered[:n_trace]

            for i in range(len(traced) - 1):
                idx_i, idx_j = int(traced[i]), int(traced[i + 1])
                if idx_i >= len(self.pts_3d) or idx_j >= len(self.pts_3d):
                    continue
                seg_i = self._idx_to_seg.get(idx_i, -1)
                seg_j = self._idx_to_seg.get(idx_j, -1)
                p0 = self.pts_3d[idx_i]
                p1 = self.pts_3d[idx_j]
                if seg_i == seg_j and seg_i >= 0:
                    color = SEGMENT_COLORS[seg_i % len(SEGMENT_COLORS)]
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                            c=color, linewidth=2.5, alpha=0.9)
                else:
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                            c=BRIDGE_COLOR, linewidth=1.0, alpha=0.35)

            traced_valid = [int(i) for i in traced if int(i) < len(self.pts_3d)]
            for si, seg in enumerate(self.segments):
                color = SEGMENT_COLORS[si % len(SEGMENT_COLORS)]
                seg_set = set(int(i) for i in seg)
                visible = [i for i in traced_valid if i in seg_set]
                if visible:
                    sp = self.pts_3d[visible]
                    ax.scatter(sp[:, 0], sp[:, 1], sp[:, 2], c=color, s=15, alpha=0.8, edgecolors="none")

            head_idx = int(traced[-1])
            if head_idx < len(self.pts_3d):
                head = self.pts_3d[head_idx]
                head_seg = self._idx_to_seg.get(head_idx, -1)
                ec = SEGMENT_COLORS[head_seg % len(SEGMENT_COLORS)] if head_seg >= 0 else "black"
                ax.scatter([head[0]], [head[1]], [head[2]], c="yellow", s=120, edgecolors=ec, linewidths=1.5, zorder=200)

        ax.set_axis_off()
        n_s = self.n_segs
        ax.set_title(f"Cycle ({n_s} segment{'s' if n_s != 1 else ''})", fontsize=10)

    def animate(self, frame: int):
        progress = frame / max(self.n_frames - 1, 1)
        azimuth = (frame / self.n_frames) * 360

        clip_frame = self._clip_frame_at(progress)

        cycle_head_seg: int | None = None
        seg_last_frame: dict[int, int] = {}
        if hasattr(self, '_cycle_ordered') and len(self._cycle_ordered) > 1:
            n_cycle = len(self._cycle_ordered)
            n_trace = max(1, int(n_cycle * progress))
            traced = self._cycle_ordered[:n_trace]
            for idx in traced:
                si = self._idx_to_seg.get(int(idx))
                if si is not None:
                    cf = min(int(idx) + self.offset, self._clip_len - 1)
                    seg_last_frame[si] = cf
            head_idx = int(traced[-1])
            cycle_head_seg = self._idx_to_seg.get(head_idx)

        self._draw_path_pca(progress, azimuth)
        self._draw_cycle_pca(clip_frame, cycle_head_seg, azimuth, progress)

        for pi, ax in enumerate(self.ax_path_imgs):
            ax.clear()
            ax.set_axis_off()
            fi = self.per_path_frame_indices[pi]
            color = self.path_colors[pi] if pi < len(self.path_colors) else "gray"
            fi_n = max(1, int(len(fi) * progress))
            if fi:
                current = fi[min(fi_n - 1, len(fi) - 1)]
                ax.imshow(self.rodent_frames[current])
                ax.set_title(f"Path {pi+1} — F{current}", fontsize=9, color=color, fontweight="bold")
            else:
                ax.set_title(f"Path {pi+1}", fontsize=9)

        ax_fr = self.ax_full_rodent
        ax_fr.clear()
        ax_fr.set_axis_off()
        ax_fr.set_title(f"Full Clip Playback — Frame {clip_frame}", fontsize=10)
        if 0 <= clip_frame < len(self.rodent_frames):
            ax_fr.imshow(self.rodent_frames[clip_frame])

        for si, ax in enumerate(self.ax_seg_imgs):
            ax.clear()
            ax.set_axis_off()
            color = SEGMENT_COLORS[si % len(SEGMENT_COLORS)]

            if si in seg_last_frame:
                show = seg_last_frame[si]
                is_active = (cycle_head_seg == si)
                if is_active:
                    label = f"Seg {si+1} [ACTIVE] — F{show}"
                    tc, bw = color, 3
                else:
                    label = f"Seg {si+1} (traced) — F{show}"
                    tc, bw = color, 1
            else:
                fs, _ = self.seg_frame_ranges[si]
                show = fs
                label = f"Seg {si+1} (waiting)"
                tc, bw = "gray", 1

            if 0 <= show < len(self.rodent_frames):
                ax.imshow(self.rodent_frames[show])
            ax.set_title(label, fontsize=8, color=tc, fontweight="bold" if bw == 3 else "normal")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(bw)
                spine.set_alpha(1.0 if bw == 3 else 0.3)

        return []

    def save_gif(self, output_path: str):
        anim = FuncAnimation(
            self.fig, self.animate,
            frames=self.n_frames,
            interval=1000 // self.fps,
            blit=False,
        )
        anim.save(output_path, writer="pillow", fps=self.fps, dpi=self.dpi)
        plt.close(self.fig)
        print(f"    Saved GIF: {output_path}")


# ---------------------------------------------------------------------------
# Stage 4b: Static 3D visualization with all paths
# ---------------------------------------------------------------------------


def plot_static_3d_paths(
    pts_3d: np.ndarray,
    paths_remapped: list[np.ndarray],
    cycle_indices: np.ndarray | None,
    min1_idx: int | None,
    min2_idx: int | None,
    topology_type: str,
    clip_id: int,
    output_path: str,
    dpi: int = 150,
) -> None:
    fig = plt.figure(figsize=(12, 5))

    for plot_idx, (elev, azim, title_suffix) in enumerate(
        [(25, 45, "Perspective"), (90, 0, "Top-Down")]
    ):
        ax = fig.add_subplot(1, 2, plot_idx + 1, projection="3d")

        temporal_colors = np.arange(len(pts_3d)) / max(len(pts_3d) - 1, 1)
        ax.scatter(
            pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2],
            c=temporal_colors, cmap="viridis", s=4, alpha=0.3, edgecolors="none",
        )

        if cycle_indices is not None and len(cycle_indices) > 1:
            cycle_pts = pts_3d[cycle_indices]
            ax.plot(
                cycle_pts[:, 0], cycle_pts[:, 1], cycle_pts[:, 2],
                c="lightgray", linewidth=1.0, alpha=0.3, label="Cycle (faint)",
            )

        for i, path_idx in enumerate(paths_remapped):
            if len(path_idx) > 1:
                valid = [idx for idx in path_idx if idx < len(pts_3d)]
                if len(valid) > 1:
                    p = pts_3d[valid]
                    color = PATH_COLORS[i % len(PATH_COLORS)]
                    ax.plot(
                        p[:, 0], p[:, 1], p[:, 2],
                        c=color, linewidth=2.5, alpha=0.9,
                        label=f"Path {i+1} (L={len(valid)})",
                    )

        for min_idx in [min1_idx, min2_idx]:
            if min_idx is not None and min_idx < len(pts_3d):
                p = pts_3d[min_idx]
                ax.scatter([p[0]], [p[1]], [p[2]], c="gold", s=200, marker="*", zorder=200)

        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title_suffix, fontsize=10)
        if plot_idx == 0:
            ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

    fig.suptitle(
        f"Clip {clip_id} | {topology_type} | {len(paths_remapped)} edge-disjoint path(s)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved static 3D: {output_path}")


# ---------------------------------------------------------------------------
# Stage 5: Summary
# ---------------------------------------------------------------------------


def build_summary_entry(
    clip_id: int,
    dm_key: str,
    conn_result: ConnectivityResult | None,
    endpoint_result: EndpointSimilarityResult | None,
    gif_files: list[str],
    discovery_hit: RearWalkRearHit | None = None,
) -> dict:
    entry = {
        "clip_id": clip_id,
        "dm_key": dm_key,
    }

    if discovery_hit is not None:
        entry["discovery"] = {
            "score": round(discovery_hit.score, 6),
            "rear1_frames": list(discovery_hit.peak1_frames),
            "rear1_max_z": round(discovery_hit.peak1_max_z, 4),
            "walk_frames": list(discovery_hit.valley_frames),
            "walk_mean_z": round(discovery_hit.valley_mean_z, 4),
            "rear2_frames": list(discovery_hit.peak2_frames),
            "rear2_max_z": round(discovery_hit.peak2_max_z, 4),
        }

    if conn_result is not None:
        entry["topology_type"] = conn_result.topology_type
        entry["edge_connectivity"] = conn_result.edge_connectivity
        entry["n_edge_disjoint_paths"] = len(conn_result.edge_disjoint_paths)
        entry["path_lengths"] = [len(p) for p in conn_result.edge_disjoint_paths]
    else:
        entry["topology_type"] = "unknown"
        entry["edge_connectivity"] = 0
        entry["n_edge_disjoint_paths"] = 0
        entry["path_lengths"] = []

    if endpoint_result is not None:
        entry["endpoint_similarity"] = {
            "cosine_matrix": endpoint_result.similarity_matrix_cosine.tolist(),
            "l2_matrix": endpoint_result.similarity_matrix_l2.tolist(),
            "labels": endpoint_result.endpoint_labels,
        }
    else:
        entry["endpoint_similarity"] = None

    entry["gifs"] = gif_files
    return entry


def print_summary_table(results: list[dict]) -> str:
    header = (
        f"{'Clip':>5} {'Feature':>10} {'Topology':>12} "
        f"{'EdgeConn':>9} {'#Paths':>7} {'PathLens':>20}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        path_lens_str = str(r.get("path_lengths", []))
        line = (
            f"{r['clip_id']:>5} {r['dm_key']:>10} {r['topology_type']:>12} "
            f"{r['edge_connectivity']:>9} {r['n_edge_disjoint_paths']:>7} {path_lens_str:>20}"
        )
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Transition Topology Demo")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--skip-gifs", action="store_true", help="Skip GIF generation")
    parser.add_argument("--clip-ids", type=int, nargs="+", default=None)
    parser.add_argument("--top-k", type=int, default=None, help="Override clip_discovery.top_k")
    parser.add_argument(
        "--no-discover", action="store_true",
        help="Disable auto-discovery, use only clip_ids from config",
    )
    args = parser.parse_args()

    default_config_path = (
        Path(__file__).parent.parent / "configs" / "demos" / "demo_transition_topology.yaml"
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
    if not os.path.isabs(cfg.data.xml_path):
        cfg.data.xml_path = str(project_root / cfg.data.xml_path)
    if not os.path.isabs(cfg.data.stac_data_path):
        cfg.data.stac_data_path = str(project_root / cfg.data.stac_data_path)

    if args.top_k is not None:
        cfg.clip_discovery.top_k = args.top_k
    if args.no_discover:
        cfg.clip_discovery.enabled = False

    print("=" * 70)
    print("Transition Topology Demo")
    print("=" * 70)
    print(f"  Output dir: {output_dir}")
    print(f"  Feature spaces: {[fs.dm_key for fs in cfg.feature_spaces]}")
    print(f"  DM params: k={cfg.dm.k_neighbors}, epsilon={cfg.dm.epsilon}")
    print(f"  Skip GIFs: {args.skip_gifs}")

    discovery_hits: list[RearWalkRearHit] = []
    if args.clip_ids:
        clip_ids = args.clip_ids
        print(f"  Clip IDs (CLI override): {clip_ids}")
    elif cfg.clip_discovery.enabled:
        print(f"\n  Auto-discovering rear-walk-rear clips (top_k={cfg.clip_discovery.top_k})...")
        discovery_hits = discover_rear_walk_rear_clips(
            cfg.data.rodent_rollout_dir, cfg.clip_discovery
        )
        clip_ids = [h.clip_id for h in discovery_hits]
        print(f"  Discovered clip IDs: {clip_ids}")
    else:
        clip_ids = cfg.clip_ids
        print(f"  Clip IDs (config): {clip_ids}")

    discovery_map: dict[int, RearWalkRearHit] = {h.clip_id: h for h in discovery_hits}

    all_results = []
    rendered_clips: dict[int, list[np.ndarray]] = {}

    total_combos = len(clip_ids) * len(cfg.feature_spaces)
    combo_idx = 0

    for clip_id in clip_ids:
        print(f"\n{'='*70}")
        print(f"Clip {clip_id}")
        print(f"{'='*70}")

        if not args.skip_gifs and clip_id not in rendered_clips:
            print(f"  Rendering rodent frames for clip {clip_id}...")
            try:
                rendered_clips[clip_id] = _load_full_clip_frames(
                    cfg.data.rodent_rollout_dir, clip_id
                )
                print(f"  Rendered {len(rendered_clips[clip_id])} frames")
            except Exception as e:
                print(f"  WARNING: Failed to render clip {clip_id}: {e}")
                rendered_clips[clip_id] = []

        for fs_cfg in cfg.feature_spaces:
            combo_idx += 1
            print(
                f"\n  [{combo_idx}/{total_combos}] "
                f"{fs_cfg.dm_key} (PCA={fs_cfg.pca_dims}, "
                f"m={fs_cfg.takens_m}, tau={fs_cfg.takens_tau}, "
                f"pers_delta={fs_cfg.pers_delta})"
            )

            clip_output_dir = str(output_dir / f"clip_{clip_id}" / fs_cfg.dm_key)
            os.makedirs(clip_output_dir, exist_ok=True)

            try:
                # Stage 1: Load + DM extraction
                clip_data, var_explained = load_single_clip(
                    clip_id, fs_cfg.dm_key, cfg.data, fs_cfg.pca_dims
                )
                print(f"    Clip data shape: {clip_data.shape}")

                dm_result = run_dm_extraction(
                    clip_data, fs_cfg, cfg.dm, clip_output_dir
                )

                minima_result = find_dm_minima(dm_result)

                # Stage 2: Edge-connectivity analysis
                remap_to_orig = parse_swr_remap(dm_result.dm_output_dir)
                n_takens = len(dm_result.takens_points)

                conn_result = analyze_edge_connectivity(
                    dm_result.G_dm, minima_result, remap_to_orig, n_takens
                )

                # Prepare PCA projection
                pca_3d = PCA(n_components=3)
                pts_3d = pca_3d.fit_transform(dm_result.takens_points)

                # Remap minima for visualization
                min1_idx = None
                min2_idx = None
                if minima_result.min1_remapped is not None and minima_result.min1_remapped in remap_to_orig:
                    orig = remap_to_orig[minima_result.min1_remapped]
                    if orig < n_takens:
                        min1_idx = orig
                if minima_result.min2_remapped is not None and minima_result.min2_remapped in remap_to_orig:
                    orig = remap_to_orig[minima_result.min2_remapped]
                    if orig < n_takens:
                        min2_idx = orig

                # Get cycle indices for context
                cycle_indices = None
                if dm_result.cycles:
                    cycle = dm_result.cycles[0]
                    valid = remap_cycle_to_original(cycle, remap_to_orig, n_takens)
                    if len(valid) > 1:
                        cycle_indices = order_cycle_vertices(dm_result.takens_points, valid)

                # Stage 3: Endpoint similarity
                segments = []
                endpoint_result = None
                if cycle_indices is not None:
                    raw_segments = split_cycle_into_segments(
                        cycle_indices, gap_limit=cfg.visualization.segment_gap_limit
                    )
                    segments = [s for s in raw_segments if len(s) >= cfg.min_segment_frames]
                    if len(segments) < len(raw_segments):
                        print(f"    Filtered segments: {len(raw_segments)} -> {len(segments)} (min {cfg.min_segment_frames} frames)")
                    if segments:
                        hdf5_key = get_hdf5_key_for_dm_key(fs_cfg.dm_key)
                        intention_raw = get_aggregate_data(
                            "/intention", [], clip_id, cfg.data.rodent_rollout_dir
                        )
                        endpoint_result = compute_endpoint_similarity(
                            segments, intention_raw, fs_cfg.takens_m, fs_cfg.takens_tau
                        )
                        if endpoint_result is not None:
                            plot_endpoint_heatmap(
                                endpoint_result, clip_id,
                                os.path.join(clip_output_dir, "endpoint_similarity.png"),
                                dpi=cfg.visualization.dpi,
                            )

                # Static 3D plot
                paths_remapped = conn_result.paths_remapped if conn_result else []
                topology_type = conn_result.topology_type if conn_result else "unknown"

                plot_static_3d_paths(
                    pts_3d, paths_remapped, cycle_indices,
                    min1_idx, min2_idx, topology_type, clip_id,
                    os.path.join(clip_output_dir, "static_3d.png"),
                    dpi=cfg.visualization.dpi,
                )

                # Stage 4: GIF (single file with all paths)
                gif_files = []
                if not args.skip_gifs and rendered_clips.get(clip_id) and conn_result and paths_remapped:
                    valid_paths = [list(p) for p in paths_remapped if len(p) >= 2]
                    if valid_paths:
                        colors = [PATH_COLORS[i % len(PATH_COLORS)] for i in range(len(valid_paths))]
                        n_p = len(valid_paths)
                        label = f"{topology_type} — {n_p} path{'s' if n_p > 1 else ''}"

                        anim = TransitionPathAnimation(
                            pts_3d=pts_3d,
                            all_path_indices=valid_paths,
                            segments=segments,
                            cycle_indices=cycle_indices,
                            min1_idx=min1_idx,
                            min2_idx=min2_idx,
                            path_colors=colors,
                            topology_label=label,
                            rodent_frames=rendered_clips[clip_id],
                            takens_points=dm_result.takens_points,
                            m=fs_cfg.takens_m,
                            tau=fs_cfg.takens_tau,
                            endpoint_result=endpoint_result,
                            n_frames=cfg.visualization.gif_n_frames,
                            fps=cfg.visualization.gif_fps,
                            dpi=cfg.visualization.dpi,
                        )

                        gif_name = "paths.gif"
                        anim.save_gif(os.path.join(clip_output_dir, gif_name))
                        gif_files.append(gif_name)

                # Build summary entry
                all_results.append(
                    build_summary_entry(
                        clip_id, fs_cfg.dm_key, conn_result, endpoint_result,
                        gif_files, discovery_hit=discovery_map.get(clip_id),
                    )
                )

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "clip_id": clip_id,
                    "dm_key": fs_cfg.dm_key,
                    "topology_type": "error",
                    "edge_connectivity": -1,
                    "n_edge_disjoint_paths": -1,
                    "path_lengths": [],
                    "error": str(e),
                })

    # Summary output
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    table = print_summary_table(all_results)
    print(table)

    summary_path = str(output_dir / "summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump({"results": all_results}, f, default_flow_style=False, sort_keys=False)
    print(f"Saved summary YAML: {summary_path}")

    table_path = str(output_dir / "summary_table.txt")
    with open(table_path, "w") as f:
        f.write(table)
    print(f"Saved summary table: {table_path}")

    print(f"\nOutput saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
