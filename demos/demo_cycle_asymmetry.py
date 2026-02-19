#!/usr/bin/env python3
"""
Cycle Asymmetry Verification Demo
===================================
Tests the hypothesis that DM-G cycles arise from behavioral asymmetry:
when a rodent walks->rears->walks back differently (non-symmetric return),
the point cloud forms a loop. Symmetric behaviors should NOT produce cycles.

Also implements DM-path extraction: finding the shortest path between
the two most persistent density minima in the DM graph.

Usage:
    python demos/demo_cycle_asymmetry.py
    python demos/demo_cycle_asymmetry.py --config configs/demos/demo_cycle_asymmetry.yaml
    python demos/demo_cycle_asymmetry.py --skip-gifs
    python demos/demo_cycle_asymmetry.py --clip-sets cycle_candidates
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
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA
import networkx as nx
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, "/home/jovyan/vast/kaiwen/TopoVNL")
sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")

from topo_mimic.dmwk.skeletonization.skeletonization import (
    extract_skeleton_pcd_dm,
    extract_cycles_from_dm_graph,
)
from topo_mimic.dmwk.utils.feature_transforms import (
    apply_dm_key_transform,
    get_hdf5_key_for_dm_key,
)
from topo_mimic.dmwk.utils.load_utils import get_aggregate_data
from topo_mimic.dmwk.skeletonization.rendering import (
    _load_full_clip_frames,
    _ensure_rgb_uint8,
)


def takens_embedding(X: np.ndarray, m: int, tau: int) -> np.ndarray:
    n_frames, D = X.shape
    start = (m - 1) * tau
    idxs = np.arange(start, n_frames)
    pts = np.stack([X[[j - a * tau for a in range(m)]].reshape(-1) for j in idxs], axis=0)
    return pts


# ---------------------------------------------------------------------------
# Step 1: Config dataclasses + load_config()
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    rodent_rollout_dir: str = "data/rodent_ar_latest.h5"
    xml_path: str = "topo_mimic/dmwk/xml/rodent.xml"
    stac_data_path: str = "data/rodent_reference_clips.h5"


@dataclass
class FeatureSpaceConfig:
    dm_key: str = "intention"
    pca_dims: int | None = None
    takens_m: int = 3
    takens_tau: int = 6
    pers_delta: float = 0.75


@dataclass
class ClipSetConfig:
    name: str = "cycle_candidates"
    clip_ids: list[int] = field(default_factory=list)
    expect_cycles: bool = True


@dataclass
class DMConfig:
    k_neighbors: int = 3
    epsilon: float = 0.99


@dataclass
class VisualizationConfig:
    dpi: int = 150
    gif_n_frames: int = 120
    gif_fps: int = 15


@dataclass
class Config:
    output_dir: str = "out/cycle_asymmetry"
    data: DataConfig = field(default_factory=DataConfig)
    feature_spaces: list[FeatureSpaceConfig] = field(default_factory=list)
    clip_sets: list[ClipSetConfig] = field(default_factory=list)
    dm: DMConfig = field(default_factory=DMConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    data_cfg = DataConfig(**cfg_dict.get("data", {}))

    feature_spaces = []
    for fs_dict in cfg_dict.get("feature_spaces", []):
        feature_spaces.append(FeatureSpaceConfig(**fs_dict))

    clip_sets = []
    for cs_dict in cfg_dict.get("clip_sets", []):
        clip_sets.append(ClipSetConfig(**cs_dict))

    dm_cfg = DMConfig(**cfg_dict.get("dm", {}))

    vis_cfg = VisualizationConfig(**cfg_dict.get("visualization", {}))

    return Config(
        output_dir=cfg_dict.get("output_dir", "out/cycle_asymmetry"),
        data=data_cfg,
        feature_spaces=feature_spaces,
        clip_sets=clip_sets,
        dm=dm_cfg,
        visualization=vis_cfg,
    )


# ---------------------------------------------------------------------------
# Step 2: Single-clip data loader
# ---------------------------------------------------------------------------


def load_single_clip(
    clip_idx: int,
    dm_key: str,
    data_config: DataConfig,
    pca_dims: int | None = None,
) -> tuple[np.ndarray, float | None]:
    """Load and transform data for a single clip.

    Args:
        clip_idx: Clip index in the HDF5 file
        dm_key: Feature space key ("intention", "qpos", or "ekp")
        data_config: Data configuration with paths
        pca_dims: If set, reduce dimensionality with PCA after transform

    Returns:
        Tuple of (clip_data shaped (n_frames, D), pca_variance_explained or None)
    """
    hdf5_key = get_hdf5_key_for_dm_key(dm_key)
    raw = get_aggregate_data(f"/{hdf5_key}", [], clip_idx, data_config.rodent_rollout_dir)

    raw_batch = raw[np.newaxis, ...]
    transformed = apply_dm_key_transform(
        raw_batch,
        dm_key,
        xml_path=data_config.xml_path,
        stac_data_path=data_config.stac_data_path,
    )
    clip_data = transformed[0]

    var_explained = None
    if pca_dims is not None and pca_dims < clip_data.shape[1]:
        pca = PCA(n_components=pca_dims)
        clip_data = pca.fit_transform(clip_data)
        var_explained = float(pca.explained_variance_ratio_.sum())

    return clip_data, var_explained


# ---------------------------------------------------------------------------
# Step 3: DM-G extraction per clip
# ---------------------------------------------------------------------------


@dataclass
class DMResult:
    cycles: list[np.ndarray]
    G_dm: nx.Graph
    takens_points: np.ndarray
    dm_output_dir: str
    n_dm_edges: int
    n_cycles: int
    longest_cycle_len: int


def run_dm_extraction(
    clip_data: np.ndarray,
    fs_cfg: FeatureSpaceConfig,
    dm_cfg: DMConfig,
    output_dir: str,
) -> DMResult:
    """Run Takens embedding + DM-G extraction for a single clip/feature space."""
    pts = takens_embedding(clip_data, fs_cfg.takens_m, fs_cfg.takens_tau)
    print(f"    Takens embedding: {clip_data.shape} -> {pts.shape}")

    os.makedirs(output_dir, exist_ok=True)
    cycles, G_dm = extract_skeleton_pcd_dm(
        points=pts,
        k=dm_cfg.k_neighbors,
        pers_delta=fs_cfg.pers_delta,
        epsilon=dm_cfg.epsilon,
        output_dir=output_dir,
        cleanup=False,
    )

    if cycles:
        cycles = sorted(cycles, key=len, reverse=True)

    n_edges = G_dm.number_of_edges()
    n_cycles = len(cycles)
    longest = len(cycles[0]) if cycles else 0

    print(f"    DM graph: {G_dm.number_of_nodes()} nodes, {n_edges} edges")
    print(f"    Cycles: {n_cycles}, longest: {longest}")

    return DMResult(
        cycles=cycles,
        G_dm=G_dm,
        takens_points=pts,
        dm_output_dir=output_dir,
        n_dm_edges=n_edges,
        n_cycles=n_cycles,
        longest_cycle_len=longest,
    )


# ---------------------------------------------------------------------------
# Step 4: DM-Path extraction
# ---------------------------------------------------------------------------


@dataclass
class DMPathResult:
    path_indices_remapped: list[int] | None
    min1_remapped: int | None
    min2_remapped: int | None
    min1_orig: int | None
    min2_orig: int | None
    path_length: int
    persistence_pairs: list[tuple]


def parse_swr_filtration(output_dir: str) -> dict[int, int]:
    """Parse sparse_weighted_rips_filtration.txt to build remap_to_orig mapping."""
    filt_path = os.path.join(output_dir, "sparse_weighted_rips_filtration.txt")
    remap_to_orig = {}
    vertex_count = 0

    with open(filt_path) as f:
        for line in f:
            parts = line.strip().split()
            dim = int(parts[0])
            if dim == 0:
                orig_idx = int(parts[1])
                remap_to_orig[vertex_count] = orig_idx
                vertex_count += 1

    return remap_to_orig


def parse_sorted_weights(output_dir: str) -> dict[int, float]:
    """Parse sorted-weights.txt to get weight per remapped vertex."""
    weights_path = os.path.join(output_dir, "sorted-weights.txt")
    weights = {}
    with open(weights_path) as f:
        for i, line in enumerate(f):
            weights[i] = float(line.strip())
    return weights


def parse_persistence_pairs(output_dir: str) -> list[tuple[int, int, int, float]]:
    """Parse edge_for_morse_only.txt.

    Returns:
        List of (v0, v1, type, persistence) tuples.
        type 0: min-saddle pairs
        type 1: saddle-max pairs
        type 2: unpaired
    """
    edge_path = os.path.join(output_dir, "edge_for_morse_only.txt")
    pairs = []

    with open(edge_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            v0 = int(parts[0])
            v1 = int(parts[1])
            pair_type = int(parts[2])
            pers_str = parts[3]
            pers = float("inf") if pers_str == "inf" else float(pers_str)
            pairs.append((v0, v1, pair_type, pers))

    return pairs


def find_persistent_minima(
    pairs: list[tuple[int, int, int, float]],
    weights: dict[int, float],
    top_k: int = 2,
) -> list[int]:
    """Find the top-k most persistent density minima from type-0 pairs.

    For each type-0 (min-saddle) pair, the minimum vertex is the endpoint
    with lower weight (denser region).
    """
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


def extract_dm_path(
    G_dm: nx.Graph,
    min1: int,
    min2: int,
) -> list[int] | None:
    """Find shortest path between two minima in the DM graph."""
    try:
        path = nx.shortest_path(G_dm, min1, min2)
        return list(path)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def run_dm_path_extraction(
    dm_result: DMResult,
) -> DMPathResult:
    """Extract DM-path between two most persistent minima."""
    output_dir = dm_result.dm_output_dir

    pairs_path = os.path.join(output_dir, "edge_for_morse_only.txt")
    weights_path = os.path.join(output_dir, "sorted-weights.txt")

    if not os.path.exists(pairs_path) or not os.path.exists(weights_path):
        print("    DM-path: persistence files not found, skipping")
        return DMPathResult(
            path_indices_remapped=None,
            min1_remapped=None,
            min2_remapped=None,
            min1_orig=None,
            min2_orig=None,
            path_length=0,
            persistence_pairs=[],
        )

    pairs = parse_persistence_pairs(output_dir)
    weights = parse_sorted_weights(output_dir)
    remap_to_orig = parse_swr_filtration(output_dir)

    minima = find_persistent_minima(pairs, weights, top_k=2)

    if len(minima) < 2:
        print(f"    DM-path: found only {len(minima)} minima, need 2")
        return DMPathResult(
            path_indices_remapped=None,
            min1_remapped=minima[0] if minima else None,
            min2_remapped=None,
            min1_orig=remap_to_orig.get(minima[0]) if minima else None,
            min2_orig=None,
            path_length=0,
            persistence_pairs=pairs,
        )

    min1, min2 = minima[0], minima[1]
    path = extract_dm_path(dm_result.G_dm, min1, min2)

    path_length = len(path) if path else 0
    min1_orig = remap_to_orig.get(min1)
    min2_orig = remap_to_orig.get(min2)

    if path:
        print(f"    DM-path: {min1}({min1_orig}) -> {min2}({min2_orig}), length={path_length}")
    else:
        print(f"    DM-path: no path found between minima {min1} and {min2}")

    return DMPathResult(
        path_indices_remapped=path,
        min1_remapped=min1,
        min2_remapped=min2,
        min1_orig=min1_orig,
        min2_orig=min2_orig,
        path_length=path_length,
        persistence_pairs=pairs,
    )


# ---------------------------------------------------------------------------
# Step 5: 3D PCA visualization (static)
# ---------------------------------------------------------------------------


def remap_indices_to_original(
    indices: list[int] | np.ndarray,
    remap_to_orig: dict[int, int],
) -> list[int]:
    """Convert remapped DM indices to original Takens point indices."""
    return [remap_to_orig[int(i)] for i in indices if int(i) in remap_to_orig]


def takens_index_to_frame(
    takens_idx: int,
    m: int,
    tau: int,
) -> int:
    """Convert Takens point index to original frame index."""
    return (m - 1) * tau + takens_idx


def plot_static_3d(
    dm_result: DMResult,
    path_result: DMPathResult,
    fs_cfg: FeatureSpaceConfig,
    clip_id: int,
    clip_set_name: str,
    output_path: str,
    dpi: int = 150,
) -> None:
    """Create static 3D PCA scatter plot with cycle and DM-path highlighted."""
    pts = dm_result.takens_points
    pca_3d = PCA(n_components=3)
    pts_3d = pca_3d.fit_transform(pts)

    remap_to_orig = parse_swr_filtration(dm_result.dm_output_dir)

    fig = plt.figure(figsize=(12, 5))

    for plot_idx, (elev, azim, title_suffix) in enumerate(
        [(25, 45, "Perspective"), (90, 0, "Top-Down")]
    ):
        ax = fig.add_subplot(1, 2, plot_idx + 1, projection="3d")

        temporal_colors = np.arange(len(pts_3d)) / max(len(pts_3d) - 1, 1)
        ax.scatter(
            pts_3d[:, 0],
            pts_3d[:, 1],
            pts_3d[:, 2],
            c=temporal_colors,
            cmap="viridis",
            s=4,
            alpha=0.3,
            edgecolors="none",
        )

        if dm_result.cycles:
            cycle = dm_result.cycles[0]
            cycle_orig = remap_indices_to_original(cycle, remap_to_orig)
            valid = [i for i in cycle_orig if i < len(pts_3d)]
            if len(valid) > 1:
                cycle_pts = pts_3d[valid]
                ax.plot(
                    cycle_pts[:, 0],
                    cycle_pts[:, 1],
                    cycle_pts[:, 2],
                    c="red",
                    linewidth=2.5,
                    alpha=0.9,
                    label=f"Cycle (L={len(valid)})",
                )

        if path_result.path_indices_remapped:
            path_orig = remap_indices_to_original(
                path_result.path_indices_remapped, remap_to_orig
            )
            valid = [i for i in path_orig if i < len(pts_3d)]
            if len(valid) > 1:
                path_pts = pts_3d[valid]
                ax.plot(
                    path_pts[:, 0],
                    path_pts[:, 1],
                    path_pts[:, 2],
                    c="dodgerblue",
                    linewidth=2.5,
                    alpha=0.9,
                    label=f"DM-path (L={len(valid)})",
                )

        if path_result.min1_orig is not None and path_result.min1_orig < len(pts_3d):
            p = pts_3d[path_result.min1_orig]
            ax.scatter([p[0]], [p[1]], [p[2]], c="gold", s=200, marker="*", zorder=200)
        if path_result.min2_orig is not None and path_result.min2_orig < len(pts_3d):
            p = pts_3d[path_result.min2_orig]
            ax.scatter([p[0]], [p[1]], [p[2]], c="gold", s=200, marker="*", zorder=200)

        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title_suffix, fontsize=10)
        if plot_idx == 0:
            ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

    fig.suptitle(
        f"Clip {clip_id} | {fs_cfg.dm_key} | {clip_set_name}\n"
        f"Cycles: {dm_result.n_cycles} | DM-path len: {path_result.path_length}",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved static 3D: {output_path}")


# ---------------------------------------------------------------------------
# Step 6: Rotating GIF with rodent rendering
# ---------------------------------------------------------------------------


class CycleAsymmetryAnimation:
    """Side-by-side rotating 3D PCA + rodent rendering animation."""

    def __init__(
        self,
        pts_3d: np.ndarray,
        trace_indices_orig: list[int],
        trace_color: str,
        trace_label: str,
        rodent_frames: list[np.ndarray],
        m: int,
        tau: int,
        n_frames: int = 120,
        fps: int = 15,
        dpi: int = 150,
    ):
        self.pts_3d = pts_3d
        self.trace_indices_orig = trace_indices_orig
        self.trace_color = trace_color
        self.trace_label = trace_label
        self.rodent_frames = rodent_frames
        self.m = m
        self.tau = tau
        self.n_frames = n_frames
        self.fps = fps
        self.dpi = dpi

        self.frame_indices = [
            takens_index_to_frame(i, m, tau) for i in trace_indices_orig
        ]
        self.frame_indices = [
            fi for fi in self.frame_indices if fi < len(rodent_frames)
        ]

        self._setup_figure()

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(14, 6), facecolor="white")
        self.ax_3d = self.fig.add_subplot(121, projection="3d", facecolor="white")
        self.ax_img = self.fig.add_subplot(122)
        self.ax_img.set_axis_off()
        self.fig.subplots_adjust(wspace=0.05)

    def animate(self, frame: int):
        self.ax_3d.clear()
        self.ax_img.clear()
        self.ax_img.set_axis_off()

        azimuth = (frame / self.n_frames) * 360
        self.ax_3d.view_init(elev=25, azim=azimuth)

        self.ax_3d.scatter(
            self.pts_3d[:, 0],
            self.pts_3d[:, 1],
            self.pts_3d[:, 2],
            c="gray",
            s=3,
            alpha=0.15,
            edgecolors="none",
        )

        progress = frame / max(self.n_frames - 1, 1)
        n_trace = max(1, int(len(self.trace_indices_orig) * progress))
        shown = self.trace_indices_orig[:n_trace]
        valid = [i for i in shown if i < len(self.pts_3d)]

        if len(valid) > 1:
            trace_pts = self.pts_3d[valid]
            self.ax_3d.plot(
                trace_pts[:, 0],
                trace_pts[:, 1],
                trace_pts[:, 2],
                c=self.trace_color,
                linewidth=2.5,
                alpha=0.9,
            )

            head = trace_pts[-1]
            self.ax_3d.scatter(
                [head[0]],
                [head[1]],
                [head[2]],
                c="yellow",
                s=100,
                edgecolors=self.trace_color,
                linewidths=2,
                zorder=200,
            )

        self.ax_3d.set_axis_off()
        self.ax_3d.set_title(f"{self.trace_label}", fontsize=10)

        fi_n = max(1, int(len(self.frame_indices) * progress))
        if self.frame_indices:
            current_frame_idx = self.frame_indices[min(fi_n - 1, len(self.frame_indices) - 1)]
            self.ax_img.imshow(self.rodent_frames[current_frame_idx])
            self.ax_img.set_title(f"Frame {current_frame_idx}", fontsize=10)
        else:
            self.ax_img.set_title("No frames", fontsize=10)

        return []

    def save_gif(self, output_path: str):
        anim = FuncAnimation(
            self.fig,
            self.animate,
            frames=self.n_frames,
            interval=1000 // self.fps,
            blit=False,
        )
        anim.save(output_path, writer="pillow", fps=self.fps, dpi=self.dpi)
        plt.close(self.fig)
        print(f"    Saved GIF: {output_path}")


def create_gifs(
    dm_result: DMResult,
    path_result: DMPathResult,
    fs_cfg: FeatureSpaceConfig,
    rodent_frames: list[np.ndarray],
    output_dir: str,
    vis_cfg: VisualizationConfig,
) -> None:
    """Create rotating GIFs for cycle and DM-path traces."""
    pts = dm_result.takens_points
    pca_3d = PCA(n_components=3)
    pts_3d = pca_3d.fit_transform(pts)

    remap_to_orig = parse_swr_filtration(dm_result.dm_output_dir)

    if dm_result.cycles:
        cycle = dm_result.cycles[0]
        cycle_orig = remap_indices_to_original(cycle, remap_to_orig)
        valid = [i for i in cycle_orig if i < len(pts_3d)]
        if len(valid) > 1:
            anim = CycleAsymmetryAnimation(
                pts_3d=pts_3d,
                trace_indices_orig=valid,
                trace_color="red",
                trace_label=f"Cycle ({fs_cfg.dm_key})",
                rodent_frames=rodent_frames,
                m=fs_cfg.takens_m,
                tau=fs_cfg.takens_tau,
                n_frames=vis_cfg.gif_n_frames,
                fps=vis_cfg.gif_fps,
                dpi=vis_cfg.dpi,
            )
            anim.save_gif(os.path.join(output_dir, "cycle_anim.gif"))

    if path_result.path_indices_remapped:
        path_orig = remap_indices_to_original(
            path_result.path_indices_remapped, remap_to_orig
        )
        valid = [i for i in path_orig if i < len(pts_3d)]
        if len(valid) > 1:
            anim = CycleAsymmetryAnimation(
                pts_3d=pts_3d,
                trace_indices_orig=valid,
                trace_color="dodgerblue",
                trace_label=f"DM-path ({fs_cfg.dm_key})",
                rodent_frames=rodent_frames,
                m=fs_cfg.takens_m,
                tau=fs_cfg.takens_tau,
                n_frames=vis_cfg.gif_n_frames,
                fps=vis_cfg.gif_fps,
                dpi=vis_cfg.dpi,
            )
            anim.save_gif(os.path.join(output_dir, "path_anim.gif"))


# ---------------------------------------------------------------------------
# Step 7: Summary table + results
# ---------------------------------------------------------------------------


def print_summary_table(results: list[dict]) -> str:
    """Print and return a formatted summary table."""
    header = (
        f"{'Set':<20} {'Clip':>5} {'Feature':>10} {'Expect':>7} "
        f"{'Cycles':>7} {'LongestC':>9} {'PathLen':>8} {'Match':>6}"
    )
    sep = "-" * len(header)

    lines = [sep, header, sep]
    for r in results:
        expect_str = "cycle" if r["expect_cycles"] else "none"
        has_cycle = r["n_cycles"] > 0
        match = (has_cycle == r["expect_cycles"])
        match_str = "OK" if match else "MISS"

        line = (
            f"{r['clip_set']:<20} {r['clip_id']:>5} {r['dm_key']:>10} {expect_str:>7} "
            f"{r['n_cycles']:>7} {r['longest_cycle']:>9} {r['path_length']:>8} {match_str:>6}"
        )
        lines.append(line)

    lines.append(sep)
    table = "\n".join(lines)
    return table


def save_summary_yaml(results: list[dict], output_path: str) -> None:
    summary = {"results": results}
    with open(output_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    print(f"Saved summary YAML: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Cycle Asymmetry Verification Demo")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--skip-gifs", action="store_true", help="Skip GIF generation")
    parser.add_argument(
        "--clip-sets",
        type=str,
        nargs="+",
        default=None,
        help="Only run specific clip sets by name",
    )
    parser.add_argument(
        "--feature-spaces",
        type=str,
        nargs="+",
        default=None,
        help="Only run specific feature spaces by dm_key",
    )
    args = parser.parse_args()

    default_config_path = (
        Path(__file__).parent.parent / "configs" / "demos" / "demo_cycle_asymmetry.yaml"
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

    # Resolve data paths to absolute before any DM extraction
    # (extract_skeleton_pcd_dm does os.chdir internally, breaking relative paths)
    if not os.path.isabs(cfg.data.rodent_rollout_dir):
        cfg.data.rodent_rollout_dir = str(project_root / cfg.data.rodent_rollout_dir)
    if not os.path.isabs(cfg.data.xml_path):
        cfg.data.xml_path = str(project_root / cfg.data.xml_path)
    if not os.path.isabs(cfg.data.stac_data_path):
        cfg.data.stac_data_path = str(project_root / cfg.data.stac_data_path)

    if args.clip_sets:
        cfg.clip_sets = [cs for cs in cfg.clip_sets if cs.name in args.clip_sets]
    if args.feature_spaces:
        cfg.feature_spaces = [
            fs for fs in cfg.feature_spaces if fs.dm_key in args.feature_spaces
        ]

    print("=" * 70)
    print("Cycle Asymmetry Verification Demo")
    print("=" * 70)
    print(f"  Output dir: {output_dir}")
    print(f"  Feature spaces: {[fs.dm_key for fs in cfg.feature_spaces]}")
    print(f"  Clip sets: {[cs.name for cs in cfg.clip_sets]}")
    print(f"  DM params: k={cfg.dm.k_neighbors}, epsilon={cfg.dm.epsilon}")
    print(f"  Skip GIFs: {args.skip_gifs}")

    all_results = []
    rendered_clips: dict[int, list[np.ndarray]] = {}

    total_combos = sum(
        len(cs.clip_ids) * len(cfg.feature_spaces) for cs in cfg.clip_sets
    )
    combo_idx = 0

    for cs_cfg in cfg.clip_sets:
        print(f"\n{'='*70}")
        print(f"Clip set: {cs_cfg.name} (expect_cycles={cs_cfg.expect_cycles})")
        print(f"{'='*70}")

        for clip_id in cs_cfg.clip_ids:
            print(f"\n  Clip {clip_id}:")

            if not args.skip_gifs and clip_id not in rendered_clips:
                print(f"    Rendering rodent frames for clip {clip_id}...")
                try:
                    rendered_clips[clip_id] = _load_full_clip_frames(
                        cfg.data.rodent_rollout_dir, clip_id
                    )
                    print(f"    Rendered {len(rendered_clips[clip_id])} frames")
                except Exception as e:
                    print(f"    WARNING: Failed to render clip {clip_id}: {e}")
                    rendered_clips[clip_id] = []

            for fs_cfg in cfg.feature_spaces:
                combo_idx += 1
                print(
                    f"\n    [{combo_idx}/{total_combos}] "
                    f"{fs_cfg.dm_key} (PCA={fs_cfg.pca_dims}, "
                    f"m={fs_cfg.takens_m}, tau={fs_cfg.takens_tau}, "
                    f"pers_delta={fs_cfg.pers_delta})"
                )

                clip_output_dir = str(
                    output_dir / cs_cfg.name / f"clip_{clip_id}" / fs_cfg.dm_key
                )
                os.makedirs(clip_output_dir, exist_ok=True)

                try:
                    clip_data, var_explained = load_single_clip(
                        clip_id, fs_cfg.dm_key, cfg.data, fs_cfg.pca_dims
                    )
                    if var_explained is not None:
                        print(f"    PCA variance explained: {var_explained:.3f}")
                    print(f"    Clip data shape: {clip_data.shape}")

                    dm_result = run_dm_extraction(
                        clip_data, fs_cfg, cfg.dm, clip_output_dir
                    )

                    path_result = run_dm_path_extraction(dm_result)

                    plot_static_3d(
                        dm_result,
                        path_result,
                        fs_cfg,
                        clip_id,
                        cs_cfg.name,
                        os.path.join(clip_output_dir, "static_3d.png"),
                        dpi=cfg.visualization.dpi,
                    )

                    if not args.skip_gifs and rendered_clips.get(clip_id):
                        create_gifs(
                            dm_result,
                            path_result,
                            fs_cfg,
                            rendered_clips[clip_id],
                            clip_output_dir,
                            cfg.visualization,
                        )

                    all_results.append(
                        {
                            "clip_set": cs_cfg.name,
                            "clip_id": clip_id,
                            "dm_key": fs_cfg.dm_key,
                            "expect_cycles": cs_cfg.expect_cycles,
                            "n_cycles": dm_result.n_cycles,
                            "longest_cycle": dm_result.longest_cycle_len,
                            "path_length": path_result.path_length,
                            "n_dm_edges": dm_result.n_dm_edges,
                            "pca_dims": fs_cfg.pca_dims,
                            "pca_var_explained": var_explained,
                            "takens_m": fs_cfg.takens_m,
                            "takens_tau": fs_cfg.takens_tau,
                            "pers_delta": fs_cfg.pers_delta,
                        }
                    )

                except Exception as e:
                    print(f"    ERROR: {e}")
                    import traceback

                    traceback.print_exc()
                    all_results.append(
                        {
                            "clip_set": cs_cfg.name,
                            "clip_id": clip_id,
                            "dm_key": fs_cfg.dm_key,
                            "expect_cycles": cs_cfg.expect_cycles,
                            "n_cycles": -1,
                            "longest_cycle": -1,
                            "path_length": -1,
                            "n_dm_edges": -1,
                            "pca_dims": fs_cfg.pca_dims,
                            "pca_var_explained": None,
                            "takens_m": fs_cfg.takens_m,
                            "takens_tau": fs_cfg.takens_tau,
                            "pers_delta": fs_cfg.pers_delta,
                            "error": str(e),
                        }
                    )

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    table = print_summary_table(all_results)
    print(table)

    save_summary_yaml(all_results, str(output_dir / "summary.yaml"))

    table_path = str(output_dir / "summary_table.txt")
    with open(table_path, "w") as f:
        f.write(table)
    print(f"Saved summary table: {table_path}")

    n_matches = sum(
        1
        for r in all_results
        if r["n_cycles"] >= 0
        and ((r["n_cycles"] > 0) == r["expect_cycles"])
    )
    n_valid = sum(1 for r in all_results if r["n_cycles"] >= 0)
    print(f"\nVerification: {n_matches}/{n_valid} match expectations")
    print(f"Output saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
