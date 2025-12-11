"""
Benchmark clustering pipeline.
Compares multiple clustering methods using Dunn Index for fair evaluation.

Follows the same structure as dm_latent_subspace_auto_dunn.py but benchmarks
multiple methods instead of just W2-based clustering.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import functools
import multiprocessing as mp
import pickle as pkl
import yaml
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import pdist, cdist

from vnl_tda.utils.load_utils import get_aggregate_data
from vnl_tda.pipelines.tda.skeletonization.postprocess import (
    _expand_frames_nonconsec,
    _uniform_downsample,
    chunk_loops_by_clustered_vpath,
)
from vnl_tda.pipelines.tda.skeletonization.rendering import render_grid_clusters_side_by_side

from benchmark.clustering_methods import get_clustering_method


@dataclass
class DMConfig:
    results_dir: str = ""


@dataclass
class ClusteringConfig:
    min_len: int = 0
    max_len: int = 500
    target_len: int = 70
    only_max: bool = True
    only_min: bool = False
    skip_clips: tuple[int, ...] = (37, 152, 163, 274, 281)


@dataclass
class AutoClusterConfig:
    k_min: int = 5
    k_max: int = 20
    joint_state_key: str = "qpos"
    plot_dunn: bool = True


@dataclass
class MethodConfig:
    name: str
    enabled: bool = True
    params: dict = field(default_factory=dict)


@dataclass
class RenderingConfig:
    pad: int = 0
    max_samples: int = 10
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)


@dataclass
class BenchmarkConfig:
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    output_dir: str = "out_benchmark"
    num_clips: int = 500
    dm: DMConfig = None
    clustering: ClusteringConfig = None
    auto_cluster: AutoClusterConfig = None
    methods: list[MethodConfig] = None
    rendering: RenderingConfig = None

    def __post_init__(self):
        if self.dm is None:
            self.dm = DMConfig()
        if self.clustering is None:
            self.clustering = ClusteringConfig()
        if self.auto_cluster is None:
            self.auto_cluster = AutoClusterConfig()
        if self.methods is None:
            self.methods = []
        if self.rendering is None:
            self.rendering = RenderingConfig()


def load_config(config_path: str) -> BenchmarkConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dm_cfg = DMConfig(**cfg.get("dm", {}))

    clustering_dict = cfg.get("clustering", {})
    if "skip_clips" in clustering_dict:
        clustering_dict["skip_clips"] = tuple(clustering_dict["skip_clips"])
    clustering_cfg = ClusteringConfig(**clustering_dict)

    auto_cluster_cfg = AutoClusterConfig(**cfg.get("auto_cluster", {}))

    methods_list = []
    for m in cfg.get("methods", []):
        methods_list.append(MethodConfig(
            name=m["name"],
            enabled=m.get("enabled", True),
            params=m.get("params", {}),
        ))

    rendering_dict = cfg.get("rendering", {})
    if "target_size" in rendering_dict:
        rendering_dict["target_size"] = tuple(rendering_dict["target_size"])
    rendering_cfg = RenderingConfig(**rendering_dict)

    return BenchmarkConfig(
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", BenchmarkConfig.rodent_rollout_dir),
        output_dir=cfg.get("output_dir", BenchmarkConfig.output_dir),
        num_clips=cfg.get("num_clips", BenchmarkConfig.num_clips),
        dm=dm_cfg,
        clustering=clustering_cfg,
        auto_cluster=auto_cluster_cfg,
        methods=methods_list,
        rendering=rendering_cfg,
    )


def prepare_windows_for_clustering(
    all_cycle_dm: list,
    intention: np.ndarray,
    clustering_cfg: ClusteringConfig,
) -> tuple[list[tuple[int, np.ndarray]], list[np.ndarray]]:
    """
    Prepare windows from cycles for clustering.

    Args:
        all_cycle_dm: List of cycles per clip
        intention: Intention array (n_clips, n_frames, D)
        clustering_cfg: Clustering configuration

    Returns:
        Tuple of (windows, cleaned_loops)
    """
    windows = []
    cleaned_loops = []
    n_clips = len(intention)

    for clip_id in range(n_clips):
        if clip_id in clustering_cfg.skip_clips:
            continue
        if clip_id >= len(all_cycle_dm):
            continue

        loops = [
            loop for loop in all_cycle_dm[clip_id]
            if clustering_cfg.min_len <= len(loop) <= clustering_cfg.max_len
        ]

        if clustering_cfg.only_max and len(loops) > 0:
            loops = [max(loops, key=len)]
        elif clustering_cfg.only_min and len(loops) > 0:
            loops = [min(loops, key=len)]

        clip_len = len(intention[clip_id])
        for loop in loops:
            expanded = _expand_frames_nonconsec(loop, pad=0, clip_len=clip_len)
            sampled = _uniform_downsample(expanded, clustering_cfg.target_len)
            if sampled is None:
                continue
            windows.append((clip_id, sampled))
            cleaned_loops.append(np.asarray(loop, dtype=int))

    if (clustering_cfg.only_max or clustering_cfg.only_min) and cleaned_loops:
        loop_lengths = np.array([len(loop) for loop in cleaned_loops])
        mode = "only_max" if clustering_cfg.only_max else "only_min"
        print(f"[{mode}] Loop length stats: mean={loop_lengths.mean():.2f}, std={loop_lengths.std():.2f}, n={len(loop_lengths)}")

    return windows, cleaned_loops


def build_feature_matrix(
    windows: list[tuple[int, np.ndarray]],
    intention_data: np.ndarray,
) -> np.ndarray:
    """
    Build feature matrix from windows for clustering.

    Args:
        windows: List of (clip_id, sampled_frames) tuples
        intention_data: Intention array (n_clips, n_frames, D)

    Returns:
        Feature matrix (n_samples, n_features)
    """
    features = []
    for clip_id, frames in windows:
        traj = intention_data[clip_id, frames].flatten()
        features.append(traj)
    return np.array(features)


def prepare_joint_trajectories(
    windows: list[tuple[int, np.ndarray]],
    joint_data: np.ndarray,
    target_len: int,
) -> np.ndarray:
    """
    Prepare flattened joint trajectories for Dunn computation.

    Args:
        windows: List of (clip_id, sampled_frames) tuples
        joint_data: Joint space array (n_clips, n_frames, D)
        target_len: Target length for uniform downsampling

    Returns:
        Flattened trajectories array (n_samples, n_features)
    """
    trajectories = []
    for clip_id, frames in windows:
        frames = np.asarray(frames, dtype=int)
        sampled = _uniform_downsample(frames, target_len)
        if sampled is None:
            sampled = frames[:target_len] if len(frames) >= target_len else frames
        traj = joint_data[clip_id, sampled].flatten()
        trajectories.append(traj)

    return np.array(trajectories)


def compute_dunn_index(
    labels: np.ndarray,
    joint_trajectories: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute Dunn Index for a clustering.

    Dunn = min(inter-cluster distance) / max(intra-cluster diameter)

    Args:
        labels: Cluster labels array (n_samples,)
        joint_trajectories: Flattened joint trajectories (n_samples, n_features)

    Returns:
        Tuple of (dunn_index, min_inter_dist, max_intra_diam)
    """
    unique_labels = [l for l in np.unique(labels) if l >= 0]
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return 0.0, 0.0, 0.0

    cluster_data = {}
    for lbl in unique_labels:
        mask = labels == lbl
        cluster_data[lbl] = joint_trajectories[mask]

    cluster_diameters = []
    for lbl in unique_labels:
        X = cluster_data[lbl]
        if len(X) < 2:
            cluster_diameters.append(0.0)
        else:
            pairwise = pdist(X, metric='euclidean')
            diameter = pairwise.max() if len(pairwise) > 0 else 0.0
            cluster_diameters.append(diameter)

    max_intra_diameter = max(cluster_diameters) if cluster_diameters else 0.0

    min_inter_distance = np.inf
    for i, lbl_i in enumerate(unique_labels):
        for j, lbl_j in enumerate(unique_labels):
            if i >= j:
                continue
            X_i = cluster_data[lbl_i]
            X_j = cluster_data[lbl_j]
            if len(X_i) == 0 or len(X_j) == 0:
                continue
            inter_dists = cdist(X_i, X_j, metric='euclidean')
            min_dist = inter_dists.min()
            min_inter_distance = min(min_inter_distance, min_dist)

    if min_inter_distance == np.inf:
        min_inter_distance = 0.0

    if max_intra_diameter < 1e-10:
        dunn_index = 0.0
    else:
        dunn_index = min_inter_distance / max_intra_diameter

    return dunn_index, min_inter_distance, max_intra_diameter


def compute_per_cluster_stats(
    labels: np.ndarray,
    joint_trajectories: np.ndarray,
) -> dict:
    """Compute per-cluster statistics (diameter, size)."""
    unique_labels = [l for l in np.unique(labels) if l >= 0]
    stats = {}

    for lbl in unique_labels:
        mask = labels == lbl
        X = joint_trajectories[mask]
        n_points = len(X)

        if n_points < 2:
            diameter = 0.0
        else:
            pairwise = pdist(X, metric='euclidean')
            diameter = pairwise.max() if len(pairwise) > 0 else 0.0

        stats[int(lbl)] = {
            "size": n_points,
            "diameter": diameter,
        }

    return stats


def labels_to_cluster_dict(
    labels: np.ndarray,
    windows: list[tuple[int, np.ndarray]],
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Convert labels array to cluster dictionary format."""
    cluster_dict = {}
    for lbl, win in zip(labels, windows):
        lbl = int(lbl)
        if lbl < 0:
            continue
        cluster_dict.setdefault(lbl, []).append(win)
    return cluster_dict


def select_optimal_k_with_dunn(
    method_name: str,
    method_params: dict,
    X: np.ndarray,
    joint_trajectories: np.ndarray,
    windows: list[tuple[int, np.ndarray]],
    k_min: int,
    k_max: int,
) -> tuple[int, np.ndarray, dict[int, list], dict]:
    """
    Select optimal number of clusters using Dunn Index.

    Args:
        method_name: Name of the clustering method
        method_params: Parameters for the method
        X: Input data for clustering (n_samples, n_features)
        joint_trajectories: Joint space trajectories for Dunn (n_samples, n_features)
        windows: List of (clip_id, sampled_frames) tuples
        k_min: Minimum number of clusters
        k_max: Maximum number of clusters

    Returns:
        Tuple of (optimal_k, best_labels, cluster_dict, results_dict)
    """
    print(f"\n=== Dunn Index Selection for {method_name} (k={k_min} to {k_max}) ===")

    results = {
        "dunn_index": [],
        "min_inter_dist": [],
        "max_intra_diam": [],
        "n_clusters_actual": [],
        "per_cluster_stats": {},
        "all_labels": {},
        "all_cluster_dicts": {},
    }

    k_range = range(k_min, k_max + 1)
    method = get_clustering_method(method_name, method_params)

    for k in tqdm(k_range, desc=f"{method_name} k sweep"):
        labels = method.fit_predict(X, n_clusters=k)

        n_actual = len(set(labels)) - (1 if -1 in labels else 0)

        dunn, min_inter, max_intra = compute_dunn_index(labels, joint_trajectories)
        per_cluster = compute_per_cluster_stats(labels, joint_trajectories)

        results["dunn_index"].append(dunn)
        results["min_inter_dist"].append(min_inter)
        results["max_intra_diam"].append(max_intra)
        results["n_clusters_actual"].append(n_actual)
        results["per_cluster_stats"][k] = per_cluster
        results["all_labels"][k] = labels.copy()
        results["all_cluster_dicts"][k] = labels_to_cluster_dict(labels, windows)

    dunn_arr = np.array(results["dunn_index"])
    optimal_idx = np.argmax(dunn_arr)
    optimal_k = list(k_range)[optimal_idx]
    best_labels = results["all_labels"][optimal_k]
    cluster_dict = results["all_cluster_dicts"][optimal_k]

    print(f"\nMetrics by k (Dunn | n_actual | min_inter | max_intra):")
    for i, k in enumerate(k_range):
        dunn = results["dunn_index"][i]
        n_actual = results["n_clusters_actual"][i]
        min_inter = results["min_inter_dist"][i]
        max_intra = results["max_intra_diam"][i]
        marker = " <-- optimal" if k == optimal_k else ""
        print(f"  k={k}: Dunn={dunn:.4f} | n={n_actual} | min_inter={min_inter:.2f} | max_intra={max_intra:.2f}{marker}")

    print(f"\nOptimal k={optimal_k} for {method_name}")

    return optimal_k, best_labels, cluster_dict, results


def plot_method_dunn_analysis(
    results: dict,
    optimal_k: int,
    method_name: str,
    k_min: int,
    output_dir: str,
) -> None:
    """Plot Dunn analysis for a single method."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_dunn, ax_inter), (ax_intra, ax_per) = axes

    k_list = list(results["all_labels"].keys())

    ax_dunn.plot(k_list, results["dunn_index"], color="darkgreen", marker="o", linewidth=2, markersize=6)
    ax_dunn.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax_dunn.set_xlabel("Number of clusters (k)", fontsize=12)
    ax_dunn.set_ylabel("Dunn Index", fontsize=12)
    ax_dunn.set_title(f"{method_name}: Dunn Index", fontsize=14)
    ax_dunn.legend()
    ax_dunn.grid(True, alpha=0.3)

    ax_inter.plot(k_list, results["min_inter_dist"], color="blue", marker="s", linewidth=2, markersize=6)
    ax_inter.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax_inter.set_xlabel("Number of clusters (k)", fontsize=12)
    ax_inter.set_ylabel("Min Inter-cluster Distance", fontsize=12)
    ax_inter.set_title("Min Inter-cluster Distance", fontsize=14)
    ax_inter.legend()
    ax_inter.grid(True, alpha=0.3)

    ax_intra.plot(k_list, results["max_intra_diam"], color="purple", marker="^", linewidth=2, markersize=6)
    ax_intra.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax_intra.set_xlabel("Number of clusters (k)", fontsize=12)
    ax_intra.set_ylabel("Max Intra-cluster Diameter", fontsize=12)
    ax_intra.set_title("Max Intra-cluster Diameter", fontsize=14)
    ax_intra.legend()
    ax_intra.grid(True, alpha=0.3)

    per_cluster_optimal = results["per_cluster_stats"].get(optimal_k, {})
    if per_cluster_optimal:
        cluster_ids = sorted(per_cluster_optimal.keys())
        diameters = [per_cluster_optimal[cid]["diameter"] for cid in cluster_ids]
        sizes = [per_cluster_optimal[cid]["size"] for cid in cluster_ids]

        bars = ax_per.bar(cluster_ids, diameters, color='steelblue', alpha=0.7)
        for bar, size in zip(bars, sizes):
            ax_per.annotate(f'n={size}',
                          xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                          ha='center', va='bottom', fontsize=9)

    ax_per.set_xlabel("Cluster ID", fontsize=12)
    ax_per.set_ylabel("Diameter", fontsize=12)
    ax_per.set_title(f"Per-cluster Diameter at k={optimal_k}", fontsize=14)
    ax_per.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"dunn_analysis.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"dunn_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {method_name} Dunn analysis to {output_dir}/dunn_analysis.pdf")


def plot_dunn_comparison(
    all_results: dict[str, dict],
    output_dir: str,
) -> None:
    """Plot Dunn Index comparison across all methods."""
    os.makedirs(output_dir, exist_ok=True)

    methods = list(all_results.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_dunn, ax_inter), (ax_intra, ax_optimal) = axes

    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))

    for i, (method_name, data) in enumerate(all_results.items()):
        results = data["results"]
        k_list = list(results["all_labels"].keys())
        ax_dunn.plot(k_list, results["dunn_index"], color=colors[i], marker="o",
                     linewidth=2, markersize=5, label=method_name)
        ax_inter.plot(k_list, results["min_inter_dist"], color=colors[i], marker="s",
                      linewidth=2, markersize=5, label=method_name)
        ax_intra.plot(k_list, results["max_intra_diam"], color=colors[i], marker="^",
                      linewidth=2, markersize=5, label=method_name)

    ax_dunn.set_xlabel("Number of clusters (k)", fontsize=12)
    ax_dunn.set_ylabel("Dunn Index", fontsize=12)
    ax_dunn.set_title("Dunn Index by Method", fontsize=14)
    ax_dunn.legend(fontsize=9)
    ax_dunn.grid(True, alpha=0.3)

    ax_inter.set_xlabel("Number of clusters (k)", fontsize=12)
    ax_inter.set_ylabel("Min Inter-cluster Distance", fontsize=12)
    ax_inter.set_title("Min Inter-cluster Distance", fontsize=14)
    ax_inter.legend(fontsize=9)
    ax_inter.grid(True, alpha=0.3)

    ax_intra.set_xlabel("Number of clusters (k)", fontsize=12)
    ax_intra.set_ylabel("Max Intra-cluster Diameter", fontsize=12)
    ax_intra.set_title("Max Intra-cluster Diameter", fontsize=14)
    ax_intra.legend(fontsize=9)
    ax_intra.grid(True, alpha=0.3)

    method_names = []
    optimal_ks = []
    optimal_dunns = []
    for method_name, data in all_results.items():
        method_names.append(method_name)
        optimal_ks.append(data["optimal_k"])
        optimal_dunns.append(data["optimal_dunn"])

    x_pos = np.arange(len(method_names))
    bars = ax_optimal.bar(x_pos, optimal_dunns, color=colors[:len(method_names)], alpha=0.7)
    for bar, k in zip(bars, optimal_ks):
        ax_optimal.annotate(f'k={k}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9)
    ax_optimal.set_xticks(x_pos)
    ax_optimal.set_xticklabels(method_names, rotation=45, ha='right', fontsize=10)
    ax_optimal.set_ylabel("Optimal Dunn Index", fontsize=12)
    ax_optimal.set_title("Optimal Dunn Index per Method", fontsize=14)
    ax_optimal.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dunn_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "dunn_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Dunn comparison to {output_dir}/dunn_comparison.pdf")


def run_rendering(
    cluster_dict: dict,
    rodent_rollout_dir: str,
    rendering_cfg: RenderingConfig,
    output_dir: str,
    cluster_labels: dict[int, str] | None = None,
    output_filename: str = "cluster_side_by_side.mp4",
    title: str = "Cluster Samples",
) -> None:
    """Render cluster video."""
    cluster_chunks, cluster_clip_ids = chunk_loops_by_clustered_vpath(
        cluster_dict,
        rodent_rollout_dir,
        pad=rendering_cfg.pad,
        sample=rendering_cfg.max_samples,
        return_clip_ids=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)

    render_grid_clusters_side_by_side(
        cluster_chunks=cluster_chunks,
        out_path=out_path,
        framerate=rendering_cfg.framerate,
        max_samples_per_cluster=rendering_cfg.max_samples,
        target_size=rendering_cfg.target_size,
        title=title,
        per_tile_labels=True,
        selected_clusters=sorted(cluster_dict.keys()),
        cluster_labels=cluster_labels,
        cluster_clip_ids=cluster_clip_ids,
    )
    print(f"Video saved to: {out_path}")


def main(config_path: str = "configs/benchmark_cluster.yaml"):
    cfg = load_config(config_path)

    if cfg.clustering.only_max and cfg.clustering.only_min:
        raise ValueError("Cannot use both only_max and only_min")

    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print("Benchmark Clustering Pipeline")
    print("=" * 60)

    print("\n=== Stage 1: Loading intention data ===")
    work_rodent = functools.partial(
        get_aggregate_data, "/intention", [], path=cfg.rodent_rollout_dir
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        int_rodent = list(tqdm(pool.imap(work_rodent, range(cfg.num_clips)), total=cfg.num_clips))
    int_rodent = np.vstack(int_rodent)
    int_rodent = int_rodent.reshape([cfg.num_clips, 499, 16])
    print(f"Loaded intention: {int_rodent.shape}")

    print(f"\n=== Stage 2: Loading joint state data ({cfg.auto_cluster.joint_state_key}) ===")
    work_joint = functools.partial(
        get_aggregate_data, f"/{cfg.auto_cluster.joint_state_key}", [], path=cfg.rodent_rollout_dir
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        joint_list = list(tqdm(pool.imap(work_joint, range(cfg.num_clips)), total=cfg.num_clips))

    first_clip = joint_list[0]
    if first_clip.ndim == 2:
        n_frames, joint_dim = first_clip.shape
    else:
        total_size = first_clip.size
        joint_dim = total_size // 500
        n_frames = 500

    joint_data = np.array([
        clip.reshape(n_frames, joint_dim)[1:]
        for clip in joint_list
    ])
    print(f"Loaded joint data: {joint_data.shape}")

    print("\n=== Stage 3: Loading DM cycles ===")
    cycles_path = os.path.join(cfg.dm.results_dir, "all_cycle_dm.pkl")
    with open(cycles_path, "rb") as f:
        all_cycle_dm = pkl.load(f)
    print(f"Loaded {len(all_cycle_dm)} cycle sets from {cfg.dm.results_dir}")

    print("\n=== Stage 4: Preparing windows for clustering ===")
    windows, cleaned_loops = prepare_windows_for_clustering(
        all_cycle_dm, int_rodent, cfg.clustering
    )
    print(f"Prepared {len(windows)} windows for clustering")

    if len(windows) == 0:
        raise RuntimeError("No valid windows found for clustering")

    X = build_feature_matrix(windows, int_rodent)
    print(f"Feature matrix shape: {X.shape}")

    joint_trajectories = prepare_joint_trajectories(
        windows, joint_data, cfg.clustering.target_len
    )
    print(f"Joint trajectories shape: {joint_trajectories.shape}")

    print("\n=== Stage 5: Running benchmark methods ===")
    enabled_methods = [m for m in cfg.methods if m.enabled]
    print(f"Running {len(enabled_methods)} methods: {[m.name for m in enabled_methods]}")

    all_results = {}

    for method_cfg in enabled_methods:
        print(f"\n{'='*60}")
        print(f"Method: {method_cfg.name}")
        print(f"{'='*60}")

        optimal_k, best_labels, cluster_dict, results = select_optimal_k_with_dunn(
            method_name=method_cfg.name,
            method_params=method_cfg.params,
            X=X,
            joint_trajectories=joint_trajectories,
            windows=windows,
            k_min=cfg.auto_cluster.k_min,
            k_max=cfg.auto_cluster.k_max,
        )

        optimal_dunn = results["dunn_index"][optimal_k - cfg.auto_cluster.k_min]

        all_results[method_cfg.name] = {
            "optimal_k": optimal_k,
            "optimal_dunn": optimal_dunn,
            "labels": best_labels,
            "cluster_dict": cluster_dict,
            "results": results,
        }

        if cfg.auto_cluster.plot_dunn:
            method_output_dir = os.path.join(cfg.output_dir, method_cfg.name)
            plot_method_dunn_analysis(
                results, optimal_k, method_cfg.name,
                cfg.auto_cluster.k_min, method_output_dir
            )

    print("\n=== Stage 6: Generating comparison plots ===")
    if cfg.auto_cluster.plot_dunn and len(all_results) > 1:
        plot_dunn_comparison(all_results, cfg.output_dir)

    print("\n=== Stage 7: Saving results ===")
    results_path = os.path.join(cfg.output_dir, "benchmark_results.pkl")
    with open(results_path, "wb") as f:
        pkl.dump(all_results, f)
    print(f"Saved results to: {results_path}")

    summary_path = os.path.join(cfg.output_dir, "benchmark_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Benchmark Clustering Summary\n")
        f.write("=" * 60 + "\n\n")

        sorted_methods = sorted(
            all_results.items(),
            key=lambda x: x[1]["optimal_dunn"],
            reverse=True,
        )

        f.write("Ranking by Dunn Index:\n")
        f.write("-" * 40 + "\n")
        for rank, (method_name, result) in enumerate(sorted_methods, 1):
            f.write(f"{rank}. {method_name}: Dunn={result['optimal_dunn']:.4f} (k={result['optimal_k']})\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Per-method Details:\n")
        f.write("-" * 40 + "\n")
        for method_name, result in sorted_methods:
            f.write(f"\n{method_name}:\n")
            f.write(f"  Optimal k: {result['optimal_k']}\n")
            f.write(f"  Dunn Index: {result['optimal_dunn']:.4f}\n")
            f.write(f"  Cluster sizes: {[len(v) for v in result['cluster_dict'].values()]}\n")

    print(f"Saved summary to: {summary_path}")

    print("\n=== Stage 8: Rendering optimal cluster videos ===")
    for method_name, result in all_results.items():
        cluster_dict = result["cluster_dict"]
        optimal_k = result["optimal_k"]

        if not cluster_dict:
            print(f"Skipping {method_name}: no clusters found")
            continue

        print(f"\nRendering {method_name} (k={optimal_k})...")

        method_output_dir = os.path.join(cfg.output_dir, method_name)
        cluster_labels = {i: f"C{i}" for i in cluster_dict.keys()}

        run_rendering(
            cluster_dict,
            rodent_rollout_dir=cfg.rodent_rollout_dir,
            rendering_cfg=cfg.rendering,
            output_dir=method_output_dir,
            cluster_labels=cluster_labels,
            output_filename=f"cluster_k{optimal_k}_optimal.mp4",
            title=f"{method_name} (k={optimal_k}, Dunn-optimal)",
        )

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)

    print("\nFinal Rankings:")
    sorted_methods = sorted(
        all_results.items(),
        key=lambda x: x[1]["optimal_dunn"],
        reverse=True,
    )
    for rank, (method_name, result) in enumerate(sorted_methods, 1):
        print(f"  {rank}. {method_name}: Dunn={result['optimal_dunn']:.4f} (k={result['optimal_k']})")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/benchmark_cluster.yaml"
    main(config_path)
