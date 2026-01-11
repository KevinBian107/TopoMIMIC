"""
Unified experiment runner for behavioral clustering benchmark.

Single entry point for running all clustering and state-space methods
with fixed number of clusters, integrated seed handling, and auto-generated
labels.yaml files for each method.
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
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import pdist, cdist

from vnl_tda.utils.load_utils import get_aggregate_data
from vnl_tda.pipelines.tda.skeletonization.postprocess import (
    _expand_frames_nonconsec,
    _uniform_downsample,
    _normalize_dist_matrix,
    chunk_loops_by_clustered_vpath,
)
from vnl_tda.pipelines.tda.skeletonization.rendering import render_grid_clusters_side_by_side

try:
    import ot
    HAS_OT = True
except ImportError:
    HAS_OT = False

from scripts.benchmark.methods import (
    get_method,
    MethodType,
    ClusteringResult,
    KPMSResult,
)
from scripts.benchmark.alignment import (
    align_clusters_to_reference,
    remap_cluster_dict,
    remap_labels,
    compute_alignment_summary,
    get_alignment_quality_label,
)


@dataclass
class DataConfig:
    dm_key: str = "intention"
    evaluation_key: str = "qpos"


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
class MethodEntry:
    name: str
    type: str = "clustering"
    enabled: bool = True
    params: dict = field(default_factory=dict)


@dataclass
class RenderingConfig:
    enabled: bool = True
    pad: int = 0
    max_samples: int = 5
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)
    per_cluster_videos: bool = False


@dataclass
class KSelectionConfig:
    mode: str = "fixed"
    n_clusters: int = 10
    k_min: int = 2
    k_max: int = 15
    evaluation_key: str = "qpos"
    plot_dunn: bool = True
    save_all_k: bool = False


@dataclass
class ExperimentConfig:
    output_dir: str = "out/experiment"
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    num_clips: int = 500

    data: DataConfig = None
    dm: DMConfig = None
    clustering: ClusteringConfig = None

    k_selection: KSelectionConfig = None
    seeds: list[int] = field(default_factory=lambda: [0])
    aggregate_seeds: bool = True

    methods: list[MethodEntry] = field(default_factory=list)
    rendering: RenderingConfig = None

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.dm is None:
            self.dm = DMConfig()
        if self.clustering is None:
            self.clustering = ClusteringConfig()
        if self.k_selection is None:
            self.k_selection = KSelectionConfig()
        if self.rendering is None:
            self.rendering = RenderingConfig()

    @property
    def n_clusters(self) -> int:
        return self.k_selection.n_clusters

    @property
    def compute_dunn(self) -> bool:
        return self.k_selection.mode == "auto" or self.k_selection.plot_dunn


@dataclass
class ExperimentResults:
    results: dict
    config: ExperimentConfig


def load_config(config_path: str) -> ExperimentConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = DataConfig(**cfg.get("data", {}))
    dm_cfg = DMConfig(**cfg.get("dm", {}))

    clustering_dict = cfg.get("clustering", {})
    if "skip_clips" in clustering_dict:
        clustering_dict["skip_clips"] = tuple(clustering_dict["skip_clips"])
    clustering_cfg = ClusteringConfig(**clustering_dict)

    if "k_selection" in cfg:
        k_selection_cfg = KSelectionConfig(**cfg["k_selection"])
    elif "n_clusters" in cfg:
        k_selection_cfg = KSelectionConfig(
            mode="fixed",
            n_clusters=cfg["n_clusters"],
            plot_dunn=cfg.get("compute_dunn", True),
        )
    else:
        k_selection_cfg = KSelectionConfig()

    methods_list = []
    for m in cfg.get("methods", []):
        methods_list.append(MethodEntry(
            name=m["name"],
            type=m.get("type", "clustering"),
            enabled=m.get("enabled", True),
            params=m.get("params", {}),
        ))

    rendering_dict = cfg.get("rendering", {})
    if "target_size" in rendering_dict:
        rendering_dict["target_size"] = tuple(rendering_dict["target_size"])
    rendering_cfg = RenderingConfig(**rendering_dict)

    return ExperimentConfig(
        output_dir=cfg.get("output_dir", ExperimentConfig.output_dir),
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", ExperimentConfig.rodent_rollout_dir),
        num_clips=cfg.get("num_clips", ExperimentConfig.num_clips),
        data=data_cfg,
        dm=dm_cfg,
        clustering=clustering_cfg,
        k_selection=k_selection_cfg,
        seeds=cfg.get("seeds", [0]),
        aggregate_seeds=cfg.get("aggregate_seeds", True),
        methods=methods_list,
        rendering=rendering_cfg,
    )


def load_data(
    data_key: str,
    rodent_rollout_dir: str,
    num_clips: int,
) -> np.ndarray:
    work_fn = functools.partial(
        get_aggregate_data, f"/{data_key}", [], path=rodent_rollout_dir
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        data_list = list(tqdm(pool.imap(work_fn, range(num_clips)), total=num_clips, desc=f"Loading {data_key}"))

    first_clip = data_list[0]
    if first_clip.ndim == 2:
        n_frames, n_dim = first_clip.shape
    else:
        total_size = first_clip.size
        n_dim = total_size // 500
        n_frames = 500

    data = np.array([
        clip.reshape(-1, n_dim) if clip.ndim != 2 else clip
        for clip in data_list
    ])

    min_frames = min(d.shape[0] for d in data)
    data = data[:, :min_frames, :]

    print(f"Loaded {data_key}: {data.shape}")
    return data


def prepare_windows_for_clustering(
    all_cycle_dm: list,
    data: np.ndarray,
    clustering_cfg: ClusteringConfig,
) -> tuple[list[tuple[int, np.ndarray]], list[np.ndarray]]:
    windows = []
    cleaned_loops = []
    n_clips = len(data)

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

        clip_len = len(data[clip_id])
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
    data: np.ndarray,
) -> np.ndarray:
    features = []
    for clip_id, frames in windows:
        traj = data[clip_id, frames].flatten()
        features.append(traj)
    return np.array(features)


def compute_w2_distance_matrix(
    windows: list[tuple[int, np.ndarray]],
    data: np.ndarray,
    target_len: int,
    squared_cost: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if not HAS_OT:
        raise ImportError("POT library not installed")

    M = len(windows)
    D = np.zeros((M, M), dtype=float)

    print(f"Computing W2 distance matrix for {M} windows...")
    for i in tqdm(range(M), desc="W2 distances"):
        ci, fi = windows[i]
        Xi = data[ci, fi]
        wi = np.full(target_len, 1.0 / target_len)

        for j in range(i + 1, M):
            cj, fj = windows[j]
            Xj = data[cj, fj]
            vj = np.full(target_len, 1.0 / target_len)

            C = ot.dist(Xi, Xj)
            if squared_cost:
                C = C ** 2

            val = ot.emd2(wi, vj, C)
            if squared_cost:
                val = np.sqrt(val)

            D[i, j] = D[j, i] = val

    Dn = _normalize_dist_matrix(D)
    return D, Dn


def prepare_joint_trajectories(
    windows: list[tuple[int, np.ndarray]],
    joint_data: np.ndarray,
    target_len: int,
) -> np.ndarray:
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


def labels_to_cluster_dict(
    labels: np.ndarray,
    windows: list[tuple[int, np.ndarray]],
) -> dict[int, list[tuple[int, np.ndarray]]]:
    cluster_dict = {}
    for lbl, win in zip(labels, windows):
        lbl = int(lbl)
        if lbl < 0:
            continue
        cluster_dict.setdefault(lbl, []).append(win)
    return cluster_dict


def compute_per_cluster_stats(
    labels: np.ndarray,
    joint_trajectories: np.ndarray,
) -> dict:
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


def auto_select_clusters(
    method,
    X: np.ndarray,
    joint_trajectories: np.ndarray,
    windows: list[tuple[int, np.ndarray]],
    k_selection: "KSelectionConfig",
    seed: int,
    output_dir: str,
) -> tuple[int, dict, dict]:
    """
    Automatically select optimal number of clusters using Dunn Index.

    Args:
        method: Clustering method instance
        X: Feature matrix for clustering
        joint_trajectories: Joint space trajectories for Dunn evaluation
        windows: List of (clip_id, frames) tuples
        k_selection: K selection configuration
        seed: Random seed
        output_dir: Output directory for plots

    Returns:
        Tuple of (optimal_k, cluster_dict, all_results)
    """
    print(f"\n=== Dunn Index Auto-Selection (k={k_selection.k_min} to {k_selection.k_max}) ===")

    k_range = range(k_selection.k_min, k_selection.k_max + 1)
    results = {
        "dunn_index": [],
        "min_inter_dist": [],
        "max_intra_diam": [],
        "per_cluster_stats": {},
    }
    all_cluster_dicts = {}

    for k in tqdm(k_range, desc="Evaluating k values"):
        labels = method.fit_predict(X, n_clusters=k, seed=seed)
        cluster_dict_k = labels_to_cluster_dict(labels, windows)

        dunn, min_inter, max_intra = compute_dunn_index(labels, joint_trajectories)
        per_cluster = compute_per_cluster_stats(labels, joint_trajectories)

        results["dunn_index"].append(dunn)
        results["min_inter_dist"].append(min_inter)
        results["max_intra_diam"].append(max_intra)
        results["per_cluster_stats"][k] = per_cluster
        all_cluster_dicts[k] = {
            "cluster_dict": cluster_dict_k,
            "labels": labels,
            "dunn_index": dunn,
            "min_inter_dist": min_inter,
            "max_intra_diam": max_intra,
        }

    dunn_arr = np.array(results["dunn_index"])
    optimal_idx = np.argmax(dunn_arr)
    optimal_k = list(k_range)[optimal_idx]
    optimal_dunn = results["dunn_index"][optimal_idx]

    print(f"\nDunn Index by k:")
    for i, k in enumerate(k_range):
        dunn = results["dunn_index"][i]
        marker = " <-- optimal" if k == optimal_k else ""
        print(f"  k={k}: Dunn={dunn:.4f}{marker}")

    print(f"\nOptimal k={optimal_k} with Dunn={optimal_dunn:.4f}")

    if k_selection.plot_dunn:
        plot_dunn_analysis(k_range, results, optimal_k, output_dir)

    return optimal_k, all_cluster_dicts[optimal_k], all_cluster_dicts


def plot_dunn_analysis(
    k_range: range,
    results: dict,
    optimal_k: int,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_dunn, ax_inter), (ax_intra, ax_per) = axes

    k_list = list(k_range)

    ax_dunn.plot(k_list, results["dunn_index"], color="darkgreen", marker="o", linewidth=2, markersize=6)
    ax_dunn.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax_dunn.set_xlabel("Number of clusters (k)", fontsize=12)
    ax_dunn.set_ylabel("Dunn Index", fontsize=12)
    ax_dunn.set_title("Dunn Index (higher = more distinct clusters)", fontsize=14)
    ax_dunn.legend()
    ax_dunn.grid(True, alpha=0.3)

    ax_inter.plot(k_list, results["min_inter_dist"], color="blue", marker="s", linewidth=2, markersize=6)
    ax_inter.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax_inter.set_xlabel("Number of clusters (k)", fontsize=12)
    ax_inter.set_ylabel("Min Inter-cluster Distance", fontsize=12)
    ax_inter.set_title("Min Inter-cluster Distance (Dunn numerator)", fontsize=14)
    ax_inter.legend()
    ax_inter.grid(True, alpha=0.3)

    ax_intra.plot(k_list, results["max_intra_diam"], color="purple", marker="^", linewidth=2, markersize=6)
    ax_intra.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax_intra.set_xlabel("Number of clusters (k)", fontsize=12)
    ax_intra.set_ylabel("Max Intra-cluster Diameter", fontsize=12)
    ax_intra.set_title("Max Intra-cluster Diameter (Dunn denominator)", fontsize=14)
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
    ax_per.set_title(f"Per-cluster Diameter at optimal k={optimal_k}", fontsize=14)
    ax_per.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dunn_analysis.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "dunn_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Dunn analysis saved to {output_dir}/dunn_analysis.pdf")


def generate_labels_yaml(
    method_name: str,
    n_clusters: int,
    seed: int,
    output_dir: str,
    experiment_id: str = None,
) -> str:
    if experiment_id is None:
        from datetime import datetime
        experiment_id = f"experiment_{datetime.now().strftime('%Y-%m-%d')}"

    labels_path = os.path.join(output_dir, "labels.yaml")

    labels_content = {
        "experiment_id": experiment_id,
        "method": method_name,
        "n_clusters": n_clusters,
        "seed": seed,
        "labels": {i: "" for i in range(n_clusters)},
    }

    with open(labels_path, "w") as f:
        f.write("# AUTO-GENERATED - Fill in labels after watching videos\n")
        f.write("# Label format: single behavior (\"rear\"), transition (\"walk rear\"), compound (\"rear left_turn\")\n\n")
        yaml.dump(labels_content, f, default_flow_style=False, sort_keys=False)

    return labels_path


def run_rendering(
    cluster_dict: dict,
    rodent_rollout_dir: str,
    rendering_cfg: RenderingConfig,
    output_dir: str,
    cluster_labels: dict[int, str] | None = None,
    output_filename: str = "cluster_side_by_side.mp4",
    title: str = "Cluster Samples",
) -> None:
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


def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("Unified Benchmark Experiment")
    print("=" * 60)
    print(f"Output dir: {config.output_dir}")
    if config.k_selection.mode == "auto":
        print(f"k_selection: auto (k_min={config.k_selection.k_min}, k_max={config.k_selection.k_max})")
    else:
        print(f"k_selection: fixed (n_clusters={config.k_selection.n_clusters})")
    print(f"Seeds: {config.seeds}")

    print(f"\n=== Stage 1: Loading {config.data.dm_key} data ===")
    dm_data = load_data(config.data.dm_key, config.rodent_rollout_dir, config.num_clips)

    print(f"\n=== Stage 2: Loading {config.data.evaluation_key} data ===")
    eval_data = load_data(config.data.evaluation_key, config.rodent_rollout_dir, config.num_clips)

    min_frames = min(dm_data.shape[1], eval_data.shape[1])
    dm_data = dm_data[:, :min_frames, :]
    eval_data = eval_data[:, :min_frames, :]

    print("\n=== Stage 3: Loading DM cycles ===")
    if not config.dm.results_dir:
        raise ValueError("dm.results_dir required")
    cycles_path = os.path.join(config.dm.results_dir, "all_cycle_dm.pkl")
    with open(cycles_path, "rb") as f:
        all_cycle_dm = pkl.load(f)
    print(f"Loaded {len(all_cycle_dm)} cycle sets")

    print("\n=== Stage 4: Preparing windows ===")
    windows, cleaned_loops = prepare_windows_for_clustering(
        all_cycle_dm, dm_data, config.clustering
    )
    print(f"Prepared {len(windows)} windows")

    if len(windows) == 0:
        raise RuntimeError("No valid windows for clustering")

    X = build_feature_matrix(windows, dm_data)
    print(f"Feature matrix: {X.shape}")

    joint_trajectories = prepare_joint_trajectories(
        windows, eval_data, config.clustering.target_len
    )
    print(f"Joint trajectories: {joint_trajectories.shape}")

    w2_cache_path = os.path.join(config.output_dir, "w2_distance_matrix.pkl")
    if os.path.exists(w2_cache_path):
        print("Loading cached W2 distance matrix...")
        with open(w2_cache_path, "rb") as f:
            w2_data = pkl.load(f)
        D, Dn = w2_data["D"], w2_data["Dn"]
    else:
        D, Dn = compute_w2_distance_matrix(
            windows, dm_data, config.clustering.target_len
        )
        with open(w2_cache_path, "wb") as f:
            pkl.dump({"D": D, "Dn": Dn}, f)
        print(f"Cached W2 matrix to {w2_cache_path}")

    print("\n=== Stage 5: Running methods ===")
    enabled_methods = [m for m in config.methods if m.enabled]
    print(f"Enabled methods: {[m.name for m in enabled_methods]}")

    all_results = {}
    from datetime import datetime
    experiment_id = f"experiment_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"

    for method_entry in enabled_methods:
        print(f"\n{'='*60}")
        print(f"Method: {method_entry.name}")
        print(f"{'='*60}")

        method_output_dir = os.path.join(config.output_dir, method_entry.name)
        os.makedirs(method_output_dir, exist_ok=True)

        method = get_method(method_entry.name, method_entry.params)
        method_results = {}

        if method_entry.name == "w2_kmeans":
            method.params["distance_matrix"] = Dn

        if method.method_type == MethodType.STATE_SPACE:
            n_clusters_to_use = config.k_selection.n_clusters
            for seed in config.seeds:
                seed_dir = os.path.join(method_output_dir, f"seed_{seed}")
                kpms_result = method.fit(
                    dm_data, n_clusters_to_use, seed_dir, seed
                )

                dunn_index = None
                if config.compute_dunn:
                    cluster_dict = kpms_labels_to_cluster_dict(kpms_result.labels)
                    joint_traj = prepare_kpms_joint_trajectories(
                        kpms_result.labels, eval_data
                    )
                    dunn_index, _, _ = compute_kpms_dunn_index(
                        kpms_result.labels, eval_data
                    )

                method_results[seed] = {
                    "n_clusters": n_clusters_to_use,
                    "labels": kpms_result.labels,
                    "dunn_index": dunn_index,
                }

                print(f"  Seed {seed}: Dunn={dunn_index:.4f}" if dunn_index else f"  Seed {seed}: complete")
        else:
            if config.k_selection.mode == "auto":
                primary_seed = config.seeds[0]
                optimal_k, optimal_result, all_k_results = auto_select_clusters(
                    method=method,
                    X=X,
                    joint_trajectories=joint_trajectories,
                    windows=windows,
                    k_selection=config.k_selection,
                    seed=primary_seed,
                    output_dir=method_output_dir,
                )

                method_results[primary_seed] = {
                    "n_clusters": optimal_k,
                    "cluster_dict": optimal_result["cluster_dict"],
                    "labels": optimal_result["labels"],
                    "dunn_index": optimal_result["dunn_index"],
                    "optimal_k": optimal_k,
                    "dunn_by_k": {k: r["dunn_index"] for k, r in all_k_results.items()},
                }

                if config.k_selection.save_all_k:
                    method_results["all_k_results"] = all_k_results

                for seed in config.seeds[1:]:
                    labels = method.fit_predict(X, optimal_k, seed)
                    cluster_dict = labels_to_cluster_dict(labels, windows)
                    dunn_index, _, _ = compute_dunn_index(labels, joint_trajectories)

                    method_results[seed] = {
                        "n_clusters": optimal_k,
                        "cluster_dict": cluster_dict,
                        "labels": labels,
                        "dunn_index": dunn_index,
                        "optimal_k": optimal_k,
                    }
                    print(f"  Seed {seed}: k={optimal_k}, Dunn={dunn_index:.4f}")
            else:
                for seed in config.seeds:
                    labels = method.fit_predict(X, config.k_selection.n_clusters, seed)
                    cluster_dict = labels_to_cluster_dict(labels, windows)

                    dunn_index = None
                    if config.compute_dunn:
                        dunn_index, _, _ = compute_dunn_index(labels, joint_trajectories)

                    method_results[seed] = {
                        "n_clusters": config.k_selection.n_clusters,
                        "cluster_dict": cluster_dict,
                        "labels": labels,
                        "dunn_index": dunn_index,
                    }

                    print(f"  Seed {seed}: Dunn={dunn_index:.4f}" if dunn_index else f"  Seed {seed}: complete")

        if len(config.seeds) > 1 and method.method_type == MethodType.CLUSTERING:
            print(f"  Aligning clusters across seeds...")
            primary_seed = config.seeds[0]
            ref_labels = method_results[primary_seed]["labels"]
            alignment_results = {}

            for seed in config.seeds[1:]:
                target_labels = method_results[seed]["labels"]
                alignment = align_clusters_to_reference(ref_labels, target_labels)
                alignment_results[seed] = alignment

                aligned_labels = remap_labels(target_labels, alignment.mapping)
                aligned_cluster_dict = remap_cluster_dict(
                    method_results[seed]["cluster_dict"], alignment.mapping
                )

                method_results[seed]["labels_aligned"] = aligned_labels
                method_results[seed]["cluster_dict_aligned"] = aligned_cluster_dict
                method_results[seed]["alignment"] = {
                    "mapping": alignment.mapping,
                    "score": alignment.alignment_score,
                    "per_cluster_scores": alignment.per_cluster_scores,
                    "quality": get_alignment_quality_label(alignment.alignment_score),
                }

                quality = get_alignment_quality_label(alignment.alignment_score)
                print(f"    Seed {seed}: alignment={alignment.alignment_score:.1%} ({quality})")

            alignment_summary = compute_alignment_summary(alignment_results)
            method_results["alignment_summary"] = alignment_summary
            print(f"  Mean alignment: {alignment_summary['mean_score']:.1%}")

        if config.aggregate_seeds and len(config.seeds) > 1:
            dunn_values = [r.get("dunn_index", 0) for r in method_results.values() if r.get("dunn_index")]
            method_results["aggregated"] = {
                "mean_dunn": np.mean(dunn_values) if dunn_values else 0.0,
                "std_dunn": np.std(dunn_values) if dunn_values else 0.0,
                "n_seeds": len(config.seeds),
            }

        primary_seed = config.seeds[0]
        result_n_clusters = method_results[primary_seed].get("n_clusters", config.k_selection.n_clusters)
        generate_labels_yaml(
            method_name=method_entry.name,
            n_clusters=result_n_clusters,
            seed=primary_seed,
            output_dir=method_output_dir,
            experiment_id=experiment_id,
        )

        with open(os.path.join(method_output_dir, "results.pkl"), "wb") as f:
            pkl.dump(method_results, f)

        all_results[method_entry.name] = method_results

    if config.rendering.enabled:
        print("\n=== Stage 6: Rendering videos ===")
        for method_entry in enabled_methods:
            method_output_dir = os.path.join(config.output_dir, method_entry.name)
            method_results = all_results[method_entry.name]

            primary_seed = config.seeds[0]
            result = method_results[primary_seed]
            result_n_clusters = result.get("n_clusters", config.k_selection.n_clusters)

            if "cluster_dict" in result:
                cluster_dict = result["cluster_dict"]
                cluster_labels = {i: f"C{i}" for i in cluster_dict.keys()}

                is_auto = config.k_selection.mode == "auto"
                title_suffix = " (auto)" if is_auto else ""
                run_rendering(
                    cluster_dict,
                    config.rodent_rollout_dir,
                    config.rendering,
                    method_output_dir,
                    cluster_labels=cluster_labels,
                    output_filename=f"clusters_k{result_n_clusters}.mp4",
                    title=f"{method_entry.name} (k={result_n_clusters}{title_suffix})",
                )

                if is_auto and config.k_selection.save_all_k and "all_k_results" in method_results:
                    k_sweep_dir = os.path.join(method_output_dir, "k_sweep")
                    os.makedirs(k_sweep_dir, exist_ok=True)
                    print(f"  Rendering k-sweep videos for {method_entry.name}...")
                    for k, k_result in tqdm(method_results["all_k_results"].items(), desc="Rendering k sweep"):
                        k_cluster_dict = k_result["cluster_dict"]
                        k_labels = {i: f"C{i}" for i in k_cluster_dict.keys()}
                        is_optimal = " (optimal)" if k == result_n_clusters else ""
                        run_rendering(
                            k_cluster_dict,
                            config.rodent_rollout_dir,
                            config.rendering,
                            k_sweep_dir,
                            cluster_labels=k_labels,
                            output_filename=f"clusters_k{k}.mp4",
                            title=f"{method_entry.name} (k={k}){is_optimal}",
                        )

    print("\n=== Stage 7: Saving summary ===")
    summary_path = os.path.join(config.output_dir, "experiment_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Unified Benchmark Experiment Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Experiment ID: {experiment_id}\n")
        if config.k_selection.mode == "auto":
            f.write(f"k_selection: auto (k_min={config.k_selection.k_min}, k_max={config.k_selection.k_max})\n")
        else:
            f.write(f"k_selection: fixed (n_clusters={config.k_selection.n_clusters})\n")
        f.write(f"Seeds: {config.seeds}\n\n")

        f.write("Results by Method:\n")
        f.write("-" * 40 + "\n")

        for method_name, results in all_results.items():
            f.write(f"\n{method_name}:\n")
            primary_result = results.get(config.seeds[0], {})
            if "optimal_k" in primary_result:
                f.write(f"  Optimal k: {primary_result['optimal_k']}\n")
            for seed in config.seeds:
                if seed in results:
                    r = results[seed]
                    dunn = r.get("dunn_index", "N/A")
                    k = r.get("n_clusters", "N/A")
                    if isinstance(dunn, float):
                        f.write(f"  Seed {seed}: k={k}, Dunn={dunn:.4f}\n")
                    else:
                        f.write(f"  Seed {seed}: k={k}, Dunn={dunn}\n")
            if "aggregated" in results:
                agg = results["aggregated"]
                f.write(f"  Aggregated: mean_dunn={agg['mean_dunn']:.4f} +/- {agg['std_dunn']:.4f}\n")
            if "alignment_summary" in results:
                align_sum = results["alignment_summary"]
                f.write(f"  Alignment: mean={align_sum['mean_score']:.1%}, ")
                f.write(f"min={align_sum['min_score']:.1%}, max={align_sum['max_score']:.1%}\n")
                for seed, score in align_sum.get("per_seed_scores", {}).items():
                    quality = get_alignment_quality_label(score)
                    f.write(f"    Seed {seed}: {score:.1%} ({quality})\n")

    print(f"Saved summary to {summary_path}")

    results_path = os.path.join(config.output_dir, "all_results.pkl")
    with open(results_path, "wb") as f:
        pkl.dump(all_results, f)
    print(f"Saved all results to {results_path}")

    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)

    return ExperimentResults(results=all_results, config=config)


def kpms_labels_to_cluster_dict(
    labels_list: list[np.ndarray],
) -> dict[int, list[tuple[int, np.ndarray]]]:
    cluster_dict = {}
    for clip_idx, syllables in enumerate(labels_list):
        syllables = np.asarray(syllables)
        unique_sylls = np.unique(syllables)

        for syll in unique_sylls:
            syll = int(syll)
            frame_indices = np.where(syllables == syll)[0]
            if syll not in cluster_dict:
                cluster_dict[syll] = []
            cluster_dict[syll].append((clip_idx, frame_indices))

    return cluster_dict


def prepare_kpms_joint_trajectories(
    labels_list: list[np.ndarray],
    joint_data: np.ndarray,
) -> np.ndarray:
    cluster_frames = {}
    for clip_idx, sylls in enumerate(labels_list):
        for t, s in enumerate(sylls):
            s = int(s)
            if s not in cluster_frames:
                cluster_frames[s] = []
            if t < joint_data.shape[1]:
                cluster_frames[s].append(joint_data[clip_idx, t])

    return cluster_frames


def compute_kpms_dunn_index(
    labels_list: list[np.ndarray],
    joint_data: np.ndarray,
) -> tuple[float, float, float]:
    cluster_frames = {}
    for clip_idx, sylls in enumerate(labels_list):
        for t, s in enumerate(sylls):
            s = int(s)
            if s not in cluster_frames:
                cluster_frames[s] = []
            if t < joint_data.shape[1]:
                cluster_frames[s].append(joint_data[clip_idx, t])

    cluster_data = {}
    for s, frames in cluster_frames.items():
        if len(frames) > 0:
            cluster_data[s] = np.array(frames)

    unique_labels = list(cluster_data.keys())
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return 0.0, 0.0, 0.0

    cluster_diameters = []
    for lbl in unique_labels:
        X = cluster_data[lbl]
        if len(X) < 2:
            cluster_diameters.append(0.0)
        else:
            n_samples = min(500, len(X))
            if len(X) > n_samples:
                idx = np.random.choice(len(X), n_samples, replace=False)
                X_sub = X[idx]
            else:
                X_sub = X
            pairwise = pdist(X_sub, metric='euclidean')
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

            n_i = min(200, len(X_i))
            n_j = min(200, len(X_j))
            if len(X_i) > n_i:
                idx_i = np.random.choice(len(X_i), n_i, replace=False)
                X_i = X_i[idx_i]
            if len(X_j) > n_j:
                idx_j = np.random.choice(len(X_j), n_j, replace=False)
                X_j = X_j[idx_j]

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


def main(config_path: str = "configs/benchmark/experiment.yaml"):
    if os.path.exists(config_path):
        cfg = load_config(config_path)
    else:
        cfg = ExperimentConfig()

    run_experiment(cfg)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/benchmark/experiment.yaml"
    main(config_path)
