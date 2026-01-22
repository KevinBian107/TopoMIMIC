"""
Unified experiment runner for behavioral clustering benchmark.

Single entry point for running all clustering and state-space methods
with fixed number of clusters, integrated seed handling, and auto-generated
labels.yaml files for each method.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["JAX_ENABLE_X64"] = "True"

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

from topo_mimic.dmwk.utils.load_utils import get_aggregate_data, get_clip_indices
from topo_mimic.dmwk.skeletonization.postprocess import (
    _expand_frames_nonconsec,
    _uniform_downsample,
    _normalize_dist_matrix,
    chunk_loops_by_clustered_vpath,
)
from topo_mimic.dmwk.skeletonization.rendering import render_grid_clusters_side_by_side

try:
    import ot
    HAS_OT = True
except ImportError:
    HAS_OT = False

from topo_mimic.benchmark.methods import (
    get_method,
    MethodType,
    ClusteringResult,
    KPMSResult,
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


@dataclass
class ExperimentConfig:
    output_dir: str = "out/experiment"
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    split: str = "train"

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
        k_sel_dict = cfg["k_selection"].copy()
        for key in ["plot_dunn", "plot_silhouette", "evaluation_key", "k_min", "k_max", "save_all_k"]:
            k_sel_dict.pop(key, None)
        k_sel_dict["mode"] = "fixed"
        k_selection_cfg = KSelectionConfig(**k_sel_dict)
    elif "n_clusters" in cfg:
        k_selection_cfg = KSelectionConfig(
            mode="fixed",
            n_clusters=cfg["n_clusters"],
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
        split=cfg.get("split", ExperimentConfig.split),
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
    split: str,
) -> tuple[np.ndarray, list[int]]:
    clip_indices = get_clip_indices(split)
    work_fn = functools.partial(
        get_aggregate_data, f"/{data_key}", [], path=rodent_rollout_dir
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        data_list = list(tqdm(pool.imap(work_fn, clip_indices), total=len(clip_indices), desc=f"Loading {data_key}"))

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

    print(f"Loaded {data_key}: {data.shape} (split={split}, clips={clip_indices[0]}-{clip_indices[-1]})")
    return data, clip_indices


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

        clip_len = len(data[clip_id])
        for loop in loops:
            expanded = _expand_frames_nonconsec(loop, pad=0, clip_len=clip_len)
            sampled = _uniform_downsample(expanded, clustering_cfg.target_len)
            if sampled is None:
                continue
            windows.append((clip_id, sampled))
            cleaned_loops.append(np.asarray(loop, dtype=int))

    if clustering_cfg.only_max and cleaned_loops:
        loop_lengths = np.array([len(loop) for loop in cleaned_loops])
        print(f"[only_max] Loop length stats: mean={loop_lengths.mean():.2f}, std={loop_lengths.std():.2f}, n={len(loop_lengths)}")

    return windows, cleaned_loops


def prepare_all_frames_for_clustering(
    data: np.ndarray,
    clustering_cfg: ClusteringConfig,
) -> list[tuple[int, np.ndarray]]:
    """
    Prepare windows using all frames from each clip (baseline).

    Instead of using topologically-discovered cycles, this function creates
    one window per clip using all available frames, uniformly downsampled
    to target_len.

    Args:
        data: Full data array (n_clips, n_frames, n_dim)
        clustering_cfg: Clustering configuration

    Returns:
        List of (clip_id, frame_indices) tuples, one per clip
    """
    windows = []
    n_clips = len(data)

    for clip_id in range(n_clips):
        if clip_id in clustering_cfg.skip_clips:
            continue

        clip_len = len(data[clip_id])
        all_frames = np.arange(clip_len)
        sampled = _uniform_downsample(all_frames, clustering_cfg.target_len)

        if sampled is not None:
            windows.append((clip_id, sampled))

    print(f"[all_frames baseline] Created {len(windows)} windows (1 per clip)")
    return windows


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


def compute_gw_distance_matrix(
    windows: list[tuple[int, np.ndarray]],
    data: np.ndarray,
    target_len: int,
    loss_fun: str = "square_loss",
    max_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Gromov-Wasserstein distance matrix between all window pairs.

    Args:
        windows: List of (clip_id, frame_indices) tuples
        data: Full data array (n_clips, n_frames, n_dim)
        target_len: Number of frames per window (for uniform weights)
        loss_fun: GW loss function ("square_loss" or "kl_loss")
        max_iter: Maximum iterations for GW solver

    Returns:
        D: Raw GW distance matrix (M, M)
        Dn: Normalized GW distance matrix (M, M)
    """
    if not HAS_OT:
        raise ImportError("POT library not installed")

    M = len(windows)
    D = np.zeros((M, M), dtype=float)

    print(f"Computing GW distance matrix for {M} windows...")
    print(f"  loss_fun={loss_fun}, max_iter={max_iter}")

    structure_matrices = []
    for clip_id, frames in tqdm(windows, desc="Precomputing structure matrices"):
        Xi = data[clip_id, frames]
        Ci = ot.dist(Xi, Xi)
        structure_matrices.append(Ci)

    w = np.full(target_len, 1.0 / target_len)

    for i in tqdm(range(M), desc="GW distances"):
        Ci = structure_matrices[i]
        for j in range(i + 1, M):
            Cj = structure_matrices[j]
            val = ot.gromov.gromov_wasserstein2(
                Ci, Cj, w, w,
                loss_fun=loss_fun,
                max_iter=max_iter,
            )
            D[i, j] = D[j, i] = val

    Dn = _normalize_dist_matrix(D)
    return D, Dn


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

    labels_path = os.path.join(output_dir, f"labels_seed_{seed}.yaml")

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
    print(f"n_clusters: {config.k_selection.n_clusters}")
    print(f"Seeds: {config.seeds}")

    print(f"\n=== Stage 1: Loading {config.data.dm_key} data ===")
    dm_data, clip_indices = load_data(config.data.dm_key, config.rodent_rollout_dir, config.split)

    print(f"\n=== Stage 2: Loading {config.data.evaluation_key} data ===")
    eval_data, _ = load_data(config.data.evaluation_key, config.rodent_rollout_dir, config.split)

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
    print(f"Prepared {len(windows)} windows (from topological cycles)")

    if len(windows) == 0:
        raise RuntimeError("No valid windows for clustering")

    X = build_feature_matrix(windows, dm_data)
    print(f"Feature matrix: {X.shape}")

    baseline_needed = any(m.name == "w2_unif_kmeans" and m.enabled for m in config.methods)
    windows_baseline = None
    X_baseline = None
    if baseline_needed:
        print("\n=== Stage 4b: Preparing baseline windows (all frames) ===")
        windows_baseline = prepare_all_frames_for_clustering(dm_data, config.clustering)
        X_baseline = build_feature_matrix(windows_baseline, dm_data)
        print(f"Baseline feature matrix: {X_baseline.shape}")

    D_w2, Dn_w2 = None, None
    D_gw, Dn_gw = None, None
    D_w2_baseline, Dn_w2_baseline = None, None

    w2_needed = any(m.name == "w2_dm_kmeans" and m.enabled for m in config.methods)
    if w2_needed:
        w2_cache_path = os.path.join(config.output_dir, "w2_distance_matrix.pkl")
        if os.path.exists(w2_cache_path):
            print("Loading cached W2 distance matrix...")
            with open(w2_cache_path, "rb") as f:
                w2_data = pkl.load(f)
            D_w2, Dn_w2 = w2_data["D"], w2_data["Dn"]
        else:
            D_w2, Dn_w2 = compute_w2_distance_matrix(
                windows, dm_data, config.clustering.target_len
            )
            with open(w2_cache_path, "wb") as f:
                pkl.dump({"D": D_w2, "Dn": Dn_w2}, f)
            print(f"Cached W2 matrix to {w2_cache_path}")

    if baseline_needed:
        w2_baseline_cache_path = os.path.join(config.output_dir, "w2_distance_matrix_baseline.pkl")
        if os.path.exists(w2_baseline_cache_path):
            print("Loading cached W2 baseline distance matrix...")
            with open(w2_baseline_cache_path, "rb") as f:
                w2_data = pkl.load(f)
            D_w2_baseline, Dn_w2_baseline = w2_data["D"], w2_data["Dn"]
        else:
            print("Computing W2 baseline distance matrix (all frames)...")
            D_w2_baseline, Dn_w2_baseline = compute_w2_distance_matrix(
                windows_baseline, dm_data, config.clustering.target_len
            )
            with open(w2_baseline_cache_path, "wb") as f:
                pkl.dump({"D": D_w2_baseline, "Dn": Dn_w2_baseline}, f)
            print(f"Cached W2 baseline matrix to {w2_baseline_cache_path}")

    gw_needed = any(m.name == "gw_dm_kmeans" and m.enabled for m in config.methods)
    if gw_needed:
        gw_method = next((m for m in config.methods if m.name == "gw_dm_kmeans"), None)
        gw_params = gw_method.params if gw_method else {}
        loss_fun = gw_params.get("loss_fun", "square_loss")
        max_iter = gw_params.get("max_iter", 1000)

        gw_cache_path = os.path.join(config.output_dir, "gw_distance_matrix.pkl")
        if os.path.exists(gw_cache_path):
            print("Loading cached GW distance matrix...")
            with open(gw_cache_path, "rb") as f:
                gw_data = pkl.load(f)
            D_gw, Dn_gw = gw_data["D"], gw_data["Dn"]
        else:
            D_gw, Dn_gw = compute_gw_distance_matrix(
                windows, dm_data, config.clustering.target_len,
                loss_fun=loss_fun, max_iter=max_iter
            )
            with open(gw_cache_path, "wb") as f:
                pkl.dump({"D": D_gw, "Dn": Dn_gw}, f)
            print(f"Cached GW matrix to {gw_cache_path}")

    print("\n=== Stage 5: Running methods ===")
    enabled_methods = [m for m in config.methods if m.enabled]
    print(f"Enabled methods: {[m.name for m in enabled_methods]}")

    results_path = os.path.join(config.output_dir, "all_results.pkl")
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            all_results = pkl.load(f)
        print(f"Loaded existing results: {list(all_results.keys())}")
    else:
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

        if method_entry.name == "w2_dm_kmeans":
            method.params["distance_matrix"] = Dn_w2
        elif method_entry.name == "w2_unif_kmeans":
            method.params["distance_matrix"] = Dn_w2_baseline
        elif method_entry.name == "gw_dm_kmeans":
            method.params["distance_matrix"] = Dn_gw

        current_windows = windows_baseline if method_entry.name == "w2_unif_kmeans" else windows
        current_X = X_baseline if method_entry.name == "w2_unif_kmeans" else X

        if method.method_type == MethodType.STATE_SPACE:
            n_clusters_to_use = config.k_selection.n_clusters

            skip_set = set(config.clustering.skip_clips)
            keep_indices = [i for i in range(dm_data.shape[0]) if i not in skip_set]
            dm_data_filtered = dm_data[keep_indices]
            print(f"  Filtered {dm_data.shape[0]} -> {dm_data_filtered.shape[0]} clips (skipped {len(skip_set)})")

            for seed in config.seeds:
                seed_dir = os.path.join(method_output_dir, f"seed_{seed}")
                kpms_result = method.fit(
                    dm_data_filtered, n_clusters_to_use, seed_dir, seed
                )

                cluster_dict_filtered = kpms_result.transition_cluster_dict
                cluster_dict = {
                    cluster_id: [(keep_indices[clip_id], frames) for clip_id, frames in instances]
                    for cluster_id, instances in cluster_dict_filtered.items()
                }
                stats = kpms_result.transition_stats

                method_results[seed] = {
                    "n_clusters": len(cluster_dict),
                    "cluster_dict": cluster_dict,
                    "labels": kpms_result.labels,
                    "transition_stats": stats,
                }

                print(f"  Seed {seed}: {stats['n_sustained_segments']} sustained segments, "
                      f"{stats['n_transition_types']} transition types, "
                      f"{len(cluster_dict)} clusters")
        else:
            for seed in config.seeds:
                labels = method.fit_predict(current_X, config.k_selection.n_clusters, seed)
                cluster_dict = labels_to_cluster_dict(labels, current_windows)

                method_results[seed] = {
                    "n_clusters": config.k_selection.n_clusters,
                    "cluster_dict": cluster_dict,
                    "labels": labels,
                }

                print(f"  Seed {seed}: complete")

        for seed in config.seeds:
            result_n_clusters = method_results[seed].get("n_clusters", config.k_selection.n_clusters)
            generate_labels_yaml(
                method_name=method_entry.name,
                n_clusters=result_n_clusters,
                seed=seed,
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

            for seed in config.seeds:
                result = method_results[seed]
                result_n_clusters = result.get("n_clusters", config.k_selection.n_clusters)

                if "cluster_dict" in result:
                    cluster_dict = result["cluster_dict"]
                    cluster_labels = {i: f"C{i}" for i in cluster_dict.keys()}

                    run_rendering(
                        cluster_dict,
                        config.rodent_rollout_dir,
                        config.rendering,
                        method_output_dir,
                        cluster_labels=cluster_labels,
                        output_filename=f"clusters_k{result_n_clusters}_seed{seed}.mp4",
                        title=f"{method_entry.name} (k={result_n_clusters}, seed={seed})",
                    )

    print("\n=== Stage 7: Saving summary ===")
    summary_path = os.path.join(config.output_dir, "experiment_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Unified Benchmark Experiment Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Experiment ID: {experiment_id}\n")
        f.write(f"n_clusters: {config.k_selection.n_clusters}\n")
        f.write(f"Seeds: {config.seeds}\n\n")

        f.write("Results by Method:\n")
        f.write("-" * 40 + "\n")

        for method_name, results in all_results.items():
            f.write(f"\n{method_name}:\n")
            for seed in config.seeds:
                if seed in results:
                    r = results[seed]
                    k = r.get("n_clusters", "N/A")
                    f.write(f"  Seed {seed}: k={k}, labels_file=labels_seed_{seed}.yaml\n")

    print(f"Saved summary to {summary_path}")

    with open(results_path, "wb") as f:
        pkl.dump(all_results, f)
    print(f"Saved all results to {results_path} (methods: {list(all_results.keys())})")

    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)

    return ExperimentResults(results=all_results, config=config)


def main(config_path: str = "configs/benchmark/experiment.yaml"):
    if os.path.exists(config_path):
        cfg = load_config(config_path)
    else:
        cfg = ExperimentConfig()

    run_experiment(cfg)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/benchmark/experiment.yaml"
    main(config_path)
