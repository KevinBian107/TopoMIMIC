import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import numpy as np
import functools
import multiprocessing as mp
import pickle as pkl
import yaml
import ot
from dataclasses import dataclass
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
from sklearn.manifold import MDS
from sklearn.cluster import KMeans

from vnl_tda.utils.load_utils import get_aggregate_data
from vnl_tda.pipelines.tda.skeletonization.postprocess import (
    chunk_loops_by_clustered_vpath,
    _expand_frames_nonconsec,
    _uniform_downsample,
    _chunked_downsample,
    _normalize_dist_matrix,
)
from vnl_tda.pipelines.tda.skeletonization.rendering import render_grid_clusters_side_by_side


M = 3
TAU = 6


@dataclass
class DMConfig:
    pers_delta: float = 0.75
    k_neighbors: int = 5
    epsilon: float = 0.99
    skip: bool = False
    results_dir: str | None = None


@dataclass
class ClusteringConfig:
    min_len: int = 0
    max_len: int = 500
    only_max: bool = True
    only_min: bool = False
    skip_clips: tuple[int, ...] = (37, 152, 163, 274, 281)


@dataclass
class TargetLenSweepConfig:
    values: list[int] = None
    n_clusters: int = 10
    sampling_method: str = "uniform"  # "uniform" or "chunked"

    def __post_init__(self):
        if self.values is None:
            self.values = [30, 50, 70, 90, 110]


@dataclass
class AutoClusterConfig:
    joint_state_key: str = "qpos"
    seed: int = 0


@dataclass
class RenderingConfig:
    pad: int = 0
    max_samples: int = 5
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)


@dataclass
class PipelineConfig:
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    output_dir: str = "out/out_sample_scale"
    num_clips: int = 500
    dm: DMConfig = None
    clustering: ClusteringConfig = None
    target_len_sweep: TargetLenSweepConfig = None
    auto_cluster: AutoClusterConfig = None
    rendering: RenderingConfig = None

    def __post_init__(self):
        if self.dm is None:
            self.dm = DMConfig()
        if self.clustering is None:
            self.clustering = ClusteringConfig()
        if self.target_len_sweep is None:
            self.target_len_sweep = TargetLenSweepConfig()
        if self.auto_cluster is None:
            self.auto_cluster = AutoClusterConfig()
        if self.rendering is None:
            self.rendering = RenderingConfig()


def load_config(config_path: str) -> PipelineConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dm_cfg = DMConfig(**cfg.get("dm", {}))

    clustering_dict = cfg.get("clustering", {})
    if "skip_clips" in clustering_dict:
        clustering_dict["skip_clips"] = tuple(clustering_dict["skip_clips"])
    clustering_cfg = ClusteringConfig(**clustering_dict)

    sweep_dict = cfg.get("target_len_sweep", {})
    sweep_cfg = TargetLenSweepConfig(**sweep_dict)

    auto_cluster_cfg = AutoClusterConfig(**cfg.get("auto_cluster", {}))

    rendering_dict = cfg.get("rendering", {})
    if "target_size" in rendering_dict:
        rendering_dict["target_size"] = tuple(rendering_dict["target_size"])
    rendering_cfg = RenderingConfig(**rendering_dict)

    return PipelineConfig(
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", PipelineConfig.rodent_rollout_dir),
        output_dir=cfg.get("output_dir", PipelineConfig.output_dir),
        num_clips=cfg.get("num_clips", PipelineConfig.num_clips),
        dm=dm_cfg,
        clustering=clustering_cfg,
        target_len_sweep=sweep_cfg,
        auto_cluster=auto_cluster_cfg,
        rendering=rendering_cfg,
    )


def prepare_windows_for_clustering(
    all_cycle_dm: list,
    intention: np.ndarray,
    clustering_cfg: ClusteringConfig,
    target_len: int,
    sampling_method: str = "uniform",
) -> tuple[list[tuple[int, np.ndarray]], list[np.ndarray]]:
    """
    Prepare windows from cycles for clustering with specified target_len.

    Args:
        all_cycle_dm: List of cycles per clip
        intention: Intention array (n_clips, n_frames, D)
        clustering_cfg: Clustering configuration
        target_len: Target length for downsampling
        sampling_method: "uniform" for evenly-spaced sampling,
                        "chunked" for non-overlapping chunks

    Returns:
        Tuple of (windows, cleaned_loops)
    """
    windows = []
    cleaned_loops = []
    n_clips = len(intention)

    for clip_id in range(n_clips):
        if clip_id in clustering_cfg.skip_clips:
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

            if sampling_method == "chunked":
                chunks = _chunked_downsample(expanded, target_len)
                for chunk in chunks:
                    windows.append((clip_id, chunk))
                    cleaned_loops.append(np.asarray(loop, dtype=int))
            else:  # uniform (default)
                sampled = _uniform_downsample(expanded, target_len)
                if sampled is None:
                    continue
                windows.append((clip_id, sampled))
                cleaned_loops.append(np.asarray(loop, dtype=int))

    return windows, cleaned_loops


def compute_w2_distance_matrix(
    windows: list[tuple[int, np.ndarray]],
    intention: np.ndarray,
    target_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise Wasserstein-2 distance matrix.

    Args:
        windows: List of (clip_id, sampled_frames) tuples
        intention: Intention array (n_clips, n_frames, D)
        target_len: Target length for uniform downsampling

    Returns:
        Tuple of (raw_distance_matrix, normalized_distance_matrix)
    """
    M = len(windows)
    D = np.zeros((M, M), dtype=float)

    for i in tqdm(range(M), desc=f"W2 distances (target_len={target_len})"):
        ci, fi = windows[i]
        Xi = intention[ci, fi]
        wi = np.full(target_len, 1.0 / target_len)

        for j in range(i + 1, M):
            cj, fj = windows[j]
            Xj = intention[cj, fj]
            vj = np.full(target_len, 1.0 / target_len)

            C = ot.dist(Xi, Xj)
            val = ot.emd2(wi, vj, C)
            D[i, j] = D[j, i] = val

    Dn = _normalize_dist_matrix(D)
    return D, Dn


def cluster_from_distance_matrix(
    Dn: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
) -> np.ndarray:
    """
    Cluster using precomputed distance matrix via MDS + KMeans.

    Args:
        Dn: Normalized distance matrix
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        Cluster labels array
    """
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=random_state, normalized_stress='auto')
    X_embedded = mds.fit_transform(Dn)
    labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto').fit_predict(X_embedded)
    return labels


def build_cluster_dict(
    labels: np.ndarray,
    windows: list[tuple[int, np.ndarray]],
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Build cluster dictionary from labels and windows."""
    clusters = {}
    for lbl, win in zip(labels, windows):
        clusters.setdefault(int(lbl), []).append(win)
    return clusters


def compute_dunn_index(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    joint_data: np.ndarray,
    target_len: int,
) -> tuple[float, float, float]:
    """
    Compute Dunn Index for a clustering in joint space.

    Args:
        cluster_dict: Dict mapping cluster_id to list of (clip_id, frame_indices)
        joint_data: Joint space array (n_clips, n_frames, D)
        target_len: Target length for uniform downsampling

    Returns:
        Tuple of (dunn_index, min_inter_dist, max_intra_diam)
    """
    cluster_ids = sorted(cluster_dict.keys())
    n_clusters = len(cluster_ids)

    if n_clusters < 2:
        return 0.0, 0.0, 0.0

    cluster_trajectories = {}
    for cid in cluster_ids:
        trajs = []
        for clip_id, frames in cluster_dict[cid]:
            frames = np.asarray(frames, dtype=int)
            sampled = _uniform_downsample(frames, target_len)
            if sampled is None:
                sampled = frames[:target_len] if len(frames) >= target_len else frames
            traj = joint_data[clip_id, sampled].flatten()
            trajs.append(traj)
        cluster_trajectories[cid] = np.array(trajs)

    cluster_diameters = []
    for cid in cluster_ids:
        X = cluster_trajectories[cid]
        if len(X) < 2:
            cluster_diameters.append(0.0)
        else:
            pairwise = pdist(X, metric='euclidean')
            diameter = pairwise.max() if len(pairwise) > 0 else 0.0
            cluster_diameters.append(diameter)

    max_intra_diameter = max(cluster_diameters) if cluster_diameters else 0.0

    min_inter_distance = np.inf
    for i, cid_i in enumerate(cluster_ids):
        for j, cid_j in enumerate(cluster_ids):
            if i >= j:
                continue
            X_i = cluster_trajectories[cid_i]
            X_j = cluster_trajectories[cid_j]
            inter_dists = cdist(X_i, X_j, metric='euclidean')
            min_dist = inter_dists.min()
            min_inter_distance = min(min_inter_distance, min_dist)

    if max_intra_diameter < 1e-10:
        dunn_index = 0.0
    else:
        dunn_index = min_inter_distance / max_intra_diameter

    return dunn_index, min_inter_distance, max_intra_diameter


def run_rendering(
    cluster_dict: dict,
    rodent_rollout_dir: str,
    rendering_cfg: RenderingConfig,
    output_dir: str,
    cluster_labels: dict[int, str] | None = None,
    output_filename: str = "cluster_side_by_side.mp4",
    title: str = "DM Cluster Samples",
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


def plot_sweep_summary(
    sweep_results: dict,
    output_dir: str,
    sampling_method: str = "uniform",
) -> None:
    """Plot summary of target_len sweep results."""
    os.makedirs(output_dir, exist_ok=True)

    target_lens = sorted(sweep_results.keys())
    n_windows = [sweep_results[tl]["n_windows"] for tl in target_lens]
    dunn_indices = [sweep_results[tl]["dunn_index"] for tl in target_lens]
    min_inter = [sweep_results[tl]["min_inter_dist"] for tl in target_lens]
    max_intra = [sweep_results[tl]["max_intra_diam"] for tl in target_lens]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    (ax_windows, ax_dunn), (ax_inter, ax_intra) = axes

    ax_windows.bar(target_lens, n_windows, color='steelblue', alpha=0.7)
    ax_windows.set_xlabel("Target Length", fontsize=12)
    ax_windows.set_ylabel("Number of Windows", fontsize=12)
    ax_windows.set_title(f"Windows by Target Length (method={sampling_method})", fontsize=14)
    ax_windows.grid(True, alpha=0.3)

    ax_dunn.plot(target_lens, dunn_indices, color="darkgreen", marker="o", linewidth=2, markersize=8)
    ax_dunn.set_xlabel("Target Length", fontsize=12)
    ax_dunn.set_ylabel("Dunn Index", fontsize=12)
    ax_dunn.set_title("Dunn Index by Target Length", fontsize=14)
    ax_dunn.grid(True, alpha=0.3)

    ax_inter.plot(target_lens, min_inter, color="blue", marker="s", linewidth=2, markersize=8)
    ax_inter.set_xlabel("Target Length", fontsize=12)
    ax_inter.set_ylabel("Min Inter-cluster Distance", fontsize=12)
    ax_inter.set_title("Min Inter-cluster Distance by Target Length", fontsize=14)
    ax_inter.grid(True, alpha=0.3)

    ax_intra.plot(target_lens, max_intra, color="purple", marker="^", linewidth=2, markersize=8)
    ax_intra.set_xlabel("Target Length", fontsize=12)
    ax_intra.set_ylabel("Max Intra-cluster Diameter", fontsize=12)
    ax_intra.set_title("Max Intra-cluster Diameter by Target Length", fontsize=14)
    ax_intra.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "target_len_sweep_summary.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "target_len_sweep_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sweep summary saved to {output_dir}/target_len_sweep_summary.pdf")


def main(config_path: str = "configs/dmwkc_experiment/sample_scale.yaml"):
    cfg = load_config(config_path)

    if cfg.clustering.only_max and cfg.clustering.only_min:
        raise ValueError("Cannot use both only_max and only_min")

    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=== Stage 1: Loading intention data ===")
    work_rodent = functools.partial(
        get_aggregate_data, "/intention", [], path=cfg.rodent_rollout_dir
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        int_rodent = list(tqdm(pool.imap(work_rodent, range(cfg.num_clips)), total=cfg.num_clips))
    int_rodent = np.vstack(int_rodent)
    int_rodent = int_rodent.reshape([cfg.num_clips, 499, 16])
    print(f"Loaded intention: {int_rodent.shape}")

    print(f"\nLoading joint space data ({cfg.auto_cluster.joint_state_key})...")
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

    print("\n=== Stage 2: Loading DM graphs ===")
    if cfg.dm.skip:
        if cfg.dm.results_dir is None:
            raise ValueError("dm.results_dir required when dm.skip is true")
        print(f"Loading existing DM results from {cfg.dm.results_dir}")
        with open(os.path.join(cfg.dm.results_dir, "all_cycle_dm.pkl"), "rb") as f:
            all_cycle_dm = pkl.load(f)
        print(f"Loaded {len(all_cycle_dm)} cycle sets")
    else:
        raise ValueError("This experiment requires pre-computed DM results (dm.skip must be true)")

    sampling_method = cfg.target_len_sweep.sampling_method
    print(f"\n=== Stage 3: Target Length Sweep ({cfg.target_len_sweep.values}) ===")
    print(f"Sampling method: {sampling_method}")
    sweep_results = {}

    for target_len in cfg.target_len_sweep.values:
        print(f"\n--- Processing target_len={target_len} ---")

        windows, cleaned_loops = prepare_windows_for_clustering(
            all_cycle_dm, int_rodent, cfg.clustering, target_len,
            sampling_method=sampling_method,
        )
        print(f"  Prepared {len(windows)} windows")

        if len(windows) < cfg.target_len_sweep.n_clusters:
            print(f"  Skipping: not enough windows ({len(windows)}) for {cfg.target_len_sweep.n_clusters} clusters")
            sweep_results[target_len] = {
                "n_windows": len(windows),
                "dunn_index": 0.0,
                "min_inter_dist": 0.0,
                "max_intra_diam": 0.0,
                "cluster_dict": {},
            }
            continue

        D, Dn = compute_w2_distance_matrix(windows, int_rodent, target_len)

        labels = cluster_from_distance_matrix(
            Dn, n_clusters=cfg.target_len_sweep.n_clusters, random_state=cfg.auto_cluster.seed
        )
        cluster_dict = build_cluster_dict(labels, windows)

        dunn, min_inter, max_intra = compute_dunn_index(
            cluster_dict, joint_data, target_len
        )

        sweep_results[target_len] = {
            "n_windows": len(windows),
            "dunn_index": dunn,
            "min_inter_dist": min_inter,
            "max_intra_diam": max_intra,
            "cluster_dict": cluster_dict,
        }

        print(f"  Dunn={dunn:.4f}, min_inter={min_inter:.2f}, max_intra={max_intra:.2f}")

        cluster_labels = {i: f"cluster_{i}" for i in cluster_dict.keys()}
        run_rendering(
            cluster_dict,
            rodent_rollout_dir=cfg.rodent_rollout_dir,
            rendering_cfg=cfg.rendering,
            output_dir=cfg.output_dir,
            cluster_labels=cluster_labels,
            output_filename=f"cluster_target_len_{target_len}.mp4",
            title=f"DM Clusters (target_len={target_len}, k={cfg.target_len_sweep.n_clusters})",
        )

    print("\n=== Stage 4: Saving results ===")
    results_path = os.path.join(cfg.output_dir, "sweep_results.pkl")
    with open(results_path, "wb") as f:
        pkl.dump(sweep_results, f)
    print(f"Results saved to {results_path}")

    plot_sweep_summary(sweep_results, cfg.output_dir, sampling_method)

    print("\n=== Summary ===")
    print(f"Sampling method: {sampling_method}")
    print(f"{'target_len':<12} {'n_windows':<12} {'Dunn':<10} {'min_inter':<12} {'max_intra':<12}")
    print("-" * 58)
    for tl in sorted(sweep_results.keys()):
        r = sweep_results[tl]
        print(f"{tl:<12} {r['n_windows']:<12} {r['dunn_index']:<10.4f} {r['min_inter_dist']:<12.2f} {r['max_intra_diam']:<12.2f}")

    print("\n=== Pipeline complete! ===")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dmwkc_experiment/sample_scale.yaml"
    main(config_path)
