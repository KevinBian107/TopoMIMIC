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
from vnl_tda.pipelines.tda.skeletonization.skeletonization import extract_skeleton_pcd_dm
from vnl_tda.pipelines.tda.skeletonization.postprocess import (
    chunk_loops_by_clustered_vpath,
    _expand_frames_nonconsec,
    _uniform_downsample,
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
    target_len: int = 50
    only_max: bool = True
    only_min: bool = False
    skip_clips: tuple[int, ...] = (37, 152, 163, 274, 281)


@dataclass
class AutoClusterConfig:
    enabled: bool = True
    k_min: int = 2
    k_max: int = 15
    joint_state_key: str = "qpos"
    plot_dunn: bool = True
    render_all_k: bool = True
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
    output_dir: str = "out_subspace_dunn"
    num_clips: int = 500
    dm: DMConfig = None
    clustering: ClusteringConfig = None
    auto_cluster: AutoClusterConfig = None
    rendering: RenderingConfig = None

    def __post_init__(self):
        if self.dm is None:
            self.dm = DMConfig()
        if self.clustering is None:
            self.clustering = ClusteringConfig()
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

    auto_cluster_dict = cfg.get("auto_cluster", {})
    if "plot_cdi" in auto_cluster_dict:
        auto_cluster_dict["plot_dunn"] = auto_cluster_dict.pop("plot_cdi")
    auto_cluster_cfg = AutoClusterConfig(**auto_cluster_dict)

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
        auto_cluster=auto_cluster_cfg,
        rendering=rendering_cfg,
    )


def takens_points(X: np.ndarray, m: int = M, tau: int = TAU) -> np.ndarray:
    n_frames, D = X.shape
    start = (m - 1) * tau
    idxs = np.arange(start, n_frames)
    pts = np.stack([X[[j - a * tau for a in range(m)]].reshape(-1) for j in idxs], axis=0)
    return pts


def run_dm_extraction(
    int_rodent: np.ndarray,
    num_clips: int,
    dm_cfg: DMConfig,
    base_output_dir: str,
) -> tuple[list, list]:
    all_cycle_dm = []
    all_G_sm = []

    os.makedirs(base_output_dir, exist_ok=True)

    for clip_id in tqdm(range(num_clips), desc="DM extraction"):
        clip = int_rodent[clip_id]
        pts = takens_points(clip, m=M, tau=TAU)
        clip_output_dir = os.path.join(base_output_dir, f"clip_{clip_id}")
        cycle_inds, G_sm = extract_skeleton_pcd_dm(
            pts,
            k=dm_cfg.k_neighbors,
            pers_delta=dm_cfg.pers_delta,
            epsilon=dm_cfg.epsilon,
            output_dir=clip_output_dir,
            cleanup=False,
        )
        cycle_inds = [np.array(inds, dtype=int) for inds in cycle_inds]
        all_cycle_dm.append(cycle_inds)
        all_G_sm.append(G_sm)
        print(f"Clip {clip_id}: {len(cycle_inds)} cycles, {G_sm.number_of_edges()} edges")

    with open(os.path.join(base_output_dir, "all_G_sm.pkl"), "wb") as f:
        pkl.dump(all_G_sm, f)
    with open(os.path.join(base_output_dir, "all_cycle_dm.pkl"), "wb") as f:
        pkl.dump(all_cycle_dm, f)

    return all_cycle_dm, all_G_sm


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


def compute_w2_distance_matrix(
    windows: list[tuple[int, np.ndarray]],
    intention: np.ndarray,
    target_len: int,
    squared_cost: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise Wasserstein-2 distance matrix (computed once, reused for all k).

    Args:
        windows: List of (clip_id, sampled_frames) tuples
        intention: Intention array (n_clips, n_frames, D)
        target_len: Target length for uniform downsampling
        squared_cost: Whether to square the cost matrix

    Returns:
        Tuple of (raw_distance_matrix, normalized_distance_matrix)
    """
    M = len(windows)
    D = np.zeros((M, M), dtype=float)

    print(f"Computing W2 distance matrix for {M} windows...")
    for i in tqdm(range(M), desc="W2 distances"):
        ci, fi = windows[i]
        Xi = intention[ci, fi]
        wi = np.full(target_len, 1.0 / target_len)

        for j in range(i + 1, M):
            cj, fj = windows[j]
            Xj = intention[cj, fj]
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
    """
    Build cluster dictionary from labels and windows.

    Args:
        labels: Cluster labels array
        windows: List of (clip_id, sampled_frames) tuples

    Returns:
        Dict mapping cluster_id to list of (clip_id, frame_indices)
    """
    clusters = {}
    for lbl, win in zip(labels, windows):
        clusters.setdefault(int(lbl), []).append(win)
    return clusters


def compute_dunn_index_from_cluster_dict(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    joint_data: np.ndarray,
    target_len: int,
) -> tuple[float, float, float]:
    """
    Compute Dunn Index for a clustering in joint space.

    Dunn = min(inter-cluster distance) / max(intra-cluster diameter)

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


def compute_per_cluster_stats(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    joint_data: np.ndarray,
    target_len: int,
) -> dict:
    """Compute per-cluster statistics (diameter, size) in joint space."""
    cluster_ids = sorted(cluster_dict.keys())
    stats = {}

    for cid in cluster_ids:
        trajs = []
        for clip_id, frames in cluster_dict[cid]:
            frames = np.asarray(frames, dtype=int)
            sampled = _uniform_downsample(frames, target_len)
            if sampled is None:
                sampled = frames[:target_len] if len(frames) >= target_len else frames
            traj = joint_data[clip_id, sampled].flatten()
            trajs.append(traj)

        X = np.array(trajs)
        n_points = len(X)

        if n_points < 2:
            diameter = 0.0
        else:
            pairwise = pdist(X, metric='euclidean')
            diameter = pairwise.max() if len(pairwise) > 0 else 0.0

        stats[int(cid)] = {
            "size": n_points,
            "diameter": diameter,
        }

    return stats


def auto_select_clusters(
    windows: list[tuple[int, np.ndarray]],
    Dn: np.ndarray,
    joint_data: np.ndarray,
    clustering_cfg: ClusteringConfig,
    auto_cfg: AutoClusterConfig,
    output_dir: str,
) -> tuple[int, dict, dict[int, dict]]:
    """
    Automatically select optimal number of clusters using Dunn Index.

    Clustering is done via MDS + KMeans on precomputed W2 distance matrix.
    Dunn Index is computed on joint space.

    Args:
        windows: List of (clip_id, sampled_frames) tuples
        Dn: Normalized W2 distance matrix (precomputed)
        joint_data: Joint space array (n_clips, n_frames, J)
        clustering_cfg: Clustering configuration
        auto_cfg: Auto cluster selection configuration
        output_dir: Output directory for plots

    Returns:
        Tuple of (optimal_k, cluster_dict, all_cluster_dicts)
    """
    print(f"\n=== Dunn Index Cluster Selection (k={auto_cfg.k_min} to {auto_cfg.k_max}) ===")
    print("Clustering: MDS + KMeans on precomputed W2 distance matrix")
    print("Scoring: Dunn Index on joint space")

    k_range = range(auto_cfg.k_min, auto_cfg.k_max + 1)
    results = {
        "dunn_index": [],
        "min_inter_dist": [],
        "max_intra_diam": [],
        "per_cluster_stats": {},
    }
    all_cluster_dicts = {}

    for k in tqdm(k_range, desc="Evaluating k values"):
        labels = cluster_from_distance_matrix(Dn, n_clusters=k, random_state=auto_cfg.seed)
        cluster_dict_k = build_cluster_dict(labels, windows)

        dunn, min_inter, max_intra = compute_dunn_index_from_cluster_dict(
            cluster_dict_k, joint_data, clustering_cfg.target_len
        )
        per_cluster = compute_per_cluster_stats(
            cluster_dict_k, joint_data, clustering_cfg.target_len
        )

        results["dunn_index"].append(dunn)
        results["min_inter_dist"].append(min_inter)
        results["max_intra_diam"].append(max_intra)
        results["per_cluster_stats"][k] = per_cluster
        all_cluster_dicts[k] = cluster_dict_k

    dunn_arr = np.array(results["dunn_index"])
    optimal_idx = np.argmax(dunn_arr)
    optimal_k = list(k_range)[optimal_idx]
    optimal_dunn = results["dunn_index"][optimal_idx]
    optimal_min_inter = results["min_inter_dist"][optimal_idx]
    optimal_max_intra = results["max_intra_diam"][optimal_idx]

    print(f"\nMetrics by k (Dunn | min_inter | max_intra):")
    for i, k in enumerate(k_range):
        dunn = results["dunn_index"][i]
        min_inter = results["min_inter_dist"][i]
        max_intra = results["max_intra_diam"][i]
        marker = " <-- optimal" if k == optimal_k else ""
        print(f"  k={k}: Dunn={dunn:.4f} | min_inter={min_inter:.2f} | max_intra={max_intra:.2f}{marker}")

    print(f"\nOptimal k={optimal_k}: Dunn={optimal_dunn:.4f}, min_inter={optimal_min_inter:.2f}, max_intra={optimal_max_intra:.2f}")

    if auto_cfg.plot_dunn:
        plot_dunn_analysis(k_range, results, optimal_k, output_dir)

    cluster_dict = all_cluster_dicts[optimal_k]

    return optimal_k, cluster_dict, all_cluster_dicts


def plot_dunn_analysis(
    k_range: range,
    results: dict,
    optimal_k: int,
    output_dir: str,
) -> None:
    """Plot Dunn Index analysis."""
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


def run_rendering(
    cluster_dict: dict,
    rodent_rollout_dir: str,
    rendering_cfg: RenderingConfig,
    output_dir: str,
    cluster_labels: dict[int, str] | None = None,
    output_filename: str = "cluster_side_by_side.mp4",
    title: str = "DM Cluster Samples (Dunn-selected)",
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


def main(config_path: str = "configs/dm_latent_subspace_auto_dunn.yaml"):
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

    print("\n=== Stage 2: Loading/Computing DM graphs ===")
    if cfg.dm.skip:
        if cfg.dm.results_dir is None:
            raise ValueError("dm.results_dir required when dm.skip is true")
        print(f"Loading existing DM results from {cfg.dm.results_dir}")
        with open(os.path.join(cfg.dm.results_dir, "all_cycle_dm.pkl"), "rb") as f:
            all_cycle_dm = pkl.load(f)
        print(f"Loaded {len(all_cycle_dm)} cycle sets")
    else:
        base_output_dir = os.path.join(
            cfg.output_dir, f"dm_results_m{M}_tau{TAU}_p{cfg.dm.pers_delta}"
        )
        all_cycle_dm, _ = run_dm_extraction(
            int_rodent,
            num_clips=cfg.num_clips,
            dm_cfg=cfg.dm,
            base_output_dir=base_output_dir,
        )

    print("\n=== Stage 3: Preparing windows and computing W2 distance matrix ===")
    windows, cleaned_loops = prepare_windows_for_clustering(
        all_cycle_dm, int_rodent, cfg.clustering
    )
    print(f"Prepared {len(windows)} windows for clustering")

    if len(windows) == 0:
        raise RuntimeError("No valid windows for clustering")

    D, Dn = compute_w2_distance_matrix(
        windows, int_rodent, cfg.clustering.target_len, squared_cost=False
    )

    w2_cache_path = os.path.join(cfg.output_dir, "w2_distance_matrix.pkl")
    with open(w2_cache_path, "wb") as f:
        pkl.dump({"D": D, "Dn": Dn, "windows": windows}, f)
    print(f"Cached W2 distance matrix to {w2_cache_path}")

    print("\n=== Stage 4: Dunn Index cluster selection (W2 clustering, joint-space scoring) ===")
    optimal_k, cluster_dict, all_cluster_dicts = auto_select_clusters(
        windows=windows,
        Dn=Dn,
        joint_data=joint_data,
        clustering_cfg=cfg.clustering,
        auto_cfg=cfg.auto_cluster,
        output_dir=cfg.output_dir,
    )

    print(f"\n=== Stage 5: Saving results ===")
    cluster_output_dir = os.path.join(cfg.output_dir, "clustering")
    os.makedirs(cluster_output_dir, exist_ok=True)

    with open(os.path.join(cluster_output_dir, "cluster_dict.pkl"), "wb") as f:
        pkl.dump(cluster_dict, f)
    with open(os.path.join(cluster_output_dir, "all_cluster_dicts.pkl"), "wb") as f:
        pkl.dump(all_cluster_dicts, f)
    with open(os.path.join(cluster_output_dir, "optimal_k.txt"), "w") as f:
        f.write(f"optimal_k={optimal_k}\n")

    print(f"Created {len(cluster_dict)} clusters (Dunn-selected k={optimal_k})")
    for cid, entries in sorted(cluster_dict.items()):
        print(f"  Cluster {cid}: {len(entries)} samples")

    print("\n=== Stage 6: Rendering cluster videos ===")
    if cfg.auto_cluster.render_all_k:
        k_sweep_dir = os.path.join(cfg.output_dir, "k_sweep")
        os.makedirs(k_sweep_dir, exist_ok=True)
        print(f"Rendering videos for all k values ({cfg.auto_cluster.k_min} to {cfg.auto_cluster.k_max})...")
        for k, k_cluster_dict in tqdm(all_cluster_dicts.items(), desc="Rendering k sweep"):
            k_labels = {i: f"cluster_{i}" for i in k_cluster_dict.keys()}
            is_optimal = " (optimal)" if k == optimal_k else ""
            run_rendering(
                k_cluster_dict,
                rodent_rollout_dir=cfg.rodent_rollout_dir,
                rendering_cfg=cfg.rendering,
                output_dir=k_sweep_dir,
                cluster_labels=k_labels,
                output_filename=f"cluster_k{k}.mp4",
                title=f"DM Clusters (k={k}){is_optimal}",
            )
    else:
        cluster_labels = {i: f"cluster_{i}" for i in cluster_dict.keys()}
        run_rendering(
            cluster_dict,
            rodent_rollout_dir=cfg.rodent_rollout_dir,
            rendering_cfg=cfg.rendering,
            output_dir=cfg.output_dir,
            cluster_labels=cluster_labels,
            output_filename=f"cluster_k{optimal_k}_optimal.mp4",
            title=f"DM Clusters (k={optimal_k}, Dunn-optimal)",
        )

    print("\n=== Pipeline complete! ===")
    print(f"Optimal number of clusters: {optimal_k}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dm_latent_subspace_auto_dunn.yaml"
    main(config_path)
