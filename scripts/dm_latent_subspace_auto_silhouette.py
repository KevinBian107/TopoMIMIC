import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import numpy as np
import functools
import multiprocessing as mp
import pickle as pkl
import yaml
import h5py
from dataclasses import dataclass
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, silhouette_samples
import ot

from vnl_tda.utils.load_utils import get_aggregate_data, load_from_h5py
from vnl_tda.pipelines.tda.skeletonization.skeletonization import extract_skeleton_pcd_dm
from vnl_tda.pipelines.tda.skeletonization.postprocess import (
    chunk_loops_by_clustered_vpath,
    _expand_frames_nonconsec,
    _uniform_downsample,
    _normalize_dist_matrix,
    drop_middle_nearby_loops,
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
    metric: str = "both"  # "intention", "joint", or "both"
    joint_state_key: str = "qpos"
    plot_silhouette: bool = True
    render_all_k: bool = True  # render video for each k value


@dataclass
class RenderingConfig:
    pad: int = 0
    max_samples: int = 5
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)


@dataclass
class PipelineConfig:
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    output_dir: str = "out_subspace_auto"
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
    n_clips: int,
) -> tuple[list[tuple[int, np.ndarray]], list[np.ndarray]]:
    """Prepare windows (clip_id, sampled_frames) for clustering."""
    windows = []
    cleaned_loops = []

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


def compute_wasserstein_distance_matrix(
    windows: list[tuple[int, np.ndarray]],
    data: np.ndarray,
    target_len: int,
) -> np.ndarray:
    """Compute pairwise Wasserstein distance matrix."""
    M_samples = len(windows)
    D = np.zeros((M_samples, M_samples), dtype=float)

    for i, (ci, fi) in enumerate(tqdm(windows, desc="Computing W2 distances")):
        Xi = data[ci, fi]
        wi = np.full(target_len, 1.0 / target_len)
        for j in range(i + 1, M_samples):
            cj, fj = windows[j]
            Xj = data[cj, fj]
            vj = np.full(target_len, 1.0 / target_len)
            C = ot.dist(Xi, Xj)
            val = ot.emd2(wi, vj, C)
            D[i, j] = D[j, i] = val

    return D


def compute_silhouette_for_k(
    D_normalized: np.ndarray,
    k: int,
    random_state: int = 0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute silhouette score for a given k using precomputed distance matrix.

    Returns:
        Tuple of (mean_silhouette, labels, X_embedded)
    """
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=random_state, normalized_stress='auto')
    X_embedded = mds.fit_transform(D_normalized)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(X_embedded)

    if len(np.unique(labels)) < 2:
        return -1.0, labels, X_embedded

    sil_score = silhouette_score(X_embedded, labels)
    return sil_score, labels, X_embedded


def auto_select_clusters(
    windows: list[tuple[int, np.ndarray]],
    intention: np.ndarray,
    joint_data: np.ndarray | None,
    clustering_cfg: ClusteringConfig,
    auto_cfg: AutoClusterConfig,
    output_dir: str,
) -> tuple[int, dict, np.ndarray, dict[int, dict]]:
    """
    Automatically select optimal number of clusters using silhouette score.

    Args:
        windows: List of (clip_id, sampled_frames) tuples
        intention: Intention array (n_clips, n_frames, D)
        joint_data: Joint space array (n_clips, n_frames, J) or None
        clustering_cfg: Clustering configuration
        auto_cfg: Auto cluster selection configuration
        output_dir: Output directory for plots

    Returns:
        Tuple of (optimal_k, cluster_dict, labels, all_cluster_dicts)
        where all_cluster_dicts maps k -> cluster_dict for all evaluated k values
    """
    print(f"\n=== Auto Cluster Selection (k={auto_cfg.k_min} to {auto_cfg.k_max}) ===")

    D_intention = compute_wasserstein_distance_matrix(
        windows, intention, clustering_cfg.target_len
    )
    Dn_intention = _normalize_dist_matrix(D_intention)

    if auto_cfg.metric in ["joint", "both"] and joint_data is not None:
        print("Computing joint space distance matrix...")
        D_joint = compute_wasserstein_distance_matrix(
            windows, joint_data, clustering_cfg.target_len
        )
        Dn_joint = _normalize_dist_matrix(D_joint)
    else:
        Dn_joint = None

    k_range = range(auto_cfg.k_min, auto_cfg.k_max + 1)
    results = {
        "intention": {"scores": [], "labels": {}, "embeddings": {}},
    }
    if Dn_joint is not None:
        results["joint"] = {"scores": [], "labels": {}, "embeddings": {}}
        results["combined"] = {"scores": [], "labels": {}, "embeddings": {}}

    for k in tqdm(k_range, desc="Evaluating cluster counts"):
        sil_int, labels_int, X_int = compute_silhouette_for_k(Dn_intention, k)
        results["intention"]["scores"].append(sil_int)
        results["intention"]["labels"][k] = labels_int
        results["intention"]["embeddings"][k] = X_int

        if Dn_joint is not None:
            sil_joint, labels_joint, X_joint = compute_silhouette_for_k(Dn_joint, k)
            results["joint"]["scores"].append(sil_joint)
            results["joint"]["labels"][k] = labels_joint
            results["joint"]["embeddings"][k] = X_joint

            Dn_combined = (Dn_intention + Dn_joint) / 2
            sil_comb, labels_comb, X_comb = compute_silhouette_for_k(Dn_combined, k)
            results["combined"]["scores"].append(sil_comb)
            results["combined"]["labels"][k] = labels_comb
            results["combined"]["embeddings"][k] = X_comb

    if auto_cfg.metric == "intention":
        scores = results["intention"]["scores"]
        labels_dict = results["intention"]["labels"]
    elif auto_cfg.metric == "joint" and Dn_joint is not None:
        scores = results["joint"]["scores"]
        labels_dict = results["joint"]["labels"]
    elif auto_cfg.metric == "both" and Dn_joint is not None:
        scores = results["combined"]["scores"]
        labels_dict = results["combined"]["labels"]
    else:
        scores = results["intention"]["scores"]
        labels_dict = results["intention"]["labels"]

    optimal_idx = np.argmax(scores)
    optimal_k = list(k_range)[optimal_idx]
    optimal_score = scores[optimal_idx]
    optimal_labels = labels_dict[optimal_k]

    print(f"\nSilhouette scores by k:")
    for k, score in zip(k_range, scores):
        marker = " <-- optimal" if k == optimal_k else ""
        print(f"  k={k}: {score:.4f}{marker}")

    print(f"\nOptimal k={optimal_k} with silhouette score={optimal_score:.4f}")

    if auto_cfg.plot_silhouette:
        plot_silhouette_analysis(
            k_range, results, optimal_k, auto_cfg.metric, output_dir
        )

    all_cluster_dicts = {}
    for k in k_range:
        k_labels = labels_dict[k]
        k_cluster_dict = {}
        for lbl, win in zip(k_labels, windows):
            k_cluster_dict.setdefault(int(lbl), []).append(win)
        all_cluster_dicts[k] = k_cluster_dict

    cluster_dict = all_cluster_dicts[optimal_k]

    return optimal_k, cluster_dict, optimal_labels, all_cluster_dicts


def plot_silhouette_analysis(
    k_range: range,
    results: dict,
    optimal_k: int,
    metric: str,
    output_dir: str,
) -> None:
    """Plot silhouette scores for different k values."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"intention": "blue", "joint": "green", "combined": "red"}
    markers = {"intention": "o", "joint": "s", "combined": "^"}

    for space_name, data in results.items():
        scores = data["scores"]
        ax.plot(
            list(k_range), scores,
            color=colors.get(space_name, "gray"),
            marker=markers.get(space_name, "o"),
            label=f"{space_name.capitalize()} space",
            linewidth=2,
            markersize=8
        )

    ax.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax.set_xlabel("Number of clusters (k)", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title(f"Silhouette Analysis for Cluster Selection (metric={metric})", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(list(k_range))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "silhouette_analysis.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "silhouette_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Silhouette plot saved to {output_dir}/silhouette_analysis.pdf")


def run_rendering(
    cluster_dict: dict,
    rodent_rollout_dir: str,
    rendering_cfg: RenderingConfig,
    output_dir: str,
    cluster_labels: dict[int, str] | None = None,
    output_filename: str = "cluster_side_by_side.mp4",
    title: str = "DM Cluster Samples (Auto-selected)",
) -> None:
    cluster_chunks = chunk_loops_by_clustered_vpath(
        cluster_dict,
        rodent_rollout_dir,
        pad=rendering_cfg.pad,
        sample=rendering_cfg.max_samples,
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
    )
    print(f"Video saved to: {out_path}")


def main(config_path: str = "configs/dm_latent_subspace_auto.yaml"):
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

    joint_data = None
    if cfg.auto_cluster.metric in ["joint", "both"]:
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
            clip.reshape(n_frames, joint_dim)[1:]  # skip first frame to match intention (499 frames)
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

    print("\n=== Stage 3: Preparing windows for clustering ===")
    windows, cleaned_loops = prepare_windows_for_clustering(
        all_cycle_dm, int_rodent, cfg.clustering, cfg.num_clips
    )
    print(f"Prepared {len(windows)} windows for clustering")

    if len(windows) == 0:
        raise RuntimeError("No valid windows for clustering")

    print("\n=== Stage 4: Auto cluster selection ===")
    optimal_k, cluster_dict, labels, all_cluster_dicts = auto_select_clusters(
        windows=windows,
        intention=int_rodent,
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
    with open(os.path.join(cluster_output_dir, "cleaned_loops.pkl"), "wb") as f:
        pkl.dump(cleaned_loops, f)
    with open(os.path.join(cluster_output_dir, "optimal_k.txt"), "w") as f:
        f.write(f"optimal_k={optimal_k}\n")

    print(f"Created {len(cluster_dict)} clusters (auto-selected k={optimal_k})")
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
            title=f"DM Clusters (k={optimal_k}, optimal)",
        )

    print("\n=== Pipeline complete! ===")
    print(f"Optimal number of clusters: {optimal_k}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dm_latent_subspace_auto.yaml"
    main(config_path)
