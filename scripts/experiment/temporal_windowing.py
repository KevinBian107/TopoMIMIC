"""Temporal Windowing DM Pipeline.

Splits clips into smaller temporal segments before running DM extraction and clustering.
"""

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

from scipy.spatial.distance import pdist, cdist
from sklearn.manifold import MDS
from sklearn.cluster import KMeans

from vnl_tda.utils.load_utils import get_aggregate_data, split_clips_by_frames
from vnl_tda.pipelines.tda.skeletonization.skeletonization import extract_skeleton_pcd_dm
from vnl_tda.pipelines.tda.skeletonization.postprocess import (
    chunk_loops_by_clustered_vpath,
    _expand_frames_nonconsec,
    _uniform_downsample,
    _normalize_dist_matrix,
)
from vnl_tda.pipelines.tda.skeletonization.rendering import render_grid_clusters_side_by_side


@dataclass
class TakensConfig:
    m: int = 3
    tau: int = 6


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
    target_len: int = 70
    only_max: bool = True
    only_min: bool = False
    skip_clips: tuple[int, ...] = (37, 152, 163, 274, 281)


@dataclass
class AutoClusterConfig:
    joint_state_key: str = "qpos"
    k_min: int = 5
    k_max: int = 20
    plot_dunn: bool = True
    render_all_k: bool = False
    seed: int = 0


@dataclass
class RenderingConfig:
    skip: bool = False
    pad: int = 0
    max_samples: int = 5
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)


@dataclass
class WindowingConfig:
    frames_per_clip: int = 100


@dataclass
class WindowingPipelineConfig:
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    output_dir: str = "out/temporal_windowing"
    num_clips: int = 500
    takens: TakensConfig = None
    windowing: WindowingConfig = None
    dm: DMConfig = None
    clustering: ClusteringConfig = None
    auto_cluster: AutoClusterConfig = None
    rendering: RenderingConfig = None

    def __post_init__(self):
        if self.takens is None:
            self.takens = TakensConfig()
        if self.windowing is None:
            self.windowing = WindowingConfig()
        if self.dm is None:
            self.dm = DMConfig()
        if self.clustering is None:
            self.clustering = ClusteringConfig()
        if self.auto_cluster is None:
            self.auto_cluster = AutoClusterConfig()
        if self.rendering is None:
            self.rendering = RenderingConfig()


def load_config(config_path: str) -> WindowingPipelineConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    takens_cfg = TakensConfig(**cfg.get("takens", {}))

    windowing_dict = cfg.get("windowing", cfg.get("multiscale", {}))
    windowing_cfg = WindowingConfig(**windowing_dict)

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

    return WindowingPipelineConfig(
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", WindowingPipelineConfig.rodent_rollout_dir),
        output_dir=cfg.get("output_dir", WindowingPipelineConfig.output_dir),
        num_clips=cfg.get("num_clips", WindowingPipelineConfig.num_clips),
        takens=takens_cfg,
        windowing=windowing_cfg,
        dm=dm_cfg,
        clustering=clustering_cfg,
        auto_cluster=auto_cluster_cfg,
        rendering=rendering_cfg,
    )


def takens_embedding(X: np.ndarray, m: int, tau: int) -> np.ndarray:
    n_frames, D = X.shape
    start = (m - 1) * tau
    idxs = np.arange(start, n_frames)
    pts = np.stack([X[[j - a * tau for a in range(m)]].reshape(-1) for j in idxs], axis=0)
    return pts


def run_dm_extraction(
    data: np.ndarray,
    num_clips: int,
    dm_cfg: DMConfig,
    takens_cfg: TakensConfig,
    base_output_dir: str,
) -> tuple[list, list]:
    os.makedirs(base_output_dir, exist_ok=True)

    all_cycle_dm = []
    all_G_sm = []

    for clip_id in tqdm(range(num_clips), desc="DM extraction"):
        clip = data[clip_id]
        pts = takens_embedding(clip, m=takens_cfg.m, tau=takens_cfg.tau)

        clip_output_dir = os.path.join(base_output_dir, "clips", f"clip_{clip_id}")
        os.makedirs(clip_output_dir, exist_ok=True)

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

    with open(os.path.join(base_output_dir, "all_cycle_dm.pkl"), "wb") as f:
        pkl.dump(all_cycle_dm, f)
    with open(os.path.join(base_output_dir, "all_G_sm.pkl"), "wb") as f:
        pkl.dump(all_G_sm, f)

    return all_cycle_dm, all_G_sm


def prepare_windows_for_clustering(
    all_cycle_dm: list,
    intention: np.ndarray,
    clustering_cfg: ClusteringConfig,
) -> tuple[list[tuple[int, np.ndarray]], list[np.ndarray]]:
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

    return windows, cleaned_loops


def compute_w2_distance_matrix(
    windows: list[tuple[int, np.ndarray]],
    intention: np.ndarray,
    target_len: int,
    squared_cost: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    M = len(windows)
    D = np.zeros((M, M), dtype=float)

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
            D[i, j] = D[j, i] = val

    Dn = _normalize_dist_matrix(D)
    return D, Dn


def cluster_from_distance_matrix(
    Dn: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
) -> np.ndarray:
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=random_state, normalized_stress='auto')
    X_embedded = mds.fit_transform(Dn)
    labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto').fit_predict(X_embedded)
    return labels


def build_cluster_dict(
    labels: np.ndarray,
    windows: list[tuple[int, np.ndarray]],
) -> dict[int, list[tuple[int, np.ndarray]]]:
    clusters = {}
    for lbl, win in zip(labels, windows):
        clusters.setdefault(int(lbl), []).append(win)
    return clusters


def compute_dunn_index(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    joint_data: np.ndarray,
    target_len: int,
) -> tuple[float, float, float]:
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


def auto_select_clusters(
    windows: list[tuple[int, np.ndarray]],
    Dn: np.ndarray,
    joint_data: np.ndarray,
    clustering_cfg: ClusteringConfig,
    auto_cfg: AutoClusterConfig,
    output_dir: str,
) -> tuple[int, dict, dict]:
    all_cluster_dicts = {}
    dunn_scores = {}

    for k in range(auto_cfg.k_min, auto_cfg.k_max + 1):
        labels = cluster_from_distance_matrix(Dn, n_clusters=k, random_state=auto_cfg.seed)
        cluster_dict = build_cluster_dict(labels, windows)
        dunn, _, _ = compute_dunn_index(cluster_dict, joint_data, clustering_cfg.target_len)

        all_cluster_dicts[k] = cluster_dict
        dunn_scores[k] = dunn
        print(f"  k={k}: Dunn={dunn:.4f}")

    optimal_k = max(dunn_scores, key=dunn_scores.get)
    print(f"\nOptimal k={optimal_k} with Dunn={dunn_scores[optimal_k]:.4f}")

    if auto_cfg.plot_dunn:
        import matplotlib.pyplot as plt
        os.makedirs(output_dir, exist_ok=True)

        ks = sorted(dunn_scores.keys())
        scores = [dunn_scores[k] for k in ks]

        plt.figure(figsize=(8, 5))
        plt.plot(ks, scores, 'o-', linewidth=2, markersize=8)
        plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Dunn Index')
        plt.title('Dunn Index vs Number of Clusters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "dunn_index.png"), dpi=150, bbox_inches='tight')
        plt.close()

    return optimal_k, all_cluster_dicts[optimal_k], all_cluster_dicts


def run_rendering(
    cluster_dict: dict,
    rodent_rollout_dir: str,
    rendering_cfg: RenderingConfig,
    output_dir: str,
    cluster_labels: dict[int, str] | None = None,
    output_filename: str = "cluster_side_by_side.mp4",
    title: str = "DM Cluster Samples",
) -> None:
    if rendering_cfg.skip:
        print("Rendering skipped")
        return

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


def main(config_path: str = "configs/experiment/temporal_windowing.yaml"):
    cfg = load_config(config_path)

    if cfg.clustering.only_max and cfg.clustering.only_min:
        raise ValueError("Cannot use both only_max and only_min")

    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")
    os.makedirs(cfg.output_dir, exist_ok=True)

    frames_per_clip = cfg.windowing.frames_per_clip
    print(f"=== Temporal Windowing Pipeline (frames_per_clip={frames_per_clip}) ===")

    print("\n=== Stage 1: Loading intention data ===")
    work_rodent = functools.partial(
        get_aggregate_data, "/intention", [], path=cfg.rodent_rollout_dir
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        int_rodent = list(tqdm(pool.imap(work_rodent, range(cfg.num_clips)), total=cfg.num_clips))
    int_rodent = np.vstack(int_rodent)
    print(f"Loaded intention (raw): {int_rodent.shape}")

    int_rodent_split = split_clips_by_frames(
        int_rodent, original_num_clips=cfg.num_clips, frames_per_clip=frames_per_clip
    )
    num_split_clips = int_rodent_split.shape[0]
    print(f"Split intention: {int_rodent_split.shape} ({num_split_clips} clips)")

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

    joint_data_raw = np.vstack([clip.reshape(n_frames, joint_dim)[1:] for clip in joint_list])
    print(f"Loaded joint data (raw): {joint_data_raw.shape}")

    joint_data_split = split_clips_by_frames(
        joint_data_raw, original_num_clips=cfg.num_clips, frames_per_clip=frames_per_clip
    )
    print(f"Split joint data: {joint_data_split.shape}")

    print("\n=== Stage 2: Computing DM graphs on split clips ===")
    if cfg.dm.skip:
        if cfg.dm.results_dir is None:
            raise ValueError("dm.results_dir required when dm.skip is true")
        print(f"Loading existing DM results from {cfg.dm.results_dir}")
        with open(os.path.join(cfg.dm.results_dir, "all_cycle_dm.pkl"), "rb") as f:
            all_cycle_dm = pkl.load(f)
        print(f"Loaded {len(all_cycle_dm)} cycle sets")
    else:
        base_output_dir = os.path.join(
            cfg.output_dir, f"dm_results_m{cfg.takens.m}_tau{cfg.takens.tau}_p{cfg.dm.pers_delta}_f{frames_per_clip}"
        )
        all_cycle_dm, _ = run_dm_extraction(
            int_rodent_split,
            num_clips=num_split_clips,
            dm_cfg=cfg.dm,
            takens_cfg=cfg.takens,
            base_output_dir=base_output_dir,
        )

    print("\n=== Stage 3: Preparing windows and computing W2 distance matrix ===")
    windows, cleaned_loops = prepare_windows_for_clustering(
        all_cycle_dm, int_rodent_split, cfg.clustering
    )
    print(f"Prepared {len(windows)} windows for clustering")

    if len(windows) == 0:
        raise RuntimeError("No valid windows for clustering")

    D, Dn = compute_w2_distance_matrix(
        windows, int_rodent_split, cfg.clustering.target_len, squared_cost=False
    )

    w2_cache_path = os.path.join(cfg.output_dir, "w2_distance_matrix.pkl")
    with open(w2_cache_path, "wb") as f:
        pkl.dump({"D": D, "Dn": Dn, "windows": windows}, f)
    print(f"Cached W2 distance matrix to {w2_cache_path}")

    print("\n=== Stage 4: Dunn Index cluster selection ===")
    optimal_k, cluster_dict, all_cluster_dicts = auto_select_clusters(
        windows=windows,
        Dn=Dn,
        joint_data=joint_data_split,
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
        f.write(f"frames_per_clip={frames_per_clip}\n")
        f.write(f"num_split_clips={num_split_clips}\n")

    print(f"Created {len(cluster_dict)} clusters (Dunn-selected k={optimal_k})")
    for cid, entries in sorted(cluster_dict.items()):
        print(f"  Cluster {cid}: {len(entries)} samples")

    print("\n=== Stage 6: Rendering cluster videos ===")
    original_frames_per_clip = 499
    usable_frames = frames_per_clip - 1
    splits_per_original = original_frames_per_clip // usable_frames

    cluster_dict_original = {}
    for cid, entries in cluster_dict.items():
        cluster_dict_original[cid] = []
        for split_clip_idx, frames in entries:
            frames_np = np.asarray(frames, dtype=int)
            original_clip = split_clip_idx // splits_per_original
            segment_idx = split_clip_idx % splits_per_original
            original_frames = segment_idx * usable_frames + frames_np
            cluster_dict_original[cid].append((original_clip, original_frames))

    if cfg.auto_cluster.render_all_k:
        k_sweep_dir = os.path.join(cfg.output_dir, "k_sweep")
        os.makedirs(k_sweep_dir, exist_ok=True)
        print(f"Rendering videos for all k values ({cfg.auto_cluster.k_min} to {cfg.auto_cluster.k_max})...")
        for k, k_cluster_dict in tqdm(all_cluster_dicts.items(), desc="Rendering k sweep"):
            k_cluster_dict_original = {}
            for cid, entries in k_cluster_dict.items():
                k_cluster_dict_original[cid] = []
                for split_clip_idx, frames in entries:
                    frames_np = np.asarray(frames, dtype=int)
                    original_clip = split_clip_idx // splits_per_original
                    segment_idx = split_clip_idx % splits_per_original
                    original_frames = segment_idx * usable_frames + frames_np
                    k_cluster_dict_original[cid].append((original_clip, original_frames))

            k_labels = {i: f"cluster_{i}" for i in k_cluster_dict_original.keys()}
            is_optimal = " (optimal)" if k == optimal_k else ""
            run_rendering(
                k_cluster_dict_original,
                rodent_rollout_dir=cfg.rodent_rollout_dir,
                rendering_cfg=cfg.rendering,
                output_dir=k_sweep_dir,
                cluster_labels=k_labels,
                output_filename=f"cluster_k{k}.mp4",
                title=f"Temporal Windowing Clusters (k={k}, f={frames_per_clip}){is_optimal}",
            )
    else:
        cluster_labels = {i: f"cluster_{i}" for i in cluster_dict_original.keys()}
        run_rendering(
            cluster_dict_original,
            rodent_rollout_dir=cfg.rodent_rollout_dir,
            rendering_cfg=cfg.rendering,
            output_dir=cfg.output_dir,
            cluster_labels=cluster_labels,
            output_filename=f"cluster_k{optimal_k}_optimal.mp4",
            title=f"Temporal Windowing Clusters (k={optimal_k}, f={frames_per_clip})",
        )

    print("\n=== Pipeline complete! ===")
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Frames per clip: {frames_per_clip}")
    print(f"Total split clips: {num_split_clips}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/experiment/temporal_windowing.yaml"
    main(config_path)
