"""
Attractor analysis pipeline for no-cycle clips.

Clusters clips without topological cycles using W2 distance + MDS + KMeans.
Compares cluster coverage against labeled behaviors (walk, rear, groom).
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import functools
import multiprocessing as mp
import numpy as np
import pickle as pkl
import yaml
import h5py
import ot
from dataclasses import dataclass, field
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import pandas as pd

from vnl_tda.utils.load_utils import get_aggregate_data, load_from_h5py
from vnl_tda.pipelines.tda.skeletonization.rendering import render_grid_clusters_side_by_side
from vnl_tda.pipelines.tda.skeletonization.postprocess import _normalize_dist_matrix


@dataclass
class ClusteringConfig:
    k_min: int = 2
    k_max: int = 15
    random_state: int = 0
    subsample_frames: int | None = None  # None = use all frames, int = randomly subsample to this many
    mds_components: int = 3
    cluster_labels: dict[int, str] = field(default_factory=dict)
    skip_clips: tuple[int, ...] = ()


@dataclass
class CoverageConfig:
    behavior_csv: str = "/home/jovyan/vast/kaiwen/TopoVNL/behavior_book.csv"
    behavior_categories: list[str] = field(default_factory=lambda: ["walk", "rear", "groom"])
    max_clips_per_behavior: int = 50


@dataclass
class RenderingConfig:
    max_samples: int = 10
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)


@dataclass
class AttractorConfig:
    dm_results_dir: str = ""
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    output_dir: str = "out_attractor"
    num_clips: int = 500
    joint_state_key: str = "qpos"
    clustering: ClusteringConfig = None
    coverage: CoverageConfig = None
    rendering: RenderingConfig = None

    def __post_init__(self):
        if self.clustering is None:
            self.clustering = ClusteringConfig()
        if self.coverage is None:
            self.coverage = CoverageConfig()
        if self.rendering is None:
            self.rendering = RenderingConfig()


def load_config(config_path: str) -> AttractorConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    clustering_dict = cfg.get("clustering", {})
    if "cluster_labels" in clustering_dict and clustering_dict["cluster_labels"]:
        clustering_dict["cluster_labels"] = {
            int(k): v for k, v in clustering_dict["cluster_labels"].items()
        }
    if "skip_clips" in clustering_dict:
        clustering_dict["skip_clips"] = tuple(clustering_dict["skip_clips"])
    clustering_cfg = ClusteringConfig(**clustering_dict)

    coverage_dict = cfg.get("coverage", {})
    coverage_cfg = CoverageConfig(**coverage_dict)

    rendering_dict = cfg.get("rendering", {})
    if "target_size" in rendering_dict and isinstance(rendering_dict["target_size"], list):
        rendering_dict["target_size"] = tuple(rendering_dict["target_size"])
    rendering_cfg = RenderingConfig(**rendering_dict)

    return AttractorConfig(
        dm_results_dir=cfg.get("dm_results_dir", ""),
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", AttractorConfig.rodent_rollout_dir),
        output_dir=cfg.get("output_dir", AttractorConfig.output_dir),
        num_clips=cfg.get("num_clips", AttractorConfig.num_clips),
        joint_state_key=cfg.get("joint_state_key", "qpos"),
        clustering=clustering_cfg,
        coverage=coverage_cfg,
        rendering=rendering_cfg,
    )


def find_no_cycle_clips(
    dm_results_dir: str,
    num_clips: int,
    skip_clips: tuple[int, ...] = (),
) -> list[int]:
    cycles_path = os.path.join(dm_results_dir, "all_cycle_dm.pkl")
    if not os.path.exists(cycles_path):
        raise FileNotFoundError(f"Cycles file not found: {cycles_path}")

    with open(cycles_path, "rb") as f:
        all_cycle_dm = pkl.load(f)

    no_cycle_clips = []
    for clip_id in range(min(num_clips, len(all_cycle_dm))):
        if clip_id in skip_clips:
            continue
        cycles = all_cycle_dm[clip_id]
        if len(cycles) == 0:
            no_cycle_clips.append(clip_id)

    return no_cycle_clips


def compute_w2_distance_matrix(
    clip_ids: list[int],
    intention_data: np.ndarray,
    subsample_frames: int | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise W2 distance matrix treating each clip as a point set.

    Args:
        clip_ids: List of clip indices
        intention_data: Intention array (n_clips, n_frames, D)
        subsample_frames: If set, randomly subsample to this many frames per clip
        random_state: Random seed for subsampling

    Returns:
        Tuple of (raw_distance_matrix, normalized_distance_matrix)
    """
    rng = np.random.default_rng(random_state)
    n_clips = len(clip_ids)
    D = np.zeros((n_clips, n_clips), dtype=float)

    clip_data = []
    for clip_id in clip_ids:
        X = intention_data[clip_id]  # (T, D)
        if subsample_frames is not None and subsample_frames < len(X):
            idx = rng.choice(len(X), size=subsample_frames, replace=False)
            idx = np.sort(idx)
            X = X[idx]
        clip_data.append(X)

    print(f"Computing W2 distance matrix for {n_clips} clips...")
    for i in tqdm(range(n_clips), desc="W2 distances"):
        Xi = clip_data[i]
        ni = len(Xi)
        wi = np.full(ni, 1.0 / ni)

        for j in range(i + 1, n_clips):
            Xj = clip_data[j]
            nj = len(Xj)
            wj = np.full(nj, 1.0 / nj)

            C = ot.dist(Xi, Xj)
            val = ot.emd2(wi, wj, C)
            D[i, j] = D[j, i] = val

    Dn = _normalize_dist_matrix(D)
    return D, Dn


def cluster_from_distance_matrix(
    Dn: np.ndarray,
    n_clusters: int,
    mds_components: int = 3,
    random_state: int = 0,
) -> np.ndarray:
    """
    Cluster using precomputed distance matrix via MDS + KMeans.
    """
    mds = MDS(
        n_components=mds_components,
        dissimilarity='precomputed',
        random_state=random_state,
        normalized_stress='auto',
    )
    X_embedded = mds.fit_transform(Dn)
    labels = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init='auto',
    ).fit_predict(X_embedded)
    return labels


def prepare_joint_trajectories(
    clip_ids: list[int],
    joint_data: np.ndarray,
) -> np.ndarray:
    trajectories = []
    for clip_id in clip_ids:
        traj = joint_data[clip_id].flatten()
        trajectories.append(traj)
    return np.array(trajectories)


def compute_dunn_index(
    labels: np.ndarray,
    trajectories: np.ndarray,
) -> tuple[float, float, float]:
    unique_labels = [l for l in np.unique(labels) if l >= 0]
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return 0.0, 0.0, 0.0

    cluster_data = {}
    for lbl in unique_labels:
        mask = labels == lbl
        cluster_data[lbl] = trajectories[mask]

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
    clip_ids: list[int],
) -> dict[int, list[int]]:
    cluster_dict = {}
    for lbl, clip_id in zip(labels, clip_ids):
        lbl = int(lbl)
        if lbl < 0:
            continue
        cluster_dict.setdefault(lbl, []).append(clip_id)
    return cluster_dict


def select_optimal_k_with_dunn(
    Dn: np.ndarray,
    joint_trajectories: np.ndarray,
    clip_ids: list[int],
    k_min: int,
    k_max: int,
    mds_components: int = 3,
    random_state: int = 0,
) -> tuple[int, np.ndarray, dict[int, list[int]], dict]:
    print(f"\n=== Dunn Index Selection (k={k_min} to {k_max}) ===")

    results = {
        "dunn_index": [],
        "min_inter_dist": [],
        "max_intra_diam": [],
        "n_clusters_actual": [],
        "all_labels": {},
        "all_cluster_dicts": {},
    }

    k_range = range(k_min, k_max + 1)

    for k in tqdm(k_range, desc="W2+MDS+KMeans k sweep"):
        labels = cluster_from_distance_matrix(
            Dn, n_clusters=k, mds_components=mds_components, random_state=random_state
        )

        n_actual = len(set(labels)) - (1 if -1 in labels else 0)

        dunn, min_inter, max_intra = compute_dunn_index(labels, joint_trajectories)

        results["dunn_index"].append(dunn)
        results["min_inter_dist"].append(min_inter)
        results["max_intra_diam"].append(max_intra)
        results["n_clusters_actual"].append(n_actual)
        results["all_labels"][k] = labels.copy()
        results["all_cluster_dicts"][k] = labels_to_cluster_dict(labels, clip_ids)

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

    print(f"\nOptimal k={optimal_k}")

    return optimal_k, best_labels, cluster_dict, results


def plot_dunn_analysis(
    results: dict,
    optimal_k: int,
    k_min: int,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    k_list = list(results["all_labels"].keys())

    axes[0].plot(k_list, results["dunn_index"], color="darkgreen", marker="o", linewidth=2, markersize=6)
    axes[0].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    axes[0].set_xlabel("Number of clusters (k)", fontsize=12)
    axes[0].set_ylabel("Dunn Index", fontsize=12)
    axes[0].set_title("Dunn Index (higher = better)", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_list, results["min_inter_dist"], color="blue", marker="s", linewidth=2, markersize=6)
    axes[1].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    axes[1].set_xlabel("Number of clusters (k)", fontsize=12)
    axes[1].set_ylabel("Min Inter-cluster Distance", fontsize=12)
    axes[1].set_title("Min Inter-cluster Distance", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(k_list, results["max_intra_diam"], color="purple", marker="^", linewidth=2, markersize=6)
    axes[2].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    axes[2].set_xlabel("Number of clusters (k)", fontsize=12)
    axes[2].set_ylabel("Max Intra-cluster Diameter", fontsize=12)
    axes[2].set_title("Max Intra-cluster Diameter", fontsize=14, fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dunn_analysis.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "dunn_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Dunn analysis to {output_dir}/dunn_analysis.pdf")


def load_behavior_labels(behavior_csv: str) -> dict[str, list[int]]:
    df = pd.read_csv(behavior_csv)

    behavior_to_clips = {
        "walk": df[df["walk"] == 1]["clip_idx"].tolist(),
        "rear": df[df["rear"] == 1]["clip_idx"].tolist(),
        "groom": df[(df["groom_down"] == 1) | (df["groom_up"] == 1)]["clip_idx"].tolist(),
    }

    return behavior_to_clips


def load_qpos(clip_id: int, rodent_rollout_dir: str) -> np.ndarray:
    with h5py.File(f"{rodent_rollout_dir}/clip_{int(clip_id)}.h5", "r") as f:
        roll = load_from_h5py(f)
    return np.asarray(roll["qpos"])


def collect_labeled_behavior_points(
    labeled_clips: dict[str, list[int]],
    rodent_rollout_dir: str,
    max_clips: int = 50,
) -> dict[str, np.ndarray]:
    behavior_points = {}

    for behavior, clip_ids in labeled_clips.items():
        points = []
        for i, clip_id in enumerate(clip_ids):
            if i >= max_clips:
                break
            try:
                qpos = load_qpos(clip_id, rodent_rollout_dir)
                points.append(qpos.mean(axis=0))
            except Exception:
                continue

        if points:
            behavior_points[behavior] = np.array(points)

    return behavior_points


def collect_cluster_points(
    cluster_dict: dict[int, list[int]],
    rodent_rollout_dir: str,
    max_clips: int = 50,
) -> dict[int, np.ndarray]:
    cluster_points = {}

    for cid, clip_ids in cluster_dict.items():
        points = []
        for i, clip_id in enumerate(clip_ids):
            if i >= max_clips:
                break
            try:
                qpos = load_qpos(clip_id, rodent_rollout_dir)
                points.append(qpos.mean(axis=0))
            except Exception:
                continue

        if points:
            cluster_points[cid] = np.array(points)

    return cluster_points


CLUSTER_COLORS = [
    "#E63946",  # Red
    "#2A9D8F",  # Teal
    "#E9C46A",  # Yellow
    "#264653",  # Dark blue
    "#F4A261",  # Orange
    "#8338EC",  # Purple
    "#06D6A0",  # Mint
    "#118AB2",  # Blue
    "#EF476F",  # Pink
    "#073B4C",  # Navy
]


def plot_coverage_comparison(
    cluster_dict: dict[int, list[int]],
    labeled_clips: dict[str, list[int]],
    behavior_categories: list[str],
    rodent_rollout_dir: str,
    output_dir: str,
    cluster_labels: dict[int, str] | None = None,
    max_clips: int = 50,
    figsize: tuple[int, int] = (15, 5),
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    labeled_points = collect_labeled_behavior_points(labeled_clips, rodent_rollout_dir, max_clips)
    cluster_points = collect_cluster_points(cluster_dict, rodent_rollout_dir, max_clips)

    if cluster_labels is None:
        cluster_labels = {cid: f"Cluster {cid}" for cid in cluster_dict.keys()}

    if not labeled_points or not cluster_points:
        print("  Warning: No points collected, skipping coverage plot")
        return

    # Group clusters by behavior based on their labels
    behavior_to_clusters: dict[str, list[int]] = {beh: [] for beh in behavior_categories}
    for cid, label in cluster_labels.items():
        label_lower = label.lower()
        for beh in behavior_categories:
            if beh.lower() in label_lower:
                behavior_to_clusters[beh].append(cid)
                break

    all_points = []
    for pts in labeled_points.values():
        all_points.append(pts)
    for pts in cluster_points.values():
        all_points.append(pts)

    all_points = np.vstack(all_points)

    pca = PCA(n_components=2, random_state=0)
    pca.fit(all_points)

    labeled_points_2d = {beh: pca.transform(pts) for beh, pts in labeled_points.items()}
    cluster_points_2d = {cid: pca.transform(pts) for cid, pts in cluster_points.items()}

    all_2d = np.vstack([pts for pts in labeled_points_2d.values()] +
                       [pts for pts in cluster_points_2d.values()])

    fig, axes = plt.subplots(1, len(behavior_categories), figsize=figsize)
    if len(behavior_categories) == 1:
        axes = [axes]

    for ax_idx, behavior in enumerate(behavior_categories):
        ax = axes[ax_idx]

        if behavior in labeled_points_2d:
            pts = labeled_points_2d[behavior]

            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    hull_points = pts[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])
                    ax.fill(hull_points[:, 0], hull_points[:, 1],
                            color='#f0f0f0', alpha=0.6, edgecolor='#666666',
                            linewidth=2.5, linestyle='-', zorder=1)
                except Exception:
                    pass

            ax.scatter(pts[:, 0], pts[:, 1], c='#999999', alpha=0.4, s=20,
                       edgecolors='none', zorder=1, label=f"{behavior} (labeled)")

        # Only show clusters that match this behavior
        related_clusters = behavior_to_clusters.get(behavior, [])
        cluster_handles = []
        for i, cid in enumerate(sorted(related_clusters)):
            if cid not in cluster_points_2d:
                continue
            pts = cluster_points_2d[cid]
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    hull_points = pts[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])
                    ax.fill(hull_points[:, 0], hull_points[:, 1],
                            color=color, alpha=0.35, edgecolor=color,
                            linewidth=2, zorder=2)
                except Exception:
                    pass

            ax.scatter(pts[:, 0], pts[:, 1], c=color, alpha=0.7, s=25,
                       edgecolors='white', linewidths=0.3, zorder=3)

            label_str = cluster_labels.get(cid, f"Cluster {cid}")
            cluster_handles.append(mpatches.Patch(
                facecolor=color, alpha=0.6, label=f"C{cid}: {label_str} (n={len(cluster_dict[cid])})",
                edgecolor=color, linewidth=1.5
            ))

        labeled_handle = mpatches.Patch(
            facecolor='#f0f0f0', alpha=0.6, label=f'{behavior} (labeled)',
            edgecolor='#666666', linewidth=1.5
        )

        margin = 0.1
        x_range = all_2d[:, 0].max() - all_2d[:, 0].min()
        y_range = all_2d[:, 1].max() - all_2d[:, 1].min()
        ax.set_xlim(all_2d[:, 0].min() - margin * x_range, all_2d[:, 0].max() + margin * x_range)
        ax.set_ylim(all_2d[:, 1].min() - margin * y_range, all_2d[:, 1].max() + margin * y_range)

        all_handles = [labeled_handle] + cluster_handles
        ax.legend(handles=all_handles, loc='upper right', fontsize=7, framealpha=0.95)

        ax.set_xlabel("PC1", fontsize=11)
        ax.set_ylabel("PC2", fontsize=11)
        n_clusters = len(related_clusters)
        ax.set_title(f"{behavior.capitalize()} ({n_clusters} clusters)", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')

    fig.suptitle("No-Cycle Clips: Cluster Coverage vs Labeled Behaviors", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "coverage_comparison.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved coverage comparison to {save_path}")


def run_rendering(
    cluster_dict: dict[int, list[int]],
    rodent_rollout_dir: str,
    rendering_cfg: RenderingConfig,
    output_dir: str,
    cluster_labels: dict[int, str] | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if cluster_labels is None:
        cluster_labels = {cid: f"C{cid}" for cid in cluster_dict.keys()}

    cluster_chunks = {}
    cluster_clip_ids = {}
    for cid, clip_ids in cluster_dict.items():
        sampled = clip_ids[:rendering_cfg.max_samples]
        chunks = []
        cids_out = []
        for clip_id in sampled:
            try:
                qpos = load_qpos(clip_id, rodent_rollout_dir)
                chunks.append(np.arange(len(qpos)))
                cids_out.append(clip_id)
            except Exception:
                continue
        cluster_chunks[cid] = chunks
        cluster_clip_ids[cid] = cids_out

    cluster_dict_for_render = {}
    for cid in cluster_chunks:
        entries = []
        for clip_id, frames in zip(cluster_clip_ids[cid], cluster_chunks[cid]):
            entries.append((clip_id, frames))
        cluster_dict_for_render[cid] = entries

    from vnl_tda.pipelines.tda.skeletonization.postprocess import chunk_loops_by_clustered_vpath

    cluster_chunks_render, cluster_clip_ids_render = chunk_loops_by_clustered_vpath(
        cluster_dict_for_render,
        rodent_rollout_dir,
        pad=0,
        sample=rendering_cfg.max_samples,
        return_clip_ids=True,
    )

    out_path = os.path.join(output_dir, "cluster_samples.mp4")

    render_grid_clusters_side_by_side(
        cluster_chunks=cluster_chunks_render,
        out_path=out_path,
        framerate=rendering_cfg.framerate,
        max_samples_per_cluster=rendering_cfg.max_samples,
        target_size=rendering_cfg.target_size,
        title="No-Cycle Clips: KMeans Clusters (Dunn-optimal)",
        per_tile_labels=True,
        selected_clusters=sorted(cluster_dict.keys()),
        cluster_labels=cluster_labels,
        cluster_clip_ids=cluster_clip_ids_render,
    )
    print(f"Video saved to: {out_path}")


def main(config_path: str = "configs/attractor.yaml"):
    cfg = load_config(config_path)

    if not cfg.dm_results_dir:
        raise ValueError("dm_results_dir must be specified in config")

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print("Attractor Analysis Pipeline (No-Cycle Clips)")
    print("=" * 60)

    print("\n=== Stage 1: Finding no-cycle clips ===")
    if cfg.clustering.skip_clips:
        print(f"Skipping clips: {cfg.clustering.skip_clips}")
    no_cycle_clips = find_no_cycle_clips(
        cfg.dm_results_dir, cfg.num_clips, cfg.clustering.skip_clips
    )
    print(f"Found {len(no_cycle_clips)} clips without cycles out of {cfg.num_clips}")
    print(f"Clip IDs: {no_cycle_clips}")

    if len(no_cycle_clips) < 2:
        print("Not enough no-cycle clips for clustering. Exiting.")
        return

    print("\n=== Stage 2: Loading intention data ===")
    work_rodent = functools.partial(
        get_aggregate_data, "/intention", [], path=cfg.rodent_rollout_dir
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        int_rodent = list(tqdm(pool.imap(work_rodent, range(cfg.num_clips)), total=cfg.num_clips))
    int_rodent = np.vstack(int_rodent)
    int_rodent = int_rodent.reshape([cfg.num_clips, 499, 16])
    print(f"Loaded intention: {int_rodent.shape}")

    print(f"\n=== Stage 3: Loading joint state data ({cfg.joint_state_key}) ===")
    work_joint = functools.partial(
        get_aggregate_data, f"/{cfg.joint_state_key}", [], path=cfg.rodent_rollout_dir
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

    print("\n=== Stage 4: Computing W2 distance matrix ===")
    if cfg.clustering.subsample_frames:
        print(f"Subsampling to {cfg.clustering.subsample_frames} frames per clip")
    D_raw, Dn = compute_w2_distance_matrix(
        no_cycle_clips,
        int_rodent,
        subsample_frames=cfg.clustering.subsample_frames,
        random_state=cfg.clustering.random_state,
    )
    print(f"Distance matrix shape: {Dn.shape}")

    with open(os.path.join(cfg.output_dir, "w2_distance_matrix.pkl"), "wb") as f:
        pkl.dump({"raw": D_raw, "normalized": Dn}, f)

    joint_trajectories = prepare_joint_trajectories(no_cycle_clips, joint_data)
    print(f"Joint trajectories shape: {joint_trajectories.shape}")

    print("\n=== Stage 5: W2+MDS+KMeans clustering with Dunn Index selection ===")
    optimal_k, best_labels, cluster_dict, results = select_optimal_k_with_dunn(
        Dn=Dn,
        joint_trajectories=joint_trajectories,
        clip_ids=no_cycle_clips,
        k_min=cfg.clustering.k_min,
        k_max=cfg.clustering.k_max,
        mds_components=cfg.clustering.mds_components,
        random_state=cfg.clustering.random_state,
    )

    print("\n=== Stage 6: Saving results ===")
    plot_dunn_analysis(results, optimal_k, cfg.clustering.k_min, cfg.output_dir)

    with open(os.path.join(cfg.output_dir, "cluster_dict.pkl"), "wb") as f:
        pkl.dump(cluster_dict, f)
    with open(os.path.join(cfg.output_dir, "all_results.pkl"), "wb") as f:
        pkl.dump(results, f)
    with open(os.path.join(cfg.output_dir, "optimal_k.txt"), "w") as f:
        f.write(f"optimal_k={optimal_k}\n")

    print(f"Created {len(cluster_dict)} clusters (Dunn-selected k={optimal_k})")

    cluster_labels = cfg.clustering.cluster_labels.copy() if cfg.clustering.cluster_labels else {}
    for cid in cluster_dict.keys():
        if cid not in cluster_labels:
            cluster_labels[cid] = f"cluster_{cid}"

    for cid, clip_ids in sorted(cluster_dict.items()):
        label = cluster_labels.get(cid, f"cluster_{cid}")
        print(f"  Cluster {cid} ({label}): {len(clip_ids)} clips")

    print("\n=== Stage 7: Computing behavior coverage ===")
    labeled_clips = load_behavior_labels(cfg.coverage.behavior_csv)
    for cat, clips in labeled_clips.items():
        print(f"  {cat}: {len(clips)} labeled clips")

    print("\n=== Stage 8: Generating coverage comparison plots ===")
    plot_coverage_comparison(
        cluster_dict=cluster_dict,
        labeled_clips=labeled_clips,
        behavior_categories=cfg.coverage.behavior_categories,
        rodent_rollout_dir=cfg.rodent_rollout_dir,
        output_dir=cfg.output_dir,
        cluster_labels=cluster_labels,
        max_clips=cfg.coverage.max_clips_per_behavior,
    )

    print("\n=== Stage 9: Rendering cluster samples ===")
    run_rendering(
        cluster_dict=cluster_dict,
        rodent_rollout_dir=cfg.rodent_rollout_dir,
        rendering_cfg=cfg.rendering,
        output_dir=cfg.output_dir,
        cluster_labels=cluster_labels,
    )

    print("\n=== Stage 10: Saving summary ===")
    summary_path = os.path.join(cfg.output_dir, "attractor_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Attractor Analysis Summary (No-Cycle Clips)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total clips: {cfg.num_clips}\n")
        f.write(f"No-cycle clips: {len(no_cycle_clips)}\n")
        f.write(f"Optimal k: {optimal_k}\n\n")
        f.write("Cluster sizes:\n")
        f.write("-" * 40 + "\n")
        for cid, clip_ids in sorted(cluster_dict.items()):
            label = cluster_labels.get(cid, f"cluster_{cid}")
            f.write(f"  Cluster {cid} ({label}): {len(clip_ids)} clips\n")
        f.write("\nBehavior overlap:\n")
        f.write("-" * 40 + "\n")
        for cid in sorted(overlap.keys()):
            label = cluster_labels.get(cid, f"cluster_{cid}")
            f.write(f"  Cluster {cid} ({label}): ")
            parts = [f"{beh}={overlap[cid][beh]}" for beh in cfg.coverage.behavior_categories]
            f.write(", ".join(parts) + "\n")

    print(f"Saved summary to {summary_path}")

    print("\n" + "=" * 60)
    print("Attractor Analysis Complete!")
    print("=" * 60)
    print(f"\nOptimal k: {optimal_k}")
    print(f"Results saved to: {cfg.output_dir}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dm_latent_attractor.yaml"
    main(config_path)
