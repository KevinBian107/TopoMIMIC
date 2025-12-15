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
from dataclasses import dataclass, field
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import pandas as pd

from vnl_tda.utils.load_utils import get_aggregate_data, load_from_h5py

sys.path.insert(0, "/home/jovyan/vast/kaiwen/TopoVNL")
from scripts.dm_latent_subspace_auto_dunn import (
    M, TAU,
    DMConfig, ClusteringConfig, AutoClusterConfig, RenderingConfig,
    prepare_windows_for_clustering,
    compute_w2_distance_matrix,
    auto_select_clusters,
    run_rendering,
)


CLUSTER_COLORS = [
    "#E63946", "#2A9D8F", "#E9C46A", "#264653", "#F4A261",
    "#8338EC", "#06D6A0", "#118AB2", "#EF476F", "#073B4C",
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
]


@dataclass
class RandomSeedConfig:
    master_output_dir: str = "out_random_seed_benchmark"
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    behavior_csv: str = "/home/jovyan/vast/kaiwen/TopoVNL/behavior_book.csv"
    num_clips: int = 500
    dm: DMConfig = None
    clustering: ClusteringConfig = None
    auto_cluster: AutoClusterConfig = None
    rendering: RenderingConfig = None
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1000])
    skip_clustering: bool = False
    skip_rendering: bool = False
    seed_labels: dict = field(default_factory=dict)
    behavior_categories: list[str] = field(default_factory=lambda: ["walk", "rear", "groom"])
    body_parts: list[str] = None
    coverage_plot: dict = field(default_factory=lambda: {"max_loops": 200, "figsize": [18, 6]})

    def __post_init__(self):
        if self.dm is None:
            self.dm = DMConfig()
        if self.clustering is None:
            self.clustering = ClusteringConfig()
        if self.auto_cluster is None:
            self.auto_cluster = AutoClusterConfig()
        if self.rendering is None:
            self.rendering = RenderingConfig()
        if self.body_parts is None:
            self.body_parts = [
                "upper_leg_R", "lower_leg_R", "upper_arm_R",
                "upper_leg_L", "lower_leg_L", "upper_arm_L",
                "skull", "torso", "pelvis",
            ]


def load_config(config_path: str) -> RandomSeedConfig:
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

    seed_labels = {}
    for key, val in cfg.get("seed_labels", {}).items():
        seed_labels[key] = {int(k): v for k, v in val.items()}

    return RandomSeedConfig(
        master_output_dir=cfg.get("master_output_dir", "out_random_seed_benchmark"),
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", RandomSeedConfig.rodent_rollout_dir),
        behavior_csv=cfg.get("behavior_csv", RandomSeedConfig.behavior_csv),
        num_clips=cfg.get("num_clips", 500),
        dm=dm_cfg,
        clustering=clustering_cfg,
        auto_cluster=auto_cluster_cfg,
        rendering=rendering_cfg,
        seeds=cfg.get("seeds", [42, 123, 456, 789, 1000]),
        skip_clustering=cfg.get("skip_clustering", False),
        skip_rendering=cfg.get("skip_rendering", False),
        seed_labels=seed_labels,
        behavior_categories=cfg.get("behavior_categories", ["walk", "rear", "groom"]),
        body_parts=cfg.get("body_parts", None),
        coverage_plot=cfg.get("coverage_plot", {"max_loops": 200, "figsize": [18, 6]}),
    )


def load_behavior_labels(behavior_csv: str) -> dict[str, list[int]]:
    df = pd.read_csv(behavior_csv)
    return {
        "walk": df[df["walk"] == 1]["clip_idx"].tolist(),
        "rear": df[df["rear"] == 1]["clip_idx"].tolist(),
        "groom": df[(df["groom_down"] == 1) | (df["groom_up"] == 1)]["clip_idx"].tolist(),
    }


def load_qpos(clip_id: int, rodent_rollout_dir: str) -> np.ndarray:
    with h5py.File(f"{rodent_rollout_dir}/clip_{int(clip_id)}.h5", "r") as f:
        roll = load_from_h5py(f)
    return np.asarray(roll["qpos"])


def group_clusters_by_behavior(
    cluster_labels: dict[int, str],
    behavior_categories: list[str],
) -> dict[str, list[int]]:
    grouped = {cat: [] for cat in behavior_categories}
    for cid, label in cluster_labels.items():
        label_lower = label.lower()
        for cat in behavior_categories:
            if cat in label_lower:
                grouped[cat].append(cid)
    return grouped


def collect_labeled_behavior_points(
    labeled_clips: dict[str, list[int]],
    rodent_rollout_dir: str,
) -> dict[str, np.ndarray]:
    behavior_points = {}
    for behavior, clip_ids in labeled_clips.items():
        points = []
        for clip_id in clip_ids:
            try:
                qpos = load_qpos(clip_id, rodent_rollout_dir)
                points.append(qpos.mean(axis=0))
            except:
                continue
        if points:
            behavior_points[behavior] = np.array(points)
    return behavior_points


def collect_cluster_points_by_behavior(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    rodent_rollout_dir: str,
    behavior_categories: list[str],
    max_loops: int = 200,
) -> dict[str, dict[str, np.ndarray]]:
    behavior_clusters = {cat: {} for cat in behavior_categories}

    for cid, entries in cluster_dict.items():
        label = cluster_labels.get(cid, f"cluster_{cid}")
        label_lower = label.lower()

        matched_behaviors = []
        for cat in behavior_categories:
            if cat in label_lower:
                matched_behaviors.append(cat)

        if not matched_behaviors:
            continue

        points = []
        for i, (clip_id, frames) in enumerate(entries):
            if i >= max_loops:
                break
            try:
                qpos = load_qpos(clip_id, rodent_rollout_dir)
            except:
                continue

            frames = np.asarray(frames, dtype=int)
            frames = frames[frames < len(qpos)]
            if len(frames) < 2:
                continue
            points.append(qpos[frames].mean(axis=0))

        if points:
            points_arr = np.array(points)
            for matched_behavior in matched_behaviors:
                behavior_clusters[matched_behavior][label] = points_arr

    return behavior_clusters


def run_clustering_for_seed(
    seed: int,
    windows: list,
    Dn: np.ndarray,
    joint_data: np.ndarray,
    clustering_cfg: ClusteringConfig,
    auto_cfg: AutoClusterConfig,
    output_dir: str,
) -> tuple[int, dict, dict]:
    auto_cfg_seed = AutoClusterConfig(
        enabled=auto_cfg.enabled,
        k_min=auto_cfg.k_min,
        k_max=auto_cfg.k_max,
        joint_state_key=auto_cfg.joint_state_key,
        plot_dunn=auto_cfg.plot_dunn,
        render_all_k=auto_cfg.render_all_k,
        seed=seed,
    )

    optimal_k, cluster_dict, all_cluster_dicts = auto_select_clusters(
        windows=windows,
        Dn=Dn,
        joint_data=joint_data,
        clustering_cfg=clustering_cfg,
        auto_cfg=auto_cfg_seed,
        output_dir=output_dir,
    )

    return optimal_k, cluster_dict, all_cluster_dicts


def plot_seed_behavior_overlay(
    seed_name: str,
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    labeled_clips: dict[str, list[int]],
    behavior_categories: list[str],
    rodent_rollout_dir: str,
    output_dir: str,
    pca_model: PCA = None,
    max_loops: int = 200,
    figsize: tuple[int, int] = (18, 6),
) -> PCA:
    os.makedirs(output_dir, exist_ok=True)

    labeled_points = collect_labeled_behavior_points(labeled_clips, rodent_rollout_dir)
    cluster_by_behavior = collect_cluster_points_by_behavior(
        cluster_dict, cluster_labels, rodent_rollout_dir,
        behavior_categories, max_loops
    )

    if not labeled_points:
        print(f"  Warning: No labeled points, skipping {seed_name}")
        return pca_model

    all_points = []
    for pts in labeled_points.values():
        all_points.append(pts)
    for behavior_dict in cluster_by_behavior.values():
        for pts in behavior_dict.values():
            all_points.append(pts)

    if not all_points:
        return pca_model

    all_points = np.vstack(all_points)

    if pca_model is None:
        pca_model = PCA(n_components=2, random_state=0)
        pca_model.fit(all_points)

    labeled_points_2d = {beh: pca_model.transform(pts) for beh, pts in labeled_points.items()}
    cluster_points_2d = {}
    for behavior, cluster_dict_beh in cluster_by_behavior.items():
        cluster_points_2d[behavior] = {
            label: pca_model.transform(pts) for label, pts in cluster_dict_beh.items()
        }

    all_2d_points = []
    for pts in labeled_points_2d.values():
        all_2d_points.append(pts)
    for behavior_dict in cluster_points_2d.values():
        for pts in behavior_dict.values():
            all_2d_points.append(pts)
    all_2d = np.vstack(all_2d_points) if all_2d_points else None

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax_idx, behavior in enumerate(behavior_categories[:3]):
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
                except:
                    pass
            ax.scatter(pts[:, 0], pts[:, 1], c='#999999', alpha=0.4, s=20,
                       edgecolors='none', zorder=1)

        cluster_handles = []
        if behavior in cluster_points_2d:
            clusters = cluster_points_2d[behavior]
            for i, (label, pts) in enumerate(sorted(clusters.items())):
                color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                if len(pts) >= 3:
                    try:
                        hull = ConvexHull(pts)
                        hull_points = pts[hull.vertices]
                        hull_points = np.vstack([hull_points, hull_points[0]])
                        ax.fill(hull_points[:, 0], hull_points[:, 1],
                                color=color, alpha=0.35, edgecolor=color,
                                linewidth=2, zorder=2)
                    except:
                        pass
                ax.scatter(pts[:, 0], pts[:, 1], c=color, alpha=0.7, s=25,
                           edgecolors='white', linewidths=0.3, zorder=3)
                cluster_handles.append(mpatches.Patch(
                    facecolor=color, alpha=0.6, label=label,
                    edgecolor=color, linewidth=1.5
                ))

        labeled_handle = mpatches.Patch(
            facecolor='#f0f0f0', alpha=0.6, label=f'{behavior} (labeled)',
            edgecolor='#666666', linewidth=1.5
        )

        if all_2d is not None:
            margin = 0.1
            x_range = all_2d[:, 0].max() - all_2d[:, 0].min()
            y_range = all_2d[:, 1].max() - all_2d[:, 1].min()
            ax.set_xlim(all_2d[:, 0].min() - margin * x_range, all_2d[:, 0].max() + margin * x_range)
            ax.set_ylim(all_2d[:, 1].min() - margin * y_range, all_2d[:, 1].max() + margin * y_range)

        all_handles = [labeled_handle] + cluster_handles
        ax.legend(handles=all_handles, loc='upper right', fontsize=8, framealpha=0.95)
        ax.set_xlabel("PC1", fontsize=11)
        ax.set_ylabel("PC2", fontsize=11)
        ax.set_title(f"{behavior.capitalize()}", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')

    fig.suptitle(f"{seed_name}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    safe_name = seed_name.replace(" ", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(output_dir, f"{safe_name}_behavior_overlay.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved behavior overlay to {save_path}")

    return pca_model


def plot_all_seeds_comparison(
    seed_results: dict,
    labeled_clips: dict[str, list[int]],
    behavior_categories: list[str],
    rodent_rollout_dir: str,
    output_dir: str,
    max_loops: int = 200,
    figsize: tuple[int, int] = (20, 12),
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    labeled_points = collect_labeled_behavior_points(labeled_clips, rodent_rollout_dir)
    if not labeled_points:
        print("Warning: No labeled points for comparison plot")
        return

    all_points_for_pca = []
    for pts in labeled_points.values():
        all_points_for_pca.append(pts)

    for seed, data in seed_results.items():
        cluster_dict = data["cluster_dict"]
        cluster_labels = data["cluster_labels"]
        cluster_by_behavior = collect_cluster_points_by_behavior(
            cluster_dict, cluster_labels, rodent_rollout_dir,
            behavior_categories, max_loops
        )
        for behavior_dict in cluster_by_behavior.values():
            for pts in behavior_dict.values():
                all_points_for_pca.append(pts)

    if not all_points_for_pca:
        return

    all_points = np.vstack(all_points_for_pca)
    pca = PCA(n_components=2, random_state=0)
    pca.fit(all_points)

    labeled_points_2d = {beh: pca.transform(pts) for beh, pts in labeled_points.items()}

    n_seeds = len(seed_results)
    n_behaviors = len(behavior_categories)
    fig, axes = plt.subplots(n_seeds, n_behaviors, figsize=figsize)

    if n_seeds == 1:
        axes = axes.reshape(1, -1)

    all_2d = pca.transform(all_points)
    x_min, x_max = all_2d[:, 0].min(), all_2d[:, 0].max()
    y_min, y_max = all_2d[:, 1].min(), all_2d[:, 1].max()
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min

    for row_idx, (seed, data) in enumerate(sorted(seed_results.items())):
        cluster_dict = data["cluster_dict"]
        cluster_labels = data["cluster_labels"]
        optimal_k = data["optimal_k"]

        cluster_by_behavior = collect_cluster_points_by_behavior(
            cluster_dict, cluster_labels, rodent_rollout_dir,
            behavior_categories, max_loops
        )

        for col_idx, behavior in enumerate(behavior_categories):
            ax = axes[row_idx, col_idx]

            if behavior in labeled_points_2d:
                pts = labeled_points_2d[behavior]
                if len(pts) >= 3:
                    try:
                        hull = ConvexHull(pts)
                        hull_points = pts[hull.vertices]
                        hull_points = np.vstack([hull_points, hull_points[0]])
                        ax.fill(hull_points[:, 0], hull_points[:, 1],
                                color='#f0f0f0', alpha=0.6, edgecolor='#666666',
                                linewidth=2, linestyle='-', zorder=1)
                    except:
                        pass
                ax.scatter(pts[:, 0], pts[:, 1], c='#999999', alpha=0.3, s=15,
                           edgecolors='none', zorder=1)

            if behavior in cluster_by_behavior:
                clusters = cluster_by_behavior[behavior]
                for i, (label, pts) in enumerate(sorted(clusters.items())):
                    color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                    pts_2d = pca.transform(pts)
                    if len(pts_2d) >= 3:
                        try:
                            hull = ConvexHull(pts_2d)
                            hull_points = pts_2d[hull.vertices]
                            hull_points = np.vstack([hull_points, hull_points[0]])
                            ax.fill(hull_points[:, 0], hull_points[:, 1],
                                    color=color, alpha=0.35, edgecolor=color,
                                    linewidth=1.5, zorder=2)
                        except:
                            pass
                    ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c=color, alpha=0.7, s=20,
                               edgecolors='white', linewidths=0.2, zorder=3)

            ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
            ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)

            if row_idx == 0:
                ax.set_title(f"{behavior.capitalize()}", fontsize=12, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"Seed {seed}\n(k={optimal_k})", fontsize=10)

            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_facecolor('#fafafa')
            ax.tick_params(labelsize=8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "all_seeds_comparison.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved all-seeds comparison to {save_path}")


def plot_seed_metrics_summary(
    seed_results: dict,
    output_dir: str,
    figsize: tuple[int, int] = (14, 8),
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    seeds = sorted(seed_results.keys())
    optimal_ks = [seed_results[s]["optimal_k"] for s in seeds]
    dunn_indices = [seed_results[s].get("dunn_index", 0) for s in seeds]
    n_samples = [sum(len(v) for v in seed_results[s]["cluster_dict"].values()) for s in seeds]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ax1 = axes[0, 0]
    ax1.bar(range(len(seeds)), optimal_ks, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(seeds)))
    ax1.set_xticklabels([str(s) for s in seeds], rotation=45, ha='right')
    ax1.set_xlabel("Seed", fontsize=11)
    ax1.set_ylabel("Optimal k", fontsize=11)
    ax1.set_title("Optimal Number of Clusters by Seed", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.bar(range(len(seeds)), dunn_indices, color='darkgreen', alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(seeds)))
    ax2.set_xticklabels([str(s) for s in seeds], rotation=45, ha='right')
    ax2.set_xlabel("Seed", fontsize=11)
    ax2.set_ylabel("Dunn Index", fontsize=11)
    ax2.set_title("Dunn Index by Seed (Higher = Better)", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    k_counts = {}
    for k in optimal_ks:
        k_counts[k] = k_counts.get(k, 0) + 1
    k_vals = sorted(k_counts.keys())
    k_freqs = [k_counts[k] for k in k_vals]
    ax3.bar(k_vals, k_freqs, color='coral', alpha=0.7, edgecolor='black')
    ax3.set_xlabel("Optimal k", fontsize=11)
    ax3.set_ylabel("Frequency", fontsize=11)
    ax3.set_title("Distribution of Optimal k", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.scatter(optimal_ks, dunn_indices, c='purple', s=100, alpha=0.7, edgecolors='black')
    for i, seed in enumerate(seeds):
        ax4.annotate(str(seed), (optimal_ks[i], dunn_indices[i]), fontsize=8, ha='left')
    ax4.set_xlabel("Optimal k", fontsize=11)
    ax4.set_ylabel("Dunn Index", fontsize=11)
    ax4.set_title("Dunn Index vs Optimal k", fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "seed_metrics_summary.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved seed metrics summary to {save_path}")


def main(config_path: str = "configs/dmwkc_experiment/random_seed_run.yaml"):
    cfg = load_config(config_path)

    if cfg.clustering.only_max and cfg.clustering.only_min:
        raise ValueError("Cannot use both only_max and only_min")

    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")
    os.makedirs(cfg.master_output_dir, exist_ok=True)

    print("=" * 60)
    print("Random Seed Benchmark Pipeline")
    print("=" * 60)
    print(f"Seeds to evaluate: {cfg.seeds}")
    print(f"Master output dir: {cfg.master_output_dir}")

    seed_results = {}
    seed_results_path = os.path.join(cfg.master_output_dir, "seed_results.pkl")

    if cfg.skip_clustering and os.path.exists(seed_results_path):
        print("\n=== Loading existing clustering results ===")
        with open(seed_results_path, "rb") as f:
            seed_results = pkl.load(f)
        print(f"Loaded results for {len(seed_results)} seeds")

        for seed in cfg.seeds:
            if seed not in seed_results:
                print(f"Warning: Seed {seed} not found in saved results")
            else:
                seed_key = f"seed_{seed}"
                if seed_key in cfg.seed_labels:
                    seed_results[seed]["cluster_labels"] = cfg.seed_labels[seed_key]
                    print(f"  Updated labels for seed {seed} from config")
    else:
        print("\n=== Stage 1: Loading data ===")
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

        print("\n=== Stage 2: Loading DM cycles ===")
        if cfg.dm.results_dir is None:
            raise ValueError("dm.results_dir required for random seed benchmark")
        with open(os.path.join(cfg.dm.results_dir, "all_cycle_dm.pkl"), "rb") as f:
            all_cycle_dm = pkl.load(f)
        print(f"Loaded {len(all_cycle_dm)} cycle sets")

        print("\n=== Stage 3: Preparing windows and computing W2 distance matrix ===")
        windows, cleaned_loops = prepare_windows_for_clustering(
            all_cycle_dm, int_rodent, cfg.clustering
        )
        print(f"Prepared {len(windows)} windows for clustering")

        if len(windows) == 0:
            raise RuntimeError("No valid windows for clustering")

        w2_cache_path = os.path.join(cfg.master_output_dir, "w2_distance_matrix.pkl")
        if os.path.exists(w2_cache_path):
            print(f"Loading cached W2 distance matrix from {w2_cache_path}")
            with open(w2_cache_path, "rb") as f:
                w2_data = pkl.load(f)
            D, Dn = w2_data["D"], w2_data["Dn"]
        else:
            D, Dn = compute_w2_distance_matrix(
                windows, int_rodent, cfg.clustering.target_len, squared_cost=False
            )
            with open(w2_cache_path, "wb") as f:
                pkl.dump({"D": D, "Dn": Dn, "windows": windows}, f)
            print(f"Cached W2 distance matrix to {w2_cache_path}")

        print("\n=== Stage 4: Running clustering for each seed ===")
        for seed in cfg.seeds:
            print(f"\n--- Seed {seed} ---")
            seed_output_dir = os.path.join(cfg.master_output_dir, f"seed_{seed}")
            os.makedirs(seed_output_dir, exist_ok=True)

            optimal_k, cluster_dict, all_cluster_dicts = run_clustering_for_seed(
                seed=seed,
                windows=windows,
                Dn=Dn,
                joint_data=joint_data,
                clustering_cfg=cfg.clustering,
                auto_cfg=cfg.auto_cluster,
                output_dir=seed_output_dir,
            )

            seed_key = f"seed_{seed}"
            cluster_labels = cfg.seed_labels.get(seed_key, {})
            if not cluster_labels:
                cluster_labels = {i: f"cluster_{i}" for i in cluster_dict.keys()}

            with open(os.path.join(seed_output_dir, "cluster_dict.pkl"), "wb") as f:
                pkl.dump(cluster_dict, f)
            with open(os.path.join(seed_output_dir, "all_cluster_dicts.pkl"), "wb") as f:
                pkl.dump(all_cluster_dicts, f)

            dunn_index = 0.0
            dunn_analysis_path = os.path.join(seed_output_dir, "dunn_analysis.pdf")
            if os.path.exists(dunn_analysis_path.replace(".pdf", ".png")):
                pass

            seed_results[seed] = {
                "optimal_k": optimal_k,
                "cluster_dict": cluster_dict,
                "all_cluster_dicts": all_cluster_dicts,
                "cluster_labels": cluster_labels,
                "dunn_index": dunn_index,
            }

            print(f"  Optimal k: {optimal_k}")
            print(f"  Clusters: {len(cluster_dict)}")
            for cid, entries in sorted(cluster_dict.items()):
                label = cluster_labels.get(cid, f"cluster_{cid}")
                print(f"    {cid} ({label}): {len(entries)} samples")

        with open(seed_results_path, "wb") as f:
            pkl.dump(seed_results, f)
        print(f"\nSaved all seed results to {seed_results_path}")

    print("\n=== Stage 5: Loading behavior labels ===")
    labeled_clips = load_behavior_labels(cfg.behavior_csv)
    for cat, clips in labeled_clips.items():
        print(f"  {cat}: {len(clips)} clips")

    print("\n=== Stage 6: Generating coverage plots ===")
    coverage_output_dir = os.path.join(cfg.master_output_dir, "coverage_plots")
    os.makedirs(coverage_output_dir, exist_ok=True)

    pca_model = None
    for seed in cfg.seeds:
        if seed not in seed_results:
            continue

        data = seed_results[seed]
        seed_name = f"Seed {seed} (k={data['optimal_k']})"

        pca_model = plot_seed_behavior_overlay(
            seed_name=seed_name,
            cluster_dict=data["cluster_dict"],
            cluster_labels=data["cluster_labels"],
            labeled_clips=labeled_clips,
            behavior_categories=cfg.behavior_categories,
            rodent_rollout_dir=cfg.rodent_rollout_dir,
            output_dir=coverage_output_dir,
            pca_model=pca_model,
            max_loops=cfg.coverage_plot.get("max_loops", 200),
            figsize=tuple(cfg.coverage_plot.get("figsize", [18, 6])),
        )

    print("\n=== Stage 7: Generating comparison plots ===")
    plot_all_seeds_comparison(
        seed_results=seed_results,
        labeled_clips=labeled_clips,
        behavior_categories=cfg.behavior_categories,
        rodent_rollout_dir=cfg.rodent_rollout_dir,
        output_dir=cfg.master_output_dir,
        max_loops=cfg.coverage_plot.get("max_loops", 200),
    )

    plot_seed_metrics_summary(
        seed_results=seed_results,
        output_dir=cfg.master_output_dir,
    )

    if not cfg.skip_rendering:
        print("\n=== Stage 8: Rendering cluster videos ===")
        for seed in cfg.seeds:
            if seed not in seed_results:
                continue

            data = seed_results[seed]
            seed_output_dir = os.path.join(cfg.master_output_dir, f"seed_{seed}")

            run_rendering(
                cluster_dict=data["cluster_dict"],
                rodent_rollout_dir=cfg.rodent_rollout_dir,
                rendering_cfg=cfg.rendering,
                output_dir=seed_output_dir,
                cluster_labels=data["cluster_labels"],
                output_filename=f"cluster_k{data['optimal_k']}.mp4",
                title=f"Seed {seed} (k={data['optimal_k']})",
            )

    print("\n=== Stage 9: Generating summary ===")
    summary_path = os.path.join(cfg.master_output_dir, "random_seed_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Random Seed Benchmark Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Seeds evaluated: {cfg.seeds}\n")
        f.write(f"Behavior categories: {cfg.behavior_categories}\n\n")

        f.write("Results by Seed:\n")
        f.write("-" * 40 + "\n")
        for seed in sorted(seed_results.keys()):
            data = seed_results[seed]
            f.write(f"Seed {seed}:\n")
            f.write(f"  Optimal k: {data['optimal_k']}\n")
            f.write(f"  Clusters:\n")
            for cid, entries in sorted(data["cluster_dict"].items()):
                label = data["cluster_labels"].get(cid, f"cluster_{cid}")
                f.write(f"    {cid} ({label}): {len(entries)} samples\n")
            f.write("\n")

        optimal_ks = [seed_results[s]["optimal_k"] for s in seed_results]
        f.write(f"\nOptimal k statistics:\n")
        f.write(f"  Mean: {np.mean(optimal_ks):.2f}\n")
        f.write(f"  Std: {np.std(optimal_ks):.2f}\n")
        f.write(f"  Min: {np.min(optimal_ks)}\n")
        f.write(f"  Max: {np.max(optimal_ks)}\n")

    print(f"Saved summary to {summary_path}")

    print("\n" + "=" * 60)
    print("Random Seed Benchmark Complete!")
    print("=" * 60)

    print("\nOptimal k by seed:")
    for seed in sorted(seed_results.keys()):
        print(f"  Seed {seed}: k={seed_results[seed]['optimal_k']}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dmwkc_experiment/random_seed_run.yaml"
    main(config_path)
