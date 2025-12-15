"""
Analyzing synthetic behavioral state space data.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

BASE_DIR = "/home/jovyan/vast/kaiwen/TopoVNL"

import sys
import numpy as np
import pickle as pkl
import yaml
from dataclasses import dataclass, field
from tqdm import tqdm

from vnl_tda.pipelines.tda.skeletonization.skeletonization import extract_skeleton_pcd_dm
from vnl_tda.pipelines.tda.skeletonization.postprocess import (
    cluster_vpath_with_wasserstein2d_fixedlen_kmeans,
)
from vnl_tda.pipelines.tda.skeletonization.preprocess import (
    takens_points,
)


@dataclass
class TakensConfig:
    m: int = 3
    tau: int = 6
    auto_estimate: bool = False
    max_tau: int = 200
    m_max: int = 15
    fnn_thr: float = 0.05


@dataclass
class DMConfig:
    pers_delta: float = 0.75
    k_neighbors: int = 5
    epsilon: float = 0.99
    skip: bool = False
    results_dir: str | None = None


@dataclass
class ClusteringConfig:
    n_clusters: int = 9
    min_len: int = 0
    max_len: int = 500
    target_len: int = 50
    only_max: bool = True
    only_min: bool = False
    skip_clips: tuple[int, ...] = ()


@dataclass
class SyntheticPipelineConfig:
    data_dir: str = "data/synthetic"
    output_dir: str = "out_synthetic"
    num_clips: int = 500
    takens: TakensConfig = None
    dm: DMConfig = None
    clustering: ClusteringConfig = None

    def __post_init__(self):
        if self.takens is None:
            self.takens = TakensConfig()
        if self.dm is None:
            self.dm = DMConfig()
        if self.clustering is None:
            self.clustering = ClusteringConfig()


def load_config(config_path: str) -> SyntheticPipelineConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    takens_cfg = TakensConfig(**cfg.get("takens", {}))

    dm_cfg = DMConfig(**cfg.get("dm", {}))

    clustering_dict = cfg.get("clustering", {})
    if "skip_clips" in clustering_dict:
        clustering_dict["skip_clips"] = tuple(clustering_dict["skip_clips"])
    clustering_cfg = ClusteringConfig(**clustering_dict)

    data_dir = cfg.get("data_dir", SyntheticPipelineConfig.data_dir)
    output_dir = cfg.get("output_dir", SyntheticPipelineConfig.output_dir)

    if not os.path.isabs(data_dir):
        data_dir = os.path.join(BASE_DIR, data_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(BASE_DIR, output_dir)

    return SyntheticPipelineConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        num_clips=cfg.get("num_clips", SyntheticPipelineConfig.num_clips),
        takens=takens_cfg,
        dm=dm_cfg,
        clustering=clustering_cfg,
    )


def load_synthetic_data(data_dir: str, num_clips: int) -> np.ndarray:
    """
    Load synthetic dataset from disk.

    Args:
        data_dir: Directory containing synthetic dataset
        num_clips: Number of clips to load

    Returns:
        Array of shape (num_clips, frames_per_clip, latent_dim)
    """
    dataset_path = os.path.join(data_dir, "dataset.pkl")
    with open(dataset_path, "rb") as f:
        dataset = pkl.load(f)

    clips_data = []
    for i in range(min(num_clips, len(dataset.clips))):
        clips_data.append(dataset.clips[i].latents)

    return np.stack(clips_data, axis=0)


def estimate_takens_params(
    latents: np.ndarray,
    takens_cfg: TakensConfig,
    output_dir: str,
    plot: bool = True,
) -> tuple[int, int]:
    """
    Estimate optimal Takens embedding parameters using AMI and FNN.

    Implements AMI/FNN directly on pre-loaded latent array to avoid
    multiprocessing pickle issues with nested functions.

    Args:
        latents: Array of shape (n_clips, n_frames, latent_dim)
        takens_cfg: Takens configuration
        output_dir: Directory to save diagnostic plots
        plot: Whether to display diagnostic plots

    Returns:
        Tuple of (tau_star, m_star)
    """
    import matplotlib.pyplot as plt

    max_tau = takens_cfg.max_tau
    m_max = takens_cfg.m_max
    fnn_thr = takens_cfg.fnn_thr
    bins = 64
    R_tol = 15.0
    A_tol = 2.0
    max_points = 20000
    rng_seed = 0

    def pc1(x):
        x = np.asarray(x, float)
        x = x[~np.isnan(x).any(axis=1)] if x.ndim == 2 else x
        if x.ndim == 1:
            return x.astype(float)
        xc = x - x.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(xc, full_matrices=False)
        return (xc @ Vt.T[:, 0]).astype(float)

    def delay_embed(x, m, tau):
        x = np.asarray(x, float)
        N = len(x)
        T = N - (m - 1) * tau
        if T <= 0:
            raise ValueError("Time series too short for m,tau.")
        s = x.strides[0]
        return np.lib.stride_tricks.as_strided(
            x, shape=(T, m), strides=(s, tau * s)
        ).copy()

    def entropy_from_hist(p, eps=1e-12):
        p = p.astype(float)
        Z = max(p.sum(), eps)
        p = p / Z
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    def average_mutual_information_1d(x, max_tau=200, bins=64):
        x = np.asarray(x, float)
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(xmin) or xmin == xmax:
            return None, None
        edges = np.linspace(xmin, xmax, bins + 1)
        hx, _ = np.histogram(x, bins=edges)
        Hx = entropy_from_hist(hx)
        I = np.zeros(max_tau, float)
        for k in range(1, max_tau + 1):
            x0 = x[:-k]
            x1 = x[k:]
            hxy, _, _ = np.histogram2d(x0, x1, bins=[edges, edges])
            hy, _ = np.histogram(x1, bins=edges)
            Hxy = entropy_from_hist(hxy.ravel())
            Hy = entropy_from_hist(hy)
            I[k - 1] = Hx + Hy - Hxy
        taus = np.arange(1, max_tau + 1)
        return taus, I

    def first_local_minimum_idx(y):
        y = np.asarray(y)
        for i in range(1, len(y) - 1):
            if y[i] < y[i - 1] and y[i] < y[i + 1]:
                return i
        return None

    def fnn_fraction_1d(x, tau, m_max=12, R_tol=15.0, A_tol=2.0,
                        theiler=None, max_points=20000, rng_seed=0):
        rs = np.random.RandomState(rng_seed)
        x = np.asarray(x, float)
        sig = np.std(x)
        if not np.isfinite(sig) or sig == 0:
            return None, None

        ms = np.arange(1, m_max + 1)
        fracs = []
        for m in ms:
            try:
                Xm = delay_embed(x, m, tau)
                Xm1 = delay_embed(x, m + 1, tau)
            except ValueError:
                fracs.append(np.nan)
                continue

            Tm = Xm.shape[0]
            idx_all = np.arange(Tm)
            if Tm > max_points:
                idx = np.sort(rs.choice(Tm, size=max_points, replace=False))
                Xm_use = Xm[idx]
            else:
                idx = idx_all
                Xm_use = Xm

            d2 = np.sum((Xm_use[:, None, :] - Xm_use[None, :, :]) ** 2, axis=2)
            np.fill_diagonal(d2, np.inf)

            theiler_ = 0 if theiler is None else theiler
            if theiler_ > 0:
                I = idx[:, None]
                J = idx[None, :]
                mask = np.abs(I - J) <= theiler_
                d2[mask] = np.inf

            nn = np.argmin(d2, axis=1)
            dist_m = np.sqrt(d2[np.arange(len(nn)), nn])
            valid0 = np.isfinite(dist_m) & (dist_m > 0)

            Tm1 = Xm1.shape[0]
            base = idx[valid0]
            nnix = nn[valid0]
            valid1 = (base < Tm1) & (nnix < Tm1)
            base = base[valid1]
            nnix = nnix[valid1]
            dist_m = dist_m[valid0][valid1]
            if len(base) == 0:
                fracs.append(np.nan)
                continue

            x_new_diff = np.abs(x[base + m * tau] - x[nnix + m * tau])
            Rm = x_new_diff / np.maximum(dist_m, 1e-12)
            crit1 = Rm > R_tol
            crit2 = (x_new_diff / sig) > A_tol
            fnn = np.logical_or(crit1, crit2)
            fracs.append(np.mean(fnn) if len(fnn) else np.nan)

        return ms, np.asarray(fracs, float)

    os.makedirs(output_dir, exist_ok=True)

    n_clips = latents.shape[0]
    taus_list, I_list, tau_star_each = [], [], []

    for clip_idx in tqdm(range(n_clips), desc="AMI per clip"):
        clip = latents[clip_idx]
        x = pc1(clip)
        if len(x) < 3:
            continue
        taus, I = average_mutual_information_1d(x, max_tau=max_tau, bins=bins)
        if taus is None:
            continue
        taus_list.append(taus)
        I_list.append(I)
        idx = first_local_minimum_idx(I)
        tau_star_each.append(taus[idx] if idx is not None else np.nan)

    tau_star_each = np.array(tau_star_each, float)
    tau_star = int(np.nanmedian(tau_star_each)) if np.any(np.isfinite(tau_star_each)) else 10

    I_mat = np.vstack(I_list) if len(I_list) > 0 else np.empty((0, max_tau))
    I_mean = np.nanmean(I_mat, axis=0) if I_mat.size else np.full(max_tau, np.nan)
    I_se = (np.nanstd(I_mat, axis=0) / np.sqrt(I_mat.shape[0])) if I_mat.size else np.full(max_tau, np.nan)

    theiler_val = tau_star
    fnn_curves = []
    ms = np.arange(1, m_max + 1)

    for clip_idx in tqdm(range(n_clips), desc=f"FNN per clip (τ={tau_star})"):
        clip = latents[clip_idx]
        x = pc1(clip)
        ms_, fnn = fnn_fraction_1d(
            x, tau_star, m_max=m_max, R_tol=R_tol, A_tol=A_tol,
            theiler=theiler_val, max_points=max_points, rng_seed=rng_seed
        )
        if fnn is None:
            continue
        fnn_curves.append(fnn)

    fnn_mat = np.vstack(fnn_curves) if len(fnn_curves) > 0 else np.empty((0, m_max))
    fnn_mean = np.nanmean(fnn_mat, axis=0) if fnn_mat.size else np.full(m_max, np.nan)
    fnn_se = (np.nanstd(fnn_mat, axis=0) / np.sqrt(fnn_mat.shape[0])) if fnn_mat.size else np.full(m_max, np.nan)

    m_star = None
    for k, v in enumerate(fnn_mean, start=1):
        if np.isfinite(v) and v <= fnn_thr:
            m_star = k
            break
    if m_star is None:
        m_star = int(np.nanargmin(fnn_mean) + 1) if np.any(np.isfinite(fnn_mean)) else 3

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        x_taus = taus_list[0] if len(taus_list) > 0 else np.arange(1, max_tau + 1)
        ax1.plot(x_taus, I_mean, label="mean AMI")
        if np.all(np.isfinite(I_se)):
            ax1.fill_between(x_taus, I_mean - I_se, I_mean + I_se, alpha=0.2, linewidth=0)
        ax1.axvline(tau_star, ls="--", label=f"τ* = {tau_star}")
        ax1.set_xlabel("Delay τ")
        ax1.set_ylabel("AMI")
        ax1.set_title("Average Mutual Information")
        ax1.legend()

        ax2.plot(ms, fnn_mean, marker='o', label="mean FNN")
        if np.all(np.isfinite(fnn_se)):
            ax2.fill_between(ms, fnn_mean - fnn_se, fnn_mean + fnn_se, alpha=0.2, linewidth=0)
        ax2.axhline(fnn_thr, ls=":", label=f"threshold = {fnn_thr}")
        ax2.axvline(m_star, ls="--", label=f"m* = {m_star}")
        ax2.set_xlabel("Embedding dimension m")
        ax2.set_ylabel("Fraction of false neighbors")
        ax2.set_title(f"FNN at τ = {tau_star} (Theiler={theiler_val})")
        ax2.legend()

        fig.suptitle("Delay-Embedding Parameter Selection (AMI → τ*, FNN → m*)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "takens_params.png"), dpi=150)
        plt.show()

    diagnostics = dict(
        taus_list=taus_list,
        I_list=I_list,
        tau_star_each=tau_star_each,
        I_mean=I_mean,
        I_se=I_se,
        fnn_curves=fnn_curves,
        fnn_mean=fnn_mean,
        fnn_se=fnn_se,
        ms=ms,
        theiler_used=theiler_val,
    )

    with open(os.path.join(output_dir, "takens_diagnostics.pkl"), "wb") as f:
        pkl.dump(diagnostics, f)

    print(f"Estimated Takens parameters: tau={tau_star}, m={m_star}")
    return tau_star, m_star


def run_dm_extraction(
    latents: np.ndarray,
    num_clips: int,
    m: int,
    tau: int,
    dm_cfg: DMConfig,
    base_output_dir: str,
) -> tuple[list, list]:
    """
    Run discrete Morse graph extraction on all clips.

    Args:
        latents: Array of shape (n_clips, n_frames, latent_dim)
        num_clips: Number of clips to process
        m: Takens embedding dimension
        tau: Takens time delay
        dm_cfg: DM configuration
        base_output_dir: Directory to save results

    Returns:
        Tuple of (all_cycle_dm, all_G_sm)
    """
    all_cycle_dm = []
    all_G_sm = []

    os.makedirs(base_output_dir, exist_ok=True)

    for clip_id in tqdm(range(num_clips), desc="DM extraction"):
        clip = latents[clip_id]
        pts = takens_points(clip, m=m, tau=tau)
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


def run_clustering(
    all_cycle_dm: list,
    latents: np.ndarray,
    clustering_cfg: ClusteringConfig,
    n_clips: int,
    output_dir: str,
) -> tuple[dict, list]:
    """
    Cluster extracted cycles using Wasserstein distance.

    Args:
        all_cycle_dm: List of cycles per clip
        latents: Array of shape (n_clips, n_frames, latent_dim)
        clustering_cfg: Clustering configuration
        n_clips: Number of clips
        output_dir: Directory to save results

    Returns:
        Tuple of (cluster_dict, cleaned_loops)
    """
    cluster_dict, cleaned_loops = cluster_vpath_with_wasserstein2d_fixedlen_kmeans(
        all_cycle_dm,
        latents,
        pad=0,
        min_len=clustering_cfg.min_len,
        max_len=clustering_cfg.max_len,
        target_len=clustering_cfg.target_len,
        n_clusters=clustering_cfg.n_clusters,
        n_clips=n_clips,
        drop_nearby_loop_num=0,
        skip_clips=clustering_cfg.skip_clips,
        squared_cost=False,
        only_max=clustering_cfg.only_max,
        only_min=clustering_cfg.only_min,
        use_full_union=False,
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "cluster_dict.pkl"), "wb") as f:
        pkl.dump(cluster_dict, f)
    with open(os.path.join(output_dir, "cleaned_loops.pkl"), "wb") as f:
        pkl.dump(cleaned_loops, f)

    return cluster_dict, cleaned_loops


def plot_cluster_comparison(
    cluster_dict: dict,
    latents: np.ndarray,
    data_dir: str,
    output_dir: str,
    figsize: tuple[int, int] = (14, 6),
    alpha: float = 0.5,
    hull_alpha: float = 0.3,
    point_size: int = 10,
) -> None:
    """
    Plot side-by-side comparison of predicted clusters vs ground truth sub-clusters.

    Args:
        cluster_dict: Clustering results from our method
        latents: Array of shape (n_clips, n_frames, latent_dim)
        data_dir: Directory containing synthetic dataset with ground truth
        output_dir: Directory to save plots
        figsize: Figure size
        alpha: Point transparency
        hull_alpha: Convex hull transparency
        point_size: Scatter point size
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.spatial import ConvexHull
    from sklearn.decomposition import PCA

    dataset_path = os.path.join(data_dir, "dataset.pkl")
    with open(dataset_path, "rb") as f:
        dataset = pkl.load(f)

    all_latents = dataset.all_latents
    pca = PCA(n_components=2, random_state=42)
    all_latents_2d = pca.fit_transform(all_latents)

    behaviors = dataset.config.behaviors
    n_behaviors = len(behaviors)
    n_subclusters = dataset.config.subclusters_per_behavior
    behavior_labels = dataset.all_behavior_labels
    subcluster_labels = np.concatenate([c.subcluster_labels for c in dataset.clips])

    behavior_colors = {
        "walk": "#2A9D8F",
        "rear": "#E63946",
        "groom": "#E9C46A",
    }
    for i, b in enumerate(behaviors):
        if b not in behavior_colors:
            behavior_colors[b] = plt.cm.tab10(i)

    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))

    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

    def get_subcluster_shades(base_color, n_shades):
        if isinstance(base_color, str) and base_color.startswith('#'):
            rgb = hex_to_rgb(base_color)
        else:
            rgb = base_color[:3] if len(base_color) >= 3 else (0.5, 0.5, 0.5)
        shades = []
        lightness_range = [0.4 + 0.6 * i / (n_shades - 1) for i in range(n_shades)] if n_shades > 1 else [0.7]
        for lightness in lightness_range:
            shade = tuple(max(0, min(1, c * lightness + (1 - lightness) * 0.9)) for c in rgb)
            shades.append(rgb_to_hex(shade))
        return shades

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    handles_gt = []

    for b_idx, behavior in enumerate(behaviors):
        base_color = behavior_colors.get(behavior, plt.cm.tab10(b_idx))
        shades = get_subcluster_shades(base_color, n_subclusters)

        for sub_idx in range(n_subclusters):
            mask = (behavior_labels == b_idx) & (subcluster_labels == sub_idx)
            pts = all_latents_2d[mask]

            if len(pts) < 3:
                continue

            shade_color = shades[sub_idx]

            ax.scatter(
                pts[:, 0], pts[:, 1],
                c=shade_color, alpha=alpha * 0.5, s=point_size,
                edgecolors='none', zorder=1
            )

            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                ax.fill(
                    hull_pts[:, 0], hull_pts[:, 1],
                    color=shade_color, alpha=hull_alpha,
                    edgecolor=shade_color, linewidth=2, zorder=2
                )
            except:
                pass

            handles_gt.append(mpatches.Patch(
                color=shade_color, alpha=0.6,
                label=f"{behavior}_{sub_idx}"
            ))

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title("Ground Truth Sub-clusters", fontsize=14, fontweight="bold")
    ax.legend(handles=handles_gt, loc="upper right", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3, linestyle="--")

    ax = axes[1]
    n_pred_clusters = len(cluster_dict)
    pred_colors = plt.cm.tab20(np.linspace(0, 1, max(n_pred_clusters, 1)))

    frames_per_clip = latents.shape[1]
    n_clips = latents.shape[0]
    total_frames = n_clips * frames_per_clip

    pred_labels_full = np.full(total_frames, -1, dtype=int)

    for cluster_id, items in cluster_dict.items():
        for clip_id, frames in items:
            for frame in frames:
                global_idx = clip_id * frames_per_clip + frame
                if global_idx < total_frames:
                    pred_labels_full[global_idx] = cluster_id

    handles_pred = []
    for cluster_id in sorted(cluster_dict.keys()):
        mask = pred_labels_full == cluster_id
        pts = all_latents_2d[mask]

        if len(pts) < 1:
            continue

        color = pred_colors[cluster_id % len(pred_colors)]

        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=[color], alpha=alpha, s=point_size * 2,
            edgecolors='none', zorder=1
        )

        handles_pred.append(mpatches.Patch(
            color=color, alpha=0.6,
            label=f"Cluster {cluster_id}"
        ))

    unclustered_mask = pred_labels_full == -1
    if unclustered_mask.sum() > 0:
        ax.scatter(
            all_latents_2d[unclustered_mask, 0],
            all_latents_2d[unclustered_mask, 1],
            c='lightgray', alpha=0.2, s=point_size // 2,
            edgecolors='none', zorder=0, label='Unclustered'
        )

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title("Predicted Clusters (Our Method)", fontsize=14, fontweight="bold")
    ax.legend(handles=handles_pred, loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")

    for a in axes:
        a.set_xlim(axes[0].get_xlim())
        a.set_ylim(axes[0].get_ylim())

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "cluster_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace(".png", ".pdf"), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved cluster comparison to {save_path}")


def evaluate_clustering(
    cluster_dict: dict,
    latents: np.ndarray,
    data_dir: str,
    output_dir: str,
) -> float:
    """
    Evaluate clustering using Pair Accuracy.

    Pair Accuracy = (TP + TN) / total_pairs
    - TP: same true label AND same predicted label
    - TN: different true label AND different predicted label

    Returns:
        Pair accuracy score (0 to 1)
    """
    dataset_path = os.path.join(data_dir, "dataset.pkl")
    with open(dataset_path, "rb") as f:
        dataset = pkl.load(f)

    pred_labels = []
    true_labels = []

    for cluster_id, items in cluster_dict.items():
        for clip_id, frames in items:
            mid_frame = frames[len(frames) // 2]
            pred_labels.append(cluster_id)
            true_labels.append(dataset.clips[clip_id].cluster_labels[mid_frame])

    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)
    n = len(pred_labels)

    if n < 2:
        print("Not enough samples for evaluation")
        return 0.0

    same_true = np.equal.outer(true_labels, true_labels)
    same_pred = np.equal.outer(pred_labels, pred_labels)

    upper_tri = np.triu_indices(n, k=1)
    same_true_pairs = same_true[upper_tri]
    same_pred_pairs = same_pred[upper_tri]

    tp = np.sum(same_true_pairs & same_pred_pairs)
    tn = np.sum(~same_true_pairs & ~same_pred_pairs)
    total_pairs = len(same_true_pairs)

    pair_accuracy = (tp + tn) / total_pairs

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "pair_accuracy.txt"), "w") as f:
        f.write(f"{pair_accuracy:.4f}\n")

    print(f"\nPair Accuracy: {pair_accuracy:.4f} (n={n} samples, {total_pairs:,} pairs)")

    return pair_accuracy


def main(config_path: str = "configs/synthetic/pipeline.yaml"):
    cfg = load_config(config_path)

    if cfg.clustering.only_max and cfg.clustering.only_min:
        raise ValueError("Cannot use both only_max and only_min")

    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")

    print("Loading synthetic data...")
    latents = load_synthetic_data(cfg.data_dir, cfg.num_clips)
    print(f"Loaded latents: {latents.shape}")

    if cfg.takens.auto_estimate:
        print("Estimating Takens embedding parameters...")
        tau, m = estimate_takens_params(
            latents,
            cfg.takens,
            os.path.join(cfg.output_dir, "takens_estimation"),
        )
    else:
        tau = cfg.takens.tau
        m = cfg.takens.m
        print(f"Using configured Takens parameters: tau={tau}, m={m}")

    if cfg.dm.skip:
        if cfg.dm.results_dir is None:
            raise ValueError("dm.results_dir required when dm.skip is true")
        print(f"Loading existing DM results from {cfg.dm.results_dir}")
        with open(os.path.join(cfg.dm.results_dir, "all_cycle_dm.pkl"), "rb") as f:
            all_cycle_dm = pkl.load(f)
        print(f"Loaded {len(all_cycle_dm)} cycle sets")
    else:
        base_output_dir = os.path.join(
            cfg.output_dir, f"dm_results_m{m}_tau{tau}_p{cfg.dm.pers_delta}"
        )
        all_cycle_dm, _ = run_dm_extraction(
            latents,
            num_clips=cfg.num_clips,
            m=m,
            tau=tau,
            dm_cfg=cfg.dm,
            base_output_dir=base_output_dir,
        )

    print("Running clustering...")
    cluster_output_dir = os.path.join(cfg.output_dir, "clustering")
    cluster_dict, cleaned_loops = run_clustering(
        all_cycle_dm,
        latents,
        clustering_cfg=cfg.clustering,
        n_clips=cfg.num_clips,
        output_dir=cluster_output_dir,
    )
    print(f"Created {len(cluster_dict)} clusters")

    print("\nPlotting cluster comparison...")
    plot_cluster_comparison(
        cluster_dict,
        latents,
        cfg.data_dir,
        cluster_output_dir,
    )

    print("\nEvaluating clustering...")
    pair_acc = evaluate_clustering(
        cluster_dict,
        latents,
        cfg.data_dir,
        cluster_output_dir,
    )

    print("\nPipeline complete!")
    return cluster_dict, cleaned_loops, pair_acc


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/synthetic/benchmark_dmwkc.yaml"
    main(config_path)
