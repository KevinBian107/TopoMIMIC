"""
Benchmark pipeline for KPMS state space models (AR-HMM and SLDS).
Evaluates clustering quality using Dunn Index and visualizes reconstruction.
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
from vnl_tda.pipelines.tda.skeletonization.rendering import render_grid_clusters_side_by_side

from benchmark.kpms_methods import (
    KPMSConfig,
    KPMSResult,
    ARHMMClustering,
    SLDSClustering,
    get_kpms_method,
    plot_reconstruction_comparison,
)


@dataclass
class RenderingConfig:
    enabled: bool = True
    pad: int = 0
    max_samples: int = 10
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)


@dataclass
class CheckpointConfig:
    """Config for loading a specific checkpoint directory."""
    method: str  # "arhmm" or "slds"
    checkpoint_dir: str  # e.g., "out_kpms_benchmark/arhmm/k5"
    k: int  # number of states


@dataclass
class KPMSBenchmarkConfig:
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    output_dir: str = "out_kpms_benchmark"
    num_clips: int = 500
    intention_key: str = "/intention"
    joint_state_key: str = "qpos"
    k_min: int = 5
    k_max: int = 20
    plot_dunn: bool = True
    plot_reconstruction: bool = True
    recon_clip_idx: int = 0
    recon_n_dims: int = 3
    use_pretrained: bool = False
    pretrained_results_path: str = ""
    checkpoints: list[CheckpointConfig] = None  # Alternative: load specific checkpoints
    kpms_config: KPMSConfig = None
    rendering: RenderingConfig = None
    methods: list[str] = field(default_factory=lambda: ["arhmm", "slds"])

    def __post_init__(self):
        if self.kpms_config is None:
            self.kpms_config = KPMSConfig()
        if self.rendering is None:
            self.rendering = RenderingConfig()
        if self.checkpoints is None:
            self.checkpoints = []


def load_config(config_path: str) -> KPMSBenchmarkConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    kpms_cfg_dict = cfg.get("kpms_config", {})
    kpms_config = KPMSConfig(
        latent_dim=kpms_cfg_dict.get("latent_dim", 10),
        kappa=kpms_cfg_dict.get("kappa", 1e5),
        ar_iters=kpms_cfg_dict.get("ar_iters", 50),
        full_iters=kpms_cfg_dict.get("full_iters", 300),
        seed=kpms_cfg_dict.get("seed", 0),
    )

    rendering_dict = cfg.get("rendering", {})
    if "target_size" in rendering_dict:
        rendering_dict["target_size"] = tuple(rendering_dict["target_size"])
    rendering_cfg = RenderingConfig(
        enabled=rendering_dict.get("enabled", True),
        pad=rendering_dict.get("pad", 0),
        max_samples=rendering_dict.get("max_samples", 10),
        framerate=rendering_dict.get("framerate", 30),
        target_size=rendering_dict.get("target_size", (320, 240)),
    )

    checkpoints_list = []
    for ckpt in cfg.get("checkpoints", []):
        checkpoints_list.append(CheckpointConfig(
            method=ckpt["method"],
            checkpoint_dir=ckpt["checkpoint_dir"],
            k=ckpt["k"],
        ))

    return KPMSBenchmarkConfig(
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", KPMSBenchmarkConfig.rodent_rollout_dir),
        output_dir=cfg.get("output_dir", KPMSBenchmarkConfig.output_dir),
        num_clips=cfg.get("num_clips", KPMSBenchmarkConfig.num_clips),
        intention_key=cfg.get("intention_key", "/intention"),
        joint_state_key=cfg.get("joint_state_key", "qpos"),
        k_min=cfg.get("k_min", 5),
        k_max=cfg.get("k_max", 20),
        plot_dunn=cfg.get("plot_dunn", True),
        plot_reconstruction=cfg.get("plot_reconstruction", True),
        recon_clip_idx=cfg.get("recon_clip_idx", 0),
        recon_n_dims=cfg.get("recon_n_dims", 3),
        use_pretrained=cfg.get("use_pretrained", False),
        pretrained_results_path=cfg.get("pretrained_results_path", ""),
        checkpoints=checkpoints_list,
        kpms_config=kpms_config,
        rendering=rendering_cfg,
        methods=cfg.get("methods", ["arhmm", "slds"]),
    )


def compute_dunn_index(
    labels_list: list[np.ndarray],
    joint_data: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute Dunn Index for KPMS syllable assignments.

    Args:
        labels_list: List of syllable arrays per clip
        joint_data: Joint space data (n_clips, n_frames, n_dim)

    Returns:
        Tuple of (dunn_index, min_inter_dist, max_intra_diam)
    """
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


def labels_to_cluster_dict(labels_list: list[np.ndarray]) -> dict[int, list[tuple[int, np.ndarray]]]:
    """
    Convert KPMS syllable labels to cluster_dict format for rendering.

    Args:
        labels_list: List of syllable arrays per clip

    Returns:
        cluster_dict[syllable_id] = [(clip_id, frame_indices), ...]
    """
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


def run_rendering(
    cluster_dict: dict,
    rodent_rollout_dir: str,
    rendering_cfg: RenderingConfig,
    output_dir: str,
    cluster_labels: dict[int, str] | None = None,
    output_filename: str = "cluster_side_by_side.mp4",
    title: str = "Cluster Samples",
) -> None:
    """Render cluster video for KPMS syllables."""
    from vnl_tda.pipelines.tda.skeletonization.postprocess import chunk_loops_by_clustered_vpath

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


def run_kpms_k_sweep(
    method_name: str,
    intention_data: np.ndarray,
    joint_data: np.ndarray,
    k_min: int,
    k_max: int,
    output_dir: str,
    kpms_config: KPMSConfig,
) -> dict:
    """
    Run KPMS method across k values and compute Dunn index.

    Args:
        method_name: "arhmm" or "slds"
        intention_data: Intention array (n_clips, n_frames, n_dim)
        joint_data: Joint state array (n_clips, n_frames, joint_dim)
        k_min: Minimum number of states
        k_max: Maximum number of states
        output_dir: Output directory
        kpms_config: KPMS configuration

    Returns:
        Results dictionary with metrics per k
    """
    print(f"\n=== Running {method_name.upper()} k-sweep (k={k_min} to {k_max}) ===")

    results = {
        "dunn_index": [],
        "min_inter_dist": [],
        "max_intra_diam": [],
        "n_clusters_actual": [],
        "all_labels": {},
        "all_results": {},
        "all_models": {},
    }

    method_dir = Path(output_dir) / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    for k in tqdm(range(k_min, k_max + 1), desc=f"{method_name} k sweep"):
        project_dir = str(method_dir / f"k{k}")

        method = get_kpms_method(method_name, kpms_config)
        kpms_result = method.fit(intention_data, k, project_dir)

        labels_list = kpms_result.labels
        n_actual = len(set(s for sylls in labels_list for s in sylls))

        dunn, min_inter, max_intra = compute_dunn_index(labels_list, joint_data)

        results["dunn_index"].append(dunn)
        results["min_inter_dist"].append(min_inter)
        results["max_intra_diam"].append(max_intra)
        results["n_clusters_actual"].append(n_actual)
        results["all_labels"][k] = labels_list
        results["all_results"][k] = kpms_result.results
        results["all_models"][k] = kpms_result.model

        print(f"  k={k}: Dunn={dunn:.4f}, n_actual={n_actual}")

    return results


def load_pretrained_results(pretrained_path: str, output_dir: str) -> dict:
    """
    Load pretrained KPMS results from a previous run.

    Args:
        pretrained_path: Path to the pretrained results pickle file
        output_dir: Output directory (to look for full results)

    Returns:
        Dictionary with results per method
    """
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained results not found: {pretrained_path}")

    with open(pretrained_path, "rb") as f:
        saved_results = pkl.load(f)

    full_results_path = os.path.join(output_dir, "kpms_benchmark_results_full.pkl")
    if os.path.exists(full_results_path):
        with open(full_results_path, "rb") as f:
            full_results = pkl.load(f)
        return full_results

    return saved_results


def load_checkpoint_results(
    checkpoint: CheckpointConfig,
    intention_data: np.ndarray,
    joint_data: np.ndarray,
) -> dict:
    """
    Load results from a specific KPMS checkpoint directory.

    Args:
        checkpoint: CheckpointConfig with method, checkpoint_dir, and k
        intention_data: Intention array for computing Dunn index
        joint_data: Joint state array for computing Dunn index

    Returns:
        Results dictionary for this checkpoint
    """
    import keypoint_moseq as kpms

    checkpoint_dir = checkpoint.checkpoint_dir
    method = checkpoint.method
    k = checkpoint.k

    if method == "arhmm":
        model_name = f"arhmm_states{k}"
    else:
        model_name = f"slds_states{k}"

    print(f"  Loading checkpoint from: {checkpoint_dir}")
    print(f"  Model name: {model_name}")

    model_ckpt, _, _, _ = kpms.load_checkpoint(checkpoint_dir, model_name)
    model_ckpt['project_dir'] = checkpoint_dir
    model_ckpt['model_name'] = model_name

    n_clips, n_frames, n_dim = intention_data.shape
    coordinates = {}
    confidences = {}
    for i in range(n_clips):
        coords = intention_data[i][:, :, np.newaxis].astype(np.float64)
        coordinates[f"recording_{i:04d}"] = coords
        confidences[f"recording_{i:04d}"] = np.ones((n_frames, n_dim), dtype=np.float64)

    config = kpms.load_config(checkpoint_dir)
    int_data, int_metadata = kpms.format_data(coordinates, confidences, **config)

    results = kpms.extract_results(model_ckpt, int_metadata, checkpoint_dir, model_name, save_results=False)

    rec_names = sorted(results.keys())
    labels_list = [results[rn]['syllable'] for rn in rec_names]

    dunn, min_inter, max_intra = compute_dunn_index(labels_list, joint_data)
    n_actual = len(set(s for sylls in labels_list for s in sylls))

    return {
        "optimal_k": k,
        "optimal_dunn": dunn,
        "dunn_index": [dunn],
        "min_inter_dist": [min_inter],
        "max_intra_diam": [max_intra],
        "n_clusters_actual": [n_actual],
        "all_labels": {k: labels_list},
        "all_results": {k: results},
        "all_models": {k: model_ckpt},
    }


def plot_dunn_analysis(
    results: dict,
    optimal_k: int,
    method_name: str,
    k_min: int,
    k_max: int,
    output_dir: str,
) -> None:
    """Plot Dunn index analysis for a KPMS method."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_dunn, ax_inter, ax_intra = axes

    k_list = list(range(k_min, k_max + 1))

    ax_dunn.plot(k_list, results["dunn_index"], color="darkgreen", marker="o", linewidth=2, markersize=6)
    ax_dunn.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax_dunn.set_xlabel("Number of states (k)", fontsize=12)
    ax_dunn.set_ylabel("Dunn Index", fontsize=12)
    ax_dunn.set_title(f"{method_name.upper()}: Dunn Index", fontsize=14)
    ax_dunn.legend()
    ax_dunn.grid(True, alpha=0.3)

    ax_inter.plot(k_list, results["min_inter_dist"], color="blue", marker="s", linewidth=2, markersize=6)
    ax_inter.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax_inter.set_xlabel("Number of states (k)", fontsize=12)
    ax_inter.set_ylabel("Min Inter-cluster Distance", fontsize=12)
    ax_inter.set_title("Min Inter-cluster Distance", fontsize=14)
    ax_inter.legend()
    ax_inter.grid(True, alpha=0.3)

    ax_intra.plot(k_list, results["max_intra_diam"], color="purple", marker="^", linewidth=2, markersize=6)
    ax_intra.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
    ax_intra.set_xlabel("Number of states (k)", fontsize=12)
    ax_intra.set_ylabel("Max Intra-cluster Diameter", fontsize=12)
    ax_intra.set_title("Max Intra-cluster Diameter", fontsize=14)
    ax_intra.legend()
    ax_intra.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dunn_analysis.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "dunn_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {method_name} Dunn analysis to {output_dir}/dunn_analysis.pdf")


def plot_comparison(
    all_results: dict[str, dict],
    k_min: int,
    k_max: int,
    output_dir: str,
) -> None:
    """Plot Dunn index comparison between AR-HMM and SLDS."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    k_list = list(range(k_min, k_max + 1))

    colors = {"arhmm": "#2E86AB", "slds": "#A23B72"}
    markers = {"arhmm": "o", "slds": "s"}

    for method_name, results in all_results.items():
        ax.plot(k_list, results["dunn_index"],
                color=colors.get(method_name, "gray"),
                marker=markers.get(method_name, "o"),
                linewidth=2, markersize=6,
                label=method_name.upper())

        optimal_idx = np.argmax(results["dunn_index"])
        optimal_k = k_list[optimal_idx]
        optimal_dunn = results["dunn_index"][optimal_idx]
        ax.scatter([optimal_k], [optimal_dunn], s=150, color=colors.get(method_name, "gray"),
                   marker="*", zorder=5, edgecolors="black", linewidths=1)

    ax.set_xlabel("Number of states (k)", fontsize=12)
    ax.set_ylabel("Dunn Index", fontsize=12)
    ax.set_title("AR-HMM vs SLDS: Dunn Index Comparison", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dunn_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "dunn_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison to {output_dir}/dunn_comparison.pdf")


def main(config_path: str = None):
    if config_path is None:
        config_path = "configs/benchmark_kpms.yaml"

    if os.path.exists(config_path):
        cfg = load_config(config_path)
    else:
        cfg = KPMSBenchmarkConfig()

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print("KPMS Benchmark Pipeline (AR-HMM vs SLDS)")
    print("=" * 60)

    int_rodent = None
    joint_data = None

    # Determine which mode to use:
    # 1. Load from specific checkpoints (highest priority)
    # 2. Load from pretrained results file
    # 3. Train from scratch
    use_checkpoints = len(cfg.checkpoints) > 0
    use_pretrained = cfg.use_pretrained and cfg.pretrained_results_path and not use_checkpoints

    if use_checkpoints:
        print("\n=== Loading from specific checkpoints ===")

        print("\n=== Stage 1: Loading intention data ===")
        work_rodent = functools.partial(
            get_aggregate_data, cfg.intention_key, [], path=cfg.rodent_rollout_dir
        )
        with mp.Pool(processes=mp.cpu_count()) as pool:
            int_rodent = list(tqdm(pool.imap(work_rodent, range(cfg.num_clips)), total=cfg.num_clips))
        int_rodent = np.vstack(int_rodent)
        n_frames = int_rodent.shape[0] // cfg.num_clips
        n_dim = int_rodent.shape[1] if int_rodent.ndim == 2 else 16
        int_rodent = int_rodent.reshape([cfg.num_clips, n_frames, n_dim])
        print(f"Loaded intention: {int_rodent.shape}")

        print(f"\n=== Stage 2: Loading joint state data ({cfg.joint_state_key}) ===")
        work_joint = functools.partial(
            get_aggregate_data, f"/{cfg.joint_state_key}", [], path=cfg.rodent_rollout_dir
        )
        with mp.Pool(processes=mp.cpu_count()) as pool:
            joint_list = list(tqdm(pool.imap(work_joint, range(cfg.num_clips)), total=cfg.num_clips))

        first_clip = joint_list[0]
        if first_clip.ndim == 2:
            joint_n_frames, joint_dim = first_clip.shape
        else:
            total_size = first_clip.size
            joint_dim = total_size // 500
            joint_n_frames = 500

        joint_data = np.array([
            clip.reshape(joint_n_frames, joint_dim)[:n_frames]
            for clip in joint_list
        ])
        print(f"Loaded joint data: {joint_data.shape}")

        print("\n=== Stage 3: Loading checkpoints ===")
        all_results = {}

        for ckpt in cfg.checkpoints:
            print(f"\n{'='*60}")
            print(f"Checkpoint: {ckpt.method.upper()} k={ckpt.k}")
            print(f"{'='*60}")

            results = load_checkpoint_results(ckpt, int_rodent, joint_data)
            all_results[ckpt.method] = results

            print(f"  Dunn={results['optimal_dunn']:.4f}")

    elif use_pretrained:
        print("\n=== Loading pretrained results ===")
        all_results = load_pretrained_results(cfg.pretrained_results_path, cfg.output_dir)
        print(f"Loaded pretrained results from: {cfg.pretrained_results_path}")

        if cfg.rendering.enabled or cfg.plot_reconstruction:
            print("\n=== Stage 1: Loading intention data (for visualization) ===")
            work_rodent = functools.partial(
                get_aggregate_data, cfg.intention_key, [], path=cfg.rodent_rollout_dir
            )
            with mp.Pool(processes=mp.cpu_count()) as pool:
                int_rodent = list(tqdm(pool.imap(work_rodent, range(cfg.num_clips)), total=cfg.num_clips))
            int_rodent = np.vstack(int_rodent)
            n_frames = int_rodent.shape[0] // cfg.num_clips
            n_dim = int_rodent.shape[1] if int_rodent.ndim == 2 else 16
            int_rodent = int_rodent.reshape([cfg.num_clips, n_frames, n_dim])
            print(f"Loaded intention: {int_rodent.shape}")

    else:
        print("\n=== Stage 1: Loading intention data ===")
        work_rodent = functools.partial(
            get_aggregate_data, cfg.intention_key, [], path=cfg.rodent_rollout_dir
        )
        with mp.Pool(processes=mp.cpu_count()) as pool:
            int_rodent = list(tqdm(pool.imap(work_rodent, range(cfg.num_clips)), total=cfg.num_clips))
        int_rodent = np.vstack(int_rodent)
        n_frames = int_rodent.shape[0] // cfg.num_clips
        n_dim = int_rodent.shape[1] if int_rodent.ndim == 2 else 16
        int_rodent = int_rodent.reshape([cfg.num_clips, n_frames, n_dim])
        print(f"Loaded intention: {int_rodent.shape}")

        print(f"\n=== Stage 2: Loading joint state data ({cfg.joint_state_key}) ===")
        work_joint = functools.partial(
            get_aggregate_data, f"/{cfg.joint_state_key}", [], path=cfg.rodent_rollout_dir
        )
        with mp.Pool(processes=mp.cpu_count()) as pool:
            joint_list = list(tqdm(pool.imap(work_joint, range(cfg.num_clips)), total=cfg.num_clips))

        first_clip = joint_list[0]
        if first_clip.ndim == 2:
            joint_n_frames, joint_dim = first_clip.shape
        else:
            total_size = first_clip.size
            joint_dim = total_size // 500
            joint_n_frames = 500

        joint_data = np.array([
            clip.reshape(joint_n_frames, joint_dim)[:n_frames]
            for clip in joint_list
        ])
        print(f"Loaded joint data: {joint_data.shape}")

        print("\n=== Stage 3: Running KPMS methods ===")
        all_results = {}

        for method_name in cfg.methods:
            print(f"\n{'='*60}")
            print(f"Method: {method_name.upper()}")
            print(f"{'='*60}")

            results = run_kpms_k_sweep(
                method_name=method_name,
                intention_data=int_rodent,
                joint_data=joint_data,
                k_min=cfg.k_min,
                k_max=cfg.k_max,
                output_dir=cfg.output_dir,
                kpms_config=cfg.kpms_config,
            )

            optimal_idx = np.argmax(results["dunn_index"])
            optimal_k = cfg.k_min + optimal_idx
            optimal_dunn = results["dunn_index"][optimal_idx]

            results["optimal_k"] = optimal_k
            results["optimal_dunn"] = optimal_dunn
            all_results[method_name] = results

            print(f"\nOptimal k={optimal_k} for {method_name.upper()} (Dunn={optimal_dunn:.4f})")

    for method_name in cfg.methods:
        if method_name not in all_results:
            continue

        results = all_results[method_name]
        optimal_k = results.get("optimal_k", cfg.k_min)
        method_output_dir = os.path.join(cfg.output_dir, method_name)
        os.makedirs(method_output_dir, exist_ok=True)

        if cfg.plot_dunn and "dunn_index" in results and len(results["dunn_index"]) > 1:
            plot_dunn_analysis(
                results, optimal_k, method_name,
                cfg.k_min, cfg.k_max, method_output_dir
            )

        if cfg.plot_reconstruction and int_rodent is not None:
            if "all_models" in results and "all_results" in results:
                model = results["all_models"].get(optimal_k)
                kpms_results = results["all_results"].get(optimal_k)

                if model is not None and kpms_results is not None:
                    rec_name = f"recording_{cfg.recon_clip_idx:04d}"
                    if rec_name in kpms_results:
                        from benchmark.kpms_methods import BaseKPMSMethod

                        class TempMethod(BaseKPMSMethod):
                            def __init__(self):
                                super().__init__("temp")
                            def fit(self, *args, **kwargs):
                                pass

                        temp = TempMethod()
                        reconstructed = temp._compute_reconstruction(model, kpms_results, rec_name)
                        original = int_rodent[cfg.recon_clip_idx]
                        syllables = results["all_labels"][optimal_k][cfg.recon_clip_idx]

                        plot_reconstruction_comparison(
                            original=original,
                            reconstructed=reconstructed,
                            syllables=syllables,
                            title=f"{method_name.upper()} Reconstruction (k={optimal_k})",
                            save_path=os.path.join(method_output_dir, f"reconstruction_clip{cfg.recon_clip_idx}.png"),
                            n_dims=cfg.recon_n_dims,
                        )
                        print(f"Saved reconstruction plot to {method_output_dir}/reconstruction_clip{cfg.recon_clip_idx}.png")

        if cfg.rendering.enabled and "all_labels" in results:
            labels_list = results["all_labels"].get(optimal_k)
            if labels_list is not None:
                cluster_dict = labels_to_cluster_dict(labels_list)
                cluster_labels = {i: f"S{i}" for i in cluster_dict.keys()}

                run_rendering(
                    cluster_dict=cluster_dict,
                    rodent_rollout_dir=cfg.rodent_rollout_dir,
                    rendering_cfg=cfg.rendering,
                    output_dir=method_output_dir,
                    cluster_labels=cluster_labels,
                    output_filename=f"syllables_k{optimal_k}.mp4",
                    title=f"{method_name.upper()} Syllables (k={optimal_k})",
                )

    print("\n=== Generating comparison plots ===")
    has_full_sweep = any(len(r.get("dunn_index", [])) > 1 for r in all_results.values())
    if len(all_results) > 1 and cfg.plot_dunn and has_full_sweep:
        plot_comparison(all_results, cfg.k_min, cfg.k_max, cfg.output_dir)

    print("\n=== Saving results ===")
    results_path = os.path.join(cfg.output_dir, "kpms_benchmark_results.pkl")
    save_results = {
        method: {
            "optimal_k": r.get("optimal_k", cfg.k_min),
            "optimal_dunn": r.get("optimal_dunn", 0.0),
            "dunn_index": r.get("dunn_index", []),
            "min_inter_dist": r.get("min_inter_dist", []),
            "max_intra_diam": r.get("max_intra_diam", []),
            "n_clusters_actual": r.get("n_clusters_actual", []),
        }
        for method, r in all_results.items()
    }
    with open(results_path, "wb") as f:
        pkl.dump(save_results, f)
    print(f"Saved results to: {results_path}")

    if not cfg.use_pretrained:
        full_results_path = os.path.join(cfg.output_dir, "kpms_benchmark_results_full.pkl")
        with open(full_results_path, "wb") as f:
            pkl.dump(all_results, f)
        print(f"Saved full results to: {full_results_path}")

    summary_path = os.path.join(cfg.output_dir, "kpms_benchmark_summary.txt")
    with open(summary_path, "w") as f:
        f.write("KPMS Benchmark Summary (AR-HMM vs SLDS)\n")
        f.write("=" * 60 + "\n\n")

        sorted_methods = sorted(
            all_results.items(),
            key=lambda x: x[1].get("optimal_dunn", 0.0),
            reverse=True,
        )

        f.write("Ranking by Dunn Index:\n")
        f.write("-" * 40 + "\n")
        for rank, (method_name, result) in enumerate(sorted_methods, 1):
            f.write(f"{rank}. {method_name.upper()}: Dunn={result.get('optimal_dunn', 0.0):.4f} (k={result.get('optimal_k', 'N/A')})\n")

    print(f"Saved summary to: {summary_path}")

    print("\n" + "=" * 60)
    print("KPMS Benchmark Complete!")
    print("=" * 60)

    print("\nFinal Rankings:")
    sorted_methods = sorted(
        all_results.items(),
        key=lambda x: x[1].get("optimal_dunn", 0.0),
        reverse=True,
    )
    for rank, (method_name, result) in enumerate(sorted_methods, 1):
        print(f"  {rank}. {method_name.upper()}: Dunn={result.get('optimal_dunn', 0.0):.4f} (k={result.get('optimal_k', 'N/A')})")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/benchmark_kpms.yaml"
    main(config_path)
