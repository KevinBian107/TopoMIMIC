"""
Discrete Morse graph extraction pipeline.

Refactored from dm_latent_subspace.py
Provides DM extraction, cycle detection, and optional W2-based clustering.
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
from tqdm import tqdm

from topo_mimic.extraction.utils.load_utils import get_aggregate_data, get_clip_indices
from topo_mimic.extraction.utils.feature_transforms import apply_dm_key_transform, get_hdf5_key_for_dm_key
from topo_mimic.extraction.utils.dataset_config import get_dataset_preset
from topo_mimic.extraction.skeletonization.skeletonization import extract_skeleton_pcd_dm
from topo_mimic.extraction.skeletonization.remap import (
    parse_swr_remap,
    remap_cycle_to_original,
    split_cycle_into_segments,
)
from topo_mimic.extraction.skeletonization.preprocess import estimate_delay_embedding_params
from topo_mimic.extraction.skeletonization.postprocess import (
    chunk_loops_by_clustered_vpath,
    _expand_frames_nonconsec,
    _uniform_downsample,
)
from topo_mimic.extraction.skeletonization.rendering import render_grid_clusters_side_by_side


def plot_ami_fnn_diagnostics(
    diagnostics: dict,
    tau_star: int,
    m_star: int,
    output_path: str,
    fnn_thr: float = 0.05,
    figsize: tuple[float, float] = (12, 4.5),
) -> None:
    """
    Plot AMI and FNN diagnostic curves from saved diagnostics.

    Args:
        diagnostics: Dict containing I_mean, I_se, fnn_mean, fnn_se, ms arrays
        tau_star: Selected delay parameter
        m_star: Selected embedding dimension
        output_path: Path to save the plot
        fnn_thr: FNN threshold used
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    I_mean = diagnostics["I_mean"]
    I_se = diagnostics["I_se"]
    fnn_mean = diagnostics["fnn_mean"]
    fnn_se = diagnostics["fnn_se"]
    ms = diagnostics["ms"]
    theiler_used = diagnostics.get("theiler_used", tau_star)

    taus = np.arange(1, len(I_mean) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(taus, I_mean, label="mean AMI")
    if np.all(np.isfinite(I_se)):
        ax1.fill_between(taus, I_mean - I_se, I_mean + I_se, alpha=0.2, linewidth=0)
    ax1.axvline(tau_star, ls="--", color="red", label=f"τ* = {tau_star}")
    ax1.set_xlabel("Delay τ")
    ax1.set_ylabel("AMI")
    ax1.set_title("Average Mutual Information")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ms, fnn_mean, label="mean FNN")
    if np.all(np.isfinite(fnn_se)):
        ax2.fill_between(ms, fnn_mean - fnn_se, fnn_mean + fnn_se, alpha=0.2, linewidth=0)
    ax2.axhline(fnn_thr, ls=":", color="gray", label=f"threshold = {fnn_thr}")
    ax2.axvline(m_star, ls="--", color="red", label=f"m* = {m_star}")
    ax2.set_xlabel("Embedding dimension m")
    ax2.set_ylabel("Fraction of false neighbors")
    ax2.set_title(f"FNN at τ = {tau_star} (Theiler={theiler_used})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Delay-Embedding Parameter Selection (AMI → τ*, FNN → m*)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved AMI/FNN plot to {output_path}")


@dataclass
class DataConfig:
    dataset: str = "rodent"
    dm_key: str = "intention"
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    split: str = "train"
    pca_dims: int | None = None
    xml_path: str = "topo_mimic/extraction/xml/rodent.xml"
    stac_data_path: str = "data/rodent_reference_clips.h5"


@dataclass
class TakensConfig:
    auto_estimate: bool = False
    m: int = 3
    tau: int = 6
    max_tau: int = 200
    bins: int = 64
    m_max: int = 15
    R_tol: float = 15.0
    A_tol: float = 2.0
    fnn_thr: float = 0.05
    theiler: str | int | None = "auto"
    max_points: int = 20000


@dataclass
class DMConfig:
    pers_delta: float = 0.75
    k_neighbors: int = 5
    epsilon: float = 0.99
    skip: bool = False
    results_dir: str | None = None
    segment_gap_limit: int = 50
    min_segment_len: int = 50
    use_dm_path: bool = False


@dataclass
class RenderingConfig:
    enabled: bool = False
    pad: int = 0
    max_samples: int = 5
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)


@dataclass
class PipelineConfig:
    output_dir: str = "outputs/dm_pipeline"
    data: DataConfig = None
    takens: TakensConfig = None
    dm: DMConfig = None
    rendering: RenderingConfig = None

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.takens is None:
            self.takens = TakensConfig()
        if self.dm is None:
            self.dm = DMConfig()
        if self.rendering is None:
            self.rendering = RenderingConfig()


def load_config(config_path: str) -> PipelineConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dict = cfg.get("data", {})
    if "pca_dims" in data_dict and data_dict["pca_dims"] is None:
        data_dict["pca_dims"] = None
    data_cfg = DataConfig(**data_dict)

    takens_raw = cfg.get("takens", {})
    theiler_val = takens_raw.get("theiler", "auto")
    if theiler_val == "auto":
        theiler_val = "auto"
    elif theiler_val is not None:
        theiler_val = int(theiler_val)
    takens_raw["theiler"] = theiler_val
    takens_cfg = TakensConfig(**takens_raw)
    dm_cfg = DMConfig(**cfg.get("dm", {}))

    rendering_dict = cfg.get("rendering", {})
    if "target_size" in rendering_dict:
        rendering_dict["target_size"] = tuple(rendering_dict["target_size"])
    rendering_cfg = RenderingConfig(**rendering_dict)

    return PipelineConfig(
        output_dir=cfg.get("output_dir", "outputs/dm_pipeline"),
        data=data_cfg,
        takens=takens_cfg,
        dm=dm_cfg,
        rendering=rendering_cfg,
    )


def load_data(config: DataConfig, output_dir: str | None = None) -> tuple[np.ndarray, list[int]]:
    preset = get_dataset_preset(config.dataset)
    clip_indices = get_clip_indices(
        config.split,
        total_clips=preset.total_clips,
        test_size=preset.test_size,
    )
    hdf5_key = get_hdf5_key_for_dm_key(config.dm_key)
    work_fn = functools.partial(
        get_aggregate_data, f"/{hdf5_key}", [], path=config.rodent_rollout_dir
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        data_list = list(tqdm(pool.imap(work_fn, clip_indices), total=len(clip_indices), desc=f"Loading {hdf5_key}"))

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
    print(f"Loaded {hdf5_key}: {data.shape} (split={config.split}, clips={clip_indices[0]}-{clip_indices[-1]})")

    render_output_dir = os.path.join(output_dir, "keypoint_renders") if output_dir and config.dm_key == "ekp" else None
    data = apply_dm_key_transform(
        data, config.dm_key, config.xml_path, config.stac_data_path, render_output_dir,
        dataset=config.dataset,
    )
    print(f"After transform ({config.dm_key}): {data.shape}")

    if config.pca_dims is not None:
        from sklearn.decomposition import PCA
        n_clips, n_frames_d, n_dim_d = data.shape
        flat = data.reshape(-1, n_dim_d)
        pca = PCA(n_components=config.pca_dims)
        flat_reduced = pca.fit_transform(flat)
        data = flat_reduced.reshape(n_clips, n_frames_d, config.pca_dims)
        print(f"PCA reduced: {n_dim_d} -> {config.pca_dims} dims (var explained: {pca.explained_variance_ratio_.sum():.3f})")

    return data, clip_indices


def takens_embedding(X: np.ndarray, m: int, tau: int) -> np.ndarray:
    n_frames, D = X.shape
    start = (m - 1) * tau
    idxs = np.arange(start, n_frames)
    pts = np.stack([X[[j - a * tau for a in range(m)]].reshape(-1) for j in idxs], axis=0)
    return pts


def run_dm_extraction(
    data: np.ndarray,
    config: PipelineConfig,
) -> tuple[list, list, list | None]:
    os.makedirs(config.output_dir, exist_ok=True)

    use_dm_path = config.dm.use_dm_path
    if use_dm_path:
        from topo_mimic.extraction.skeletonization.dm_path import extract_dm_paths

    n_total = len(data)

    cycle_pkl = os.path.join(config.output_dir, "all_cycle_dm.pkl")
    g_pkl = os.path.join(config.output_dir, "all_G_sm.pkl")
    path_pkl = os.path.join(config.output_dir, "all_path_dm.pkl")

    def _load_existing(path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pkl.load(f)
        return None

    existing_cycle = _load_existing(cycle_pkl)
    existing_G = _load_existing(g_pkl)
    existing_path = _load_existing(path_pkl) if use_dm_path else None

    all_cycle_dm: list = list(existing_cycle) if existing_cycle is not None else []
    all_G_sm: list = list(existing_G) if existing_G is not None else []
    all_path_dm: list | None = list(existing_path) if (existing_path is not None) else ([] if use_dm_path else None)

    sentinel = object()
    while len(all_cycle_dm) < n_total:
        all_cycle_dm.append(sentinel)
    while len(all_G_sm) < n_total:
        all_G_sm.append(sentinel)
    if use_dm_path:
        while len(all_path_dm) < n_total:
            all_path_dm.append(sentinel)

    n_cached = sum(1 for x in all_cycle_dm[:n_total] if x is not sentinel)
    print(f"[run_dm_extraction] Found {n_cached}/{n_total} clip results already in aggregate pkls; will skip those clips.")

    for clip_id in tqdm(range(n_total), desc="DM extraction"):
        clip_output_dir = os.path.join(config.output_dir, "clips", f"clip_{clip_id}")

        if all_cycle_dm[clip_id] is not sentinel and all_G_sm[clip_id] is not sentinel and (not use_dm_path or all_path_dm[clip_id] is not sentinel):
            if os.path.isdir(clip_output_dir):
                continue

        os.makedirs(clip_output_dir, exist_ok=True)
        clip = data[clip_id]
        pts = takens_embedding(clip, m=config.takens.m, tau=config.takens.tau)

        cycle_inds, G_sm = extract_skeleton_pcd_dm(
            pts,
            k=config.dm.k_neighbors,
            pers_delta=config.dm.pers_delta,
            epsilon=config.dm.epsilon,
            output_dir=clip_output_dir,
            cleanup=False,
        )

        cycle_inds = [np.array(inds, dtype=int) for inds in cycle_inds]
        all_G_sm[clip_id] = G_sm

        remap = parse_swr_remap(clip_output_dir)
        n_takens = len(pts)

        if not cycle_inds:
            all_cycle_dm[clip_id] = []
            if use_dm_path:
                path_result = extract_dm_paths(
                    G_sm, clip_output_dir, remap, n_takens,
                    gap_limit=config.dm.segment_gap_limit,
                    min_segment_len=config.dm.min_segment_len,
                )
                all_path_dm[clip_id] = path_result.paths_remapped
                print(f"Clip {clip_id}: 0 cycles, {len(path_result.paths_remapped)} paths, {G_sm.number_of_edges()} edges")
            else:
                print(f"Clip {clip_id}: 0 cycles, {G_sm.number_of_edges()} edges")
            continue

        remapped_cycles = []
        for cycle in cycle_inds:
            orig = remap_cycle_to_original(cycle, remap, n_takens)
            if len(orig) > 0:
                remapped_cycles.append(orig)

        longest = max(remapped_cycles, key=len)
        segments = split_cycle_into_segments(longest, gap_limit=config.dm.segment_gap_limit)
        segments = [seg for seg in segments if len(seg) >= config.dm.min_segment_len]

        all_cycle_dm[clip_id] = segments

        if use_dm_path:
            path_result = extract_dm_paths(
                G_sm, clip_output_dir, remap, n_takens,
                gap_limit=config.dm.segment_gap_limit,
                min_segment_len=config.dm.min_segment_len,
            )
            all_path_dm[clip_id] = path_result.paths_remapped
            print(f"Clip {clip_id}: {len(cycle_inds)} cycles -> {len(segments)} segments, {len(path_result.paths_remapped)} paths, {G_sm.number_of_edges()} edges")
        else:
            print(f"Clip {clip_id}: {len(cycle_inds)} cycles -> {len(segments)} segments, {G_sm.number_of_edges()} edges")

    all_cycle_dm = [x for x in all_cycle_dm[:n_total] if x is not sentinel]
    all_G_sm = [x for x in all_G_sm[:n_total] if x is not sentinel]
    if use_dm_path:
        all_path_dm = [x for x in all_path_dm[:n_total] if x is not sentinel]

    assert len(all_cycle_dm) == n_total, f"aggregate cycle list has {len(all_cycle_dm)} entries, expected {n_total}"
    assert len(all_G_sm) == n_total, f"aggregate G_sm list has {len(all_G_sm)} entries, expected {n_total}"

    with open(g_pkl, "wb") as f:
        pkl.dump(all_G_sm, f)
    with open(cycle_pkl, "wb") as f:
        pkl.dump(all_cycle_dm, f)

    if use_dm_path:
        assert len(all_path_dm) == n_total, f"aggregate path list has {len(all_path_dm)} entries, expected {n_total}"
        with open(path_pkl, "wb") as f:
            pkl.dump(all_path_dm, f)

    print(f"\nSaved DM results to {config.output_dir}")

    return all_cycle_dm, all_G_sm, all_path_dm


def get_cycle_statistics(all_cycle_dm: list) -> dict:
    n_clips = len(all_cycle_dm)
    total_cycles = sum(len(cycles) for cycles in all_cycle_dm)
    clips_with_cycles = sum(1 for cycles in all_cycle_dm if len(cycles) > 0)

    all_lengths = []
    for cycles in all_cycle_dm:
        for cycle in cycles:
            all_lengths.append(len(cycle))

    stats = {
        "n_clips": n_clips,
        "total_cycles": total_cycles,
        "clips_with_cycles": clips_with_cycles,
        "cycles_per_clip": total_cycles / n_clips if n_clips > 0 else 0,
    }

    if all_lengths:
        stats["cycle_length_mean"] = np.mean(all_lengths)
        stats["cycle_length_std"] = np.std(all_lengths)
        stats["cycle_length_min"] = np.min(all_lengths)
        stats["cycle_length_max"] = np.max(all_lengths)

    return stats


def get_path_statistics(all_path_dm: list) -> dict:
    n_clips = len(all_path_dm)
    total_paths = sum(len(paths) for paths in all_path_dm)
    clips_with_paths = sum(1 for paths in all_path_dm if len(paths) > 0)

    all_lengths = []
    for paths in all_path_dm:
        for path in paths:
            all_lengths.append(len(path))

    stats = {
        "n_clips": n_clips,
        "total_paths": total_paths,
        "clips_with_paths": clips_with_paths,
        "paths_per_clip": total_paths / n_clips if n_clips > 0 else 0,
    }

    if all_lengths:
        stats["path_length_mean"] = np.mean(all_lengths)
        stats["path_length_std"] = np.std(all_lengths)
        stats["path_length_min"] = np.min(all_lengths)
        stats["path_length_max"] = np.max(all_lengths)

    return stats


def run_pipeline(config: PipelineConfig) -> tuple[list, list, list | None]:
    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("Discrete Morse Pipeline")
    print("=" * 60)

    all_path_dm = None

    if config.dm.skip:
        if config.dm.results_dir is None:
            raise ValueError("dm.results_dir required when dm.skip is True")

        print(f"Loading existing DM results from {config.dm.results_dir}")
        with open(os.path.join(config.dm.results_dir, "all_cycle_dm.pkl"), "rb") as f:
            all_cycle_dm = pkl.load(f)

        all_G_sm = None
        graph_path = os.path.join(config.dm.results_dir, "all_G_sm.pkl")
        if os.path.exists(graph_path):
            with open(graph_path, "rb") as f:
                all_G_sm = pkl.load(f)

        path_path = os.path.join(config.dm.results_dir, "all_path_dm.pkl")
        if os.path.exists(path_path):
            with open(path_path, "rb") as f:
                all_path_dm = pkl.load(f)
            print(f"Loaded {len(all_path_dm)} path sets")

        print(f"Loaded {len(all_cycle_dm)} cycle sets")
    else:
        print("\n=== Stage 1: Loading data ===")
        data, clip_indices = load_data(config.data, output_dir=config.output_dir)

        if config.takens.auto_estimate:
            print("\n=== Stage 2: Auto-estimating Takens parameters (AMI + FNN) ===")
            hdf5_key = get_hdf5_key_for_dm_key(config.data.dm_key)
            transform_fn = None
            if config.data.dm_key != "intention":
                transform_fn = functools.partial(
                    apply_dm_key_transform,
                    dm_key=config.data.dm_key,
                    xml_path=config.data.xml_path,
                    stac_data_path=config.data.stac_data_path,
                    dataset=config.data.dataset,
                )
            tau_star, m_star, diagnostics = estimate_delay_embedding_params(
                rodent_rollout_dir=config.data.rodent_rollout_dir,
                get_aggregate_data_fn=get_aggregate_data,
                num_clips=len(clip_indices),
                intention_key=f"/{hdf5_key}",
                clip_indices=clip_indices,
                transform_fn=transform_fn,
                max_tau=config.takens.max_tau,
                bins=config.takens.bins,
                m_max=config.takens.m_max,
                R_tol=config.takens.R_tol,
                A_tol=config.takens.A_tol,
                fnn_thr=config.takens.fnn_thr,
                theiler=config.takens.theiler,
                max_points=config.takens.max_points,
                plot=False,
            )
            config.takens.tau = tau_star
            config.takens.m = m_star
            print(f"Estimated: tau* = {tau_star}, m* = {m_star}")

            diag_path = os.path.join(config.output_dir, "ami_fnn_diagnostics.pkl")
            with open(diag_path, "wb") as f:
                pkl.dump({"tau_star": tau_star, "m_star": m_star, "diagnostics": diagnostics}, f)
            print(f"Saved diagnostics to {diag_path}")

            plot_path = os.path.join(config.output_dir, "ami_fnn_diagnostics.png")
            plot_ami_fnn_diagnostics(
                diagnostics,
                tau_star,
                m_star,
                plot_path,
                fnn_thr=config.takens.fnn_thr,
            )
        else:
            print(f"\n=== Stage 2: Using fixed Takens parameters (m={config.takens.m}, tau={config.takens.tau}) ===")

        print("\n=== Stage 3: Running DM extraction ===")
        all_cycle_dm, all_G_sm, all_path_dm = run_dm_extraction(data, config)

    stats = get_cycle_statistics(all_cycle_dm)
    print("\n=== Cycle Statistics ===")
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.2f}")
        else:
            print(f"  {key}: {val}")

    stats_path = os.path.join(config.output_dir, "cycle_stats.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(stats, f)
    print(f"\nSaved stats to {stats_path}")

    if all_path_dm is not None:
        path_stats = get_path_statistics(all_path_dm)
        print("\n=== Path Statistics ===")
        for key, val in path_stats.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.2f}")
            else:
                print(f"  {key}: {val}")

        path_stats_path = os.path.join(config.output_dir, "path_stats.yaml")
        with open(path_stats_path, "w") as f:
            yaml.dump(path_stats, f)
        print(f"Saved path stats to {path_stats_path}")

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)

    return all_cycle_dm, all_G_sm, all_path_dm


def plot_diagnostics_from_pickle(
    pickle_path: str,
    output_path: str | None = None,
    fnn_thr: float = 0.05,
) -> tuple[int, int]:
    """
    Load diagnostics from pickle and generate AMI/FNN plot.

    Args:
        pickle_path: Path to ami_fnn_diagnostics.pkl
        output_path: Path to save plot (defaults to same dir as pickle)
        fnn_thr: FNN threshold for plot

    Returns:
        Tuple of (tau_star, m_star)
    """
    with open(pickle_path, "rb") as f:
        data = pkl.load(f)

    tau_star = data["tau_star"]
    m_star = data["m_star"]
    diagnostics = data["diagnostics"]

    if output_path is None:
        output_path = pickle_path.replace(".pkl", ".png")

    plot_ami_fnn_diagnostics(
        diagnostics,
        tau_star,
        m_star,
        output_path,
        fnn_thr=fnn_thr,
    )

    return tau_star, m_star


def main(config_path: str = "configs/dmwkc/dm_pipeline.yaml"):
    if os.path.exists(config_path):
        cfg = load_config(config_path)
    else:
        cfg = PipelineConfig()

    run_pipeline(cfg)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dmwkc/dm_pipeline.yaml"
    main(config_path)
