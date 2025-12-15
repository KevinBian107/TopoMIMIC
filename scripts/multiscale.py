"""Multiscale DM Latent Subspace Pipeline.

Splits clips into smaller segments before running DM extraction and clustering.
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
from dataclasses import dataclass
from tqdm import tqdm

from vnl_tda.utils.load_utils import get_aggregate_data, split_clips_by_frames
from scripts.dm_latent_subspace_auto_dunn import (
    DMConfig,
    ClusteringConfig,
    AutoClusterConfig,
    RenderingConfig,
    takens_points,
    run_dm_extraction,
    prepare_windows_for_clustering,
    compute_w2_distance_matrix,
    auto_select_clusters,
    run_rendering,
)


M = 3
TAU = 6


@dataclass
class MultiscaleConfig:
    frames_per_clip: int = 100


@dataclass
class MultiscalePipelineConfig:
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    output_dir: str = "out/out_multiscale"
    num_clips: int = 500
    multiscale: MultiscaleConfig = None
    dm: DMConfig = None
    clustering: ClusteringConfig = None
    auto_cluster: AutoClusterConfig = None
    rendering: RenderingConfig = None

    def __post_init__(self):
        if self.multiscale is None:
            self.multiscale = MultiscaleConfig()
        if self.dm is None:
            self.dm = DMConfig()
        if self.clustering is None:
            self.clustering = ClusteringConfig()
        if self.auto_cluster is None:
            self.auto_cluster = AutoClusterConfig()
        if self.rendering is None:
            self.rendering = RenderingConfig()


def load_config(config_path: str) -> MultiscalePipelineConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    multiscale_cfg = MultiscaleConfig(**cfg.get("multiscale", {}))
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

    return MultiscalePipelineConfig(
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", MultiscalePipelineConfig.rodent_rollout_dir),
        output_dir=cfg.get("output_dir", MultiscalePipelineConfig.output_dir),
        num_clips=cfg.get("num_clips", MultiscalePipelineConfig.num_clips),
        multiscale=multiscale_cfg,
        dm=dm_cfg,
        clustering=clustering_cfg,
        auto_cluster=auto_cluster_cfg,
        rendering=rendering_cfg,
    )


def main(config_path: str = "configs/dmwkc_experiment/multiscale.yaml"):
    cfg = load_config(config_path)

    if cfg.clustering.only_max and cfg.clustering.only_min:
        raise ValueError("Cannot use both only_max and only_min")

    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")
    os.makedirs(cfg.output_dir, exist_ok=True)

    frames_per_clip = cfg.multiscale.frames_per_clip
    print(f"=== Multiscale Pipeline (frames_per_clip={frames_per_clip}) ===")

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
            cfg.output_dir, f"dm_results_m{M}_tau{TAU}_p{cfg.dm.pers_delta}_f{frames_per_clip}"
        )
        all_cycle_dm, _ = run_dm_extraction(
            int_rodent_split,
            num_clips=num_split_clips,
            dm_cfg=cfg.dm,
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
    print("NOTE: Rendering uses split clip indices, which map to original clips.")

    # For rendering, we need to map split clip indices back to original clips
    # split_clip_idx -> (original_clip_idx, segment_idx)
    original_frames_per_clip = 499  # Standard 500 frames minus 1
    usable_frames = frames_per_clip - 1
    splits_per_original = original_frames_per_clip // usable_frames

    def map_split_to_original(split_idx: int, frame_idx: int) -> tuple[int, int]:
        """Map split clip index and frame to original clip index and frame."""
        original_clip = split_idx // splits_per_original
        segment_idx = split_idx % splits_per_original
        original_frame = segment_idx * usable_frames + frame_idx
        return original_clip, original_frame

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
                title=f"Multiscale DM Clusters (k={k}, f={frames_per_clip}){is_optimal}",
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
            title=f"Multiscale DM Clusters (k={optimal_k}, f={frames_per_clip})",
        )

    print("\n=== Pipeline complete! ===")
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Frames per clip: {frames_per_clip}")
    print(f"Total split clips: {num_split_clips}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dmwkc_experiment/multiscale.yaml"
    main(config_path)
