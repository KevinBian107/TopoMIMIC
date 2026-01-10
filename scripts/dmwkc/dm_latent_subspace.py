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

from vnl_tda.utils.load_utils import get_aggregate_data
from vnl_tda.pipelines.tda.skeletonization.skeletonization import extract_skeleton_pcd_dm
from vnl_tda.pipelines.tda.skeletonization.postprocess import (
    cluster_vpath_with_wasserstein2d_fixedlen_kmeans,
    chunk_loops_by_clustered_vpath,
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
    n_clusters: int = 10
    min_len: int = 0
    max_len: int = 500
    target_len: int = 50
    only_max: bool = True
    only_min: bool = False
    skip_clips: tuple[int, ...] = (37, 152, 163, 274, 281)


@dataclass
class RenderingConfig:
    pad: int = 0
    max_samples: int = 5
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)


@dataclass
class PipelineConfig:
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    output_dir: str = "out_videos"
    num_clips: int = 500
    takens: TakensConfig = None
    dm: DMConfig = None
    clustering: ClusteringConfig = None
    rendering: RenderingConfig = None

    def __post_init__(self):
        if self.takens is None:
            self.takens = TakensConfig()
        if self.dm is None:
            self.dm = DMConfig()
        if self.clustering is None:
            self.clustering = ClusteringConfig()
        if self.rendering is None:
            self.rendering = RenderingConfig()


def load_config(config_path: str) -> PipelineConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    takens_cfg = TakensConfig(**cfg.get("takens", {}))
    dm_cfg = DMConfig(**cfg.get("dm", {}))

    clustering_dict = cfg.get("clustering", {})
    if "skip_clips" in clustering_dict:
        clustering_dict["skip_clips"] = tuple(clustering_dict["skip_clips"])
    clustering_cfg = ClusteringConfig(**clustering_dict)

    rendering_dict = cfg.get("rendering", {})
    if "target_size" in rendering_dict:
        rendering_dict["target_size"] = tuple(rendering_dict["target_size"])
    rendering_cfg = RenderingConfig(**rendering_dict)

    return PipelineConfig(
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", PipelineConfig.rodent_rollout_dir),
        output_dir=cfg.get("output_dir", PipelineConfig.output_dir),
        num_clips=cfg.get("num_clips", PipelineConfig.num_clips),
        takens=takens_cfg,
        dm=dm_cfg,
        clustering=clustering_cfg,
        rendering=rendering_cfg,
    )


def takens_points(X: np.ndarray, m: int, tau: int) -> np.ndarray:
    n_frames, D = X.shape
    start = (m - 1) * tau
    idxs = np.arange(start, n_frames)
    pts = np.stack([X[[j - a * tau for a in range(m)]].reshape(-1) for j in idxs], axis=0)
    return pts


def run_dm_extraction(
    int_rodent: np.ndarray,
    num_clips: int,
    dm_cfg: DMConfig,
    takens_cfg: TakensConfig,
    base_output_dir: str,
) -> tuple[list, list]:
    all_cycle_dm = []
    all_G_sm = []

    os.makedirs(base_output_dir, exist_ok=True)

    for clip_id in tqdm(range(num_clips), desc="DM extraction"):
        clip = int_rodent[clip_id]
        pts = takens_points(clip, m=takens_cfg.m, tau=takens_cfg.tau)
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
    int_rodent: np.ndarray,
    clustering_cfg: ClusteringConfig,
    n_clips: int,
    output_dir: str,
) -> tuple[dict, list]:
    cluster_dict, cleaned_loops = cluster_vpath_with_wasserstein2d_fixedlen_kmeans(
        all_cycle_dm,
        int_rodent,
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


def run_rendering(
    cluster_dict: dict,
    rodent_rollout_dir: str,
    rendering_cfg: RenderingConfig,
    output_dir: str,
    cluster_labels: dict[int, str] | None = None,
) -> None:
    cluster_chunks = chunk_loops_by_clustered_vpath(
        cluster_dict,
        rodent_rollout_dir,
        pad=rendering_cfg.pad,
        sample=rendering_cfg.max_samples,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "cluster_side_by_side.mp4")

    render_grid_clusters_side_by_side(
        cluster_chunks=cluster_chunks,
        out_path=out_path,
        framerate=rendering_cfg.framerate,
        max_samples_per_cluster=rendering_cfg.max_samples,
        target_size=rendering_cfg.target_size,
        title="DM Cluster Samples",
        per_tile_labels=True,
        selected_clusters=sorted(cluster_dict.keys()),
        cluster_labels=cluster_labels,
    )
    print(f"Video saved to: {out_path}")


def main(config_path: str = "configs/dmwkc/dm_latent_subspace.yaml"):
    cfg = load_config(config_path)

    if cfg.clustering.only_max and cfg.clustering.only_min:
        raise ValueError("Cannot use both only_max and only_min")

    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")

    print("Loading intention data...")
    work_rodent = functools.partial(
        get_aggregate_data, "/intention", [], path=cfg.rodent_rollout_dir
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        int_rodent = list(tqdm(pool.imap(work_rodent, range(cfg.num_clips)), total=cfg.num_clips))
    int_rodent = np.vstack(int_rodent)
    int_rodent = int_rodent.reshape([cfg.num_clips, 499, 16])
    print(f"Loaded intention: {int_rodent.shape}")

    if cfg.dm.skip:
        if cfg.dm.results_dir is None:
            raise ValueError("dm.results_dir required when dm.skip is true")
        print(f"Loading existing DM results from {cfg.dm.results_dir}")
        with open(os.path.join(cfg.dm.results_dir, "all_cycle_dm.pkl"), "rb") as f:
            all_cycle_dm = pkl.load(f)
        print(f"Loaded {len(all_cycle_dm)} cycle sets")
    else:
        base_output_dir = os.path.join(
            cfg.output_dir, f"dm_results_m{cfg.takens.m}_tau{cfg.takens.tau}_p{cfg.dm.pers_delta}"
        )
        all_cycle_dm, _ = run_dm_extraction(
            int_rodent,
            num_clips=cfg.num_clips,
            dm_cfg=cfg.dm,
            takens_cfg=cfg.takens,
            base_output_dir=base_output_dir,
        )

    print("Running clustering...")
    cluster_output_dir = os.path.join(cfg.output_dir, "clustering")
    cluster_dict, cleaned_loops = run_clustering(
        all_cycle_dm,
        int_rodent,
        clustering_cfg=cfg.clustering,
        n_clips=cfg.num_clips,
        output_dir=cluster_output_dir,
    )
    print(f"Created {len(cluster_dict)} clusters")

    print("Rendering cluster videos...")
    cluster_labels = {i: f"cluster_{i}" for i in cluster_dict.keys()}
    run_rendering(
        cluster_dict,
        rodent_rollout_dir=cfg.rodent_rollout_dir,
        rendering_cfg=cfg.rendering,
        output_dir=cfg.output_dir,
        cluster_labels=cluster_labels,
    )

    print("Pipeline complete!")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dmwkc/dm_latent_subspace.yaml"
    main(config_path)
