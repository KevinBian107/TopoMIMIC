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

from vnl_tda.utils.load_utils import get_aggregate_data, get_clip_indices
from vnl_tda.pipelines.tda.skeletonization.skeletonization import extract_skeleton_pcd_dm
from vnl_tda.pipelines.tda.skeletonization.postprocess import (
    chunk_loops_by_clustered_vpath,
    _expand_frames_nonconsec,
    _uniform_downsample,
)
from vnl_tda.pipelines.tda.skeletonization.rendering import render_grid_clusters_side_by_side


@dataclass
class DataConfig:
    dm_key: str = "intention"
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    split: str = "train"


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
class RenderingConfig:
    enabled: bool = False
    pad: int = 0
    max_samples: int = 5
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)


@dataclass
class PipelineConfig:
    output_dir: str = "out/dm_pipeline"
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

    data_cfg = DataConfig(**cfg.get("data", {}))
    takens_cfg = TakensConfig(**cfg.get("takens", {}))
    dm_cfg = DMConfig(**cfg.get("dm", {}))

    rendering_dict = cfg.get("rendering", {})
    if "target_size" in rendering_dict:
        rendering_dict["target_size"] = tuple(rendering_dict["target_size"])
    rendering_cfg = RenderingConfig(**rendering_dict)

    return PipelineConfig(
        output_dir=cfg.get("output_dir", "out/dm_pipeline"),
        data=data_cfg,
        takens=takens_cfg,
        dm=dm_cfg,
        rendering=rendering_cfg,
    )


def load_data(config: DataConfig) -> tuple[np.ndarray, list[int]]:
    clip_indices = get_clip_indices(config.split)
    work_fn = functools.partial(
        get_aggregate_data, f"/{config.dm_key}", [], path=config.rodent_rollout_dir
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        data_list = list(tqdm(pool.imap(work_fn, clip_indices), total=len(clip_indices), desc=f"Loading {config.dm_key}"))

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

    print(f"Loaded {config.dm_key}: {data.shape} (split={config.split}, clips={clip_indices[0]}-{clip_indices[-1]})")
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
) -> tuple[list, list]:
    os.makedirs(config.output_dir, exist_ok=True)

    all_cycle_dm = []
    all_G_sm = []

    for clip_id in tqdm(range(len(data)), desc="DM extraction"):
        clip = data[clip_id]
        pts = takens_embedding(clip, m=config.takens.m, tau=config.takens.tau)

        clip_output_dir = os.path.join(config.output_dir, "clips", f"clip_{clip_id}")
        os.makedirs(clip_output_dir, exist_ok=True)

        cycle_inds, G_sm = extract_skeleton_pcd_dm(
            pts,
            k=config.dm.k_neighbors,
            pers_delta=config.dm.pers_delta,
            epsilon=config.dm.epsilon,
            output_dir=clip_output_dir,
            cleanup=False,
        )

        cycle_inds = [np.array(inds, dtype=int) for inds in cycle_inds]
        all_cycle_dm.append(cycle_inds)
        all_G_sm.append(G_sm)

        print(f"Clip {clip_id}: {len(cycle_inds)} cycles, {G_sm.number_of_edges()} edges")

    with open(os.path.join(config.output_dir, "all_G_sm.pkl"), "wb") as f:
        pkl.dump(all_G_sm, f)
    with open(os.path.join(config.output_dir, "all_cycle_dm.pkl"), "wb") as f:
        pkl.dump(all_cycle_dm, f)

    print(f"\nSaved DM results to {config.output_dir}")

    return all_cycle_dm, all_G_sm


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


def run_pipeline(config: PipelineConfig) -> tuple[list, list]:
    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("Discrete Morse Pipeline")
    print("=" * 60)

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

        print(f"Loaded {len(all_cycle_dm)} cycle sets")
    else:
        print("\n=== Stage 1: Loading data ===")
        data, clip_indices = load_data(config.data)

        print("\n=== Stage 2: Running DM extraction ===")
        all_cycle_dm, all_G_sm = run_dm_extraction(data, config)

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

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)

    return all_cycle_dm, all_G_sm


def main(config_path: str = "configs/dmwkc/dm_pipeline.yaml"):
    if os.path.exists(config_path):
        cfg = load_config(config_path)
    else:
        cfg = PipelineConfig()

    run_pipeline(cfg)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dmwkc/dm_pipeline.yaml"
    main(config_path)
