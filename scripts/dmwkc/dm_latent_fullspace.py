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

import networkx as nx
import matplotlib.pyplot as plt
import subprocess
from sklearn.decomposition import PCA

from vnl_tda.utils.load_utils import get_aggregate_data, load_from_h5py
from vnl_tda.pipelines.tda.skeletonization.skeletonization import extract_skeleton_pcd_dm
from vnl_tda.utils.rendering import render_from_saved_rollout
import h5py


@dataclass
class TakensConfig:
    m: int = 3
    tau: int = 6


@dataclass
class PerClipDMConfig:
    results_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM/dm_results_m3_tau6_p0.75"
    loop_select: str = "shortest"
    target_len: int = 50
    min_len: int = 0
    max_len: int = 500


@dataclass
class PooledDMConfig:
    pers_delta: float = 0.75
    k_neighbors: int = 5
    epsilon: float = 0.99


@dataclass
class RenderingConfig:
    framerate: int = 30
    target_size: tuple[int, int] = (640, 480)
    pca_size: tuple[int, int] = (400, 400)
    show_skeleton: bool = True
    show_progression: bool = True


@dataclass
class FullspaceConfig:
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    output_dir: str = "out_fullspace"
    takens: TakensConfig = None
    per_clip_dm: PerClipDMConfig = None
    pooled_dm: PooledDMConfig = None
    rendering: RenderingConfig = None

    def __post_init__(self):
        if self.takens is None:
            self.takens = TakensConfig()
        if self.per_clip_dm is None:
            self.per_clip_dm = PerClipDMConfig()
        if self.pooled_dm is None:
            self.pooled_dm = PooledDMConfig()
        if self.rendering is None:
            self.rendering = RenderingConfig()


def load_config(config_path: str) -> FullspaceConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    takens_cfg = TakensConfig(**cfg.get("takens", {}))
    per_clip_cfg = PerClipDMConfig(**cfg.get("per_clip_dm", {}))
    pooled_cfg = PooledDMConfig(**cfg.get("pooled_dm", {}))

    rendering_dict = cfg.get("rendering", {})
    if "target_size" in rendering_dict:
        rendering_dict["target_size"] = tuple(rendering_dict["target_size"])
    if "pca_size" in rendering_dict:
        rendering_dict["pca_size"] = tuple(rendering_dict["pca_size"])
    rendering_cfg = RenderingConfig(**rendering_dict)

    return FullspaceConfig(
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", FullspaceConfig.rodent_rollout_dir),
        output_dir=cfg.get("output_dir", FullspaceConfig.output_dir),
        takens=takens_cfg,
        per_clip_dm=per_clip_cfg,
        pooled_dm=pooled_cfg,
        rendering=rendering_cfg,
    )


def takens_points(X: np.ndarray, m: int, tau: int) -> np.ndarray:
    n_frames, D = X.shape
    start = (m - 1) * tau
    if n_frames <= start:
        return np.empty((0, m * D), dtype=X.dtype)
    idxs = np.arange(start, n_frames)
    return np.stack([X[[j - a * tau for a in range(m)]].reshape(-1) for j in idxs], axis=0)


def ensure_2d(x: np.ndarray, expected_D: int | None = None) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        return x
    if x.ndim == 1:
        if expected_D is not None and x.size % expected_D == 0:
            return x.reshape(-1, expected_D)
        for D in [16, 32, 64]:
            if x.size % D == 0:
                return x.reshape(-1, D)
        raise ValueError(f"Cannot infer feature dim from 1D array of size {x.size}")
    raise ValueError(f"Unexpected array shape {x.shape}")


def select_loop(cycles: list[np.ndarray], mode: str = "shortest") -> np.ndarray | None:
    if not cycles:
        return None
    if mode == "longest":
        return max(cycles, key=len)
    elif mode == "shortest":
        return min(cycles, key=len)
    else:
        raise ValueError(f"Unknown loop selection mode: {mode}")


def extract_loop_intentions(
    int_clip: np.ndarray,
    loop_indices: np.ndarray,
    m: int,
    tau: int,
) -> np.ndarray:
    start_offset = (m - 1) * tau
    original_frame_indices = loop_indices + start_offset
    original_frame_indices = np.clip(original_frame_indices, 0, len(int_clip) - 1)
    return int_clip[original_frame_indices]


def uniform_downsample(indices: np.ndarray, target_len: int) -> np.ndarray | None:
    if len(indices) < target_len:
        return None
    step = len(indices) / target_len
    sampled_idx = [int(i * step) for i in range(target_len)]
    return indices[sampled_idx]


def load_dm_results(
    results_dir: str,
    loop_select: str,
    target_len: int,
    min_len: int,
    max_len: int,
) -> tuple[list[int], list[np.ndarray]]:
    cycle_path = os.path.join(results_dir, "all_cycle_dm.pkl")
    with open(cycle_path, "rb") as f:
        all_cycles = pkl.load(f)

    matched_clips = []
    selected_loops = []

    for clip_id, cycles in enumerate(all_cycles):
        cycles = [np.asarray(c, dtype=int) for c in cycles]
        cycles = [c for c in cycles if min_len <= len(c) <= max_len]

        if not cycles:
            continue

        selected = select_loop(cycles, mode=loop_select)
        if selected is None:
            continue

        sampled = uniform_downsample(selected, target_len)
        if sampled is None:
            continue

        matched_clips.append(clip_id)
        selected_loops.append(sampled)

    print(f"\nFirst 3 clips - {loop_select} loop (target_len={target_len}):")
    for i in range(min(3, len(matched_clips))):
        clip_id = matched_clips[i]
        loop = selected_loops[i]
        print(f"  Clip {clip_id}: {len(loop)} points (downsampled to target_len)")

    return matched_clips, selected_loops


def render_skeleton_with_pca(
    pooled_int: np.ndarray,
    pooled_G: nx.Graph,
    clip_loop_metadata: list[dict],
    rodent_rollout_dir: str,
    matched_clips: list[int],
    output_path: str,
    rendering_cfg: RenderingConfig,
    m: int,
    tau: int,
) -> None:
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(pooled_int)

    x_min, x_max = pca_coords[:, 0].min(), pca_coords[:, 0].max()
    y_min, y_max = pca_coords[:, 1].min(), pca_coords[:, 1].max()
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1

    edges = list(pooled_G.edges())

    clip_frames_cache = {}
    for meta in clip_loop_metadata:
        clip_idx = meta["clip_idx"]
        if clip_idx not in clip_frames_cache:
            with h5py.File(f"{rodent_rollout_dir}/clip_{clip_idx}.h5", "r") as h5file:
                rollout = load_from_h5py(h5file)
            clip_frames_cache[clip_idx] = render_from_saved_rollout(rollout)

    frame_to_clip_and_local = {}
    for meta in clip_loop_metadata:
        clip_idx = meta["clip_idx"]
        start_idx = meta["start_idx_in_pool"]
        loop_len = meta["loop_length"]
        for i in range(loop_len):
            pool_frame_idx = start_idx + i
            local_frame_idx = i + (m - 1) * tau
            frame_to_clip_and_local[pool_frame_idx] = (clip_idx, local_frame_idx)

    temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    n_frames = len(pooled_int)
    pca_w, pca_h = rendering_cfg.pca_size
    mj_w, mj_h = rendering_cfg.target_size

    print(f"Rendering {n_frames} frames...")
    for frame_idx in tqdm(range(n_frames), desc="Rendering frames"):
        fig, (ax_pca, ax_mj) = plt.subplots(1, 2, figsize=(12, 5))

        ax_pca.scatter(pca_coords[:, 0], pca_coords[:, 1], c='lightgray', s=5, alpha=0.5)

        if rendering_cfg.show_skeleton:
            for u, v in edges:
                if u < len(pca_coords) and v < len(pca_coords):
                    ax_pca.plot(
                        [pca_coords[u, 0], pca_coords[v, 0]],
                        [pca_coords[u, 1], pca_coords[v, 1]],
                        'b-', linewidth=1.5, alpha=0.7
                    )

        if rendering_cfg.show_progression:
            ax_pca.scatter(
                pca_coords[:frame_idx + 1, 0],
                pca_coords[:frame_idx + 1, 1],
                c=np.arange(frame_idx + 1),
                cmap='viridis',
                s=20,
                alpha=0.8
            )

        ax_pca.scatter(
            pca_coords[frame_idx, 0],
            pca_coords[frame_idx, 1],
            c='red', s=100, marker='o', edgecolors='black', linewidths=2, zorder=10
        )

        ax_pca.set_xlim(x_min - x_pad, x_max + x_pad)
        ax_pca.set_ylim(y_min - y_pad, y_max + y_pad)
        ax_pca.set_xlabel("PC1")
        ax_pca.set_ylabel("PC2")
        ax_pca.set_title(f"PCA Projection (Frame {frame_idx}/{n_frames})")
        ax_pca.set_aspect('equal')

        if frame_idx in frame_to_clip_and_local:
            clip_idx, local_idx = frame_to_clip_and_local[frame_idx]
            mj_frames = clip_frames_cache[clip_idx]
            if local_idx < len(mj_frames):
                mj_frame = mj_frames[local_idx]
                ax_mj.imshow(mj_frame)
                ax_mj.set_title(f"Clip {clip_idx}, Frame {local_idx}")
            else:
                ax_mj.text(0.5, 0.5, "Frame out of range", ha='center', va='center', transform=ax_mj.transAxes)
                ax_mj.set_title(f"Clip {clip_idx}")
        else:
            ax_mj.text(0.5, 0.5, "No mapping", ha='center', va='center', transform=ax_mj.transAxes)
            ax_mj.set_title("MuJoCo Rendering")

        ax_mj.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(temp_dir, f"frame_{frame_idx:06d}.png"), dpi=100)
        plt.close(fig)

    print("Encoding video...")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(rendering_cfg.framerate),
        "-i", os.path.join(temp_dir, "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)

    print(f"Video saved to: {output_path}")


def main(config_path: str = "configs/dmwkc/dm_latent_fullspace.yaml"):
    cfg = load_config(config_path)

    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"=== Stage 1: Loading DM results from {cfg.per_clip_dm.results_dir} ===")
    matched_clips, selected_loops = load_dm_results(
        cfg.per_clip_dm.results_dir,
        cfg.per_clip_dm.loop_select,
        cfg.per_clip_dm.target_len,
        cfg.per_clip_dm.min_len,
        cfg.per_clip_dm.max_len,
    )
    print(f"Loaded {len(matched_clips)} clips with loops (target_len={cfg.per_clip_dm.target_len})")

    print("Loading intention data for matched clips...")
    work_rodent = functools.partial(get_aggregate_data, "/intention", [], path=cfg.rodent_rollout_dir)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        int_list = list(tqdm(pool.imap(work_rodent, matched_clips), total=len(matched_clips)))
    int_per_clip = [ensure_2d(x) for x in int_list]

    print("=== Stage 2: Pooling loop intentions ===")
    loop_lengths = [len(l) for l in selected_loops if l is not None]
    print(f"Loop length stats: mean={np.mean(loop_lengths):.2f}, std={np.std(loop_lengths):.2f}")
    pooled_intentions = []
    clip_loop_metadata = []

    for clip_idx, int_clip, loop_inds in zip(matched_clips, int_per_clip, selected_loops):
        if loop_inds is None or len(loop_inds) == 0:
            continue
        loop_int = extract_loop_intentions(int_clip, loop_inds, m=cfg.takens.m, tau=cfg.takens.tau)
        pooled_intentions.append(loop_int)
        clip_loop_metadata.append({
            "clip_idx": clip_idx,
            "loop_length": len(loop_int),
            "start_idx_in_pool": sum(len(p) for p in pooled_intentions[:-1]),
        })

    if not pooled_intentions:
        raise RuntimeError("No loops extracted. Cannot proceed.")

    pooled_int = np.vstack(pooled_intentions)
    print(f"Pooled {len(pooled_intentions)} loops, {len(pooled_int)} total frames")

    print("=== Stage 3: Takens embedding on pooled data ===")
    pooled_pts = takens_points(pooled_int, m=cfg.takens.m, tau=cfg.takens.tau)
    print(f"Pooled Takens points: {len(pooled_pts)}")

    print("=== Stage 4: Pooled DM graph ===")
    pooled_output_dir = os.path.join(cfg.output_dir, "pooled")
    os.makedirs(pooled_output_dir, exist_ok=True)

    pooled_cycles, pooled_G = extract_skeleton_pcd_dm(
        pooled_pts,
        k=cfg.pooled_dm.k_neighbors,
        pers_delta=cfg.pooled_dm.pers_delta,
        epsilon=cfg.pooled_dm.epsilon,
        output_dir=pooled_output_dir,
        cleanup=False,
    )
    pooled_cycles = [np.array(inds, dtype=int) for inds in pooled_cycles]
    print(f"Pooled DM: {len(pooled_cycles)} cycles, {pooled_G.number_of_edges()} edges")

    print("=== Stage 5: Saving results ===")
    with open(os.path.join(cfg.output_dir, "pooled_intentions.pkl"), "wb") as f:
        pkl.dump(pooled_int, f)
    with open(os.path.join(cfg.output_dir, "pooled_cycles.pkl"), "wb") as f:
        pkl.dump(pooled_cycles, f)
    with open(os.path.join(cfg.output_dir, "pooled_G.pkl"), "wb") as f:
        pkl.dump(pooled_G, f)
    with open(os.path.join(cfg.output_dir, "clip_loop_metadata.pkl"), "wb") as f:
        pkl.dump(clip_loop_metadata, f)
    with open(os.path.join(cfg.output_dir, "matched_clips.pkl"), "wb") as f:
        pkl.dump(matched_clips, f)

    print("=== Stage 6: Rendering video ===")
    output_video = os.path.join(cfg.output_dir, "skeleton_walkthrough.mp4")
    render_skeleton_with_pca(
        pooled_int=pooled_int,
        pooled_G=pooled_G,
        clip_loop_metadata=clip_loop_metadata,
        rodent_rollout_dir=cfg.rodent_rollout_dir,
        matched_clips=matched_clips,
        output_path=output_video,
        rendering_cfg=cfg.rendering,
        m=cfg.takens.m,
        tau=cfg.takens.tau,
    )

    print("=== Pipeline complete! ===")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dmwkc/dm_latent_fullspace.yaml"
    main(config_path)
