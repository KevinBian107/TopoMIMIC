"""Main entry point for granularity analysis pipeline."""

import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["JAX_ENABLE_X64"] = "True"

import functools
import multiprocessing as mp
import pickle as pkl
import yaml

import numpy as np
from tqdm import tqdm

from .config import GranularityConfig, load_config
from .dm_multi_persistence import run_dm_multi_persistence
from .stationary_analysis import run_stationary_analysis
from .redundancy import run_redundancy_analysis
from .plotting import generate_stationary_plots, generate_redundancy_plots
from .rendering import render_all_similar_pairs, render_stationary_analysis, render_similar_pairs_grid


def load_data(
    data_key: str,
    rodent_rollout_dir: str,
    split: str,
    n_clips: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    """
    Load data from HDF5 files.

    Args:
        data_key: Key to load (e.g., "intention", "qpos")
        rodent_rollout_dir: Path to data directory or consolidated H5 file
        split: Data split ("train", "test", "all")
        n_clips: Number of clips to load (None for all)

    Returns:
        Tuple of (data_array, clip_indices)
    """
    from topo_mimic.dmwk.utils.load_utils import get_aggregate_data, get_clip_indices

    clip_indices = get_clip_indices(split, n_clips=n_clips)
    work_fn = functools.partial(
        get_aggregate_data, f"/{data_key}", [], path=rodent_rollout_dir
    )

    with mp.Pool(processes=mp.cpu_count()) as pool:
        data_list = list(tqdm(
            pool.imap(work_fn, clip_indices),
            total=len(clip_indices),
            desc=f"Loading {data_key}"
        ))

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

    print(f"Loaded {data_key}: {data.shape} (split={split}, clips={clip_indices[0]}-{clip_indices[-1]})")
    return data, clip_indices


def save_summary(
    config: GranularityConfig,
    dm_results: dict | None,
    stationary_output,
    redundancy_output,
) -> None:
    """Save analysis summary to YAML file."""
    output_dir = Path(config.output_dir)
    summary = {
        "config": {
            "data": {
                "dm_key": config.data.dm_key,
                "qpos_key": config.data.qpos_key,
                "rodent_rollout_dir": config.data.rodent_rollout_dir,
                "split": config.data.split,
            },
            "multi_persistence": {
                "enabled": config.multi_persistence.enabled,
                "pers_delta_values": config.multi_persistence.pers_delta_values,
            },
            "stationary": {
                "enabled": config.stationary.enabled,
                "root_displacement_threshold": config.stationary.root_displacement_threshold,
            },
            "redundancy": {
                "enabled": config.redundancy.enabled,
                "n_clusters_range": config.redundancy.n_clusters_range,
                "dtw_threshold": config.redundancy.dtw_threshold,
                "dtw_percentile": config.redundancy.dtw_percentile,
            },
        },
        "results": {},
    }

    if dm_results is not None:
        summary["results"]["dm_extraction"] = {
            pers: {
                "n_total_cycles": r.cycle_stats["n_total_cycles"],
                "n_clips_with_cycles": r.cycle_stats["n_clips_with_cycles"],
                "mean_cycle_length": r.cycle_stats["mean_cycle_length"],
            }
            for pers, r in dm_results.items()
        }

    if stationary_output is not None:
        summary["results"]["stationary"] = {
            "n_stationary_clips": len(stationary_output.stationary_clips),
        }

    if redundancy_output is not None:
        summary["results"]["redundancy"] = {
            "n_configurations": len(redundancy_output.results),
        }

    with open(output_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    print(f"Saved summary to {output_dir / 'summary.yaml'}")


def run_granularity_analysis(config: GranularityConfig) -> None:
    """
    Run the full granularity analysis pipeline.

    Args:
        config: Configuration object
    """
    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")

    original_cwd = os.getcwd()

    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(original_cwd) / output_dir
    config.output_dir = str(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Granularity Analysis Pipeline")
    print("=" * 60)
    print(f"Output directory: {output_dir}")

    print(f"\n=== Stage 1: Loading {config.data.dm_key} data ===")
    intention_data, clip_indices = load_data(
        config.data.dm_key,
        config.data.rodent_rollout_dir,
        config.data.split,
    )

    print(f"\n=== Stage 2: Loading {config.data.qpos_key} data ===")
    qpos_data, _ = load_data(
        config.data.qpos_key,
        config.data.rodent_rollout_dir,
        config.data.split,
    )

    min_frames = min(intention_data.shape[1], qpos_data.shape[1])
    intention_data = intention_data[:, :min_frames, :]
    qpos_data = qpos_data[:, :min_frames, :]
    print(f"Aligned data shapes: intention={intention_data.shape}, qpos={qpos_data.shape}")

    dm_results = None
    if config.multi_persistence.enabled:
        print("\n=== Stage 3: Multi-Persistence DM Extraction ===")
        dm_results = run_dm_multi_persistence(intention_data, config, clip_indices)
    else:
        print("\n=== Stage 3: Skipping DM Extraction (disabled) ===")

    stationary_output = None
    if config.stationary.enabled and dm_results is not None:
        print("\n=== Stage 4: Stationary Clip Analysis ===")
        stationary_output = run_stationary_analysis(
            qpos_data, dm_results, config, clip_indices
        )
        stationary_plots = generate_stationary_plots(stationary_output, config)
        print(f"  Generated {len(stationary_plots)} stationary plots")

        if config.rendering.enabled:
            stationary_videos = render_stationary_analysis(
                stationary_output,
                dm_results,
                clip_indices,
                config.data.rodent_rollout_dir,
                Path(config.output_dir) / "stationary",
                config,
            )
            print(f"  Rendered {len(stationary_videos)} stationary videos")
    else:
        print("\n=== Stage 4: Skipping Stationary Analysis (disabled or no DM results) ===")

    redundancy_output = None
    if config.redundancy.enabled and dm_results is not None:
        print("\n=== Stage 5: Redundancy Analysis ===")
        redundancy_output = run_redundancy_analysis(
            dm_results, intention_data, qpos_data, config, clip_indices
        )
        redundancy_plots = generate_redundancy_plots(redundancy_output, config)
        print(f"  Generated {len(redundancy_plots)} redundancy plots")
    else:
        print("\n=== Stage 5: Skipping Redundancy Analysis (disabled or no DM results) ===")

    rendered_videos = []
    if config.rendering.enabled and redundancy_output is not None:
        print("\n=== Stage 6: Rendering Similar Pairs ===")

        grid_video_path = str(Path(config.output_dir) / "redundancy" / "similar_pairs_grid.mp4")
        try:
            print("  Rendering similar pairs grid (persistence × cluster count)...")
            render_similar_pairs_grid(
                redundancy_output.results,
                config.data.rodent_rollout_dir,
                grid_video_path,
                config,
            )
            rendered_videos.append(grid_video_path)
            print(f"  Rendered: {grid_video_path}")
        except Exception as e:
            print(f"  Error rendering similar pairs grid: {e}")

        for (pers_delta, n_clusters), result in redundancy_output.results.items():
            if result.top_similar_pairs:
                print(f"\n  Rendering individual pairs for pers_delta={pers_delta}, k={n_clusters}")
                videos_dir = Path(config.output_dir) / "redundancy" / f"pers_{pers_delta:.2f}" / f"n_clusters_{n_clusters}"
                videos = render_all_similar_pairs(
                    result,
                    result.cluster_dict,
                    config.data.rodent_rollout_dir,
                    videos_dir,
                    config,
                )
                rendered_videos.extend(videos)
        print(f"\n  Total videos rendered: {len(rendered_videos)}")
    else:
        print("\n=== Stage 6: Skipping Rendering (disabled or no redundancy results) ===")

    print("\n=== Stage 7: Saving Summary ===")
    save_summary(config, dm_results, stationary_output, redundancy_output)

    print("\n" + "=" * 60)
    print("Granularity Analysis Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


def main(config_path: str | None = None) -> None:
    """
    Main entry point.

    Args:
        config_path: Path to config YAML file (uses default if None)
    """
    if config_path is None:
        config_path = "configs/granularity/granularity.yaml"

    if os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        config = load_config(config_path)
    else:
        print(f"Config not found at {config_path}, using defaults")
        config = GranularityConfig()

    run_granularity_analysis(config)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
