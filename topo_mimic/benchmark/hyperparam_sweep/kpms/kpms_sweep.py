"""Main entry point for KPMS hyperparameter sweep with WandB integration."""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["JAX_ENABLE_X64"] = "True"

import argparse
import functools
import multiprocessing as mp
import pickle as pkl
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from jax_moseq.models.keypoint_slds import estimate_coordinates
    HAS_JAXMOSEQ = True
except ImportError:
    HAS_JAXMOSEQ = False

from topo_mimic.dmwk.utils.load_utils import get_aggregate_data, get_clip_indices
from topo_mimic.dmwk.utils.rendering import render_from_saved_rollout
from topo_mimic.dmwk.utils.load_utils import load_rollout_from_source

from topo_mimic.benchmark.hyperparam_sweep.config import (
    SweepConfig,
    load_config,
)
from topo_mimic.benchmark.hyperparam_sweep.kpms.config import (
    KPMSHyperparams,
    hyperparams_to_dict,
)
from topo_mimic.benchmark.hyperparam_sweep.keypoint_loader import (
    setup_stac_reference,
    qpos_to_keypoints,
)
from topo_mimic.benchmark.hyperparam_sweep.kpms.dual_fit import (
    run_dual_fit,
    run_ensemble_dual_fit,
)
from topo_mimic.benchmark.hyperparam_sweep.metrics import (
    compute_duration_stats,
    compute_active_syllables,
    compute_transition_entropy,
)
from topo_mimic.benchmark.hyperparam_sweep.metrics import collect_all_metrics
from topo_mimic.benchmark.hyperparam_sweep.wandb_logger import (
    WandBLogger,
    log_metrics_as_bar_charts,
)
from topo_mimic.benchmark.hyperparam_sweep.visualization import (
    create_all_plots,
    render_dual_syllable_video,
    render_keypoint_3d_video,
)


def load_data(
    data_key: str,
    rodent_rollout_dir: str,
    clip_range: tuple[int, int],
) -> tuple[np.ndarray, list[int]]:
    start, end = clip_range
    clip_indices = list(range(start, end))

    work_fn = functools.partial(
        get_aggregate_data, f"/{data_key}", [], path=rodent_rollout_dir
    )

    with mp.Pool(processes=mp.cpu_count()) as pool:
        data_list = list(
            tqdm(pool.imap(work_fn, clip_indices), total=len(clip_indices), desc=f"Loading {data_key}")
        )

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

    print(f"Loaded {data_key}: {data.shape} (clips {start}-{end})")
    return data, clip_indices


def load_qpos_and_convert_keypoints(
    rodent_rollout_dir: str,
    clip_range: tuple[int, int],
    stac_xml_path: str,
    stac_data_path: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    qpos_data, clip_indices = load_data("qpos", rodent_rollout_dir, clip_range)

    print("Setting up mujoco model for keypoint conversion...")
    mjx_model, mjx_data, site_ids, kp_names = setup_stac_reference(stac_xml_path, stac_data_path)
    print(f"Found {len(site_ids)} keypoint sites: {kp_names[:5]}..." if len(kp_names) > 5 else f"Found keypoint sites: {kp_names}")

    print("Converting qpos to keypoints...")
    keypoint_data = qpos_to_keypoints(qpos_data, mjx_model, mjx_data, site_ids)
    print(f"Keypoints shape: {keypoint_data.shape}")

    return qpos_data, keypoint_data, kp_names


def render_sample_videos(
    config: SweepConfig,
    int_results: dict,
    kp_results: dict,
    rodent_rollout_dir: str,
    clip_indices: list[int],
    output_dir: str,
    n_samples: int = 5,
) -> dict[str, str]:
    video_paths = {}

    rec_names = sorted(int_results.keys())[:n_samples]

    for i, rec_name in enumerate(rec_names):
        idx_in_batch = int(rec_name.split("_")[1])
        clip_id = clip_indices[idx_in_batch]

        int_syllables = np.array(int_results[rec_name]["syllable"])
        kp_syllables = np.array(kp_results[rec_name]["syllable"])

        try:
            ro = load_rollout_from_source(clip_id, rodent_rollout_dir)
            frames = render_from_saved_rollout(ro, camera_name=config.rendering.camera_name)
        except Exception as e:
            print(f"Warning: Could not render clip {clip_id}: {e}")
            continue

        video_name = f"dual_syllable_clip{clip_id}"
        video_path = f"{output_dir}/videos/{video_name}.mp4"

        try:
            render_dual_syllable_video(
                frames=frames,
                intention_syllables=int_syllables,
                keypoint_syllables=kp_syllables,
                output_path=video_path,
                fps=config.rendering.fps,
            )
            video_paths[video_name] = video_path
            print(f"Rendered video: {video_path}")
        except Exception as e:
            print(f"Warning: Could not render video for clip {clip_id}: {e}")

    return video_paths


def single_run(
    config: SweepConfig,
    hyperparams: KPMSHyperparams,
    seed: int = 0,
    split: str = "test",
) -> dict[str, Any]:
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/videos", exist_ok=True)

    if split == "test":
        clip_range = config.data.test_clips
    else:
        clip_range = config.data.train_clips

    print("=" * 60)
    print(f"KPMS Hyperparameter Sweep - Single Run")
    print("=" * 60)
    print(f"Hyperparameters: {hyperparams_to_dict(hyperparams)}")
    print(f"Clip range: {clip_range}")
    print(f"Seed: {seed}")

    print("\n=== Loading data ===")
    intention_data, clip_indices = load_data("intention", config.data.rodent_rollout_dir, clip_range)

    qpos_data, keypoint_data, kp_names = load_qpos_and_convert_keypoints(
        rodent_rollout_dir=config.data.rodent_rollout_dir,
        clip_range=clip_range,
        stac_xml_path=config.keypoint.xml_path,
        stac_data_path=config.keypoint.stac_data_path,
    )

    print("\n=== Running dual fit ===")
    dual_result = run_dual_fit(
        intention_data=intention_data,
        keypoint_data=keypoint_data,
        hyperparams=hyperparams,
        output_dir=output_dir,
        seed=seed,
        kp_names=kp_names,
        compute_eml_flag=True,
    )

    print("\n=== Computing metrics ===")
    metrics = collect_all_metrics(
        int_model=dual_result.intention_result.model,
        kp_model=dual_result.keypoint_result.model,
        int_results=dual_result.intention_result.results,
        kp_results=dual_result.keypoint_result.results,
        int_data=dual_result.intention_result.data,
        kp_data=dual_result.keypoint_result.data,
        int_metadata=dual_result.intention_result.metadata,
        kp_metadata=dual_result.keypoint_result.metadata,
        num_states=hyperparams.num_states,
    )

    metrics["dual_fit/syllable_ari"] = dual_result.syllable_ari
    metrics["dual_fit/syllable_nmi"] = dual_result.syllable_nmi
    metrics["dual_fit/duration_kl"] = dual_result.duration_kl

    if dual_result.eml_intention is not None:
        metrics["eml_intention"] = dual_result.eml_intention.mean_score
        metrics["eml_intention_std"] = dual_result.eml_intention.mean_std_err

    if dual_result.eml_keypoint is not None:
        metrics["eml_keypoint"] = dual_result.eml_keypoint.mean_score
        metrics["eml_keypoint_std"] = dual_result.eml_keypoint.mean_std_err

    print("\n=== Computing reconstructions ===")
    int_original = None
    int_reconstructed = None
    kp_original = None
    kp_reconstructed = None

    def stack_data_array(data_dict, key):
        """Stack data from dict format to array format."""
        val = data_dict.get(key)
        if val is None:
            return None
        if isinstance(val, dict):
            keys = sorted(val.keys())
            return np.stack([np.array(val[k]) for k in keys], axis=0)
        else:
            return np.array(val)

    try:
        import jax.numpy as jnp
        from jax_moseq.utils import unbatch

        int_model = dual_result.intention_result.model
        int_metadata = dual_result.intention_result.metadata
        int_data = dual_result.intention_result.data

        int_original = stack_data_array(int_data, "Y")
        if int_original is not None:
            int_original = int_original.squeeze(-1) if int_original.ndim > 3 else int_original

        if 'states' in int_model and 'params' in int_model:
            Y_est_int = estimate_coordinates(
                jnp.array(int_model['states']['x']),
                jnp.array(int_model['states']['v']),
                jnp.array(int_model['states']['h']),
                jnp.array(int_model['params']['Cd'])
            )
            coords_est_int = unbatch(Y_est_int, *int_metadata)
            rec_keys = sorted(coords_est_int.keys())
            int_reconstructed = np.stack([np.array(coords_est_int[k]).squeeze(-1) for k in rec_keys], axis=0)
            print(f"Intention reconstruction: {int_original.shape} -> {int_reconstructed.shape}")
        else:
            print(f"Intention model missing states/params. Keys: {list(int_model.keys())}")

        kp_model = dual_result.keypoint_result.model
        kp_metadata = dual_result.keypoint_result.metadata
        kp_data = dual_result.keypoint_result.data

        kp_original = stack_data_array(kp_data, "Y")
        if kp_original is not None:
            if kp_original.ndim == 4:
                n_clips, n_frames, n_kp, n_dim = kp_original.shape
                kp_original = kp_original.reshape(n_clips, n_frames, n_kp * n_dim)

        if 'states' in kp_model and 'params' in kp_model:
            Y_est_kp = estimate_coordinates(
                jnp.array(kp_model['states']['x']),
                jnp.array(kp_model['states']['v']),
                jnp.array(kp_model['states']['h']),
                jnp.array(kp_model['params']['Cd'])
            )
            coords_est_kp = unbatch(Y_est_kp, *kp_metadata)
            rec_keys = sorted(coords_est_kp.keys())
            kp_recon_list = []
            for k in rec_keys:
                arr = np.array(coords_est_kp[k])
                if arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = arr.squeeze(-1)
                elif arr.ndim == 4:
                    arr = arr.reshape(arr.shape[0], -1)
                kp_recon_list.append(arr)
            kp_reconstructed = np.stack(kp_recon_list, axis=0)
            if kp_reconstructed.ndim == 4:
                n_clips, n_frames, n_kp, n_dim = kp_reconstructed.shape
                kp_reconstructed = kp_reconstructed.reshape(n_clips, n_frames, n_kp * n_dim)
            print(f"Keypoint reconstruction: {kp_original.shape} -> {kp_reconstructed.shape}")
        else:
            print(f"Keypoint model missing states/params. Keys: {list(kp_model.keys())}")

        if int_original is not None and int_reconstructed is not None:
            min_len = min(int_original.shape[1], int_reconstructed.shape[1])
            int_recon_mse = np.mean((int_original[:, :min_len] - int_reconstructed[:, :min_len]) ** 2)
            metrics["intention_fit/reconstruction_mse"] = float(int_recon_mse)
            print(f"Intention Reconstruction MSE: {int_recon_mse:.6f}")

        if kp_original is not None and kp_reconstructed is not None:
            min_len = min(kp_original.shape[1], kp_reconstructed.shape[1])
            kp_recon_mse = np.mean((kp_original[:, :min_len] - kp_reconstructed[:, :min_len]) ** 2)
            metrics["keypoint_fit/reconstruction_mse"] = float(kp_recon_mse)
            print(f"Keypoint Reconstruction MSE: {kp_recon_mse:.6f}")

    except Exception as e:
        print(f"Warning: Reconstruction computation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Generating plots ===")
    plots = create_all_plots(
        int_model=dual_result.intention_result.model,
        kp_model=dual_result.keypoint_result.model,
        int_results=dual_result.intention_result.results,
        kp_results=dual_result.keypoint_result.results,
        int_original=int_original,
        int_reconstructed=int_reconstructed,
        kp_original=kp_original,
        kp_reconstructed=kp_reconstructed,
        n_recon_dims=5,
    )

    print("\n=== Rendering sample videos ===")
    if config.rendering.enabled:
        video_paths = render_sample_videos(
            config=config,
            int_results=dual_result.intention_result.results,
            kp_results=dual_result.keypoint_result.results,
            rodent_rollout_dir=config.data.rodent_rollout_dir,
            clip_indices=clip_indices,
            output_dir=output_dir,
            n_samples=config.wandb.n_video_samples,
        )

        print("\n=== Rendering 3D keypoint verification video ===")
        try:
            kp_3d_video_path = f"{output_dir}/videos/keypoint_3d_verification.mp4"
            render_keypoint_3d_video(
                keypoint_data=keypoint_data,
                output_path=kp_3d_video_path,
                kp_names=kp_names,
                clip_idx=0,
                fps=config.rendering.fps,
                title="Keypoint 3D Trajectory (Verification)",
            )
            video_paths["keypoint_3d_verification"] = kp_3d_video_path
        except Exception as e:
            print(f"Warning: 3D keypoint video rendering failed: {e}")
    else:
        video_paths = {}

    print("\n=== Saving results ===")
    results_path = f"{output_dir}/results.pkl"
    with open(results_path, "wb") as f:
        pkl.dump({
            "metrics": metrics,
            "hyperparams": hyperparams_to_dict(hyperparams),
            "seed": seed,
            "clip_range": clip_range,
        }, f)
    print(f"Saved results to {results_path}")

    print("\n=== Summary ===")
    print(f"ARI: {metrics['dual_fit/syllable_ari']:.4f}")
    print(f"NMI: {metrics['dual_fit/syllable_nmi']:.4f}")
    print(f"Duration KL: {metrics['dual_fit/duration_kl']:.4f}")
    if "eml_intention" in metrics:
        print(f"EML Intention: {metrics['eml_intention']:.4f}")
    if "eml_keypoint" in metrics:
        print(f"EML Keypoint: {metrics['eml_keypoint']:.4f}")

    return {
        "metrics": metrics,
        "plots": plots,
        "video_paths": video_paths,
        "dual_result": dual_result,
    }


def ensemble_run(
    config: SweepConfig,
    hyperparams: KPMSHyperparams,
    seeds: list[int],
    split: str = "test",
) -> dict[str, Any]:
    """Run ensemble of models with multiple seeds to compute valid EML."""
    from topo_mimic.benchmark.hyperparam_sweep.metrics import compute_eml

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/videos", exist_ok=True)

    if split == "test":
        clip_range = config.data.test_clips
    else:
        clip_range = config.data.train_clips

    print("=" * 60)
    print(f"KPMS Ensemble Run - {len(seeds)} seeds")
    print("=" * 60)
    print(f"Hyperparameters: {hyperparams_to_dict(hyperparams)}")
    print(f"Seeds: {seeds}")

    print("\n=== Loading data ===")
    intention_data, clip_indices = load_data("intention", config.data.rodent_rollout_dir, clip_range)

    qpos_data, keypoint_data, kp_names = load_qpos_and_convert_keypoints(
        rodent_rollout_dir=config.data.rodent_rollout_dir,
        clip_range=clip_range,
        stac_xml_path=config.keypoint.xml_path,
        stac_data_path=config.keypoint.stac_data_path,
    )

    print("\n=== Running ensemble dual fit ===")
    results, ensemble_metrics = run_ensemble_dual_fit(
        intention_data=intention_data,
        keypoint_data=keypoint_data,
        hyperparams=hyperparams,
        output_dir=output_dir,
        seeds=seeds,
        kp_names=kp_names,
    )

    metrics = {}
    last_result = results[-1]

    int_duration = compute_duration_stats(last_result.intention_result.results)
    metrics["intention_fit/duration_mean"] = int_duration.mean
    metrics["intention_fit/duration_std"] = int_duration.std
    metrics["intention_fit/n_segments"] = int_duration.total_segments

    kp_duration = compute_duration_stats(last_result.keypoint_result.results)
    metrics["keypoint_fit/duration_mean"] = kp_duration.mean
    metrics["keypoint_fit/duration_std"] = kp_duration.std
    metrics["keypoint_fit/n_segments"] = kp_duration.total_segments

    metrics["intention_fit/n_syllables_active"] = compute_active_syllables(last_result.intention_result.results)
    metrics["keypoint_fit/n_syllables_active"] = compute_active_syllables(last_result.keypoint_result.results)

    metrics["intention_fit/transition_entropy"] = compute_transition_entropy(last_result.intention_result.model)
    metrics["keypoint_fit/transition_entropy"] = compute_transition_entropy(last_result.keypoint_result.model)

    metrics["dual_fit/syllable_ari"] = ensemble_metrics["syllable_ari_mean"]
    metrics["dual_fit/syllable_ari_std"] = ensemble_metrics["syllable_ari_std"]
    metrics["dual_fit/syllable_nmi"] = ensemble_metrics["syllable_nmi_mean"]
    metrics["dual_fit/syllable_nmi_std"] = ensemble_metrics["syllable_nmi_std"]
    metrics["dual_fit/duration_kl"] = ensemble_metrics["duration_kl_mean"]

    if ensemble_metrics["eml_intention"] is not None:
        metrics["eml_intention"] = ensemble_metrics["eml_intention"].mean_score
        metrics["eml_intention_std"] = ensemble_metrics["eml_intention"].mean_std_err
    if ensemble_metrics["eml_keypoint"] is not None:
        metrics["eml_keypoint"] = ensemble_metrics["eml_keypoint"].mean_score
        metrics["eml_keypoint_std"] = ensemble_metrics["eml_keypoint"].mean_std_err

    print("\n=== Generating plots ===")
    plots = create_all_plots(
        int_model=last_result.intention_result.model,
        kp_model=last_result.keypoint_result.model,
        int_results=last_result.intention_result.results,
        kp_results=last_result.keypoint_result.results,
    )

    print("\n=== Summary ===")
    print(f"ARI: {metrics['dual_fit/syllable_ari']:.4f} +/- {metrics.get('dual_fit/syllable_ari_std', 0):.4f}")
    print(f"NMI: {metrics['dual_fit/syllable_nmi']:.4f} +/- {metrics.get('dual_fit/syllable_nmi_std', 0):.4f}")
    if "eml_intention" in metrics:
        print(f"EML Intention: {metrics['eml_intention']:.4f}")
    if "eml_keypoint" in metrics:
        print(f"EML Keypoint: {metrics['eml_keypoint']:.4f}")

    return {
        "metrics": metrics,
        "plots": plots,
        "video_paths": {},
        "results": results,
    }


def sweep_agent_fn(config: SweepConfig) -> None:
    if not HAS_WANDB:
        raise ImportError("wandb not installed")

    with wandb.init() as run:
        hyperparams = KPMSHyperparams(
            kappa=wandb.config.get("kappa", config.hyperparams.kappa),
            latent_dim=wandb.config.get("latent_dim", config.hyperparams.latent_dim),
            num_states=wandb.config.get("num_states", config.hyperparams.num_states),
            ar_iters=wandb.config.get("ar_iters", config.hyperparams.ar_iters),
            full_iters=wandb.config.get("full_iters", config.hyperparams.full_iters),
            model_type=wandb.config.get("model_type", config.hyperparams.model_type),
        )

        run_name = f"k{hyperparams.kappa:.0e}_s{hyperparams.num_states}_d{hyperparams.latent_dim}"
        run.name = run_name

        run_output_dir = f"{config.output_dir}/{run.id}"
        config_copy = SweepConfig(
            output_dir=run_output_dir,
            hyperparams=hyperparams,
            data=config.data,
            keypoint=config.keypoint,
            wandb=config.wandb,
            evaluation=config.evaluation,
            rendering=config.rendering,
        )

        result = single_run(config_copy, hyperparams, seed=0)

        log_metrics_as_bar_charts(result["metrics"])

        for name, fig in result["plots"].items():
            wandb.log({name: wandb.Image(fig)})

        for name, path in result["video_paths"].items():
            if os.path.exists(path):
                wandb.log({name: wandb.Video(path, fps=config.rendering.fps)})


def run_sweep(
    config_path: str,
    sweep_config_path: str,
    count: int | None = None,
) -> None:
    if not HAS_WANDB:
        raise ImportError("wandb not installed")

    config = load_config(config_path)

    import yaml
    with open(sweep_config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )

    print(f"Created sweep: {sweep_id}")

    wandb.agent(
        sweep_id,
        function=lambda: sweep_agent_fn(config),
        count=count,
    )


def main():
    parser = argparse.ArgumentParser(description="KPMS Hyperparameter Sweep")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark/hyperparam_sweep/kpms_sweep.yaml",
        help="Path to sweep config YAML",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run as WandB sweep agent",
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        default="configs/benchmark/hyperparam_sweep/sweep_wandb.yaml",
        help="Path to WandB sweep config YAML",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of sweep runs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for single run",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Data split to use",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Run ensemble of models (multiple seeds) to compute valid EML",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of seeds for ensemble mode",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.no_wandb:
        config.wandb.enabled = False

    if args.sweep:
        run_sweep(args.config, args.sweep_config, args.count)
    elif args.ensemble:
        logger = WandBLogger(config)
        hp = config.hyperparams
        seeds = list(range(args.n_seeds))

        if config.wandb.enabled:
            run_name = f"ensemble_k{hp.kappa:.0e}_s{hp.num_states}_d{hp.latent_dim}_n{args.n_seeds}"
            logger.init(
                hyperparams=config.hyperparams,
                run_name=run_name,
                tags=["ensemble", args.split, f"n_seeds={args.n_seeds}"],
            )

        try:
            result = ensemble_run(config, config.hyperparams, seeds, args.split)

            if config.wandb.enabled:
                logger.log_as_bar_charts(result["metrics"])
                logger.log_plots(result["plots"])
                logger.log_videos(result["video_paths"], fps=config.rendering.fps)

        finally:
            logger.finish()
    else:
        logger = WandBLogger(config)

        if config.wandb.enabled:
            hp = config.hyperparams
            run_name = f"k{hp.kappa:.0e}_s{hp.num_states}_d{hp.latent_dim}_seed{args.seed}"
            logger.init(
                hyperparams=config.hyperparams,
                run_name=run_name,
                tags=["single_run", args.split],
            )

        try:
            result = single_run(config, config.hyperparams, args.seed, args.split)

            if config.wandb.enabled:
                logger.log_as_bar_charts(result["metrics"])
                logger.log_plots(result["plots"])
                logger.log_videos(result["video_paths"], fps=config.rendering.fps)

        finally:
            logger.finish()


if __name__ == "__main__":
    main()
