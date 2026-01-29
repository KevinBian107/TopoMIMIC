"""Main entry point for VAE hyperparameter sweep with WandB integration."""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

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
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from topo_mimic.dmwk.utils.load_utils import get_aggregate_data, load_rollout_from_source
from topo_mimic.dmwk.utils.rendering import render_from_saved_rollout

from topo_mimic.benchmark.hyperparam_sweep.custom_vae.config import (
    VAESweepConfig,
    VAEHyperparams,
    load_vae_config,
    vae_hyperparams_to_dict,
)
from topo_mimic.benchmark.hyperparam_sweep.custom_vae.dual_fit import (
    fit_vae,
)
from topo_mimic.benchmark.hyperparam_sweep.metrics import (
    compute_duration_stats,
    compute_active_syllables,
)
from topo_mimic.benchmark.hyperparam_sweep.wandb_logger import (
    WandBLogger,
    log_metrics_as_bar_charts,
)
from topo_mimic.benchmark.hyperparam_sweep.visualization import (
    plot_duration_distribution,
    plot_syllable_frequencies,
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


def plot_reconstruction_curves(
    original: np.ndarray,
    reconstructed: np.ndarray,
    clip_idx: int = 0,
    n_dims: int = 5,
    max_frames: int = 500,
    title: str = "VAE Reconstruction",
) -> "plt.Figure":
    if not HAS_MPL:
        raise ImportError("matplotlib not installed")

    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    n_frames = min(original.shape[1], reconstructed.shape[1], max_frames)
    t = np.arange(n_frames)

    for dim_idx in range(n_dims):
        ax = axes[dim_idx]

        orig = original[clip_idx, :n_frames, dim_idx]
        recon = reconstructed[clip_idx, :n_frames, dim_idx]

        ax.plot(t, orig, label="Original", color="blue", alpha=0.7, linewidth=1)
        ax.plot(t, recon, label="Reconstructed", color="red", alpha=0.7, linewidth=1)

        mse = np.mean((orig - recon) ** 2)
        ax.set_ylabel(f"Dim {dim_idx}")
        ax.set_title(f"Dimension {dim_idx} (MSE: {mse:.4f})", fontsize=10)
        if dim_idx == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Frame")
    fig.suptitle(f"{title} - Clip {clip_idx}", fontsize=12)
    plt.tight_layout()

    return fig


def _get_syllable_color(syllable_id: int, n_colors: int = 30) -> tuple[int, int, int]:
    if HAS_MPL:
        cmap = plt.cm.get_cmap("tab20", n_colors)
        color = cmap(syllable_id % n_colors)
        return (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    else:
        hue = (syllable_id * 137) % 360
        r = int(128 + 127 * np.sin(np.radians(hue)))
        g = int(128 + 127 * np.sin(np.radians(hue + 120)))
        b = int(128 + 127 * np.sin(np.radians(hue + 240)))
        return (r, g, b)


def _ensure_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def render_syllable_video(
    frames: list[np.ndarray],
    syllables: np.ndarray,
    output_path: str,
    fps: int = 30,
    bar_height: int = 30,
) -> str:
    if not HAS_IMAGEIO:
        raise ImportError("imageio-ffmpeg or PIL not installed")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if len(frames) == 0:
        raise ValueError("No frames provided")

    first_frame = _ensure_rgb_uint8(frames[0])
    H, W = first_frame.shape[:2]

    canvas_h = H + bar_height
    canvas_w = W

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    n_frames = min(len(frames), len(syllables))

    timeline_bar = Image.new("RGB", (canvas_w, bar_height), (255, 255, 255))
    timeline_draw = ImageDraw.Draw(timeline_bar)

    for t in range(n_frames):
        syll = int(syllables[t])
        color = _get_syllable_color(syll)
        x = int(t * canvas_w / n_frames)
        x_next = int((t + 1) * canvas_w / n_frames)
        timeline_draw.rectangle([x, 0, x_next, bar_height], fill=color)

    writer = imageio_ffmpeg.write_frames(
        output_path, (canvas_w, canvas_h), fps=fps, codec="h264", macro_block_size=1
    )
    writer.send(None)

    for t in range(n_frames):
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        frame = _ensure_rgb_uint8(frames[t])
        frame_img = Image.fromarray(frame)
        canvas.paste(frame_img, (0, 0))

        syll = int(syllables[t])
        syll_color = _get_syllable_color(syll)
        draw.rounded_rectangle([5, 5, 95, 28], radius=5, fill=syll_color)
        draw.text((12, 7), f"Syllable {syll}", font=font, fill=(255, 255, 255))

        canvas.paste(timeline_bar, (0, H))

        pos_x = int(t * canvas_w / n_frames)
        draw.line([(pos_x, H), (pos_x, H + bar_height)], fill=(255, 255, 255), width=2)

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
    return output_path


def render_sample_videos(
    config: VAESweepConfig,
    results: dict,
    rodent_rollout_dir: str,
    clip_indices: list[int],
    output_dir: str,
    n_samples: int = 2,
) -> dict[str, str]:
    video_paths = {}

    rec_names = sorted(results.keys())[:n_samples]

    for i, rec_name in enumerate(rec_names):
        idx_in_batch = int(rec_name.split("_")[1])
        clip_id = clip_indices[idx_in_batch]

        syllables = np.array(results[rec_name]["syllable"])

        try:
            ro = load_rollout_from_source(clip_id, rodent_rollout_dir)
            frames = render_from_saved_rollout(ro, camera_name=config.rendering.camera_name)
        except Exception as e:
            print(f"Warning: Could not render clip {clip_id}: {e}")
            continue

        video_name = f"vae_motif_clip{clip_id}"
        video_path = f"{output_dir}/videos/{video_name}.mp4"

        try:
            render_syllable_video(
                frames=frames,
                syllables=syllables,
                output_path=video_path,
                fps=config.rendering.fps,
            )
            video_paths[video_name] = video_path
            print(f"Rendered video: {video_path}")
        except Exception as e:
            print(f"Warning: Could not render video for clip {clip_id}: {e}")

    return video_paths


def create_vae_plots(
    results: dict,
    original: np.ndarray | None = None,
    reconstructed: np.ndarray | None = None,
    n_recon_dims: int = 5,
) -> dict[str, Any]:
    plots = {}

    plots["intention/duration_dist"] = plot_duration_distribution(
        results, title="Intention Motif Duration Distribution"
    )
    plots["intention/motif_freq"] = plot_syllable_frequencies(
        results, title="Intention Motif Frequencies"
    )

    if original is not None and reconstructed is not None:
        plots["intention/reconstruction"] = plot_reconstruction_curves(
            original, reconstructed,
            clip_idx=0, n_dims=n_recon_dims, title="VAE Reconstruction"
        )

    return plots


def single_run(
    config: VAESweepConfig,
    hyperparams: VAEHyperparams,
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
    print("VAE Hyperparameter Sweep - Single Run")
    print("=" * 60)
    print(f"Hyperparameters: {vae_hyperparams_to_dict(hyperparams)}")
    print(f"Clip range: {clip_range}")
    print(f"Seed: {seed}")

    print("\n=== Loading data ===")
    intention_data, clip_indices = load_data("intention", config.data.rodent_rollout_dir, clip_range)

    logger = None
    if config.wandb.enabled and HAS_WANDB:
        class DummySweepConfig:
            def __init__(self, wandb_cfg):
                self.wandb = wandb_cfg

        dummy_config = DummySweepConfig(config.wandb)
        logger = WandBLogger(dummy_config)

    print("\n=== Running VAE fit ===")
    result = fit_vae(
        data=intention_data,
        hyperparams=hyperparams,
        wandb_logger=logger,
        modality_name="intention",
        seed=seed,
    )

    print("\n=== Computing reconstruction ===")
    from topo_mimic.benchmark.hyperparam_sweep.custom_vae.trainer import VAETrainer
    trainer = VAETrainer(result.model, hyperparams)
    reconstructed = trainer.reconstruct(intention_data)
    recon_mse = np.mean((intention_data - reconstructed) ** 2)
    print(f"Reconstruction MSE: {recon_mse:.6f}")

    print("\n=== Computing metrics ===")
    metrics = {}

    duration_stats = compute_duration_stats(result.results)
    metrics["intention_fit/duration_mean"] = duration_stats.mean
    metrics["intention_fit/duration_std"] = duration_stats.std
    metrics["intention_fit/n_segments"] = duration_stats.total_segments

    metrics["intention_fit/n_motifs_active"] = compute_active_syllables(result.results)
    metrics["intention_fit/motif_usage_ratio"] = (
        metrics["intention_fit/n_motifs_active"] / hyperparams.n_clusters
    )

    train_result = result.train_result
    metrics["intention_fit/final_recon_loss"] = train_result.recon_losses[-1]
    metrics["intention_fit/final_kl_loss"] = train_result.kl_losses[-1]
    metrics["intention_fit/final_total_loss"] = train_result.train_losses[-1]
    metrics["intention_fit/reconstruction_mse"] = recon_mse

    latent = result.latent_embeddings
    metrics["latent/mean_norm"] = float(np.mean(np.linalg.norm(latent, axis=-1)))
    metrics["latent/std_norm"] = float(np.std(np.linalg.norm(latent, axis=-1)))

    print("\n=== Generating plots ===")
    plots = create_vae_plots(
        results=result.results,
        original=intention_data,
        reconstructed=reconstructed,
        n_recon_dims=5,
    )

    print("\n=== Rendering sample videos ===")
    video_paths = {}
    if config.rendering.enabled:
        video_paths = render_sample_videos(
            config=config,
            results=result.results,
            rodent_rollout_dir=config.data.rodent_rollout_dir,
            clip_indices=clip_indices,
            output_dir=output_dir,
            n_samples=config.wandb.n_video_samples,
        )

    print("\n=== Saving results ===")
    results_path = f"{output_dir}/results.pkl"
    with open(results_path, "wb") as f:
        pkl.dump({
            "metrics": metrics,
            "hyperparams": vae_hyperparams_to_dict(hyperparams),
            "seed": seed,
            "clip_range": clip_range,
        }, f)
    print(f"Saved results to {results_path}")

    print("\n=== Summary ===")
    print(f"Active motifs: {metrics['intention_fit/n_motifs_active']}/{hyperparams.n_clusters}")
    print(f"Duration mean: {metrics['intention_fit/duration_mean']:.2f} frames")
    print(f"Reconstruction MSE: {metrics['intention_fit/reconstruction_mse']:.6f}")
    print(f"Final recon loss: {metrics['intention_fit/final_recon_loss']:.4f}")
    print(f"Final KL loss: {metrics['intention_fit/final_kl_loss']:.4f}")

    return {
        "metrics": metrics,
        "plots": plots,
        "video_paths": video_paths,
        "result": result,
    }


def sweep_agent_fn(config: VAESweepConfig) -> None:
    if not HAS_WANDB:
        raise ImportError("wandb not installed")

    with wandb.init() as run:
        hyperparams = VAEHyperparams(
            latent_dim=wandb.config.get("latent_dim", config.hyperparams.latent_dim),
            time_window=wandb.config.get("time_window", config.hyperparams.time_window),
            hidden_dim=wandb.config.get("hidden_dim", config.hyperparams.hidden_dim),
            rnn_layers=wandb.config.get("rnn_layers", config.hyperparams.rnn_layers),
            rnn_type=wandb.config.get("rnn_type", config.hyperparams.rnn_type),
            beta=wandb.config.get("beta", config.hyperparams.beta),
            n_clusters=wandb.config.get("n_clusters", config.hyperparams.n_clusters),
            segmentation_algo=wandb.config.get("segmentation_algo", config.hyperparams.segmentation_algo),
            learning_rate=wandb.config.get("learning_rate", config.hyperparams.learning_rate),
            batch_size=wandb.config.get("batch_size", config.hyperparams.batch_size),
            max_epochs=wandb.config.get("max_epochs", config.hyperparams.max_epochs),
        )

        run_name = f"vae_z{hyperparams.latent_dim}_w{hyperparams.time_window}_k{hyperparams.n_clusters}"
        run.name = run_name

        run_output_dir = f"{config.output_dir}/{run.id}"
        config_copy = VAESweepConfig(
            output_dir=run_output_dir,
            hyperparams=hyperparams,
            data=config.data,
            keypoint=config.keypoint,
            wandb=config.wandb,
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

    config = load_vae_config(config_path)

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
    parser = argparse.ArgumentParser(description="VAE Hyperparameter Sweep")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark/hyperparam_sweep/vae_sweep.yaml",
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
        default="configs/benchmark/hyperparam_sweep/vae_sweep_wandb.yaml",
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

    args = parser.parse_args()

    config = load_vae_config(args.config)

    if args.no_wandb:
        config.wandb.enabled = False

    if args.sweep:
        run_sweep(args.config, args.sweep_config, args.count)
    else:
        class DummySweepConfig:
            def __init__(self, vae_config):
                self.wandb = vae_config.wandb
                self.evaluation = type("obj", (object,), {"n_models_per_setting": 1})()

        logger = WandBLogger(DummySweepConfig(config))

        if config.wandb.enabled:
            hp = config.hyperparams
            run_name = f"vae_z{hp.latent_dim}_w{hp.time_window}_k{hp.n_clusters}_seed{args.seed}"

            class DummyHyperparams:
                def __init__(self, hp):
                    for k, v in vae_hyperparams_to_dict(hp).items():
                        setattr(self, k, v)

            logger.init(
                hyperparams=DummyHyperparams(hp),
                run_name=run_name,
                tags=["vae", "single_run", args.split],
            )

        try:
            result = single_run(config, config.hyperparams, args.seed, args.split)

            if config.wandb.enabled:
                logger.log_as_bar_charts(result["metrics"])
                logger.log_plots(result["plots"])
                for name, path in result["video_paths"].items():
                    if os.path.exists(path):
                        logger.log_videos({name: path}, fps=config.rendering.fps)

        finally:
            logger.finish()


if __name__ == "__main__":
    main()
