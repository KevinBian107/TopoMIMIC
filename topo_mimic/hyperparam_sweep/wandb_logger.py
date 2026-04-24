"""WandB logging utilities for hyperparameter sweep."""

from pathlib import Path
from typing import Any

import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def init_sweep_run(
    config: Any,
    hyperparams: Any | None = None,
    run_name: str | None = None,
    tags: list[str] | None = None,
) -> Any:
    if not HAS_WANDB:
        raise ImportError("wandb not installed")

    if not config.wandb.enabled:
        return None

    hp = hyperparams if hyperparams is not None else config.hyperparams
    hp_dict = {}
    for attr in dir(hp):
        if not attr.startswith("_"):
            val = getattr(hp, attr)
            if isinstance(val, (int, float, str, bool, list, tuple)):
                hp_dict[attr] = val

    run_config = {**hp_dict}

    if hasattr(config, "evaluation") and hasattr(config.evaluation, "n_models_per_setting"):
        run_config["n_models_per_setting"] = config.evaluation.n_models_per_setting
    if hasattr(config, "data"):
        if hasattr(config.data, "train_clips"):
            run_config["train_clips"] = config.data.train_clips
        if hasattr(config.data, "test_clips"):
            run_config["test_clips"] = config.data.test_clips

    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        config=run_config,
        name=run_name,
        tags=tags,
        reinit=True,
    )

    return run


def log_metrics(metrics: dict[str, Any]) -> None:
    if not HAS_WANDB:
        return

    if wandb.run is None:
        return

    wandb.log(metrics)


def log_metrics_as_bar_charts(
    metrics: dict[str, Any],
    group_prefix: str | None = None,
) -> None:
    """Log metrics as individual panels, one per metric."""
    if not HAS_WANDB:
        return

    if wandb.run is None:
        return

    flat_metrics = {}
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue

        if group_prefix and not key.startswith(group_prefix):
            continue

        flat_key = key.replace("/", "_")
        flat_metrics[flat_key] = value
        wandb.run.summary[flat_key] = value

    wandb.log(flat_metrics)


def log_summary_metrics(metrics: dict[str, Any]) -> None:
    if not HAS_WANDB:
        return

    if wandb.run is None:
        return

    for key, value in metrics.items():
        wandb.run.summary[key] = value


def log_plots(plots: dict[str, "Figure"]) -> None:
    if not HAS_WANDB or not HAS_MPL:
        return

    if wandb.run is None:
        return

    images = {}
    for name, fig in plots.items():
        images[name] = wandb.Image(fig)
        plt.close(fig)

    wandb.log(images)


def log_videos(
    videos: dict[str, str | Path],
    fps: int = 30,
) -> None:
    if not HAS_WANDB:
        return

    if wandb.run is None:
        return

    video_logs = {}
    for name, path in videos.items():
        path_str = str(path)
        if Path(path_str).exists():
            video_logs[name] = wandb.Video(path_str, fps=fps, format="mp4")

    if video_logs:
        wandb.log(video_logs)


def log_artifact(
    name: str,
    artifact_type: str,
    file_paths: list[str | Path],
    metadata: dict[str, Any] | None = None,
) -> None:
    if not HAS_WANDB:
        return

    if wandb.run is None:
        return

    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)

    for path in file_paths:
        path = Path(path)
        if path.exists():
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))

    wandb.log_artifact(artifact)


def log_table(
    name: str,
    columns: list[str],
    data: list[list[Any]],
) -> None:
    if not HAS_WANDB:
        return

    if wandb.run is None:
        return

    table = wandb.Table(columns=columns, data=data)
    wandb.log({name: table})


def log_training_curves(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    recon_losses: list[float] | None = None,
    kl_losses: list[float] | None = None,
    prefix: str = "train",
) -> None:
    if not HAS_WANDB or not HAS_MPL:
        return

    if wandb.run is None:
        return

    fig, axes = plt.subplots(1, 2 if val_losses else 1, figsize=(12, 4))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    epochs = list(range(len(train_losses)))

    ax = axes[0]
    ax.plot(epochs, train_losses, label="Total Loss", color="blue")
    if recon_losses:
        ax.plot(epochs, recon_losses, label="Recon Loss", color="green", linestyle="--")
    if kl_losses:
        ax.plot(epochs, kl_losses, label="KL Loss", color="red", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{prefix} Training Curves")
    ax.legend()

    if val_losses and len(axes) > 1:
        ax = axes[1]
        ax.plot(epochs, train_losses, label="Train", color="blue")
        ax.plot(epochs, val_losses, label="Validation", color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        ax.set_title(f"{prefix} Train vs Validation")
        ax.legend()

    plt.tight_layout()
    wandb.log({f"{prefix}/training_curves": wandb.Image(fig)})
    plt.close(fig)


def finish_run() -> None:
    if not HAS_WANDB:
        return

    if wandb.run is not None:
        wandb.finish()


def create_sweep(
    sweep_config_path: str,
    project: str | None = None,
    entity: str | None = None,
) -> str:
    if not HAS_WANDB:
        raise ImportError("wandb not installed")

    import yaml
    with open(sweep_config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    if project is not None:
        sweep_config["project"] = project

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
        entity=entity,
    )

    return sweep_id


def run_sweep_agent(
    sweep_id: str,
    function: callable,
    count: int | None = None,
    project: str | None = None,
    entity: str | None = None,
) -> None:
    if not HAS_WANDB:
        raise ImportError("wandb not installed")

    wandb.agent(
        sweep_id,
        function=function,
        count=count,
        project=project,
        entity=entity,
    )


class WandBLogger:
    def __init__(self, config: Any):
        self.config = config
        self.enabled = config.wandb.enabled and HAS_WANDB
        self.run = None

    def init(
        self,
        hyperparams: Any | None = None,
        run_name: str | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        if not self.enabled:
            return None
        self.run = init_sweep_run(self.config, hyperparams, run_name, tags)
        return self.run

    def log(self, metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return
        log_metrics(metrics)

    def log_as_bar_charts(self, metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return
        log_metrics_as_bar_charts(metrics)

    def log_summary(self, metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return
        log_summary_metrics(metrics)

    def log_plots(self, plots: dict[str, "Figure"]) -> None:
        if not self.enabled:
            return
        log_plots(plots)

    def log_videos(self, videos: dict[str, str | Path], fps: int = 30) -> None:
        if not self.enabled:
            return
        log_videos(videos, fps)

    def log_training_curves(
        self,
        train_losses: list[float],
        val_losses: list[float] | None = None,
        recon_losses: list[float] | None = None,
        kl_losses: list[float] | None = None,
        prefix: str = "train",
    ) -> None:
        if not self.enabled:
            return
        log_training_curves(train_losses, val_losses, recon_losses, kl_losses, prefix)

    def finish(self) -> None:
        if not self.enabled:
            return
        finish_run()
        self.run = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False
