"""VAE training loop with WandB logging."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from topo_mimic.benchmark.hyperparam_sweep.custom_vae.config import VAEHyperparams
from topo_mimic.benchmark.hyperparam_sweep.custom_vae.model import RNNVAE
from topo_mimic.benchmark.hyperparam_sweep.custom_vae.dataset import (
    WindowedDataset,
    create_train_val_datasets,
)


@dataclass
class VAETrainResult:
    model: RNNVAE
    train_losses: list[float]
    val_losses: list[float]
    recon_losses: list[float]
    kl_losses: list[float]
    latent_embeddings: np.ndarray
    final_epoch: int


class VAETrainer:
    def __init__(
        self,
        model: RNNVAE,
        hyperparams: VAEHyperparams,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.hyperparams = hyperparams
        self.device = device

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hyperparams.learning_rate,
        )

    def compute_loss(
        self,
        x: torch.Tensor,
        recon_x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + self.hyperparams.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> tuple[float, float, float]:
        self.model.train()
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()
            recon_x, mu, logvar = self.model(batch)
            total_loss, recon_loss, kl_loss = self.compute_loss(batch, recon_x, mu, logvar)

            total_loss.backward()
            self.optimizer.step()

            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            n_batches += 1

        return (
            total_loss_sum / n_batches,
            recon_loss_sum / n_batches,
            kl_loss_sum / n_batches,
        )

    def eval_epoch(self, dataloader: DataLoader) -> tuple[float, float, float]:
        self.model.eval()
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                recon_x, mu, logvar = self.model(batch)
                total_loss, recon_loss, kl_loss = self.compute_loss(batch, recon_x, mu, logvar)

                total_loss_sum += total_loss.item()
                recon_loss_sum += recon_loss.item()
                kl_loss_sum += kl_loss.item()
                n_batches += 1

        return (
            total_loss_sum / n_batches,
            recon_loss_sum / n_batches,
            kl_loss_sum / n_batches,
        )

    def train(
        self,
        train_data: np.ndarray,
        wandb_logger: Any | None = None,
        modality_name: str = "intention",
        val_ratio: float = 0.1,
        seed: int = 0,
    ) -> VAETrainResult:
        train_dataset, val_dataset = create_train_val_datasets(
            train_data,
            time_window=self.hyperparams.time_window,
            stride=1,
            val_ratio=val_ratio,
            seed=seed,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.hyperparams.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.hyperparams.batch_size,
            shuffle=False,
            num_workers=0,
        )

        train_losses = []
        val_losses = []
        recon_losses = []
        kl_losses = []

        best_val_loss = float("inf")
        patience = 20
        patience_counter = 0
        final_epoch = 0

        pbar = tqdm(range(self.hyperparams.max_epochs), desc=f"Training VAE ({modality_name})")
        for epoch in pbar:
            train_loss, train_recon, train_kl = self.train_epoch(train_loader)
            val_loss, val_recon, val_kl = self.eval_epoch(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            recon_losses.append(train_recon)
            kl_losses.append(train_kl)

            pbar.set_postfix({
                "train": f"{train_loss:.4f}",
                "val": f"{val_loss:.4f}",
                "recon": f"{train_recon:.4f}",
                "kl": f"{train_kl:.4f}",
            })

            if wandb_logger is not None and wandb_logger.enabled:
                wandb_logger.log({
                    f"{modality_name}_fit/train/total_loss": train_loss,
                    f"{modality_name}_fit/train/recon_loss": train_recon,
                    f"{modality_name}_fit/train/kl_loss": train_kl,
                    f"{modality_name}_fit/val/total_loss": val_loss,
                    f"{modality_name}_fit/val/recon_loss": val_recon,
                    f"{modality_name}_fit/val/kl_loss": val_kl,
                    f"{modality_name}_fit/train/epoch": epoch,
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            final_epoch = epoch

        latent_embeddings = self.encode_all(train_data)

        return VAETrainResult(
            model=self.model,
            train_losses=train_losses,
            val_losses=val_losses,
            recon_losses=recon_losses,
            kl_losses=kl_losses,
            latent_embeddings=latent_embeddings,
            final_epoch=final_epoch,
        )

    def encode_all(self, data: np.ndarray) -> np.ndarray:
        self.model.eval()

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        n_clips, n_frames, n_dim = data.shape
        time_window = self.hyperparams.time_window

        all_embeddings = []

        with torch.no_grad():
            for clip_idx in range(n_clips):
                clip_embeddings = []

                for start in range(0, n_frames - time_window + 1, time_window):
                    window = data[clip_idx, start:start + time_window, :]
                    window_tensor = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
                    z = self.model.encode_sequence(window_tensor)
                    clip_embeddings.append(z.cpu().numpy())

                last_start = n_frames - time_window
                if last_start > (len(clip_embeddings) - 1) * time_window:
                    window = data[clip_idx, last_start:last_start + time_window, :]
                    window_tensor = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
                    z = self.model.encode_sequence(window_tensor)
                    clip_embeddings.append(z.cpu().numpy())

                all_embeddings.append(np.concatenate(clip_embeddings, axis=0))

        return np.array(all_embeddings)

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        self.model.eval()

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        n_clips, n_frames, n_dim = data.shape
        time_window = self.hyperparams.time_window

        reconstructed = np.zeros_like(data)
        count_matrix = np.zeros((n_clips, n_frames, 1))

        with torch.no_grad():
            for clip_idx in range(n_clips):
                for start in range(0, n_frames - time_window + 1):
                    window = data[clip_idx, start:start + time_window, :]
                    window_tensor = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
                    recon, _, _ = self.model(window_tensor)
                    recon = recon.cpu().numpy()[0]

                    reconstructed[clip_idx, start:start + time_window, :] += recon
                    count_matrix[clip_idx, start:start + time_window, :] += 1

        count_matrix = np.maximum(count_matrix, 1)
        reconstructed = reconstructed / count_matrix

        return reconstructed

    def encode_per_frame(self, data: np.ndarray) -> np.ndarray:
        self.model.eval()

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        n_clips, n_frames, n_dim = data.shape
        time_window = self.hyperparams.time_window
        latent_dim = self.hyperparams.latent_dim

        all_embeddings = np.zeros((n_clips, n_frames, latent_dim))
        count_matrix = np.zeros((n_clips, n_frames, 1))

        with torch.no_grad():
            for clip_idx in range(n_clips):
                for start in range(0, n_frames - time_window + 1):
                    window = data[clip_idx, start:start + time_window, :]
                    window_tensor = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
                    z = self.model.encode_sequence(window_tensor).cpu().numpy()[0]

                    center = start + time_window // 2
                    all_embeddings[clip_idx, center, :] += z
                    count_matrix[clip_idx, center, :] += 1

        count_matrix = np.maximum(count_matrix, 1)
        all_embeddings = all_embeddings / count_matrix

        for clip_idx in range(n_clips):
            first_valid = time_window // 2
            last_valid = n_frames - time_window // 2 - 1

            for t in range(first_valid):
                all_embeddings[clip_idx, t, :] = all_embeddings[clip_idx, first_valid, :]
            for t in range(last_valid + 1, n_frames):
                all_embeddings[clip_idx, t, :] = all_embeddings[clip_idx, last_valid, :]

        return all_embeddings
