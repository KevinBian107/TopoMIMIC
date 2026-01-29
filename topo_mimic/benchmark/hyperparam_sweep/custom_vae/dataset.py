"""Windowed dataset for VAE training."""

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowedDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        time_window: int = 30,
        stride: int = 1,
    ):
        self.time_window = time_window
        self.stride = stride

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        self.n_clips, self.n_frames, self.n_dim = data.shape
        self.data = data

        self.windows = []
        self.clip_indices = []
        self.frame_indices = []

        for clip_idx in range(self.n_clips):
            n_windows = (self.n_frames - time_window) // stride + 1
            for w_idx in range(n_windows):
                start = w_idx * stride
                self.windows.append((clip_idx, start))
                self.clip_indices.append(clip_idx)
                self.frame_indices.append(start)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        clip_idx, start = self.windows[idx]
        window = self.data[clip_idx, start:start + self.time_window, :]
        return torch.from_numpy(window).float()

    def get_window_info(self, idx: int) -> tuple[int, int]:
        return self.windows[idx]


def create_train_val_datasets(
    data: np.ndarray,
    time_window: int = 30,
    stride: int = 1,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[WindowedDataset, WindowedDataset]:
    full_dataset = WindowedDataset(data, time_window, stride)

    n_total = len(full_dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_windows = [full_dataset.windows[i] for i in train_indices]
    val_windows = [full_dataset.windows[i] for i in val_indices]

    train_dataset = WindowedDataset.__new__(WindowedDataset)
    train_dataset.time_window = time_window
    train_dataset.stride = stride
    train_dataset.n_clips = full_dataset.n_clips
    train_dataset.n_frames = full_dataset.n_frames
    train_dataset.n_dim = full_dataset.n_dim
    train_dataset.data = full_dataset.data
    train_dataset.windows = train_windows
    train_dataset.clip_indices = [w[0] for w in train_windows]
    train_dataset.frame_indices = [w[1] for w in train_windows]

    val_dataset = WindowedDataset.__new__(WindowedDataset)
    val_dataset.time_window = time_window
    val_dataset.stride = stride
    val_dataset.n_clips = full_dataset.n_clips
    val_dataset.n_frames = full_dataset.n_frames
    val_dataset.n_dim = full_dataset.n_dim
    val_dataset.data = full_dataset.data
    val_dataset.windows = val_windows
    val_dataset.clip_indices = [w[0] for w in val_windows]
    val_dataset.frame_indices = [w[1] for w in val_windows]

    return train_dataset, val_dataset
