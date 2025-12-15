"""Utils for helping loading data"""

from typing import Any, Union
import os
import h5py
import numpy as np
import re
import jax.numpy as jnp


def save_to_h5py(
    file: h5py.File,
    data: Union[dict, list, int, float, str, bool, np.ndarray],
    group_path: str = "/",
) -> None:
    """
    Save a pytree (like a dictionary) into an HDF5 file.

    Args:
        file: Open HDF5 file handle
        data: Data structure to save (dict, list, or array-like)
        group_path: HDF5 group path to save under
    """
    if isinstance(data, dict):
        for key, value in data.items():
            sub_group_path = f"{group_path}/{key}"
            save_to_h5py(file, value, sub_group_path)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            sub_group_path = f"{group_path}/{i}"
            save_to_h5py(file, item, sub_group_path)
    elif isinstance(data, (int, float, str, bool, np.ndarray)):
        file.create_dataset(group_path, data=data)
    elif hasattr(data, "numpy"):  # For NumPy arrays or PyTorch tensors
        file.create_dataset(group_path, data=data.numpy())
    elif isinstance(data, jnp.ndarray):  # Convert JAX array to NumPy
        file.create_dataset(group_path, data=np.array(data))
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def load_from_h5py(
    file: h5py.File,
    group_path: str = "/",
) -> Union[dict, list, np.ndarray]:
    """
    Load a pytree structure from an HDF5 file.

    Args:
        file: Open HDF5 file handle
        group_path: HDF5 group path to load from

    Returns:
        Loaded data structure (dict, list, or array)
    """
    group = file[group_path]
    if isinstance(group, h5py.Dataset):
        return group[()]  # Read dataset value
    elif isinstance(group, h5py.Group):
        if all(k.isdigit() for k in group.keys()):  # Likely a list
            return [
                load_from_h5py(file, f"{group_path}/{k}")
                for k in sorted(group.keys(), key=int)
            ]
        else:  # Dictionary-like group
            return {k: load_from_h5py(file, f"{group_path}/{k}") for k in group.keys()}
    else:
        raise TypeError(f"Unsupported group type: {type(group)}")


def get_aggregate_data(
    group_path: str,
    keys: list[str],
    clip_idx: int,
    path: str,
) -> Union[dict, list, np.ndarray]:
    """
    Get the aggregate data from the hdf5 file.

    Args:
        group_path: HDF5 group path to read from
        keys: List of keys to traverse in the data structure
        clip_idx: Index of the clip file to load
        path: Directory containing clip HDF5 files

    Returns:
        Extracted data from the specified path and keys
    """
    with h5py.File(path + f"/clip_{clip_idx}.h5", "r") as h5file:
        data = load_from_h5py(h5file, group_path=group_path)
        for key in keys:
            if type(data) == list and type(data[0]) == dict:
                data = [d[key] for d in data]
            elif type(data) == dict:
                data = data[key]
            else:
                raise ValueError("Data structure not supported")
    return data


def extract_clip_info(
    snippet_path: str,
) -> tuple[str | None, int | None]:
    """
    Extracts the behavior label and clip number from a snippet filename.

    Args:
        snippet_path: Path to the snippet file

    Returns:
        Tuple of (behavior_label, clip_number) or (None, None) if parsing fails
    """
    filename = os.path.basename(snippet_path)
    name, _ = os.path.splitext(filename)

    match = re.match(r"([a-zA-Z]+)_(\d+)", name)
    if match:
        behavior = match.group(1)
        clip_number = int(match.group(2))
        return behavior, clip_number
    else:
        return None, None


def subsample_data(
    X: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    """
    Randomly subsamples the dataset for faster computation.

    Args:
        X: Input array of shape (N, ...)
        sample_size: Number of samples to draw

    Returns:
        Subsampled array of shape (sample_size, ...)
    """
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], sample_size, replace=False)
    return X[indices]


def split_clips_by_frames(
    data: np.ndarray,
    original_num_clips: int,
    frames_per_clip: int = 500,
) -> np.ndarray:
    """
    Split loaded clip data into smaller segments based on frames_per_clip.

    Args:
        data: Stacked data array from multiple clips (total_frames, dim)
        original_num_clips: Number of original clips loaded
        frames_per_clip: Number of frames per output clip segment

    Returns:
        Reshaped array (new_num_clips, frames_per_clip - 1, dim)
        where new_num_clips = original_num_clips * (original_frames // frames_per_clip)

    Example:
        - frames_per_clip=500: 500 clips -> 500 clips with 499 frames each
        - frames_per_clip=100: 500 clips -> 2500 clips with 99 frames each
    """
    dim = data.shape[-1]
    total_frames = data.shape[0]
    original_frames_per_clip = total_frames // original_num_clips

    if frames_per_clip >= original_frames_per_clip + 1:
        return data.reshape(original_num_clips, original_frames_per_clip, dim)

    data_reshaped = data.reshape(original_num_clips, original_frames_per_clip, dim)

    usable_frames = frames_per_clip - 1
    splits_per_clip = original_frames_per_clip // usable_frames

    segments = []
    for clip_idx in range(original_num_clips):
        clip_data = data_reshaped[clip_idx]
        for seg_idx in range(splits_per_clip):
            start = seg_idx * usable_frames
            end = start + usable_frames
            segment = clip_data[start:end]
            segments.append(segment)

    return np.array(segments)
