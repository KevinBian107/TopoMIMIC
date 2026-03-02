"""Write generation results directly to TopoVNL unified format (H5 + CSV)."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

WALKER_MODES = [
    "stand",         # 0
    "walk_slow",     # 1
    "run",           # 2
    "knee_down",     # 3
    "walk_backward", # 4
    "hop",           # 5
    "walk_fast",     # 6
    "tiptoe",        # 7
]


def write_unified_h5(
    h5_path: str,
    latent_mean: np.ndarray,
    agent_qpos: np.ndarray,
) -> None:
    """Write /intention and /qpos_rollout to TopoVNL H5.

    Args:
        h5_path: Output HDF5 file path.
        latent_mean: (N, T, 16) latent intention embeddings.
        agent_qpos: (N, T, 9) agent joint positions.
    """
    Path(h5_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("intention", data=latent_mean)
        f.create_dataset("qpos_rollout", data=agent_qpos)

    size_mb = Path(h5_path).stat().st_size / 1024 / 1024
    print(f"Wrote {h5_path} ({size_mb:.1f} MB)")
    print(f"  /intention:    {latent_mean.shape}")
    print(f"  /qpos_rollout: {agent_qpos.shape}")


def write_labels_csv(
    csv_path: str,
    src_modes: np.ndarray,
    dst_modes: np.ndarray,
    transition_starts: np.ndarray,
    transition_ends: np.ndarray,
) -> pd.DataFrame:
    """Write behavior labels CSV from clip metadata arrays.

    Args:
        csv_path: Output CSV file path.
        src_modes: (N,) source mode indices per clip.
        dst_modes: (N,) destination mode indices per clip.
        transition_starts: (N,) transition start frame within each clip.
        transition_ends: (N,) transition end frame within each clip.

    Returns:
        DataFrame with Clip, Behavior, Start_Frame, End_Frame columns.
    """
    n_clips = len(src_modes)
    rows = []
    for i in range(n_clips):
        b1_name = WALKER_MODES[int(src_modes[i])]
        b2_name = WALKER_MODES[int(dst_modes[i])]
        t_start = int(transition_starts[i])
        t_end = int(transition_ends[i])

        rows.append({
            "Clip": i,
            "Behavior": b1_name,
            "Start_Frame": 0,
            "End_Frame": t_start,
        })
        rows.append({
            "Clip": i,
            "Behavior": b2_name,
            "Start_Frame": t_end,
            "End_Frame": -1,
        })

    df = pd.DataFrame(rows)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}: {len(df)} rows ({n_clips} clips x 2 segments)")
    return df
