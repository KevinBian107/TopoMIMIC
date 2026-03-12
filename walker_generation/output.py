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


def write_multi_segment_labels_csv(
    csv_path: str,
    clip_segment_lists: list[list[dict]],
) -> pd.DataFrame:
    """Write behavior labels CSV for long-sequence clips with multiple segments.

    Args:
        csv_path: Output CSV file path.
        clip_segment_lists: Per-clip list of segment dicts, each with keys:
            "behavior" (str), "start_frame" (int), "end_frame" (int).

    Returns:
        DataFrame with Clip, Behavior, Start_Frame, End_Frame columns.
    """
    rows = []
    for clip_id, segments in enumerate(clip_segment_lists):
        for seg in segments:
            rows.append({
                "Clip": clip_id,
                "Behavior": seg["behavior"],
                "Start_Frame": seg["start_frame"],
                "End_Frame": seg["end_frame"],
            })

    df = pd.DataFrame(rows)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    n_clips = len(clip_segment_lists)
    n_segs = len(df)
    print(f"Wrote {csv_path}: {n_segs} rows ({n_clips} clips, avg {n_segs / max(n_clips, 1):.1f} segments/clip)")
    return df
