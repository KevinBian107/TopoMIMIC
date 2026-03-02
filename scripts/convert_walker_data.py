"""Convert walker HDF5 files to unified TopoVNL format."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


WALKER_MODES = [
    "stand",        # 0
    "walk_slow",    # 1
    "run",          # 2  (excluded from dataset but index preserved)
    "knee_down",    # 3
    "walk_backward",# 4
    "hop",          # 5
    "walk_fast",    # 6
    "tiptoe",       # 7
]


def convert_activations(
    src_path: str,
    dst_path: str,
) -> dict[str, tuple]:
    with h5py.File(src_path, "r") as src:
        latent_mean = src["latent_mean"][:]
        agent_qpos = src["agent_qpos"][:]

    print(f"latent_mean: {latent_mean.shape}")
    print(f"agent_qpos:  {agent_qpos.shape}")

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dst_path, "w") as dst:
        dst.create_dataset("intention", data=latent_mean)
        dst.create_dataset("qpos_rollout", data=agent_qpos)

    shapes = {
        "intention": latent_mean.shape,
        "qpos_rollout": agent_qpos.shape,
    }
    print(f"Wrote {dst_path}")
    for k, v in shapes.items():
        print(f"  /{k}: {v}")
    return shapes


def build_labels_csv(
    dataset_path: str,
    csv_path: str,
) -> pd.DataFrame:
    with h5py.File(dataset_path, "r") as f:
        cm = f["clip_metadata"]
        behavior_1 = cm["behavior_1"][:]
        behavior_2 = cm["behavior_2"][:]
        transition_start = cm["transition_start"][:]
        transition_end = cm["transition_end"][:]

    n_clips = len(behavior_1)
    rows = []
    for i in range(n_clips):
        b1_idx = int(behavior_1[i])
        b2_idx = int(behavior_2[i])
        t_start = int(transition_start[i])
        t_end = int(transition_end[i])

        b1_name = WALKER_MODES[b1_idx]
        b2_name = WALKER_MODES[b2_idx]

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

    print("\nBehavior distribution:")
    for b in sorted(df["Behavior"].unique()):
        count = (df["Behavior"] == b).sum()
        print(f"  {b}: {count}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Convert walker data to unified TopoVNL format")
    parser.add_argument(
        "--activations",
        default="walker/data/topo_activations_2000x500_7modes.h5",
        help="Path to walker activations HDF5",
    )
    parser.add_argument(
        "--dataset",
        default="walker/data/topo_dataset_2000x500_7modes.h5",
        help="Path to walker dataset HDF5 (contains clip_metadata)",
    )
    parser.add_argument(
        "--output-h5",
        default="data/walker.h5",
        help="Output unified HDF5 path",
    )
    parser.add_argument(
        "--output-csv",
        default="walker_behavior_labels.csv",
        help="Output behavior labels CSV path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Walker Data Conversion")
    print("=" * 60)

    shapes = convert_activations(args.activations, args.output_h5)
    df = build_labels_csv(args.dataset, args.output_csv)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Clips:     {shapes['intention'][0]}")
    print(f"  Frames:    {shapes['intention'][1]}")
    print(f"  Intention: {shapes['intention'][2]}D")
    print(f"  Qpos:      {shapes['qpos_rollout'][2]}D")
    print(f"  CSV rows:  {len(df)}")


if __name__ == "__main__":
    main()
