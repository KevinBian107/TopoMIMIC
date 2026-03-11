import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import numpy as np
import pickle as pkl
import yaml
import h5py
from dataclasses import dataclass
from tqdm import tqdm

import matplotlib.pyplot as plt

from vnl_tda.utils.load_utils import load_from_h5py, load_rollout_from_source
from vnl_tda.pipelines.tda.skeletonization.rendering import render_clip_grid


@dataclass
class DistributionConfig:
    n_cycle_clips: int = 10
    n_no_cycle_clips: int = 10
    body_parts: list[str] = None
    figsize: tuple[int, int] = (14, 8)

    def __post_init__(self):
        if self.body_parts is None:
            self.body_parts = [
                "torso", "pelvis",
                "upper_leg_L", "lower_leg_L", "foot_L",
                "upper_leg_R", "lower_leg_R", "foot_R",
                "skull", "upper_arm_L", "upper_arm_R",
            ]


@dataclass
class RenderConfig:
    dm_results_dir: str = ""
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    output_dir: str = "out_no_cycle_clips"
    num_clips: int = 500
    grid_cols: int = 5
    grid_rows: int = 4
    fps: int = 30
    target_size: tuple[int, int] = (320, 240)
    distribution: DistributionConfig = None

    def __post_init__(self):
        if self.distribution is None:
            self.distribution = DistributionConfig()


QPOS_BODY_NAMES = [
    "world", "floor", "walker", "torso", "vertebra_1", "vertebra_2", "vertebra_3",
    "vertebra_4", "vertebra_5", "vertebra_6", "pelvis", "upper_leg_L", "lower_leg_L",
    "foot_L", "toe_L", "upper_leg_R", "lower_leg_R", "foot_R", "toe_R",
    "vertebra_C1", "vertebra_C2", "vertebra_C3", "vertebra_C4", "vertebra_C5",
    "vertebra_C6", "vertebra_C7", "vertebra_C8", "vertebra_C9", "vertebra_C10",
    "vertebra_C11", "vertebra_C12", "vertebra_C13", "vertebra_C14", "vertebra_C15",
    "vertebra_C16", "vertebra_C17", "vertebra_C18", "vertebra_C19", "vertebra_C20",
    "vertebra_C21", "vertebra_C22", "vertebra_C23", "vertebra_C24", "vertebra_C25",
    "vertebra_C26", "vertebra_C27", "vertebra_C28", "vertebra_C29", "vertebra_C30",
    "vertebra_cervical_5", "vertebra_cervical_4", "vertebra_cervical_3",
    "vertebra_cervical_2", "vertebra_cervical_1", "vertebra_axis", "vertebra_atlant",
    "skull", "jaw", "scapula_L", "upper_arm_L", "lower_arm_L", "hand_L", "finger_L",
    "scapula_R", "upper_arm_R", "lower_arm_R", "hand_R", "finger_R",
]


def load_config(config_path: str) -> RenderConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    target_size = cfg.get("target_size", [320, 240])
    if isinstance(target_size, list):
        target_size = tuple(target_size)

    dist_dict = cfg.get("distribution", {})
    if "figsize" in dist_dict and isinstance(dist_dict["figsize"], list):
        dist_dict["figsize"] = tuple(dist_dict["figsize"])
    dist_cfg = DistributionConfig(**dist_dict)

    return RenderConfig(
        dm_results_dir=cfg.get("dm_results_dir", ""),
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", RenderConfig.rodent_rollout_dir),
        output_dir=cfg.get("output_dir", RenderConfig.output_dir),
        num_clips=cfg.get("num_clips", RenderConfig.num_clips),
        grid_cols=cfg.get("grid_cols", RenderConfig.grid_cols),
        grid_rows=cfg.get("grid_rows", RenderConfig.grid_rows),
        fps=cfg.get("fps", RenderConfig.fps),
        target_size=target_size,
        distribution=dist_cfg,
    )


def find_cycle_and_no_cycle_clips(
    dm_results_dir: str,
    num_clips: int,
) -> tuple[list[int], list[int]]:
    cycles_path = os.path.join(dm_results_dir, "all_cycle_dm.pkl")
    if not os.path.exists(cycles_path):
        raise FileNotFoundError(f"Cycles file not found: {cycles_path}")

    with open(cycles_path, "rb") as f:
        all_cycle_dm = pkl.load(f)

    cycle_clips = []
    no_cycle_clips = []
    for clip_id in range(min(num_clips, len(all_cycle_dm))):
        cycles = all_cycle_dm[clip_id]
        if len(cycles) == 0:
            no_cycle_clips.append(clip_id)
        else:
            cycle_clips.append(clip_id)

    return cycle_clips, no_cycle_clips


def load_qpos(clip_id: int, rodent_rollout_dir: str) -> np.ndarray:
    roll = load_rollout_from_source(int(clip_id), rodent_rollout_dir)
    return np.asarray(roll["qpos"])


def compute_delta_xyz_stats(
    clip_ids: list[int],
    rodent_rollout_dir: str,
    max_clips: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    all_deltas = []
    for i, clip_id in enumerate(clip_ids):
        if i >= max_clips:
            break
        try:
            qpos = load_qpos(clip_id, rodent_rollout_dir)
            delta = np.diff(qpos[:, :3], axis=0)
            all_deltas.append(np.abs(delta).mean(axis=0))
        except Exception:
            continue

    if not all_deltas:
        return np.zeros(3), np.zeros(3)

    all_deltas = np.vstack(all_deltas)
    return all_deltas.mean(axis=0), all_deltas.std(axis=0)


def compute_body_variance(
    clip_ids: list[int],
    rodent_rollout_dir: str,
    body_parts: list[str],
    max_clips: int = 10,
) -> np.ndarray:
    body_indices = []
    valid_body_parts = []
    for bp in body_parts:
        if bp in QPOS_BODY_NAMES:
            body_indices.append(QPOS_BODY_NAMES.index(bp))
            valid_body_parts.append(bp)

    all_qpos_by_body = {bp: [] for bp in valid_body_parts}

    for i, clip_id in enumerate(clip_ids):
        if i >= max_clips:
            break
        try:
            qpos = load_qpos(clip_id, rodent_rollout_dir)
            for bp, body_idx in zip(valid_body_parts, body_indices):
                if body_idx < qpos.shape[1]:
                    all_qpos_by_body[bp].append(qpos[:, body_idx])
        except Exception:
            continue

    variances = np.zeros(len(valid_body_parts))
    for i, bp in enumerate(valid_body_parts):
        if all_qpos_by_body[bp]:
            combined = np.concatenate(all_qpos_by_body[bp])
            variances[i] = np.var(combined)

    return variances, valid_body_parts


def plot_cycle_vs_no_cycle_comparison(
    cycle_clips: list[int],
    no_cycle_clips: list[int],
    rodent_rollout_dir: str,
    output_dir: str,
    dist_cfg: DistributionConfig,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    n_cycle = min(dist_cfg.n_cycle_clips, len(cycle_clips))
    n_no_cycle = min(dist_cfg.n_no_cycle_clips, len(no_cycle_clips))

    np.random.seed(42)
    sampled_cycle = np.random.choice(cycle_clips, n_cycle, replace=False).tolist() if len(cycle_clips) >= n_cycle else cycle_clips
    sampled_no_cycle = np.random.choice(no_cycle_clips, n_no_cycle, replace=False).tolist() if len(no_cycle_clips) >= n_no_cycle else no_cycle_clips

    print(f"\nSampled {len(sampled_cycle)} cycle clips: {sampled_cycle}")
    print(f"Sampled {len(sampled_no_cycle)} no-cycle clips: {sampled_no_cycle}")

    cycle_delta_mean, cycle_delta_std = compute_delta_xyz_stats(
        sampled_cycle, rodent_rollout_dir, max_clips=n_cycle
    )
    no_cycle_delta_mean, no_cycle_delta_std = compute_delta_xyz_stats(
        sampled_no_cycle, rodent_rollout_dir, max_clips=n_no_cycle
    )

    cycle_var, body_parts = compute_body_variance(
        sampled_cycle, rodent_rollout_dir, dist_cfg.body_parts, max_clips=n_cycle
    )
    no_cycle_var, _ = compute_body_variance(
        sampled_no_cycle, rodent_rollout_dir, dist_cfg.body_parts, max_clips=n_no_cycle
    )

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    x = np.arange(3)
    width = 0.35
    labels = ["|Δx|", "|Δy|", "|Δz|"]

    ax1.bar(x - width/2, cycle_delta_mean, width, yerr=cycle_delta_std,
            label=f"Cycle clips (n={len(sampled_cycle)})", color="#1f78b4", capsize=4, alpha=0.8)
    ax1.bar(x + width/2, no_cycle_delta_mean, width, yerr=no_cycle_delta_std,
            label=f"No-cycle clips (n={len(sampled_no_cycle)})", color="#e31a1c", capsize=4, alpha=0.8)

    ax1.set_xlabel("Axis", fontsize=12)
    ax1.set_ylabel("Mean |Δ| per frame", fontsize=12)
    ax1.set_title("XYZ Motion Comparison:\nCycle vs No-Cycle Clips", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.legend(fontsize=10, frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    xyz_path = os.path.join(output_dir, "xyz_motion_comparison.pdf")
    plt.savefig(xyz_path, dpi=300, bbox_inches="tight")
    plt.savefig(xyz_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved XYZ motion comparison to {xyz_path}")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    x2 = np.arange(len(body_parts))
    body_labels = [bp.replace("_", "\n") for bp in body_parts]

    ax2.bar(x2 - width/2, cycle_var, width,
            label=f"Cycle clips (n={len(sampled_cycle)})", color="#1f78b4", alpha=0.8)
    ax2.bar(x2 + width/2, no_cycle_var, width,
            label=f"No-cycle clips (n={len(sampled_no_cycle)})", color="#e31a1c", alpha=0.8)

    ax2.set_xlabel("Body Part", fontsize=12)
    ax2.set_ylabel("Variance", fontsize=12)
    ax2.set_title("Body Part Variance Comparison:\nCycle vs No-Cycle Clips", fontsize=14, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(body_labels, fontsize=9, rotation=45, ha="right")
    ax2.legend(fontsize=10, frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    variance_path = os.path.join(output_dir, "body_variance_comparison.pdf")
    plt.savefig(variance_path, dpi=300, bbox_inches="tight")
    plt.savefig(variance_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved body variance comparison to {variance_path}")

    summary_path = os.path.join(output_dir, "distribution_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Cycle vs No-Cycle Distribution Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Cycle clips sampled: {sampled_cycle}\n")
        f.write(f"No-cycle clips sampled: {sampled_no_cycle}\n\n")
        f.write("XYZ Motion (mean |Δ| per frame):\n")
        f.write("-" * 30 + "\n")
        for i, axis in enumerate(["x", "y", "z"]):
            f.write(f"  Δ{axis}: Cycle={cycle_delta_mean[i]:.4f}±{cycle_delta_std[i]:.4f}, "
                    f"No-cycle={no_cycle_delta_mean[i]:.4f}±{no_cycle_delta_std[i]:.4f}\n")
        f.write("\nBody Part Variance:\n")
        f.write("-" * 30 + "\n")
        for i, bp in enumerate(body_parts):
            f.write(f"  {bp}: Cycle={cycle_var[i]:.4f}, No-cycle={no_cycle_var[i]:.4f}\n")

    print(f"Saved distribution summary to {summary_path}")


def main(config_path: str = "configs/dmwkc_experiment/extract_filled_space.yaml"):
    cfg = load_config(config_path)

    if not cfg.dm_results_dir:
        raise ValueError("dm_results_dir must be specified in config")

    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"=== Finding clips with and without cycles ===")
    print(f"DM results dir: {cfg.dm_results_dir}")

    cycle_clips, no_cycle_clips = find_cycle_and_no_cycle_clips(cfg.dm_results_dir, cfg.num_clips)

    print(f"Found {len(cycle_clips)} clips WITH cycles")
    print(f"Found {len(no_cycle_clips)} clips WITHOUT cycles out of {cfg.num_clips}")
    print(f"No-cycle clip IDs: {no_cycle_clips}")

    with open(os.path.join(cfg.output_dir, "no_cycle_clip_ids.txt"), "w") as f:
        f.write(f"# Clips with no cycles from {cfg.dm_results_dir}\n")
        f.write(f"# Total: {len(no_cycle_clips)} / {cfg.num_clips}\n")
        for cid in no_cycle_clips:
            f.write(f"{cid}\n")

    with open(os.path.join(cfg.output_dir, "cycle_clip_ids.txt"), "w") as f:
        f.write(f"# Clips with cycles from {cfg.dm_results_dir}\n")
        f.write(f"# Total: {len(cycle_clips)} / {cfg.num_clips}\n")
        for cid in cycle_clips:
            f.write(f"{cid}\n")

    print(f"\n=== Computing distribution comparison ===")
    plot_cycle_vs_no_cycle_comparison(
        cycle_clips=cycle_clips,
        no_cycle_clips=no_cycle_clips,
        rodent_rollout_dir=cfg.rodent_rollout_dir,
        output_dir=cfg.output_dir,
        dist_cfg=cfg.distribution,
    )

    if len(no_cycle_clips) == 0:
        print("\nNo clips without cycles found. Skipping rendering.")
        return

    print(f"\n=== Rendering no-cycle clips ===")

    clips_per_video = cfg.grid_rows * cfg.grid_cols
    n_videos = (len(no_cycle_clips) + clips_per_video - 1) // clips_per_video

    for vid_idx in range(n_videos):
        start_idx = vid_idx * clips_per_video
        end_idx = min(start_idx + clips_per_video, len(no_cycle_clips))
        batch_clips = no_cycle_clips[start_idx:end_idx]

        out_path = os.path.join(cfg.output_dir, f"no_cycle_clips_batch_{vid_idx}.mp4")

        print(f"Rendering batch {vid_idx + 1}/{n_videos}: clips {batch_clips}")

        render_clip_grid(
            rodent_rollout_dir=cfg.rodent_rollout_dir,
            clip_ids=batch_clips,
            rows=cfg.grid_rows,
            cols=cfg.grid_cols,
            out_path=out_path,
            fps=cfg.fps,
            target_size=cfg.target_size,
            title=f"No-Cycle Clips (batch {vid_idx + 1}/{n_videos})",
            per_tile_labels=True,
            label_fmt="clip_{id}",
        )

        print(f"Saved: {out_path}")

    print(f"\n=== Done ===")
    print(f"Results saved to: {cfg.output_dir}")
    print(f"Total no-cycle clips: {len(no_cycle_clips)}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/demos/demo_stationary.yaml"
    main(config_path)
