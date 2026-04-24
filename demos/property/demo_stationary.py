"""
Stationary Clip Analysis (Appendix: DM-G Identifies Stationary Clips)

Compares clips with DM-G cycles (behaviorally rich) vs clips without cycles
(stationary/simple topology). Produces Nature-style bar plots and a filmstrip
grid of representative no-cycle clips.

Usage:
    python -m demos.property.demo_stationary
    python -m demos.property.demo_stationary --config configs/demos/property/demo_stationary.yaml
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import math
import argparse
import pickle as pkl
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

from topo_mimic.extraction.utils.load_utils import load_rollout_from_source


NATURE_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'axes.linewidth': 0.6,
    'axes.labelsize': 8, 'axes.titlesize': 9,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
}

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


@dataclass
class DistributionConfig:
    n_cycle_clips: int = 50
    n_no_cycle_clips: int = 50
    body_parts: list[str] = field(default_factory=lambda: [
        "torso", "pelvis", "upper_leg_L", "lower_leg_L", "foot_L",
        "upper_leg_R", "lower_leg_R", "foot_R", "skull",
        "upper_arm_L", "upper_arm_R",
    ])


@dataclass
class Config:
    dm_results_dir: str = "outputs/dm_pipeline_intention"
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar.h5"
    xml_path: str = "topo_mimic/extraction/xml/rodent.xml"
    output_dir: str = "outputs/stationary_clips"
    num_clips: int = 500
    filmstrip_n_clips: int = 8
    filmstrip_n_poses: int = 6
    filmstrip_sample_every: int = 80
    distribution: DistributionConfig = field(default_factory=DistributionConfig)


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dist_dict = cfg.get("distribution", {})
    dist_cfg = DistributionConfig(**dist_dict)

    return Config(
        dm_results_dir=cfg.get("dm_results_dir", Config.dm_results_dir),
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", Config.rodent_rollout_dir),
        xml_path=cfg.get("xml_path", Config.xml_path),
        output_dir=cfg.get("output_dir", Config.output_dir),
        num_clips=cfg.get("num_clips", Config.num_clips),
        filmstrip_n_clips=cfg.get("filmstrip_n_clips", Config.filmstrip_n_clips),
        filmstrip_n_poses=cfg.get("filmstrip_n_poses", Config.filmstrip_n_poses),
        filmstrip_sample_every=cfg.get("filmstrip_sample_every", Config.filmstrip_sample_every),
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


def compute_delta_xyz_stats(
    clip_ids: list[int],
    rodent_rollout_dir: str,
    max_clips: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    all_deltas = []
    for i, clip_id in enumerate(clip_ids):
        if i >= max_clips:
            break
        try:
            roll = load_rollout_from_source(int(clip_id), rodent_rollout_dir)
            qpos = np.asarray(roll["qpos"])
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
    max_clips: int = 50,
) -> tuple[np.ndarray, list[str]]:
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
            roll = load_rollout_from_source(int(clip_id), rodent_rollout_dir)
            qpos = np.asarray(roll["qpos"])
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

    print(f"\nSampled {len(sampled_cycle)} cycle clips, {len(sampled_no_cycle)} no-cycle clips")

    cycle_delta_mean, cycle_delta_std = compute_delta_xyz_stats(sampled_cycle, rodent_rollout_dir, n_cycle)
    no_cycle_delta_mean, no_cycle_delta_std = compute_delta_xyz_stats(sampled_no_cycle, rodent_rollout_dir, n_no_cycle)

    cycle_var, body_parts = compute_body_variance(sampled_cycle, rodent_rollout_dir, dist_cfg.body_parts, n_cycle)
    no_cycle_var, _ = compute_body_variance(sampled_no_cycle, rodent_rollout_dir, dist_cfg.body_parts, n_no_cycle)

    with plt.rc_context(NATURE_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8),
                                        gridspec_kw={'width_ratios': [1, 2.2]})

        # Panel A: XYZ motion
        x = np.arange(3)
        width = 0.32
        labels = ["|Δx|", "|Δy|", "|Δz|"]

        ax1.bar(x - width/2, cycle_delta_mean, width, yerr=cycle_delta_std,
                label=f"With cycles (n={len(sampled_cycle)})",
                color='#4878CF', capsize=2.5, edgecolor='none',
                error_kw={'linewidth': 0.7, 'capthick': 0.7})
        ax1.bar(x + width/2, no_cycle_delta_mean, width, yerr=no_cycle_delta_std,
                label=f"No cycles (n={len(sampled_no_cycle)})",
                color='#C44E52', capsize=2.5, edgecolor='none',
                error_kw={'linewidth': 0.7, 'capthick': 0.7})

        ax1.set_xlabel("Root displacement axis")
        ax1.set_ylabel("Mean |Δ| per frame")
        ax1.set_title("Root motion", fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend(fontsize=5.5, frameon=True, framealpha=0.9, edgecolor='#cccccc')
        for spine in ['top', 'right']:
            ax1.spines[spine].set_visible(False)
        ax1.tick_params(direction='out', length=2.5, width=0.5)

        # Panel B: Body part variance
        x2 = np.arange(len(body_parts))
        body_labels = [bp.replace("_", "\n") for bp in body_parts]

        ax2.bar(x2 - width/2, cycle_var, width,
                label=f"With cycles", color='#4878CF', edgecolor='none')
        ax2.bar(x2 + width/2, no_cycle_var, width,
                label=f"No cycles", color='#C44E52', edgecolor='none')

        ax2.set_xlabel("Body part")
        ax2.set_ylabel("Variance")
        ax2.set_title("Body part variance", fontweight='bold')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(body_labels, fontsize=5.5, rotation=45, ha="right")
        ax2.legend(fontsize=5.5, frameon=True, framealpha=0.9, edgecolor='#cccccc')
        for spine in ['top', 'right']:
            ax2.spines[spine].set_visible(False)
        ax2.tick_params(direction='out', length=2.5, width=0.5)

        plt.tight_layout()
        out_path = os.path.join(output_dir, "stationary_comparison.pdf")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.savefig(out_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
    print(f"Saved: {out_path}")


def render_filmstrip_grid(
    clip_ids: list[int],
    rodent_rollout_dir: str,
    xml_path: str,
    output_path: str,
    n_poses: int = 6,
    sample_every: int = 80,
    grid_cols: int = 4,
) -> None:
    from demos.illustrative.demo_bodies import (
        _render_poses_individually,
        _tile_cropped_poses,
    )
    from matplotlib.patches import FancyBboxPatch

    project_root = Path(__file__).parent.parent.parent.resolve()
    if not os.path.isabs(xml_path):
        xml_path = str(project_root / xml_path)

    strips = []
    valid_ids = []
    for clip_id in clip_ids:
        try:
            roll = load_rollout_from_source(int(clip_id), rodent_rollout_dir)
            qposes = np.asarray(roll["qpos"])
            n_total = min(500, qposes.shape[0])
            frame_indices = list(range(0, n_total, sample_every))[:n_poses]

            fovy_rad = math.radians(45.0)
            view_h = 0.35
            distance = (view_h / 2.0) / math.tan(fovy_rad / 2.0) * 1.3

            def rodent_lookat(qpos):
                return [qpos[0], qpos[1], qpos[2] + 0.02]

            cropped = _render_poses_individually(
                qposes, frame_indices, xml_path,
                camera_elevation=-55.0, camera_azimuth=90.0,
                camera_distance=distance, camera_lookat_fn=rodent_lookat,
                render_h=720, render_w=720,
            )
            strip = _tile_cropped_poses(cropped, gap_fraction=0.25)
            strips.append(strip)
            valid_ids.append(clip_id)
        except Exception as e:
            print(f"  Skipping clip {clip_id}: {e}")
            continue

    if not strips:
        print("No filmstrips rendered")
        return

    grid_rows = math.ceil(len(strips) / grid_cols)
    box_color = '#e8e8e8'
    box_edge = '#b0b0b0'
    dpi = 300
    panel_w_in = 2.5
    bx_pad = 0.03
    by_pad = 0.03
    label_h = 0.2
    gap_x = 0.1
    gap_y = 0.1
    target_w = int(panel_w_in * dpi)

    from PIL import Image as PILImage
    resized = []
    for s in strips:
        oh, ow = s.shape[:2]
        scale = target_w / ow
        nh = int(oh * scale)
        resized.append(np.array(PILImage.fromarray(s).resize((target_w, nh), PILImage.LANCZOS)))

    box_rgb = tuple(int(box_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    for i, img in enumerate(resized):
        white_mask = np.all(img >= 245, axis=-1)
        img[white_mask] = box_rgb
        resized[i] = img

    max_h_in = max(img.shape[0] for img in resized) / dpi
    box_w = panel_w_in + 2 * bx_pad
    box_h = max_h_in + 2 * by_pad + label_h

    fig_w = grid_cols * box_w + (grid_cols - 1) * gap_x + 0.1
    fig_h = grid_rows * box_h + (grid_rows - 1) * gap_y + 0.1

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        fig.patch.set_facecolor('white')
        ax.set_xlim(0, fig_w)
        ax.set_ylim(0, fig_h)
        ax.set_axis_off()
        ax.set_aspect('equal')

        for idx, (clip_id, img) in enumerate(zip(valid_ids, resized)):
            row = idx // grid_cols
            col = idx % grid_cols

            iw_in = img.shape[1] / dpi
            ih_in = img.shape[0] / dpi

            x0 = 0.05 + col * (box_w + gap_x)
            y0 = fig_h - 0.05 - (row + 1) * box_h - row * gap_y

            rect = FancyBboxPatch(
                (x0, y0), box_w, box_h,
                boxstyle='round,pad=0.01,rounding_size=0.04',
                facecolor=box_color, edgecolor=box_edge, linewidth=0.6, zorder=0,
            )
            ax.add_patch(rect)

            img_x = x0 + bx_pad
            center_y = y0 + by_pad + label_h + max_h_in / 2
            extent = [img_x, img_x + iw_in, center_y - ih_in / 2, center_y + ih_in / 2]
            ax.imshow(img, extent=extent, aspect='auto', zorder=1)

            ax.text(x0 + box_w / 2, y0 + by_pad + label_h * 0.45,
                    f'Clip {clip_id} (no cycles)', ha='center', va='center',
                    fontsize=5.5, fontweight='bold', color='#333333', zorder=2)

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white',
                    pad_inches=0.02)
        plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stationary Clip Analysis (Appendix)")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent.resolve()
    default_config = project_root / "configs" / "demos" / "property" / "demo_stationary.yaml"

    if args.config:
        cfg = load_config(args.config)
    elif default_config.exists():
        print(f"Loading config from {default_config}")
        cfg = load_config(str(default_config))
    else:
        cfg = Config()

    if not os.path.isabs(cfg.dm_results_dir):
        cfg.dm_results_dir = str(project_root / cfg.dm_results_dir)
    if not os.path.isabs(cfg.output_dir):
        cfg.output_dir = str(project_root / cfg.output_dir)
    if not os.path.isabs(cfg.xml_path):
        cfg.xml_path = str(project_root / cfg.xml_path)

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print("Stationary Clip Analysis (Appendix)")
    print("=" * 60)
    print(f"  DM results: {cfg.dm_results_dir}")
    print(f"  Output: {cfg.output_dir}")

    print("\n=== Finding clips with/without cycles ===")
    cycle_clips, no_cycle_clips = find_cycle_and_no_cycle_clips(cfg.dm_results_dir, cfg.num_clips)
    print(f"  {len(cycle_clips)} clips WITH cycles")
    print(f"  {len(no_cycle_clips)} clips WITHOUT cycles")

    with open(os.path.join(cfg.output_dir, "no_cycle_clip_ids.txt"), "w") as f:
        for cid in no_cycle_clips:
            f.write(f"{cid}\n")

    print("\n=== Distribution comparison ===")
    plot_cycle_vs_no_cycle_comparison(
        cycle_clips, no_cycle_clips,
        cfg.rodent_rollout_dir, cfg.output_dir,
        cfg.distribution,
    )

    print("\n=== Filmstrip grid of no-cycle clips ===")
    np.random.seed(42)
    n_film = min(cfg.filmstrip_n_clips, len(no_cycle_clips))
    film_clips = np.random.choice(no_cycle_clips, n_film, replace=False).tolist() if len(no_cycle_clips) >= n_film else no_cycle_clips
    print(f"  Rendering {len(film_clips)} clips: {film_clips}")

    render_filmstrip_grid(
        film_clips, cfg.rodent_rollout_dir, cfg.xml_path,
        os.path.join(cfg.output_dir, "stationary_filmstrip_grid.png"),
        n_poses=cfg.filmstrip_n_poses,
        sample_every=cfg.filmstrip_sample_every,
    )

    print(f"\nAll outputs saved to: {cfg.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
