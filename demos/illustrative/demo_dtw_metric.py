#!/usr/bin/env python3
"""
DTW Metric Visualization Demo
==============================
For each unique cluster in a benchmark method, renders a Nature-style figure
with the DTW cost matrix on the left and two chronophotography filmstrips
(discovered top, GT bottom) on the right.

Usage:
    python -m demos.illustrative.demo_dtw_metric
    python -m demos.illustrative.demo_dtw_metric --config configs/demos/illustrative/demo_dtw_metric.yaml
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import yaml
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from topo_mimic.analysis.feature_analysis.data_loading import (
    load_cluster_dict,
    load_labels_yaml,
)
from topo_mimic.analysis.feature_analysis.ground_truth import (
    find_all_transitions,
    TransitionSegment,
)
from topo_mimic.analysis.feature_analysis.behavior_parsing import (
    parse_cluster_label,
    parse_turn_direction,
    TurnDirection,
)
from topo_mimic.analysis.feature_analysis.dtw_visualization import compute_dtw_with_path


@dataclass
class DTWDemoConfig:
    experiment_dir: str = "outputs/benchmark_experiment_transitions"
    method: str = "w2_dmp_qpos_kmeans"
    seed: int = 0
    behavior_labels_csv: str = "behavior_labels.csv"
    data_path: str = "data/rodent_ar.h5"
    data_key: str = "qpos_rollout"
    xml_path: str = "topo_mimic/extraction/xml/rodent.xml"
    dataset: str = "rodent"
    output_dir: str = "outputs/dtw_metric"
    skip_root_dims: int = 7
    short_threshold: int = 50
    merge_rear: bool = True
    min_seq_length: int = 3
    n_filmstrip_poses: int = 8
    filmstrip_gap_fraction: float = 0.15
    dpi: int = 300


def load_config(config_path: str | None) -> DTWDemoConfig:
    if config_path is None or not os.path.exists(config_path):
        return DTWDemoConfig()
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return DTWDemoConfig(**{k: v for k, v in raw.items() if k in DTWDemoConfig.__dataclass_fields__})


def _load_data(data_path: str, data_key: str) -> np.ndarray:
    with h5py.File(data_path, "r") as f:
        return np.array(f[data_key])


def _render_filmstrip(
    qposes: np.ndarray,
    frame_indices: list[int],
    xml_path: str,
    dataset: str,
    n_poses: int = 8,
    gap_fraction: float = 0.15,
    tint_start: np.ndarray | None = None,
    tint_end: np.ndarray | None = None,
) -> np.ndarray:
    from demos.transition_demos.skeleton_filmstrip import render_filmstrip_for_frames
    return render_filmstrip_for_frames(
        qposes, frame_indices, xml_path, dataset,
        n_poses=n_poses, gap_fraction=gap_fraction,
        tint_start=tint_start, tint_end=tint_end,
    )


def render_dtw_panel(
    cost_matrix: np.ndarray,
    warping_path: list[tuple[int, int]],
    dtw_dist: float,
    disc_filmstrip: np.ndarray,
    gt_filmstrip: np.ndarray,
    title: str,
    output_path: str,
    dpi: int = 300,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
    })

    fig = plt.figure(figsize=(10, 4.5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1],
                           wspace=0.25, hspace=0.35)

    # Left: DTW cost matrix (spans both rows)
    ax_matrix = fig.add_subplot(gs[:, 0])
    cmap = LinearSegmentedColormap.from_list("cost", ["#2ECC71", "#F1C40F", "#E74C3C"], N=256)
    im = ax_matrix.imshow(cost_matrix, cmap=cmap, aspect="auto", origin="lower")
    if warping_path:
        path_j = [p[1] for p in warping_path]
        path_i = [p[0] for p in warping_path]
        ax_matrix.plot(path_j, path_i, "w-", linewidth=1.5, alpha=0.9)
    ax_matrix.set_xlabel("GT frames", fontsize=8)
    ax_matrix.set_ylabel("Discovered frames", fontsize=8)
    ax_matrix.set_title(f"DTW = {dtw_dist:.4f}", fontsize=9, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax_matrix, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=6)

    # Top-right: discovered filmstrip
    ax_disc = fig.add_subplot(gs[0, 1])
    ax_disc.imshow(disc_filmstrip)
    ax_disc.set_title("Discovered", fontsize=8, fontweight="bold", color="#2980b9")
    ax_disc.axis("off")

    # Bottom-right: GT filmstrip
    ax_gt = fig.add_subplot(gs[1, 1])
    ax_gt.imshow(gt_filmstrip)
    ax_gt.set_title("Ground Truth", fontsize=8, fontweight="bold", color="#c0392b")
    ax_gt.axis("off")

    fig.suptitle(title, fontsize=10, fontweight="bold", y=0.98)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="DTW metric visualization demo")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.method is not None:
        cfg.method = args.method
    if args.seed is not None:
        cfg.seed = args.seed

    output_dir = Path(cfg.output_dir) / cfg.method / f"seed_{cfg.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    method_dir = os.path.join(cfg.experiment_dir, cfg.method)
    print(f"Loading clusters from {method_dir} (seed {cfg.seed})...")

    cluster_dict = load_cluster_dict(method_dir, cfg.seed)
    cluster_labels, turn_directions = load_labels_yaml(method_dir, cfg.seed)

    if not cluster_labels:
        cluster_labels = {i: f"C{i}" for i in cluster_dict.keys()}
    if turn_directions is None:
        turn_directions = {}

    print(f"  {len(cluster_dict)} clusters, {len(cluster_labels)} labels")

    print(f"Loading data from {cfg.data_path}...")
    data = _load_data(cfg.data_path, cfg.data_key)
    data_for_dtw = data[..., cfg.skip_root_dims:] if cfg.skip_root_dims > 0 else data

    print("Loading GT transitions...")
    all_transitions = find_all_transitions(
        cfg.behavior_labels_csv,
        short_threshold=cfg.short_threshold,
        merge_rear=cfg.merge_rear,
    )

    # Group clusters by (label, turn)
    grouped = defaultdict(list)
    for cid, label in cluster_labels.items():
        if not label or label.startswith("C"):
            continue
        parsed = parse_cluster_label(label, merge_rear=cfg.merge_rear)
        if parsed is None or parsed.comparison_type != "transition":
            continue
        if len(parsed.primitives) < 2:
            continue
        target = tuple(sorted(parsed.primitives[:2]))
        turn = turn_directions.get(cid)
        grouped[(target, turn)].append(cid)

    print(f"\n{len(grouped)} unique transition groups to visualize")

    for (target_behaviors, turn_str), cluster_ids in grouped.items():
        turn_dir = parse_turn_direction(turn_str)
        label_str = f"{target_behaviors[0]}↔{target_behaviors[1]}"
        turn_suffix = f" [{turn_str}]" if turn_str else ""
        safe_name = label_str.replace("↔", "_").replace(" ", "_")
        safe_turn = f"_{turn_str}" if turn_str else ""

        # Get discovered entries
        disc_entries = []
        for cid in cluster_ids:
            disc_entries.extend(cluster_dict.get(cid, []))
        if not disc_entries:
            continue

        disc_clip_id, disc_frame_arr = disc_entries[0]
        disc_clip_id = int(disc_clip_id)
        disc_frames = np.sort(np.asarray(disc_frame_arr).flatten())
        valid_disc = disc_frames[(disc_frames >= 0) & (disc_frames < data.shape[1])]
        if len(valid_disc) < cfg.min_seq_length:
            continue

        # Get GT
        gt_transitions = [t for t in all_transitions if t.transition_key == target_behaviors]
        if turn_dir not in (None, TurnDirection.STRAIGHT, TurnDirection.NON_DETERMINISTIC):
            gt_transitions = [t for t in gt_transitions
                              if t.from_turn == turn_dir or t.to_turn == turn_dir]
        if not gt_transitions:
            continue

        gt_t = gt_transitions[0]
        gt_frames = np.sort(np.array(gt_t.all_frames))
        valid_gt = gt_frames[(gt_frames >= 0) & (gt_frames < data.shape[1])]
        if len(valid_gt) < cfg.min_seq_length:
            continue

        # Compute DTW
        disc_seq = data_for_dtw[disc_clip_id, valid_disc]
        gt_seq = data_for_dtw[gt_t.clip_id, valid_gt]
        dtw_dist, cost_matrix, warping_path = compute_dtw_with_path(disc_seq, gt_seq)

        # Render filmstrips
        disc_qpos = data[disc_clip_id]
        gt_qpos = data[gt_t.clip_id]

        print(f"  {label_str}{turn_suffix}: DTW={dtw_dist:.4f}, rendering...")

        try:
            disc_film = _render_filmstrip(
                disc_qpos, valid_disc.tolist(), cfg.xml_path, cfg.dataset,
                n_poses=cfg.n_filmstrip_poses, gap_fraction=cfg.filmstrip_gap_fraction,
                tint_start=np.array([0.55, 0.70, 0.90]),
                tint_end=np.array([0.15, 0.30, 0.60]),
            )
            gt_film = _render_filmstrip(
                gt_qpos, valid_gt.tolist(), cfg.xml_path, cfg.dataset,
                n_poses=cfg.n_filmstrip_poses, gap_fraction=cfg.filmstrip_gap_fraction,
                tint_start=np.array([0.90, 0.60, 0.55]),
                tint_end=np.array([0.60, 0.20, 0.15]),
            )
        except Exception as e:
            print(f"    Filmstrip rendering failed: {e}")
            continue

        for ext in ("pdf", "png"):
            out_path = str(output_dir / f"{safe_name}{safe_turn}.{ext}")
            render_dtw_panel(
                cost_matrix, warping_path, dtw_dist,
                disc_film, gt_film,
                title=f"{label_str}{turn_suffix}  (C{cluster_ids[0]}, clip {disc_clip_id}) vs GT (clip {gt_t.clip_id})",
                output_path=out_path,
                dpi=cfg.dpi,
            )
            print(f"    Saved {out_path}")

    print(f"\nDone. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
