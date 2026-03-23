#!/usr/bin/env python3
"""
Cluster Filmstrip Grid
======================
For each unique labeled cluster in a benchmark method, renders a grid of
chronophotography filmstrips (grey-box style). Each row is one cluster label,
each column is a different sample from that cluster.

Usage:
    python -m demos.illustrative.demo_cluster_filmstrips
    python -m demos.illustrative.demo_cluster_filmstrips --config configs/demos/illustrative/demo_cluster_filmstrips.yaml
    python -m demos.illustrative.demo_cluster_filmstrips --method w2_dmc_int_kmeans
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
import math
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import yaml
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from topo_mimic.benchmark.analysis.data_loading import (
    load_cluster_dict,
    load_labels_yaml,
)
from topo_mimic.benchmark.analysis.behavior_parsing import (
    parse_cluster_label,
)


@dataclass
class ClusterFilmstripConfig:
    experiment_dir: str = "outputs/benchmark_experiment_transitions"
    method: str = "w2_dmp_qpos_kmeans"
    seed: int = 0
    data_path: str = "data/rodent_ar.h5"
    data_key: str = "qpos_rollout"
    xml_path: str = "topo_mimic/dmwk/xml/rodent.xml"
    dataset: str = "rodent"
    output_dir: str = "outputs/cluster_filmstrips"
    k_samples: int = 5
    n_poses: int = 6
    filmstrip_gap_fraction: float = 0.25
    dpi: int = 300


def load_config(path: str | None) -> ClusterFilmstripConfig:
    if path is None or not os.path.exists(path):
        return ClusterFilmstripConfig()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return ClusterFilmstripConfig(**{
        k: v for k, v in raw.items() if k in ClusterFilmstripConfig.__dataclass_fields__
    })


def _render_filmstrip(
    qposes: np.ndarray,
    frame_indices: list[int],
    xml_path: str,
    dataset: str,
    n_poses: int = 6,
    gap_fraction: float = 0.25,
) -> np.ndarray:
    from demos.transition_demos.skeleton_filmstrip import render_filmstrip_for_frames
    return render_filmstrip_for_frames(
        qposes, frame_indices, xml_path, dataset,
        n_poses=n_poses, gap_fraction=gap_fraction,
    )


def _group_clusters_by_label(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    turn_directions: dict[int, str] | None,
    merge_rear: bool = True,
) -> dict[str, list[tuple[int, int, np.ndarray]]]:
    """Group cluster entries by their human-assigned label + turn.

    Returns: {display_label: [(clip_id, cluster_id, frame_indices), ...]}
    """
    if turn_directions is None:
        turn_directions = {}

    grouped = defaultdict(list)
    for cid, label in cluster_labels.items():
        if not label or label.startswith("C"):
            continue
        turn = turn_directions.get(cid, "")
        display = f"{label} [{turn}]" if turn else label

        for clip_id, frames in cluster_dict.get(cid, []):
            grouped[display].append((int(clip_id), cid, np.sort(np.asarray(frames).flatten())))

    return dict(sorted(grouped.items()))


def render_cluster_grid(
    grouped: dict[str, list[tuple[int, int, np.ndarray]]],
    data: np.ndarray,
    cfg: ClusterFilmstripConfig,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    xml_path = cfg.xml_path
    if not os.path.isabs(xml_path):
        xml_path = str(project_root / xml_path)

    n_rows = len(grouped)
    n_cols = cfg.k_samples
    labels = list(grouped.keys())

    box_color = "#e8e8e8"
    box_edge = "#b0b0b0"
    dpi = cfg.dpi
    panel_w_in = 2.2
    bx_pad = 0.03
    by_pad = 0.03
    label_h = 0.22
    gap_x = 0.08
    gap_y = 0.08

    print(f"Rendering {n_rows} rows x {n_cols} columns...")

    all_strips = {}
    target_w = int(panel_w_in * dpi)

    for label in labels:
        entries = grouped[label]
        samples = entries[:n_cols]
        strips = []
        for clip_id, cid, frames in samples:
            valid = frames[(frames >= 0) & (frames < data.shape[1])]
            if len(valid) < 2:
                strips.append(np.full((60, target_w, 3), 232, dtype=np.uint8))
                continue
            try:
                qposes = data[clip_id]
                strip = _render_filmstrip(
                    qposes, valid.tolist(), xml_path, cfg.dataset,
                    n_poses=cfg.n_poses, gap_fraction=cfg.filmstrip_gap_fraction,
                )
                oh, ow = strip.shape[:2]
                scale = target_w / ow
                nh = max(1, int(oh * scale))
                strip = np.array(Image.fromarray(strip).resize((target_w, nh), Image.LANCZOS))
                strips.append(strip)
            except Exception as e:
                print(f"  Skipping {label} clip {clip_id}: {e}")
                strips.append(np.full((60, target_w, 3), 232, dtype=np.uint8))

        while len(strips) < n_cols:
            strips.append(np.full((60, target_w, 3), 255, dtype=np.uint8))

        all_strips[label] = strips

    max_h_px = max(
        max(s.shape[0] for s in strips)
        for strips in all_strips.values()
    )
    max_h_in = max_h_px / dpi

    box_w = panel_w_in + 2 * bx_pad
    box_h = max_h_in + 2 * by_pad + label_h

    row_label_w = 2.0
    fig_w = row_label_w + n_cols * box_w + (n_cols - 1) * gap_x + 0.15
    fig_h = n_rows * box_h + (n_rows - 1) * gap_y + 0.15

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_axis_off()
    ax.set_aspect("equal")

    box_rgb = tuple(int(box_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

    for row_idx, label in enumerate(labels):
        strips = all_strips[label]
        y0 = fig_h - 0.05 - (row_idx + 1) * box_h - row_idx * gap_y

        ax.text(
            row_label_w - 0.08, y0 + box_h / 2,
            label, ha="right", va="center",
            fontsize=5.5, fontweight="bold", color="#333333",
            zorder=2,
        )

        for col_idx, img in enumerate(strips[:n_cols]):
            x0 = row_label_w + col_idx * (box_w + gap_x)

            rect = FancyBboxPatch(
                (x0, y0), box_w, box_h,
                boxstyle="round,pad=0.01,rounding_size=0.04",
                facecolor=box_color, edgecolor=box_edge, linewidth=0.5, zorder=0,
            )
            ax.add_patch(rect)

            white_mask = np.all(img >= 245, axis=-1)
            img[white_mask] = box_rgb

            ih_in = img.shape[0] / dpi
            iw_in = img.shape[1] / dpi
            img_x = x0 + bx_pad
            center_y = y0 + by_pad + label_h + max_h_in / 2
            extent = [img_x, img_x + iw_in, center_y - ih_in / 2, center_y + ih_in / 2]
            ax.imshow(img, extent=extent, aspect="auto", zorder=1)

            entry = grouped[label][col_idx] if col_idx < len(grouped[label]) else None
            if entry is not None:
                clip_id, cid, _ = entry
                ax.text(
                    x0 + box_w / 2, y0 + by_pad + label_h * 0.4,
                    f"C{cid}, clip {clip_id}",
                    ha="center", va="center",
                    fontsize=4, color="#555555", zorder=2,
                )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_dir = Path(cfg.output_dir) / cfg.method / f"seed_{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf", "png"):
        out_path = str(out_dir / f"cluster_filmstrip_grid.{ext}")
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.02)
        print(f"  Saved {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Cluster filmstrip grid")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    default_cfg = Path(__file__).resolve().parents[2] / "configs/demos/illustrative/demo_cluster_filmstrips.yaml"
    cfg = load_config(args.config or (str(default_cfg) if default_cfg.exists() else None))
    if args.method is not None:
        cfg.method = args.method
    if args.seed is not None:
        cfg.seed = args.seed

    method_dir = os.path.join(cfg.experiment_dir, cfg.method)
    print(f"Loading {cfg.method} seed {cfg.seed}...")

    cluster_dict = load_cluster_dict(method_dir, cfg.seed)
    cluster_labels, turn_directions = load_labels_yaml(method_dir, cfg.seed)
    if not cluster_labels:
        cluster_labels = {i: f"C{i}" for i in cluster_dict.keys()}

    print(f"  {len(cluster_dict)} clusters, {len(cluster_labels)} labels")

    print(f"Loading data from {cfg.data_path}...")
    with h5py.File(cfg.data_path, "r") as f:
        data = np.array(f[cfg.data_key])

    grouped = _group_clusters_by_label(cluster_dict, cluster_labels, turn_directions)
    print(f"  {len(grouped)} unique cluster groups")

    render_cluster_grid(grouped, data, cfg)
    print("Done.")


if __name__ == "__main__":
    main()
