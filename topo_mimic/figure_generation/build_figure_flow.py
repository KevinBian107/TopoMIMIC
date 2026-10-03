#!/usr/bin/env python3
"""Build paper/figures/figure1.png: the method data-flow overview (Figure 1).

  trajectory T_i --Step 1--> DM skeleton G_i
       skeleton --Step 2 decompose--> DM-path (split tree)  and  DM-cycle (loop)
       path + cycle --converge--> [ W2 + K-means ] --> behaviour clusters

DM-path / DM-cycle / skeleton come from dm_primitives (the corrected primitives);
the input is trajectory.png and the clusters are freshly-rendered filmstrips.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from topo_mimic.figure_generation.dm_primitives import (  # noqa: E402
    render_dm_cycle, render_dm_path_2d, render_skeleton)

TRAJ = Path(__file__).resolve().parent / "external_data/trajectory.png"
OUT = REPO / "paper/figures/figure1.png"


def _img(p) -> np.ndarray:
    img = Image.open(p) if not isinstance(p, Image.Image) else p
    return np.asarray(img.convert("RGB"))


def _render_cluster_filmstrips(n: int = 3, method: str = "w2_dmp_qpos_kmeans",
                               seed: int = 0) -> list[tuple[str, np.ndarray]]:
    """Fresh, label-free filmstrips for a few DISTINCT discovered clusters, using
    the demo_cluster_filmstrips renderer. Returns (transition_name, filmstrip)."""
    import re

    import h5py

    from demos.demo_cluster_filmstrips import _group_clusters_by_label, _render_filmstrip
    from topo_mimic.analysis.feature_analysis.data_loading import (
        load_cluster_dict, load_labels_yaml, recenter_cluster_dict)
    from topo_mimic.experiment.auto_label import (
        load_behavior_segments, split_segments_by_kind)

    method_dir = str(REPO / "outputs/experiment/transitions" / method)
    cluster_dict = recenter_cluster_dict(load_cluster_dict(method_dir, seed), method)
    cluster_labels, turn_dirs = load_labels_yaml(method_dir, seed)
    with h5py.File(REPO / "data/rodent_ar.h5", "r") as f:
        data = np.array(f["qpos_rollout"])
    behav_segs, dir_segs = split_segments_by_kind(
        load_behavior_segments(str(REPO / "behavior_labels/rodent_labels.csv")))
    grouped = _group_clusters_by_label(cluster_dict, cluster_labels, turn_dirs,
                                       behav_segs=behav_segs, dir_segs=dir_segs)
    xml = str(REPO / "topo_mimic/extraction/xml/rodent.xml")

    out, seen = [], set()
    for label in grouped:
        base = re.sub(r"\s+(Left|Right|Straight)$", "", label)
        if base in seen or "->" not in base:
            continue
        seen.add(base)
        clip_id, _cid, frames, _score = sorted(grouped[label], key=lambda x: -x[3])[0]
        frame_idx = list(np.asarray(frames).astype(int))
        film = np.asarray(_render_filmstrip(data[int(clip_id)], frame_idx, xml, "rodent"))
        out.append((base.replace("->", r"$\to$"), film))
        if len(out) >= n:
            break
    return out


def _imgpanel(fig, rect, img, label=None, loff: float = -0.05) -> None:
    ax = fig.add_axes(rect)
    ax.imshow(img)
    ax.axis("off")
    if label:
        # loff is the label's vertical offset below the panel (axes fraction);
        # a larger gap keeps the caption clear of the drawing's lowest node.
        ax.text(0.5, loff, label, ha="center", va="top", fontsize=24,
                transform=ax.transAxes)


def _arrow(fig, p0, p1, color="#333") -> None:
    fig.add_artist(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18,
                                   lw=1.9, color=color))


def _line(fig, p0, p1, color="#333") -> None:
    fig.add_artist(FancyArrowPatch(p0, p1, arrowstyle="-", lw=1.9, color=color))


def main() -> None:
    print("rendering panels...")
    traj = _img(TRAJ)
    skel = _img(render_skeleton(900))
    cyc = _img(render_dm_cycle(900))
    pth = _img(render_dm_path_2d(900, labels=False))
    clusters = _render_cluster_filmstrips()

    fig = plt.figure(figsize=(20, 5.0))

    _imgpanel(fig, [0.005, 0.33, 0.105, 0.40], traj, "$T_i$: behavior trajectory")
    _imgpanel(fig, [0.165, 0.30, 0.105, 0.44], skel, "skeleton $G_i$", loff=-0.10)
    # DM-cycle / DM-path drawings raised and given a wider label gap so the
    # skeleton's lowest node no longer crowds the "DM-cycle" / "DM-path" caption,
    # while leaving room for the "DM-cycle" caption between the two drawings.
    _imgpanel(fig, [0.315, 0.62, 0.135, 0.35], cyc, "DM-cycle", loff=-0.10)
    _imgpanel(fig, [0.315, 0.15, 0.135, 0.35], pth, "DM-path", loff=-0.12)

    # cluster filmstrips -- transition name beneath. Vertical spacing is wide
    # enough that the (enlarged) labels do not touch the neighbouring filmstrip.
    cl_x, cl_w = 0.52, 0.20
    cl_h = 0.115
    cl_gap = 0.30  # center-to-center spacing; symmetric about the fan origin (oy=0.50)
    cl_y = [0.50 + cl_gap - cl_h / 2, 0.50 - cl_h / 2, 0.50 - cl_gap - cl_h / 2]
    cl_axes = []
    for (lab, film), y in zip(clusters, cl_y):
        ax = fig.add_axes([cl_x, y, cl_w, cl_h])
        ax.imshow(film)
        ax.axis("off")
        ax.text(0.5, -0.12, lab, transform=ax.transAxes, ha="center", va="top",
                fontsize=19, color="#333")
        cl_axes.append((ax, lab, y + cl_h / 2))

    fw, fh = fig.get_size_inches()

    # Step 1
    _arrow(fig, (0.115, 0.52), (0.16, 0.52))
    # Step 2: skeleton -> cycle (up) and -> path (down, after removing the cycle)
    _arrow(fig, (0.275, 0.60), (0.31, 0.72))
    _arrow(fig, (0.275, 0.44), (0.31, 0.28))
    # short fan arrows from just right of the two graphs to the clusters
    ox, oy = 0.47, 0.50
    for _ax, _lab, cy in cl_axes:
        _arrow(fig, (ox, oy), (cl_x - 0.004, cy))
    # Step captions on one common baseline along the bottom of the figure.
    y_step = 0.04
    fig.text(0.13, y_step, "Step 1\nTakens + DM-G", ha="center", va="bottom", fontsize=20)
    fig.text(0.25, y_step, "Step 2\ndecompose", ha="center", va="bottom", fontsize=20)
    fig.text(0.50, y_step, "Step 3\ncluster", ha="center", va="bottom", fontsize=20)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {OUT}  (size {Image.open(OUT).size})")


if __name__ == "__main__":
    main()
