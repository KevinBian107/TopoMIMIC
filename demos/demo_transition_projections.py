"""
Per-transition projection with DM-path overlays.

For each of the top-N GT transition types (e.g., Rear<->Walking, Immobile<->Turn):

  1. Find clips dominated by the transition: frames labeled A or B cover at
     least ``majority_threshold`` of the clip.
  2. Pool the real frames from those clips, fit PCA / UMAP / t-SNE, color
     blue (behavior A) vs red (behavior B) vs grey (other).
  3. Load ``all_path_dm.pkl`` for the chosen feature space, filter to paths
     that live in the dominated clips, take the K longest. Project the
     path vertices through the same 2D fit.
  4. Render one image per projection method. Each image: N rows x 5 cols.

     Row = one transition type.
     Cols (K=2 default):
       0: Real frames only (blue/red/grey).
       1: Real as grey background + DM path 1.
       2: Real as grey background + DM path 2.
       3: Real as grey background + all K paths + minima (endpoint) markers.
       4: Paths only, colored by temporal index along the path.

     For K paths other than 2, the column count is K + 3.

Usage:
    python -m demos.demo_transition_projections \\
           --config configs/demos/demo_transition_projections.yaml
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from topo_mimic.analysis.feature_analysis.ground_truth import find_all_transitions
from topo_mimic.extraction.utils.feature_transforms import (
    apply_dm_key_transform,
    get_hdf5_key_for_dm_key,
)


A_COLOR = "#1F4E79"   # Nature blue
B_COLOR = "#B22222"   # Nature red
OTHER_COLOR = (0.82, 0.82, 0.82, 0.55)
PATH_PALETTE = [
    "#2CA02C",  # green — path 1
    "#FF7F0E",  # orange — path 2
    "#9467BD",  # purple — path 3
    "#17BECF",  # teal — path 4
    "#8C564B",  # brown — path 5
]

PROJECTION_DISPLAY = {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}


@dataclass
class DemoConfig:
    h5_path: str = "data/rodent_ar.h5"
    behavior_labels_csv: str = "behavior_labels/rodent_labels.csv"
    xml_path: str = "topo_mimic/extraction/xml/rodent.xml"
    stac_data_path: str = "data/rodent_reference_clips.h5"
    feature_space: str = "ekp"
    extraction_dir: str = "outputs/extraction/dm_pipeline_ekp"
    seed: int = 0
    majority_threshold: float = 0.5
    min_per_behavior_fraction: float = 0.15
    n_transitions: int = 4
    k_paths: int = 2
    real_subsample_n: int = 4000
    tsne_subsample_n: int = 3000
    projections: list[str] = field(default_factory=lambda: ["pca", "umap", "tsne"])
    skip_clips: list[int] = field(default_factory=list)
    merge_rear: bool = True
    output_dir: str = "outputs/demos/illustrative/transition_projections"


def load_config(path: str | None) -> DemoConfig:
    cfg = DemoConfig()
    if not path or not os.path.exists(path):
        return cfg
    with open(path) as f:
        blob = yaml.safe_load(f) or {}
    for k, v in blob.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def parse_clip_id(video: str) -> int | None:
    name = Path(str(video)).stem
    if name.startswith("clip_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return None
    return None


def load_feature_data(cfg: DemoConfig) -> np.ndarray:
    hdf5_key = get_hdf5_key_for_dm_key(cfg.feature_space)
    with h5py.File(cfg.h5_path, "r") as f:
        raw = np.asarray(f[hdf5_key][:])
    return apply_dm_key_transform(
        raw, cfg.feature_space,
        xml_path=cfg.xml_path, stac_data_path=cfg.stac_data_path,
    )


def load_behavior_segments(csv_path: str) -> dict[int, list[tuple[str, int, int]]]:
    """Return {clip_id: [(behavior, start_frame, end_frame), ...]} (exclusive end).

    Only the four primitive behaviors are kept (Immobile, Turn, Rear, Walk).
    Direction labels (Left, Right, Straight) are dropped.
    """
    PRIM = {"Immobile", "Turn", "Rear", "Walk"}
    df = pd.read_csv(csv_path)
    df = df[df["Label"].isin(PRIM)]
    out: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    for _, r in df.iterrows():
        cid = parse_clip_id(r["Video"])
        if cid is None:
            continue
        out[cid].append((r["Label"], int(r["Start_Frame"]), int(r["End_Frame"]) + 1))
    return dict(out)


def rank_transitions(
    csv_path: str, top_k: int, merge_rear: bool,
) -> list[tuple[str, str]]:
    transitions = find_all_transitions(csv_path, merge_rear=merge_rear)
    counter: Counter[tuple[str, str]] = Counter()
    for t in transitions:
        counter[t.transition_key] += 1
    return [k for k, _ in counter.most_common(top_k)]


def find_dominated_clips(
    segments: dict[int, list[tuple[str, int, int]]],
    A: str, B: str,
    n_frames_per_clip: int,
    threshold: float,
    min_per_behavior_fraction: float,
    skip_clips: set[int],
    n_clips_total: int,
) -> list[int]:
    """Clips where BOTH A and B are substantially present AND their combined
    coverage is a majority.

    Requires:
      A-frames >= min_per_behavior_fraction * T, AND
      B-frames >= min_per_behavior_fraction * T, AND
      (A+B)-frames >= threshold * T.

    The combined threshold alone is insufficient because one behavior
    (e.g. Immobile) often dominates most rodent clips regardless of the
    other.
    """
    out = []
    min_per = int(np.floor(min_per_behavior_fraction * n_frames_per_clip))
    for cid in range(n_clips_total):
        if cid in skip_clips:
            continue
        segs = segments.get(cid, [])
        if not segs:
            continue
        cov_a = cov_b = 0
        for lab, s, e in segs:
            length = max(0, min(e, n_frames_per_clip) - max(s, 0))
            if lab == A:
                cov_a += length
            elif lab == B:
                cov_b += length
        if cov_a < min_per or cov_b < min_per:
            continue
        if (cov_a + cov_b) >= threshold * n_frames_per_clip:
            out.append(cid)
    return out


def build_frame_labels(
    segments: dict[int, list[tuple[str, int, int]]],
    clip_ids: list[int],
    n_frames_per_clip: int,
    A: str, B: str,
) -> np.ndarray:
    """Per (clip_idx, frame_idx) label in {A, B, ""}. Shape: (len(clip_ids), T)."""
    out = np.full((len(clip_ids), n_frames_per_clip), "", dtype=object)
    for row, cid in enumerate(clip_ids):
        for lab, s, e in segments.get(cid, []):
            if lab not in (A, B):
                continue
            lo = max(s, 0)
            hi = min(e, n_frames_per_clip)
            if lo < hi:
                out[row, lo:hi] = lab
    return out


def load_paths_for_clips(
    extraction_dir: str, clip_ids: list[int],
) -> dict[int, list[list[int]]]:
    pkl = Path(extraction_dir) / "all_path_dm.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"missing {pkl}")
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        full = {int(k): list(v) for k, v in data.items()}
    else:
        full = {i: list(c) for i, c in enumerate(data)}
    return {cid: full.get(cid, []) for cid in clip_ids}


def select_top_k_paths(
    paths_by_clip: dict[int, list[list[int]]],
    k: int,
) -> list[tuple[int, list[int]]]:
    flat: list[tuple[int, list[int]]] = []
    for cid, paths in paths_by_clip.items():
        for path in paths:
            verts = list(path)
            if len(verts) > 0:
                flat.append((cid, verts))
    flat.sort(key=lambda cp: -len(cp[1]))
    return flat[:k]


def project_together(
    real_points: np.ndarray,
    path_points: np.ndarray,  # concatenated path points, may be empty
    method: str,
    tsne_cap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a 2D projection on pooled (real + path) points and split back.

    Real frames are subsampled to ``tsne_cap`` for t-SNE tractability. Path
    points are NEVER subsampled — they are the primary overlay.
    """
    real = real_points
    if method == "tsne" and real.shape[0] > tsne_cap:
        idx = rng.choice(real.shape[0], tsne_cap, replace=False)
        real = real[idx]
    else:
        idx = np.arange(real.shape[0])

    n_real = real.shape[0]
    pooled = np.vstack([real, path_points]) if path_points.size else real
    if pooled.shape[0] < 2:
        return (
            np.zeros((n_real, 2), dtype=np.float32),
            np.zeros((path_points.shape[0], 2), dtype=np.float32),
        )

    if method == "pca":
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=0).fit_transform(pooled).astype(np.float32)
    elif method == "umap":
        import umap
        coords = umap.UMAP(
            n_components=2, random_state=0, n_neighbors=15,
            min_dist=0.1, metric="euclidean",
        ).fit_transform(pooled).astype(np.float32)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        coords = TSNE(
            n_components=2,
            perplexity=min(30, max(5, pooled.shape[0] // 30)),
            random_state=0, init="pca", learning_rate="auto",
        ).fit_transform(pooled).astype(np.float32)
    else:
        raise ValueError(f"unknown projection {method}")

    real_coords = coords[:n_real]
    path_coords = coords[n_real:]

    # Re-expand real_coords to all original real indices (pad unused with NaN).
    full_real = np.full((real_points.shape[0], 2), np.nan, dtype=np.float32)
    full_real[idx] = real_coords
    return full_real, path_coords


def nature_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(labelsize=6, width=0.8, length=2.5)
    ax.set_xticks([])
    ax.set_yticks([])


def scatter_real(ax, coords: np.ndarray, labels: np.ndarray, A: str, B: str,
                 grey_background: bool = False):
    mask_ok = ~np.isnan(coords[:, 0])
    coords = coords[mask_ok]
    labels = labels[mask_ok]
    if grey_background:
        ax.scatter(coords[:, 0], coords[:, 1], s=2.0,
                   c=[OTHER_COLOR], linewidths=0, rasterized=True, zorder=1)
        return
    m_other = labels == ""
    if m_other.any():
        ax.scatter(coords[m_other, 0], coords[m_other, 1], s=1.8,
                   c=[OTHER_COLOR], linewidths=0, rasterized=True, zorder=1)
    m_a = labels == A
    if m_a.any():
        ax.scatter(coords[m_a, 0], coords[m_a, 1], s=2.2,
                   c=A_COLOR, linewidths=0, rasterized=True, zorder=2)
    m_b = labels == B
    if m_b.any():
        ax.scatter(coords[m_b, 0], coords[m_b, 1], s=2.2,
                   c=B_COLOR, linewidths=0, rasterized=True, zorder=2)


def overlay_path(ax, path_coords: np.ndarray, color: str, label: str | None = None):
    if path_coords.size == 0:
        return
    ax.plot(path_coords[:, 0], path_coords[:, 1], "-",
            color=color, linewidth=0.9, alpha=0.9, zorder=3, label=label)
    ax.scatter(path_coords[:, 0], path_coords[:, 1], s=5.5,
               c=color, edgecolor="#222", linewidth=0.3, rasterized=True, zorder=4)


def overlay_minima(ax, path_coords: np.ndarray, color: str):
    if path_coords.size == 0:
        return
    ends = np.vstack([path_coords[0], path_coords[-1]])
    ax.scatter(ends[:, 0], ends[:, 1], marker="*", s=120, c="gold",
               edgecolor=color, linewidth=1.2, zorder=6)


def overlay_path_temporal(ax, path_coords: np.ndarray, cmap_name: str = "viridis"):
    if path_coords.shape[0] < 2:
        return
    cmap = plt.get_cmap(cmap_name)
    t = np.linspace(0, 1, path_coords.shape[0])
    for i in range(path_coords.shape[0] - 1):
        ax.plot(path_coords[i:i + 2, 0], path_coords[i:i + 2, 1],
                "-", color=cmap(t[i]), linewidth=1.2, alpha=0.95, zorder=3)
    ax.scatter(path_coords[:, 0], path_coords[:, 1],
               c=t, cmap=cmap_name, s=8, edgecolor="none",
               rasterized=True, zorder=4)


def render_transition_row(
    axes_row,
    A: str, B: str,
    real_coords: np.ndarray, real_labels: np.ndarray,
    paths_coords: list[np.ndarray],
    k: int,
    n_clips: int,
    n_paths_found: int,
    row_title: str,
):
    # Col 0: real only
    nature_axis(axes_row[0])
    scatter_real(axes_row[0], real_coords, real_labels, A, B, grey_background=False)

    # Individual path overlays
    for i in range(k):
        ax = axes_row[1 + i]
        nature_axis(ax)
        scatter_real(ax, real_coords, real_labels, A, B, grey_background=True)
        if i < len(paths_coords):
            overlay_path(ax, paths_coords[i],
                         color=PATH_PALETTE[i % len(PATH_PALETTE)])
            overlay_minima(ax, paths_coords[i],
                           color=PATH_PALETTE[i % len(PATH_PALETTE)])

    # Col K+1: all paths + minima on real-labeled background
    ax_all = axes_row[1 + k]
    nature_axis(ax_all)
    scatter_real(ax_all, real_coords, real_labels, A, B, grey_background=False)
    for i, pc in enumerate(paths_coords):
        overlay_path(ax_all, pc, color=PATH_PALETTE[i % len(PATH_PALETTE)])
        overlay_minima(ax_all, pc, color=PATH_PALETTE[i % len(PATH_PALETTE)])

    # Col K+2: paths only, temporal coloring
    ax_temp = axes_row[2 + k]
    nature_axis(ax_temp)
    for pc in paths_coords:
        overlay_path_temporal(ax_temp, pc)

    # Row title (y-label of first axis)
    axes_row[0].set_ylabel(
        f"{A} ↔ {B}\n(n clips={n_clips}, paths found={n_paths_found})",
        fontsize=9, fontweight="bold", labelpad=10,
    )


def render_figure(
    projection: str,
    rows_data: list,  # list of dicts, one per transition
    k: int,
    cfg: DemoConfig,
    out_path: Path,
):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
    })

    n_cols = k + 3
    n_rows = len(rows_data)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 3.1 * n_rows),
        squeeze=False,
    )

    col_titles = (
        ["Real (labeled)"]
        + [f"DM path {i + 1}" for i in range(k)]
        + ["All paths + minima", "Paths (temporal)"]
    )
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold", pad=6)

    for row_idx, row_data in enumerate(rows_data):
        render_transition_row(
            axes[row_idx],
            A=row_data["A"], B=row_data["B"],
            real_coords=row_data["real_coords"],
            real_labels=row_data["real_labels"],
            paths_coords=row_data["path_coords"],
            k=k,
            n_clips=row_data["n_clips"],
            n_paths_found=row_data["n_paths_found"],
            row_title=row_data["row_title"],
        )

    fig.suptitle(
        f"Transition-centred projection — {PROJECTION_DISPLAY[projection]}"
        f" (feature space: {cfg.feature_space})",
        fontsize=12, fontweight="bold", y=1.0,
    )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=A_COLOR, edgecolor="none", label="behavior A"),
        Patch(facecolor=B_COLOR, edgecolor="none", label="behavior B"),
        Patch(facecolor=OTHER_COLOR, edgecolor="none", label="other"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="#222", markersize=10, label="path minima"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))

    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    plt.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"  wrote {out_path} (+.png)")


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    os.makedirs(cfg.output_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    print("Loading feature data...")
    feature_data = load_feature_data(cfg)
    n_clips, T, D = feature_data.shape
    print(f"  {cfg.feature_space}: {feature_data.shape}")

    print("Loading behavior segments...")
    segments = load_behavior_segments(cfg.behavior_labels_csv)
    print(f"  {len(segments)} clips with GT labels")

    print("Ranking transitions...")
    transitions = rank_transitions(
        cfg.behavior_labels_csv, cfg.n_transitions, cfg.merge_rear,
    )
    print(f"  top {cfg.n_transitions}: {transitions}")

    # Build per-transition data bundles, one per row.
    row_bundles = []
    for A, B in transitions:
        dominated = find_dominated_clips(
            segments, A, B, T,
            cfg.majority_threshold, cfg.min_per_behavior_fraction,
            set(cfg.skip_clips), n_clips,
        )
        print(f"\n{A} <-> {B}: {len(dominated)} clips dominated "
              f"(both >= {cfg.min_per_behavior_fraction:.0%}, "
              f"sum >= {cfg.majority_threshold:.0%})")
        if not dominated:
            row_bundles.append(None)
            continue

        paths_by_clip = load_paths_for_clips(cfg.extraction_dir, dominated)
        top_paths = select_top_k_paths(paths_by_clip, cfg.k_paths)
        n_paths_found = len(top_paths)
        print(f"  {n_paths_found} DM paths chosen "
              f"(top {cfg.k_paths} by length from dominated clips)")

        # Real frames: (cid, frame_idx) flattened across dominated clips.
        real_list: list[np.ndarray] = []
        label_list: list[np.ndarray] = []
        labels_grid = build_frame_labels(segments, dominated, T, A, B)
        for i, cid in enumerate(dominated):
            real_list.append(feature_data[cid])
            label_list.append(labels_grid[i])
        real_X = np.concatenate(real_list, axis=0)
        real_L = np.concatenate(label_list, axis=0)

        # Stride-subsample real frames uniformly to keep projection tractable.
        if real_X.shape[0] > cfg.real_subsample_n:
            step = real_X.shape[0] // cfg.real_subsample_n
            keep_idx = np.arange(0, real_X.shape[0], step)[:cfg.real_subsample_n]
            real_X = real_X[keep_idx]
            real_L = real_L[keep_idx]

        # Path points: for each path, pull feature_data[cid][frame_idx] per vertex.
        path_feature_list: list[np.ndarray] = []
        path_lens: list[int] = []
        for cid, verts in top_paths:
            verts_arr = np.asarray(verts, dtype=np.int64)
            verts_arr = verts_arr[(verts_arr >= 0) & (verts_arr < T)]
            if verts_arr.size == 0:
                path_feature_list.append(np.zeros((0, D), dtype=np.float32))
                path_lens.append(0)
                continue
            path_feature_list.append(feature_data[cid, verts_arr])
            path_lens.append(verts_arr.size)

        row_bundles.append({
            "A": A, "B": B,
            "dominated": dominated,
            "top_paths": top_paths,
            "real_X": real_X,
            "real_L": real_L,
            "path_features": path_feature_list,
            "path_lens": path_lens,
            "n_paths_found": n_paths_found,
        })

    # One figure per projection.
    for projection in cfg.projections:
        print(f"\nProjection: {projection}")
        rendered_rows = []
        for (A, B), bundle in zip(transitions, row_bundles):
            if bundle is None:
                rendered_rows.append({
                    "A": A, "B": B,
                    "real_coords": np.zeros((0, 2)),
                    "real_labels": np.array([], dtype=object),
                    "path_coords": [],
                    "n_clips": 0, "n_paths_found": 0,
                    "row_title": f"{A} ↔ {B} (no dominated clips)",
                })
                continue

            all_path_pts = (
                np.concatenate(bundle["path_features"], axis=0)
                if bundle["path_features"] else np.zeros((0, bundle["real_X"].shape[1]))
            )
            real_coords, path_coords_flat = project_together(
                bundle["real_X"], all_path_pts, projection,
                cfg.tsne_subsample_n, rng,
            )

            # Split flat path coords back per path using path_lens.
            per_path_coords: list[np.ndarray] = []
            cursor = 0
            for L in bundle["path_lens"]:
                per_path_coords.append(path_coords_flat[cursor:cursor + L])
                cursor += L

            rendered_rows.append({
                "A": A, "B": B,
                "real_coords": real_coords,
                "real_labels": bundle["real_L"],
                "path_coords": per_path_coords,
                "n_clips": len(bundle["dominated"]),
                "n_paths_found": bundle["n_paths_found"],
                "row_title": f"{A} ↔ {B}",
            })

        out_path = Path(cfg.output_dir) / f"transition_projections_{projection}.pdf"
        render_figure(projection, rendered_rows, cfg.k_paths, cfg, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(args.config)
