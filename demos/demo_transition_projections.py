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
from topo_mimic.extraction.skeletonization.dm_path import (
    _parse_persistence_pairs,
    _parse_sorted_weights,
)
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


def _add_colored_transition_title(
    ax,
    A: str,
    B: str,
    *,
    A_color: str = A_COLOR,
    B_color: str = B_COLOR,
    arrow_color: str = "#222",
    fontsize: int = 11,
    y: float = 1.04,
    sep: int = 3,
):
    """Place a horizontal `A ↔ B` title above ``ax`` with per-segment colors.

    Uses ``HPacker`` so the three text runs are measured and laid out side
    by side — avoids the overlap from anchoring all three texts at the same
    x with ha=right/center/left.
    """
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

    common = dict(fontsize=fontsize, fontweight="bold")
    boxes = [
        TextArea(A, textprops=dict(color=A_color, **common)),
        TextArea("↔", textprops=dict(color=arrow_color, **common)),
        TextArea(B, textprops=dict(color=B_color, **common)),
    ]
    pack = HPacker(children=boxes, align="center", pad=0, sep=sep)
    abox = AnchoredOffsetbox(
        loc="lower center", child=pack, pad=0.0, borderpad=0.0,
        frameon=False, bbox_to_anchor=(0.5, y),
        bbox_transform=ax.transAxes,
    )
    ax.add_artist(abox)


@dataclass
class DemoConfig:
    h5_path: str = "data/rodent_ar.h5"
    behavior_labels_csv: str = "behavior_labels/rodent_labels.csv"
    xml_path: str = "topo_mimic/extraction/xml/rodent.xml"
    stac_data_path: str = "data/rodent_reference_clips.h5"
    feature_spaces: list[str] = field(default_factory=lambda: ["intention", "qpos", "ekp"])
    extraction_dirs: dict[str, str] = field(default_factory=lambda: {
        "intention": "outputs/extraction/dm_pipeline_intention",
        "qpos":      "outputs/extraction/dm_pipeline_qpos",
        "ekp":       "outputs/extraction/dm_pipeline_ekp",
    })
    seed: int = 0
    majority_threshold: float = 0.5
    min_per_behavior_fraction: float = 0.15
    min_clips_per_transition: int = 10
    min_clips_per_direction_variant: int = 10
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


def load_feature_data(cfg: DemoConfig, feature_space: str) -> np.ndarray:
    hdf5_key = get_hdf5_key_for_dm_key(feature_space)
    with h5py.File(cfg.h5_path, "r") as f:
        raw = np.asarray(f[hdf5_key][:])
    return apply_dm_key_transform(
        raw, feature_space,
        xml_path=cfg.xml_path, stac_data_path=cfg.stac_data_path,
    )


def load_direction_segments(csv_path: str) -> dict[int, list[tuple[str, int, int]]]:
    """Return {clip_id: [(direction, start_frame, end_frame), ...]} (exclusive end).

    Direction labels are overlays (Left, Right, Straight) that coexist with
    behavior labels; a frame may have one of each.
    """
    DIRS = {"Left", "Right", "Straight"}
    df = pd.read_csv(csv_path)
    df = df[df["Label"].isin(DIRS)]
    out: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    for _, r in df.iterrows():
        cid = parse_clip_id(r["Video"])
        if cid is None:
            continue
        out[cid].append((r["Label"], int(r["Start_Frame"]), int(r["End_Frame"]) + 1))
    return dict(out)


def dominant_direction_for_behavior(
    behavior_segs: list[tuple[str, int, int]],
    direction_segs: list[tuple[str, int, int]],
    behavior: str,
    n_frames: int,
) -> str:
    """Return the direction that overlaps the majority of `behavior`-labeled
    frames in the clip, or '' if behavior never appears."""
    counts = {"Left": 0, "Right": 0, "Straight": 0}
    any_beh_frames = False
    for lab, s, e in behavior_segs:
        if lab != behavior:
            continue
        any_beh_frames = True
        lo, hi = max(s, 0), min(e, n_frames)
        if lo >= hi:
            continue
        for dlab, ds, de in direction_segs:
            if dlab not in counts:
                continue
            overlap = max(0, min(hi, de) - max(lo, ds))
            if overlap > 0:
                counts[dlab] += overlap
    if not any_beh_frames:
        return ""
    total = sum(counts.values())
    if total == 0:
        return "Straight"  # behavior present but no direction overlay -> default
    return max(counts, key=counts.get)


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


def _frame_label(
    segs: list[tuple[str, int, int]], frame: int,
) -> str:
    for lab, s, e in segs:
        if s <= frame < e:
            return lab
    return ""


_PERS_CACHE: dict[tuple[str, int], float] = {}


def clip_minima_persistence_score(extraction_dir: str, clip_id: int) -> float:
    """Score a clip by how persistent its top-2 density minima are.

    Reads the per-clip persistence pairs DiMoSC emits; ranks type-0 pairs
    by |persistence| descending; takes the lower vertex (min-side) of each
    pair as a minimum, deduped. Returns min(persistence of top-1 minimum,
    persistence of top-2 minimum) -- the "limiting" persistence. Higher =
    the clip has a clearer two-basin structure; paths between these minima
    are more canonical A<->B transition skeletons.

    inf values (immortal generators) are clipped to a large finite value
    so they don't dominate min(); practically this only affects the top
    minimum of each clip, which under DiMoSC's convention is immortal.
    """
    key = (extraction_dir, int(clip_id))
    if key in _PERS_CACHE:
        return _PERS_CACHE[key]
    clip_dir = Path(extraction_dir) / "clips" / f"clip_{clip_id}"
    pairs_path = clip_dir / "edge_for_morse_only.txt"
    weights_path = clip_dir / "sorted-weights.txt"
    if not pairs_path.exists() or not weights_path.exists():
        _PERS_CACHE[key] = 0.0
        return 0.0
    try:
        pairs = _parse_persistence_pairs(str(clip_dir))
        weights = _parse_sorted_weights(str(clip_dir))
    except Exception:
        _PERS_CACHE[key] = 0.0
        return 0.0
    type0 = [(v0, v1, pers) for v0, v1, t, pers in pairs if t == 0]
    type0.sort(key=lambda x: float("inf") if x[2] == float("inf") else abs(x[2]),
               reverse=True)
    seen: set[int] = set()
    minima_pers: list[float] = []
    for v0, v1, pers in type0:
        w0 = weights.get(v0, float("inf"))
        w1 = weights.get(v1, float("inf"))
        min_v = v0 if w0 < w1 else v1
        if min_v in seen:
            continue
        seen.add(min_v)
        p = 1e9 if pers == float("inf") else abs(pers)
        minima_pers.append(p)
        if len(minima_pers) >= 2:
            break
    if len(minima_pers) < 2:
        _PERS_CACHE[key] = 0.0
        return 0.0
    score = min(minima_pers[0], minima_pers[1])
    _PERS_CACHE[key] = score
    return score


def _frame_direction(
    dir_segs: list[tuple[str, int, int]], frame: int,
) -> str:
    for lab, s, e in dir_segs:
        if s <= frame < e:
            return lab
    return ""


def select_top_k_paths_with_direction(
    paths_by_clip: dict[int, list[list[int]]],
    k: int,
    segments: dict[int, list[tuple[str, int, int]]],
    direction_segments: dict[int, list[tuple[str, int, int]]],
    A: str, B: str,
    dir_A: str, dir_B: str,
    extraction_dir: str | None = None,
) -> list[tuple[int, list[int], str, str]]:
    """Variant-aware path filter, ranked by minima persistence (same as
    ``select_top_k_paths``). Requires one endpoint to fall inside an
    A-segment with direction ``dir_A`` and the other inside a B-segment
    with direction ``dir_B``.
    """
    flat: list[tuple[float, int, list[int], str, str]] = []
    for cid, paths in paths_by_clip.items():
        beh_segs = segments.get(cid, [])
        dir_segs = direction_segments.get(cid, [])
        pers_score = (
            clip_minima_persistence_score(extraction_dir, cid)
            if extraction_dir else float(len(paths))
        )
        for path in paths:
            verts = list(path)
            if len(verts) < 2:
                continue
            f0, fN = int(verts[0]), int(verts[-1])
            lab0 = _frame_label(beh_segs, f0)
            labN = _frame_label(beh_segs, fN)
            d0 = _frame_direction(dir_segs, f0) or "Straight"
            dN = _frame_direction(dir_segs, fN) or "Straight"
            endpoints = {(lab0, d0), (labN, dN)}
            if endpoints == {(A, dir_A), (B, dir_B)}:
                score = pers_score if extraction_dir else float(len(verts))
                flat.append((score, cid, verts, lab0, labN))
    flat.sort(key=lambda t: -t[0])
    return [(c, v, fl, ll) for _, c, v, fl, ll in flat[:k]]


def select_top_k_paths(
    paths_by_clip: dict[int, list[list[int]]],
    k: int,
    segments: dict[int, list[tuple[str, int, int]]],
    A: str, B: str,
    extraction_dir: str | None = None,
) -> list[tuple[int, list[int], str, str]]:
    """Keep paths whose endpoint frames fall inside A-segment and B-segment,
    ranked by the persistence of the clip's top-2 density minima (descending).

    If ``extraction_dir`` is given, paths are scored by
    ``clip_minima_persistence_score(extraction_dir, clip_id)``: the min of
    the two most persistent minima for that clip. Among clips with matching
    {A, B} endpoint labels, paths from the clips with the deepest minima
    come first.
    If ``extraction_dir`` is None, falls back to ranking by path length.
    """
    flat: list[tuple[float, int, list[int], str, str]] = []
    for cid, paths in paths_by_clip.items():
        segs = segments.get(cid, [])
        pers_score = (
            clip_minima_persistence_score(extraction_dir, cid)
            if extraction_dir else float(len(paths))
        )
        for path in paths:
            verts = list(path)
            if len(verts) < 2:
                continue
            first_lab = _frame_label(segs, int(verts[0]))
            last_lab  = _frame_label(segs, int(verts[-1]))
            if {first_lab, last_lab} != {A, B}:
                continue
            score = pers_score if extraction_dir else float(len(verts))
            flat.append((score, cid, verts, first_lab, last_lab))
    flat.sort(key=lambda t: -t[0])
    return [(c, v, fl, ll) for _, c, v, fl, ll in flat[:k]]


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

    explained_variance: tuple[float, float] | None = None
    if method == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(pooled).astype(np.float32)
        explained_variance = (
            float(pca.explained_variance_ratio_[0]),
            float(pca.explained_variance_ratio_[1]),
        )
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
    return full_real, path_coords, explained_variance


def nature_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(labelsize=6, width=0.8, length=2.5)
    ax.set_xticks([])
    ax.set_yticks([])


def scatter_real(
    ax, coords: np.ndarray, labels: np.ndarray, A: str, B: str,
    grey_background: bool = False,
    alpha_ab: float = 1.0,
    alpha_other: float = 1.0,
    size_ab: float = 2.2,
    size_other: float = 1.8,
):
    mask_ok = ~np.isnan(coords[:, 0])
    coords = coords[mask_ok]
    labels = labels[mask_ok]
    grey = (OTHER_COLOR[0], OTHER_COLOR[1], OTHER_COLOR[2], OTHER_COLOR[3] * alpha_other)
    if grey_background:
        ax.scatter(coords[:, 0], coords[:, 1], s=size_other * (alpha_other + 0.2),
                   c=[grey], linewidths=0, rasterized=True, zorder=1)
        return
    m_other = labels == ""
    if m_other.any():
        ax.scatter(coords[m_other, 0], coords[m_other, 1], s=size_other,
                   c=[grey], linewidths=0, rasterized=True, zorder=1)
    m_a = labels == A
    if m_a.any():
        ax.scatter(coords[m_a, 0], coords[m_a, 1], s=size_ab,
                   c=A_COLOR, alpha=alpha_ab,
                   linewidths=0, rasterized=True, zorder=2)
    m_b = labels == B
    if m_b.any():
        ax.scatter(coords[m_b, 0], coords[m_b, 1], s=size_ab,
                   c=B_COLOR, alpha=alpha_ab,
                   linewidths=0, rasterized=True, zorder=2)


def overlay_path(ax, path_coords: np.ndarray, color: str, label: str | None = None):
    if path_coords.size == 0:
        return
    ax.plot(path_coords[:, 0], path_coords[:, 1], "-",
            color=color, linewidth=0.9, alpha=0.9, zorder=3, label=label)
    ax.scatter(path_coords[:, 0], path_coords[:, 1], s=5.5,
               c=color, edgecolor="#222", linewidth=0.3, rasterized=True, zorder=4)


def overlay_path_endpoints(
    ax,
    path_coords: np.ndarray,
    first_label: str,
    last_label: str,
    A: str, B: str,
    star_size: float = 240,
):
    """Place gold stars at the path's two endpoints. Star edge color
    indicates which behavior each endpoint falls in (blue for A, red for B).
    """
    if path_coords.shape[0] < 2:
        return
    for coord, lab in (
        (path_coords[0],  first_label),
        (path_coords[-1], last_label),
    ):
        edge = A_COLOR if lab == A else (B_COLOR if lab == B else "#222")
        ax.scatter([coord[0]], [coord[1]], marker="*",
                   s=star_size, c="gold",
                   edgecolor=edge, linewidth=1.6, zorder=7)


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
    paths_endpoint_labels: list[tuple[str, str]],
    k: int,
    n_clips: int,
    n_paths_found: int,
    row_title: str,
):
    # Col 0: real only
    nature_axis(axes_row[0])
    scatter_real(axes_row[0], real_coords, real_labels, A, B, grey_background=False)

    # Individual path overlays: real as grey background, path line + stars
    # at the path's own endpoints (blue-edge for A, red-edge for B).
    for i in range(k):
        ax = axes_row[1 + i]
        nature_axis(ax)
        scatter_real(ax, real_coords, real_labels, A, B, grey_background=True)
        if i < len(paths_coords):
            overlay_path(ax, paths_coords[i],
                         color=PATH_PALETTE[i % len(PATH_PALETTE)])
            overlay_path_endpoints(
                ax, paths_coords[i],
                paths_endpoint_labels[i][0], paths_endpoint_labels[i][1],
                A, B,
            )

    # Col K+1: all paths on labeled background + endpoint stars per path.
    ax_all = axes_row[1 + k]
    nature_axis(ax_all)
    scatter_real(ax_all, real_coords, real_labels, A, B, grey_background=False)
    for i, pc in enumerate(paths_coords):
        overlay_path(ax_all, pc, color=PATH_PALETTE[i % len(PATH_PALETTE)])
        overlay_path_endpoints(
            ax_all, pc,
            paths_endpoint_labels[i][0], paths_endpoint_labels[i][1],
            A, B,
        )

    # Col K+2: paths only, temporal coloring
    ax_temp = axes_row[2 + k]
    nature_axis(ax_temp)
    for pc in paths_coords:
        overlay_path_temporal(ax_temp, pc)

    # Row title (y-label of first axis)
    axes_row[0].set_ylabel(
        f"{A} ↔ {B}",
        fontsize=9, fontweight="bold", labelpad=10,
    )


def render_direction_panel_figure(
    projection: str,
    grid_data: dict,  # {fs: list[variant_row_dict]}
    feature_spaces: list[str],
    variant_specs: list[tuple[str, str, str, str, list[int]]],
    cfg: DemoConfig,
    out_path: Path,
):
    """Direction-conditioned panel. Rows = feature spaces, cols = kept
    direction variants (A-dA <-> B-dB) across all transitions.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
    })

    n_rows = len(feature_spaces)
    n_cols = len(variant_specs)
    if n_cols == 0:
        print(f"  no direction variants survived filter; skipping {out_path}")
        return

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.8 * n_cols, 2.9 * n_rows),
        squeeze=False,
    )

    for col_idx, (A, B, dA, dB, cids) in enumerate(variant_specs):
        axes[0, col_idx].set_title(
            f"{A}-{dA}\n↔\n{B}-{dB}",
            fontsize=9, fontweight="bold", pad=6,
        )

    for row_idx, fs in enumerate(feature_spaces):
        axes[row_idx, 0].set_ylabel(
            fs, fontsize=10, fontweight="bold", labelpad=12,
        )
        rows = grid_data.get(fs, [])
        for col_idx, spec in enumerate(variant_specs):
            A, B, dA, dB, _cids = spec
            ax = axes[row_idx, col_idx]
            nature_axis(ax)
            if col_idx >= len(rows):
                continue
            row = rows[col_idx]
            if row["real_coords"].shape[0] == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#777")
                continue
            scatter_real(
                ax, row["real_coords"], row["real_labels"], A, B,
                grey_background=False,
                alpha_ab=0.55, alpha_other=0.45,
                size_ab=6.0, size_other=4.0,
            )
            if row["path_coords"]:
                import matplotlib.patheffects as PE
                pc = row["path_coords"][0]
                if pc.size > 0:
                    ax.plot(
                        pc[:, 0], pc[:, 1],
                        linestyle=(0, (5, 3)),
                        color="#0a0a0a", linewidth=1.8, alpha=1.0, zorder=5,
                        path_effects=[PE.withStroke(linewidth=3.8, foreground="white")],
                    )
                    overlay_path_endpoints(
                        ax, pc,
                        row["paths_endpoint_labels"][0][0],
                        row["paths_endpoint_labels"][0][1],
                        A, B, star_size=180,
                    )

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=A_COLOR, edgecolor="none", label="behavior A"),
        Patch(facecolor=B_COLOR, edgecolor="none", label="behavior B"),
        Patch(facecolor=OTHER_COLOR, edgecolor="none", label="other"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="#222", markersize=10,
               label="path endpoint (minimum)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle(
        f"Transition x direction-variant panel — {PROJECTION_DISPLAY[projection]}",
        fontsize=12, fontweight="bold", y=1.0,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"  wrote {out_path} (+.png)")


def _hex_to_rgb01(hex_str: str) -> np.ndarray:
    h = hex_str.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def render_transition_filmstrip_figure(
    grid_data: dict,  # {fs: list[row_dict]} where each row carries top_paths + clip_ids
    feature_spaces: list[str],
    transitions: list[tuple[str, str]],
    qpos_all: np.ndarray,          # (n_clips, T, qpos_dims) raw qpos for rendering
    xml_path: str,
    dataset: str,
    cfg: DemoConfig,
    out_path: Path,
    n_poses: int = 6,
    show_row_labels: bool = True,
):
    """Grid of rendered MuJoCo pose filmstrips for the top-1 DM path per
    (feature_space, transition) cell. Tinted from A-color (start) to
    B-color (end) using the same style as demo_transitions_projections_single_clip.
    """
    from demos.skeleton_filmstrip import render_filmstrip_for_frames

    n_rows = len(feature_spaces)
    n_cols = len(transitions)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 1.3 * n_rows + 0.8),
        squeeze=False,
        gridspec_kw={"wspace": 0.18, "hspace": 0.25},
    )

    for col_idx, (A, B) in enumerate(transitions):
        _add_colored_transition_title(axes[0, col_idx], A, B, fontsize=11, y=1.04)

    for row_idx, fs in enumerate(feature_spaces):
        if show_row_labels:
            axes[row_idx, 0].set_ylabel(
                fs, fontsize=10, fontweight="bold", labelpad=12,
            )
        rows = grid_data.get(fs, [])
        for col_idx, (A, B) in enumerate(transitions):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ("top", "right", "left", "bottom"):
                ax.spines[s].set_visible(False)
            if col_idx >= len(rows):
                continue
            row = rows[col_idx]
            top_paths = row.get("top_paths_raw") or []
            if not top_paths:
                ax.text(0.5, 0.5, "no path", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#777")
                continue

            # Top path = index 0.
            cid, verts, first_lab, last_lab = top_paths[0]
            verts_arr = np.asarray(verts, dtype=np.int64)
            T = qpos_all.shape[1]
            verts_arr = verts_arr[(verts_arr >= 0) & (verts_arr < T)]
            if verts_arr.size < 2:
                ax.text(0.5, 0.5, "path too short", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#777")
                continue

            # Ensure temporal ordering so the A->B color sweep reads L->R.
            if first_lab == A:
                frame_indices = sorted(verts_arr.tolist())
                tint_start = _hex_to_rgb01(A_COLOR)
                tint_end   = _hex_to_rgb01(B_COLOR)
            else:
                frame_indices = sorted(verts_arr.tolist(), reverse=True)
                tint_start = _hex_to_rgb01(B_COLOR)
                tint_end   = _hex_to_rgb01(A_COLOR)
                # But since render_filmstrip_for_frames subsamples in order,
                # keep frame_indices ascending; just swap tints so left = A.
                frame_indices = sorted(verts_arr.tolist())

            strip = render_filmstrip_for_frames(
                qpos_all[cid], frame_indices, xml_path, dataset=dataset,
                n_poses=n_poses, gap_fraction=0.25,
                tint_start=tint_start, tint_end=tint_end,
            )
            ax.imshow(strip)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=A_COLOR, edgecolor="none", label="behavior A (start)"),
        Patch(facecolor=B_COLOR, edgecolor="none", label="behavior B (end)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.03, 1, 0.99))
    plt.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"  wrote {out_path} (+.png)")


def render_combined_panel_figure(
    projection: str,
    grid_data: dict,  # {fs: list[row_dict]} where row_dict matches render_transition_row input
    feature_spaces: list[str],
    transitions: list[tuple[str, str]],
    cfg: DemoConfig,
    out_path: Path,
    show_legend: bool = True,
    show_support_annotation: bool = False,
):
    """One image per projection method. Rows = feature spaces, cols =
    transition types. Each cell = compact scatter of real A/B background +
    K DM paths + gold stars at path endpoints.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
    })

    n_rows = len(feature_spaces)
    n_cols = len(transitions)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    for col_idx, (A, B) in enumerate(transitions):
        # Color-code the column title so the reader can read off which scatter
        # color is which behavior in this column (A = first/blue, B = second/red).
        # HPacker lays the three runs out side by side without overlap.
        _add_colored_transition_title(axes[0, col_idx], A, B, fontsize=11, y=1.04)

    for row_idx, fs in enumerate(feature_spaces):
        axes[row_idx, 0].set_ylabel(
            fs, fontsize=10, fontweight="bold", labelpad=12,
        )
        rows = grid_data.get(fs, [])
        for col_idx, (A, B) in enumerate(transitions):
            ax = axes[row_idx, col_idx]
            nature_axis(ax)
            if col_idx >= len(rows):
                continue
            row = rows[col_idx]
            if row["real_coords"].shape[0] == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#777")
                continue
            scatter_real(
                ax, row["real_coords"], row["real_labels"], A, B,
                grey_background=False,
                alpha_ab=0.55, alpha_other=0.45,
                size_ab=6.0, size_other=4.0,
            )
            # Single top path: dark dashed line with a white halo for contrast.
            if row["path_coords"]:
                pc = row["path_coords"][0]
                if pc.size > 0:
                    import matplotlib.patheffects as PE
                    ax.plot(
                        pc[:, 0], pc[:, 1],
                        linestyle=(0, (5, 3)),
                        color="#0a0a0a", linewidth=1.8, alpha=1.0, zorder=5,
                        path_effects=[PE.withStroke(linewidth=3.8, foreground="white")],
                    )
                    overlay_path_endpoints(
                        ax, pc,
                        row["paths_endpoint_labels"][0][0],
                        row["paths_endpoint_labels"][0][1],
                        A, B, star_size=200,
                    )
            if show_support_annotation:
                ax.text(
                    0.02, 0.98,
                    f"n clips={row['n_clips']}, paths={row['n_paths_found']}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=7, color="#555",
                )
            # PCA-only: show explained variance per axis.
            if projection == "pca":
                ev = row.get("explained_variance")
                if ev is not None:
                    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=8)
                    if col_idx == 0:
                        ax.set_ylabel(
                            f"{fs}\nPC2 ({ev[1]*100:.1f}%)",
                            fontsize=9, fontweight="bold", labelpad=12,
                        )

    if show_legend:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        handles = [
            Patch(facecolor=A_COLOR, edgecolor="none",
                  label="behavior A (left of ↔ in column title)"),
            Patch(facecolor=B_COLOR, edgecolor="none",
                  label="behavior B (right of ↔ in column title)"),
            Patch(facecolor=OTHER_COLOR, edgecolor="none", label="other frames"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                   markeredgecolor="#222", markersize=10,
                   label="path endpoint (basin minimum)"),
            Line2D([0], [0], linestyle=(0, (5, 3)), color="#0a0a0a",
                   linewidth=1.8, label="DM path"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))
        fig.tight_layout(rect=(0, 0.04, 1, 0.99))
    else:
        fig.tight_layout(rect=(0, 0.0, 1, 0.99))
    plt.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"  wrote {out_path} (+.png)")


def render_figure(
    projection: str,
    feature_space: str,
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
            paths_endpoint_labels=row_data["paths_endpoint_labels"],
            k=k,
            n_clips=row_data["n_clips"],
            n_paths_found=row_data["n_paths_found"],
            row_title=row_data["row_title"],
        )

    fig.suptitle(
        f"Transition-centred projection — {PROJECTION_DISPLAY[projection]}"
        f" (feature space: {feature_space})",
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

    print("Loading behavior segments...")
    segments = load_behavior_segments(cfg.behavior_labels_csv)
    direction_segments = load_direction_segments(cfg.behavior_labels_csv)
    print(f"  {len(segments)} clips with GT labels")

    print("Ranking transitions...")
    transitions = rank_transitions(
        cfg.behavior_labels_csv, cfg.n_transitions, cfg.merge_rear,
    )
    print(f"  top {cfg.n_transitions}: {transitions}")

    # Drop transitions with too few dominated clips. We do this on the
    # first feature space (intention); support is nearly identical across
    # the three, so a single check is representative.
    kept_transitions: list[tuple[str, str]] = []
    for A, B in transitions:
        # A lightweight check: dominance is a property of the GT CSV, not
        # the feature space, so any T would give the same answer. Use
        # T=500 (rodent clip length) as the representative.
        dom_preview = find_dominated_clips(
            segments, A, B, 500,
            cfg.majority_threshold, cfg.min_per_behavior_fraction,
            set(cfg.skip_clips), 842,
        )
        if len(dom_preview) < cfg.min_clips_per_transition:
            print(f"  dropping {A}<->{B}: only {len(dom_preview)} clips "
                  f"(<{cfg.min_clips_per_transition})")
            continue
        kept_transitions.append((A, B))
    transitions = kept_transitions
    print(f"  kept transitions: {transitions}")

    combined_grid: dict[str, dict[str, list]] = {p: {} for p in cfg.projections}

    # Per-transition direction variants, same for all projections; we
    # compute the set of kept (A, B, dir_A, dir_B) variants once.
    direction_variants: list[tuple[str, str, str, str, list[int]]] = []
    for A, B in transitions:
        dom_clips = find_dominated_clips(
            segments, A, B, 500,
            cfg.majority_threshold, cfg.min_per_behavior_fraction,
            set(cfg.skip_clips), 842,
        )
        # Group by (dominant dir during A, dominant dir during B).
        by_variant: dict[tuple[str, str], list[int]] = defaultdict(list)
        for cid in dom_clips:
            beh_segs = segments.get(cid, [])
            dir_segs = direction_segments.get(cid, [])
            dA = dominant_direction_for_behavior(beh_segs, dir_segs, A, 500) or "Straight"
            dB = dominant_direction_for_behavior(beh_segs, dir_segs, B, 500) or "Straight"
            by_variant[(dA, dB)].append(cid)
        for (dA, dB), cids in by_variant.items():
            if len(cids) >= cfg.min_clips_per_direction_variant:
                direction_variants.append((A, B, dA, dB, cids))
                print(f"  variant kept: {A}-{dA} <-> {B}-{dB}  "
                      f"({len(cids)} clips)")
            else:
                print(f"  variant dropped: {A}-{dA} <-> {B}-{dB}  "
                      f"({len(cids)} clips < {cfg.min_clips_per_direction_variant})")

    # Store per (projection, feature_space, variant) rendered rows for the
    # direction-panel figure.
    direction_grid: dict[str, dict[str, list]] = {p: {} for p in cfg.projections}

    for feature_space in cfg.feature_spaces:
        print(f"\n{'=' * 56}\nFeature space: {feature_space}\n{'=' * 56}")
        extraction_dir = cfg.extraction_dirs.get(feature_space)
        if extraction_dir is None:
            print(f"  no extraction_dir configured for {feature_space}, skipping")
            continue

        print(f"Loading feature data ({feature_space})...")
        feature_data = load_feature_data(cfg, feature_space)
        n_clips, T, D = feature_data.shape
        print(f"  shape: {feature_data.shape}")

        # Build per-transition data bundles for this feature space.
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

            paths_by_clip = load_paths_for_clips(extraction_dir, dominated)
            top_paths = select_top_k_paths(
                paths_by_clip, cfg.k_paths, segments, A, B,
                extraction_dir=extraction_dir,
            )
            n_paths_found = len(top_paths)
            print(f"  {n_paths_found} DM paths chosen "
                  f"(top {cfg.k_paths} by length, endpoints in {{{A}, {B}}})")

            labels_grid = build_frame_labels(segments, dominated, T, A, B)
            real_list, label_list = [], []
            for i, cid in enumerate(dominated):
                real_list.append(feature_data[cid])
                label_list.append(labels_grid[i])
            real_X = np.concatenate(real_list, axis=0)
            real_L = np.concatenate(label_list, axis=0)

            if real_X.shape[0] > cfg.real_subsample_n:
                step = real_X.shape[0] // cfg.real_subsample_n
                keep_idx = np.arange(0, real_X.shape[0], step)[:cfg.real_subsample_n]
                real_X = real_X[keep_idx]
                real_L = real_L[keep_idx]

            path_feature_list, path_lens, endpoint_labels = [], [], []
            for cid, verts, first_lab, last_lab in top_paths:
                verts_arr = np.asarray(verts, dtype=np.int64)
                verts_arr = verts_arr[(verts_arr >= 0) & (verts_arr < T)]
                if verts_arr.size == 0:
                    path_feature_list.append(np.zeros((0, D), dtype=np.float32))
                    path_lens.append(0)
                    endpoint_labels.append((first_lab, last_lab))
                    continue
                path_feature_list.append(feature_data[cid, verts_arr])
                path_lens.append(verts_arr.size)
                endpoint_labels.append((first_lab, last_lab))

            row_bundles.append({
                "A": A, "B": B,
                "dominated": dominated,
                "top_paths": top_paths,
                "real_X": real_X,
                "real_L": real_L,
                "path_features": path_feature_list,
                "path_lens": path_lens,
                "endpoint_labels": endpoint_labels,
                "n_paths_found": n_paths_found,
            })

        for projection in cfg.projections:
            print(f"\nProjection: {projection} ({feature_space})")
            rendered_rows = []
            for (A, B), bundle in zip(transitions, row_bundles):
                if bundle is None:
                    rendered_rows.append({
                        "A": A, "B": B,
                        "real_coords": np.zeros((0, 2)),
                        "real_labels": np.array([], dtype=object),
                        "path_coords": [],
                        "paths_endpoint_labels": [],
                        "n_clips": 0, "n_paths_found": 0,
                        "row_title": f"{A} ↔ {B} (no dominated clips)",
                    })
                    continue

                all_path_pts = (
                    np.concatenate(bundle["path_features"], axis=0)
                    if bundle["path_features"]
                    else np.zeros((0, bundle["real_X"].shape[1]))
                )
                real_coords, path_coords_flat, explained_var = project_together(
                    bundle["real_X"], all_path_pts, projection,
                    cfg.tsne_subsample_n, rng,
                )

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
                    "paths_endpoint_labels": bundle["endpoint_labels"],
                    "top_paths_raw": bundle["top_paths"],
                    "n_clips": len(bundle["dominated"]),
                    "n_paths_found": bundle["n_paths_found"],
                    "row_title": f"{A} ↔ {B}",
                    "explained_variance": explained_var,
                })

            combined_grid[projection][feature_space] = rendered_rows

        # Direction-variant bundles for this feature space.
        for projection in cfg.projections:
            direction_grid[projection][feature_space] = []

        for A, B, dA, dB, variant_clips in direction_variants:
            variant_clips_set = set(variant_clips)
            clips_in_data = [c for c in variant_clips if c < n_clips]
            if not clips_in_data:
                for projection in cfg.projections:
                    direction_grid[projection][feature_space].append({
                        "A": A, "B": B, "dA": dA, "dB": dB,
                        "real_coords": np.zeros((0, 2)),
                        "real_labels": np.array([], dtype=object),
                        "path_coords": [],
                        "paths_endpoint_labels": [],
                        "n_clips": 0, "n_paths_found": 0,
                    })
                continue

            labels_grid = build_frame_labels(segments, clips_in_data, T, A, B)
            real_list, label_list = [], []
            for i, cid in enumerate(clips_in_data):
                real_list.append(feature_data[cid])
                label_list.append(labels_grid[i])
            real_X = np.concatenate(real_list, axis=0)
            real_L = np.concatenate(label_list, axis=0)
            if real_X.shape[0] > cfg.real_subsample_n:
                step = real_X.shape[0] // cfg.real_subsample_n
                keep_idx = np.arange(0, real_X.shape[0], step)[:cfg.real_subsample_n]
                real_X = real_X[keep_idx]
                real_L = real_L[keep_idx]

            paths_by_clip = load_paths_for_clips(extraction_dir, clips_in_data)
            top_paths = select_top_k_paths_with_direction(
                paths_by_clip, cfg.k_paths, segments, direction_segments,
                A, B, dA, dB, extraction_dir=extraction_dir,
            )
            n_paths_found = len(top_paths)

            path_feature_list, path_lens, endpoint_labels = [], [], []
            for cid, verts, first_lab, last_lab in top_paths:
                verts_arr = np.asarray(verts, dtype=np.int64)
                verts_arr = verts_arr[(verts_arr >= 0) & (verts_arr < T)]
                if verts_arr.size == 0:
                    path_feature_list.append(np.zeros((0, D), dtype=np.float32))
                    path_lens.append(0)
                    endpoint_labels.append((first_lab, last_lab))
                    continue
                path_feature_list.append(feature_data[cid, verts_arr])
                path_lens.append(verts_arr.size)
                endpoint_labels.append((first_lab, last_lab))

            all_path_pts = (
                np.concatenate(path_feature_list, axis=0)
                if path_feature_list else np.zeros((0, real_X.shape[1]))
            )
            for projection in cfg.projections:
                real_coords, path_coords_flat, _ = project_together(
                    real_X, all_path_pts, projection,
                    cfg.tsne_subsample_n, rng,
                )
                per_path_coords = []
                cursor = 0
                for L in path_lens:
                    per_path_coords.append(path_coords_flat[cursor:cursor + L])
                    cursor += L
                direction_grid[projection][feature_space].append({
                    "A": A, "B": B, "dA": dA, "dB": dB,
                    "real_coords": real_coords,
                    "real_labels": real_L,
                    "path_coords": per_path_coords,
                    "paths_endpoint_labels": endpoint_labels,
                    "n_clips": len(clips_in_data),
                    "n_paths_found": n_paths_found,
                })
            print(f"  variant bundle {A}-{dA}<->{B}-{dB}: "
                  f"n_clips={len(clips_in_data)}, paths={n_paths_found}")

        del feature_data

    # Combined panel: one image per projection, rows = feature spaces,
    # cols = transition types. Shows the top-1 path as a grey dashed line
    # with endpoint stars.
    for projection in cfg.projections:
        out_path = (
            Path(cfg.output_dir)
            / f"transition_panel_{projection}.pdf"
        )
        render_combined_panel_figure(
            projection=projection,
            grid_data=combined_grid[projection],
            feature_spaces=cfg.feature_spaces,
            transitions=transitions,
            cfg=cfg,
            out_path=out_path,
        )
        # Intention-only single-row version for the main paper figure.
        # Drop the verbose legend — figure 2 uses a single shared legend
        # at the bottom of the filmstrip below.
        if "intention" in cfg.feature_spaces:
            out_path_int = (
                Path(cfg.output_dir)
                / f"transition_panel_{projection}_intention.pdf"
            )
            render_combined_panel_figure(
                projection=projection,
                grid_data={"intention": combined_grid[projection]["intention"]},
                feature_spaces=["intention"],
                transitions=transitions,
                cfg=cfg,
                out_path=out_path_int,
                show_legend=False,
            )

    # Direction-variant panel: one image per projection. Rows = feature
    # spaces, cols = kept (A, B, dir_A, dir_B) variants across all kept
    # transitions. Uses the same scatter+dashed-path rendering as the
    # combined panel, with direction-filtered paths per variant.
    for projection in cfg.projections:
        out_dir_path = (
            Path(cfg.output_dir)
            / f"transition_directions_{projection}.pdf"
        )
        render_direction_panel_figure(
            projection=projection,
            grid_data=direction_grid[projection],
            feature_spaces=cfg.feature_spaces,
            variant_specs=direction_variants,
            cfg=cfg,
            out_path=out_dir_path,
        )

    # Filmstrip grid: one figure total. Rows = feature spaces, cols =
    # transitions. Each cell = MuJoCo-rendered filmstrip of the top-1 DM
    # path's frames, tinted A -> B.
    print("\nLoading raw qpos for filmstrip rendering...")
    with h5py.File(cfg.h5_path, "r") as f:
        qpos_all = np.asarray(f["qpos_rollout"][:]) if "qpos_rollout" in f else np.asarray(f["qpos"][:])
    print(f"  qpos shape: {qpos_all.shape}")

    # grid_data for filmstrips can come from any projection's combined_grid
    # (we only need top_paths_raw, not projection coords).
    any_projection = cfg.projections[0]
    filmstrip_path = Path(cfg.output_dir) / "transition_filmstrips.pdf"
    render_transition_filmstrip_figure(
        grid_data=combined_grid[any_projection],
        feature_spaces=cfg.feature_spaces,
        transitions=transitions,
        qpos_all=qpos_all,
        xml_path=cfg.xml_path,
        dataset="rodent",
        cfg=cfg,
        out_path=filmstrip_path,
    )
    # Intention-only single-row version for the main paper figure.
    # No "intention" row label — figure 2 keeps only the bottom legend.
    if "intention" in cfg.feature_spaces:
        filmstrip_path_int = Path(cfg.output_dir) / "transition_filmstrips_intention.pdf"
        render_transition_filmstrip_figure(
            grid_data={"intention": combined_grid[any_projection]["intention"]},
            feature_spaces=["intention"],
            transitions=transitions,
            qpos_all=qpos_all,
            xml_path=cfg.xml_path,
            dataset="rodent",
            cfg=cfg,
            out_path=filmstrip_path_int,
            show_row_labels=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(args.config)
