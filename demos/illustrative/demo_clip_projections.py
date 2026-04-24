"""Clip-level projection comparison: each clip / each DM member as one point.

Parallel to ``demo_feature_space_projections.py`` but aggregated one level
up. For each (DM method ∈ {DMC, DMP}) × (projection ∈ {PCA, UMAP, t-SNE})
we produce one figure. Each figure is a 2-row × 3-column grid:

  rows = [real clips, DM paths/cycles]
  cols = [intention, qpos, ekp]

Top row: 842 real clips, each clip's (500, D) trajectory flattened to a
single (500·D,)-dim vector. Clip color = dominant GT transition in that
clip (longest ``(from_behavior, to_behavior)`` span).

Bottom row: every DM cluster member from
``outputs/benchmark_experiment_transitions/w2_{dm}_{fs}_kmeans/results.pkl``.
Each member's native frame span is linearly resampled to K=100 and
flattened to (100·D,)-dim vector. Color = the cluster's auto-label with the
trailing direction token stripped (so a cluster labeled "Rear -> Immobile
Left" colors its members as "Rear -> Immobile", matching the clip-level
palette).

Rows use SEPARATE projections — the two input dims differ (500·D vs 100·D)
and each row is fit independently on its own points.
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D

from topo_mimic.benchmark.analysis.ground_truth import find_all_transitions
from topo_mimic.dmwk.utils.feature_transforms import (
    apply_dm_key_transform,
    get_hdf5_key_for_dm_key,
)


@dataclass
class DemoConfig:
    h5_path: str = "data/rodent_ar.h5"
    behavior_labels_csv: str = "behavior_labels/rodent_labels.csv"
    experiment_dir: str = "outputs/benchmark_experiment_transitions"
    xml_path: str = "topo_mimic/dmwk/xml/rodent.xml"
    stac_data_path: str = "data/rodent_reference_clips.h5"
    seed: int = 0
    window_frames: int = 100
    dm_resample_k: int = 100
    feature_spaces: list[str] = field(default_factory=lambda: ["intention", "qpos", "ekp"])
    dm_methods: list[str] = field(default_factory=lambda: ["dmc", "dmp"])
    projections: list[str] = field(default_factory=lambda: ["pca", "umap", "tsne"])
    point_size_clip: float = 18.0
    point_size_dm: float = 8.0
    output_dir: str = "outputs/demos/clip_level_projections"


FEATURE_SPACE_DISPLAY = {"intention": "Intention", "qpos": "qpos", "ekp": "EKP"}
DM_METHOD_DISPLAY = {"dmc": "DMC (cycles)", "dmp": "DMP (paths)"}
PROJECTION_DISPLAY = {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}
DIRECTION_TOKENS = {"Left", "Right", "Straight"}


def load_config(path: str | None) -> DemoConfig:
    if path is None or not os.path.exists(path):
        return DemoConfig()
    with open(path) as f:
        blob = yaml.safe_load(f) or {}
    cfg = DemoConfig()
    for k, v in blob.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def load_feature_data(h5_path: str, feature_space: str, xml_path: str,
                      stac_data_path: str) -> np.ndarray:
    hdf5_key = get_hdf5_key_for_dm_key(feature_space)
    with h5py.File(h5_path, "r") as f:
        raw = np.asarray(f[hdf5_key][:])
    return apply_dm_key_transform(raw, feature_space, xml_path=xml_path,
                                   stac_data_path=stac_data_path)


def clip_dominant_transition_label(
    transitions: list,
    n_clips: int,
) -> np.ndarray:
    """For each clip, label = the transition with the LONGEST span in the
    clip. Clips without any transition are labeled ""."""
    best_span = np.full(n_clips, -1, dtype=np.int64)
    best_label = np.full(n_clips, "", dtype=object)
    for t in transitions:
        if t.clip_id < 0 or t.clip_id >= n_clips:
            continue
        span = t.to_end - t.from_start + 1
        if span > best_span[t.clip_id]:
            best_span[t.clip_id] = span
            best_label[t.clip_id] = f"{t.from_behavior} -> {t.to_behavior}"
    return best_label


def flatten_clip_vectors(feature_data: np.ndarray) -> np.ndarray:
    """(n_clips, T, D) → (n_clips, T*D)."""
    n_clips, T, D = feature_data.shape
    return feature_data.reshape(n_clips, T * D).astype(np.float32, copy=False)


def resample_linear(arr: np.ndarray, K: int) -> np.ndarray:
    """Linear resample (T, D) → (K, D)."""
    T, D = arr.shape
    if T == K:
        return arr.astype(np.float32, copy=False)
    if T == 1:
        return np.broadcast_to(arr, (K, D)).astype(np.float32, copy=True)
    src_idx = np.linspace(0, T - 1, K, dtype=np.float64)
    lo = np.floor(src_idx).astype(np.int64)
    hi = np.clip(lo + 1, 0, T - 1)
    frac = (src_idx - lo).astype(np.float32)
    out = (1.0 - frac[:, None]) * arr[lo] + frac[:, None] * arr[hi]
    return out.astype(np.float32, copy=False)


def strip_direction_suffix(label: str) -> str:
    """'Rear -> Immobile Left' -> 'Rear -> Immobile'; 'Rear Straight' -> 'Rear'."""
    if not label:
        return ""
    toks = label.strip().split()
    if toks and toks[-1] in DIRECTION_TOKENS:
        toks = toks[:-1]
    return " ".join(toks)


def load_cluster_label_map(
    experiment_dir: str,
    dm_method: str,
    feature_space: str,
    seed: int,
) -> dict[int, str]:
    """cluster_id -> auto-label string."""
    fs_tag = "int" if feature_space == "intention" else feature_space
    method_name = f"w2_{dm_method}_{fs_tag}_kmeans"
    labels_path = Path(experiment_dir) / method_name / f"labels_seed_{seed}.yaml"
    if not labels_path.exists():
        return {}
    with open(labels_path) as f:
        blob = yaml.safe_load(f) or {}
    raw = blob.get("labels", {}) or {}
    return {int(k): str(v) for k, v in raw.items()}


def load_dm_cluster_dict(
    experiment_dir: str,
    dm_method: str,
    feature_space: str,
    seed: int,
) -> dict:
    fs_tag = "int" if feature_space == "intention" else feature_space
    method_name = f"w2_{dm_method}_{fs_tag}_kmeans"
    pkl_path = Path(experiment_dir) / method_name / "results.pkl"
    if not pkl_path.exists():
        return {}
    with open(pkl_path, "rb") as f:
        r = pickle.load(f)
    if seed not in r:
        return {}
    return r[seed].get("cluster_dict", {})


def extract_dm_member_vectors(
    feature_data: np.ndarray,
    cluster_dict: dict,
    cluster_label_map: dict[int, str],
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, labels). X[i] = flattened K×D vector for member i.
    labels[i] = member's cluster label (direction stripped)."""
    n_clips, T, D = feature_data.shape
    vectors: list[np.ndarray] = []
    labels_out: list[str] = []
    for cid, members in cluster_dict.items():
        raw_label = cluster_label_map.get(int(cid), "")
        label = strip_direction_suffix(raw_label)
        # only keep transition labels ("A -> B") in the palette; singletons
        # pass through as themselves and will likely render grey because
        # they don't match any of the top-12 transition colors.
        for clip_id, frames in members:
            ci = int(clip_id)
            if ci < 0 or ci >= n_clips:
                continue
            f_arr = np.asarray(frames, dtype=np.int64)
            if f_arr.size < 2:
                continue
            lo = int(f_arr.min())
            hi = min(T, int(f_arr.max()) + 1)
            if lo >= hi or hi - lo < 2:
                continue
            slab = feature_data[ci, lo:hi]
            res = resample_linear(slab, K)
            vectors.append(res.reshape(-1))
            labels_out.append(label)
    if not vectors:
        return (
            np.zeros((0, K * D), dtype=np.float32),
            np.array([], dtype=object),
        )
    X = np.stack(vectors, axis=0)
    return X, np.asarray(labels_out, dtype=object)


def project_points(
    X: np.ndarray,
    method: str,
) -> np.ndarray:
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2), dtype=np.float32)
    if method == "pca":
        from sklearn.decomposition import PCA
        n_comp = min(2, X.shape[0], X.shape[1])
        if n_comp < 2:
            return np.zeros((X.shape[0], 2), dtype=np.float32)
        return PCA(n_components=2, random_state=0).fit_transform(X).astype(np.float32)
    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError("umap-learn required for UMAP projection") from e
        n_neighbors = min(15, max(2, X.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=2, random_state=0, n_neighbors=n_neighbors,
            min_dist=0.1, metric="euclidean",
        )
        return reducer.fit_transform(X).astype(np.float32)
    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30.0, max(5.0, (X.shape[0] - 1) / 3.0))
        return TSNE(
            n_components=2, perplexity=perplexity, random_state=0,
            init="pca", learning_rate="auto",
        ).fit_transform(X).astype(np.float32)
    raise ValueError(f"Unknown projection: {method}")


def pick_transition_colors(labels_iter) -> tuple[list[str], dict]:
    """Build a color palette over all non-empty transition labels observed."""
    all_labels: list[str] = []
    for arr in labels_iter:
        all_labels.extend(arr.tolist())
    unique, counts = np.unique(np.asarray(all_labels, dtype=object), return_counts=True)
    pairs = [(u, c) for u, c in zip(unique, counts) if u and " -> " in u]
    pairs.sort(key=lambda p: -p[1])
    transitions = [p[0] for p in pairs]
    cmap = plt.get_cmap("tab20")
    color_map = {t: cmap(i % 20) for i, t in enumerate(transitions)}
    color_map[""] = (0.82, 0.82, 0.82, 0.5)
    return transitions, color_map


def nature_style_axis(ax: plt.Axes) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=7, width=0.8, length=3)
    ax.set_xticks([])
    ax.set_yticks([])


def scatter_projection(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    color_map: dict,
    point_size: float,
) -> None:
    nature_style_axis(ax)
    lbl_arr = np.asarray(labels, dtype=object)
    mask_other = np.array([l not in color_map or l == "" for l in lbl_arr], dtype=bool)
    if mask_other.any():
        ax.scatter(coords[mask_other, 0], coords[mask_other, 1],
                   s=point_size * 0.6, c=[color_map[""]], linewidths=0,
                   rasterized=True, zorder=1)
    for key, color in color_map.items():
        if key == "":
            continue
        m = lbl_arr == key
        if not m.any():
            continue
        ax.scatter(coords[m, 0], coords[m, 1], s=point_size,
                   c=[color], linewidths=0, edgecolors="white",
                   rasterized=True, zorder=2)


def render_figure(
    dm_method: str,
    projection: str,
    feature_spaces: list[str],
    real_coords: dict[str, np.ndarray],
    real_labels: dict[str, np.ndarray],
    dm_coords: dict[str, np.ndarray],
    dm_labels: dict[str, np.ndarray],
    transitions: list[str],
    color_map: dict,
    cfg: DemoConfig,
    out_path: str,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
    })

    n_cols = len(feature_spaces)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.2 * n_cols, 8.4))
    if n_cols == 1:
        axes = np.asarray(axes).reshape(2, 1)

    row_titles = ["Real clips\n(842 × 500·D flattened)",
                  f"{DM_METHOD_DISPLAY[dm_method]}\n(members × 100·D flattened)"]

    for col, fs in enumerate(feature_spaces):
        rc = real_coords.get(fs)
        rl = real_labels.get(fs)
        dc = dm_coords.get(fs)
        dl = dm_labels.get(fs)

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        if rc is not None and rc.size:
            scatter_projection(ax_top, rc, rl, color_map, cfg.point_size_clip)
        else:
            ax_top.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax_top.transAxes, fontsize=8, color="#777")
            nature_style_axis(ax_top)

        if dc is not None and dc.size:
            scatter_projection(ax_bot, dc, dl, color_map, cfg.point_size_dm)
        else:
            ax_bot.text(0.5, 0.5, "no DM members", ha="center", va="center",
                        transform=ax_bot.transAxes, fontsize=8, color="#777")
            nature_style_axis(ax_bot)

        ax_top.set_title(FEATURE_SPACE_DISPLAY.get(fs, fs),
                         fontsize=10, fontweight="bold", pad=6)

    for row, title in enumerate(row_titles):
        axes[row, 0].set_ylabel(title, fontsize=9, fontweight="bold", labelpad=14)

    fig.suptitle(
        f"Clip level: {DM_METHOD_DISPLAY[dm_method]}  —  "
        f"{PROJECTION_DISPLAY[projection]} projection",
        fontsize=12, fontweight="bold", y=0.99,
    )

    handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=color_map[t], markeredgecolor="none",
               markersize=7, label=t)
        for t in transitions
    ]
    ncol = min(6, max(1, len(handles)))
    fig.legend(handles=handles, loc="lower center", ncol=ncol,
               frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path} (+.pdf)")


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("Loading GT transitions...")
    transitions = find_all_transitions(
        cfg.behavior_labels_csv, merge_rear=True,
        window_frames=cfg.window_frames,
    )
    print(f"  {len(transitions)} transitions")

    feature_data: dict[str, np.ndarray] = {}
    for fs in cfg.feature_spaces:
        print(f"Loading feature space '{fs}'...")
        feature_data[fs] = load_feature_data(
            cfg.h5_path, fs, cfg.xml_path, cfg.stac_data_path,
        )
        print(f"  shape {feature_data[fs].shape}")

    n_clips = feature_data[cfg.feature_spaces[0]].shape[0]
    clip_labels = clip_dominant_transition_label(transitions, n_clips)
    print(f"  clip-level dominant-transition labels: "
          f"{sum(1 for l in clip_labels if l)} / {n_clips} clips labeled")

    # Pool label sources to build a consistent color palette.
    label_sources: list[np.ndarray] = [clip_labels]
    # Pre-load DM label maps to include in palette too.
    dm_label_preview: dict[tuple[str, str], np.ndarray] = {}
    for dm_method in cfg.dm_methods:
        for fs in cfg.feature_spaces:
            cdict = load_dm_cluster_dict(cfg.experiment_dir, dm_method, fs, cfg.seed)
            lmap = load_cluster_label_map(cfg.experiment_dir, dm_method, fs, cfg.seed)
            member_labels: list[str] = []
            for cid, members in cdict.items():
                lbl = strip_direction_suffix(lmap.get(int(cid), ""))
                member_labels.extend([lbl] * len(members))
            arr = np.asarray(member_labels, dtype=object)
            dm_label_preview[(dm_method, fs)] = arr
            label_sources.append(arr)

    transitions_order, color_map = pick_transition_colors(label_sources)
    print(f"  {len(transitions_order)} unique transition labels in palette")

    # Pre-flatten real clip vectors once per feature space.
    real_X_per_fs: dict[str, np.ndarray] = {
        fs: flatten_clip_vectors(feature_data[fs]) for fs in cfg.feature_spaces
    }
    for fs in cfg.feature_spaces:
        print(f"  {fs}: real clip matrix {real_X_per_fs[fs].shape}")

    for dm_method in cfg.dm_methods:
        print(f"\nDM method: {dm_method}")
        dm_X: dict[str, np.ndarray] = {}
        dm_L: dict[str, np.ndarray] = {}
        for fs in cfg.feature_spaces:
            cdict = load_dm_cluster_dict(cfg.experiment_dir, dm_method, fs, cfg.seed)
            lmap = load_cluster_label_map(cfg.experiment_dir, dm_method, fs, cfg.seed)
            X, L = extract_dm_member_vectors(
                feature_data[fs], cdict, lmap, cfg.dm_resample_k,
            )
            dm_X[fs] = X
            dm_L[fs] = L
            print(f"  {fs}: DM matrix {X.shape}, labels unique={len(np.unique(L))}")

        for projection in cfg.projections:
            print(f"  projection: {projection}")
            real_coords: dict[str, np.ndarray] = {}
            real_L_map: dict[str, np.ndarray] = {}
            dm_coords: dict[str, np.ndarray] = {}
            dm_L_map: dict[str, np.ndarray] = {}
            for fs in cfg.feature_spaces:
                real_coords[fs] = project_points(real_X_per_fs[fs], projection)
                real_L_map[fs] = clip_labels
                dm_coords[fs] = project_points(dm_X[fs], projection)
                dm_L_map[fs] = dm_L[fs]

            out_png = os.path.join(
                cfg.output_dir, f"proj_clip_{dm_method}_{projection}.png",
            )
            render_figure(
                dm_method=dm_method,
                projection=projection,
                feature_spaces=cfg.feature_spaces,
                real_coords=real_coords,
                real_labels=real_L_map,
                dm_coords=dm_coords,
                dm_labels=dm_L_map,
                transitions=transitions_order,
                color_map=color_map,
                cfg=cfg,
                out_path=out_png,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(args.config)
