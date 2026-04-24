"""Feature-space projection comparison: real frames vs. DM-selected frames.

For each of (DM-method ∈ {DMC, DMP}) × (projection ∈ {PCA, UMAP, t-SNE}) we
produce one figure. Each figure is a 2-row × 3-column grid:

  rows = [real frames, DM-selected frames]
  cols = [intention, qpos, ekp]

Each cell is a 2D scatter of frames projected under the named method,
colored by the GT transition each frame belongs to (per the unified
`find_all_transitions` rule with ``window_frames=100``). Frames not
inside any transition span are greyed out as "other".

Real rows are stride-subsampled before fitting/plotting so t-SNE and UMAP
stay tractable (N ≈ 10k for UMAP/PCA, N ≈ 5k for t-SNE). DM rows use all
DM-selected frames for the (dm_method, feature_space) method reading
``outputs/benchmark_experiment_transitions/w2_{dm}_{fs}_kmeans/results.pkl``
for the configured seed.

Usage:
  python -m demos.illustrative.demo_feature_space_projections
  python -m demos.illustrative.demo_feature_space_projections --config \\
    configs/demos/illustrative/demo_feature_space_projections.yaml
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

from topo_mimic.analysis.feature_analysis.ground_truth import find_all_transitions
from topo_mimic.extraction.utils.feature_transforms import (
    apply_dm_key_transform,
    get_hdf5_key_for_dm_key,
)


@dataclass
class DemoConfig:
    h5_path: str = "data/rodent_ar.h5"
    behavior_labels_csv: str = "behavior_labels/rodent_labels.csv"
    experiment_dir: str = "outputs/benchmark_experiment_transitions"
    xml_path: str = "topo_mimic/extraction/xml/rodent.xml"
    stac_data_path: str = "data/rodent_reference_clips.h5"
    seed: int = 0
    window_frames: int = 100
    feature_spaces: list[str] = field(default_factory=lambda: ["intention", "qpos", "ekp"])
    dm_methods: list[str] = field(default_factory=lambda: ["dmc", "dmp"])
    projections: list[str] = field(default_factory=lambda: ["pca", "umap", "tsne"])
    real_subsample_n: int = 10000
    tsne_subsample_n: int = 5000
    point_size: float = 2.0
    point_size_dm: float = 5.0
    output_dir: str = "outputs/demos/feature_space_projections"


FEATURE_SPACE_DISPLAY = {"intention": "Intention", "qpos": "qpos", "ekp": "EKP"}
DM_METHOD_DISPLAY = {"dmc": "DMC (cycles)", "dmp": "DMP (paths)"}
PROJECTION_DISPLAY = {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}


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
    """Return (N_clips, T, D) feature array for the given space."""
    hdf5_key = get_hdf5_key_for_dm_key(feature_space)
    with h5py.File(h5_path, "r") as f:
        raw = np.asarray(f[hdf5_key][:])
    return apply_dm_key_transform(raw, feature_space, xml_path=xml_path,
                                   stac_data_path=stac_data_path)


def build_frame_label_map(
    transitions: list,
    n_clips: int,
    n_frames: int,
) -> np.ndarray:
    """Per-(clip, frame) transition label. Shape (n_clips, n_frames), dtype=object.

    A frame gets the transition label of the FIRST TransitionSegment whose
    ``[from_start, to_end]`` span contains it. Frames outside any span → "".
    """
    labels = np.full((n_clips, n_frames), "", dtype=object)
    for t in transitions:
        if t.clip_id >= n_clips:
            continue
        lo = max(0, t.from_start)
        hi = min(n_frames, t.to_end + 1)
        if lo >= hi:
            continue
        key = f"{t.from_behavior} -> {t.to_behavior}"
        mask = labels[t.clip_id, lo:hi] == ""
        labels[t.clip_id, lo:hi] = np.where(mask, key, labels[t.clip_id, lo:hi])
    return labels


def stride_subsample(
    n_clips: int,
    n_frames: int,
    n_total: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (clip_idx, frame_idx) arrays sampling ~n_total frames by stride."""
    total_frames = n_clips * n_frames
    stride = max(1, total_frames // n_total)
    flat_idx = np.arange(0, total_frames, stride, dtype=np.int64)
    clip_idx = (flat_idx // n_frames).astype(np.int64)
    frame_idx = (flat_idx % n_frames).astype(np.int64)
    return clip_idx, frame_idx


def load_dm_frames(
    experiment_dir: str,
    dm_method: str,
    feature_space: str,
    seed: int,
) -> list[tuple[int, int]]:
    """Return a flat list of (clip_id, frame_idx) tuples for DM-selected frames."""
    # Convention: experiment directories use "int" for intention.
    fs_tag = "int" if feature_space == "intention" else feature_space
    method_name = f"w2_{dm_method}_{fs_tag}_kmeans"
    pkl_path = Path(experiment_dir) / method_name / "results.pkl"
    if not pkl_path.exists():
        print(f"  [warn] {pkl_path} missing; returning no DM frames")
        return []
    with open(pkl_path, "rb") as f:
        r = pickle.load(f)
    if seed not in r:
        print(f"  [warn] seed {seed} not in {pkl_path}; available {list(r.keys())}")
        return []
    cluster_dict = r[seed].get("cluster_dict", {})
    out: list[tuple[int, int]] = []
    for _cid, members in cluster_dict.items():
        for clip_id, frames in members:
            for f in np.asarray(frames, dtype=np.int64):
                out.append((int(clip_id), int(f)))
    return out


def project_points(
    X: np.ndarray,
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=0).fit_transform(X).astype(np.float32)
    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError("umap-learn required for UMAP projection") from e
        reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=15,
                            min_dist=0.1, metric="euclidean")
        return reducer.fit_transform(X).astype(np.float32)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(
            n_components=2, perplexity=min(30, max(5, X.shape[0] // 30)),
            random_state=0, init="pca", learning_rate="auto",
        ).fit_transform(X).astype(np.float32)
    raise ValueError(f"Unknown projection: {method}")


def pick_transition_colors(
    all_labels: np.ndarray,
) -> tuple[list[str], dict[str, tuple[float, float, float, float]]]:
    """Return (ordered_transition_list, color_map)."""
    unique, counts = np.unique(all_labels, return_counts=True)
    pairs = [(u, c) for u, c in zip(unique, counts) if u != ""]
    pairs.sort(key=lambda p: -p[1])
    transitions = [p[0] for p in pairs]
    cmap = plt.get_cmap("tab20")
    color_map: dict[str, tuple[float, float, float, float]] = {
        t: cmap(i % 20) for i, t in enumerate(transitions)
    }
    color_map[""] = (0.82, 0.82, 0.82, 0.5)  # grey "other"
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
    lbl_arr = np.asarray(labels)
    # grey background for "other" first
    mask_other = lbl_arr == ""
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
                   c=[color], linewidths=0, rasterized=True, zorder=2)


def render_figure(
    dm_method: str,
    projection: str,
    feature_spaces: list[str],
    real_coords: dict[str, np.ndarray],      # fs -> (n, 2)
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

    row_titles = ["Real frames", f"{DM_METHOD_DISPLAY[dm_method]} frames"]

    for col, fs in enumerate(feature_spaces):
        rc = real_coords.get(fs)
        rl = real_labels.get(fs)
        dc = dm_coords.get(fs)
        dl = dm_labels.get(fs)

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        if rc is not None and rc.size:
            scatter_projection(ax_top, rc, rl, color_map, cfg.point_size)
        else:
            ax_top.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax_top.transAxes, fontsize=8, color="#777")
            nature_style_axis(ax_top)

        if dc is not None and dc.size:
            scatter_projection(ax_bot, dc, dl, color_map, cfg.point_size_dm)
        else:
            ax_bot.text(0.5, 0.5, "no DM frames", ha="center", va="center",
                        transform=ax_bot.transAxes, fontsize=8, color="#777")
            nature_style_axis(ax_bot)

        ax_top.set_title(FEATURE_SPACE_DISPLAY.get(fs, fs), fontsize=10,
                         fontweight="bold", pad=6)

    for row, title in enumerate(row_titles):
        axes[row, 0].set_ylabel(title, fontsize=10, fontweight="bold",
                                labelpad=12)

    fig.suptitle(
        f"{DM_METHOD_DISPLAY[dm_method]}  —  {PROJECTION_DISPLAY[projection]} projection",
        fontsize=12, fontweight="bold", y=0.99,
    )

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color_map[t],
               markeredgecolor="none", markersize=7, label=t)
        for t in transitions
    ]
    ncol = min(6, max(1, len(handles)))
    fig.legend(handles=handles, loc="lower center",
               ncol=ncol, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path} (+.pdf)")


def prepare_real_subsample(
    feature_data: np.ndarray,
    frame_label_map: np.ndarray,
    n_target: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stride-subsample feature_data (n_clips, T, D) to ~n_target frames."""
    n_clips, n_frames, _D = feature_data.shape
    clip_idx, frame_idx = stride_subsample(n_clips, n_frames, n_target)
    X = feature_data[clip_idx, frame_idx]
    labels = frame_label_map[clip_idx, frame_idx]
    return X, labels


def prepare_dm_subset(
    feature_data: np.ndarray,
    dm_frames: list[tuple[int, int]],
    frame_label_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull DM-selected frames' feature vectors and labels."""
    if not dm_frames:
        return np.zeros((0, feature_data.shape[-1]), dtype=feature_data.dtype), \
               np.array([], dtype=object)
    clip_idx = np.asarray([c for c, _ in dm_frames], dtype=np.int64)
    frame_idx = np.asarray([f for _, f in dm_frames], dtype=np.int64)
    n_clips, n_frames, _ = feature_data.shape
    keep = (clip_idx < n_clips) & (frame_idx < n_frames)
    clip_idx = clip_idx[keep]
    frame_idx = frame_idx[keep]
    X = feature_data[clip_idx, frame_idx]
    labels = frame_label_map[clip_idx, frame_idx]
    return X, labels


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    os.makedirs(cfg.output_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    print("Loading GT transitions...")
    transitions = find_all_transitions(
        cfg.behavior_labels_csv, merge_rear=True,
        window_frames=cfg.window_frames,
    )
    print(f"  {len(transitions)} transitions")

    # One feature_data per space (cached locally).
    feature_data_cache: dict[str, np.ndarray] = {}
    frame_label_cache: dict[str, np.ndarray] = {}
    for fs in cfg.feature_spaces:
        print(f"Loading feature space '{fs}'...")
        data = load_feature_data(cfg.h5_path, fs, cfg.xml_path, cfg.stac_data_path)
        print(f"  shape {data.shape}")
        feature_data_cache[fs] = data
        frame_label_cache[fs] = build_frame_label_map(
            transitions, data.shape[0], data.shape[1],
        )

    # Pick colors over labels actually present anywhere.
    pooled_labels = np.concatenate(
        [lbl_map.ravel() for lbl_map in frame_label_cache.values()]
    )
    transitions_order, color_map = pick_transition_colors(pooled_labels)
    print(f"  {len(transitions_order)} unique transition labels")

    # Pre-subsample real data per feature space (same indices used across
    # all projection methods so colors are consistent).
    real_X: dict[str, np.ndarray] = {}
    real_L: dict[str, np.ndarray] = {}
    for fs in cfg.feature_spaces:
        X, L = prepare_real_subsample(
            feature_data_cache[fs], frame_label_cache[fs],
            cfg.real_subsample_n,
        )
        real_X[fs] = X
        real_L[fs] = L
        print(f"  {fs}: real subsample N={X.shape[0]} D={X.shape[1]}")

    for dm_method in cfg.dm_methods:
        print(f"\nDM method: {dm_method}")
        # DM frames per feature space
        dm_X: dict[str, np.ndarray] = {}
        dm_L: dict[str, np.ndarray] = {}
        for fs in cfg.feature_spaces:
            frames = load_dm_frames(cfg.experiment_dir, dm_method, fs, cfg.seed)
            X, L = prepare_dm_subset(
                feature_data_cache[fs], frames, frame_label_cache[fs],
            )
            dm_X[fs] = X
            dm_L[fs] = L
            print(f"  {fs}: DM subset N={X.shape[0]} D={X.shape[1]}")

        for projection in cfg.projections:
            print(f"  projection: {projection}")
            # For t-SNE, subsample further.
            real_coords: dict[str, np.ndarray] = {}
            real_L_proj: dict[str, np.ndarray] = {}
            dm_coords: dict[str, np.ndarray] = {}
            dm_L_proj: dict[str, np.ndarray] = {}
            for fs in cfg.feature_spaces:
                X_r = real_X[fs]
                L_r = real_L[fs]
                if projection == "tsne" and X_r.shape[0] > cfg.tsne_subsample_n:
                    ridx = rng.choice(X_r.shape[0], cfg.tsne_subsample_n, replace=False)
                    X_r = X_r[ridx]
                    L_r = L_r[ridx]
                if X_r.shape[0] >= 2:
                    real_coords[fs] = project_points(X_r, projection, rng)
                    real_L_proj[fs] = L_r
                else:
                    real_coords[fs] = np.zeros((0, 2), dtype=np.float32)
                    real_L_proj[fs] = np.array([], dtype=object)

                X_d = dm_X[fs]
                L_d = dm_L[fs]
                if projection == "tsne" and X_d.shape[0] > cfg.tsne_subsample_n:
                    didx = rng.choice(X_d.shape[0], cfg.tsne_subsample_n, replace=False)
                    X_d = X_d[didx]
                    L_d = L_d[didx]
                if X_d.shape[0] >= 2:
                    dm_coords[fs] = project_points(X_d, projection, rng)
                    dm_L_proj[fs] = L_d
                else:
                    dm_coords[fs] = np.zeros((0, 2), dtype=np.float32)
                    dm_L_proj[fs] = np.array([], dtype=object)

            out_png = os.path.join(
                cfg.output_dir, f"proj_{dm_method}_{projection}.png",
            )
            render_figure(
                dm_method=dm_method,
                projection=projection,
                feature_spaces=cfg.feature_spaces,
                real_coords=real_coords,
                real_labels=real_L_proj,
                dm_coords=dm_coords,
                dm_labels=dm_L_proj,
                transitions=transitions_order,
                color_map=color_map,
                cfg=cfg,
                out_path=out_png,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config (optional)")
    args = parser.parse_args()
    main(args.config)
