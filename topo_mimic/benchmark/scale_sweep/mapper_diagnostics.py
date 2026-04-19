"""Mapper-specific fit-quality diagnostics.

Two visualizations per Mapper variant:
  1. `graph_gt_colored.png` — the Mapper graph in 2D (force-directed), with each
     node colored by the GT behavior that dominates the frames inside it. The
     analog to KPMS's "predicted vs real signal" overlap: shows whether Mapper
     carved the data into behaviorally-coherent regions.
  2. `lens_histogram.png` — the lens-value distribution. A flat / wide histogram
     means the lens is informative; a single spike means the cover degenerates.

These supplement the per-clip trace PNGs produced by `plots.save_trace_png`,
which work for both KPMS and Mapper.
"""

from __future__ import annotations

import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from topo_mimic.benchmark.auto_label import BehaviorSegment

log = logging.getLogger(__name__)

_TAB20 = plt.get_cmap("tab20").colors


def reconstruct_labels_list_from_cluster_dict(
    cluster_dict_path: str | Path,
    clip_ids: list[int],
    n_frames_per_clip: int,
    out_path: str | Path,
) -> None:
    """Self-heal: rebuild labels_list.pkl from cluster_dict.pkl when missing.

    cluster_dict format: {cluster_id -> [(clip_id, frame_indices), ...]} with
    absolute clip_ids. Output: list aligned with clip_ids, each entry an (n_frames,)
    int array where the entry is the cluster_id active at that frame (or -1).
    """
    with Path(cluster_dict_path).open("rb") as f:
        cluster_dict = pickle.load(f)
    cid_to_idx = {cid: i for i, cid in enumerate(clip_ids)}
    labels_list = [
        np.full(n_frames_per_clip, -1, dtype=np.int64) for _ in clip_ids
    ]
    for cluster_id, members in cluster_dict.items():
        for clip_id, frames in members:
            i = cid_to_idx.get(int(clip_id))
            if i is None:
                continue
            labels_list[i][np.asarray(frames)] = int(cluster_id)
    with Path(out_path).open("wb") as f:
        pickle.dump(labels_list, f)
    log.info(f"reconstructed labels_list.pkl -> {out_path}")


def _node_dominant_behavior(
    members: list[tuple[int, np.ndarray]],
    segments: dict[int, list[BehaviorSegment]],
) -> tuple[str, int, int]:
    """Return (dominant_behavior, n_frames_for_dominant, n_frames_with_gt)."""
    counter: Counter[str] = Counter()
    n_with_gt = 0
    for clip_id, frames in members:
        clip_segs = segments.get(int(clip_id), [])
        if not clip_segs:
            continue
        starts = np.array([s.start for s in clip_segs])
        ends = np.array([s.end for s in clip_segs])
        for f in np.asarray(frames):
            mask = (starts <= f) & (ends > f)
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                continue
            counter[clip_segs[int(idxs[0])].behavior] += 1
            n_with_gt += 1
    if not counter:
        return "<no-gt>", 0, 0
    top, top_count = counter.most_common(1)[0]
    return top, top_count, n_with_gt


def _behavior_color_map(all_behaviors: list[str]) -> dict[str, tuple]:
    ordered = sorted(set(all_behaviors))
    cmap = dict(zip(ordered, _TAB20[: len(ordered)]))
    cmap.setdefault("<no-gt>", (0.7, 0.7, 0.7))
    return cmap


def save_mapper_graph_gt_colored_png(
    graph_path: str | Path,
    cluster_dict_path: str | Path,
    segments: dict[int, list[BehaviorSegment]],
    out_path: Path,
    title: str,
    behavior_colors: dict[str, tuple],
) -> None:
    """Render the Mapper graph in 2D with nodes colored by dominant GT behavior."""
    import networkx as nx

    with Path(graph_path).open("rb") as f:
        graph_blob = pickle.load(f)
    with Path(cluster_dict_path).open("rb") as f:
        cluster_dict = pickle.load(f)

    G = nx.Graph()
    for node_id, attrs in graph_blob["nodes"]:
        G.add_node(int(node_id), **attrs)
    G.add_edges_from((int(u), int(v)) for u, v in graph_blob["edges"])

    node_dominant: dict[int, str] = {}
    node_purity: dict[int, float] = {}
    node_size: dict[int, int] = {}
    for cid, members in cluster_dict.items():
        dom, dom_count, n_gt = _node_dominant_behavior(members, segments)
        node_dominant[int(cid)] = dom
        node_purity[int(cid)] = dom_count / max(n_gt, 1)
        node_size[int(cid)] = sum(int(f.size) for _, f in members)

    for nid in G.nodes:
        if nid not in node_dominant:
            node_dominant[nid] = "<pruned>"
            node_purity[nid] = 0.0
            node_size[nid] = 0

    pos = nx.spring_layout(G, seed=0, k=0.4)

    fig, ax = plt.subplots(figsize=(10, 8))

    sizes = np.array([node_size.get(nid, 0) for nid in G.nodes], dtype=float)
    s_min, s_max = max(20.0, sizes.min() if sizes.size else 20.0), max(200.0, sizes.max() if sizes.size else 200.0)
    if s_max > s_min:
        sizes_scaled = 20.0 + 480.0 * (sizes - s_min) / (s_max - s_min)
    else:
        sizes_scaled = np.full_like(sizes, 60.0)

    colors = []
    for nid in G.nodes:
        d = node_dominant.get(nid, "<no-gt>")
        if d == "<pruned>":
            colors.append((0.85, 0.85, 0.85))
        else:
            colors.append(behavior_colors.get(d, (0.5, 0.5, 0.5)))

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", width=0.5, alpha=0.4)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=sizes_scaled.tolist(), node_color=colors,
        edgecolors="black", linewidths=0.4,
    )

    legend_handles = []
    for behav, color in behavior_colors.items():
        legend_handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                         markerfacecolor=color, markersize=8, label=behav))
    legend_handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor=(0.85, 0.85, 0.85), markersize=8, label="<pruned>"))
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=7, frameon=False)

    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _compute_pca_2d_for_feature_space(
    data: np.ndarray, takens_m: int, takens_tau: int,
) -> np.ndarray:
    """Takens-embed per clip, concatenate, PCA to 2D. Returns (N_points, 2)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    from topo_mimic.benchmark.mapper.mapper_method import _takens_embedding

    n_clips = data.shape[0]
    pts_list = []
    for i in range(n_clips):
        pts, _ = _takens_embedding(data[i], m=takens_m, tau=takens_tau)
        pts_list.append(pts)
    all_pts = np.concatenate(pts_list, axis=0)
    pipe = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=2))])
    return pipe.fit_transform(all_pts)


def save_mapper_graph_pca_overlay_png(
    graph_path: str | Path,
    cluster_dict_path: str | Path,
    pca_2d: np.ndarray,
    segments: dict[int, list[BehaviorSegment]],
    out_path: Path,
    title: str,
    behavior_colors: dict[str, tuple],
) -> None:
    """Render the Mapper graph overlaid on the real-data PCA-2D scatter.

    Background: all Takens-embedded points scattered in PCA-2D, gray + transparent.
    Foreground: one marker per Mapper node, placed at the PCA-2D centroid of its
    member points, colored by the node's dominant GT behavior, sized by frame count.
    Edges: drawn between node centroids using the Mapper graph connectivity.
    """
    import networkx as nx

    with Path(graph_path).open("rb") as f:
        graph_blob = pickle.load(f)
    with Path(cluster_dict_path).open("rb") as f:
        cluster_dict = pickle.load(f)

    G = nx.Graph()
    for node_id, attrs in graph_blob["nodes"]:
        G.add_node(int(node_id), **attrs)
    G.add_edges_from((int(u), int(v)) for u, v in graph_blob["edges"])

    node_centroid: dict[int, np.ndarray] = {}
    node_dominant: dict[int, str] = {}
    node_size: dict[int, int] = {}

    for nid, attrs in G.nodes(data=True):
        ids = np.asarray(attrs.get("ids", []), dtype=int)
        if ids.size:
            node_centroid[nid] = pca_2d[ids].mean(axis=0)
        else:
            node_centroid[nid] = pca_2d.mean(axis=0)
        node_dominant[nid] = "<pruned>"
        node_size[nid] = 0

    for cid, members in cluster_dict.items():
        dom, dom_count, n_gt = _node_dominant_behavior(members, segments)
        node_dominant[int(cid)] = dom
        node_size[int(cid)] = sum(int(f.size) for _, f in members)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pca_2d[:, 0], pca_2d[:, 1], s=2, color="lightgray", alpha=0.25,
               linewidths=0, zorder=1)

    for u, v in G.edges:
        xu, yu = node_centroid[u]
        xv, yv = node_centroid[v]
        ax.plot([xu, xv], [yu, yv], color="gray", lw=0.6, alpha=0.6, zorder=2)

    sizes = np.array([node_size[n] for n in G.nodes], dtype=float)
    s_max = max(sizes.max(), 1.0)
    sizes_scaled = 40.0 + 460.0 * sizes / s_max

    centroids = np.array([node_centroid[n] for n in G.nodes])
    node_colors = []
    for n in G.nodes:
        d = node_dominant[n]
        if d == "<pruned>":
            node_colors.append((0.85, 0.85, 0.85))
        else:
            node_colors.append(behavior_colors.get(d, (0.5, 0.5, 0.5)))

    ax.scatter(centroids[:, 0], centroids[:, 1], s=sizes_scaled,
               c=node_colors, edgecolors="black", linewidths=0.5, zorder=3)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markersize=8, label=b)
        for b, c in behavior_colors.items()
    ]
    legend_handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor=(0.85, 0.85, 0.85),
                                     markersize=8, label="<pruned>"))
    ax.legend(handles=legend_handles, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_per_clip_pca_overlay_png(
    clip_id: int,
    clip_data: np.ndarray,
    takens_m: int,
    takens_tau: int,
    predicted_labels_per_frame: np.ndarray,
    gt_segments: list[BehaviorSegment],
    out_path: Path,
    title: str,
    behavior_colors: dict[str, tuple],
) -> None:
    """One clip's Takens trajectory in its own PCA-2D, side by side: GT vs predicted node.

    Left panel: each Takens-embedded point coloured by the GT behaviour at its frame.
    Right panel: same scatter coloured by Mapper node id at that frame.

    Showing them side-by-side makes it visually obvious whether the Mapper node
    partition tracks the behavioural partition for this clip.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    from topo_mimic.benchmark.mapper.mapper_method import _takens_embedding

    pts, idxs = _takens_embedding(clip_data, m=takens_m, tau=takens_tau)
    if pts.shape[0] < 3:
        return
    n_components = 2 if pts.shape[1] >= 2 else 1
    pipe = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=n_components))])
    proj = pipe.fit_transform(pts)
    if n_components == 1:
        proj = np.column_stack([proj, np.zeros_like(proj)])

    starts = np.array([s.start for s in gt_segments])
    ends = np.array([s.end for s in gt_segments])
    behaviors_arr = np.array([s.behavior for s in gt_segments])
    gt_labels = np.full(len(idxs), "<no-gt>", dtype=object)
    for i, frame_idx in enumerate(idxs):
        mask = (starts <= frame_idx) & (ends > frame_idx)
        m_idxs = np.where(mask)[0]
        if m_idxs.size:
            gt_labels[i] = behaviors_arr[int(m_idxs[0])]

    pred_labels = predicted_labels_per_frame[idxs]

    gt_colors_arr = np.array([
        behavior_colors.get(b, (0.85, 0.85, 0.85)) for b in gt_labels
    ])
    unique_pred = sorted(set(int(p) for p in pred_labels))
    pred_color_map = {p: _TAB20[i % 20] for i, p in enumerate(unique_pred)}
    pred_color_map[-1] = (0.85, 0.85, 0.85)
    pred_colors_arr = np.array([pred_color_map[int(p)] for p in pred_labels])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax in axes:
        ax.plot(proj[:, 0], proj[:, 1], color="lightgray", lw=0.5, alpha=0.5, zorder=1)
    axes[0].scatter(proj[:, 0], proj[:, 1], c=gt_colors_arr, s=18,
                    edgecolors="black", linewidths=0.2, zorder=2)
    axes[0].set_title("GT behaviour", fontsize=10)
    axes[1].scatter(proj[:, 0], proj[:, 1], c=pred_colors_arr, s=18,
                    edgecolors="black", linewidths=0.2, zorder=2)
    axes[1].set_title("Mapper node id", fontsize=10)

    legend_handles_gt = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markersize=8, label=b)
        for b, c in behavior_colors.items()
        if b in set(gt_labels.tolist())
    ]
    if "<no-gt>" in gt_labels.tolist():
        legend_handles_gt.append(plt.Line2D([0], [0], marker="o", color="w",
                                            markerfacecolor=(0.85, 0.85, 0.85),
                                            markersize=8, label="<no-gt>"))
    axes[0].legend(handles=legend_handles_gt, loc="best", fontsize=7, frameon=False)

    legend_handles_pred = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=pred_color_map[p], markersize=8, label=f"node {p}")
        for p in unique_pred[:12]
    ]
    if len(unique_pred) > 12:
        legend_handles_pred.append(plt.Line2D([0], [0], marker="o", color="w",
                                              markerfacecolor="black", markersize=8,
                                              label=f"... +{len(unique_pred)-12} more"))
    axes[1].legend(handles=legend_handles_pred, loc="best", fontsize=7, frameon=False)

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_lens_histogram_png(
    lens_values_path: str | Path,
    out_path: Path,
    title: str,
    n_intervals: int,
) -> None:
    """Lens-value distribution; vertical lines at the cover bin edges."""
    lens = np.load(lens_values_path)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(lens, bins=50, color="steelblue", alpha=0.85, edgecolor="black", linewidth=0.3)
    bin_edges = np.linspace(lens.min(), lens.max(), n_intervals + 1)
    for e in bin_edges:
        ax.axvline(e, color="red", lw=0.5, alpha=0.5)
    ax.set_xlabel("lens value")
    ax.set_ylabel("count")
    ax.set_title(title + f"  (std={lens.std():.3f}, range=[{lens.min():.3f}, {lens.max():.3f}])",
                 fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def generate_mapper_diagnostics(
    variant_id: str,
    feature_space: str,
    lens_name: str,
    n_intervals: int,
    variant_dir: Path,
    segments: dict[int, list[BehaviorSegment]],
    behavior_colors: dict[str, tuple],
    out_root: Path,
    pca_2d: np.ndarray | None = None,
) -> list[Path]:
    """Emit the diagnostic PNGs for one Mapper variant.

    Three files when `pca_2d` is provided:
      - graph_gt_colored.png      (force-directed layout)
      - graph_gt_colored_pca.png  (overlaid on real-data PCA-2D)
      - lens_histogram.png
    """
    out_dir = out_root / variant_id
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    graph_path = variant_dir / "graph.pkl"
    cluster_dict_path = variant_dir / "cluster_dict.pkl"
    lens_path = variant_dir / "lens_values.npy"

    graph_png = out_dir / "graph_gt_colored.png"
    if graph_path.exists() and cluster_dict_path.exists() and not graph_png.exists():
        save_mapper_graph_gt_colored_png(
            graph_path=graph_path,
            cluster_dict_path=cluster_dict_path,
            segments=segments,
            out_path=graph_png,
            title=f"{variant_id}  | nodes coloured by dominant GT behaviour",
            behavior_colors=behavior_colors,
        )
        written.append(graph_png)

    pca_png = out_dir / "graph_gt_colored_pca.png"
    if (
        pca_2d is not None
        and graph_path.exists()
        and cluster_dict_path.exists()
        and not pca_png.exists()
    ):
        save_mapper_graph_pca_overlay_png(
            graph_path=graph_path,
            cluster_dict_path=cluster_dict_path,
            pca_2d=pca_2d,
            segments=segments,
            out_path=pca_png,
            title=f"{variant_id}  | nodes on PCA-2D of Takens-embedded data",
            behavior_colors=behavior_colors,
        )
        written.append(pca_png)

    lens_png = out_dir / "lens_histogram.png"
    if lens_path.exists() and not lens_png.exists():
        save_lens_histogram_png(
            lens_values_path=lens_path,
            out_path=lens_png,
            title=f"{variant_id}  | lens={lens_name}",
            n_intervals=n_intervals,
        )
        written.append(lens_png)

    return written
