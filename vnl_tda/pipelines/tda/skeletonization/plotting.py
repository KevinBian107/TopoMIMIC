"""Functions for plotting the skeletons and post-processed clusters"""

from typing import Any
import numpy as np
import os
import random
import h5py

import math
from scipy.spatial.distance import cdist, squareform
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import linkage
from scipy.linalg import subspace_angles, eigh
from sklearn.cluster import AgglomerativeClustering
from collections import Counter, defaultdict

import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import ot

from vnl_tda.utils.load_utils import (
    load_from_h5py,
    get_aggregate_data,
    extract_clip_info,
    load_rollout_from_source,
)


def _load_dimo_edges(path: str) -> np.ndarray:
    out = []
    with open(path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 3:
                continue
            out.append((int(p[0]), int(p[1]), int(p[2])))
    return np.array(out, dtype=int)


def _load_edges_first2(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 2:
                continue
            try:
                a, b = int(p[0]), int(p[1])
                out.append((a, b))
            except ValueError:
                continue
    return np.array(out, dtype=int) if out else np.empty((0, 2), dtype=int)


def _find_feature_file(dirpath):
    cand = []
    for fn in os.listdir(dirpath):
        low = fn.lower()
        if any(k in low for k in ["feature", "point", "pts"]) and (low.endswith(".txt") or low.endswith(".npy")):
            cand.append(os.path.join(dirpath, fn))
    if not cand:
        raise FileNotFoundError(f"No feature file found in {dirpath}")
    cand.sort()
    return cand[0]


def _load_features(path):
    if path.lower().endswith(".npy"):
        return np.load(path)
    return np.loadtxt(path)


def _embed_tsne_3d(X, perplexity=None, early_exaggeration=12.0, n_iter=1000, random_state=42):
    N = X.shape[0]
    if perplexity is None:
        perplexity = max(5.0, min(30.0, (N - 1) / 3.0))
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(
            n_components=3,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            n_iter=n_iter,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
            metric="euclidean",
            verbose=0,
        )
        return tsne.fit_transform(X)
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(n_components=3, random_state=random_state).fit_transform(X)


def _segments_from_edges(coords, edgelist):
    if not edgelist:
        return np.empty((0, 2, 3))
    keep = [(u, v) for (u, v) in edgelist if (u in coords) and (v in coords)]
    if not keep:
        return np.empty((0, 2, 3))
    a = np.array([coords[u] for u, v in keep])
    b = np.array([coords[v] for u, v in keep])
    return np.stack([a, b], axis=1)


def _build_undirected_skeleton_from_dimo(dimo_edges_array, n_nodes=None):
    if dimo_edges_array.ndim == 1:
        dimo_edges_array = dimo_edges_array.reshape(1, -1)
    uv = dimo_edges_array[:, :2].astype(int)
    if n_nodes is None:
        n_nodes = int(np.max(uv)) + 1 if uv.size else 0
    UG = nx.Graph()
    UG.add_nodes_from(range(n_nodes))
    for u, v in uv:
        if u != v and 0 <= u < n_nodes and 0 <= v < n_nodes:
            UG.add_edge(int(u), int(v))
    return UG


def _two_core(G):
    try:
        return nx.k_core(G, k=2)
    except nx.NetworkXError:
        return G.copy()


def _find_cycles_mcb(UG, min_len=3, max_cycles=None):
    cycles = []
    for comp_nodes in sorted(nx.connected_components(UG), key=len, reverse=True):
        H = UG.subgraph(comp_nodes).copy()
        if H.number_of_edges() == 0:
            continue
        mcb = nx.minimum_cycle_basis(H)
        for c in mcb:
            if len(c) >= min_len:
                cycles.append(c)
                if max_cycles is not None and len(cycles) >= max_cycles:
                    return cycles
    return cycles


def _segments3d_from_cycle(coords, cyc):
    seq = [n for n in cyc if n in coords]
    if len(seq) < 2:
        return np.empty((0, 2, 3))
    a = np.array([coords[u] for u in seq])
    b = np.array([coords[v] for v in seq[1:] + [seq[0]]])
    return np.stack([a, b], axis=1)


def _frame_label_vote(cluster_dict):
    """
    Majority vote helper
    """
    # cluster_dict: cid → list of (clip_id, frames_array)
    votes = defaultdict(list)
    for cid, wins in cluster_dict.items():
        for clip, frs in wins:
            for f in frs:
                votes[(clip, f)].append(cid)

    # pick the most‐common cid for each frame
    frame_labels = {}
    for (clip, f), clist in votes.items():
        frame_labels[(clip, f)] = Counter(clist).most_common(1)[0][0]
    return frame_labels


def plot_dimo(
    edge_file: str,
    feature_dir: str,
    figsize: tuple[int, int] = (10, 10),
    node_size: int = 8,
    tsne_kwargs: dict | None = None,
) -> tuple[nx.DiGraph, dict[int, np.ndarray], np.ndarray]:
    """
    Plot discrete Morse edges on a t-SNE surface.

    Args:
        edge_file: Path to the edge file from DM computation
        feature_dir: Directory containing feature/point files
        figsize: Figure size
        node_size: Size of scatter points
        tsne_kwargs: Additional kwargs for t-SNE embedding

    Returns:
        Tuple of (graph, coords_dict, coords3d_array)
    """
    
    edges = _load_dimo_edges(edge_file)
    feat_path = _find_feature_file(feature_dir)
    X = _load_features(feat_path)
    tsne_kwargs = tsne_kwargs or {}
    coords3d = _embed_tsne_3d(X, **tsne_kwargs)
    nodes = np.unique(edges[:, :2].ravel())
    valid = nodes[nodes < len(coords3d)]
    keep = np.isin(edges[:,0], valid) & np.isin(edges[:,1], valid)
    edges = edges[keep]
    asc  = [(a,b) for a,b,t in edges if t == 1]
    desc = [(a,b) for a,b,t in edges if t == -1]
    crit = [(a,b) for a,b,t in edges if t == 2]
    n_desc = int(np.sum(edges[:,2] == -1))
    n_asc  = int(np.sum(edges[:,2] == 1))
    n_crit = int(np.sum(edges[:,2] == 2))
    print(f"Counts -> -1: {n_desc}, +1: {n_asc}, 2: {n_crit}")
    morse_only_path = os.path.join(os.path.dirname(feat_path), "edge_for_morse_only.txt")
    morse_edges = _load_edges_first2(morse_only_path) if os.path.exists(morse_only_path) else np.empty((0,2), dtype=int)
    if morse_edges.size > 0:
        morse_valid_mask = (morse_edges[:,0] < len(coords3d)) & (morse_edges[:,1] < len(coords3d))
        morse_edges = morse_edges[morse_valid_mask]
        morse_edgelist = [tuple(e) for e in morse_edges if (e[0] in valid) and (e[1] in valid)]
    else:
        morse_edgelist = []
    G = nx.DiGraph()
    G.add_nodes_from(valid.tolist())
    G.add_edges_from([(a,b) for a,b,_ in edges])
    coords = {i: coords3d[i] for i in valid.tolist()}
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if morse_edgelist:
        seg_morse = _segments_from_edges(coords, morse_edgelist)
        if len(seg_morse):
            lc_morse = Line3DCollection(seg_morse, colors="lightgray", linewidths=0.5, alpha=0.35)
            ax.add_collection3d(lc_morse)
    if asc:
        seg_asc = _segments_from_edges(coords, asc)
        if len(seg_asc):
            lc_asc = Line3DCollection(seg_asc, colors="tab:blue", linewidths=1.0, alpha=0.7)
            ax.add_collection3d(lc_asc)
    if desc:
        seg_desc = _segments_from_edges(coords, desc)
        if len(seg_desc):
            lc_desc = Line3DCollection(seg_desc, colors="tab:green", linewidths=1.0, alpha=0.6)
            ax.add_collection3d(lc_desc)
    if crit:
        seg_crit = _segments_from_edges(coords, crit)
        if len(seg_crit):
            lc_crit = Line3DCollection(seg_crit, colors="tab:red", linewidths=2.0, alpha=0.8)
            ax.add_collection3d(lc_crit)
    if len(valid) < 2000:
        pts = np.array([coords[i] for i in valid.tolist()])
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=node_size, c="tab:green", alpha=0.6)
    ax.set_title("DM graph over 3D t-SNE of clip features")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    h_morse = mlines.Line2D([], [], color="lightgray", alpha=0.35, label="Morse-only")
    h_asc   = mlines.Line2D([], [], color="tab:blue",  label="Unstable (+1)")
    h_desc  = mlines.Line2D([], [], color="tab:green", label="Stable (-1)")
    h_crit  = mlines.Line2D([], [], color="tab:red",   label="Saddle (2)")
    handles = []
    if morse_edgelist: handles.append(h_morse)
    if asc: handles.append(h_asc)
    if desc: handles.append(h_desc)
    if crit: handles.append(h_crit)
    if handles:
        ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.show()
    return G, coords, coords3d


def plot_dimo_longest_cycle(
    edge_file: str,
    feature_dir: str,
    figsize: tuple[int, int] = (10, 10),
    node_size: int = 8,
    tsne_kwargs: dict | None = None,
    top_k: int = 5,
) -> tuple[list[list[int]], nx.Graph, dict[int, np.ndarray], np.ndarray]:
    """
    Plot discrete Morse edges with highlighted longest cycles.

    Args:
        edge_file: Path to the edge file from DM computation
        feature_dir: Directory containing feature/point files
        figsize: Figure size
        node_size: Size of scatter points
        tsne_kwargs: Additional kwargs for t-SNE embedding
        top_k: Number of longest cycles to highlight

    Returns:
        Tuple of (cycles_sorted, graph, coords_dict, coords3d_array)
    """
    
    edges = _load_dimo_edges(edge_file)
    feat_path = _find_feature_file(feature_dir)
    X = _load_features(feat_path)
    tsne_kwargs = tsne_kwargs or {}
    coords3d = _embed_tsne_3d(X, **tsne_kwargs)
    nodes = np.unique(edges[:, :2].ravel())
    valid = nodes[nodes < len(coords3d)]
    keep = np.isin(edges[:,0], valid) & np.isin(edges[:,1], valid)
    edges = edges[keep]
    desc = [(a,b) for a,b,t in edges if t == -1]
    crit = [(a,b) for a,b,t in edges if t == 2]
    morse_only_path = os.path.join(os.path.dirname(feat_path), "edge_for_morse_only.txt")
    morse_edges = _load_edges_first2(morse_only_path) if os.path.exists(morse_only_path) else np.empty((0,2), dtype=int)
    if morse_edges.size > 0:
        morse_valid_mask = (morse_edges[:,0] < len(coords3d)) & (morse_edges[:,1] < len(coords3d))
        morse_edges = morse_edges[morse_valid_mask]
        morse_edgelist = [tuple(e) for e in morse_edges if (e[0] in valid) and (e[1] in valid)]
    else:
        morse_edgelist = []
    coords = {int(i): coords3d[int(i)] for i in valid.tolist()}
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if morse_edgelist:
        seg_morse = _segments_from_edges(coords, morse_edgelist)
        if len(seg_morse):
            lc_morse = Line3DCollection(seg_morse, colors="lightgray", linewidths=0.4, alpha=0.15, zorder=0)
            ax.add_collection3d(lc_morse)
    if desc:
        seg_desc = _segments_from_edges(coords, desc)
        if len(seg_desc):
            lc_desc = Line3DCollection(seg_desc, colors="tab:green", linewidths=0.6, alpha=0.15, zorder=1)
            ax.add_collection3d(lc_desc)
    if crit:
        seg_crit = _segments_from_edges(coords, crit)
        if len(seg_crit):
            lc_crit = Line3DCollection(seg_crit, colors="tab:red", linewidths=1.2, alpha=0.35, zorder=1)
            ax.add_collection3d(lc_crit)
    if len(valid) < 2000:
        pts = np.array([coords[i] for i in coords])
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=node_size, c="tab:green", alpha=0.5, depthshade=False, zorder=0)
    UG = _build_undirected_skeleton_from_dimo(edges, n_nodes=int(nodes.max())+1 if len(nodes) else None)
    UG = UG.subgraph(list(coords.keys())).copy()
    UG = _two_core(UG)
    cycles = _find_cycles_mcb(UG, min_len=3, max_cycles=None)
    cycles_sorted = sorted(cycles, key=len, reverse=True)[:max(0, int(top_k))]
    cmap = plt.cm.get_cmap("tab10")
    cycle_handles = []
    for i, cyc in enumerate(cycles_sorted):
        seg = _segments3d_from_cycle(coords, cyc)
        if len(seg):
            col = cmap(i % 10)
            lc_glow = Line3DCollection(seg, colors=[(1,1,1,0.8)], linewidths=5.0, zorder=999)
            ax.add_collection3d(lc_glow)
            lc = Line3DCollection(seg, colors=[col], linewidths=2.6, alpha=1.0, zorder=1000)
            try:
                lc.set_zsort("max")
            except Exception:
                pass
            ax.add_collection3d(lc)
            cycle_handles.append(mlines.Line2D([], [], color=col, linewidth=3.0, label=f"Cycle {i+1} (|C|={len(cyc)})"))
    ax.set_title("DM graph over 3D t-SNE with top-k longest cycles")
    # ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_axis_off()
    h_morse = mlines.Line2D([], [], color="lightgray", alpha=0.5, label="kNN")
    # h_desc  = mlines.Line2D([], [], color="tab:green", label="Stable (-1)")
    # h_crit  = mlines.Line2D([], [], color="tab:red",   label="Saddle (2)")
    handles = []
    if morse_edgelist: handles.append(h_morse)
    # if desc: handles.append(h_desc)
    # if crit: handles.append(h_crit)
    handles.extend(cycle_handles)
    if handles:
        ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(
        "dimo_longest_cycle.pdf",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
        transparent=True,
        format="pdf"
    )
    plt.show()
    return cycles_sorted, UG, coords, coords3d


def plot_desc_flow(
    edge_path: str,
    coords3d: np.ndarray | dict[int, np.ndarray],
    show_cycle_edges: bool = True,
    cycle_alpha: float = 0.15,
    top_k: int | None = None,
) -> None:
    """
    Plot the descending flow paths from discrete Morse graph.

    Args:
        edge_path: Path to the edge file
        coords3d: 3D coordinates as array or dict
        show_cycle_edges: Whether to show cycle edges in background
        cycle_alpha: Alpha for cycle edges
        top_k: Number of top minima to show (None for all)
    """
    
    E = load_dimo_edges(edge_path)
    if E.size == 0: return
    if isinstance(coords3d, np.ndarray):
        coord_has = lambda i: 0 <= int(i) < coords3d.shape[0]
        getp = lambda i: coords3d[int(i)].astype(float)
    else:
        coord_has = lambda i: int(i) in coords3d
        getp = lambda i: np.asarray(coords3d[int(i)], dtype=float)

    desc = E[E[:,2] == -1]
    cyc  = E[E[:,2] ==  2]

    G = nx.DiGraph()
    for u,v,_ in desc:
        if coord_has(u) and coord_has(v):
            G.add_edge(int(u), int(v))

    nodes = list(G.nodes())
    minima_all  = [n for n in nodes if G.out_degree(n) == 0]
    sources_all = [n for n in nodes if G.in_degree(n)  == 0]
    if not minima_all: return

    Grev = G.reverse(copy=False)
    maxlen = {}
    for m in minima_all:
        lengths = nx.single_source_shortest_path_length(Grev, m)
        maxlen[m] = max(lengths.values()) if lengths else 0
    if top_k is not None:
        mins = [m for m,_ in sorted(maxlen.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
    else:
        mins = minima_all

    basin = {}
    for m in mins:
        for n in nx.descendants(Grev, m) | {m}:
            basin[n] = m
    sources = [s for s in sources_all if s in basin]

    cmap = plt.cm.get_cmap("tab10", max(1,len(mins)))
    col = {m: cmap(i % 10) for i,m in enumerate(mins)}

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection="3d")

    if show_cycle_edges and len(cyc):
        for u,v,_ in cyc:
            if coord_has(u) and coord_has(v):
                p1, p2 = getp(u), getp(v)
                ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]], c="gray", lw=0.6, alpha=cycle_alpha, zorder=0)

    for u,v in G.edges():
        if u not in basin: continue
        p1, p2 = getp(u), getp(v)
        c = col[basin[u]]
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]], c=c, lw=1.6, alpha=0.9)
        d = p2 - p1
        L = np.linalg.norm(d)
        if L < 1e-8: continue
        udir = d / L
        base = p1 + 0.72*d
        alen = 0.25 * L
        ax.quiver(base[0], base[1], base[2], udir[0], udir[1], udir[2], length=alen, normalize=False, color=c, alpha=0.95)

    for i,m in enumerate(mins):
        if not coord_has(m): continue
        pm = getp(m)
        ax.scatter(pm[0], pm[1], pm[2], s=80, color=col[m], edgecolors='k', linewidths=0.7, label=f"min {i} (L={maxlen[m]})")

    if sources:
        sx = []; sy = []; sz = []
        for s in sources:
            if not coord_has(s): continue
            ps = getp(s); sx.append(ps[0]); sy.append(ps[1]); sz.append(ps[2])
        if sx:
            ax.scatter(sx, sy, sz, s=70, c="crimson", marker="^", edgecolors='k', linewidths=0.7, label="sources")

    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Descending flow toward selected minima (colors by basin) with sources")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    plt.tight_layout(); plt.show()


def plot_cycle_projections(
    windows: np.ndarray,
    cycle_idx: np.ndarray | list[int],
    clip_index: int,
    m_marker: int = 10,
    loop_marker: int = 20,
    random_state: int = 42,
    tsne_perplexity: int = 10,
    umap_neighbors: int = 15,
    figsize: tuple[int, int] = (12, 4),
) -> tuple[Figure, list[Axes]]:
    """
    Plot a representative cycle in reduced dimensions (PCA, t-SNE, UMAP).

    Args:
        windows: Point cloud array of shape (N, D)
        cycle_idx: Indices of points in the cycle
        clip_index: Clip ID for labeling
        m_marker: Marker size for non-cycle points
        loop_marker: Marker size for cycle points
        random_state: Random seed for reproducibility
        tsne_perplexity: Perplexity for t-SNE
        umap_neighbors: Number of neighbors for UMAP
        figsize: Figure size

    Returns:
        Tuple of (figure, list_of_axes)
    """
    windows   = np.asarray(windows)
    cycle_idx = np.asarray(cycle_idx, dtype=int)

    # mask for quick boolean indexing
    is_loop = np.zeros(len(windows), dtype=bool)
    is_loop[cycle_idx] = True

    def _scatter(ax, emb, title):
        ax.scatter(*emb[~is_loop].T, s=m_marker, alpha=.35, label="window")
        ax.scatter(*emb[is_loop].T,  s=loop_marker, color="tab:red", label=f"cycle for clip {clip_index}")

        # draw closed polygon along the loop order
        loop_closed = np.r_[emb[is_loop], emb[is_loop][0:1]]
        ax.plot(loop_closed[:, 0], loop_closed[:, 1],
                lw=2, color="tab:red")

        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig, axes = plt.subplots(1, 3, figsize=figsize,
                             constrained_layout=True)

    pca = PCA(n_components=2, random_state=random_state)
    _scatter(axes[0], pca.fit_transform(windows), "PCA")

    perplexity = min(tsne_perplexity, len(windows) - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca", # faster & more stable
        n_iter=1000,
        random_state=random_state
    )
    _scatter(axes[1], tsne.fit_transform(windows), f"t‑SNE (perp={perplexity})")

    n_neighbors = min(umap_neighbors, len(windows) - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        random_state=random_state
    )
    _scatter(axes[2], reducer.fit_transform(windows),
             f"UMAP (k={n_neighbors})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", fontsize="small")
        
    return fig, axes



def plot_vpath_cluster(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    embeds3d: list[tuple[np.ndarray, str]],
    clip_frame_to_global: dict[int, dict[int, int]],
    alpha_bg: float = 0.02,
) -> None:
    """
    Plot clustered windows in multiple 3D embedding projections.

    Args:
        cluster_dict: Dict mapping cluster_id to list of (clip_id, frame_indices)
        embeds3d: List of (embedding_array, title) tuples
        clip_frame_to_global: Mapping from (clip, frame) to global index
        alpha_bg: Alpha for background points
    """
    assert len(embeds3d) == 3, "This function expects exactly 3 embeddings for now."

    palette = ["tab:red", "tab:blue", "tab:green", "tab:orange",
               "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
    colors = {cid: palette[i % len(palette)] for i, cid in enumerate(sorted(cluster_dict))}
    frame_labels = _frame_label_vote(cluster_dict)

    fig = plt.figure(figsize=(36, 10), dpi=150)
    ncols = len(embeds3d)

    for i, (emb, title) in enumerate(embeds3d, start=1):
        ax = fig.add_subplot(1, ncols, i, projection="3d")

        ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2],
                   s=2, alpha=alpha_bg, color="lightgrey")

        for cid in sorted(cluster_dict):
            idxs = []
            for (clip, f), lbl in frame_labels.items():
                if lbl != cid:
                    continue
                global_map = clip_frame_to_global.get(clip, {})
                g = global_map.get(int(f))
                if g is not None:
                    idxs.append(g)

            if not idxs:
                continue

            idxs = np.asarray(idxs, dtype=int)
            ax.scatter(emb[idxs, 0], emb[idxs, 1], emb[idxs, 2],
                       s=10, alpha=0.5, color=colors[cid],
                       label=f"cluster {cid}")

        ax.set_title(title, fontsize=14)
        ax.set_axis_off()

        if i == ncols:
            ax.legend(loc="upper right")

    plt.tight_layout()
    output_path = "vpath_cluster_all.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved combined plot to {output_path}")


def plot_eigenbase_compression_loss(
    clip_ids: int | list[int],
    all_cycle_dm: list[list[np.ndarray]],
    rodent_rollout_dir: str,
    k_cycles: int = 1,
    k_eig: int = 10,
    pad: int = 0,
    Kshow: int = 30,
    state_key: str = "qpos",
    plot: bool = True,
) -> None:
    """
    Plot eigenbase compression loss between full and cycle subspaces.

    Args:
        clip_ids: Single clip ID or list of clip IDs
        all_cycle_dm: List of cycle arrays per clip
        rodent_rollout_dir: Directory containing clip HDF5 files
        k_cycles: Number of top cycles per clip to use
        k_eig: Number of eigenvectors for subspace angle computation
        pad: Frame padding around cycles
        Kshow: Number of PCs to show in plots
        state_key: Key for state data in HDF5
        plot: Whether to display the plot
    """
    if isinstance(clip_ids, int):
        clip_ids = [clip_ids]
    full_list, cyc_list = [], []
    D = None
    for clip_id in clip_ids:
        roll = load_rollout_from_source(clip_id, rodent_rollout_dir)
        try:
            Q = np.asarray(roll[state_key])
        except KeyError:
            Q = np.asarray(roll["states"])
        Q = Q[:, 7:]
        if D is None:
            D = Q.shape[1]
        T = Q.shape[0]
        cycles = all_cycle_dm[clip_id]
        cycles_sorted = sorted(cycles, key=lambda arr: len(arr), reverse=True)[:k_cycles]
        idx = set()
        for arr in cycles_sorted:
            a = np.maximum(0, np.asarray(arr, dtype=int) - pad)
            b = np.minimum(T - 1, np.asarray(arr, dtype=int) + pad)
            for s, e in zip(a, b):
                idx.update(range(int(s), int(e) + 1))
        idx = np.array(sorted(idx), dtype=int)
        if idx.size == 0:
            continue
        full_list.append(Q)
        cyc_list.append(Q[idx])
    if not full_list or not cyc_list:
        return {"error": "no data after pooling"}
    Q_full = np.vstack(full_list)
    Q_cyc  = np.vstack(cyc_list)

    Qf = Q_full - Q_full.mean(0, keepdims=True)
    Qc = Q_cyc  - Q_cyc.mean(0, keepdims=True)

    Cf = np.cov(Qf.T); Cc = np.cov(Qc.T)
    ef, Uf = np.linalg.eigh(Cf); ec, Uc = np.linalg.eigh(Cc)
    ord_f = ef.argsort()[::-1]; ord_c = ec.argsort()[::-1]
    ef = ef[ord_f]; ec = ec[ord_c]
    Uf = Uf[:, ord_f]; Uc = Uc[:, ord_c]

    pf = ef / (ef.sum() + 1e-12)
    pc = ec / (ec.sum() + 1e-12)

    H_full = -(np.clip(pf,1e-12,1)*np.log2(np.clip(pf,1e-12,1))).sum()
    H_cyc  = -(np.clip(pc,1e-12,1)*np.log2(np.clip(pc,1e-12,1))).sum()
    Hn_full = H_full / np.log2(D)
    Hn_cyc  = H_cyc  / np.log2(D)

    l2_spec = np.linalg.norm(pf - pc)

    s = min(k_eig, D)
    ang = subspace_angles(Uf[:, :s], Uc[:, :s])

    if plot:
        kshow = min(Kshow, D)
        x = np.arange(1, kshow+1)
        fig = plt.figure(figsize=(12, 9), dpi=140)
        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)

        ax1 = fig.add_subplot(gs[0,0])
        ax1.plot(x, pf[:kshow], marker='o', linewidth=1)
        ax1.plot(x, pc[:kshow], marker='o', linewidth=1)
        ax1.set_title("Scree (normalized eigen-spectrum)")
        ax1.set_xlabel("PC index"); ax1.set_ylabel("Variance fraction")
        ax1.legend(["Full (pooled)","Cycles (pooled)"])

        ax2 = fig.add_subplot(gs[0,1])
        ax2.plot(x, np.cumsum(pf[:kshow]), marker='o', linewidth=1)
        ax2.plot(x, np.cumsum(pc[:kshow]), marker='o', linewidth=1)
        ax2.set_title("Cumulative variance")
        ax2.set_xlabel("PC index"); ax2.set_ylabel("Cumulative fraction")
        ax2.legend(["Full (pooled)","Cycles (pooled)"])

        ax3 = fig.add_subplot(gs[1,0])
        ax3.bar(x, (pf[:kshow]-pc[:kshow]))
        ax3.axhline(0, linewidth=1)
        ax3.set_title("Spectrum difference (full - cycles)")
        ax3.set_xlabel("PC index"); ax3.set_ylabel("Δ variance fraction")

        ax4 = fig.add_subplot(gs[1,1])
        ax4.bar(np.arange(1, s+1), ang * 180/np.pi)
        ax4.set_title(f"Principal angles (top-{s})")
        ax4.set_xlabel("Angle index"); ax4.set_ylabel("Degrees")
        
        if len(clip_ids) != 1:
            fig.suptitle(
                f"{len(clip_ids)} clips | H(bits): full={H_full:.3f}, cycles={H_cyc:.3f} | "
                f"H_norm: full={Hn_full:.3f}, cycles={Hn_cyc:.3f} | "
                f"N_full={Q_full.shape[0]}, N_cycles={Q_cyc.shape[0]} | k_cycles/clip={k_cycles}"
                f"\nL2(spec)={l2_spec:.4f}"
            )
        else:
            fig.suptitle(
                f"Clips {clip_ids} | H(bits): full={H_full:.3f}, cycles={H_cyc:.3f} | "
                f"H_norm: full={Hn_full:.3f}, cycles={Hn_cyc:.3f} | "
                f"N_full={Q_full.shape[0]}, N_cycles={Q_cyc.shape[0]} | k_cycles/clip={k_cycles}"
                f"\nL2(spec)={l2_spec:.4f}"
            )
        plt.tight_layout()
        plt.show()
        

def plot_eigenbase_cross_reconstruction(
    clip_ids: list[int],
    all_cycle_dm: list[list[np.ndarray]],
    rodent_rollout_dir: str,
    k_cycles: int = 1,
    pad: int = 0,
    Kmax: int | None = None,
    state_key: str = "qpos",
    plot: bool = True,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """
    Compute and plot cross-reconstruction errors between clip eigenspaces.

    Args:
        clip_ids: List of clip IDs to analyze
        all_cycle_dm: List of cycle arrays per clip
        rodent_rollout_dir: Directory containing clip HDF5 files
        k_cycles: Number of top cycles per clip
        pad: Frame padding around cycles
        Kmax: Maximum number of eigenvectors (None for adaptive)
        state_key: Key for state data in HDF5
        plot: Whether to display heatmap

    Returns:
        Tuple of (error_matrix, clip_ids, eigenbases)
    """
    if isinstance(clip_ids, int):
        clip_ids = [clip_ids]

    full_list, cyc_list = [], []
    D = None
    for clip_id in clip_ids:
        roll = load_rollout_from_source(clip_id, rodent_rollout_dir)
        try:
            Q = np.asarray(roll[state_key])
        except KeyError:
            Q = np.asarray(roll["states"])
        Q = Q[:, 7:]
        if D is None:
            D = Q.shape[1]
        T = Q.shape[0]

        cycles = all_cycle_dm[clip_id]
        cycles_sorted = sorted(cycles, key=lambda arr: len(arr), reverse=True)[:k_cycles]
        idx = set()
        for arr in cycles_sorted:
            a = np.maximum(0, np.asarray(arr, dtype=int) - pad)
            b = np.minimum(T - 1, np.asarray(arr, dtype=int) + pad)
            for s, e in zip(a, b):
                idx.update(range(int(s), int(e) + 1))
        idx = np.array(sorted(idx), dtype=int)
        if idx.size == 0:
            continue

        full_list.append(Q)
        cyc_list.append(Q[idx])

    if not full_list or not cyc_list:
        return {"error": "no data after pooling"}

    Q_full = np.vstack(full_list)
    Q_cyc  = np.vstack(cyc_list)

    Qf = Q_full - Q_full.mean(0, keepdims=True)
    Qc = Q_cyc  - Q_cyc.mean(0, keepdims=True)

    Cf = np.cov(Qf.T)
    Cc = np.cov(Qc.T)

    ef, Uf = np.linalg.eigh(Cf); ord_f = ef.argsort()[::-1]; ef = ef[ord_f]; Uf = Uf[:, ord_f]
    ec, Uc = np.linalg.eigh(Cc); ord_c = ec.argsort()[::-1]; ec = ec[ord_c]; Uc = Uc[:, ord_c]

    tr_f = np.trace(Cf); tr_c = np.trace(Cc)
    D = Qf.shape[1]
    if Kmax is None: Kmax = min(40, D)

    # self-curves (upper bounds)
    ev_full_by_full  = np.cumsum(ef[:Kmax]) / (tr_f + 1e-12)
    ev_cycle_by_cycle = np.cumsum(ec[:Kmax]) / (tr_c + 1e-12)

    # cross curves: variance of target captured by source PCs
    ev_full_by_cycle  = np.zeros(Kmax)
    ev_cycle_by_full  = np.zeros(Kmax)
    # incremental computation using quadratic forms
    Qt_f = Cf
    Qt_c = Cc
    A_fc = Uc.T @ Qt_f @ Uc    # variance of FULL in CYCLE basis
    A_cf = Uf.T @ Qt_c @ Uf    # variance of CYCLE in FULL basis
    # cumulative diagonals
    ev_full_by_cycle  = np.cumsum(np.diag(A_fc)[:Kmax]) / (tr_f + 1e-12)
    ev_cycle_by_full  = np.cumsum(np.diag(A_cf)[:Kmax]) / (tr_c + 1e-12)

    recon_err_full_by_cycle = 1.0 - ev_full_by_cycle
    recon_err_cycle_by_full = 1.0 - ev_cycle_by_full
    recon_err_full_by_full  = 1.0 - ev_full_by_full
    recon_err_cycle_by_cycle = 1.0 - ev_cycle_by_cycle

    if plot:
        x = np.arange(1, Kmax+1)
        fig = plt.figure(figsize=(12, 8), dpi=140)
        gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25)

        ax1 = fig.add_subplot(gs[0,0])
        ax1.plot(x, ev_full_by_cycle, marker='o', linewidth=1)
        ax1.plot(x, ev_cycle_by_full, marker='o', linewidth=1)
        # ax1.plot(x, ev_full_by_full, linestyle='--', linewidth=1)
        # ax1.plot(x, ev_cycle_by_cycle, linestyle='--', linewidth=1)
        ax1.set_title("Explained variance by PCs")
        ax1.set_xlabel("# PCs (k)"); ax1.set_ylabel("Fraction of variance explained")
        ax1.legend(["Full ← Cycle PCs", "Cycle ← Full PCs"])

        ax2 = fig.add_subplot(gs[0,1])
        ax2.plot(x, recon_err_full_by_cycle, marker='o', linewidth=1)
        ax2.plot(x, recon_err_cycle_by_full, marker='o', linewidth=1)
        # ax2.plot(x, recon_err_full_by_full, linestyle='--', linewidth=1)
        # ax2.plot(x, recon_err_cycle_by_cycle, linestyle='--', linewidth=1)
        ax2.set_title("Normalized reconstruction error (1 - EV)")
        ax2.set_xlabel("# PCs (k)"); ax2.set_ylabel("Error")

        # show marginal gains
        ax3 = fig.add_subplot(gs[1,0])
        ax3.bar(x, np.r_[ev_full_by_cycle[0], np.diff(ev_full_by_cycle)])
        ax3.set_title("Marginal EV gain per PC: Full ← Cycle")
        ax3.set_xlabel("PC index"); ax3.set_ylabel("Δ EV")

        ax4 = fig.add_subplot(gs[1,1])
        ax4.bar(x, np.r_[ev_cycle_by_full[0], np.diff(ev_cycle_by_full)])
        ax4.set_title("Marginal EV gain per PC: Cycle ← Full")
        ax4.set_xlabel("PC index"); ax4.set_ylabel("Δ EV")

        fig.suptitle(
            f"Cross reconstruction over {len(clip_ids)} clips | "
            f"N_full={Q_full.shape[0]}, N_cycles={Q_cyc.shape[0]} | "
            f"k_cycles/clip={k_cycles}, pad={pad}"
        )
        plt.tight_layout()
        plt.show()


def plot_cluster_eigenbase_cross_reconstruction(
    cluster_cycles_by_id: dict[int, list[tuple[int, np.ndarray]]],
    cluster_labels: dict[int, str],
    target_clusters: list[int | str],
    csv_path: str,
    rodent_rollout_dir: str,
    k_cycles_per_clip: int = 1,
    pad: int = 0,
    Kmax: int = 30,
    state_key: str = "qpos",
    plot_error: bool = True,
    n_cycles_sample: int | None = None,
    sample_strategy: str = "longest",
    seed: int = 0,
) -> None:
    """
    Compute and plot cross-reconstruction errors between cluster eigenspaces.

    Args:
        cluster_cycles_by_id: Dict mapping cluster_id to list of (clip_id, frame_indices)
        cluster_labels: Dict mapping cluster_id to label string
        target_clusters: List of cluster IDs or names to analyze
        csv_path: Path to behavior labels CSV
        rodent_rollout_dir: Directory containing clip HDF5 files
        k_cycles_per_clip: Number of top cycles per clip
        pad: Frame padding around cycles
        Kmax: Maximum number of eigenvectors
        state_key: Key for state data in HDF5
        plot_error: Whether to display error plots
        n_cycles_sample: Number of cycles to sample per cluster
        sample_strategy: Sampling strategy ("longest" or "random")
        seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(csv_path)

    def load_qpos(i):
        roll = load_rollout_from_source(int(i), rodent_rollout_dir)
        return np.asarray(roll[state_key])[:,7:]

    def has_label(r,k):
        if k=="walk":  return r["walk"]==1
        if k=="groom": return (r["groom_up"]==1) or (r["groom_down"]==1)
        if k=="turn":  return r["head_turn_xy"]==1
        if k=="stay":  return r["walk_wait"]==1
        if k=="rear":  return r["rear"]==1
        return False

    def parse_reqs(n):
        n=n.lower(); o=[]
        if "walk" in n:  o.append("walk")
        if "groom" in n: o.append("groom")
        if "turn" in n:  o.append("turn")
        if "stay" in n:  o.append("stay")
        if "rear" in n:  o.append("rear")
        return o

    def to_cid(x):
        if isinstance(x,int): return x
        m=[k for k,v in cluster_labels.items() if v==str(x)]
        if not m: raise ValueError(f"cluster '{x}' not found")
        return m[0]

    cids=[to_cid(c) for c in target_clusters]
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)

    ev_ref = {}
    ev_cross = {}
    meta = {}

    for j,cid in enumerate(cids):
        cname = cluster_labels[cid]
        reqs = parse_reqs(cname)
        clips = [int(r.clip_idx) for _,r in df.iterrows() if all(has_label(r,k) for k in reqs)]
        if not clips: 
            continue

        qcache={}
        Q_full = np.vstack([load_qpos(c) for c in clips])
        D = Q_full.shape[1]; Kk = min(Kmax, D)
        Qf = Q_full - Q_full.mean(0)
        Cf = np.cov(Qf.T)
        ef,_ = np.linalg.eigh(Cf); ef = np.sort(ef)[::-1]
        ev_ref[cid] = np.cumsum(ef[:Kk])/(np.sum(ef)+1e-12)

        entries = cluster_cycles_by_id.get(cid, [])
        if not entries: 
            continue

        # Build a pool of all cycles across the cluster
        # cycle_pool: list of tuples (clip_id, frames_array, length)
        cycle_pool = [(int(cc), np.asarray(arr,int), len(arr)) for cc, arr in entries]

        # Choose which cycles to use
        if (n_cycles_sample is not None) and (len(cycle_pool) > n_cycles_sample):
            if sample_strategy == "random":
                sel_idx = rng.choice(len(cycle_pool), size=n_cycles_sample, replace=False)
                cycle_pool = [cycle_pool[i] for i in sel_idx]
            else:  # "longest"
                cycle_pool = sorted(cycle_pool, key=lambda t: t[2], reverse=True)[:n_cycles_sample]
        else:
            # fall back to per-clip top-k if requested
            if n_cycles_sample is None and k_cycles_per_clip is not None:
                by_clip={}
                for cc, arr, _ in cycle_pool:
                    by_clip.setdefault(cc, []).append(arr)
                cycle_pool = []
                for cc, fl in by_clip.items():
                    keep = sorted(fl, key=len, reverse=True)[:k_cycles_per_clip]
                    for a in keep:
                        cycle_pool.append((cc, a, len(a)))

        # Build Q_cyc with padding from the selected cycles
        cyc=[]
        used_cycle_count = 0
        for cc, a, _ in cycle_pool:
            if cc not in qcache: qcache[cc]=load_qpos(cc)
            T=qcache[cc].shape[0]
            lb=np.clip(a-pad,0,T-1); ub=np.clip(a+pad,0,T-1)
            idx=[]
            for s,e in zip(lb,ub): idx+=range(s,e+1)
            if idx:
                cyc.append(qcache[cc][np.unique(idx)])
                used_cycle_count += 1
        if not cyc: 
            continue

        Q_cyc = np.vstack(cyc)
        Qc = Q_cyc - Q_cyc.mean(0)
        Cc = np.cov(Qc.T)
        ec,Uc = np.linalg.eigh(Cc)
        Uc = Uc[:, ec.argsort()[::-1]]
        A = Uc.T @ Cf @ Uc
        ev_cross[cid] = np.cumsum(np.diag(A)[:Kk])/(np.trace(Cf)+1e-12)

        meta[cid] = dict(
            name=cname, 
            n_clips=len(clips), 
            N_full=Q_full.shape[0], 
            N_cyc=Q_cyc.shape[0],
            Kmax=Kk,
            cycles_used=used_cycle_count
        )

    if not ev_ref:
        return dict(ev_ref=ev_ref, ev_cross=ev_cross, meta=meta)

    x = np.arange(1, list(meta.values())[0]["Kmax"]+1)

    plt.figure(figsize=(12,8), dpi=140)
    handles=[]; labels=[]
    for i,cid in enumerate(cids):
        if cid not in meta: continue
        col=None if colors is None else colors[i%len(colors)]
        m = meta[cid]
        lbl=(f"{cid}: {m['name']} ({m['n_clips']} clips, "
             f"{m['N_full']} full fr, {m['N_cyc']} cyc fr, "
             f"{m['cycles_used']} cycles)")
        plt.plot(x, ev_ref[cid],  color=col, linestyle='-',  linewidth=2.5, marker='o')
        plt.plot(x, ev_cross[cid],color=col, linestyle='--', linewidth=2.5, marker='o')
        h=Line2D([0],[0],color=col,linestyle='-',linewidth=2.5,marker='o')
        handles.append(h); labels.append(lbl)
    style = [
        Line2D([0],[0], color='k', ls='-',  lw=2.5, label='FULL ← FULL PCs'),
        Line2D([0],[0], color='k', ls='--', lw=2.5, label='FULL ← CYCLE PCs'),
    ]
    handles += style; labels += [h.get_label() for h in style]
    plt.legend(handles, labels, bbox_to_anchor=(1.02,1), loc="upper left", handlelength=3.2)
    plt.title("Explained variance\nsolid = FULL←FULL, dashed = FULL←CYCLE")
    plt.xlabel("# PCs (k)"); plt.ylabel("EV")
    plt.tight_layout(); plt.show()

    if plot_error:
        plt.figure(figsize=(12,8), dpi=140)
        handles=[]; labels=[]
        for i,cid in enumerate(cids):
            if cid not in meta: continue
            col=None if colors is None else colors[i%len(colors)]
            m = meta[cid]
            lbl=(f"{cid}: {m['name']} ({m['n_clips']} clips, "
                 f"{m['N_full']} full fr, {m['N_cyc']} cyc fr, "
                 f"{m['cycles_used']} cycles)")
            plt.plot(x, 1-ev_ref[cid],   color=col, linestyle='-',  linewidth=2.5, marker='o')
            plt.plot(x, 1-ev_cross[cid], color=col, linestyle='--', linewidth=2.5, marker='o')
            h=Line2D([0],[0],color=col,linestyle='-',linewidth=2.5,marker='o')
            handles.append(h); labels.append(lbl)
        style = [
            Line2D([0],[0], color='k', ls='-',  lw=2.5, label='FULL ← FULL PCs'),
            Line2D([0],[0], color='k', ls='--', lw=2.5, label='FULL ← CYCLE PCs'),
        ]
        handles += style; labels += [h.get_label() for h in style]
        plt.legend(handles, labels, bbox_to_anchor=(1.02,1), loc="upper left", handlelength=3.2)
        plt.title("Reconstruction error (1−EV)\nsolid = FULL←FULL, dashed = FULL←CYCLE")
        plt.xlabel("# PCs (k)"); plt.ylabel("Error")
        plt.tight_layout(); plt.show()

    return dict(ev_ref=ev_ref, ev_cross=ev_cross, meta=meta)


def plot_cluster_principal_angles_to_target_union(
    cluster_cycles_by_id: dict[int, list[tuple[int, np.ndarray]]],
    cluster_labels: dict[int, str],
    target_clusters: list[int | str],
    csv_path: str,
    rodent_rollout_dir: str,
    target_union: str = "groom",
    k_cycles: int = 5,
    s_top: int = 10,
    pad: int = 0,
    state_key: str = "qpos",
    sample_strategy: str = "random",
    repeats: int = 1,
    seed: int = 0,
    Kmax_plot: int | None = None,
    plot: bool = True,
) -> None:
    """
    Compute and plot principal angles between cluster subspaces and target union.

    Args:
        cluster_cycles_by_id: Dict mapping cluster_id to list of (clip_id, frame_indices)
        cluster_labels: Dict mapping cluster_id to label string
        target_clusters: List of cluster IDs or names
        csv_path: Path to behavior labels CSV
        rodent_rollout_dir: Directory containing clip HDF5 files
        target_union: Target union behavior type
        k_cycles: Number of cycles per cluster
        s_top: Number of top eigenvectors
        pad: Frame padding around cycles
        state_key: Key for state data in HDF5
        sample_strategy: Sampling strategy ("random" or "longest")
        repeats: Number of repetitions
        seed: Random seed
        Kmax_plot: Maximum K for plotting
        plot: Whether to display plots
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(csv_path)

    def load_qpos(i):
        roll = load_rollout_from_source(int(i), rodent_rollout_dir)
        try:
            Q = np.asarray(roll[state_key])
        except KeyError:
            Q = np.asarray(roll["states"])
        return Q[:, 7:]

    def to_cid(x):
        if isinstance(x, int): return x
        m = [k for k,v in cluster_labels.items() if v == str(x)]
        if not m: raise ValueError(f"cluster '{x}' not found")
        return m[0]

    def motif_sets(df):
        walk_only=set(); rear_only=set(); groom_only=set(); turn_any=set(); wait_any=set()
        for _,r in df.iterrows():
            c=int(r["clip_idx"])
            w=int(r.get("walk",0))==1
            ww=int(r.get("walk_wait",0))==1
            rz=int(r.get("head_turn_z",0))==1
            re=int(r.get("rear",0))==1
            gu=int(r.get("groom_up",0))==1
            gd=int(r.get("groom_down",0))==1
            txy=int(r.get("head_turn_xy",0))==1
            walk_only_cond=(w or ww) and not (rz or re or gu or gd or txy)
            rear_only_cond=(rz or re) and not (w or ww or gu or gd or txy)
            groom_only_cond=(gu or gd) and not (w or ww or rz or re or txy)
            if walk_only_cond: walk_only.add(c)
            if rear_only_cond: rear_only.add(c)
            if groom_only_cond: groom_only.add(c)
            if txy: turn_any.add(c)
            if ww: wait_any.add(c)
        return {"walk":walk_only,"rear":rear_only,"groom":groom_only,"turn":turn_any,"wait":wait_any}

    ms = motif_sets(df)
    if isinstance(target_union, (set, list, tuple, np.ndarray)):
        target_clips = sorted(set(int(x) for x in target_union))
        target_name = "custom"
    else:
        key = str(target_union).lower()
        if key not in ms: raise ValueError(f"target_union must be one of {list(ms.keys())} or a set/list of clip ids")
        target_clips = sorted(ms[key])
        target_name = key
    if not target_clips: raise ValueError("No clips found for the chosen target union")

    full_list = []
    D = None
    for clip_id in target_clips:
        Q = load_qpos(clip_id)
        if D is None: D = Q.shape[1]
        full_list.append(Q)
    if not full_list: raise ValueError("Target union produced no data")
    Q_full = np.vstack(full_list)

    cids  = [to_cid(c) for c in target_clusters]
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    out = {"angles_mean":{}, "angles_std":{}, "labels":{}, "s_used": None}
    plotted_any = False
    s_used_global = None

    for cid in cids:
        cname = cluster_labels[cid]
        entries = cluster_cycles_by_id.get(cid, [])
        if not entries: continue

        qcache = {}
        cycle_pool = []
        for cc, arr in entries:
            cc = int(cc)
            if cc not in qcache:
                qcache[cc] = load_qpos(cc)
            a = np.asarray(arr, int)
            raw_len = len(np.unique(a))
            cycle_pool.append({"clip": cc, "frames": a, "raw_len": raw_len})
        if not cycle_pool: continue

        def build_Q_cyc(sel_items):
            by_clip = {}
            for it in sel_items:
                by_clip.setdefault(it["clip"], []).append(it["frames"])
            chunks = []
            for cc, fr_lists in by_clip.items():
                Qc = qcache.get(cc, load_qpos(cc))
                T  = Qc.shape[0]
                union = set()
                for a in fr_lists:
                    lb = np.clip(a - pad, 0, T-1)
                    ub = np.clip(a + pad, 0, T-1)
                    for s,e in zip(lb,ub):
                        union.update(range(int(s), int(e)+1))
                if union:
                    idx = np.array(sorted(union), int)
                    chunks.append(Qc[idx])
            if not chunks:
                return None
            return np.vstack(chunks)

        angles_all = []
        n_eff_max = min(k_cycles, len(cycle_pool))
        if n_eff_max <= 0: continue

        for _ in range(repeats):
            perm = rng.permutation(len(cycle_pool))
            shuffled = [cycle_pool[i] for i in perm]
            if sample_strategy == "longest":
                sel = sorted(shuffled, key=lambda t: t["raw_len"], reverse=True)[:n_eff_max]
            elif sample_strategy == "shortest":
                sel = sorted(shuffled, key=lambda t: t["raw_len"])[:n_eff_max]
            else:
                idx = rng.choice(len(cycle_pool), size=n_eff_max, replace=False)
                sel = [cycle_pool[i] for i in idx]
            Q_cyc = build_Q_cyc(sel)
            if Q_cyc is None or Q_cyc.shape[0] < 2:
                continue

            Qf = Q_full - Q_full.mean(0, keepdims=True)
            Qc = Q_cyc  - Q_cyc.mean(0, keepdims=True)
            Cf = np.cov(Qf.T); Cc = np.cov(Qc.T)
            ef, Uf = np.linalg.eigh(Cf)
            ec, Uc = np.linalg.eigh(Cc)
            ord_f = ef.argsort()[::-1]; ord_c = ec.argsort()[::-1]
            Uf = Uf[:, ord_f]; Uc = Uc[:, ord_c]
            s_show = min(s_top, Uf.shape[0], Uc.shape[0])
            ang = subspace_angles(Uf[:, :s_show], Uc[:, :s_show]) * 180/np.pi
            angles_all.append(ang)

        if not angles_all: continue
        A = np.vstack(angles_all)
        ang_mean = A.mean(axis=0)
        ang_std  = A.std(axis=0)
        out["angles_mean"][cid] = ang_mean
        out["angles_std"][cid]  = ang_std
        out["labels"][cid]      = f"{cid}: {cname}"
        s_used_global = len(ang_mean) if s_used_global is None else min(s_used_global, len(ang_mean))
        plotted_any = True

    if not plotted_any:
        out["s_used"] = 0
        return out

    s_used = min(Kmax_plot, s_used_global) if Kmax_plot is not None else s_used_global
    out["s_used"] = s_used

    if plot:
        x = np.arange(1, s_used + 1)
        plt.figure(figsize=(12, 6), dpi=140)
        ax = plt.gca()

        order = sorted(
            out["angles_mean"].keys(),
            key=lambda cid: np.mean(out["angles_mean"][cid][:s_used])
        )

        for i, cid in enumerate(order):
            col = None if colors is None else colors[i % len(colors)]
            m = out["angles_mean"][cid][:s_used]
            s = out["angles_std"][cid][:s_used]
            ax.plot(x, m, marker='o', linewidth=2, color=col, label=out["labels"][cid])
            if repeats > 1:
                ax.fill_between(x, m - s, m + s, color=col, alpha=0.15, linewidth=0)

        ax.set_xlabel("Principal angle index (top-s PCs)")
        ax.set_ylabel("Principal angle (degrees)")
        ax.set_title(f"Principal angles compared to union of [{target_name}] clips sorted ascending based on averaged angle")
        ax.set_xticks(x)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            "principal_angle.pdf",
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
            transparent=True,
            format="pdf",
        )
        plt.show()
    

def compare_clusters_axis_motion_with_union(
    cluster_cycles_by_id: dict[int, list[tuple[int, np.ndarray]]],
    cluster_labels: dict[int, str],
    target_clusters: list[int | str],
    csv_path: str,
    rodent_rollout_dir: str,
    target_union: str = "groom",
    k_cycles: int = 5,
    s_top: int = 10,
    pad: int = 0,
    state_key: str = "qpos",
    sample_strategy: str = "random",
    repeats: int = 1,
    seed: int = 0,
    Kmax_plot: int | None = None,
    plot: bool = True,
) -> None:
    """
    Compare axis motion across clusters vs target union.

    Args:
        cluster_cycles_by_id: Dict mapping cluster_id to list of (clip_id, frame_indices)
        cluster_labels: Dict mapping cluster_id to label string
        target_clusters: List of cluster IDs or names
        csv_path: Path to behavior labels CSV
        rodent_rollout_dir: Directory containing clip HDF5 files
        target_union: Target union behavior type
        k_cycles: Number of cycles per cluster
        s_top: Number of top eigenvectors
        pad: Frame padding around cycles
        state_key: Key for state data in HDF5
        sample_strategy: Sampling strategy
        repeats: Number of repetitions
        seed: Random seed
        Kmax_plot: Maximum K for plotting
        plot: Whether to display plots
    """
    _compare_clusters_axis_motion_with_union(
        cluster_cycles_by_id, cluster_labels, target_clusters, rodent_rollout_dir,
        target_union, k_cycles, s_top, pad, state_key, sample_strategy,
        repeats, seed, Kmax_plot, plot
    )


def _compare_clusters_axis_motion_with_union(
    cluster_dict: dict,
    cluster_labels: dict,
    target_cluster_names: list,
    rodent_rollout_dir: str,
    max_loops=None,
    figsize=(12,7),
    out_pdf=None,
):
    def qpos_loader(clip_id):
        roll = load_rollout_from_source(int(clip_id), rodent_rollout_dir)
        return np.asarray(roll["qpos"])

    loop_means, loop_stds, union_means, union_stds, names = [], [], [], [], []

    for cname in target_cluster_names:
        cid = next((k for k,v in cluster_labels.items() if v == cname), None)
        if cid is None or cid not in cluster_dict: 
            continue

        # A) Loops: mean |Δ| per frame for each loop, then aggregate
        loop_vals = []
        for i, (clip_id, frames) in enumerate(cluster_dict[cid]):
            if max_loops is not None and i >= max_loops: break
            Q = qpos_loader(int(clip_id))
            idx = np.asarray(frames, dtype=int)
            if idx.size < 2: continue
            D = np.diff(Q[idx, :3], axis=0)
            loop_vals.append(np.mean(np.abs(D), axis=0))
        if not loop_vals: 
            continue
        L = np.vstack(loop_vals)
        loop_means.append(L.mean(axis=0))
        loop_stds.append(L.std(axis=0))

        # B) Union of clips: collect unique clips, compute per-clip mean |Δ| per frame, then aggregate
        clip_ids = sorted({int(clip_id) for (clip_id, _) in cluster_dict[cid]})
        clip_vals = []
        for clip_id in clip_ids:
            Q = qpos_loader(int(clip_id))
            if Q.shape[0] < 2: continue
            D = np.diff(Q[:, :3], axis=0)
            clip_vals.append(np.mean(np.abs(D), axis=0))
        if not clip_vals:
            continue
        C = np.vstack(clip_vals)
        union_means.append(C.mean(axis=0))
        union_stds.append(C.std(axis=0))

        names.append(cname)

    if not names:
        raise ValueError("No valid data for requested clusters")

    loop_means, loop_stds = np.vstack(loop_means), np.vstack(loop_stds)
    union_means, union_stds = np.vstack(union_means), np.vstack(union_stds)

    # Sort clusters by (x+y) mean of LOOPS, then z (descending)
    sort_key = (loop_means[:,0] + loop_means[:,1]) + 1e-3*loop_means[:,2]
    order = np.argsort(-sort_key)
    names = [names[i] for i in order]
    lm, ls = loop_means[order], loop_stds[order]
    um, us = union_means[order], union_stds[order]

    x = np.arange(len(names))
    w = 0.25
    fig = plt.figure(figsize=figsize, dpi=300)
    ax = fig.add_subplot(111)

    # Poster colors (same as your PC figure)
    blue = "#1f78b4"
    red  = "#e31a1c"

    # Cycles (solid)
    ax.bar(x - w, lm[:,0], width=w, yerr=ls[:,0], capsize=3,
        color=blue, alpha=0.4, label="|Δx|/frame (cycles)")

    ax.bar(x,     lm[:,1], width=w, yerr=ls[:,1], capsize=3,
        color=red,  alpha=0.4, label="|Δy|/frame (cycles)")

    ax.bar(x + w, lm[:,2], width=w, yerr=ls[:,2], capsize=3,
        color=blue, alpha=0.4, label="|Δz|/frame (cycles)")

    # Union baseline (same color, transparent)
    ax.bar(x - w, um[:,0], width=w, yerr=us[:,0], capsize=3,
        color=blue, alpha=0.6, edgecolor="white", linewidth=0.6,
        label="|Δx|/frame (union)")

    ax.bar(x,     um[:,1], width=w, yerr=us[:,1], capsize=3,
        color=red, alpha=0.6, edgecolor="white", linewidth=0.6,
        label="|Δy|/frame (union)")

    ax.bar(x + w, um[:,2], width=w, yerr=us[:,2], capsize=3,
        color=blue, alpha=0.6, edgecolor="white", linewidth=0.6,
        label="|Δz|/frame (union)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("mean |Δ| per frame")
    ax.set_title("Per-axis motion by cluster: loops vs union of clips")
    ax.axhline(0, ls="--", lw=0.8, alpha=0.4)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2)

    plt.tight_layout()
    if out_pdf is not None:
        plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0, transparent=True, format="pdf")
    plt.show()
    

def plot_cluster_subspace_mds(
    cluster_cycles_by_id,
    cluster_labels,
    target_clusters,
    rodent_rollout_dir,
    csv_path,
    label_unions=("walk", "rear", "groom", "turn", "wait"),
    k_cycles=5,
    s_top=10,
    pad=0,
    state_key="qpos",
    sample_strategy="random",
    seed=0,
    plot=True,
    figsize=(4.5, 4.5),
    save_path=None,
):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(csv_path)

    def load_qpos(i):
        roll = load_rollout_from_source(int(i), rodent_rollout_dir)
        try:
            Q = np.asarray(roll[state_key])
        except KeyError:
            Q = np.asarray(roll["states"])
        return Q[:, 7:]

    def to_cid(x):
        if isinstance(x, int):
            return x
        m = [k for k, v in cluster_labels.items() if v == str(x)]
        if not m:
            raise ValueError(f"cluster '{x}' not found")
        return m[0]

    def motif_sets(df_):
        walk_only = set()
        rear_only = set()
        groom_only = set()
        turn_any = set()
        wait_any = set()
        for _, r in df_.iterrows():
            c = int(r["clip_idx"])
            w = int(r.get("walk", 0)) == 1
            ww = int(r.get("walk_wait", 0)) == 1
            rz = int(r.get("head_turn_z", 0)) == 1
            re = int(r.get("rear", 0)) == 1
            gu = int(r.get("groom_up", 0)) == 1
            gd = int(r.get("groom_down", 0)) == 1
            txy = int(r.get("head_turn_xy", 0)) == 1

            walk_only_cond = (w or ww) and not (rz or re or gu or gd or txy)
            rear_only_cond = (rz or re) and not (w or ww or gu or gd or txy)
            groom_only_cond = (gu or gd) and not (w or ww or rz or re or txy)

            if walk_only_cond:
                walk_only.add(c)
            if rear_only_cond:
                rear_only.add(c)
            if groom_only_cond:
                groom_only.add(c)
            if txy:
                turn_any.add(c)
            if ww:
                wait_any.add(c)
        return {
            "walk": walk_only,
            "rear": rear_only,
            "groom": groom_only,
            "turn": turn_any,
            "wait": wait_any,
        }

    ms = motif_sets(df)

    bases = {}
    labels_pretty = {}
    types = {}
    s_global = None

    cids = [to_cid(c) for c in target_clusters]
    for cid in cids:
        entries = cluster_cycles_by_id.get(cid, [])
        if not entries:
            continue

        qcache = {}
        cycle_pool = []
        for cc, arr in entries:
            cc = int(cc)
            if cc not in qcache:
                qcache[cc] = load_qpos(cc)
            a = np.asarray(arr, int)
            raw_len = len(np.unique(a))
            cycle_pool.append({"clip": cc, "frames": a, "raw_len": raw_len})
        if not cycle_pool:
            continue

        def build_Q_cyc(sel_items):
            by_clip = {}
            for it in sel_items:
                by_clip.setdefault(it["clip"], []).append(it["frames"])
            chunks = []
            for cc, fr_lists in by_clip.items():
                Qc = qcache.get(cc, load_qpos(cc))
                T = Qc.shape[0]
                union = set()
                for a in fr_lists:
                    lb = np.clip(a - pad, 0, T - 1)
                    ub = np.clip(a + pad, 0, T - 1)
                    for s_, e_ in zip(lb, ub):
                        union.update(range(int(s_), int(e_) + 1))
                if union:
                    idx = np.array(sorted(union), int)
                    chunks.append(Qc[idx])
            if not chunks:
                return None
            return np.vstack(chunks)

        n_eff = min(k_cycles, len(cycle_pool))
        if n_eff <= 0:
            continue

        perm = rng.permutation(len(cycle_pool))
        shuffled = [cycle_pool[i] for i in perm]
        if sample_strategy == "longest":
            sel = sorted(shuffled, key=lambda t: t["raw_len"], reverse=True)[:n_eff]
        elif sample_strategy == "shortest":
            sel = sorted(shuffled, key=lambda t: t["raw_len"])[:n_eff]
        else:
            idx = rng.choice(len(cycle_pool), size=n_eff, replace=False)
            sel = [cycle_pool[i] for i in idx]

        Q_cyc = build_Q_cyc(sel)
        if Q_cyc is None or Q_cyc.shape[0] < 2:
            continue

        Qc = Q_cyc - Q_cyc.mean(0, keepdims=True)
        Cc = np.cov(Qc.T)
        e, U = np.linalg.eigh(Cc)
        ord_c = e.argsort()[::-1]
        U = U[:, ord_c]
        s_use = min(s_top, U.shape[0])

        key = f"cycle_{cid}"
        bases[key] = U[:, :s_use]
        labels_pretty[key] = f"{cluster_labels[cid]}"
        types[key] = "cycle"
        s_global = s_use if s_global is None else min(s_global, s_use)

    for lu in label_unions:
        lu_key = str(lu).lower()
        if lu_key not in ms:
            continue
        clips = sorted(ms[lu_key])
        if not clips:
            continue

        full_list = []
        for clip_id in clips:
            Q = load_qpos(clip_id)
            full_list.append(Q)
        if not full_list:
            continue
        Q_full = np.vstack(full_list)
        Qf = Q_full - Q_full.mean(0, keepdims=True)
        Cf = np.cov(Qf.T)
        e, U = np.linalg.eigh(Cf)
        ord_f = e.argsort()[::-1]
        U = U[:, ord_f]
        s_use = min(s_top, U.shape[0])

        key = f"label_{lu_key}"
        bases[key] = U[:, :s_use]
        labels_pretty[key] = lu_key
        types[key] = "label"
        s_global = s_use if s_global is None else min(s_global, s_use)

    keys = sorted(bases.keys())
    n = len(keys)
    if n == 0 or s_global is None:
        raise RuntimeError("No valid bases for MDS.")

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            Ui = bases[keys[i]][:, :s_global]
            Uj = bases[keys[j]][:, :s_global]
            ang = subspace_angles(Ui, Uj) * 180.0 / np.pi
            d = ang.mean()
            D[i, j] = D[j, i] = d

    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    pos = w > 0
    if pos.sum() < 2:
        raise RuntimeError("Not enough positive eigenvalues for 2D MDS.")
    k = 2
    lam = np.sqrt(w[:k])
    X = V[:, :k] * lam

    if plot:
        fig, ax = plt.subplots(figsize=figsize)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        base_roots = []
        for key in keys:
            lbl = labels_pretty[key]
            root = str(lbl).lower().split("_")[0]
            base_roots.append(root)
        unique_roots = sorted(set(base_roots))
        base_colors = {
            root: palette[i % len(palette)]
            for i, root in enumerate(unique_roots)
        }

        cycle_marker = "o"
        label_marker = "s"
        cycle_size = 70
        label_size = 120

        cycle_first = True
        label_first = True

        label_xy = X.copy()
        label_xy[:, 0] += 0.03
        label_xy[:, 1] += 0.03

        min_dist = 0.07
        for _ in range(200):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = label_xy[i, 0] - label_xy[j, 0]
                    dy = label_xy[i, 1] - label_xy[j, 1]
                    dist2 = dx * dx + dy * dy
                    if dist2 < min_dist * min_dist:
                        dist = np.sqrt(dist2) + 1e-8
                        push = 0.5 * (min_dist - dist)
                        ux = dx / dist
                        uy = dy / dist
                        label_xy[i, 0] += ux * push
                        label_xy[i, 1] += uy * push
                        label_xy[j, 0] -= ux * push
                        label_xy[j, 1] -= uy * push
                        moved = True
            if not moved:
                break

        for i, key in enumerate(keys):
            x, y = X[i, 0], X[i, 1]
            lx, ly = label_xy[i, 0], label_xy[i, 1]
            lbl = labels_pretty[key]
            root = str(lbl).lower().split("_")[0]
            color = base_colors[root]

            if types[key] == "cycle":
                ax.scatter(
                    x,
                    y,
                    s=cycle_size,
                    marker=cycle_marker,
                    edgecolor="k",
                    linewidth=0.6,
                    zorder=3,
                    color=color,
                    label="Cycle cluster" if cycle_first else None,
                )
                cycle_first = False
            else:
                ax.scatter(
                    x,
                    y,
                    s=label_size,
                    marker=label_marker,
                    edgecolor="k",
                    linewidth=0.6,
                    zorder=4,
                    color=color,
                    label="Label union" if label_first else None,
                )
                label_first = False

            ax.text(
                lx,
                ly,
                lbl,
                fontsize=16,
                ha="left",
                va="bottom",
            )

        ax.set_xlabel("MDS dimension 1", fontsize=13)
        ax.set_ylabel("MDS dimension 2", fontsize=13)

        ax.axhline(0, linewidth=0.7, color="k", alpha=0.2)
        ax.axvline(0, linewidth=0.7, color="k", alpha=0.2)

        handles, leg_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, fontsize=11, loc="best")

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=True)

        plt.show()
        

def plot_cluster_eigenbase_reconstruction_box(
    cluster_cycles_by_id,
    cluster_labels,
    target_clusters,
    csv_path,
    rodent_rollout_dir,
    k_pc=10,
    n_cycles=5,
    repeats=5,
    sample_strategy="shortest",
    pad=0,
    state_key="qpos",
    seed=0,
):
    import numpy as np, pandas as pd, h5py, matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    df = pd.read_csv(csv_path)

    def load_qpos(i):
        roll = load_rollout_from_source(int(i), rodent_rollout_dir)
        return np.asarray(roll[state_key])[:, 7:]

    def to_cid(x):
        if isinstance(x, int):
            return x
        m = [k for k, v in cluster_labels.items() if v == str(x)]
        if not m:
            raise ValueError(f"cluster '{x}' not found")
        return m[0]

    def motif_sets(df_):
        walk, rear, groom, turn, wait = set(), set(), set(), set(), set()
        for _, r in df_.iterrows():
            c = int(r["clip_idx"])
            w = int(r.get("walk", 0)) == 1
            ww = int(r.get("walk_wait", 0)) == 1
            rz = int(r.get("head_turn_z", 0)) == 1
            re = int(r.get("rear", 0)) == 1
            gu = int(r.get("groom_up", 0)) == 1
            gd = int(r.get("groom_down", 0)) == 1
            txy = int(r.get("head_turn_xy", 0)) == 1

            walk_only = (w or ww) and not (rz or re or gu or gd or txy)
            rear_only = (rz or re) and not (w or ww or gu or gd or txy)
            groom_only = (gu or gd) and not (w or ww or rz or re or txy)

            if walk_only:
                walk.add(c)
            if rear_only:
                rear.add(c)
            if groom_only:
                groom.add(c)
            if txy:
                turn.add(c)
            if ww:
                wait.add(c)
        return {"walk": walk, "rear": rear, "groom": groom, "turn": turn, "wait": wait}

    def parse_tokens(n):
        n = str(n).lower()
        out = []
        for k in ["walk", "rear", "groom", "turn", "wait"]:
            if k in n:
                out.append(k)
        return out

    ms = motif_sets(df)
    cids = [to_cid(c) for c in target_clusters]

    real_all = []
    cyc_all = []
    coverage_all = []

    for cid in cids:
        cname = cluster_labels[cid]
        toks = parse_tokens(cname)
        clips = set()
        for t in toks:
            clips |= ms.get(t, set())
        clips = sorted(clips)
        if not clips:
            continue

        Q_full = np.vstack([load_qpos(c) for c in clips])
        Qf = Q_full - Q_full.mean(0)
        Cf = np.cov(Qf.T)
        ef, _ = np.linalg.eigh(Cf)
        ef = np.sort(ef)[::-1]
        total = np.sum(ef) + 1e-12
        k_eff = min(k_pc, len(ef))
        real = np.cumsum(ef[:k_eff]) / total
        real_all.append(real)

        entries = cluster_cycles_by_id.get(cid, [])
        if not entries:
            continue

        qcache = {}
        cyc_pool = []
        for cc, arr in entries:
            cc = int(cc)
            if cc not in qcache:
                qcache[cc] = load_qpos(cc)
            T = qcache[cc].shape[0]
            a = np.asarray(arr, int)
            lb = np.clip(a - pad, 0, T - 1)
            ub = np.clip(a + pad, 0, T - 1)
            idx = []
            for s, e in zip(lb, ub):
                idx += range(s, e + 1)
            cyc_pool.append({"clip": cc, "frames": a, "len": len(np.unique(idx))})

        if not cyc_pool:
            continue

        for _ in range(repeats):
            n_eff = min(n_cycles, len(cyc_pool))
            if sample_strategy == "longest":
                sel = sorted(cyc_pool, key=lambda t: t["len"], reverse=True)[:n_eff]
            elif sample_strategy == "shortest":
                sel = sorted(cyc_pool, key=lambda t: t["len"])[:n_eff]
            else:
                idx = rng.choice(len(cyc_pool), size=n_eff, replace=False)
                sel = [cyc_pool[i] for i in idx]

            cyc_by = {}
            for it in sel:
                cyc_by.setdefault(it["clip"], []).append(it["frames"])

            chunks = []
            for cc, arrs in cyc_by.items():
                Qc = qcache[cc]
                T = Qc.shape[0]
                u = set()
                for a in arrs:
                    lb = np.clip(a - pad, 0, T - 1)
                    ub = np.clip(a + pad, 0, T - 1)
                    for s, e in zip(lb, ub):
                        u.update(range(s, e + 1))
                if u:
                    idx_sorted = np.array(sorted(u), int)
                    chunks.append(Qc[idx_sorted])

            if not chunks:
                continue

            Q = np.vstack(chunks)
            coverage_all.append(100.0 * Q.shape[0] / Q_full.shape[0])

            Q = Q - Q.mean(0)
            C = np.cov(Q.T)
            ec, U = np.linalg.eigh(C)
            U = U[:, np.argsort(ec)[::-1]]
            A = U.T @ Cf @ U
            diag = np.sort(np.diag(A))[::-1]
            diag = diag[:k_eff]
            cyc = np.cumsum(diag) / total
            cyc_all.append(cyc)

    if not real_all or not cyc_all:
        return None

    real_all = np.vstack(real_all)
    cyc_all = np.vstack(cyc_all)
    coverage_all = np.asarray(coverage_all, float)
    cov_mean = float(np.nanmean(coverage_all))

    k_vals = np.arange(1, real_all.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=180)

    data = []
    for i, _ in enumerate(k_vals):
        r = real_all[:, i]
        c = cyc_all[:, i]
        data.append([r, c])

    colors = ["#1f78b4", "#e31a1c"]

    positions = []
    vals = []
    for i, (r, c) in enumerate(data):
        base = 2 * i
        positions.append(base)
        vals.append(r)
        positions.append(base + 1)
        vals.append(c)

    bp = ax.boxplot(
        vals,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.1},
        capprops={"linewidth": 1.1},
    )

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % 2])
        patch.set_alpha(0.72)

    ax.set_xticks([(2 * i + 0.5) for i in range(len(k_vals))])
    ax.set_xticklabels([str(k) for k in k_vals], fontsize=11)
    ax.set_xlabel("Number of PCs (k)")
    ax.set_ylabel("Variance explained")
    ax.set_ylim(0, 1.05)

    ax.plot([], [], color=colors[0], label="Full PC")
    ax.plot(
        [],
        [],
        color=colors[1],
        label=f"Cycle PC (using {cov_mean:.1f}%)",
    )
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(
        "reconstruction_boxplot_cov.pdf",
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )
    plt.show()