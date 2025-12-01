'''Functions for post-processing, namely chunking and clustering'''

from typing import Callable, Any
import numpy as np
import random
import h5py

from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from collections import Counter, defaultdict

import networkx as nx
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import ot

from vnl_tda.utils.rendering import render_from_saved_rollout
from vnl_tda.utils.load_utils import load_from_h5py


def _expand_frames_nonconsec(
    loop_frames: np.ndarray | list[int],
    pad: int,
    clip_len: int,
) -> np.ndarray:
    if pad <= 0:
        return np.unique(np.asarray(loop_frames, dtype=int))
    chunks = [np.arange(max(0, f - pad), min(clip_len, f + pad + 1)) for f in np.asarray(loop_frames, dtype=int)]
    return np.unique(np.concatenate(chunks))


def _expand_indices(
    indices: list[int],
    pad: int,
    T: int,
) -> list[int]:
    idx = set()
    for i in indices:
        a = max(0, int(i) - pad)
        b = min(T - 1, int(i) + pad)
        idx.update(range(a, b + 1))
    return sorted(idx)


def _uniform_downsample(
    frames: np.ndarray | list[int],
    target_len: int,
) -> np.ndarray | None:
    frames = np.unique(np.asarray(frames, dtype=int))
    if len(frames) < target_len:
        return None
    idx = np.linspace(0, len(frames) - 1, target_len)
    idx = np.rint(idx).astype(int)
    return frames[idx]


def _normalize_dist_matrix(
    D: np.ndarray,
) -> np.ndarray:
    d = D.astype(float).copy()
    np.fill_diagonal(d, np.nan)
    row_med = np.nanmedian(d, axis=1)
    col_med = np.nanmedian(d, axis=0)
    s = np.sqrt(np.outer(row_med, col_med))
    s[s == 0] = 1.0
    Dn = D / s
    Dn[np.isnan(Dn)] = 0.0
    np.fill_diagonal(Dn, 0.0)
    return Dn


def _compute_wasserstein_distances(
    X: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise Wasserstein distances for intention vectors.

    Args:
        X: Array of shape (n_samples, dim)

    Returns:
        Symmetric distance matrix of shape (n_samples, n_samples)
    """
    n = X.shape[0]
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = wasserstein_distance(X[i], X[j])
            W[i, j] = W[j, i] = d
    return W


def _merge_overlapping_loops(
    loops: list[np.ndarray | list[int]],
) -> list[np.ndarray]:
    """
    Merge loops that share any frame index.

    Args:
        loops: List of loops, each a list/array of frame indices

    Returns:
        List of merged loops (union of overlapping indices)
    """
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(len(loops)))

    for i in range(len(loops)):
        for j in range(i + 1, len(loops)):
            if set(loops[i]).intersection(loops[j]):
                G.add_edge(i, j)

    merged_loops = []
    for component in nx.connected_components(G):
        merged = set()
        for idx in component:
            merged.update(loops[idx])
        merged_loops.append(np.array(sorted(merged)))

    return merged_loops


def drop_middle_nearby_loops(
    loops_at_len: list[np.ndarray],
    frame_tol: int = 5,
) -> list[np.ndarray]:
    """
    Filter out loops that are sandwiched between nearby loops.

    Args:
        loops_at_len: List of loop arrays
        frame_tol: Frame tolerance for considering loops as nearby

    Returns:
        Filtered list of loops
    """
    if not loops_at_len:
        return []

    # sort loops by starting frame so that "previous" and "next" make sense.
    loops_sorted = sorted(loops_at_len, key=lambda loop: loop[0])
    keep_mask = [True] * len(loops_sorted) # assume we keep everything

    for i in range(1, len(loops_sorted) - 1):
        first_prev = loops_sorted[i - 1][0]
        first_curr = loops_sorted[i][0]
        first_next = loops_sorted[i + 1][0]

        close_prev = abs(first_curr - first_prev) <= frame_tol
        close_next = abs(first_next - first_curr) <= frame_tol

        if close_prev and close_next:
        # if close_prev or close_next:
            keep_mask[i] = False

    return [loop for loop, keep in zip(loops_sorted, keep_mask) if keep]


def chunk_frames_by_loop(
    rodent_rollout_dir: str,
    loops_idx_cocycle: list[list[int]],
) -> tuple[dict[int, list], dict[int, list]]:
    """
    Extract loop and non-loop frames for video rendering.

    Args:
        rodent_rollout_dir: Directory containing clip HDF5 files
        loops_idx_cocycle: List of loop indices per clip

    Returns:
        Tuple of (loop_chunks, non_loop_chunks) dictionaries
    """
    loop_chunks = {}
    non_loop_chunks = {}
    for clip_idx, loop_indices in enumerate(loops_idx_cocycle):
        if not loop_indices:
            continue
        loop_indices = np.array(loop_indices)
        min_idx, max_idx = loop_indices.min(), loop_indices.max()
        with h5py.File(f"{rodent_rollout_dir}/clip_{clip_idx}.h5", "r") as h5file:
            rollout = load_from_h5py(h5file)
        frames = render_from_saved_rollout(rollout)
        loop_chunks[clip_idx] = frames[min_idx:max_idx+1]
        non_loop_chunks[clip_idx] = frames[:min_idx] + frames[max_idx+1:]
    return loop_chunks, non_loop_chunks


def chunk_loops_by_clustered_vpath(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    rodent_rollout_dir: str,
    pad: int = 25,
    sample: int | None = None,
    rng: random.Random | None = None,
) -> dict[int, list[list[np.ndarray]]]:
    """
    Build video-frame chunks for every behavior cluster.

    Args:
        cluster_dict: Dict mapping cluster_id to list of (clip_id, frame_indices)
        rodent_rollout_dir: Directory containing clip HDF5 files
        pad: Padding around each frame index
        sample: Maximum samples per cluster (None for all)
        rng: Random number generator for sampling

    Returns:
        Dict mapping cluster_id to list of frame chunks
    """
    if rng is None:
        rng = random

    cluster_chunks = {}
    for cid, items in cluster_dict.items():
        if sample is not None and len(items) > sample:
            items = rng.sample(items, sample)

        cluster_chunks[cid] = []

        by_clip = {}
        for clip_id, frames in items:
            by_clip.setdefault(clip_id, []).append(frames)

        for clip_id, frame_lists in by_clip.items():
            with h5py.File(f"{rodent_rollout_dir}/clip_{clip_id}.h5", "r") as h5file:
                rollout = load_from_h5py(h5file)
            full_frames = render_from_saved_rollout(rollout)
            T = len(full_frames)

            for frames in frame_lists:
                ordered = _expand_indices(frames, pad=pad, T=T) if pad and pad > 0 else sorted(int(f) for f in frames)
                chunk = [full_frames[i] for i in ordered]
                cluster_chunks[cid].append(chunk)

    return cluster_chunks


def cluster_vpath_with_wasserstein2d_fixedlen_agg(
    all_loops: list[list[np.ndarray]],
    intention: np.ndarray,
    pad: int = 0,
    min_len: int = 30,
    max_len: int = 50,
    n_clusters: int = 3,
    n_clips: int = 800,
    drop_nearby_loop_num: int = 20,
    skip_clips: list[int] = [37, 152, 163, 274, 281],
    emd_fn: Callable = ot.emd2,
    squared_cost: bool = False,
    only_max: bool = False,
    use_full_union: bool = False,
) -> tuple[dict[int, list], list[list[np.ndarray]]]:
    """
    Cluster loops via multi-dimensional Wasserstein distance (EMD).

    Args:
        all_loops: List of loops per clip
        intention: Intention array of shape (n_clips, n_frames, D)
        pad: Frame padding around loop indices
        min_len: Minimum loop length to consider
        max_len: Maximum loop length to consider
        n_clusters: Number of clusters
        n_clips: Number of clips to process
        drop_nearby_loop_num: Frame tolerance for dropping nearby loops
        skip_clips: Clip indices to skip
        emd_fn: EMD computation function
        squared_cost: Whether to square the cost matrix
        only_max: Whether to keep only the longest loop per clip
        use_full_union: Whether to merge overlapping loops

    Returns:
        Tuple of (clusters_dict, cleaned_loops)
    """
    windows = []
    all_cleaned_loops = []
    for clip_id in range(n_clips):
        if clip_id in skip_clips:
            continue
        
        loops = [
            loop for loop in all_loops[clip_id]
            if min_len <= len(loop) <= max_len
        ]
        
        if only_max and len(loops) > 0:
            loops = [max(loops, key=len)]
        elif use_full_union:
            loops = drop_middle_nearby_loops(loops, frame_tol=drop_nearby_loop_num)
            loops = _merge_overlapping_loops(loops)
        else:
            loops = drop_middle_nearby_loops(loops, frame_tol=drop_nearby_loop_num)
        
        all_cleaned_loops.append(loops)
            
        # print(f'In clip {clip_id}, we still have {len(loops)} number of loops')
        for loop in loops:
            f0, f1 = loop.min(), loop.max()
            start = max(0, f0 - pad)
            end = min(len(intention[clip_id]), f1 + pad + 1)
            frames = np.arange(start, end)
            windows.append((clip_id, frames))

    M = len(windows)
    D = np.zeros((M, M), dtype=float)

    # fill distance matrix via multi-dimensional EMD
    for i, (cid_i, fr_i) in enumerate(windows):
        X = intention[cid_i, fr_i] # shape (n_i, dim)
        w = np.ones(len(fr_i)) / len(fr_i) # uniform weights
        for j in range(i+1, M):
            cid_j, fr_j = windows[j]
            Y = intention[cid_j, fr_j] # shape (n_j, dim)
            v = np.ones(len(fr_j)) / len(fr_j)
            C = ot.dist(X, Y) # pairwise cost matrix
            if squared_cost:
                C **= 2
            emd_cost = emd_fn(w, v, C)
            if squared_cost:
                emd_cost = np.sqrt(emd_cost)
            D[i, j] = D[j, i] = emd_cost

    # cluster on the precomputed distance matrix
    labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average"
    ).fit_predict(D)

    clusters = {}
    for lbl, win in zip(labels, windows):
        clusters.setdefault(lbl, []).append(win)
    
    return clusters, all_cleaned_loops


def cluster_vpath_with_wasserstein2d_fixedlen_kmeans(
    all_loops: list[list[np.ndarray]],
    intention: np.ndarray,
    pad: int = 0,
    min_len: int = 30,
    max_len: int = 200,
    target_len: int = 50,
    n_clusters: int = 3,
    n_clips: int = 800,
    drop_nearby_loop_num: int = 0,
    skip_clips: tuple[int, ...] = (37, 152, 163, 274, 281),
    squared_cost: bool = False,
    only_max: bool = False,
    use_full_union: bool = False,
    random_state: int = 0,
    return_mats: bool = False,
) -> tuple[dict[int, list], list[np.ndarray]] | tuple[dict, list, np.ndarray, np.ndarray, list, np.ndarray]:
    """
    Cluster loops via Wasserstein distance with KMeans in MDS-embedded space.

    Args:
        all_loops: List of loops per clip
        intention: Intention array of shape (n_clips, n_frames, D)
        pad: Frame padding around loop indices
        min_len: Minimum loop length to consider
        max_len: Maximum loop length to consider
        target_len: Target length for uniform downsampling
        n_clusters: Number of clusters for KMeans
        n_clips: Number of clips to process
        drop_nearby_loop_num: Frame tolerance for dropping nearby loops
        skip_clips: Clip indices to skip
        squared_cost: Whether to square the cost matrix
        only_max: Whether to keep only the longest loop per clip
        use_full_union: Whether to merge overlapping loops
        random_state: Random seed for reproducibility
        return_mats: Whether to return distance matrices

    Returns:
        Tuple of (clusters_dict, cleaned_loops) or extended tuple with matrices
    """
    windows = []
    cleaned_loops = []
    for clip_id in range(n_clips):
        if clip_id in skip_clips:
            continue
        loops = [loop for loop in all_loops[clip_id] if min_len <= len(loop) <= max_len]
        if only_max and len(loops) > 0:
            loops = [max(loops, key=len)]
        elif use_full_union:
            from vnl_tda.pipelines.tda.skeletonization.rendering_postprocess import drop_middle_nearby_loops
            from vnl_tda.pipelines.tda.skeletonization.skeletonization import _merge_overlapping_loops
            loops = drop_middle_nearby_loops(loops, frame_tol=drop_nearby_loop_num)
            loops = _merge_overlapping_loops(loops)
        else:
            if drop_nearby_loop_num > 0:
                from vnl_tda.pipelines.tda.skeletonization.rendering_postprocess import drop_middle_nearby_loops
                loops = drop_middle_nearby_loops(loops, frame_tol=drop_nearby_loop_num)
        clip_len = len(intention[clip_id])
        for loop in loops:
            expanded = _expand_frames_nonconsec(loop, pad, clip_len)
            sampled = _uniform_downsample(expanded, target_len)
            if sampled is None:
                continue
            windows.append((clip_id, sampled))
            cleaned_loops.append(np.asarray(loop, dtype=int))

    M = len(windows)
    if M == 0:
        return ({}, cleaned_loops) if not return_mats else ({}, cleaned_loops, None, None)

    D = np.zeros((M, M), dtype=float)
    for i, (ci, fi) in enumerate(windows):
        Xi = intention[ci, fi]
        wi = np.full(target_len, 1.0 / target_len)
        for j in range(i + 1, M):
            cj, fj = windows[j]
            Xj = intention[cj, fj]
            vj = np.full(target_len, 1.0 / target_len)
            C = ot.dist(Xi, Xj)
            if squared_cost:
                C = C ** 2
            val = ot.emd2(wi, vj, C)
            if squared_cost:
                val = np.sqrt(val)
            D[i, j] = D[j, i] = val

    Dn = _normalize_dist_matrix(D)

    # project the precomputed distances to Euclidean space using MDS
    # finds the 3d eucledian space such that distance are ollowing the distance matrix
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=random_state)
    X_embedded = mds.fit_transform(Dn)

    # run standard KMeans in the embedded space
    labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto').fit_predict(X_embedded)

    clusters = {}
    for lbl, win in zip(labels, windows):
        clusters.setdefault(int(lbl), []).append(win)

    if return_mats:
        return clusters, cleaned_loops, D, Dn, windows, labels
    return clusters, cleaned_loops


def build_cluster_prototypes_from_dict(
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]],
    int_rodent: np.ndarray,
    target_len: int = 50,
    agg: str = "mean",
) -> dict[int, np.ndarray]:
    """
    Build prototype sequences for each cluster.

    Args:
        cluster_dict: Dict mapping cluster_id to list of (clip_id, frame_indices)
        int_rodent: Intention array of shape (n_clips, n_frames, D)
        target_len: Target length for uniform downsampling
        agg: Aggregation method ("mean" or "medoid")

    Returns:
        Dict mapping cluster_id to prototype array
    """
    protos = {}
    for cid, entries in cluster_dict.items():
        seqs = []
        for clip_id, frames in entries:
            f = _uniform_downsample(frames, target_len)
            if f is None:
                continue
            Xi = int_rodent[clip_id, f]
            seqs.append(Xi)
        if not seqs:
            continue
        X = np.stack(seqs, axis=0)
        if agg == "mean":
            protos[cid] = X.mean(axis=0)
        elif agg == "medoid":
            n = len(X)
            D = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(X[i] - X[j])
                    D[i, j] = D[j, i] = d
            protos[cid] = X[np.argmin(D.sum(axis=1))]
    return protos


def cluster_distance_matrix(
    protos: dict[int, np.ndarray],
    metric: str = "emd",
) -> tuple[np.ndarray, list[int]]:
    """
    Compute pairwise distance matrix between cluster prototypes.

    Args:
        protos: Dict mapping cluster_id to prototype array
        metric: Distance metric ("emd" or "euclid")

    Returns:
        Tuple of (distance_matrix, cluster_ids)
    """
    ids = sorted(protos.keys())
    m = len(ids)
    C = np.zeros((m, m), dtype=float)
    wi = None
    for i in range(m):
        Xi = protos[ids[i]]
        if wi is None:
            T = Xi.shape[0]
            wi = np.full(T, 1.0 / T)
        for j in range(i + 1, m):
            Xj = protos[ids[j]]
            if metric == "emd":
                M = ot.dist(Xi, Xj)
                d = ot.emd2(wi, wi, M)
                C[i, j] = C[j, i] = float(np.sqrt(d))
            elif metric == "euclid":
                d = np.linalg.norm(Xi - Xj)
                C[i, j] = C[j, i] = float(d)
    return C, ids


def cluster_knn_graph(
    C: np.ndarray,
    ids: list[int],
    cluster_labels: dict[int, str],
    k: int = 2,
    threshold: float | None = None,
) -> nx.Graph:
    """
    Build a k-NN graph from cluster distance matrix.

    Args:
        C: Distance matrix of shape (n_clusters, n_clusters)
        ids: List of cluster IDs
        cluster_labels: Dict mapping cluster_id to label string
        k: Number of nearest neighbors
        threshold: Optional distance threshold for additional edges

    Returns:
        NetworkX graph with clusters as nodes
    """
    G = nx.Graph()
    for cid in ids:
        G.add_node(cid, label=cluster_labels[cid])
    m = len(ids)
    for i in range(m):
        if k is not None and k > 0:
            nbrs = np.argsort(C[i])
            for idx in nbrs[1:k+1]:
                u, v = ids[i], ids[idx]
                w = C[i, idx]
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=w)
        if threshold is not None:
            for j in range(i + 1, m):
                if C[i, j] <= threshold and not G.has_edge(ids[i], ids[j]):
                    G.add_edge(ids[i], ids[j], weight=C[i, j])
    return G