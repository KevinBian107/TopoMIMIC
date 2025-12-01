'''Functions for post-processing, namely chunking and clustering'''

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


def _expand_frames_nonconsec(loop_frames, pad, clip_len):
    if pad <= 0:
        return np.unique(np.asarray(loop_frames, dtype=int))
    chunks = [np.arange(max(0, f - pad), min(clip_len, f + pad + 1)) for f in np.asarray(loop_frames, dtype=int)]
    return np.unique(np.concatenate(chunks))


def _expand_indices(indices, pad, T):
    idx = set()
    for i in indices:
        a = max(0, int(i) - pad)
        b = min(T - 1, int(i) + pad)
        idx.update(range(a, b + 1))
    return sorted(idx)


def _uniform_downsample(frames, target_len):
    frames = np.unique(np.asarray(frames, dtype=int))
    if len(frames) < target_len:
        return None
    idx = np.linspace(0, len(frames) - 1, target_len)
    idx = np.rint(idx).astype(int)
    return frames[idx]


def _normalize_dist_matrix(D):
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


def _compute_wasserstein_distances(X: np.ndarray):
    """
    Computing wasserstein distances for intentions
    """
    n = X.shape[0]
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = wasserstein_distance(X[i], X[j])
            W[i, j] = W[j, i] = d
    return W


def _merge_overlapping_loops(loops):
    """
    Given a list of loops (each a list/array of frame indices), 
    merge loops that share any frame index.
    
    - Loops that intersect are merged (union of indices).
    - Non-intersecting loops are kept separate.
    """
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(len(loops)))

    # edges between loops that share any frame index
    for i in range(len(loops)):
        for j in range(i + 1, len(loops)):
            if set(loops[i]).intersection(loops[j]):
                G.add_edge(i, j)

    # connected components = merged loops
    merged_loops = []
    for component in nx.connected_components(G):
        merged = set()
        for idx in component:
            merged.update(loops[idx])
        merged_loops.append(np.array(sorted(merged)))

    return merged_loops


def _uniform_downsample(frames, target_len):
    frames = np.unique(np.asarray(frames, dtype=int))
    if len(frames) < target_len:
        return None
    idx = np.linspace(0, len(frames) - 1, target_len)
    return frames[np.rint(idx).astype(int)]


def drop_middle_nearby_loops(
    loops_at_len,
    frame_tol=5
    ):
    """
    Keep only loops whose first index is not within frame_tol frames of BOTH its immediate predecessor and successor when
    loops are ordered by their first index. (i.e. if loop[0] has FI as 2, loop[1] should not have FI < 22 if fram_tol = 20)

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


def chunk_frames_by_loop(rodent_rollout_dir, loops_idx_cocycle):
    """
    Extract loop & non-loop index for stacking video rendering
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
    cluster_dict,
    rodent_rollout_dir,
    pad=25,
    sample=None,
    rng=None
):
    """
    Build video-frame chunks for every behaviour cluster identified by each vpath.

    Instead of taking the min/max span, this collects exactly the frames that
    belong to the cluster (optionally expanded by +/- `pad` around each index),
    in chronological order (non-consecutive indices allowed).
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
    all_loops,
    intention,
    pad=0,
    min_len=30,
    max_len=50,
    n_clusters=3,
    n_clips=800,
    drop_nearby_loop_num=20,
    skip_clips=[37, 152, 163, 274, 281],
    emd_fn=ot.emd2,
    squared_cost=False,
    only_max=False,
    use_full_union=False,
):
    """
    Cluster each vpath by computing the full multi-dimensional
    Wasserstein distance between their intention-path point clouds.
    
    - no longer need to aggregate intention vectors
    - EMD naturally handles different path lengths.
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
    all_loops,
    intention,
    pad=0,
    min_len=30,
    max_len=200,
    target_len=50,
    n_clusters=3,
    n_clips=800,
    drop_nearby_loop_num=0,
    skip_clips=(37, 152, 163, 274, 281),
    squared_cost=False,
    only_max=False,
    use_full_union=False,
    random_state=0,
    return_mats=False,
):
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


def build_cluster_prototypes_from_dict(cluster_dict, int_rodent, target_len=50, agg="mean"):
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


def cluster_distance_matrix(protos, metric="emd"):
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


def cluster_knn_graph(C, ids, cluster_labels, k=2, threshold=None):
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