'''Functions for doing skeletonization of the intention space; 
persistent homology + discrete Morse graph'''

import numpy as np
import networkx as nx
import os

import tempfile

from ripser import ripser
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from typing import Union


def _knn_weights_for_persistence(pts, k=10):
    """
    For cocycle use: w_p  = sqrt( (1/k) Σ_{q∈kNN(p)} d(p,q)^2 )
    
    returns the weighted distance of distance to measure
    """
    k = min(k, pts.shape[0] - 1) 
    D = pairwise_distances(pts) 
    idx = np.argpartition(D, kth=k+1, axis=1)[:, 1:k+1]   # skip self-dist (0)
    w2 = (D[np.arange(len(pts))[:, None], idx] ** 2).mean(axis=1)
    return np.sqrt(w2)


def _weighted_distance_matrix(pts, w):
    """
    For cocycle use, return full power-distance matrix 
        - D_pow[i,j] = sqrt(||p_i-p_j||^2 + w_i^2 + w_j^2)
    """
    d2 = squareform(pdist(pts, metric='sqeuclidean'))
    W2 = w[:, None] ** 2 + w[None, :] ** 2
    return np.sqrt(d2 + W2)


def _weighted_rips_cocycle_ripser(pts, k=10):
    """
    For cocycle use, most persistent H1 generator as a set of point indices, via ripser on
    the power-distance matrix.
    """
    w      = _knn_weights_for_persistence(pts, k=k)
    D_pow  = _weighted_distance_matrix(pts, w)

    res = ripser(D_pow, distance_matrix=True, maxdim=1, do_cocycles=True)
    H1 = res["dgms"][1]
    
    # cocycle gives edges between point‐indices,
    # it tracks when new co-homology is born when adding in which simplex
    C1 = res["cocycles"][1]
    if len(H1) == 0:
        return []

    # pick generator with longest persistence
    best_idx   = np.argmax(H1[:, 1] - H1[:, 0])
    best_cycle = C1[best_idx]        # list of (i,j, coeff) triples

    verts = sorted({i for (i, j, _) in best_cycle} |
                   {j for (i, j, _) in best_cycle})
    return verts


def _get_representative_cycle_ripser(points, maxdim=1):
    """
    For cocycle use, finds the single most persistent H1 cycle in points via Ripser's cocycles.
    Returns the set of point-indices that participate in that cycle.
    """
    r = ripser(points, maxdim=maxdim, do_cocycles=True)
    dgm1 = r['dgms'][1]
    cocycles = r['cocycles'][1]

    if len(dgm1) == 0:
        return []

    # pick the index of the interval with largest persistence
    pers = dgm1[:,1] - dgm1[:,0]
    idx = np.argmax(pers)
    cocycle = cocycles[idx]  # list of (i, j, weight)

    # collect all points i or j that appear with nonzero weight
    verts = set()
    for i,j,w in cocycle:
        if abs(w) > 1e-6:
            verts.add(i)
            verts.add(j)
    return list(verts)


def _unstack_and_average(points, cycle_inds, m, D):
    """
    For cocycle use, given flattened windows points and indices cycle_inds,
    reshape back to (2m+1, D) and average over the temporal axis.
    """
    expected = (2*m+1)*D
    seqs = []
    for i in cycle_inds:
        v = points[i]
        if v.shape[0] != expected:
            continue
        window = v.reshape(2*m+1, D)
        seqs.append(window.mean(axis=0))
    return np.vstack(seqs) if seqs else np.zeros((0, D))


def _reorder_via_tsp(sequence):
    """For cocycle use,
    solve a TSP on the averaged cycle points to make a smooth loop."""
    if len(sequence) <= 1:
        return sequence
    Dmat = cdist(sequence, sequence)
    G = nx.complete_graph(len(sequence))
    for u,v in G.edges():
        G[u][v]['weight'] = Dmat[u,v]
    tour = nx.approximation.traveling_salesman_problem(G, cycle=True)
    return sequence[tour]


def extract_cycles_from_dm_graph(G, points):
    """
    For DM Graph, extract cycles from the DM graph by constructing an MST and
    finding the edges on the graph that does not lay in the MST
    """
    cycles = []
    
    # make a minimum spanning tree
    if len(G.edges()) > 0:
        try:
            spanning_tree = nx.minimum_spanning_tree(G)
            
            # For each edge not in spanning tree, find the cycle it creates
            for u, v in G.edges():
                if not spanning_tree.has_edge(u, v):
                    try:
                        path = nx.shortest_path(spanning_tree, u, v)
                        cycle = path + [path[0]]  # Close the loop
                        cycles.append(np.array(cycle, dtype=int))
                    except nx.NetworkXNoPath:
                        continue
        except nx.NetworkXException:
            # Graph might be empty or disconnected
            pass
    
    return cycles


def extract_skeleton_pcd_dm(
    points,
    k=10,
    pers_delta=0.25,
    epsilon=0.99,
    output_dir=None,
    cleanup=True
):
    """
    For DM graph, PCD DM graph reconstruction API:
        - Build sparse weighted rips filtraion
        - Calculate persistence
        - Order weights
        - Construct the DM graph and save out
    """
    
    repo_path = "/root/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM"
    os.chdir(repo_path)
    print("Changed working directory to:", os.getcwd())
    import dmpcd.pcd as pcd
    
    if output_dir is None:
        temp_dir = tempfile.mkdtemp()
        output_dir = temp_dir
        should_cleanup = cleanup
    else:
        should_cleanup = False
        os.makedirs(output_dir, exist_ok=True)
    
    input_filename = os.path.join(output_dir, 'features.txt')
    np.savetxt(input_filename, points, fmt='%.8f')
    
    try:
        weights_filename = os.path.join(output_dir, 'weights.txt')
        
        # build sparse weighted Rips filtration
        pcd.build_sparse_weighted_rips_filtration(
            input_filename, 
            output_dir, 
            k=k, 
            metric='euclidean', 
            epsilon=epsilon
        )
        
        # compute persistence
        filtration_filename = os.path.join(output_dir, 'sparse_weighted_rips_filtration.txt')
        pcd.compute_persistence_swr(filtration_filename, output_dir)
        
        # reorder weights
        edge_filename = os.path.join(output_dir, "edge_for_morse_only.txt")
        sorted_weights_filename = os.path.join(output_dir, "sorted-weights.txt")
        pcd.reorder_weights(weights_filename, sorted_weights_filename)
        
        # graph reconstruction
        morse_dir = os.path.join(output_dir, str(pers_delta) + '/')
        os.makedirs(morse_dir, exist_ok=True)
        pcd.compute_graph_reconstruction(
            sorted_weights_filename, 
            edge_filename, 
            pers_delta, 
            morse_dir
        )
        
        # read in for constructing nx graph
        result_edge_filename = os.path.join(morse_dir, 'dimo_edge.txt')
        
        G = nx.Graph()
        G.add_nodes_from(range(len(points)))
        
        if os.path.exists(result_edge_filename):
            edges = np.loadtxt(result_edge_filename, dtype=int)
            if edges.size > 0:
                if edges.ndim == 1:  # Single edge case
                    edges = edges.reshape(1, -1)
                for edge in edges:
                    if len(edge) >= 2:
                        G.add_edge(edge[0], edge[1])
        
        # extract cycles
        cycles = extract_cycles_from_dm_graph(G, points)
        
        return cycles, G
        
    finally:
        if should_cleanup and output_dir == temp_dir:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def extract_generative_loops_cocycles_based(intentions, m=5, max_clips=None):
    """
    we extracts a representative H1 loop and find its coordinate in the original intention space.
    Then we unstack the stacked intention vector to a matrix and average them.
    We return these loop significant intention coordinates when walking one complete tour with TSP
    """
    D = intentions.shape[2]
    n_clips = len(intentions) if max_clips is None else min(len(intentions), max_clips)
    loops = []
    loops_idx = []
    all_pts = []
    
    for clip_id in range(n_clips):
        clip = intentions[clip_id]
        
        # build windows
        pts = np.stack([clip[i-m:i+m+1].ravel()
                        for i in range(m, len(clip)-m)], axis=0)

        if pts.shape[0] == 0:
            loops.append(None)
            continue

        # find cycle indices
        cycle_inds = _get_representative_cycle_ripser(pts)
        if not cycle_inds:
            loops.append(None)
            continue

        # average windows at those indices
        avg_seq = _unstack_and_average(pts, cycle_inds, m, D)
        if avg_seq.shape[0] < 2:
            loops.append(None)
            continue

        # reorder into a smooth loop
        loop = _reorder_via_tsp(avg_seq)
        loops.append(loop)
        loops_idx.append(cycle_inds)
        all_pts.append(pts)

    return loops, loops_idx, all_pts


def extract_generative_loops_cocycles_power_based(intentions, m=5, max_clips=None, k=10):
    """
    we extracts a representative H1 loop and find its coordinate in the original intention space.
        - examined via power approximations with KNN density metric
        - rest the same with no KNN weighting
    """
    D  = intentions.shape[2]
    n  = len(intentions) if max_clips is None else min(len(intentions), max_clips)

    loops, loops_idx, all_pts = [], [], []

    for clip_id in range(n):
        clip = intentions[clip_id]

        # build (2m+1) number of windowed points
        pts = np.stack([clip[i-m : i+m+1].ravel()
                        for i in range(m, len(clip)-m)], axis=0)
        if pts.size == 0:
            loops.append(None); continue

        cycle_inds = _weighted_rips_cocycle_ripser(pts, k=k)
        if not cycle_inds:
            loops.append(None); continue

        avg_seq = _unstack_and_average(pts, cycle_inds, m, D)
        if avg_seq.shape[0] < 2:
            loops.append(None); continue

        # loop = _reorder_via_tsp(avg_seq)
        # loops.append(loop)
        
        loops_idx.append(cycle_inds)
        all_pts.append(pts)

    return loops, loops_idx, all_pts