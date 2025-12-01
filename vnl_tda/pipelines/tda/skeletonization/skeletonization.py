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


def _knn_weights_for_persistence(
    pts: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """
    Compute k-NN based weights for persistence computation.

    Args:
        pts: Point cloud array of shape (N, D)
        k: Number of nearest neighbors

    Returns:
        Weight array of shape (N,) with distance-to-measure values
    """
    k = min(k, pts.shape[0] - 1) 
    D = pairwise_distances(pts) 
    idx = np.argpartition(D, kth=k+1, axis=1)[:, 1:k+1]   # skip self-dist (0)
    w2 = (D[np.arange(len(pts))[:, None], idx] ** 2).mean(axis=1)
    return np.sqrt(w2)


def _weighted_distance_matrix(
    pts: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Compute power-distance matrix for weighted Rips filtration.

    Args:
        pts: Point cloud array of shape (N, D)
        w: Weight array of shape (N,)

    Returns:
        Power-distance matrix of shape (N, N)
    """
    d2 = squareform(pdist(pts, metric='sqeuclidean'))
    W2 = w[:, None] ** 2 + w[None, :] ** 2
    return np.sqrt(d2 + W2)


def _weighted_rips_cocycle_ripser(
    pts: np.ndarray,
    k: int = 10,
) -> list[int]:
    """
    Find the most persistent H1 generator via weighted Rips filtration.

    Args:
        pts: Point cloud array of shape (N, D)
        k: Number of neighbors for weight computation

    Returns:
        List of vertex indices participating in the most persistent cycle
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


def _get_representative_cycle_ripser(
    points: np.ndarray,
    maxdim: int = 1,
) -> list[int]:
    """
    Find the most persistent H1 cycle via Ripser's cocycles.

    Args:
        points: Point cloud array of shape (N, D)
        maxdim: Maximum homology dimension to compute

    Returns:
        List of point indices that participate in the most persistent cycle
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


def _unstack_and_average(
    points: np.ndarray,
    cycle_inds: list[int],
    m: int,
    D: int,
) -> np.ndarray:
    """
    Reshape flattened windows and average over the temporal axis.

    Args:
        points: Flattened window array of shape (N, (2m+1)*D)
        cycle_inds: Indices of cycle points
        m: Half-window size
        D: Feature dimension

    Returns:
        Averaged sequence array of shape (len(cycle_inds), D)
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


def _reorder_via_tsp(
    sequence: np.ndarray,
) -> np.ndarray:
    """
    Reorder cycle points via TSP to form a smooth loop.

    Args:
        sequence: Array of points to reorder, shape (N, D)

    Returns:
        Reordered sequence array
    """
    if len(sequence) <= 1:
        return sequence
    Dmat = cdist(sequence, sequence)
    G = nx.complete_graph(len(sequence))
    for u,v in G.edges():
        G[u][v]['weight'] = Dmat[u,v]
    tour = nx.approximation.traveling_salesman_problem(G, cycle=True)
    return sequence[tour]


def extract_cycles_from_dm_graph(
    G: nx.Graph,
    points: np.ndarray,
) -> list[np.ndarray]:
    """
    Extract cycles from DM graph by finding non-tree edges.

    Args:
        G: NetworkX graph from discrete Morse reconstruction
        points: Point cloud array of shape (N, D)

    Returns:
        List of cycle arrays, each containing vertex indices
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
    points: np.ndarray,
    k: int = 10,
    pers_delta: float = 0.25,
    epsilon: float = 0.99,
    output_dir: str | None = None,
    cleanup: bool = True,
) -> tuple[list[np.ndarray], nx.Graph]:
    """
    Reconstruct discrete Morse graph from point cloud data.

    Args:
        points: Point cloud array of shape (N, D)
        k: Number of neighbors for graph construction
        pers_delta: Persistence threshold for filtering
        epsilon: Sparse Rips approximation parameter
        output_dir: Directory for intermediate files (temp dir if None)
        cleanup: Whether to remove intermediate files

    Returns:
        Tuple of (cycle_indices, DM_graph) where cycle_indices is a list of
        arrays containing vertex indices for each cycle
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


def extract_generative_loops_cocycles_based(
    intentions: np.ndarray,
    m: int = 5,
    max_clips: int | None = None,
) -> tuple[list[np.ndarray | None], list[list[int]], list[np.ndarray]]:
    """
    Extract representative H1 loops from intention trajectories via cocycles.

    Args:
        intentions: Intention array of shape (n_clips, n_frames, D)
        m: Half-window size for Takens embedding
        max_clips: Maximum number of clips to process (None for all)

    Returns:
        Tuple of (loops, loops_idx, all_pts) where loops contains TSP-ordered
        cycle coordinates, loops_idx contains raw cycle indices, and all_pts
        contains the windowed point clouds
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


def extract_generative_loops_cocycles_power_based(
    intentions: np.ndarray,
    m: int = 5,
    max_clips: int | None = None,
    k: int = 10,
) -> tuple[list, list[list[int]], list[np.ndarray]]:
    """
    Extract H1 loops using power-weighted Rips filtration.

    Args:
        intentions: Intention array of shape (n_clips, n_frames, D)
        m: Half-window size for Takens embedding
        max_clips: Maximum number of clips to process (None for all)
        k: Number of neighbors for density weighting

    Returns:
        Tuple of (loops, loops_idx, all_pts) containing cycle data
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