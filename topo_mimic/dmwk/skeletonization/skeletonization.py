'''Functions for doing skeletonization of the intention space via discrete Morse graph'''

import numpy as np
import networkx as nx
import os
import tempfile


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

    if len(G.edges()) > 0:
        try:
            spanning_tree = nx.minimum_spanning_tree(G)

            for u, v in G.edges():
                if not spanning_tree.has_edge(u, v):
                    try:
                        path = nx.shortest_path(spanning_tree, u, v)
                        cycle = path + [path[0]]
                        cycles.append(np.array(cycle, dtype=int))
                    except nx.NetworkXNoPath:
                        continue
        except nx.NetworkXException:
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

    orig_cwd = os.getcwd()
    repo_path = "/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM"
    os.chdir(repo_path)
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

        pcd.build_sparse_weighted_rips_filtration(
            input_filename,
            output_dir,
            k=k,
            metric='euclidean',
            epsilon=epsilon
        )

        filtration_filename = os.path.join(output_dir, 'sparse_weighted_rips_filtration.txt')
        pcd.compute_persistence_swr(filtration_filename, output_dir)

        edge_filename = os.path.join(output_dir, "edge_for_morse_only.txt")
        sorted_weights_filename = os.path.join(output_dir, "sorted-weights.txt")
        pcd.reorder_weights(weights_filename, sorted_weights_filename)

        morse_dir = os.path.join(output_dir, str(pers_delta) + '/')
        os.makedirs(morse_dir, exist_ok=True)
        pcd.compute_graph_reconstruction(
            sorted_weights_filename,
            edge_filename,
            pers_delta,
            morse_dir
        )

        result_edge_filename = os.path.join(morse_dir, 'dimo_edge.txt')

        G = nx.Graph()
        G.add_nodes_from(range(len(points)))

        if os.path.exists(result_edge_filename):
            edges = np.loadtxt(result_edge_filename, dtype=int)
            if edges.size > 0:
                if edges.ndim == 1:
                    edges = edges.reshape(1, -1)
                for edge in edges:
                    if len(edge) >= 2:
                        G.add_edge(edge[0], edge[1])

        cycles = extract_cycles_from_dm_graph(G, points)

        return cycles, G

    finally:
        os.chdir(orig_cwd)
        if should_cleanup and output_dir == temp_dir:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

