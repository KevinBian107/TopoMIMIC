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


def extract_all_cycles(
    G: nx.Graph,
    vertex_weight: dict[int, float] | None = None,
) -> list[np.ndarray]:
    """All beta1 loops of the DM skeleton as a (filtration-weighted) minimum cycle basis.

    Replaces the old "single longest MST fundamental cycle". Returns one vertex array per
    independent loop (count == beta1). When vertex_weight is given, edges are weighted by the
    max endpoint weight (the lower-star filtration value) so the basis prefers denser loops,
    the practical stand-in for the persistence-optimal basis (Magee-Wang Thm 3.5).
    """
    if G.number_of_edges() == 0:
        return []
    H = G
    weight_key = None
    if vertex_weight is not None:
        H = G.copy()
        for u, v in H.edges():
            H[u][v]["fweight"] = max(
                vertex_weight.get(u, 0.0), vertex_weight.get(v, 0.0)
            ) + 1e-9
        weight_key = "fweight"
    try:
        basis = nx.minimum_cycle_basis(H, weight=weight_key)
    except nx.NetworkXException:
        return []
    return [np.array(c, dtype=int) for c in basis if len(c) > 0]


def _maximal_bridge_chains(Hb: nx.Graph) -> list[list[int]]:
    """Maximal degree!=2 vertex chains (corridors) of a forest Hb (bridge subgraph)."""
    chains: list[list[int]] = []
    seen: set = set()
    junctions = [n for n in Hb if Hb.degree(n) != 2]
    for j in junctions:
        for nb in Hb.neighbors(j):
            if frozenset((j, nb)) in seen:
                continue
            chain, prev, cur = [j], j, nb
            seen.add(frozenset((prev, cur)))
            chain.append(cur)
            while Hb.degree(cur) == 2:
                nxt = [x for x in Hb.neighbors(cur) if x != prev]
                if not nxt or frozenset((cur, nxt[0])) in seen:
                    break
                seen.add(frozenset((cur, nxt[0])))
                chain.append(nxt[0])
                prev, cur = cur, nxt[0]
            chains.append(chain)
    return chains


def extract_all_corridors(G: nx.Graph) -> list[np.ndarray]:
    """All bridge-corridors (paths) of the DM skeleton.

    A corridor is a maximal junction-to-junction chain on the bridge (acyclic) part
    of G. Each fork in the bridge forest is a branch point of the trajectory, where
    a subtle behavior change occurs, so every non-branching corridor between forks
    is kept as its own path primitive (a finer decomposition than mode-to-mode).
    Corridors live only on bridges, so they never lie inside a cycle (disjoint
    typing from DM-cycles).
    """
    if G.number_of_edges() == 0:
        return []
    bridges = list(nx.bridges(G))
    if not bridges:
        return []
    Hb = nx.Graph()
    Hb.add_edges_from(bridges)
    return [np.array(c, dtype=int) for c in _maximal_bridge_chains(Hb) if len(c) > 0]


def _lower_weight_spanning_forest(
    G: nx.Graph, fval
) -> nx.Graph:
    """Kruskal spanning forest of G with edges ordered by saddle level max(f(u), f(v)).

    This is the merge-tree carrier: a non-tree edge closes a cycle (an H1 feature,
    handled elsewhere), so the forest holds only the H0 (component) structure.
    """
    T = nx.Graph()
    T.add_nodes_from(G.nodes())
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    for u, v in sorted(G.edges(), key=lambda e: max(fval(e[0]), fval(e[1]))):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[rv] = ru
            T.add_edge(u, v)
    return T


def extract_persistence_paths(
    G: nx.Graph,
    vertex_weight: dict[int, float],
    persistence_delta: float = 0.0,
) -> list[np.ndarray]:
    """Extended-persistence branch decomposition of the DM skeleton into paths.

    On the lower-weight spanning forest of G, every H0 persistence pair contributes
    one branch: ordinary (min, saddle) branches plus one essential (min, max)
    backbone per connected component. Unlike bridge-corridors, an elder branch flows
    *through* a fork instead of being cut there, so a long trajectory stays one path;
    the cut criterion is filtration persistence (density), not raw graph degree.
    Branches partition the forest edges, so they are edge-disjoint and never lie
    inside a DM-cycle (disjoint typing from DM-cycles).

    Args:
        G: DM skeleton graph; node indices share the indexing of vertex_weight.
        vertex_weight: node -> filtration value f (lower weight = denser basin), so
            minima of f are behavioral modes and saddles are basin boundaries.
        persistence_delta: keep an ordinary branch when f(saddle) - f(min) >= this;
            essential backbones are always kept. 0.0 keeps every branch present in G.

    Returns:
        One vertex-index array per kept branch (the minimum first, the saddle/max last).
    """
    if G.number_of_edges() == 0:
        return []

    def fval(n: int) -> float:
        return vertex_weight.get(n, 0.0)

    T = _lower_weight_spanning_forest(G, fval)

    paths: list[np.ndarray] = []
    for comp in nx.connected_components(T):
        if len(comp) < 2:
            continue
        sub = T.subgraph(comp)
        root = max(comp, key=fval)  # global max of the component

        # Root the forest, record parents in a pre-order (parent before children).
        par: dict[int, int | None] = {root: None}
        order: list[int] = []
        stack = [root]
        visited = {root}
        while stack:
            n = stack.pop()
            order.append(n)
            for nb in sub.neighbors(n):
                if nb not in visited:
                    visited.add(nb)
                    par[nb] = n
                    stack.append(nb)

        children: dict[int, list[int]] = {n: [] for n in comp}
        for n in comp:
            if par[n] is not None:
                children[par[n]].append(n)

        # Deepest (min-f) vertex in each subtree, accumulated children-before-parents.
        deep: dict[int, int] = {n: n for n in comp}
        for n in reversed(order):
            p = par[n]
            if p is not None and fval(deep[n]) < fval(deep[p]):
                deep[p] = deep[n]

        # Each forest edge (n, par[n]) is owned by deep[n]; an owner's owned edges form
        # one contiguous chain from its minimum up to the junction where a deeper minimum
        # takes over (the global minimum owns up to root => the essential min-max path).
        global_min = deep[root]
        owners = {deep[n] for n in comp if par[n] is not None}
        for owner in owners:
            seq = [owner]
            cur = owner
            while par[cur] is not None and deep[cur] == owner:
                cur = par[cur]
                seq.append(cur)
            if len(seq) < 2:
                continue

            if owner == global_min:
                # essential pair: deepest mode up to the global max (sparsest point).
                paths.append(np.array(seq, dtype=int))
                continue

            # Ordinary pair: the feature dies where it merges into the deeper basin
            # deep[junction]. The death level is the minimax barrier (max f along the
            # full route min -> junction -> deeper-min), not the junction's own f, which
            # is non-monotone on the connectivity spine.
            junction = seq[-1]
            deeper = deep[junction]
            route = list(seq)
            node = junction
            while node != deeper:
                nxt = next((c for c in children[node] if deep[c] == deeper), None)
                if nxt is None:
                    break
                route.append(nxt)
                node = nxt
            pers = max(fval(x) for x in route) - fval(owner)
            if pers < persistence_delta:
                continue
            paths.append(np.array(seq, dtype=int))

    return paths


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

    if output_dir is not None:
        output_dir = os.path.abspath(output_dir)

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

