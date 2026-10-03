"""
Mapper baseline for the benchmark: each node on the Mapper graph is its own cluster.

Mapper is a topological-clustering tool that operates on a point cloud with a
"lens" (filter function). It partitions the lens range into overlapping
intervals, applies inner clustering within each interval, and emits nodes
(lens-interval, inner-cluster) pairs connected by overlap edges.

This implementation:
  - Takens-embeds per-clip trajectories with the same (m, tau) used by DM-G.
  - Uses the inverse-density score rho(v) = RMS k-NN distance as the 1D lens,
    matching the DM-G density weighting for apples-to-apples comparison.
  - Runs tda-mapper (`tdamapper.learn.MapperAlgorithm`) with CubicalCover and
    DBSCAN inner clustering.
  - Assigns each frame to at most one node via a deepest-bin rule: the node
    whose lens-bin centre is closest to the frame's lens value, among nodes
    that contain the frame. Ties broken by smallest node id.
  - Emits the standard {cluster_id -> [(clip_id, frame_indices), ...]}
    cluster_dict contract used by KPMS/VAE methods.

Each MapperMethod variant is parameterized by a DM-key ("intention", "qpos",
"ekp"); the preprocessing (root strip, PCA, Takens params) mirrors the DM
pipeline config for that feature space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from topo_mimic.experiment.methods import (
    BaseMethod,
    KPMSResult,
    MethodType,
    extract_sustained_transitions,
    print_transition_stats,
)


def _takens_embedding(X: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Takens delay embedding, same convention as dm_pipeline.takens_embedding."""
    n_frames, D = X.shape
    start = (m - 1) * tau
    idxs = np.arange(start, n_frames)
    pts = np.stack(
        [X[[j - a * tau for a in range(m)]].reshape(-1) for j in idxs],
        axis=0,
    )
    return pts, idxs  # idxs[i] gives the original-trajectory frame that pts[i] corresponds to


def _knn_inverse_density(points: np.ndarray, k: int) -> np.ndarray:
    """RMS k-NN distance per point; matches `rho(v_i)` from DM-G."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(points)))
    nn.fit(points)
    distances, _ = nn.kneighbors(points)
    neigh_dist = distances[:, 1:]
    return np.sqrt(np.mean(neigh_dist ** 2, axis=1))


def _canonical_sign(v: np.ndarray) -> np.ndarray:
    """Pin the sign of an eigenvector deterministically.

    Eigensolvers return +/-v arbitrarily. This is not cosmetic here: CubicalCover
    enumerates intervals in increasing lens order, so node ids are monotone in bin
    index, and `_assign_frame_to_single_node` breaks ties toward the smallest id,
    sending every overlap point to the lower-lens bin. Flipping the sign mirrors
    the lens and re-partitions every overlap region, silently changing all cluster
    ids between two runs of an identical config.
    """
    s = float(np.sum(v ** 3))
    if s == 0.0 or not np.isfinite(s):
        s = float(v[int(np.argmax(np.abs(v)))])
    if s == 0.0:
        nz = np.flatnonzero(v)
        s = float(v[nz[0]]) if nz.size else 1.0
    return v if s >= 0 else -v


def _fiedler_lens(
    points: np.ndarray,
    k: int = 15,
    max_k: int = 100,
    rank_normalize: bool = False,
    random_state: int = 0,
    min_giant_frac: float = 0.90,
) -> np.ndarray:
    """First non-trivial eigenvector of the normalized Laplacian of a k-NN graph.

    A disconnected graph is the dominant silent-failure mode: `spectral_embedding`
    does not raise on one, it returns an eigenvector of eigenvalue 0, i.e. a
    component indicator, which is finite and non-constant and so passes every
    validator downstream while being piecewise constant. We therefore grow k until
    the graph is connected, and only then fall back to the giant component.

    Unweighted connectivity is used rather than distance-weighted edges because the
    qpos/ekp spaces have heterogeneous per-dimension scales, so one global Gaussian
    bandwidth would be dominated by the densest region.
    """
    import warnings

    from scipy.sparse.csgraph import connected_components
    from sklearn.manifold import spectral_embedding
    from sklearn.neighbors import NearestNeighbors, kneighbors_graph

    n = len(points)
    k_used = int(min(k, max(1, n - 1)))
    k_cap = int(min(max_k, n - 1))
    while True:
        adj = kneighbors_graph(points, k_used, mode="connectivity", include_self=False)
        # maximum(), not 0.5*(A+A.T): the latter half-weights one-directional edges.
        adj = adj.maximum(adj.T)
        n_comp, comp_labels = connected_components(adj, directed=False)
        if n_comp == 1 or k_used >= k_cap:
            break
        k_used = int(min(k_used * 2, k_cap))

    if n_comp == 1:
        keep = np.arange(n)
        adj_sub = adj
    else:
        giant = int(np.argmax(np.bincount(comp_labels)))
        keep = np.flatnonzero(comp_labels == giant)
        frac = len(keep) / n
        if frac < min_giant_frac:
            raise RuntimeError(
                f"Fiedler lens: graph still has {n_comp} components at k={k_used} and the "
                f"giant one holds only {frac:.1%} of points; refusing to invent lens values."
            )
        adj_sub = adj[keep][:, keep]

    vec, failures = None, []
    for solver in ("amg", "arpack", "lobpcg"):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "error", message=".*not fully connected.*", category=UserWarning
                )
                vec = spectral_embedding(
                    adj_sub, n_components=1, drop_first=True, norm_laplacian=True,
                    eigen_solver=solver, random_state=random_state,
                ).ravel()
            break
        except Exception as exc:  # noqa: BLE001 - fall through to the next solver
            failures.append(f"{solver}: {type(exc).__name__}: {exc}")
            vec = None
    if vec is None:
        raise RuntimeError("Fiedler lens: every eigensolver failed -> " + " | ".join(failures))

    out = np.empty(n, dtype=float)
    out[keep] = vec
    if len(keep) < n:
        rest = np.setdiff1d(np.arange(n), keep)
        nn = NearestNeighbors(n_neighbors=1).fit(points[keep])
        out[rest] = vec[nn.kneighbors(points[rest])[1][:, 0]]

    out = _canonical_sign(out)
    if out.std() > 0:
        out = out / out.std()
    if rank_normalize:
        out = np.argsort(np.argsort(out)) / max(1, n - 1)

    print(
        f"[fiedler] n={n} k_used={k_used} components={n_comp} "
        f"giant_frac={len(keep)/n:.3f} rank_norm={rank_normalize}"
    )
    return out


def _assign_frame_to_single_node(
    graph: Any,
    lens_values: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """
    Deterministic frame-to-node partition for an overlapping Mapper graph.

    Rule: each point is assigned to the smallest-id node it belongs to, among
    all nodes that include it in their `ids`. Points in no node get -1.

    Args:
        graph: networkx graph produced by tda-mapper. Each node has an `ids`
            attribute listing the indices of the input points inside it.
        lens_values: (n_points,) lens values; reserved for future bin-distance
            tie-breaking. Unused in the current smallest-id rule but kept on
            the signature for forward-compatibility.
        n_points: total number of input points.

    Returns:
        assignments: (n_points,) int array; -1 means unassigned (noise).
    """
    del lens_values
    assignments = np.full(n_points, -1, dtype=np.int64)
    for node_id in sorted(graph.nodes):
        ids = graph.nodes[node_id].get("ids", [])
        for pt_id in ids:
            if assignments[pt_id] == -1:
                assignments[pt_id] = int(node_id)
    return assignments


class _AdaptiveDBSCAN:
    """DBSCAN whose radius tracks the local scale of each cover element.

    tda-mapper applies one clustering object to every cover element. Because the
    lens here is an inverse-density score, those elements are density strata, so
    a single global eps is far too large in the dense strata (they collapse to
    one node) and too small in the sparse ones (they become all noise). This
    derives eps from each element's own k-NN distances instead.

    Must never raise: tda-mapper's FailSafeClustering turns an exception into a
    single all-points cluster without surfacing it.
    """

    def __init__(self, scale: float, k: int = 3, min_samples: int = 5):
        self.scale = float(scale)
        self.k = int(k)
        self.min_samples = int(min_samples)
        self.labels_ = None
        self.eps_ = None

    def fit(self, X, y=None):
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors

        X = np.asarray(X)
        n = len(X)
        if n < self.min_samples:
            self.labels_ = np.full(n, -1, dtype=np.int64)
            return self

        nn = NearestNeighbors(n_neighbors=min(self.k + 1, n)).fit(X)
        dists, _ = nn.kneighbors(X)
        local = float(np.median(np.sqrt(np.mean(dists[:, 1:] ** 2, axis=1))))
        if not np.isfinite(local) or local <= 0:
            # degenerate element (duplicate points); one cluster is the sane read
            self.labels_ = np.zeros(n, dtype=np.int64)
            return self

        self.eps_ = self.scale * local
        self.labels_ = DBSCAN(
            eps=self.eps_, min_samples=self.min_samples
        ).fit_predict(X)
        return self


class MapperMethod(BaseMethod):
    """
    Mapper-based benchmark baseline. Each node on the Mapper graph becomes its
    own cluster; frames are partitioned across nodes by a deepest-bin rule.

    Parameters (via self.params):
        takens_m, takens_tau (int): Takens delay embedding. Defaults match the
            DM pipeline for each feature space.
        knn_k (int): neighbors for the inverse-density lens (default 3, matches DM-G).
        n_intervals (int): Mapper cover intervals (default 10).
        overlap_frac (float): Mapper cover overlap fraction (default 0.5).
        dbscan_eps (float): DBSCAN eps for inner clustering (default "auto",
            set to median pairwise distance in each bin).
        dbscan_min_samples (int): DBSCAN min_samples (default 5).
        min_total_frames (int): drop mapper nodes whose total frames across
            clips is below this threshold (default 0).
    """

    def __init__(self, name: str, dm_key: str, params: dict | None = None):
        super().__init__(name, params)
        self.dm_key = dm_key
        self.takens_m = int(self.params.get("takens_m", 3))
        self.takens_tau = int(self.params.get("takens_tau", 6))
        self.knn_k = int(self.params.get("knn_k", 3))
        # Lens is selectable so the historical inverse-density runs stay reproducible.
        # Default stays "knn_density" so untouched configs are byte-identical; the
        # shipped Mapper configs select "fiedler" explicitly.
        self.lens_name = str(self.params.get("lens", "knn_density"))
        self.lens_params = dict(self.params.get("lens_params", {}))
        if self.lens_name == "knn_density":
            self.lens_params.setdefault("knn_k", self.knn_k)
        self.n_intervals = int(self.params.get("n_intervals", 10))
        self.overlap_frac = float(self.params.get("overlap_frac", 0.5))
        self.dbscan_eps = self.params.get("dbscan_eps", "auto")
        self.dbscan_min_samples = int(self.params.get("dbscan_min_samples", 5))
        # "global" reproduces the published runs (one eps for every cover
        # element). "per_bin" scales eps to each element's own k-NN distance,
        # which the inverse-density lens makes necessary; see _AdaptiveDBSCAN.
        self.eps_mode = str(self.params.get("eps_mode", "global"))
        self.eps_scale = float(self.params.get("eps_scale", 3.0))
        self.min_total_frames = int(self.params.get("min_total_frames", 0))
        # When the concatenated Takens-embedded point cloud has more points
        # than this cap, build the Mapper graph on a random subsample of size
        # `max_points` and propagate node assignments to the rest via 1-NN in
        # the subsample. Without this, tdamapper's fit_transform hangs for
        # >~300k points.
        self.max_points = int(self.params.get("max_points", 100000))

    @property
    def method_type(self) -> MethodType:
        return MethodType.STATE_SPACE

    def fit_predict(self, X: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
        raise NotImplementedError("Use fit() for Mapper methods.")

    def fit(
        self,
        data: np.ndarray,
        n_states: int,
        project_dir: str,
        seed: int = 0,
    ) -> KPMSResult:
        """
        Build Mapper graph over all clips' Takens-embedded points and partition
        frames into nodes-as-clusters.

        Args:
            data: (n_clips, n_frames, n_dim) after DM-key transform (root strip +
                PCA applied upstream by the caller).
            n_states: ignored; Mapper chooses node count via hyperparameters.
            project_dir: unused, accepted for KPMS-signature parity.
            seed: DBSCAN random_state.

        Returns:
            KPMSResult with:
                labels: list of (n_frames,) int arrays, per-frame node id
                    (-1 for noise) per clip, in original-frame indexing.
                transition_cluster_dict: {cluster_id -> [(clip_id, frame_indices)]}
                transition_stats: summary from extract_sustained_transitions.
        """
        from tdamapper.learn import MapperAlgorithm
        from tdamapper.cover import CubicalCover
        from sklearn.cluster import DBSCAN

        del project_dir  # unused

        n_clips = data.shape[0]
        pts_list = []
        idxs_list = []
        clip_start_offsets = [0]
        for clip_id in range(n_clips):
            clip = data[clip_id]
            pts, idxs = _takens_embedding(clip, m=self.takens_m, tau=self.takens_tau)
            pts_list.append(pts)
            idxs_list.append(idxs)
            clip_start_offsets.append(clip_start_offsets[-1] + len(pts))

        all_pts = np.concatenate(pts_list, axis=0)
        print(f"[Mapper/{self.dm_key}] embedded points: {all_pts.shape}")

        # Subsample for graph fit if the full point cloud is too large for
        # tdamapper. Assignments for non-subsample points are propagated via
        # 1-NN in the subsample at the end. This must happen BEFORE the lens is
        # computed, because a graph-scoped lens has to be built on exactly the
        # set Mapper will cover.
        if len(all_pts) > self.max_points:
            rng_sub = np.random.default_rng(seed + 991)
            sub_idx = rng_sub.choice(len(all_pts), size=self.max_points, replace=False)
            sub_pts = all_pts[sub_idx]
            print(f"[Mapper/{self.dm_key}] subsampling {len(all_pts)} -> {self.max_points} points for graph fit")
        else:
            sub_idx = np.arange(len(all_pts))
            sub_pts = all_pts

        from topo_mimic.analysis.scale_sweep.lenses import LENS_SCOPE, compute_lens

        if LENS_SCOPE.get(self.lens_name, "pointwise") == "graph":
            # The Fiedler vector of the full cloud restricted to a random subsample
            # is not the Fiedler vector of that subsample, so compute it directly.
            sub_lens = compute_lens(self.lens_name, sub_pts, self.lens_params)
        else:
            sub_lens = compute_lens(self.lens_name, all_pts, self.lens_params)[sub_idx]
        print(
            f"[Mapper/{self.dm_key}] lens={self.lens_name} "
            f"range: [{sub_lens.min():.4f}, {sub_lens.max():.4f}]"
        )

        cover = CubicalCover(n_intervals=self.n_intervals, overlap_frac=self.overlap_frac)
        eps = self.dbscan_eps
        if self.eps_mode == "per_bin":
            inner = _AdaptiveDBSCAN(
                scale=self.eps_scale, k=self.knn_k,
                min_samples=self.dbscan_min_samples,
            )
            eps = float("nan")  # set per cover element, not globally
            print(f"[Mapper/{self.dm_key}] per-bin eps, scale={self.eps_scale}")
        else:
            if eps == "auto":
                rng = np.random.default_rng(seed)
                sub_n = min(1000, len(sub_pts))
                sub = sub_pts[rng.choice(len(sub_pts), size=sub_n, replace=False)]
                from sklearn.metrics import pairwise_distances
                d = pairwise_distances(sub)
                eps = float(np.median(d[d > 0]))
                print(f"[Mapper/{self.dm_key}] auto dbscan_eps = {eps:.4f}")
            inner = DBSCAN(eps=float(eps), min_samples=self.dbscan_min_samples)
        mapper = MapperAlgorithm(cover=cover, clustering=inner)
        graph = mapper.fit_transform(sub_pts, sub_lens)
        print(f"[Mapper/{self.dm_key}] mapper graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        # Sub-assignment on the subsample (graph ids reference sub_pts indices).
        sub_assign = _assign_frame_to_single_node(graph, sub_lens, n_points=len(sub_pts))

        if len(sub_pts) < len(all_pts):
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1).fit(sub_pts)
            _, nearest = nn.kneighbors(all_pts)
            assignments = sub_assign[nearest[:, 0]]
            print(f"[Mapper/{self.dm_key}] propagated {len(all_pts) - len(sub_pts)} non-subsample points via 1-NN")
        else:
            assignments = sub_assign
        n_noise = int(np.sum(assignments == -1))
        print(f"[Mapper/{self.dm_key}] noise frames (unassigned): {n_noise}/{len(assignments)}")

        labels_list: list[np.ndarray] = []
        for clip_id in range(n_clips):
            n_frames_clip = data.shape[1]
            per_clip_labels = np.full(n_frames_clip, -1, dtype=np.int64)
            start = clip_start_offsets[clip_id]
            end = clip_start_offsets[clip_id + 1]
            per_clip_point_node = assignments[start:end]
            per_clip_orig_frames = idxs_list[clip_id]
            per_clip_labels[per_clip_orig_frames] = per_clip_point_node
            labels_list.append(per_clip_labels)

        # Reverted to transition-based clustering: each cluster is a (from_node ->
        # to_node) sustained transition pair, not a single Mapper node. The -1 noise
        # label is handled inside extract_sustained_transitions (transitions
        # involving -1 are dropped).
        cluster_dict, stats = extract_sustained_transitions(
            labels_list,
            min_duration=int(self.params.get("min_sustained_duration", 50)),
            top_k=int(self.params.get("top_k_transitions", 10)),
            min_instances=int(self.params.get("min_transition_instances", 3)),
        )
        stats["n_noise_frames"] = int(n_noise)
        stats["mapper_n_nodes"] = int(graph.number_of_nodes())
        stats["mapper_n_edges"] = int(graph.number_of_edges())
        stats["mapper_dbscan_eps"] = float(eps)
        stats["mapper_eps_mode"] = self.eps_mode
        stats["mapper_eps_scale"] = self.eps_scale
        stats["dm_key"] = self.dm_key

        print_transition_stats(stats)

        return KPMSResult(
            labels=labels_list,
            n_states=len(cluster_dict),
            method_name=self.name,
            transition_cluster_dict=cluster_dict,
            transition_stats=stats,
        )


class MapperIntMethod(MapperMethod):
    def __init__(self, params: dict | None = None):
        p = dict(params or {})
        p.setdefault("takens_m", 3)
        p.setdefault("takens_tau", 6)
        super().__init__(name="mapper_int", dm_key="intention", params=p)


class MapperQposMethod(MapperMethod):
    def __init__(self, params: dict | None = None):
        p = dict(params or {})
        p.setdefault("takens_m", 3)
        p.setdefault("takens_tau", 9)
        super().__init__(name="mapper_qpos", dm_key="qpos", params=p)


class MapperEKPMethod(MapperMethod):
    def __init__(self, params: dict | None = None):
        p = dict(params or {})
        p.setdefault("takens_m", 3)
        p.setdefault("takens_tau", 8)
        super().__init__(name="mapper_ekp", dm_key="ekp", params=p)
