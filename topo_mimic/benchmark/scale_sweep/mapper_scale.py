"""Mapper scale sweep: lens x n_intervals x feature_space.

45 variants total (3 lens x 5 n_intervals x 3 feature_spaces). Each variant:
  1. Applies `apply_dm_key_transform(raw_qpos_or_intention, feature_space)` to get the
     per-feature-space signal.
  2. Takens-embeds each clip's trajectory (per-feature-space m, tau).
  3. Concatenates embedded points across clips, computes the lens.
  4. Builds a Mapper graph (tdamapper CubicalCover + DBSCAN).
  5. Assigns each frame to a single node via smallest-id rule.
  6. Remaps internal enumerate indices -> absolute clip_ids and emits cluster_dict.
  7. Prunes nodes with < min_total_frames, writes graph.pkl + cluster_dict.pkl + metrics.

The feature_space -> input-data map is:
    intention -> ScaleSweepData.intention
    qpos      -> ScaleSweepData.qpos
    ekp       -> ScaleSweepData.qpos  (apply_dm_key_transform handles the FK + egocentric rotate)
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from topo_mimic.benchmark.mapper.mapper_method import (
    _assign_frame_to_single_node,
    _takens_embedding,
)
from topo_mimic.benchmark.scale_sweep.config import (
    MapperCfg,
    MapperLensSpec,
    ScaleSweepConfig,
)
from topo_mimic.benchmark.scale_sweep.data import ScaleSweepData
from topo_mimic.benchmark.scale_sweep.lenses import compute_lens
from topo_mimic.dmwk.utils.feature_transforms import apply_dm_key_transform

log = logging.getLogger(__name__)


@dataclass
class MapperVariantResult:
    variant_id: str
    feature_space: str
    lens: str
    n_intervals: int
    n_nodes: int
    n_edges: int
    n_clusters: int
    n_noise_frames: int
    dbscan_eps: float
    cluster_dict_path: str
    metrics_path: str

    @property
    def column_name(self) -> str:
        return self.variant_id


def _variant_id(feature_space: str, lens: str, n_intervals: int) -> str:
    return f"mapper_{feature_space}_{lens}_n{n_intervals}"


def _select_source_data(data: ScaleSweepData, feature_space: str) -> np.ndarray:
    """Returns the raw array to feed apply_dm_key_transform."""
    if feature_space == "intention":
        return data.intention
    if feature_space in ("qpos", "ekp"):
        return data.qpos
    raise ValueError(f"Unknown feature_space {feature_space!r}")


def _preprocess(data: ScaleSweepData, feature_space: str, cfg: ScaleSweepConfig) -> np.ndarray:
    raw = _select_source_data(data, feature_space)
    return apply_dm_key_transform(
        raw,
        dm_key=feature_space,
        xml_path=cfg.data.xml_path,
        stac_data_path=cfg.data.stac_data_path,
        dataset=cfg.data.dataset,
    )


def _auto_dbscan_eps(points: np.ndarray, seed: int = 0) -> float:
    from sklearn.metrics import pairwise_distances

    rng = np.random.default_rng(seed)
    n_sub = min(1000, len(points))
    sub = points[rng.choice(len(points), size=n_sub, replace=False)]
    d = pairwise_distances(sub)
    return float(np.median(d[d > 0]))


def _mapper_fit_diagnostics(
    graph: Any,
    lens_values: np.ndarray,
    assignments: np.ndarray,
    n_points: int,
) -> dict[str, Any]:
    """Per-variant fit-quality diagnostics for Mapper graphs."""
    import networkx as nx

    nodes_view = graph.nodes(data=True)
    node_sizes = []
    for _, attrs in nodes_view:
        ids = attrs.get("ids", [])
        node_sizes.append(len(ids))
    node_sizes_arr = np.asarray(node_sizes, dtype=float) if node_sizes else np.zeros(1)

    membership_counts = np.zeros(n_points, dtype=int)
    for _, attrs in nodes_view:
        for pt in attrs.get("ids", []):
            membership_counts[int(pt)] += 1
    cov_dist = np.bincount(membership_counts, minlength=4)[:4].tolist()

    nx_graph = graph.to_undirected() if graph.is_directed() else graph
    n_components = nx.number_connected_components(nx_graph)
    biggest = max(nx.connected_components(nx_graph), key=len, default=set())

    n_unassigned_frames = int(np.sum(assignments == -1))

    return {
        "n_connected_components": int(n_components),
        "biggest_component_n_nodes": int(len(biggest)),
        "node_size_min": int(node_sizes_arr.min()),
        "node_size_max": int(node_sizes_arr.max()),
        "node_size_median": float(np.median(node_sizes_arr)),
        "node_size_p25": float(np.quantile(node_sizes_arr, 0.25)),
        "node_size_p75": float(np.quantile(node_sizes_arr, 0.75)),
        "coverage_in_zero_nodes": int(cov_dist[0]),
        "coverage_in_one_node": int(cov_dist[1]),
        "coverage_in_two_nodes": int(cov_dist[2]),
        "coverage_in_three_plus_nodes": int(np.sum(membership_counts >= 3)),
        "lens_p10": float(np.quantile(lens_values, 0.10)),
        "lens_p50": float(np.quantile(lens_values, 0.50)),
        "lens_p90": float(np.quantile(lens_values, 0.90)),
        "n_unassigned_frames_post_partition": n_unassigned_frames,
    }


def _build_cluster_dict(
    assignments: np.ndarray,
    clip_start_offsets: list[int],
    idxs_per_clip: list[np.ndarray],
    clip_ids: list[int],
    n_frames_per_clip: int,
    min_total_frames: int,
) -> tuple[dict[int, list[tuple[int, np.ndarray]]], list[np.ndarray], dict[str, int]]:
    """Remap enumerate index -> absolute clip_id; prune small nodes."""
    labels_list: list[np.ndarray] = []
    for i, cid in enumerate(clip_ids):
        per_clip = np.full(n_frames_per_clip, -1, dtype=np.int64)
        start = clip_start_offsets[i]
        end = clip_start_offsets[i + 1]
        per_clip_point_node = assignments[start:end]
        per_clip_orig_frames = idxs_per_clip[i]
        per_clip[per_clip_orig_frames] = per_clip_point_node
        labels_list.append(per_clip)

    node_totals: dict[int, int] = {}
    for per_clip in labels_list:
        uniques, counts = np.unique(per_clip, return_counts=True)
        for u, c in zip(uniques, counts):
            if int(u) == -1:
                continue
            node_totals[int(u)] = node_totals.get(int(u), 0) + int(c)

    kept_nodes = {nid for nid, total in node_totals.items() if total >= min_total_frames}
    n_noise_frames = sum(int(np.sum(per_clip == -1)) for per_clip in labels_list)
    n_noise_frames += sum(
        int(total) for nid, total in node_totals.items() if nid not in kept_nodes
    )

    cluster_dict: dict[int, list[tuple[int, np.ndarray]]] = {}
    for nid in sorted(kept_nodes):
        members: list[tuple[int, np.ndarray]] = []
        for i, per_clip in enumerate(labels_list):
            frames = np.where(per_clip == nid)[0].astype(np.int64)
            if frames.size == 0:
                continue
            members.append((clip_ids[i], frames))
        if members:
            cluster_dict[nid] = members

    summary = {
        "n_clusters": len(cluster_dict),
        "n_noise_frames": int(n_noise_frames),
        "n_pruned_nodes": int(len(node_totals) - len(kept_nodes)),
    }
    return cluster_dict, labels_list, summary


def run_mapper_variant(
    data: np.ndarray,
    feature_space: str,
    lens_spec: MapperLensSpec,
    n_intervals: int,
    cfg: MapperCfg,
    clip_ids: list[int],
    variant_out_dir: Path,
) -> MapperVariantResult:
    """Fit one (lens, n_intervals, feature_space) Mapper variant."""
    from sklearn.cluster import DBSCAN
    from tdamapper.cover import CubicalCover
    from tdamapper.learn import MapperAlgorithm

    variant_out_dir.mkdir(parents=True, exist_ok=True)
    variant_id = _variant_id(feature_space, lens_spec.name, n_intervals)

    takens = cfg.takens_per_feature_space[feature_space]
    n_clips, n_frames, _ = data.shape

    pts_list: list[np.ndarray] = []
    idxs_list: list[np.ndarray] = []
    clip_start_offsets = [0]
    for i in range(n_clips):
        pts, idxs = _takens_embedding(data[i], m=takens.m, tau=takens.tau)
        pts_list.append(pts)
        idxs_list.append(idxs)
        clip_start_offsets.append(clip_start_offsets[-1] + len(pts))
    all_pts = np.concatenate(pts_list, axis=0)

    lens_values = compute_lens(lens_spec.name, all_pts, lens_spec.params)
    np.save(variant_out_dir / "lens_values.npy", lens_values)

    eps_cfg = cfg.dbscan_eps
    eps = _auto_dbscan_eps(all_pts, seed=0) if eps_cfg == "auto" else float(eps_cfg)

    cover = CubicalCover(n_intervals=n_intervals, overlap_frac=cfg.overlap_frac)
    inner = DBSCAN(eps=eps, min_samples=cfg.dbscan_min_samples)
    mapper = MapperAlgorithm(cover=cover, clustering=inner)
    graph = mapper.fit_transform(all_pts, lens_values)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    assignments = _assign_frame_to_single_node(graph, lens_values, n_points=len(all_pts))

    cluster_dict, labels_list, summary = _build_cluster_dict(
        assignments=assignments,
        clip_start_offsets=clip_start_offsets,
        idxs_per_clip=idxs_list,
        clip_ids=clip_ids,
        n_frames_per_clip=n_frames,
        min_total_frames=cfg.min_total_frames,
    )

    cluster_dict_path = variant_out_dir / "cluster_dict.pkl"
    with cluster_dict_path.open("wb") as f:
        pickle.dump(cluster_dict, f)
    with (variant_out_dir / "labels_list.pkl").open("wb") as f:
        pickle.dump(labels_list, f)

    diag = _mapper_fit_diagnostics(
        graph=graph,
        lens_values=lens_values,
        assignments=assignments,
        n_points=len(all_pts),
    )

    metrics = {
        "variant_id": variant_id,
        "feature_space": feature_space,
        "lens": lens_spec.name,
        "lens_params": lens_spec.params,
        "n_intervals": n_intervals,
        "overlap_frac": cfg.overlap_frac,
        "dbscan_eps": float(eps),
        "dbscan_min_samples": cfg.dbscan_min_samples,
        "takens_m": takens.m,
        "takens_tau": takens.tau,
        "n_points": int(len(all_pts)),
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "n_clusters": int(summary["n_clusters"]),
        "n_pruned_nodes": int(summary["n_pruned_nodes"]),
        "n_noise_frames": int(summary["n_noise_frames"]),
        "lens_min": float(lens_values.min()),
        "lens_max": float(lens_values.max()),
        "lens_std": float(lens_values.std()),
        **diag,
    }
    metrics_path = variant_out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    graph_path = variant_out_dir / "graph.pkl"
    with graph_path.open("wb") as f:
        pickle.dump({"nodes": list(graph.nodes(data=True)), "edges": list(graph.edges())}, f)

    return MapperVariantResult(
        variant_id=variant_id,
        feature_space=feature_space,
        lens=lens_spec.name,
        n_intervals=n_intervals,
        n_nodes=int(n_nodes),
        n_edges=int(n_edges),
        n_clusters=int(summary["n_clusters"]),
        n_noise_frames=int(summary["n_noise_frames"]),
        dbscan_eps=float(eps),
        cluster_dict_path=str(cluster_dict_path),
        metrics_path=str(metrics_path),
    )


def run_mapper_sweep(
    data: ScaleSweepData,
    cfg: ScaleSweepConfig,
    output_root: Path,
    force: bool = False,
) -> list[MapperVariantResult]:
    """Iterate all 45 variants."""
    results: list[MapperVariantResult] = []
    preprocessed_cache: dict[str, np.ndarray] = {}

    for feature_space in cfg.mapper.feature_spaces:
        if feature_space not in preprocessed_cache:
            log.info(f"[Mapper] Preprocessing feature_space={feature_space}")
            preprocessed_cache[feature_space] = _preprocess(data, feature_space, cfg)
        fs_data = preprocessed_cache[feature_space]

        for lens_spec in cfg.mapper.lens_grid:
            for n_intervals in cfg.mapper.n_intervals_grid:
                variant_id = _variant_id(feature_space, lens_spec.name, n_intervals)
                variant_dir = output_root / feature_space / f"lens_{lens_spec.name}" / f"n_{n_intervals}"
                metrics_path = variant_dir / "metrics.json"
                cluster_dict_path = variant_dir / "cluster_dict.pkl"
                labels_list_path = variant_dir / "labels_list.pkl"
                if not force and metrics_path.exists() and cluster_dict_path.exists() and labels_list_path.exists():
                    log.info(f"[Mapper] skip {variant_id} (complete outputs exist)")
                    with metrics_path.open() as f:
                        m = json.load(f)
                    results.append(MapperVariantResult(
                        variant_id=variant_id,
                        feature_space=feature_space,
                        lens=lens_spec.name,
                        n_intervals=n_intervals,
                        n_nodes=int(m["n_nodes"]),
                        n_edges=int(m["n_edges"]),
                        n_clusters=int(m["n_clusters"]),
                        n_noise_frames=int(m["n_noise_frames"]),
                        dbscan_eps=float(m["dbscan_eps"]),
                        cluster_dict_path=str(cluster_dict_path),
                        metrics_path=str(metrics_path),
                    ))
                    continue

                log.info(f"[Mapper] fitting {variant_id}")
                r = run_mapper_variant(
                    data=fs_data,
                    feature_space=feature_space,
                    lens_spec=lens_spec,
                    n_intervals=n_intervals,
                    cfg=cfg.mapper,
                    clip_ids=data.clip_ids,
                    variant_out_dir=variant_dir,
                )
                log.info(
                    f"[Mapper] {variant_id}: nodes={r.n_nodes} clusters={r.n_clusters} "
                    f"noise={r.n_noise_frames}"
                )
                results.append(r)

    return results
