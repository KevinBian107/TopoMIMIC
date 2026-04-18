"""Shared pipeline for the global-manifold transition demo.

Pools clips whose GT labels contain configured transition pairs, runs DM on the
pooled Takens point cloud per feature space, and produces:
  - projection_grid.pdf  : PCA / t-SNE / UMAP overlays with DM primitives
  - filmstrip_grid.pdf   : stitched cross-clip filmstrips per minima pair
"""

from __future__ import annotations

import os
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_DM_PKG = str(REPO_ROOT / "PCD-Graph-Recon-DM")
if _DM_PKG not in sys.path:
    sys.path.append(_DM_PKG)

from topo_mimic.benchmark.analysis.behavior_parsing import (
    PRIMITIVE_BEHAVIORS,
    merge_rear_category,
    normalize_behavior_name,
)
from topo_mimic.benchmark.analysis.ground_truth import (
    BehaviorSegment,
    TurnDirection,
    load_behavior_segments,
)
from topo_mimic.dmwk.skeletonization.dm_path import (
    _find_persistent_minima,
    _parse_persistence_pairs,
    _parse_sorted_weights,
)
from topo_mimic.dmwk.skeletonization.remap import (
    parse_swr_remap,
    remap_cycle_to_original,
)
from topo_mimic.dmwk.skeletonization.skeletonization import (
    extract_cycles_from_dm_graph,
    extract_skeleton_pcd_dm,
)
from topo_mimic.dmwk.utils.dataset_config import get_dataset_preset
from topo_mimic.dmwk.utils.feature_transforms import (
    apply_dm_key_transform,
    get_hdf5_key_for_dm_key,
)
from topo_mimic.dmwk.utils.load_utils import get_aggregate_data


# ───────────────────────────── Config ─────────────────────────────


@dataclass
class TakensConfig:
    auto_estimate: bool = False
    m: int = 3
    tau: int = 6


@dataclass
class FeatureSpaceConfig:
    dm_key: str
    pca_dims: int | None = None
    pers_delta: float = 0.75
    subsample_n: int = 3
    takens: TakensConfig = field(default_factory=TakensConfig)


@dataclass
class DMConfig:
    k_neighbors: int = 3
    epsilon: float = 0.99


@dataclass
class MinimaSelectionConfig:
    primitives: list[str] = field(default_factory=list)
    neighbor_votes: int = 15
    candidate_pool: int = 50


@dataclass
class PathSamplingConfig:
    k_per_pair: int = 5
    simple_path_cap: int = 200
    gap_limit: int = 50
    min_segment_len: int = 10
    rng_seed: int = 0


@dataclass
class ProjectionConfig:
    methods: list[str] = field(default_factory=lambda: ["pca", "tsne", "umap"])
    tsne_perplexity: float = 30.0
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    max_points: int = 20000
    rng_seed: int = 0


@dataclass
class VisualizationConfig:
    dpi: int = 300
    behavior_colors: dict[str, str] = field(default_factory=dict)
    turn_shapes: dict[str, str] = field(default_factory=lambda: {"straight": "o"})
    filmstrip_n_poses: int = 6
    background_alpha: float = 0.25
    background_size: int = 4
    overlay_linewidth: float = 1.4
    minima_marker_size: int = 140


@dataclass
class GlobalManifoldConfig:
    dataset: str
    data_path: str
    labels_csv: str
    transition_types: list[tuple[str, str]]
    feature_spaces: list[FeatureSpaceConfig]
    output_dir: str
    cache_dir: str
    dm: DMConfig = field(default_factory=DMConfig)
    minima_selection: MinimaSelectionConfig = field(default_factory=MinimaSelectionConfig)
    path_sampling: PathSamplingConfig = field(default_factory=PathSamplingConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    xml_path: str = ""
    stac_data_path: str = ""


def load_config(path: str) -> GlobalManifoldConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    def _mk_fs(d: dict) -> FeatureSpaceConfig:
        t = d.get("takens", {})
        return FeatureSpaceConfig(
            dm_key=d["dm_key"],
            pca_dims=d.get("pca_dims"),
            pers_delta=d.get("pers_delta", 0.75),
            subsample_n=d.get("subsample_n", 3),
            takens=TakensConfig(**t) if isinstance(t, dict) else TakensConfig(),
        )

    preset = get_dataset_preset(raw["dataset"])
    cfg = GlobalManifoldConfig(
        dataset=raw["dataset"],
        data_path=raw.get("data_path", preset.default_data_path),
        labels_csv=raw.get("labels_csv", preset.labels_csv),
        transition_types=[tuple(pair) for pair in raw["transition_types"]],
        feature_spaces=[_mk_fs(d) for d in raw["feature_spaces"]],
        output_dir=raw["output_dir"],
        cache_dir=raw.get("cache_dir", os.path.join(raw["output_dir"], "cache")),
        dm=DMConfig(**raw.get("dm", {})),
        minima_selection=MinimaSelectionConfig(**raw.get("minima_selection", {})),
        path_sampling=PathSamplingConfig(**raw.get("path_sampling", {})),
        projection=ProjectionConfig(**raw.get("projection", {})),
        visualization=VisualizationConfig(**raw.get("visualization", {})),
        xml_path=raw.get("xml_path", preset.xml_path),
        stac_data_path=raw.get("stac_data_path", preset.stac_data_path),
    )
    return cfg


# ─────────────────────── GT label lookup ────────────────────────


@dataclass
class FrameLabels:
    primitive_by_frame: dict[int, dict[int, str]]          # clip_id → frame → label
    turn_by_frame: dict[int, dict[int, str]]                # clip_id → frame → turn
    segments_by_clip: dict[int, list[BehaviorSegment]]      # clip_id → sorted segs


def build_frame_labels(
    csv_path: str,
    selected_clips: set[int] | None = None,
) -> FrameLabels:
    segments, turn_overlaps = load_behavior_segments(csv_path, merge_rear=True)
    by_clip: dict[int, list[BehaviorSegment]] = defaultdict(list)
    for s in segments:
        if selected_clips is not None and s.clip_id not in selected_clips:
            continue
        by_clip[s.clip_id].append(s)
    for cid in by_clip:
        by_clip[cid].sort(key=lambda s: s.start_frame)

    prim_by_frame: dict[int, dict[int, str]] = defaultdict(dict)
    turn_by_frame: dict[int, dict[int, str]] = defaultdict(dict)

    for cid, segs in by_clip.items():
        for s in segs:
            for fr in range(s.start_frame, s.end_frame + 1):
                prim_by_frame[cid][fr] = s.behavior

        for turn_start, turn_end, turn_type in turn_overlaps.get(cid, []):
            if turn_type == "Left Turn":
                tag = "left"
            elif turn_type == "Right Turn":
                tag = "right"
            else:
                tag = "straight"
            for fr in range(turn_start, turn_end + 1):
                turn_by_frame[cid][fr] = tag

    return FrameLabels(
        primitive_by_frame=dict(prim_by_frame),
        turn_by_frame=dict(turn_by_frame),
        segments_by_clip=dict(by_clip),
    )


def select_clips_for_transitions(
    csv_path: str,
    transition_types: list[tuple[str, str]],
    max_clips: int | None = None,
) -> set[int]:
    segments, _ = load_behavior_segments(csv_path, merge_rear=True)
    by_clip: dict[int, list[BehaviorSegment]] = defaultdict(list)
    for s in segments:
        by_clip[s.clip_id].append(s)
    for cid in by_clip:
        by_clip[cid].sort(key=lambda s: s.start_frame)

    wanted = {frozenset(p) for p in transition_types}
    selected: set[int] = set()

    for cid, segs in sorted(by_clip.items()):
        for a, b in zip(segs[:-1], segs[1:]):
            if a.behavior == b.behavior:
                continue
            if frozenset({a.behavior, b.behavior}) in wanted:
                selected.add(cid)
                break
        if max_clips is not None and len(selected) >= max_clips:
            break
    return selected


# ──────────────────── Pooled feature extraction ────────────────────


@dataclass
class PooledData:
    X_full: np.ndarray                     # (N_full, D)
    clip_ids: np.ndarray                   # (N_full,) source clip_id per frame
    frame_idxs: np.ndarray                 # (N_full,) source frame idx per frame
    per_clip_offsets: dict[int, slice]     # clip_id → slice into X_full
    pca_var: tuple[float, float] | None    # (var_ratio[0], var_ratio[1]) if PCA
    pca_model: PCA | None


def _load_clip_raw(data_path: str, dm_key: str, clip_id: int) -> np.ndarray:
    hdf5_key = get_hdf5_key_for_dm_key(dm_key)
    return get_aggregate_data(f"/{hdf5_key}", [], clip_id, data_path)


def build_pooled_features(
    cfg: GlobalManifoldConfig,
    fs: FeatureSpaceConfig,
    clip_ids_sorted: list[int],
) -> PooledData:
    preset = get_dataset_preset(cfg.dataset)

    raws: list[np.ndarray] = []
    for cid in clip_ids_sorted:
        raws.append(_load_clip_raw(cfg.data_path, fs.dm_key, cid))

    min_frames = min(r.shape[0] for r in raws)
    raws = [r[:min_frames] for r in raws]
    batch = np.stack(raws, axis=0)

    transformed = apply_dm_key_transform(
        batch,
        fs.dm_key,
        xml_path=cfg.xml_path,
        stac_data_path=cfg.stac_data_path,
        dataset=cfg.dataset,
    )

    n_clips, n_frames, D = transformed.shape
    pca_var = None
    pca_model = None

    if fs.pca_dims is not None and fs.pca_dims < D:
        flat = transformed.reshape(-1, D)
        pca_model = PCA(n_components=fs.pca_dims)
        flat_pca = pca_model.fit_transform(flat)
        transformed = flat_pca.reshape(n_clips, n_frames, fs.pca_dims)
        D = fs.pca_dims
        pca_var = (
            float(pca_model.explained_variance_ratio_[0]),
            float(pca_model.explained_variance_ratio_[1])
            if fs.pca_dims >= 2
            else 0.0,
        )
    else:
        flat = transformed.reshape(-1, D)
        pm = PCA(n_components=min(2, D))
        pm.fit(flat)
        pca_var = (
            float(pm.explained_variance_ratio_[0]),
            float(pm.explained_variance_ratio_[1]) if D >= 2 else 0.0,
        )

    X_full = transformed.reshape(-1, D).astype(np.float32)
    clip_ids_arr = np.repeat(np.array(clip_ids_sorted, dtype=np.int32), n_frames)
    frame_idxs = np.tile(np.arange(n_frames, dtype=np.int32), n_clips)

    per_clip_offsets = {
        cid: slice(i * n_frames, (i + 1) * n_frames)
        for i, cid in enumerate(clip_ids_sorted)
    }

    return PooledData(
        X_full=X_full,
        clip_ids=clip_ids_arr,
        frame_idxs=frame_idxs,
        per_clip_offsets=per_clip_offsets,
        pca_var=pca_var,
        pca_model=pca_model,
    )


# ──────────────────── Takens + subsampling ────────────────────


@dataclass
class TakensPool:
    X_takens: np.ndarray                   # (N_takens_sub, m·D)
    clip_ids: np.ndarray                   # (N_takens_sub,) source clip
    frame_idxs: np.ndarray                 # (N_takens_sub,) source frame (anchor)
    m: int
    tau: int


def _takens_one_clip(X: np.ndarray, m: int, tau: int) -> tuple[np.ndarray, np.ndarray]:
    n_frames, D = X.shape
    start = (m - 1) * tau
    if start >= n_frames:
        return np.empty((0, m * D), dtype=X.dtype), np.empty((0,), dtype=np.int32)
    anchor_idxs = np.arange(start, n_frames, dtype=np.int32)
    pts = np.stack(
        [X[[j - a * tau for a in range(m)]].reshape(-1) for j in anchor_idxs],
        axis=0,
    )
    return pts.astype(np.float32), anchor_idxs


def build_takens_pool(
    pooled: PooledData,
    clip_ids_sorted: list[int],
    fs: FeatureSpaceConfig,
    auto_params: tuple[int, int] | None = None,
) -> TakensPool:
    if fs.takens.auto_estimate and auto_params is not None:
        m, tau = auto_params
    else:
        m, tau = fs.takens.m, fs.takens.tau

    all_pts: list[np.ndarray] = []
    all_cid: list[np.ndarray] = []
    all_fr: list[np.ndarray] = []

    for cid in clip_ids_sorted:
        sl = pooled.per_clip_offsets[cid]
        X = pooled.X_full[sl]
        pts, anchors = _takens_one_clip(X, m, tau)
        if len(pts) == 0:
            continue

        sub = slice(0, len(pts), fs.subsample_n)
        pts = pts[sub]
        anchors_sub = anchors[sub]

        all_pts.append(pts)
        all_cid.append(np.full(len(pts), cid, dtype=np.int32))
        all_fr.append(anchors_sub.astype(np.int32))

    X_takens = np.concatenate(all_pts, axis=0) if all_pts else np.empty((0, 0))
    clip_ids = np.concatenate(all_cid) if all_cid else np.empty((0,), dtype=np.int32)
    frame_idxs = np.concatenate(all_fr) if all_fr else np.empty((0,), dtype=np.int32)
    return TakensPool(X_takens=X_takens, clip_ids=clip_ids, frame_idxs=frame_idxs, m=m, tau=tau)


def estimate_takens_params(
    cfg: GlobalManifoldConfig,
    fs: FeatureSpaceConfig,
    clip_ids_sorted: list[int],
    max_clips: int = 40,
) -> tuple[int, int]:
    from topo_mimic.dmwk.skeletonization.preprocess import estimate_delay_embedding_params

    sample_clips = clip_ids_sorted[:max_clips]
    hdf5_key = get_hdf5_key_for_dm_key(fs.dm_key)
    group_path = f"/{hdf5_key}"

    def transform_fn(stacked: np.ndarray) -> np.ndarray:
        out = apply_dm_key_transform(
            stacked,
            fs.dm_key,
            xml_path=cfg.xml_path,
            stac_data_path=cfg.stac_data_path,
            dataset=cfg.dataset,
        )
        return out

    tau_star, m_star, _ = estimate_delay_embedding_params(
        cfg.data_path,
        get_aggregate_data,
        num_clips=len(sample_clips),
        intention_key=group_path,
        clip_indices=sample_clips,
        plot=False,
        transform_fn=transform_fn if fs.dm_key != "intention" else None,
    )
    return int(m_star), int(tau_star)


# ───────────────────── DM extraction (cached) ─────────────────────


@dataclass
class DMBundle:
    cycles: list[np.ndarray]
    G_dm: nx.Graph
    n_takens: int
    dm_output_dir: str
    remap_to_orig: dict[int, int]


def _dm_cache_ok(dm_output_dir: str) -> bool:
    needed = [
        "sparse_weighted_rips_filtration.txt",
        "edge_for_morse_only.txt",
        "sorted-weights.txt",
        "features.txt",
    ]
    return all(os.path.exists(os.path.join(dm_output_dir, f)) for f in needed)


def run_global_dm(
    takens: TakensPool,
    fs: FeatureSpaceConfig,
    cfg: GlobalManifoldConfig,
    cache_space_dir: str,
) -> DMBundle:
    dm_output_dir = os.path.join(cache_space_dir, "dm_output")
    bundle_pkl = os.path.join(cache_space_dir, "dm_bundle.pkl")
    os.makedirs(dm_output_dir, exist_ok=True)

    if os.path.exists(bundle_pkl) and _dm_cache_ok(dm_output_dir):
        with open(bundle_pkl, "rb") as f:
            cached = pickle.load(f)
        return cached

    print(f"  [DM] running on {takens.X_takens.shape[0]} pooled Takens points (dim={takens.X_takens.shape[1]})")

    cycles, G_dm = extract_skeleton_pcd_dm(
        takens.X_takens,
        k=cfg.dm.k_neighbors,
        pers_delta=fs.pers_delta,
        epsilon=cfg.dm.epsilon,
        output_dir=dm_output_dir,
        cleanup=False,
    )
    remap_to_orig = parse_swr_remap(dm_output_dir)
    bundle = DMBundle(
        cycles=cycles,
        G_dm=G_dm,
        n_takens=takens.X_takens.shape[0],
        dm_output_dir=dm_output_dir,
        remap_to_orig=remap_to_orig,
    )
    with open(bundle_pkl, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


# ───────────────────── Minima per label ─────────────────────


@dataclass
class LabelMinima:
    by_label: dict[str, int]                # label → takens_idx (original space)
    by_label_remapped: dict[str, int]       # label → DM-graph vertex id
    all_candidates: list[int]               # takens_idxs of top-N persistent minima


def _takens_label(
    takens_idx: int,
    takens: TakensPool,
    frame_labels: FrameLabels,
) -> str | None:
    cid = int(takens.clip_ids[takens_idx])
    fr = int(takens.frame_idxs[takens_idx])
    fm = frame_labels.primitive_by_frame.get(cid, {})
    return fm.get(fr)


def find_minima_per_label(
    bundle: DMBundle,
    takens: TakensPool,
    frame_labels: FrameLabels,
    sel: MinimaSelectionConfig,
) -> LabelMinima:
    pairs = _parse_persistence_pairs(bundle.dm_output_dir)
    weights = _parse_sorted_weights(bundle.dm_output_dir)
    minima_remapped = _find_persistent_minima(pairs, weights, top_k=sel.candidate_pool)

    # Convert to original (Takens) indices and keep those that lie in the Takens pool
    candidates_orig: list[int] = []
    candidates_remapped: list[int] = []
    for rm in minima_remapped:
        if rm in bundle.remap_to_orig:
            orig = bundle.remap_to_orig[rm]
            if 0 <= orig < bundle.n_takens:
                candidates_orig.append(int(orig))
                candidates_remapped.append(int(rm))

    if not candidates_orig:
        return LabelMinima(by_label={}, by_label_remapped={}, all_candidates=[])

    # kNN in Takens-point space
    k = min(sel.neighbor_votes, bundle.n_takens)
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(takens.X_takens)

    by_label: dict[str, int] = {}
    by_label_remapped: dict[str, int] = {}

    for label in sel.primitives:
        best_orig = None
        best_rm = None
        best_score = -1
        best_density_weight = float("inf")
        for orig, rm in zip(candidates_orig, candidates_remapped):
            _, nbr_ix = nn.kneighbors(takens.X_takens[orig : orig + 1])
            votes = 0
            for nb in nbr_ix[0]:
                lab = _takens_label(int(nb), takens, frame_labels)
                if lab == label:
                    votes += 1
            rm_weight = weights.get(int(rm), float("inf"))
            better = votes > best_score or (
                votes == best_score and rm_weight < best_density_weight
            )
            if better:
                best_score = votes
                best_density_weight = rm_weight
                best_orig = orig
                best_rm = rm
        if best_score > 0 and best_orig is not None:
            by_label[label] = int(best_orig)
            by_label_remapped[label] = int(best_rm)
        else:
            print(f"    [warn] no candidate minimum matched label={label!r}")

    return LabelMinima(
        by_label=by_label,
        by_label_remapped=by_label_remapped,
        all_candidates=candidates_orig,
    )


# ───────────────────── Path / cycle sampling ─────────────────────


@dataclass
class SampledPath:
    label_pair: tuple[str, str]
    takens_idxs: np.ndarray                 # ordered list of takens-point indices
    is_edge_disjoint: bool


def _remap_path_to_takens(
    path_vertices: list[int],
    bundle: DMBundle,
) -> np.ndarray:
    return remap_cycle_to_original(path_vertices, bundle.remap_to_orig, bundle.n_takens)


def sample_paths_per_pair(
    bundle: DMBundle,
    label_minima: LabelMinima,
    transition_types: list[tuple[str, str]],
    sel: PathSamplingConfig,
) -> dict[tuple[str, str], list[SampledPath]]:
    rng = np.random.default_rng(sel.rng_seed)
    out: dict[tuple[str, str], list[SampledPath]] = {}

    for a, b in transition_types:
        if a not in label_minima.by_label_remapped or b not in label_minima.by_label_remapped:
            out[(a, b)] = []
            continue
        src = label_minima.by_label_remapped[a]
        dst = label_minima.by_label_remapped[b]
        if src not in bundle.G_dm or dst not in bundle.G_dm:
            out[(a, b)] = []
            continue

        edp_raw: list[list[int]] = []
        try:
            edp_raw = list(nx.edge_disjoint_paths(bundle.G_dm, src, dst))
        except (nx.NetworkXError, nx.NetworkXNoPath, nx.NetworkXUnfeasible):
            edp_raw = []

        sampled: list[SampledPath] = []
        for p in edp_raw[: sel.k_per_pair]:
            idxs = _remap_path_to_takens(p, bundle)
            if len(idxs) >= sel.min_segment_len:
                sampled.append(SampledPath(label_pair=(a, b), takens_idxs=idxs, is_edge_disjoint=True))

        if len(sampled) < sel.k_per_pair:
            need = sel.k_per_pair - len(sampled)
            try:
                simple_gen = nx.shortest_simple_paths(bundle.G_dm, src, dst)
                pool: list[list[int]] = []
                used = {tuple(p) for p in edp_raw}
                for count, p in enumerate(simple_gen):
                    if count >= sel.simple_path_cap:
                        break
                    if tuple(p) in used:
                        continue
                    pool.append(p)
                if pool:
                    choose = rng.choice(len(pool), size=min(need, len(pool)), replace=False)
                    for ci in choose:
                        idxs = _remap_path_to_takens(pool[int(ci)], bundle)
                        if len(idxs) >= sel.min_segment_len:
                            sampled.append(SampledPath(
                                label_pair=(a, b), takens_idxs=idxs, is_edge_disjoint=False,
                            ))
            except (nx.NetworkXError, nx.NetworkXNoPath):
                pass

        out[(a, b)] = sampled[: sel.k_per_pair]
    return out


# ───────────────────── Projections ─────────────────────


@dataclass
class ProjectionBundle:
    method: str
    coords: np.ndarray                      # (N_full, 2)
    axis_labels: tuple[str, str]


def compute_projections(
    pooled: PooledData,
    proj_cfg: ProjectionConfig,
) -> list[ProjectionBundle]:
    X = pooled.X_full
    results: list[ProjectionBundle] = []

    if X.shape[0] > proj_cfg.max_points:
        rng = np.random.default_rng(proj_cfg.rng_seed)
        sub = rng.choice(X.shape[0], size=proj_cfg.max_points, replace=False)
        sub.sort()
        nn = NearestNeighbors(n_neighbors=1).fit(X[sub])
    else:
        sub = None
        nn = None

    X_fit = X[sub] if sub is not None else X

    for method in proj_cfg.methods:
        if method == "pca":
            pm = PCA(n_components=2)
            if sub is not None:
                coords_sub = pm.fit_transform(X_fit)
                _, nn_idx = nn.kneighbors(X)
                coords = coords_sub[nn_idx[:, 0]]
            else:
                coords = pm.fit_transform(X)
            axis_labels = (
                f"PC1 ({pm.explained_variance_ratio_[0] * 100:.0f}% var)",
                f"PC2 ({pm.explained_variance_ratio_[1] * 100:.0f}% var)",
            )
        elif method == "tsne":
            tsne = TSNE(
                n_components=2,
                perplexity=proj_cfg.tsne_perplexity,
                init="pca",
                random_state=proj_cfg.rng_seed,
                learning_rate="auto",
            )
            coords_fit = tsne.fit_transform(X_fit)
            if sub is not None:
                _, nn_idx = nn.kneighbors(X)
                coords = coords_fit[nn_idx[:, 0]]
            else:
                coords = coords_fit
            axis_labels = ("t-SNE 1", "t-SNE 2")
        elif method == "umap":
            import umap
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=proj_cfg.umap_n_neighbors,
                min_dist=proj_cfg.umap_min_dist,
                random_state=proj_cfg.rng_seed,
            )
            coords_fit = reducer.fit_transform(X_fit)
            if sub is not None:
                _, nn_idx = nn.kneighbors(X)
                coords = coords_fit[nn_idx[:, 0]]
            else:
                coords = coords_fit
            axis_labels = ("UMAP 1", "UMAP 2")
        else:
            raise ValueError(f"unknown projection method: {method}")
        results.append(ProjectionBundle(method=method, coords=coords, axis_labels=axis_labels))

    return results


# ──────────── Takens-vertex → full-data-index mapping ───────────


def takens_anchor_in_full(
    takens: TakensPool,
    pooled: PooledData,
    takens_idx: int,
) -> int:
    cid = int(takens.clip_ids[takens_idx])
    fr = int(takens.frame_idxs[takens_idx])
    start = pooled.per_clip_offsets[cid].start
    return start + fr


def takens_idxs_to_full_idxs(
    takens: TakensPool,
    pooled: PooledData,
    takens_idxs: np.ndarray,
) -> np.ndarray:
    return np.array([takens_anchor_in_full(takens, pooled, int(i)) for i in takens_idxs], dtype=np.int64)


# ───────────────────── Per-feature-space result ─────────────────────


@dataclass
class SpaceResult:
    fs: FeatureSpaceConfig
    pooled: PooledData
    takens: TakensPool
    bundle: DMBundle
    label_minima: LabelMinima
    sampled_paths: dict[tuple[str, str], list[SampledPath]]
    sampled_cycles: dict[tuple[str, str], list[SampledPath]]
    projections: list[ProjectionBundle]


def process_feature_space(
    cfg: GlobalManifoldConfig,
    fs: FeatureSpaceConfig,
    frame_labels: FrameLabels,
    clip_ids_sorted: list[int],
) -> SpaceResult:
    space_cache = os.path.join(cfg.cache_dir, fs.dm_key)
    os.makedirs(space_cache, exist_ok=True)

    pool_npz = os.path.join(space_cache, "pool.npz")
    if os.path.exists(pool_npz):
        loaded = np.load(pool_npz, allow_pickle=True)
        pooled = PooledData(
            X_full=loaded["X_full"],
            clip_ids=loaded["clip_ids"],
            frame_idxs=loaded["frame_idxs"],
            per_clip_offsets={
                int(cid): slice(int(s), int(e))
                for cid, s, e in loaded["per_clip_offsets"]
            },
            pca_var=tuple(loaded["pca_var"]) if "pca_var" in loaded.files else None,
            pca_model=None,
        )
    else:
        print(f"  [pool] building features for {fs.dm_key}  (n_clips={len(clip_ids_sorted)})")
        pooled = build_pooled_features(cfg, fs, clip_ids_sorted)
        offs = np.array(
            [(k, v.start, v.stop) for k, v in pooled.per_clip_offsets.items()],
            dtype=np.int64,
        )
        np.savez(
            pool_npz,
            X_full=pooled.X_full,
            clip_ids=pooled.clip_ids,
            frame_idxs=pooled.frame_idxs,
            per_clip_offsets=offs,
            pca_var=np.array(pooled.pca_var) if pooled.pca_var is not None else np.array([]),
        )

    auto_params = None
    if fs.takens.auto_estimate:
        auto_cache = os.path.join(space_cache, "takens_params.pkl")
        if os.path.exists(auto_cache):
            with open(auto_cache, "rb") as f:
                auto_params = pickle.load(f)
        else:
            print(f"  [takens] auto-estimating m, tau for {fs.dm_key}")
            auto_params = estimate_takens_params(cfg, fs, clip_ids_sorted)
            with open(auto_cache, "wb") as f:
                pickle.dump(auto_params, f)

    takens = build_takens_pool(pooled, clip_ids_sorted, fs, auto_params=auto_params)
    print(f"  [takens] {fs.dm_key}: m={takens.m} tau={takens.tau} pts={takens.X_takens.shape[0]}")

    bundle = run_global_dm(takens, fs, cfg, space_cache)

    label_minima = find_minima_per_label(bundle, takens, frame_labels, cfg.minima_selection)
    print(f"  [minima] {fs.dm_key}: {list(label_minima.by_label.keys())}")

    sampled_paths = sample_paths_per_pair(
        bundle, label_minima, cfg.transition_types, cfg.path_sampling,
    )
    for pair, paths in sampled_paths.items():
        print(f"    paths {pair}: {len(paths)} sampled")

    sampled_cycles: dict[tuple[str, str], list[SampledPath]] = {}
    for (a, b), paths in sampled_paths.items():
        edp = [p for p in paths if p.is_edge_disjoint]
        if len(edp) >= 2:
            sampled_cycles[(a, b)] = edp[: min(cfg.path_sampling.k_per_pair, len(edp))]
        else:
            sampled_cycles[(a, b)] = []

    projections_pkl = os.path.join(space_cache, "projections.pkl")
    if os.path.exists(projections_pkl):
        with open(projections_pkl, "rb") as f:
            projections = pickle.load(f)
    else:
        print(f"  [proj] computing PCA/t-SNE/UMAP for {fs.dm_key}")
        projections = compute_projections(pooled, cfg.projection)
        with open(projections_pkl, "wb") as f:
            pickle.dump(projections, f)

    return SpaceResult(
        fs=fs,
        pooled=pooled,
        takens=takens,
        bundle=bundle,
        label_minima=label_minima,
        sampled_paths=sampled_paths,
        sampled_cycles=sampled_cycles,
        projections=projections,
    )


# ───────────────────── Projection grid figure ─────────────────────


_NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

_DEFAULT_PALETTE = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC9A6C", "#7E6148", "#B09C85",
]


def _build_color_map(primitives: list[str], override: dict[str, str]) -> dict[str, str]:
    cmap: dict[str, str] = {}
    for i, p in enumerate(primitives):
        cmap[p] = override.get(p, _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)])
    return cmap


def _frame_primitive(
    full_idx: int,
    pooled: PooledData,
    frame_labels: FrameLabels,
) -> str | None:
    cid = int(pooled.clip_ids[full_idx])
    fr = int(pooled.frame_idxs[full_idx])
    return frame_labels.primitive_by_frame.get(cid, {}).get(fr)


def _frame_turn(
    full_idx: int,
    pooled: PooledData,
    frame_labels: FrameLabels,
) -> str:
    cid = int(pooled.clip_ids[full_idx])
    fr = int(pooled.frame_idxs[full_idx])
    return frame_labels.turn_by_frame.get(cid, {}).get(fr, "straight")


def render_projection_grid(
    cfg: GlobalManifoldConfig,
    results: list[SpaceResult],
    frame_labels: FrameLabels,
    output_pdf: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    color_map = _build_color_map(cfg.minima_selection.primitives, cfg.visualization.behavior_colors)
    turn_shapes = cfg.visualization.turn_shapes

    n_spaces = len(results)
    n_rows = 2 * n_spaces   # dmc + dmp per space
    n_cols = len(cfg.projection.methods)

    with plt.rc_context(_NATURE_RC):
        fig_w = 2.4 * n_cols
        fig_h = 2.4 * n_rows + 0.8
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

        for ri, res in enumerate(results):
            pooled = res.pooled
            labels_full = np.array(
                [_frame_primitive(i, pooled, frame_labels) for i in range(len(pooled.X_full))]
            )
            turns_full = np.array(
                [_frame_turn(i, pooled, frame_labels) for i in range(len(pooled.X_full))]
            )

            for row_type_idx, row_type in enumerate(["dmc", "dmp"]):
                grid_row = 2 * ri + row_type_idx

                for ci, proj in enumerate(res.projections):
                    ax = axes[grid_row][ci]
                    _draw_projection_panel(
                        ax=ax,
                        coords=proj.coords,
                        labels_full=labels_full,
                        turns_full=turns_full,
                        color_map=color_map,
                        turn_shapes=turn_shapes,
                        primitives=cfg.minima_selection.primitives,
                        vis_cfg=cfg.visualization,
                    )
                    _overlay_dm_primitives(
                        ax=ax,
                        coords=proj.coords,
                        res=res,
                        row_type=row_type,
                        color_map=color_map,
                        vis_cfg=cfg.visualization,
                    )
                    title = f"{res.fs.dm_key} / {row_type.upper()} — {proj.method.upper()}"
                    ax.set_title(title, fontweight="bold", pad=3)
                    ax.set_xlabel(proj.axis_labels[0])
                    ax.set_ylabel(proj.axis_labels[1])
                    ax.set_xticks([])
                    ax.set_yticks([])

        handles = []
        for p in cfg.minima_selection.primitives:
            handles.append(mlines.Line2D(
                [], [], marker="o", color="w",
                markerfacecolor=color_map[p], markersize=6,
                markeredgecolor="none", label=p,
            ))
        for turn_name, shape in turn_shapes.items():
            if turn_name == "straight":
                continue
            handles.append(mlines.Line2D(
                [], [], marker=shape, color="w",
                markerfacecolor="gray", markersize=6,
                markeredgecolor="none", label=f"turn: {turn_name}",
            ))
        handles.append(mlines.Line2D(
            [], [], marker="*", color="w",
            markerfacecolor="gold", markeredgecolor="black",
            markersize=10, label="density minimum",
        ))
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=min(len(handles), 8),
            frameon=False,
        )

        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
        fig.savefig(output_pdf, dpi=cfg.visualization.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  [fig] wrote {output_pdf}")


def _draw_projection_panel(
    ax,
    coords: np.ndarray,
    labels_full: np.ndarray,
    turns_full: np.ndarray,
    color_map: dict[str, str],
    turn_shapes: dict[str, str],
    primitives: list[str],
    vis_cfg: VisualizationConfig,
) -> None:
    prim_set = set(primitives)
    for p in primitives:
        for turn_name, shape in turn_shapes.items():
            mask = (labels_full == p) & (turns_full == turn_name)
            if not np.any(mask):
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=color_map[p], s=vis_cfg.background_size,
                alpha=vis_cfg.background_alpha, marker=shape,
                edgecolors="none", rasterized=True,
            )
    other_mask = np.array([l not in prim_set and l is not None for l in labels_full])
    if np.any(other_mask):
        ax.scatter(
            coords[other_mask, 0], coords[other_mask, 1],
            c="lightgray", s=vis_cfg.background_size * 0.7,
            alpha=vis_cfg.background_alpha * 0.6,
            edgecolors="none", rasterized=True,
        )


def _overlay_dm_primitives(
    ax,
    coords: np.ndarray,
    res: "SpaceResult",
    row_type: str,
    color_map: dict[str, str],
    vis_cfg: VisualizationConfig,
) -> None:
    if row_type == "dmc":
        paths_by_pair = res.sampled_cycles
    else:
        paths_by_pair = res.sampled_paths

    for (a, b), paths in paths_by_pair.items():
        if not paths:
            continue
        pair_color = color_map.get(a, "#888888")
        edge_color = color_map.get(b, "#555555")
        for p in paths:
            if len(p.takens_idxs) < 2:
                continue
            full_idxs = takens_idxs_to_full_idxs(res.takens, res.pooled, p.takens_idxs)
            xy = coords[full_idxs]
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list(
                f"g_{a}_{b}", [pair_color, edge_color], N=max(len(xy), 2),
            )
            t = np.linspace(0, 1, len(xy))
            for i in range(len(xy) - 1):
                ax.plot(
                    xy[i : i + 2, 0], xy[i : i + 2, 1],
                    color=cmap(t[i]),
                    linewidth=vis_cfg.overlay_linewidth,
                    alpha=0.85,
                    solid_capstyle="round",
                )

    for label, takens_idx in res.label_minima.by_label.items():
        if takens_idx >= res.bundle.n_takens:
            continue
        full_idx = takens_anchor_in_full(res.takens, res.pooled, takens_idx)
        ax.scatter(
            coords[full_idx, 0], coords[full_idx, 1],
            marker="*", s=vis_cfg.minima_marker_size,
            c=color_map.get(label, "gold"),
            edgecolors="black", linewidths=0.6, zorder=10,
        )


# ───────────────────── Qpos lookup helper ─────────────────────


def load_qpos_for_clips(data_path: str, clip_ids: list[int]) -> dict[int, np.ndarray]:
    import h5py
    qpos_by_clip: dict[int, np.ndarray] = {}
    with h5py.File(data_path, "r") as f:
        key = "qpos_rollout" if "qpos_rollout" in f else "qpos"
        for cid in clip_ids:
            qpos_by_clip[cid] = np.array(f[key][cid])
    return qpos_by_clip
