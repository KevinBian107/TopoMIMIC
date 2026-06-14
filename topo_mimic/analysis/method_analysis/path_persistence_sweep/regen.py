"""Regenerate `all_path_dm` at a given path-persistence delta, reusing the cached
DM graph + clip-dir filtration with the EXACT production code path (no C++ re-run).

Validated: at delta=0 this reproduces the production all_path_dm.pkl per clip.
"""
from __future__ import annotations

import pickle as pkl
from pathlib import Path

import networkx as nx

from topo_mimic.extraction.skeletonization.dm_path import _parse_sorted_weights
from topo_mimic.extraction.skeletonization.remap import (
    parse_swr_remap, remap_cycle_to_original, split_cycle_into_segments,
)
from topo_mimic.extraction.skeletonization.skeletonization import extract_persistence_paths


def _primitives(vertex_lists, remap, n_takens, gap_limit, min_frames):
    """Exact copy of dm_pipeline._primitives: vertices -> original frames ->
    split at gaps -> keep segments of at least `min_frames`."""
    prims = []
    for vl in vertex_lists:
        orig = remap_cycle_to_original(vl, remap, n_takens)
        if len(orig) == 0:
            continue
        for seg in split_cycle_into_segments(orig, gap_limit=gap_limit):
            if len(seg) >= min_frames:
                prims.append(seg)
    return prims


def regenerate_all_path_dm(extraction_dir: str | Path, delta: float,
                           gap_limit: int = 50, min_frames: int = 30) -> list:
    extraction_dir = Path(extraction_dir)
    G_list = pkl.load(open(extraction_dir / "all_G_sm.pkl", "rb"))
    out = []
    for i, G in enumerate(G_list):
        clip_dir = extraction_dir / "clips" / f"clip_{i}"
        if not isinstance(G, nx.Graph) or G.number_of_edges() == 0:
            out.append([])
            continue
        try:
            vweight = _parse_sorted_weights(str(clip_dir))
            remap = parse_swr_remap(str(clip_dir))
            n_takens = sum(1 for _ in open(clip_dir / "features.txt"))
        except Exception:
            out.append([])
            continue
        vlists = extract_persistence_paths(G, vweight, persistence_delta=delta)
        out.append(_primitives(vlists, remap, n_takens, gap_limit, min_frames))
    return out
