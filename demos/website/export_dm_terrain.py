"""Export the real DM-G reconstruction of the Figure 3A road network for the website.

Runs the PCD backend on the synthetic road-network cloud at a sweep of persistence
thresholds and writes one JSON with the cloud, its inverse-density score rho, a
display terrain, and the raw recovered graph (plus its DM-cycles and DM-paths) per delta.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from topo_mimic.extraction.skeletonization.dm_path import _parse_sorted_weights  # noqa: E402
from topo_mimic.extraction.skeletonization.remap import parse_swr_remap  # noqa: E402
from topo_mimic.extraction.skeletonization.skeletonization import (  # noqa: E402
    extract_all_cycles,
    extract_persistence_paths,
    extract_skeleton_pcd_dm,
)
from topo_mimic.figure_generation.build_figure1 import (  # noqa: E402
    _road_density,
    _road_network_segments,
    _sample_road_cloud,
)


@dataclass
class TerrainConfig:
    k: int = 5
    deltas: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2)
    grid: int = 72
    lim: float = 4.7
    out_json: Path = REPO / "TopoMimic-website/src/data/dm_terrain.json"
    work_dir: Path = REPO / "outputs/demos/website/dm_terrain"


def reconstruct(cloud: np.ndarray, cfg: TerrainConfig) -> list[dict]:
    levels = []
    for delta in cfg.deltas:
        _, G = extract_skeleton_pcd_dm(
            cloud, k=cfg.k, pers_delta=delta, output_dir=str(cfg.work_dir), cleanup=False
        )
        G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])
        # Backend graph indices are in sparse-Rips (weight-sorted) order; decompose there,
        # exactly as dm_pipeline does, then map every index back to the original points.
        w = _parse_sorted_weights(str(cfg.work_dir))
        remap = parse_swr_remap(str(cfg.work_dir))
        cycles = extract_all_cycles(G, vertex_weight=w)
        paths = extract_persistence_paths(G, w)
        to_orig = lambda idx: [int(remap[int(i)]) for i in idx if int(i) in remap]  # noqa: E731
        edges_o = [to_orig([a, b]) for a, b in G.edges()]
        cycles = [to_orig(c) for c in cycles]
        paths = [to_orig(p) for p in paths]
        levels.append(
            {
                "delta": delta,
                "edges": [e for e in edges_o if len(e) == 2],
                "cycles": cycles,
                "paths": [p for p in paths if len(p) >= 2],
                "beta1": len(cycles),
            }
        )
        print(f"delta={delta}: {G.number_of_edges()} edges, {len(cycles)} cycles, {len(paths)} paths")
    return levels


def main() -> None:
    cfg = TerrainConfig()
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    cloud = _sample_road_cloud()
    levels = reconstruct(cloud, cfg)
    rho = np.loadtxt(cfg.work_dir / "weights.txt")

    g = np.linspace(-cfg.lim, cfg.lim, cfg.grid)
    GX, GY = np.meshgrid(g, g)
    Z = _road_density(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    point_h = _road_density(cloud)

    out = {
        "points": np.round(cloud, 3).tolist(),
        "rho": np.round(rho, 4).tolist(),
        "height": np.round(point_h, 3).tolist(),
        "hidden": [[np.round(a, 3).tolist(), np.round(b, 3).tolist()] for a, b in _road_network_segments()],
        "grid": {"lim": cfg.lim, "n": cfg.grid, "z": np.round(Z, 3).tolist()},
        "k": cfg.k,
        "levels": levels,
    }
    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_json.write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", cfg.out_json, cfg.out_json.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
