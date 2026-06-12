"""Tier-1 structural stats (label-free) from the extraction pickles: how the
recovered topology changes with k. No clustering, no ground truth."""
from __future__ import annotations

import pickle as pkl
from pathlib import Path

import networkx as nx
import numpy as np

from .config import REPO, KSweepConfig
from .driver import extraction_dir


def _load(p: Path):
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pkl.load(f)


def _clip_lens(prims) -> list[int]:
    """Total frames covered by a clip's primitive list (cycles or paths)."""
    return [int(np.asarray(c).size) for c in prims] if prims else []


def structural_stats(cfg: KSweepConfig) -> list[dict]:
    rows = []
    for k in cfg.k_values:
        for space in cfg.spaces:
            d = extraction_dir(cfg, k, space)
            cycles = _load(d / "all_cycle_dm.pkl")
            paths = _load(d / "all_path_dm.pkl")
            graphs = _load(d / "all_G_sm.pkl")
            if cycles is None:
                print(f"[k={k} {space}] no extraction at {d}, skip")
                continue

            n_cyc = [len(c) for c in cycles]
            n_pth = [len(p) for p in paths] if paths else [0] * len(cycles)
            cyc_cov = [sum(_clip_lens(c)) for c in cycles]
            pth_cov = [sum(_clip_lens(p)) for p in paths] if paths else [0] * len(cycles)

            nodes, edges, comps, giant = [], [], [], []
            for G in (graphs or []):
                if not isinstance(G, nx.Graph):
                    continue
                nn = G.number_of_nodes()
                nodes.append(nn)
                edges.append(G.number_of_edges())
                cc = list(nx.connected_components(G)) if nn else []
                comps.append(len(cc))
                giant.append(max((len(c) for c in cc), default=0) / nn if nn else 0.0)

            def ms(x):
                a = np.asarray(x, float)
                return (float(a.mean()), float(a.std())) if a.size else (0.0, 0.0)

            row = {"k": k, "space": space, "n_clips": len(cycles)}
            for name, vals in [("n_cycles", n_cyc), ("n_paths", n_pth),
                               ("cycle_frames", cyc_cov), ("path_frames", pth_cov),
                               ("g_nodes", nodes), ("g_edges", edges),
                               ("g_components", comps), ("giant_cc_frac", giant)]:
                m, s = ms(vals)
                row[f"{name}_mean"], row[f"{name}_std"] = round(m, 3), round(s, 3)
            row["frac_empty_clips"] = round(float(np.mean(np.asarray(n_cyc) == 0)), 3)
            rows.append(row)
            print(f"[k={k} {space}] beta1={row['n_cycles_mean']:.2f} "
                  f"paths={row['n_paths_mean']:.2f} comps={row['g_components_mean']:.2f} "
                  f"giantCC={row['giant_cc_frac_mean']:.2f}")
    return rows


def write_structural(cfg: KSweepConfig) -> Path:
    import csv
    rows = structural_stats(cfg)
    out = REPO / cfg.sandbox_dir / "structural_stats.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return out
