"""Compute the per-space reduction statistics for the appendix table.

For each feature space we report the size of a trajectory at each stage of the
pipeline, all in frames except the skeleton (graph vertices):

  frames     raw trajectory length n, recovered as |X| + (m-1) * tau
  skeleton   mean DM-skeleton vertex count |V(G_T)| over clips with a skeleton
  cycle      mean DM-cycle length (frames) over all extracted cycle primitives
  path       mean DM-path  length (frames) over all extracted path  primitives

Reads ``all_G_sm.pkl`` (skeletons) and ``all_cycle_dm.pkl`` / ``all_path_dm.pkl``
(the extracted primitives that feed clustering) from the cached extraction.
"""
from __future__ import annotations

import csv
import pickle as pkl
from pathlib import Path

import networkx as nx
import numpy as np
import yaml

from .config import REPO, ReductionConfig

SPACE_LABEL = {"intention": "Intention", "qpos": "Qpos", "ekp": "EKP"}


def _takens_mt(cfg: ReductionConfig, space: str) -> tuple[int, int]:
    with open(REPO / cfg.extraction_cfg.format(space=space)) as f:
        raw = yaml.safe_load(f) or {}
    tak = raw.get("takens", {})
    return int(tak.get("m", 3)), int(tak.get("tau", 0))


def _load(path: Path):
    with open(path, "rb") as f:
        return pkl.load(f)


def _prim_lengths(all_prims) -> np.ndarray:
    return np.array([len(np.asarray(seg)) for clip in all_prims for seg in clip])


def _space_stats(cfg: ReductionConfig, space: str) -> dict:
    base = REPO / cfg.extraction_out.format(space=space)
    graphs = _load(base / "all_G_sm.pkl")
    m, tau = _takens_mt(cfg, space)

    pts, verts = [], []
    for g in graphs:
        if not isinstance(g, nx.Graph):
            continue
        pts.append(g.number_of_nodes())
        nz = [n for n, d in g.degree() if d > 0]
        if nz:
            verts.append(len(nz))
    pc = float(np.mean(pts))
    verts = np.asarray(verts, float)

    cyc = _prim_lengths(_load(base / "all_cycle_dm.pkl"))
    pth = _prim_lengths(_load(base / "all_path_dm.pkl"))

    return {
        "space": space,
        "frames": round(pc + (m - 1) * tau),
        "skeleton": round(float(verts.mean())),
        "skeleton_std": round(float(verts.std())),
        "cycle": round(float(cyc.mean())) if len(cyc) else 0,
        "cycle_n": int(len(cyc)),
        "path": round(float(pth.mean())) if len(pth) else 0,
        "path_n": int(len(pth)),
    }


def collect(cfg: ReductionConfig) -> list[dict]:
    return [_space_stats(cfg, s) for s in cfg.spaces]


def write_csv(cfg: ReductionConfig, rows: list[dict]) -> Path:
    out = REPO / cfg.out_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return out


def write_tex(cfg: ReductionConfig, rows: list[dict]) -> Path:
    out = REPO / cfg.out_tex
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Feature space & Frames & Skeleton & DM-cycle & DM-path \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{SPACE_LABEL.get(r['space'], r['space'])} & "
            f"{r['frames']} & {r['skeleton']} & {r['cycle']} & {r['path']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    return out


def run(cfg: ReductionConfig) -> list[dict]:
    rows = collect(cfg)
    for r in rows:
        print(f"  {r['space']:>9}: frames={r['frames']} skeleton={r['skeleton']} "
              f"cycle={r['cycle']} path={r['path']}")
    write_csv(cfg, rows)
    write_tex(cfg, rows)
    return rows
