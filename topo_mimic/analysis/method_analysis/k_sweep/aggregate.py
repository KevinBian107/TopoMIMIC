"""Collect the per-k downstream metrics (DTW / MMD / KID / consistency /
coverage) the feature-analysis stage wrote, into `metric vs k` tables.

Reads the same per-(method, seed) CSVs the production analysis produces:
  {k}/analysis/{method}/seed_{s}/dtw_metrics/*dtw_results.csv  -> Mean_DTW
  {k}/analysis/{method}/seed_{s}/mmd/*.csv                     -> KID, MMD_RBF
  {k}/analysis/{method}/seed_{s}/consistency/*.csv            -> Mean_Intra_DTW
coverage = number of discovered transition rows in the DTW csv.
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path

import numpy as np

from .config import REPO, KSweepConfig

# method name -> (feature space, primitive type)
METHOD_META = {
    "w2_dmc_int_kmeans": ("intention", "cycle"),
    "w2_dmc_qpos_kmeans": ("qpos", "cycle"),
    "w2_dmc_ekp_kmeans": ("ekp", "cycle"),
    "w2_dmp_int_kmeans": ("intention", "path"),
    "w2_dmp_qpos_kmeans": ("qpos", "path"),
    "w2_dmp_ekp_kmeans": ("ekp", "path"),
}


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _col_mean(rows: list[dict], col: str) -> float | None:
    vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
    return float(np.mean(vals)) if vals else None


def _one(globpat: str) -> Path | None:
    hits = sorted(glob.glob(globpat))
    return Path(hits[0]) if hits else None


def collect(cfg: KSweepConfig) -> list[dict]:
    rows = []
    for k in cfg.k_values:
        adir = REPO / cfg.sandbox_dir / f"k_{k}" / "analysis"
        for method, (space, prim) in METHOD_META.items():
            mdir = adir / method
            if not mdir.exists():
                continue
            per_seed = {"dtw": [], "kid": [], "mmd": [], "consistency": [], "coverage": []}
            for sdir in sorted(mdir.glob("seed_*")):
                dtw = _read(_one(f"{sdir}/dtw_metrics/*dtw_results*.csv") or "")
                mmd = _read(_one(f"{sdir}/mmd/*.csv") or "")
                con = _read(_one(f"{sdir}/consistency/*.csv") or "")
                if dtw:
                    per_seed["dtw"].append(_col_mean(dtw, "Mean_DTW"))
                    per_seed["coverage"].append(len(dtw))
                if mmd:
                    per_seed["kid"].append(_col_mean(mmd, "KID"))
                    per_seed["mmd"].append(_col_mean(mmd, "MMD_RBF"))
                if con:
                    per_seed["consistency"].append(_col_mean(con, "Mean_Intra_DTW"))

            def ms(key):
                vals = [v for v in per_seed[key] if v is not None]
                if not vals:
                    return (None, None)
                return (round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4))

            row = {"k": k, "space": space, "primitive": prim, "method": method,
                   "n_seeds": len(per_seed["dtw"])}
            for key in ("dtw", "kid", "mmd", "consistency", "coverage"):
                m, s = ms(key)
                row[f"{key}_mean"], row[f"{key}_std"] = m, s
            rows.append(row)
    return rows


def write_downstream(cfg: KSweepConfig) -> Path:
    rows = collect(cfg)
    out = REPO / cfg.sandbox_dir / "downstream_metrics_vs_k.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return out
