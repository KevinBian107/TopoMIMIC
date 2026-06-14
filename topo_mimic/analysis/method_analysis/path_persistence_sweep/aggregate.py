"""Collect the downstream DMP metrics (DTW/MMD/KID/coverage) at each path-
persistence multiplier into downstream_vs_delta.csv. mult=0 is the production
DMP run (path_persistence_delta=0.0), read from the main feature-analysis output.
"""
from __future__ import annotations

import csv
import glob

import numpy as np

from .config import REPO, PPSweepConfig

DMP = {"w2_dmp_int_kmeans": "intention", "w2_dmp_qpos_kmeans": "qpos",
       "w2_dmp_ekp_kmeans": "ekp"}


def _rd(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _one(globpat: str):
    hits = sorted(glob.glob(globpat))
    return hits[0] if hits else None


def _agg_dir(adir: str) -> dict:
    out = {}
    for m, sp in DMP.items():
        dtw, mmd, kid, cov = [], [], [], []
        for sd in sorted(glob.glob(f"{adir}/{m}/seed_*")):
            d = _rd(_one(f"{sd}/dtw_metrics/*dtw_results*.csv") or "")
            mm = _rd(_one(f"{sd}/mmd/*.csv") or "")
            if d:
                dtw.append(np.mean([float(r["Mean_DTW"]) for r in d]))
                cov.append(len(d))
            if mm:
                mmd.append(np.mean([float(r["MMD_RBF"]) for r in mm]))
                kid.append(np.mean([float(r["KID"]) for r in mm]))
        if dtw:
            out[sp] = (round(float(np.mean(dtw)), 4), round(float(np.mean(mmd)), 4),
                       round(float(np.mean(kid)), 4), round(float(np.mean(cov)), 2))
    return out


def write_downstream_vs_delta(cfg: PPSweepConfig) -> str:
    sources = {0.0: str(REPO / "outputs/analysis/feature_analysis")}
    for m in cfg.downstream_multipliers:
        sources[m] = str(REPO / cfg.sandbox_dir / f"delta_m{m}" / "analysis")

    rows = []
    for m in sorted(sources):
        a = _agg_dir(sources[m])
        for sp in cfg.spaces:
            if sp not in a:
                continue
            dtw, mmd, kid, cov = a[sp]
            rows.append({"space": sp, "mult": m,
                         "delta": round(m * cfg.construct_delta[sp], 6),
                         "dtw": dtw, "mmd": mmd, "kid": kid, "cov": cov})
    out = REPO / cfg.sandbox_dir / "downstream_vs_delta.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return str(out)
