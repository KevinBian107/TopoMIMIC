"""Headline metric: transition information I(T;Y) / AMI of the DMP path
coarse-graining as a function of the path-selection persistence delta.

Reuses the information_loss machinery (load_transition_labels for Y, compute_mi
for the metrics). T is the per-frame global path-instance id at each delta."""
from __future__ import annotations

import csv

import numpy as np

from topo_mimic.analysis.method_analysis.information_loss.labels import load_transition_labels
from topo_mimic.analysis.method_analysis.information_loss.metrics import compute_mi

from .config import REPO, PPSweepConfig
from .regen import regenerate_all_path_dm


def _build_T(all_path_dm: list, n_clips: int, frames: int) -> np.ndarray:
    """Per-frame global path-instance id; -1 for frames in no path."""
    T = np.full((n_clips, frames), -1, dtype=np.int64)
    gid = 0
    for c in range(min(len(all_path_dm), n_clips)):
        for prim in all_path_dm[c]:
            f = np.asarray(prim, dtype=int)
            f = f[(f >= 0) & (f < frames)]
            if f.size:
                T[c, f] = gid
            gid += 1
    return T


def run_ami(cfg: PPSweepConfig) -> str:
    labels = load_transition_labels(
        csv_path=str(REPO / cfg.labels_csv), n_clips=cfg.n_clips,
        frames_per_clip=cfg.frames_per_clip, granularity=cfg.y_granularity,
        min_class_count=cfg.min_class_count)
    Y = labels.Y

    rows = []
    for space in cfg.spaces:
        ext = REPO / cfg.prod_extraction_out.format(space=space)
        for m, delta in zip(cfg.delta_multipliers, cfg.deltas_for(space)):
            apd = regenerate_all_path_dm(ext, delta, cfg.segment_gap_limit,
                                         cfg.min_primitive_frames)
            n_paths = sum(len(p) for p in apd)
            T = _build_T(apd, cfg.n_clips, cfg.frames_per_clip)
            r = compute_mi(T, Y, method_name=f"dmp_{space}", family="DMP",
                           feature_space=space, seed=None, n_bootstrap=cfg.n_bootstrap)
            lift = r.I_TY - r.I_TY_shuffled
            rows.append({
                "space": space, "mult": m, "delta": delta, "n_paths": n_paths,
                "n_primitives": r.n_primitives, "I_TY": round(r.I_TY, 4),
                "AMI": round(r.AMI, 4), "NMI": round(r.NMI, 4),
                "H_T": round(r.H_T, 4), "efficiency": round(r.efficiency, 4),
                "lift": round(lift, 4)})
            print(f"[{space} m={m:>3} d={delta:<5}] paths={n_paths:4d} "
                  f"I(T;Y)={r.I_TY:.3f} AMI={r.AMI:.3f} H(T)={r.H_T:.2f} lift={lift:.3f}")
    out = REPO / cfg.sandbox_dir / "ami_vs_delta.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return str(out)
