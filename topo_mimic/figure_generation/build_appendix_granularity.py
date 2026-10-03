#!/usr/bin/env python3
"""Whole-body cycle-variance box plot, replacing the per-joint cycle_lifetime figure.

For each feature space (intention, qpos, ekp) and each persistence value $\\delta$
we have run DM-G on, we collect every cycle from every clip and compute the
**whole-body variance per cycle** — the trace of the covariance matrix of the
67-dim root-stripped qpos vector restricted to the cycle's frames:

    cycle_variance = sum_d Var_t( qpos[clip, frames_in_cycle, 7+d] )    for d=0..66

This replaces the earlier per-joint variance (one DOF only) with a single-scalar
summary of the whole body's pose spread inside the cycle. We then box-plot the
distribution of cycle variances per persistence bin, one panel per feature
space (3 panels total, instead of the original 9).

Frame index extraction follows the original code in
`topo_mimic.analysis.method_analysis.granularity.stationary_analysis.extract_cycle_frame_indices`
(cycle vertex indices live in Takens space; we add `(m-1)*tau` to recover qpos
indices).

Outputs:
    paper/appendix_figures/appendix_lifetime_wholebody_<space>.pdf
    outputs/analysis/method_analysis/granularity_<space>/stationary/plots/wholebody_cycle_variance.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from topo_mimic.analysis.method_analysis.granularity.dm_multi_persistence import (  # noqa: E402
    _LegacyAwareUnpickler,
)

SKIP_ROOT = 7
H5_PATH = REPO / "data/rodent_ar.h5"

# Mirror the per-space Takens settings used by the original granularity configs.
TAKENS = {
    "intention": (3, 6),   # m, tau
    "qpos":      (3, 9),
    "ekp":       (3, 8),
}

SPACES = ["intention", "qpos", "ekp"]
COLOR = "#4C72B0"
EDGE = "#222222"


def load_qpos() -> np.ndarray:
    with h5py.File(H5_PATH, "r") as h:
        if "qpos" in h:
            return h["qpos"][:]
        return h["qpos_rollout"][:]


def cycle_frame_indices(cycles: list, takens_offset: int) -> list[np.ndarray]:
    """List of qpos-frame index arrays, one per cycle. Mirrors stationary_analysis.extract_cycle_frame_indices."""
    out = []
    for cycle in cycles:
        if len(cycle) < 2:
            continue
        out.append(np.asarray(cycle, dtype=int) + takens_offset)
    return out


def cycle_wholebody_variance(qpos_clip: np.ndarray, frames: np.ndarray) -> float | None:
    """Trace of cov matrix of root-stripped qpos restricted to `frames`. (rad²)"""
    n_frames = qpos_clip.shape[0]
    valid = frames[(frames >= 0) & (frames < n_frames)]
    if valid.size < 3:
        return None
    body = qpos_clip[valid, SKIP_ROOT:]   # (n_frames_in_cycle, 67)
    return float(np.var(body, axis=0).sum())


def collect_per_space(space: str, qpos: np.ndarray) -> tuple[list[float], list[list[float]]]:
    """Return (sorted_pers_values, list of per-cycle variance lists per pers)."""
    cache_root = REPO / f"outputs/analysis/method_analysis/granularity_{space}/dm_cache"
    pers_dirs = sorted(cache_root.glob("pers_*"))
    m, tau = TAKENS[space]
    takens_offset = (m - 1) * tau

    pers_values: list[float] = []
    per_pers_vars: list[list[float]] = []

    for pdir in pers_dirs:
        try:
            pers = float(pdir.name.split("_")[1])
        except ValueError:
            continue
        pkl = pdir / "dm_result.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as f:
            try:
                res = _LegacyAwareUnpickler(f).load()
            except Exception as e:
                print(f"  skip {pkl}: {e}")
                continue

        bin_vars: list[float] = []
        for clip_idx, cycles in enumerate(res.all_cycle_dm):
            if not cycles or clip_idx >= qpos.shape[0]:
                continue
            for frames in cycle_frame_indices(cycles, takens_offset):
                v = cycle_wholebody_variance(qpos[clip_idx], frames)
                if v is not None:
                    bin_vars.append(v)

        pers_values.append(pers)
        per_pers_vars.append(bin_vars)
        print(f"  pers={pers:.3f}: {len(bin_vars):>5d} cycles")

    return pers_values, per_pers_vars


def boxplot_per_space(
    pers_values: list[float],
    per_pers_vars: list[list[float]],
    space: str,
    out_path: Path,
) -> None:
    if not any(per_pers_vars):
        print(f"  no data for {space}")
        return
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    positions = list(range(len(pers_values)))
    bp = ax.boxplot(
        per_pers_vars,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR)
        patch.set_alpha(0.7)
        patch.set_edgecolor(EDGE)
        patch.set_linewidth(0.7)
    for elem in ("whiskers", "caps", "medians"):
        for line in bp[elem]:
            line.set_color(EDGE)
            line.set_linewidth(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{p:.2f}" for p in pers_values], rotation=45, ha="right")
    ax.set_xlabel(r"max persistence $\delta$ survived")
    ax.set_ylabel(r"whole-body variance per cycle (rad$^2$)")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(direction="out", length=2.5, width=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


def main() -> None:
    print(f"Loading qpos from {H5_PATH}...")
    qpos = load_qpos()
    print(f"  shape {qpos.shape}")

    appendix_dir = REPO / "paper/appendix_figures"

    for space in SPACES:
        print(f"\n=== {space} ===")
        pers_values, per_pers_vars = collect_per_space(space, qpos)
        local_out = REPO / f"outputs/analysis/method_analysis/granularity_{space}/stationary/plots/wholebody_cycle_variance.pdf"
        local_out.parent.mkdir(parents=True, exist_ok=True)
        boxplot_per_space(pers_values, per_pers_vars, space, local_out)

        appendix_target = appendix_dir / f"appendix_lifetime_wholebody_{space}.pdf"
        try:
            appendix_target.write_bytes(local_out.read_bytes())
            print(f"  copied -> {appendix_target.name}")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
