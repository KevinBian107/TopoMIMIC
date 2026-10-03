"""Appendix cluster-example figures: in-scene ghost panels, one per discovered transition cluster.

Usage:
    python -m topo_mimic.figure_generation.build_cluster_examples METHOD OUT.png [SEED]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from topo_mimic.figure_generation import ghost_scene as G
from topo_mimic.analysis.feature_analysis.data_loading import (
    load_qpos, load_cluster_dict, load_labels_yaml)
from topo_mimic.experiment.auto_label import load_behavior_segments, split_segments_by_kind

ROLL, CSV = "data/rodent_ar.h5", "behavior_labels/rodent_labels.csv"
A_COL, B_COL, O_COL = "#1F4E79", "#B22222", "#9E9E9E"     # matches Figure 6
NP = 6


def _hex(c, a):
    c = c.lstrip("#")
    return [int(c[i:i+2], 16)/255 for i in (0, 2, 4)] + [a]


def beh_of(clip, frames, segs):
    out = []
    for f in frames:
        lab = next((s.behavior for s in segs.get(int(clip), []) if s.start <= f < s.end), "?")
        out.append(lab)
    return out


def build(method, out, n_panels=6, seed=0, ncol=4, W=1900, H=1650):
    mdir = f"outputs/experiment/transitions/{method}"
    cd = load_cluster_dict(mdir, seed)
    labels, _ = load_labels_yaml(mdir, seed)
    behav, _ = split_segments_by_kind(load_behavior_segments(CSV))

    def extent(clip, fr):
        q = load_qpos(clip, ROLL)
        idx = np.linspace(0, len(fr)-1, min(NP, len(fr))).astype(int)
        xy = q[[int(fr[i]) for i in idx]][:, :2]
        return float(np.linalg.norm(xy - xy.mean(0), axis=1).max())

    def score(lab, clip, fr):
        """How cleanly a member realises its label: both behaviours present, in
        order, with few frames belonging to neither."""
        parts = lab.split("->")
        A0 = parts[0].strip()
        B0 = parts[1].strip().rsplit(" ", 1)[0].strip() if len(parts) > 1 else ""
        idx = np.linspace(0, len(fr)-1, min(NP, len(fr))).astype(int)
        bs = beh_of(clip, [int(fr[i]) for i in idx], behav)
        if A0 not in bs or B0 not in bs:
            return -1.0
        purity = sum(b in (A0, B0) for b in bs) / len(bs)
        ordered = bs.index(A0) < len(bs) - 1 - bs[::-1].index(B0)
        return purity + (0.5 if ordered else 0.0)

    pairs, seen = [], set()
    for cid, mem in sorted(cd.items()):
        lab = labels.get(cid, "")
        if "->" not in lab or lab in seen:
            continue
        cands = []
        for c, f in mem[:14]:                       # score a pool, keep the best 2
            fr = np.sort(np.asarray(f).ravel())
            cands.append((score(lab, int(c), fr), int(c), fr))
        # penalise members that will not fit the shared frame
        cands = [(sc - 1.2 * max(0.0, extent(c, f) - 0.16), c, f) for sc, c, f in cands]
        cands.sort(key=lambda t: -t[0])
        if cands[0][0] < 0:
            continue
        seen.add(lab)
        ex = [(c, f) for _, c, f in cands[:2]]
        if len(ex) < 2:
            ex += [ex[-1]]
        pairs.append((lab, ex))
        if len(pairs) >= n_panels:
            break
    picks = [(lab, c, f) for lab, ex in pairs for c, f in ex]

    spans = []
    for _, clip, fr in picks:
        q = load_qpos(clip, ROLL)
        idx = np.linspace(0, len(fr)-1, min(NP, len(fr))).astype(int)
        xy = q[[int(fr[i]) for i in idx]][:, :2]
        spans.append(float(np.linalg.norm(xy - xy.mean(0), axis=1).max()))
    CAM = max(0.68, float(np.percentile(spans, 90)) * 2.0 * 2.45)

    ncol = 4
    nrow = int(np.ceil(len(pairs) / 2))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9*ncol, 3.7*nrow), dpi=200)
    axes = np.atleast_1d(axes).ravel()

    for ax, (lab, clip, fr) in zip(axes, picks):
        q = load_qpos(clip, ROLL)
        idx = np.linspace(0, len(fr)-1, min(NP, len(fr))).astype(int)
        sel = [int(fr[i]) for i in idx]
        behs = beh_of(clip, sel, behav)
        order = [b for i, b in enumerate(behs) if b != "?" and b not in behs[:i]]
        A = order[0] if order else "?"
        B = next((b for b in order if b != A), A)
        n = len(sel)
        cols = []
        for i in range(n):
            g = 0.72 - 0.34 * (i / max(1, n - 1))       # light -> dark with time
            cols.append([g, g, g * 1.02, 0.97])
        # behaviour is carried by the trace instead, so the bodies stay neutral
        lo, hi = sel[0], sel[-1]
        tidx = np.linspace(lo, hi, 48).astype(int)
        tbeh = beh_of(clip, tidx, behav)
        tcols = []
        for b in tbeh:
            c = A_COL if b == A else (B_COL if b == B else O_COL)
            tcols.append(_hex(c, 1.0))
        mcols = []
        for b in behs:
            c = A_COL if b == A else (B_COL if b == B else O_COL)
            mcols.append(_hex(c, 0.95))
        ax.imshow(G.render(q, sel, cols, width=W, height=H, cam_dist=CAM,
                           trace=False))
        ax.axis("off")
    for k, (lab, _) in enumerate(pairs):
        ax0 = axes[k * 2]
        ax0.set_title(lab, fontsize=20, pad=8, loc="left", x=0.0)
        ax0.title.set_position((1.02, 1.0))
        ax0.title.set_ha("center")
    for ax in axes[len(picks):]:
        ax.axis("off")

    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], seed=int(sys.argv[3]) if len(sys.argv) > 3 else 0)
