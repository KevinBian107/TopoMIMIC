"""Figure 5 (behaviour lexicon) as ghost panels: one directed GT transition per
cell, one example each, titled with the transition and its instance count.

Usage:
    python -m topo_mimic.figure_generation.build_lexicon OUT.png

Example selection is behaviour-aware. A generic "largest pose change" score picks
instances that move a lot but do not visibly *read* as the labelled pair, so each
side of the boundary is scored by the kinematic signature of its own label
(speed / yaw rate / torso height) and the two ranks are combined with a min, so a
strong second half cannot compensate for a first half that shows nothing.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topo_mimic.figure_generation import ghost_scene as G
from topo_mimic.analysis.feature_analysis.data_loading import load_qpos
from topo_mimic.analysis.feature_analysis.ground_truth import find_all_transitions

ROLL = "data/rodent_ar.h5"
CSV = "behavior_labels/rodent_labels.csv"
NP = 6
HALF = 40      # frames of each label kept around the boundary
MAXEXT = 0.40  # xy extent cap, so one example cannot blow up the shared frame


def _yaw(q):
    w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _kin(q, lo, hi):
    """(speed, yaw rate, torso height) over [lo, hi)."""
    seg = q[lo:hi]
    if len(seg) < 3:
        return 0.0, 0.0, float(seg[:, 2].mean()) if len(seg) else 0.0
    speed = float(np.linalg.norm(np.diff(seg[:, :2], axis=0), axis=1).mean())
    yaw = float(np.abs(np.diff(np.unwrap(_yaw(seg)))).mean())
    return speed, yaw, float(seg[:, 2].mean())


def _behaviour_score(name, speed, yaw, zh):
    """Higher = the window looks more like `name`."""
    if name == "Immobile":
        return -(speed + 0.3 * yaw)
    if name == "Walk":
        return speed - 0.5 * yaw       # translation without pivoting
    if name == "Turn":
        return yaw
    if name == "Rear":
        return zh
    return 0.0


def _rank(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return np.ones(len(v))
    o = np.argsort(np.argsort(v))
    return o / (len(v) - 1)


def build(out, n_types=None, W=1700, H=1500):
    trans = find_all_transitions(CSV)
    by = {}
    for t in trans:
        by.setdefault((t.from_behavior, t.to_behavior), []).append(t)
    order = sorted(by.items(), key=lambda kv: -len(kv[1]))
    if n_types:
        order = order[:n_types]

    picks = []
    for (a, b), items in order:
        cands = []
        for t in items:
            q = load_qpos(int(t.clip_id), ROLL)
            n = q.shape[0]
            lo = max(0, min(t.from_end, n) - min(HALF, t.from_end - t.from_start))
            hi = min(n, t.to_start + min(HALF, t.to_end - t.to_start))
            if hi - lo < 25 or t.from_end - t.from_start < 8 or t.to_end - t.to_start < 8:
                continue
            fr = np.linspace(lo, hi - 1, NP).astype(int)
            if len(set(fr.tolist())) < NP:
                continue
            xy = q[fr][:, :2]
            if float(np.linalg.norm(xy - xy.mean(0), axis=1).max()) > MAXEXT:
                continue
            sa = _behaviour_score(a, *_kin(q, lo, min(t.from_end, n)))
            sb = _behaviour_score(b, *_kin(q, t.to_start, hi))
            cands.append((sa, sb, int(t.clip_id), fr))
        if not cands:
            t = max(items, key=lambda x: x.to_end - x.from_start)
            q = load_qpos(int(t.clip_id), ROLL)
            fr = np.linspace(t.from_start, min(t.to_end, q.shape[0] - 1), NP).astype(int)
            best = (int(t.clip_id), fr)
        else:
            ra, rb = _rank([c[0] for c in cands]), _rank([c[1] for c in cands])
            j = int(np.argmax(np.minimum(ra, rb)))
            best = (cands[j][2], cands[j][3])
        print(f"  {a:>8} -> {b:<8} n={len(items):<4} clip={best[0]:<4} "
              f"frames={best[1][0]}..{best[1][-1]}")
        picks.append((f"{a} $\\rightarrow$ {b}  (n={len(items)})", [best]))

    spans = []
    for _, ex in picks:
        for c, f in ex:
            xy = load_qpos(c, ROLL)[f][:, :2]
            spans.append(float(np.linalg.norm(xy - xy.mean(0), axis=1).max()))
    CAM = max(0.68, float(np.percentile(spans, 90)) * 2.0 * 2.45)

    ncol = 4
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.7 * nrow), dpi=200)
    axes = np.atleast_1d(axes).ravel()
    k = 0
    for lab, ex in picks:
        for c, f in ex:
            q = load_qpos(c, ROLL)
            n = len(f)
            cols = [[0.72 - 0.34 * (i / max(1, n - 1))] * 2
                    + [(0.72 - 0.34 * (i / max(1, n - 1))) * 1.02, 0.97] for i in range(n)]
            axes[k].imshow(G.render(q, [int(x) for x in f], cols, width=W, height=H,
                                    cam_dist=CAM, trace=False))
            axes[k].axis("off")
            k += 1
    for ax in axes[k:]:
        ax.axis("off")
    for j, (lab, _) in enumerate(picks):
        axes[j].set_title(lab, fontsize=16, pad=6)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    build(sys.argv[1])
