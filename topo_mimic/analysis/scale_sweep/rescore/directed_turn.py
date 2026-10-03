"""Directed, direction-turn discovery for every swept baseline variant.

The shipped connection scorer stores state pairs UNORDERED and never consults
turn direction, so it can only answer at the 6-canonical-pair resolution. Table 1
scores the benchmark methods at the 17 directed + direction-turn classes, so the
two were not comparable. This scores the sweep variants in Table 1's label space.

Convention, matched to Table 1: GT classes come from `find_all_transitions` over
the same eval clips (642-841). A variant discovers class (A->B, turn) if some
ordered pair of pure-state clusters is temporally adjacent A->B inside the span
of a GT transition of that class, in at least `min_member_clips` distinct clips.

Note this is GENEROUS to the baselines relative to Table 1: it asks only that a
state-pair boundary co-occur with a GT transition, not that a whole cluster be
majority-voted to that class. That bias is deliberate, since the claim it feeds
is an oracle-union upper bound on baseline coverage.

Part of the Table 3 pipeline (`python -m topo_mimic.analysis.scale_sweep.rescore`).
"""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from topo_mimic.paths import REPO_ROOT as REPO

from topo_mimic.analysis.feature_analysis.ground_truth import find_all_transitions
from topo_mimic.analysis.scale_sweep.config import load_config
from topo_mimic.experiment.auto_label import (
    auto_label_clusters,
    load_behavior_segments,
    split_segments_by_kind,
)

from .rescore_all import ROOT, kpms_metas, mapper_metas

TURN_TAG = {"LEFT": "L", "RIGHT": "R", "STRAIGHT": "S"}
MIN_CLIPS = 3
EVAL = range(642, 842)


def gt_by_clip() -> dict[int, list]:
    out: dict[int, list] = defaultdict(list)
    for t in find_all_transitions(str(REPO / "behavior_labels/rodent_labels.csv")):
        if int(t.clip_id) in EVAL:
            out[int(t.clip_id)].append(t)
    return out


def classes_from(gt: dict[int, list]) -> list[str]:
    seen, seen2, seen3 = {}, {}, {}
    for ts in gt.values():
        for t in ts:
            k1 = f"{t.from_behavior} -> {t.to_behavior} [{TURN_TAG[t.turn.name]}]"
            k2 = f"{t.from_behavior} -> {t.to_behavior}"
            k3 = " <-> ".join(sorted((t.from_behavior, t.to_behavior)))
            seen[k1] = seen.get(k1, 0) + 1
            seen2[k2] = seen2.get(k2, 0) + 1
            seen3[k3] = seen3.get(k3, 0) + 1
    return seen, seen2, seen3


def score(meta: dict, segments, clip_ids, gt) -> set[str]:
    cd_path = Path(meta["cluster_dict_path"])
    with cd_path.open("rb") as f:
        cluster_dict = pickle.load(f)
    lp = cd_path.parent / "labels_list.pkl"
    if not lp.exists():
        return set()
    with lp.open("rb") as f:
        labels_list = pickle.load(f)

    labels = auto_label_clusters(cluster_dict, segments, dominance_threshold=0.20)
    pure = {cid: l.primary for cid, l in labels.items() if l.primary and " <-> " not in l.primary}

    sup_dt: dict[str, set[int]] = defaultdict(set)
    sup_d: dict[str, set[int]] = defaultdict(set)
    sup_u: dict[str, set[int]] = defaultdict(set)
    for i, seq in enumerate(labels_list):
        clip = clip_ids[i]
        gts = gt.get(clip)
        if not gts:
            continue
        seq = np.asarray(seq)
        prev, prev_f = None, None
        for f, s in enumerate(seq):
            cur = int(s)
            if cur not in pure:
                prev, prev_f = None, None
                continue
            if prev is not None and prev != cur and pure[prev] != pure[cur]:
                a, b = pure[prev], pure[cur]
                for t in gts:
                    if (t.from_behavior, t.to_behavior) == (a, b) and t.from_start <= f <= t.to_end:
                        sup_dt[f"{a} -> {b} [{TURN_TAG[t.turn.name]}]"].add(clip)
                        sup_d[f"{a} -> {b}"].add(clip)
                        sup_u[" <-> ".join(sorted((a, b)))].add(clip)
                        break
            prev, prev_f = cur, f
    return ({c for c, v in sup_dt.items() if len(v) >= MIN_CLIPS},
            {c for c, v in sup_d.items() if len(v) >= MIN_CLIPS},
            {c for c, v in sup_u.items() if len(v) >= MIN_CLIPS})


def main() -> None:
    cfg = load_config(REPO / "configs/analysis/scale_sweep/scale_sweep.yaml")
    raw = load_behavior_segments(cfg.data.behavior_labels_csv)
    segments, _ = split_segments_by_kind(raw)
    gt = gt_by_clip()
    counts, counts_d, counts_u = classes_from(gt)
    clip_ids = sorted(c for c in EVAL if c in segments and segments[c])
    print(f"eval clips {len(clip_ids)}, GT directed+turn classes {len(counts)}")

    metas = kpms_metas() + mapper_metas()
    rows, rows_d, rows_u = {}, {}, {}
    for i, m in enumerate(metas, 1):
        (rows[m["variant_id"]], rows_d[m["variant_id"]],
         rows_u[m["variant_id"]]) = score(m, segments, clip_ids, gt)
        if i % 20 == 0:
            print(f"  scored {i}/{len(metas)}")

    for lvl, (cnts, rws, name) in {
        "dt": (counts, rows, "discovery_directed_turn_binary_wide.csv"),
        "d": (counts_d, rows_d, "discovery_directed_binary_wide.csv"),
        "u": (counts_u, rows_u, "discovery_unordered_gtmatched_binary_wide.csv"),
    }.items():
        cls = sorted(cnts, key=lambda c: -cnts[c])
        pd.DataFrame({v: [1 if c in h else 0 for c in cls] for v, h in rws.items()},
                     index=cls).rename_axis("transition").to_csv(ROOT / "discovery_all" / name)
        pd.Series(cnts).to_csv(ROOT / "discovery_all" / f"gt_{lvl}_counts.csv")
        print(f"  {lvl}: {len(cls)} classes -> {name}")

    classes = sorted(counts, key=lambda c: -counts[c])
    df = pd.DataFrame(
        {v: [1 if c in hits else 0 for c in classes] for v, hits in rows.items()},
        index=classes,
    )
    df.index.name = "transition"
    out = ROOT / "discovery_all" / "discovery_directed_turn_binary_wide.csv"
    df.to_csv(out)
    pd.Series(counts).to_csv(ROOT / "discovery_all" / "gt_directed_turn_counts.csv")
    k = [c for c in df.columns if "kpms" in c]
    m_ = [c for c in df.columns if "mapper" in c]
    print(f"\nwrote {out}")
    print(f"  KPMS union   {(df[k].max(axis=1)==1).sum()}/{len(classes)}")
    print(f"  Mapper union {(df[m_].max(axis=1)==1).sum()}/{len(classes)}")
    print(f"  COMBINED     {(df.max(axis=1)==1).sum()}/{len(classes)}")


if __name__ == "__main__":
    main()
