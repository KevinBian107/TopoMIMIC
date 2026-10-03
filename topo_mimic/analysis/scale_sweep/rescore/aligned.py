"""Score every swept variant with the BENCHMARK's primitive construction.

The sweep's own connection scorer builds primitives from adjacency of pure-state
clusters, whereas the benchmark (Table 1) builds them from sustained syllable
PAIRS and majority-votes each cluster to a directed + direction-turn class. That
made Table 2 incomparable with Table 1: KPMS-KP scores 9/17 there and 3/25 here.

This applies the benchmark path to each swept fit, reusing the very functions the
benchmark uses (`extract_sustained_transitions`, `auto_label_clusters_directional`),
so the only thing that differs between a sweep cell and a Table 1 baseline row is
the swept hyperparameters.

Part of the Table 3 pipeline (`python -m topo_mimic.analysis.scale_sweep.rescore`). No refitting.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

from topo_mimic.paths import REPO_ROOT as REPO

from topo_mimic.analysis.feature_analysis.ground_truth import find_all_transitions
from topo_mimic.analysis.scale_sweep.config import load_config
from topo_mimic.experiment.auto_label import (
    auto_label_clusters_directional,
    load_behavior_segments,
    split_segments_by_kind,
)
from topo_mimic.experiment.methods import extract_sustained_transitions
from .rescore_all import ROOT, kpms_metas, mapper_metas

TURN_TAG = {"Left": "L", "Right": "R", "Straight": "S"}
GT_TURN = {"LEFT": "L", "RIGHT": "R", "STRAIGHT": "S"}
EVAL = range(642, 842)
# benchmark per-family primitive knobs (configs/experiment/transitions.yaml)
PARAMS = {"kpms": dict(min_duration=25, top_k=25, min_instances=3),
          "mapper": dict(min_duration=50, top_k=25, min_instances=3)}


def gt_classes() -> dict[str, int]:
    out: dict[str, int] = {}
    for t in find_all_transitions(str(REPO / "behavior_labels/rodent_labels.csv")):
        if int(t.clip_id) in EVAL:
            k = f"{t.from_behavior} -> {t.to_behavior} [{GT_TURN[t.turn.name]}]"
            out[k] = out.get(k, 0) + 1
    return out


def score(meta: dict, behav, direc, clip_ids: list[int]) -> set[str]:
    lp = Path(meta["cluster_dict_path"]).parent / "labels_list.pkl"
    if not lp.exists():
        return set()
    with lp.open("rb") as f:
        labels_list = pickle.load(f)

    cd, _ = extract_sustained_transitions(labels_list, **PARAMS[meta["method"]])
    # extract_sustained_transitions enumerates labels_list, so members carry the
    # POSITIONAL index; remap to absolute clip ids before GT lookup.
    cd = {cid: [(clip_ids[i], fr) for i, fr in mem if i < len(clip_ids)]
          for cid, mem in cd.items()}

    labels = auto_label_clusters_directional(cd, behav, direc)
    found = set()
    for lab in labels.values():
        s = (lab.primary or "").strip()
        if " -> " not in s:
            continue
        pair, _, direction = s.rpartition(" ")
        if direction not in TURN_TAG:
            continue
        found.add(f"{pair} [{TURN_TAG[direction]}]")
    return found


def main() -> None:
    cfg = load_config(REPO / "configs/analysis/scale_sweep/scale_sweep.yaml")
    raw = load_behavior_segments(cfg.data.behavior_labels_csv)
    behav, direc = split_segments_by_kind(raw)
    clip_ids = sorted(c for c in EVAL if c in behav and behav[c])
    counts = gt_classes()
    print(f"eval clips {len(clip_ids)}, GT classes {len(counts)}")

    metas = kpms_metas() + mapper_metas()
    rows = {}
    for i, m in enumerate(metas, 1):
        rows[m["variant_id"]] = score(m, behav, direc, clip_ids)
        if i % 25 == 0:
            print(f"  {i}/{len(metas)}")

    classes = sorted(counts, key=lambda c: -counts[c])
    df = pd.DataFrame({v: [1 if c in h else 0 for c in classes] for v, h in rows.items()},
                      index=classes)
    df.index.name = "transition"
    out = ROOT / "discovery_all" / "discovery_aligned_binary_wide.csv"
    df.to_csv(out)
    k = [c for c in df.columns if "kpms" in c]
    m_ = [c for c in df.columns if "mapper" in c]
    print(f"\nwrote {out}")
    print(f"  best single KPMS {int(df[k].sum().max())}, best single Mapper {int(df[m_].sum().max())}")
    print(f"  unions: KPMS {int((df[k].max(axis=1)==1).sum())}, Mapper {int((df[m_].max(axis=1)==1).sum())}, "
          f"combined {int((df.max(axis=1)==1).sum())} of {len(classes)}")
    # off-label classes (voted to a class with no GT instances in eval) would be dropped silently
    extra = {c for h in rows.values() for c in h} - set(classes)
    if extra:
        print(f"  NOTE {len(extra)} voted classes absent from GT eval set: {sorted(extra)[:6]}")


if __name__ == "__main__":
    main()
