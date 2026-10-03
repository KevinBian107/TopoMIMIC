"""Rescore connection-discovery over ALL swept variants, not just EML-selected ones.

The shipped `assemble_variant_metas` drops every KPMS fit that was not the
best-per-kappa EML pick, so only 10 of 50 KPMS fits ever reached discovery. That
makes the paper's "95-variant oracle union" false: the union was over 55.
Model selection is the wrong instrument here, since the question is whether the
baseline's coverage is limited by these knobs at all, which is an oracle-union
question. This rebuilds the metas straight from the cached cluster dicts.

Part of the Table 3 pipeline (`python -m topo_mimic.analysis.scale_sweep.rescore`). No refitting: reads the
cluster_dict.pkl / labels_list.pkl each fit already wrote.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from topo_mimic.paths import REPO_ROOT as REPO

from topo_mimic.analysis.scale_sweep.config import load_config
from topo_mimic.analysis.scale_sweep.data import load_scale_sweep_data
from topo_mimic.analysis.scale_sweep.labels import load_real_transitions_yaml
from topo_mimic.analysis.scale_sweep.discovery_connections import (
    build_connection_discovery_tables,
)

# The sweep writes each config to outputs/.../cfg_<config hash>; rescore that run.
SWEEP_CONFIG = REPO / "configs/analysis/scale_sweep/scale_sweep.yaml"
ROOT = REPO / Path(load_config(SWEEP_CONFIG).io.output_dir) / f"cfg_{load_config(SWEEP_CONFIG).config_hash()}"
MOD_TAG = {"intention": "int", "keypoint": "kp"}


def kpms_metas() -> list[dict]:
    out = []
    for p in sorted(ROOT.glob("kpms_*/kappa_*/states_*/seed_*/cluster_dict.pkl")):
        mod = p.parents[3].name.replace("kpms_", "")
        ke = p.parents[2].name.replace("kappa_", "")
        ns = int(p.parents[1].name.replace("states_", ""))
        out.append({
            "variant_id": f"kpms_{MOD_TAG[mod]}_k{ke}_n{ns}",
            "method": "kpms",
            "feature_space": mod,          # shipped code hardcodes "keypoint" here
            "kappa": float(f"1e{ke[2:]}"),
            "num_states": ns,
            "lens": None,
            "n_intervals": None,
            "cluster_dict_path": str(p),
        })
    return out


def mapper_metas() -> list[dict]:
    out = []
    for p in sorted(ROOT.glob("mapper/*/lens_*/n_*/cluster_dict.pkl")):
        space = p.parents[2].name
        lens = p.parents[1].name.replace("lens_", "")
        n = int(p.parents[0].name.replace("n_", ""))
        out.append({
            "variant_id": f"mapper_{space}_{lens}_n{n}",
            "method": "mapper",
            "feature_space": space,
            "kappa": None,
            "num_states": None,
            "lens": lens,
            "n_intervals": n,
            "cluster_dict_path": str(p),
        })
    return out


def main() -> None:
    cfg = load_config(SWEEP_CONFIG)
    transitions, transition_counts = load_real_transitions_yaml(ROOT / "real_transitions.yaml")
    data = load_scale_sweep_data(cfg)

    metas = kpms_metas() + mapper_metas()
    n_k = sum(1 for m in metas if m["method"] == "kpms")
    print(f"variants to score: {len(metas)}  (KPMS {n_k}, Mapper {len(metas)-n_k})")

    out_dir = ROOT / "discovery_all"
    build_connection_discovery_tables(
        transitions=transitions,
        transition_counts=transition_counts,
        variant_metas=metas,
        segments=data.segments_by_clip,
        clip_ids=data.clip_ids,
        cfg=cfg.discovery,
        output_dir=out_dir,
    )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
