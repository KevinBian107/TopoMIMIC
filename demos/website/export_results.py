"""Result data for the website: the Information Bottleneck plane (Figure 2 / Table 8)
and the exact W2 distances between the three DMC-Int primitives shown in the W2 step.

    python -m demos.website.export_results
"""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import ot

from topo_mimic.analysis.feature_analysis.data_loading import load_labels_yaml

SITE = Path("TopoMimic-website/src/data")
MI_CSV = Path("outputs/analysis/method_analysis/information_loss_reduced/data/per_method_mi.csv")
W2 = Path("outputs/experiment/transitions/w2_distance_matrix_intention.pkl")
DMC_INT = Path("outputs/experiment/transitions/w2_dmc_int_kmeans")
W2_PAIRS = Path("outputs/demos/website/w2_pairs.json")
FIELDS = ("H_T_bits", "I_TY_bits", "AMI", "NMI", "lift_over_floor_bits", "n_primitives")


def export_ib() -> None:
    rows = []
    with MI_CSV.open() as f:
        for r in csv.DictReader(f):
            rows.append(
                {"method": r["method"], "family": r["family"], "space": r["feature_space"], "seed": int(r["seed"])}
                | {k: round(float(r[k]), 4) for k in FIELDS}
            )
    (SITE / "ib_plane.json").write_text(json.dumps(rows, indent=1))
    print("ib_plane:", len(rows), "rows")


def _ot_w2sq(X: np.ndarray, Y: np.ndarray) -> float:
    """Exact OT cost between two equal-mass point sets, as in compute_w2_distance_matrix."""
    a = np.full(len(X), 1.0 / len(X))
    b = np.full(len(Y), 1.0 / len(Y))
    return float(ot.emd2(a, b, ot.dist(X, Y)))


def export_w2_pairs(seed: int = 0) -> None:
    """Exact W2^2 between three DMC-Int primitives: two of the largest transition cluster
    and one of another (the website's W2 step renders the same three)."""
    res = pickle.load((DMC_INT / "results.pkl").open("rb"))[seed]
    cd = res["cluster_dict"]
    names, _ = load_labels_yaml(str(DMC_INT), seed)
    with h5py.File("data/rodent_ar.h5", "r") as f:
        intention = np.asarray(f["intention"])
    trans = sorted((cid for cid, n in names.items() if "->" in n), key=lambda c: -len(cd[c]))
    ca, cb = trans[0], next(c for c in trans[1:] if names[c].split("->")[0] != names[trans[0]].split("->")[0])
    pts = lambda m: intention[int(m[0]), np.asarray(m[1]).ravel()]  # noqa: E731
    A0, A1, B0 = pts(cd[ca][0]), pts(cd[ca][1]), pts(cd[cb][0])
    out = {"same": round(_ot_w2sq(A0, A1), 3), "different": round(_ot_w2sq(A0, B0), 3)}
    W2_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    W2_PAIRS.write_text(json.dumps(out))
    print("w2 pairs:", names[ca], "vs", names[cb], out)


if __name__ == "__main__":
    import sys

    if "--pairs-only" in sys.argv:
        export_w2_pairs()
    else:
        export_ib()
        export_w2_pairs()
