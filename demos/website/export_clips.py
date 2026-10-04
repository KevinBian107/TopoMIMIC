"""Per-clip DM skeleton data for the website pipeline explorer, in every feature space.

Clips are chosen by a fixed rule: for each behavior pair, among clips dominated by that
pair (each behavior >= 15% of frames, together >= 50%) whose root stays within
``MAX_EXTENT`` and which yield at least one primitive in every feature space, take the
clip with the most DM primitives summed over the three spaces.

For each (clip, space) the Takens cloud the skeleton was built on is projected with
t-SNE for display only; skeleton, rho, cycles and paths are the stored pipeline outputs.

    python -m demos.website.export_clips
"""
from __future__ import annotations

import collections
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

from demos.website.ghost_video import qpos_rollout
from topo_mimic.analysis.feature_analysis.data_loading import center_offset_for_space
from topo_mimic.extraction.skeletonization.remap import parse_swr_remap
from topo_mimic.experiment.auto_label import load_behavior_segments, split_segments_by_kind

CSV = "behavior_labels/rodent_labels.csv"
SPACES = ("intention", "qpos", "ekp")
PAIRS = (("Immobile", "Rear"), ("Immobile", "Turn"), ("Rear", "Turn"), ("Rear", "Walk"))
MAX_EXTENT = 0.35
SITE = Path("TopoMimic-website")
OUT = SITE / "src/data/clips.json"


@dataclass
class SpaceData:
    graphs: list
    cycles: list
    paths: list


def load_space(space: str) -> SpaceData:
    d = Path(f"outputs/extraction/dm_pipeline_{space}")
    load = lambda n: pickle.load(open(d / f"{n}.pkl", "rb"))  # noqa: E731
    return SpaceData(load("all_G_sm"), load("all_cycle_dm"), load("all_path_dm"))


def frame_labels(segs: dict, clip: int, n: int) -> list[str]:
    lab = [""] * n
    for s in segs.get(clip, []):
        for f in range(max(0, s.start), min(n, s.end)):
            lab[f] = s.behavior
    return lab


def choose_clips(data: dict[str, SpaceData], behav: dict) -> list[tuple[tuple[str, str], int]]:
    best: dict[tuple[str, str], tuple[int, int]] = {}
    for c in range(len(data["intention"].graphs)):
        n_prim = [len(data[s].cycles[c]) + len(data[s].paths[c]) for s in SPACES]
        if min(n_prim) < 1:
            continue
        cnt = collections.Counter(frame_labels(behav, c, 500))
        cnt.pop("", None)
        top = cnt.most_common(2)
        if len(top) < 2:
            continue
        (a, na), (b, nb) = top
        pair = tuple(sorted((a, b)))
        if pair not in PAIRS or na < 75 or nb < 75 or na + nb < 250:
            continue
        xy = qpos_rollout(c)[:, :2]
        if float(np.linalg.norm(xy - xy.mean(0), axis=1).max()) > MAX_EXTENT:
            continue
        score = sum(n_prim)
        if pair not in best or score > best[pair][0]:
            best[pair] = (score, c)
    return [(p, best[p][1]) for p in PAIRS if p in best]


def space_payload(space: str, sd: SpaceData, clip: int) -> dict:
    d = Path(f"outputs/extraction/dm_pipeline_{space}/clips/clip_{clip}")
    X = np.loadtxt(d / "features.txt")
    rho = np.loadtxt(d / "weights.txt")
    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0)).max()
    G = sd.graphs[clip]
    remap = parse_swr_remap(str(d))
    prims = [{"type": "cycle", "nodes": [int(v) for v in c]} for c in sd.cycles[clip]]
    prims += [{"type": "path", "nodes": [int(v) for v in p]} for p in sd.paths[clip]]
    for p in prims:
        p["span"] = [min(p["nodes"]), max(p["nodes"])]
    prims.sort(key=lambda p: p["span"][0])
    return {
        "offset": center_offset_for_space(space),
        "xy": np.round(xy, 4).tolist(),
        "rho": np.round(rho / rho.max(), 3).tolist(),
        "edges": [[remap[int(a)], remap[int(b)]] for a, b in G.edges() if int(a) in remap and int(b) in remap],
        "primitives": prims,
    }


def main() -> None:
    behav, _ = split_segments_by_kind(load_behavior_segments(CSV))
    data = {s: load_space(s) for s in SPACES}
    picks = choose_clips(data, behav)
    out = []
    for pair, clip in picks:
        n = len(qpos_rollout(clip))
        print(pair, "clip", clip)
        out.append(
            {
                "clip": clip,
                "pair": list(pair),
                "n_frames": n,
                "labels": frame_labels(behav, clip, n),
                "spaces": {s: space_payload(s, data[s], clip) for s in SPACES},
            }
        )
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
