"""Figures 10-13 (discovered cluster examples) as videos for the website.

Same clusters, members, poses, tint and shared camera as ``build_cluster_examples``
(seed 0). Each member video plays from its first to its last frame; DM primitives
store Takens node indices, so their window is extended by the (m-1)*tau footprint of
the last node. Windows longer than ``MAX_FRAMES`` are played with a frame stride.

    python -m demos.website.render_clusters [METHOD ...]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import topo_mimic.figure_generation.build_cluster_examples as BCE
from demos.website.ghost_video import GhostClip, GhostVideoRenderer, VideoStyle, qpos_rollout, write_mp4
from demos.website.render_lexicon import frame_behaviors
from topo_mimic.analysis.feature_analysis.data_loading import center_offset_for_method

SITE = Path("TopoMimic-website")
MEDIA = SITE / "public/media/clusters"
MANIFEST = SITE / "src/data/clusters.json"
METHODS = {
    "w2_dmp_qpos_kmeans": "DMP-Qpos",
    "w2_dmc_int_kmeans": "DMC-Int",
    "kpms_keypoint": "KPMS-KP",
    "mapper_qpos": "Mapper-Qpos",
}
MAX_FRAMES = 120
NP = 6


def pose_frames(fr: np.ndarray) -> list[int]:
    idx = np.linspace(0, len(fr) - 1, min(NP, len(fr))).astype(int)
    return [int(fr[i]) for i in idx]


def main() -> None:
    methods = sys.argv[1:] or list(METHODS)
    BCE.load_qpos = lambda clip_id, _roll: qpos_rollout(int(clip_id))
    style = VideoStyle()
    rend = GhostVideoRenderer(NP, style)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

    for method in methods:
        pairs, behav = BCE.select_cluster_examples(method, n_panels=6, seed=0)
        picks = [(c, f) for _, ex, _, _ in pairs for c, f in ex]
        spans = []
        for c, f in picks:
            xy = qpos_rollout(c)[pose_frames(f)][:, :2]
            spans.append(float(np.linalg.norm(xy - xy.mean(0), axis=1).max()))
        cam = max(0.68, float(np.percentile(spans, 90)) * 2.0 * 2.45)
        foot = 2 * center_offset_for_method(method)

        clusters = []
        for k, (lab, ex, cid, size) in enumerate(pairs):
            members = []
            for j, (c, f) in enumerate(ex):
                n_clip = len(qpos_rollout(c))
                lo, hi = int(f[0]), min(n_clip, int(f[-1]) + foot + 1)
                stride = max(1, math.ceil((hi - lo) / MAX_FRAMES))
                clip = GhostClip(
                    clip_id=c,
                    window=(lo, hi),
                    pose_frames=pose_frames(f),
                    behaviors=frame_behaviors(behav, c, lo, hi),
                    cam_dist=cam,
                    stride=stride,
                )
                name = f"{method}/c{k:02d}_m{j}"
                write_mp4(rend.frames(clip), MEDIA / f"{name}.mp4", style)
                print("wrote", name, f"frames {lo}..{hi} stride {stride}")
                members.append(
                    {
                        "clip": c,
                        "frames": [lo, hi],
                        "stride": stride,
                        "src": f"media/clusters/{name}.mp4",
                        "poster": f"media/clusters/{name}.jpg",
                    }
                )
            clusters.append({"label": lab, "cluster": int(cid), "size": int(size), "members": members})
        manifest[method] = {"name": METHODS[method], "seed": 0, "clusters": clusters}
        MANIFEST.write_text(json.dumps(manifest, indent=1))
    rend.close()
    print("wrote", MANIFEST)


if __name__ == "__main__":
    main()
