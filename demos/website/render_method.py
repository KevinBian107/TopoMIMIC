"""Body-space videos for the website's "How the method works" explorer.

Every step of the pipeline is shown on the animal rather than in the embedding:
  overview  - one clip, real time
  takens    - the live pose at t with ghosts at t - tau and t - 2 tau (one Takens point)
  density   - the body tinted by the clip's k-NN inverse density rho (dense = sustained)
  primitives- one ghost video per stored intention DM-cycle / DM-path of the clip
  w2        - three DMC-Int primitives: two from one cluster, one from another

    python -m demos.website.render_method [CLIP_ID]
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

from demos.website.ghost_video import GhostClip, GhostVideoRenderer, VideoStyle, qpos_rollout, write_mp4
from demos.website.render_lexicon import frame_behaviors
from topo_mimic.analysis.feature_analysis.data_loading import center_offset_for_space, load_labels_yaml
from topo_mimic.experiment.auto_label import load_behavior_segments, split_segments_by_kind
from topo_mimic.extraction.skeletonization.remap import parse_swr_remap

CSV = "behavior_labels/rodent_labels.csv"
SITE = Path("TopoMimic-website")
MEDIA = SITE / "public/media/method"
OUT = SITE / "src/data/method.json"
W2_PAIRS = Path("outputs/demos/website/w2_pairs.json")
DMC_INT = Path("outputs/experiment/transitions/w2_dmc_int_kmeans")
SPACE = "intention"
TAU = 6
TAKENS_WINDOW = (176, 320)
CAM_DIST = 0.62
TRACK_SIGMA = 20.0
DENSE_RGB = np.array([0.22, 0.42, 0.85])
SPARSE_RGB = np.array([0.92, 0.30, 0.18])
MAX_FRAMES = 120


def rho_per_frame(clip: int, n_frames: int) -> np.ndarray:
    """Inverse density of the Takens node centred on each frame, scaled to [0, 1] within the clip."""
    rho = np.loadtxt(f"outputs/extraction/dm_pipeline_{SPACE}/clips/clip_{clip}/weights.txt")
    off = center_offset_for_space(SPACE)
    idx = np.clip(np.arange(n_frames) - off, 0, len(rho) - 1)
    r = rho[idx]
    lo, hi = np.percentile(r, 5), np.percentile(r, 95)
    return np.clip((r - lo) / (hi - lo), 0, 1)


def speed_per_frame(q: np.ndarray, w: int = 13) -> np.ndarray:
    """Joint-angle speed, averaged over a Takens-length window and scaled to [0, 1] within the clip."""
    v = np.r_[0.0, np.linalg.norm(np.diff(q[:, 7:], axis=0), axis=1)]
    v = np.convolve(v, np.ones(w) / w, mode="same")
    lo, hi = np.percentile(v, 5), np.percentile(v, 95)
    return np.clip((v - lo) / (hi - lo), 0, 1)


def skeleton_frames(clip: int) -> list[int]:
    d = Path(f"outputs/extraction/dm_pipeline_{SPACE}/clips/clip_{clip}")
    G = pickle.load(open(f"outputs/extraction/dm_pipeline_{SPACE}/all_G_sm.pkl", "rb"))[clip]
    remap = parse_swr_remap(str(d))
    off = center_offset_for_space(SPACE)
    return sorted({remap[int(v)] + off for v in G.nodes() if G.degree(v) > 0 and int(v) in remap})


def pose_frames(fr: np.ndarray, n: int = 6) -> list[int]:
    idx = np.linspace(0, len(fr) - 1, min(n, len(fr))).astype(int)
    return [int(fr[i]) for i in idx]


def main() -> None:
    clip = int(sys.argv[1]) if len(sys.argv) > 1 else 447
    q = qpos_rollout(clip)
    n = len(q)
    behav, _ = split_segments_by_kind(load_behavior_segments(CSV))
    labels = frame_behaviors(behav, clip, 0, n)
    cam = CAM_DIST
    out: dict = {"clip": clip, "n_frames": n, "labels": labels, "tau": TAU, "offset": center_offset_for_space(SPACE)}

    rt = VideoStyle(fps=30)
    live = GhostVideoRenderer(0, rt)
    write_mp4(live.trailing(clip, 0, n, cam, track_sigma=TRACK_SIGMA), MEDIA / "overview.mp4", rt)
    out["overview"] = {"src": "media/method/overview.mp4", "poster": "media/method/overview.jpg", "fps": 30, "start": 0}

    rho = rho_per_frame(clip, n)
    tint = lambda t: list((1 - rho[t]) * DENSE_RGB + rho[t] * SPARSE_RGB) + [1.0]  # noqa: E731
    write_mp4(live.trailing(clip, 0, n, cam, live_color=tint, track_sigma=TRACK_SIGMA), MEDIA / "density.mp4", rt)
    out["density"] = {"src": "media/method/density.mp4", "poster": "media/method/density.jpg", "fps": 30, "start": 0,
                      "rho": np.round(rho, 3).tolist(), "speed": np.round(speed_per_frame(q), 3).tolist(),
                      "skeleton": skeleton_frames(clip)}
    live.close()

    half = VideoStyle(fps=15)
    tk = GhostVideoRenderer(2, half)
    lo, hi = TAKENS_WINDOW
    write_mp4(tk.trailing(clip, lo, hi, cam, lags=(TAU, 2 * TAU), lag_colors=([0.62, 0.62, 0.66, 0.97], [0.80, 0.80, 0.84, 0.97]), track_sigma=TRACK_SIGMA),
              MEDIA / "takens.mp4", half)
    tk.close()
    out["takens"] = {"src": "media/method/takens.mp4", "poster": "media/method/takens.jpg", "fps": 15, "start": lo, "window": [lo, hi]}

    d = Path(f"outputs/extraction/dm_pipeline_{SPACE}")
    cyc = pickle.load(open(d / "all_cycle_dm.pkl", "rb"))[clip]
    pth = pickle.load(open(d / "all_path_dm.pkl", "rb"))[clip]
    prims = sorted([{"type": "cycle", "nodes": list(map(int, c))} for c in cyc] + [{"type": "path", "nodes": list(map(int, p))} for p in pth],
                   key=lambda p: min(p["nodes"]))
    gr = GhostVideoRenderer(6, half)
    out["primitives"] = []
    for k, p in enumerate(prims):
        fr = np.sort(np.asarray(p["nodes"]))
        w_lo, w_hi = int(fr[0]), min(n, int(fr[-1]) + 2 * TAU + 1)
        gc = GhostClip(clip, (w_lo, w_hi), pose_frames(fr), frame_behaviors(behav, clip, w_lo, w_hi),
                       stride=max(1, math.ceil((w_hi - w_lo) / MAX_FRAMES)))
        write_mp4(gr.frames(gc), MEDIA / f"prim_{k}.mp4", half)
        out["primitives"].append({"type": p["type"], "frames": [w_lo, w_hi], "stride": gc.stride,
                                  "src": f"media/method/prim_{k}.mp4", "poster": f"media/method/prim_{k}.jpg"})
        print("primitive", k, p["type"], w_lo, w_hi)

    res = pickle.load((DMC_INT / "results.pkl").open("rb"))[0]
    cd = res["cluster_dict"]
    names, _ = load_labels_yaml(str(DMC_INT), 0)
    trans = sorted((cid for cid, nm in names.items() if "->" in nm), key=lambda c: -len(cd[c]))
    ca, cb = trans[0], next(c for c in trans[1:] if names[c].split("->")[0] != names[trans[0]].split("->")[0])
    w2 = json.loads(W2_PAIRS.read_text())
    members = [("anchor", ca, cd[ca][0]), ("same", ca, cd[ca][1]), ("different", cb, cd[cb][0])]
    picks = [(role, cid, int(c), np.sort(np.asarray(f).ravel())) for role, cid, (c, f) in members]
    spans = [float(np.linalg.norm(qpos_rollout(c)[pose_frames(f)][:, :2] - qpos_rollout(c)[pose_frames(f)][:, :2].mean(0), axis=1).max()) for *_, c, f in picks]
    wcam = max(0.68, max(spans) * 2.0 * 2.45)
    out["w2"] = []
    for role, cid, c, fr in picks:
        nc = len(qpos_rollout(c))
        w_lo, w_hi = int(fr[0]), min(nc, int(fr[-1]) + 2 * TAU + 1)
        gc = GhostClip(c, (w_lo, w_hi), pose_frames(fr), frame_behaviors(behav, c, w_lo, w_hi), cam_dist=wcam,
                       stride=max(1, math.ceil((w_hi - w_lo) / MAX_FRAMES)))
        write_mp4(gr.frames(gc), MEDIA / f"w2_{role}.mp4", half)
        out["w2"].append({"role": role, "cluster": int(cid), "label": names[cid], "clip": c, "frames": [w_lo, w_hi],
                          "w2sq": None if role == "anchor" else w2[role],
                          "src": f"media/method/w2_{role}.mp4", "poster": f"media/method/w2_{role}.jpg"})
        print("w2", role, names[cid], c, w_lo, w_hi)
    gr.close()

    # The first rear yields a DM-cycle and a DM-path over identical frames (a small noise loop);
    # the website keeps their timeline bars as an example but does not show their videos.
    spans = [tuple(p["frames"]) for p in out["primitives"]]
    out["noise"] = [i for i, s in enumerate(spans) if spans.count(s) > 1]
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
