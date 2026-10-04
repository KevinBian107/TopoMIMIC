"""Figure 5 (rat behavior lexicon) as videos for the website.

Same instances, poses, tint and shared camera as ``build_lexicon``; each video plays the
window at half speed, drops the six paper poses as it passes them and holds on the panel.

    python -m demos.website.render_lexicon [--only N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import topo_mimic.figure_generation.build_lexicon as build_lexicon
from demos.website.ghost_video import GhostClip, GhostVideoRenderer, VideoStyle, qpos_rollout, write_mp4
from topo_mimic.experiment.auto_label import load_behavior_segments, split_segments_by_kind
from topo_mimic.figure_generation.build_lexicon import (
    CSV,
    select_lexicon_examples,
    shared_cam_dist,
)

SITE = Path("TopoMimic-website")
MEDIA = SITE / "public/media/lexicon"
MANIFEST = SITE / "src/data/lexicon.json"


def frame_behaviors(segs: dict, clip_id: int, lo: int, hi: int) -> list[str]:
    out = []
    for f in range(lo, hi):
        out.append(next((s.behavior for s in segs.get(int(clip_id), []) if s.start <= f < s.end), ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()

    build_lexicon.load_qpos = lambda clip_id, _roll: qpos_rollout(int(clip_id))
    style = VideoStyle()
    picks = select_lexicon_examples()
    cam = shared_cam_dist(picks)
    behav, _ = split_segments_by_kind(load_behavior_segments(CSV))
    rend = GhostVideoRenderer(6, style)

    items = []
    for i, p in enumerate(picks):
        lo, hi = int(p.frames[0]), int(p.frames[-1]) + 1
        clip = GhostClip(
            clip_id=p.clip_id,
            window=(lo, hi),
            pose_frames=[int(x) for x in p.frames],
            behaviors=frame_behaviors(behav, p.clip_id, lo, hi),
            cam_dist=cam,
        )
        name = f"{i:02d}_{p.a.lower()}_{p.b.lower()}"
        if not args.manifest_only and (args.only is None or args.only == i):
            write_mp4(rend.frames(clip), MEDIA / f"{name}.mp4", style)
            print("wrote", name)
        items.append(
            {
                "from": p.a,
                "to": p.b,
                "n": p.n,
                "clip": p.clip_id,
                "frames": [lo, hi],
                "turn": getattr(getattr(p.transition, "turn", None), "name", None),
                "src": f"media/lexicon/{name}.mp4",
                "poster": f"media/lexicon/{name}.jpg",
            }
        )
    rend.close()
    if args.only is None:
        MANIFEST.write_text(json.dumps(items, indent=1))
        print("wrote", MANIFEST)


if __name__ == "__main__":
    main()
