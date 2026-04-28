#!/usr/bin/env python3
"""GT vs Discovered transition comparison videos.

For a single behavior pair (e.g., Immobile <-> Rear), pick two GT example
clips and two clips that DM-WKC discovered for that same behavior pair, render
each as a side-view MJX rollout video, and composite them into a single 2x2
grid MP4: top row = ground truth, bottom row = discovered. We skip behavior
pairs where the chosen method discovered nothing.

Usage:
    python -m demos.demo_gt_vs_discovered --method w2_dmp_qpos_kmeans --seed 0 \
        --behavior Immobile_Rear --output-dir outputs/demos/gt_vs_discovered

The default behavior list is the six unordered pairs over {Walk, Immobile, Turn, Rear};
pass --behavior all to render one MP4 per pair.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from topo_mimic.analysis.feature_analysis.ground_truth import (  # noqa: E402
    find_all_transitions, TransitionSegment,
)
from topo_mimic.analysis.feature_analysis.data_loading import (  # noqa: E402
    load_cluster_dict, load_labels_yaml,
)
from topo_mimic.analysis.method_analysis.property.theta_filmstrip import (  # noqa: E402
    compute_filmstrip_camera, render_fixed_camera_frames,
)
from topo_mimic.extraction.utils.load_utils import load_rollout_from_source  # noqa: E402

DEFAULT_PAIRS = [
    ("Immobile", "Rear"), ("Immobile", "Turn"), ("Immobile", "Walk"),
    ("Rear", "Turn"), ("Rear", "Walk"), ("Turn", "Walk"),
]


@dataclass
class Clip:
    label: str       # human-readable transition label
    clip_id: int
    frames: list[int]


def _canonical(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted([a, b]))


def gt_clips_for_pair(
    csv_path: str,
    pair: tuple[str, str],
    short_threshold: int = 10,
    max_gap: int = 10,
    min_total_frames: int = 30,
) -> list[Clip]:
    """All directed GT transition segments whose canonical pair == `pair`."""
    segs = find_all_transitions(csv_path, short_threshold=short_threshold, max_gap=max_gap)
    out: list[Clip] = []
    for s in segs:
        if _canonical(s.from_behavior, s.to_behavior) != pair:
            continue
        total = (s.from_end - s.from_start + 1) + (s.to_end - s.to_start + 1)
        if total < min_total_frames:
            continue
        frames = list(range(s.from_start, s.to_end + 1))
        label = f"GT  {s.from_behavior} {chr(0x2192)} {s.to_behavior}"
        out.append(Clip(label=label, clip_id=s.clip_id, frames=frames))
    return out


def discovered_clips_for_pair(
    method_dir: str, seed: int, pair: tuple[str, str],
) -> list[Clip]:
    cluster_dict = load_cluster_dict(method_dir, seed)
    labels, turns = load_labels_yaml(method_dir, seed)
    out: list[Clip] = []
    KNOWN_DIRS = {"Left", "Right", "Straight"}
    for cid, lbl in (labels or {}).items():
        if not lbl or lbl.startswith("C"):
            continue
        # Strip trailing direction (Left/Right/Straight) if present.
        body = lbl.strip()
        parts = body.rsplit(" ", 1)
        if len(parts) == 2 and parts[1] in KNOWN_DIRS:
            body = parts[0].strip()
        # Parse the cluster's transition label, e.g. "Immobile -> Rear"
        if " -> " not in body:
            continue
        a, b = [x.strip() for x in body.split(" -> ")]
        if _canonical(a, b) != pair:
            continue
        for clip_id, frames in cluster_dict.get(cid, []):
            f = np.sort(np.asarray(frames).flatten()).tolist()
            if len(f) < 2:
                continue
            out.append(Clip(label=f"DMC  {a} {chr(0x2192)} {b}",
                            clip_id=int(clip_id), frames=f))
    return out


def render_clip_video(
    clip: Clip,
    qposes: np.ndarray,
    xml_path: str,
    n_frames: int,
    render_h: int,
    render_w: int,
) -> list[np.ndarray]:
    """Sample `n_frames` from clip.frames uniformly and render each via MJX."""
    valid = [f for f in clip.frames if 0 <= f < qposes.shape[0]]
    if len(valid) < 2:
        return []
    if len(valid) > n_frames:
        idx = np.linspace(0, len(valid) - 1, n_frames).astype(int)
        chosen = [valid[i] for i in idx]
    else:
        chosen = valid
    lookat, dist, az = compute_filmstrip_camera(
        qposes, chosen, elevation=-25.0, azimuth=90.0, distance_margin=2.2,
    )
    frames = render_fixed_camera_frames(
        qposes, chosen,
        camera_lookat=lookat, camera_distance=dist,
        camera_elevation=-25.0, camera_azimuth=az,
        render_height=render_h, render_width=render_w, xml_path=xml_path,
    )
    return frames


def _label_strip(text: str, width: int, height: int) -> np.ndarray:
    """White strip with centered text label."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(14, height // 2))
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), text, fill=(40, 40, 40), font=font)
    return np.asarray(img)


def composite_grid(
    gt_videos: list[list[np.ndarray]],
    disc_videos: list[list[np.ndarray]],
    titles_gt: list[str],
    titles_disc: list[str],
    label_h: int = 36,
) -> list[np.ndarray]:
    """Single row: GT | GT | ... | Disc | Disc ... (left to right)."""
    all_videos = list(gt_videos) + list(disc_videos)
    all_titles = list(titles_gt) + list(titles_disc)
    n_frames = max(len(v) for v in all_videos)
    fh, fw = all_videos[0][0].shape[:2]
    out = []
    for fi in range(n_frames):
        cells = []
        for v, t in zip(all_videos, all_titles):
            f = v[min(fi, len(v) - 1)]
            cells.append(np.vstack([_label_strip(t, fw, label_h), f]))
        out.append(np.hstack(cells))
    return out


def _first_method_with_pair(
    method_candidates: list[str], experiment_dir: str, seed: int,
    pair: tuple[str, str],
) -> tuple[str, list[Clip]] | None:
    for m in method_candidates:
        method_dir = os.path.join(experiment_dir, m)
        if not os.path.isdir(method_dir):
            continue
        clips = discovered_clips_for_pair(method_dir, seed, pair)
        if clips:
            return m, clips
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dmc-methods", type=str, nargs="+",
                   default=["w2_dmc_int_kmeans", "w2_dmc_qpos_kmeans", "w2_dmc_ekp_kmeans"],
                   help="DMC variants to draw from, in priority order.")
    p.add_argument("--dmp-methods", type=str, nargs="+",
                   default=["w2_dmp_qpos_kmeans", "w2_dmp_int_kmeans", "w2_dmp_ekp_kmeans"],
                   help="DMP variants to draw from, in priority order.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--experiment-dir", type=str, default="outputs/experiment/transitions")
    p.add_argument("--data-path", type=str, default="data/rodent_ar.h5")
    p.add_argument("--xml-path", type=str,
                   default=str(REPO / "topo_mimic/extraction/xml/rodent.xml"))
    p.add_argument("--csv", type=str, default="behavior_labels/rodent_labels.csv")
    p.add_argument("--output-dir", type=str, default="outputs/demos/gt_vs_discovered")
    p.add_argument("--behavior", type=str, default="all",
                   help="Pair as A_B (e.g. Immobile_Rear), or 'all'.")
    p.add_argument("--n-gt", type=int, default=2,
                   help="Number of GT examples shown to the left.")
    p.add_argument("--n-frames", type=int, default=60,
                   help="Frames per clip in the rendered video.")
    p.add_argument("--render-h", type=int, default=360)
    p.add_argument("--render-w", type=int, default=480)
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()

    csv_path = str(REPO / args.csv) if not os.path.isabs(args.csv) else args.csv
    data_path = str(REPO / args.data_path) if not os.path.isabs(args.data_path) else args.data_path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.behavior == "all":
        pairs = DEFAULT_PAIRS
    else:
        a, b = args.behavior.split("_")
        pairs = [tuple(sorted([a, b]))]

    qpos_cache: dict[int, np.ndarray] = {}
    def get_qpos(clip_id: int) -> np.ndarray | None:
        if clip_id not in qpos_cache:
            try:
                ro = load_rollout_from_source(clip_id, data_path)
                qpos_cache[clip_id] = ro["qpos"]
            except Exception as e:
                print(f"   skip clip {clip_id}: {e}")
                qpos_cache[clip_id] = None
        return qpos_cache[clip_id]

    def render(clips: list[Clip], label_prefix: str) -> tuple[list[list[np.ndarray]], list[str]]:
        vids, titles = [], []
        for c in clips:
            qp = get_qpos(c.clip_id)
            if qp is None:
                continue
            v = render_clip_video(c, qp, args.xml_path,
                                   args.n_frames, args.render_h, args.render_w)
            if v:
                vids.append(v)
                titles.append(f"{label_prefix}  (clip {c.clip_id})")
        return vids, titles

    for pair in pairs:
        gt = gt_clips_for_pair(csv_path, pair)
        dmc_pick = _first_method_with_pair(args.dmc_methods, args.experiment_dir, args.seed, pair)
        dmp_pick = _first_method_with_pair(args.dmp_methods, args.experiment_dir, args.seed, pair)
        print(f"=== {pair[0]}<->{pair[1]}: GT={len(gt)}  "
              f"DMC={dmc_pick[0] if dmc_pick else 'NONE'}  "
              f"DMP={dmp_pick[0] if dmp_pick else 'NONE'} ===")
        if not gt:
            print("   skipping (no GT)"); continue
        if dmc_pick is None or dmp_pick is None:
            print("   skipping (DMC or DMP did not discover this pair)"); continue

        gt_vids, gt_titles = render(gt[: args.n_gt], "GT")
        if len(gt_vids) < args.n_gt:
            while len(gt_vids) < args.n_gt and gt_vids:
                gt_vids.append(gt_vids[-1]); gt_titles.append(gt_titles[-1])

        dmc_vids, dmc_titles = render(dmc_pick[1][:1], f"DMC ({dmc_pick[0].split('_')[2]})")
        dmp_vids, dmp_titles = render(dmp_pick[1][:1], f"DMP ({dmp_pick[0].split('_')[2]})")
        if not gt_vids or not dmc_vids or not dmp_vids:
            print("   skipping (rendering failed)"); continue

        disc_vids = dmc_vids + dmp_vids
        disc_titles = dmc_titles + dmp_titles
        composite = composite_grid(gt_vids, disc_vids, gt_titles, disc_titles)

        # Pad dims to multiple of 16 for codec compatibility.
        H, W = composite[0].shape[:2]
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h or pad_w:
            composite = [np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)),
                                 mode="constant", constant_values=255) for f in composite]

        slug = f"{pair[0]}_{pair[1]}".replace(" ", "")
        out_mp4 = out_dir / f"gt_vs_disc_seed{args.seed}_{slug}.mp4"
        imageio.mimsave(
            out_mp4, composite, fps=args.fps, codec="libx264",
            output_params=["-pix_fmt", "yuv420p", "-crf", "22",
                            "-preset", "slow", "-movflags", "+faststart"],
        )
        print(f"   wrote {out_mp4}  ({os.path.getsize(out_mp4)//1024} kB)")


if __name__ == "__main__":
    main()
