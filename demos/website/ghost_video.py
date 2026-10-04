"""Video counterpart of the paper's in-scene ghost panels (Figures 5, 7, 10-13).

A live body (amber) plays through a frame window while the paper's six sampled poses are
dropped into the scene as it passes them, so the final frame is the static panel. Ghosts
stay opaque: any geom alpha below ~0.9 darkens the whole scene in MuJoCo's reflection pass.
Scene, lights, tint and camera are those of ``topo_mimic.figure_generation.ghost_scene``.

Run with the EGL loader on the path:
    LD_LIBRARY_PATH=<track-mjx>/.venv/lib/gl_stubs MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python ...
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import h5py
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from topo_mimic.figure_generation import ghost_scene as G

ROLL = "data/rodent_ar.h5"
BEHAVIOR_RGB: dict[str, tuple[int, int, int]] = {
    "Immobile": (140, 140, 140),
    "Rear": (58, 163, 86),
    "Walk": (77, 128, 217),
    "Turn": (217, 115, 31),
}
UNLABELED_RGB = (210, 210, 210)
HIDDEN_Z = -3.0


@dataclass
class VideoStyle:
    width: int = 1088
    height: int = 960
    fps: int = 15
    hold_s: float = 1.4
    crf: int = 23
    azimuth: float = 125.0
    elevation: float = -32.0
    ghost_alpha: float = 0.97
    live_rgba: tuple[float, float, float, float] = (0.93, 0.62, 0.32, 1.0)
    timeline_px: int = 14


@dataclass
class GhostClip:
    """One video: a frame window of one clip with the paper's pose frames."""

    clip_id: int
    window: tuple[int, int]
    pose_frames: list[int]
    behaviors: list[str] = field(default_factory=list)
    cam_dist: float | None = None
    stride: int = 1


@lru_cache(maxsize=None)
def qpos_rollout(clip_id: int) -> np.ndarray:
    with h5py.File(ROLL, "r") as f:
        return np.asarray(f["qpos_rollout"][clip_id])


def paper_tint(i: int, n: int) -> float:
    return 0.72 - 0.34 * (i / max(1, n - 1))


class GhostVideoRenderer:
    """Holds one compiled scene of ``n_ghosts`` poses plus one live body."""

    def __init__(self, n_ghosts: int, style: VideoStyle):
        self.style = style
        self.n = n_ghosts
        colors = [[0.6, 0.6, 0.6, 1.0]] * (n_ghosts + 1)
        self.m, self.d, self.nq1 = G.build(n_ghosts + 1, colors, trace=False)
        self.slot_geoms = self._geoms_per_slot()
        self.renderer = mujoco.Renderer(self.m, height=style.height, width=style.width)

    def _geoms_per_slot(self) -> list[np.ndarray]:
        slots: list[list[int]] = [[] for _ in range(self.n + 1)]
        for gid in range(self.m.ngeom):
            root = self.m.body_rootid[self.m.geom_bodyid[gid]]
            rname = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, root) or ""
            if rname == "walker":
                slots[0].append(gid)
            elif rname.startswith("walker-g"):
                slots[int(rname.rsplit("-g", 1)[1])].append(gid)
        return [np.asarray(s, dtype=int) for s in slots]

    def _set_slot(self, slot: int, q: np.ndarray | None, rgba: list[float]) -> None:
        sl = slice(slot * self.nq1, (slot + 1) * self.nq1)
        if q is None:
            self.d.qpos[sl] = 0.0
            self.d.qpos[sl.start + 2] = HIDDEN_Z
            self.d.qpos[sl.start + 3] = 1.0
        else:
            self.d.qpos[sl] = q
        self.m.geom_rgba[self.slot_geoms[slot]] = rgba

    def _camera(self, lookat: np.ndarray, dist: float) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = dist
        cam.azimuth = self.style.azimuth
        cam.elevation = self.style.elevation
        cam.lookat[:] = [lookat[0], lookat[1], 0.05]
        return cam

    def frames(self, clip: GhostClip):
        """Yield RGB frames: playback of the window, then a hold on the full panel."""
        st = self.style
        q = qpos_rollout(clip.clip_id)
        lo, hi = clip.window
        hi = min(hi, len(q))
        poses = list(clip.pose_frames)
        P = q[poses]
        ctr = P[:, :2].mean(0)
        span = float(np.linalg.norm(P[:, :2] - ctr, axis=1).max())
        dist = clip.cam_dist or max(0.62, span * 2.0 * 2.6)
        cam = self._camera(ctr, dist)
        live = self.n

        for t in range(lo, hi, clip.stride):
            for i, f in enumerate(poses):
                g = paper_tint(i, len(poses))
                shown = f <= t
                self._set_slot(i, q[f] if shown else None, [g, g, g * 1.02, st.ghost_alpha])
            self._set_slot(live, q[t], list(st.live_rgba))
            yield self._render(cam, clip, lo, hi, t)

        for i, f in enumerate(poses):
            g = paper_tint(i, len(poses))
            self._set_slot(i, q[f], [g, g, g * 1.02, st.ghost_alpha])
        self._set_slot(live, None, [0, 0, 0, 0])
        still = self._render(cam, clip, lo, hi, None)
        for _ in range(int(st.hold_s * st.fps)):
            yield still

    def _render(self, cam, clip: GhostClip, lo: int, hi: int, t: int | None) -> np.ndarray:
        mujoco.mj_forward(self.m, self.d)
        self.renderer.update_scene(self.d, camera=cam)
        img = self.renderer.render().copy()
        if clip.behaviors:
            img = draw_timeline(img, clip, lo, hi, t, self.style.timeline_px)
        return img

    def playback(self, clip_id: int, lo: int, hi: int, cam_dist: float):
        """Live body only, fixed camera over the clip's root path (no ghosts, no strip)."""
        q = qpos_rollout(clip_id)
        ctr = q[lo:hi, :2].mean(0)
        cam = self._camera(ctr, cam_dist)
        for i in range(self.n):
            self._set_slot(i, None, [0, 0, 0, 0])
        for t in range(lo, hi):
            self._set_slot(self.n, q[t], list(self.style.live_rgba))
            mujoco.mj_forward(self.m, self.d)
            self.renderer.update_scene(self.d, camera=cam)
            yield self.renderer.render().copy()

    def trailing(self, clip_id: int, lo: int, hi: int, cam_dist: float, lags=(), live_color=None, lag_colors=None, track_sigma=0.0):
        """Live body at t, optionally recoloured per frame, plus ghosts at t - lag for each lag.

        With track_sigma > 0 the camera follows the Gaussian-smoothed root path (sigma in frames)."""
        q = qpos_rollout(clip_id)
        ctr = q[lo:hi, :2].mean(0)
        cam = self._camera(ctr, cam_dist)
        if track_sigma > 0:
            from scipy.ndimage import gaussian_filter1d

            path = gaussian_filter1d(q[:, :2].astype(float), track_sigma, axis=0, mode="nearest")
        for i in range(self.n):
            self._set_slot(i, None, [0, 0, 0, 0])
        for t in range(lo, hi):
            for i, lag in enumerate(lags):
                self._set_slot(i, q[max(0, t - lag)], list(lag_colors[i]))
            rgba = list(live_color(t)) if live_color else list(self.style.live_rgba)
            self._set_slot(self.n, q[t], rgba)
            if track_sigma > 0:
                cam.lookat[:2] = path[t]
            mujoco.mj_forward(self.m, self.d)
            self.renderer.update_scene(self.d, camera=cam)
            yield self.renderer.render().copy()

    def poster(self, clip: GhostClip) -> np.ndarray:
        *_, last = self.frames(clip)
        return last

    def close(self) -> None:
        self.renderer.close()


def draw_timeline(img: np.ndarray, clip: GhostClip, lo: int, hi: int, t: int | None, px: int) -> np.ndarray:
    """Ground-truth behavior strip along the bottom edge with pose ticks and a playhead."""
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    W, H = im.size
    n = hi - lo
    y0 = H - px
    for j in range(n):
        b = clip.behaviors[j] if j < len(clip.behaviors) else ""
        x0, x1 = int(j * W / n), int((j + 1) * W / n)
        dr.rectangle([x0, y0, x1, H], fill=BEHAVIOR_RGB.get(b, UNLABELED_RGB))
    for f in clip.pose_frames:
        x = int((f - lo + 0.5) * W / n)
        dr.rectangle([x - 1, y0 - 6, x + 1, y0], fill=(40, 40, 40))
    if t is not None:
        x = int((t - lo + 0.5) * W / n)
        dr.rectangle([x - 2, y0 - 10, x + 2, H], fill=(255, 255, 255), outline=(20, 20, 20))
    return np.asarray(im)


def write_mp4(frames, path: Path, style: VideoStyle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = imageio_ffmpeg.write_frames(
        str(path),
        (style.width, style.height),
        fps=style.fps,
        codec="libx264",
        macro_block_size=16,
        output_params=["-crf", str(style.crf), "-preset", "slow", "-movflags", "+faststart"],
    )
    w.send(None)
    last = None
    for fr in frames:
        w.send(np.ascontiguousarray(fr))
        last = fr
    w.close()
    if last is not None:
        Image.fromarray(last).save(path.with_suffix(".jpg"), quality=88)
