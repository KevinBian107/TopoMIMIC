"""In-scene ghost rendering: N recoloured copies of the walker in ONE MuJoCo scene.

This is the NEmo-paper construction, not an image-space composite. Every ghost is
a real body, so the render keeps the floor, lighting, shadows and reflections, and
bodies occlude each other correctly. A root-path polyline is drawn on the floor as
thin capsule segments so the trajectory reads even when poses overlap.
"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import mujoco

XML = "topo_mimic/extraction/xml/rodent.xml"
_CACHE: dict = {}


def _kill_lights(body):
    """Each attached copy brings its own lights; N copies wash the scene out.
    Keep only the base scene's lighting (c2a does the same)."""
    for l in body.lights:
        l.active = 0
    for c in body.bodies:
        _kill_lights(c)


def _recolour(body, rgba):
    for g in body.geoms:
        g.rgba = rgba
    for c in body.bodies:
        _recolour(c, rgba)


def build(n_ghosts: int, colors, xml=XML, trace=True, n_trace=48):
    """Compile a scene holding `n_ghosts` recoloured walkers plus a trace strip."""
    key = (n_ghosts, tuple(map(tuple, colors)), trace, n_trace)
    if key in _CACHE:
        return _CACHE[key]
    base = mujoco.MjSpec.from_file(xml)
    # The stock floor texture is a saturated blue that competes with the ghosts.
    # Flatten it to a neutral grey checker so colour carries the signal.
    for tex in base.textures:
        if tex.name == "grid":
            tex.rgb1 = [0.46, 0.52, 0.60]
            tex.rgb2 = [0.64, 0.70, 0.78]
    for mat in base.materials:
        if mat.name == "grid":
            mat.rgba = [1.0, 1.0, 1.0, 1.0]
            mat.reflectance = 0.08
    orig = [b for b in base.worldbody.bodies if b.name == "walker"][0]
    _recolour(orig, list(colors[0]))
    # The stock light is a tracking light on the torso, so it would follow ghost 0
    # and change the lighting from panel to panel. Replace it with a fixed key
    # light so every panel is lit identically.
    _kill_lights(orig)
    key = base.worldbody.add_light()
    key.pos = [0.0, -0.6, 1.1]
    key.dir = [0.0, 0.45, -1.0]
    key.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    key.castshadow = True
    key.diffuse = [0.75, 0.75, 0.75]
    key.specular = [0.12, 0.12, 0.12]
    fill = base.worldbody.add_light()
    fill.pos = [0.7, 0.5, 0.8]
    fill.dir = [-0.5, -0.4, -1.0]
    fill.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    fill.castshadow = False
    fill.diffuse = [0.32, 0.32, 0.36]
    fill.specular = [0.0, 0.0, 0.0]
    for gi in range(1, n_ghosts):
        g = mujoco.MjSpec.from_file(xml)
        gw = [b for b in g.worldbody.bodies if b.name == "walker"][0]
        _recolour(gw, list(colors[gi]))
        _kill_lights(gw)
        base.worldbody.add_frame(pos=(0, 0, 0)).attach_body(gw, "", f"-g{gi}")
    for i in range(n_ghosts):
        mk = base.worldbody.add_site()
        mk.name = f"mark{i}"
        mk.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        mk.size = [0.020, 0.0012, 0]
        mk.rgba = [0, 0, 0, 0]
        mk.pos = [0, 0, -5]
    if trace:
        for i in range(n_trace):
            s = base.worldbody.add_site()
            s.name = f"trace{i}"
            s.type = mujoco.mjtGeom.mjGEOM_SPHERE
            s.size = [0.011, 0, 0]
            s.rgba = [0.15, 0.15, 0.15, 0.0]
            s.pos = [0, 0, -5]
    m = base.compile()
    _CACHE[key] = (m, mujoco.MjData(m), m.nq // n_ghosts)
    return _CACHE[key]


def render(qposes, frames, colors, *, width=1600, height=1400,
           azimuth=125.0, elevation=-32.0, pad=2.6, trace=True, n_trace=48,
           cam_dist=None, trace_colors=None, mark_colors=None):
    n = len(frames)
    m, d, nq1 = build(n, colors, trace=trace, n_trace=n_trace)
    d.qpos[:] = 0.0
    P = qposes[frames]
    for i in range(n):
        d.qpos[i * nq1:(i + 1) * nq1] = P[i]

    if trace:
        lo, hi = frames[0], frames[-1]
        idx = np.linspace(lo, hi, n_trace).astype(int)
        self_idx = idx  # exposed for callers that colour by behaviour
        path = qposes[idx][:, :3]
        for i in range(n_trace):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"trace{i}")
            if sid < 0:
                continue
            m.site_pos[sid] = [path[i, 0], path[i, 1], 0.008]
            m.site_rgba[sid] = (trace_colors[i] if trace_colors is not None
                                else [0.85, 0.30, 0.05, 1.0])
    for i in range(n):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"mark{i}")
        if sid >= 0:
            m.site_pos[sid] = [P[i, 0], P[i, 1], 0.0016]
            m.site_rgba[sid] = (mark_colors[i] if mark_colors is not None
                                else [0, 0, 0, 0])   # invisible unless asked for
    mujoco.mj_forward(m, d)

    xy = P[:, :3]
    ctr = xy.mean(0)
    span = float(np.linalg.norm(xy[:, :2] - xy[:, :2].mean(0), axis=1).max())
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = cam_dist if cam_dist else max(0.62, span * 2.0 * pad)
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.lookat[:] = [ctr[0], ctr[1], 0.05]

    r = mujoco.Renderer(m, height=height, width=width)
    r.update_scene(d, camera=cam)
    img = r.render().copy()
    r.close()
    return img
