#!/usr/bin/env python3
"""
Figure 1E: Increasing Complexity Arrow (Walker → Rodent)
=========================================================
Produces a side-by-side chronophotography figure with a walker filmstrip
on the left, an arrow in the middle, and one or two rodent filmstrips on
the right, illustrating increasing morphological complexity.

Usage:
    python demos/illustrative/demo_bodies.py
    python demos/illustrative/demo_bodies.py --config configs/demos/illustrative/demo_bodies.yaml
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import math
import argparse
import yaml
import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

from demos.property.demo_theta_filmstrip import (
    FilmstripConfig,
    _load_mujoco_model,
)
from demos.illustrative.demo_transition_filmstrip import load_qpos_walker
from topo_mimic.dmwk.utils.load_utils import load_rollout_from_source


@dataclass
class PanelSpec:
    dataset: str = "walker"
    clip_id: int = 0
    n_frames: int = 500
    sample_every: int = 100
    rollout_path: str = ""
    xml_path: str = ""


@dataclass
class Config:
    output_dir: str = "outputs/complexity_arrow"
    output_format: str = "pdf"
    filmstrip: FilmstripConfig = field(default_factory=FilmstripConfig)
    walker: PanelSpec = field(default_factory=lambda: PanelSpec(
        dataset="walker",
        clip_id=0,
        n_frames=500,
        sample_every=100,
        rollout_path="data/walker_long.h5",
        xml_path="topo_mimic/dmwk/xml/walker.xml",
    ))
    rodent_panels: list[PanelSpec] = field(default_factory=lambda: [
        PanelSpec(
            dataset="rodent",
            clip_id=650,
            n_frames=500,
            sample_every=100,
            rollout_path="data/rodent_ar.h5",
            xml_path="topo_mimic/dmwk/xml/rodent.xml",
        ),
    ])
    dpi: int = 300
    target_strip_height: int = 400
    gap_fraction: float = 0.3
    walker_label: str = "Generated, labeled\nmulti-transition walker"
    rodent_label: str = "Imitated, unlabeled\nmulti-transition rodent"


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    film_cfg = FilmstripConfig(**cfg_dict.get("filmstrip", {}))

    def _parse_panel(d: dict) -> PanelSpec:
        return PanelSpec(**d)

    walker = _parse_panel(cfg_dict.get("walker", {}))

    rodent_panels = []
    for rd in cfg_dict.get("rodent_panels", []):
        rodent_panels.append(_parse_panel(rd))

    return Config(
        output_dir=cfg_dict.get("output_dir", "outputs/complexity_arrow"),
        output_format=cfg_dict.get("output_format", "pdf"),
        filmstrip=film_cfg,
        walker=walker,
        rodent_panels=rodent_panels if rodent_panels else Config().rodent_panels,
        dpi=cfg_dict.get("dpi", 300),
        target_strip_height=cfg_dict.get("target_strip_height", 400),
        gap_fraction=cfg_dict.get("gap_fraction", 0.3),
        walker_label=cfg_dict.get("walker_label", "Generated, labeled\nmulti-transition walker"),
        rodent_label=cfg_dict.get("rodent_label", "Imitated, unlabeled\nmulti-transition rodent"),
    )


def resolve_path(p: str, root: Path) -> str:
    if os.path.isabs(p):
        return p
    return str(root / p)


def load_qpos(spec: PanelSpec) -> np.ndarray:
    if spec.dataset in ("walker", "walker_long"):
        return load_qpos_walker(spec.clip_id, spec.rollout_path)
    else:
        ro = load_rollout_from_source(spec.clip_id, spec.rollout_path)
        return ro["qpos"]


def _render_poses_individually(
    qposes: np.ndarray,
    frame_indices: list[int],
    xml_path: str,
    camera_elevation: float,
    camera_azimuth: float,
    camera_distance: float,
    camera_lookat_fn,
    render_h: int = 720,
    render_w: int = 480,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Render each pose individually centered, return list of (cropped_rgb, body_mask)."""
    import mujoco

    mj_model, mj_data = _load_mujoco_model(xml_path)
    floor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    rgb_renderer = mujoco.Renderer(mj_model, height=render_h, width=render_w)
    seg_renderer = mujoco.Renderer(mj_model, height=render_h, width=render_w)
    seg_renderer.enable_segmentation_rendering()

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = camera_distance
    camera.elevation = camera_elevation
    camera.azimuth = camera_azimuth

    cropped_poses = []
    for fi in frame_indices:
        camera.lookat[:] = camera_lookat_fn(qposes[fi])

        mj_data.qpos[:] = qposes[fi]
        mujoco.mj_forward(mj_model, mj_data)

        rgb_renderer.update_scene(mj_data, camera=camera)
        rgb = rgb_renderer.render().copy()

        seg_renderer.update_scene(mj_data, camera=camera)
        seg = seg_renderer.render()
        geom_ids = seg[:, :, 0].astype(np.int32)
        body_mask = (geom_ids >= 0) & (geom_ids != floor_id)

        rows_with_body = np.any(body_mask, axis=1)
        cols_with_body = np.any(body_mask, axis=0)
        if not np.any(rows_with_body):
            cropped_poses.append((
                np.full((100, 60, 3), 255, dtype=np.uint8),
                np.zeros((100, 60), dtype=bool),
            ))
            continue

        r_min, r_max = np.where(rows_with_body)[0][[0, -1]]
        c_min, c_max = np.where(cols_with_body)[0][[0, -1]]
        pad = 6
        r_min = max(0, r_min - pad)
        r_max = min(render_h - 1, r_max + pad)
        c_min = max(0, c_min - pad)
        c_max = min(render_w - 1, c_max + pad)

        crop_rgb = rgb[r_min:r_max + 1, c_min:c_max + 1]
        crop_mask = body_mask[r_min:r_max + 1, c_min:c_max + 1]
        cropped_poses.append((crop_rgb, crop_mask))

    rgb_renderer.close()
    seg_renderer.close()
    return cropped_poses


def _tile_cropped_poses(
    cropped_poses: list[tuple[np.ndarray, np.ndarray]],
    gap_fraction: float = 0.3,
) -> np.ndarray:
    """Tint and tile cropped poses into a tight strip with no padding."""
    n_poses = len(cropped_poses)
    tint_start = np.array([0.68, 0.80, 0.88])
    tint_end = np.array([0.22, 0.38, 0.50])

    max_h = max(c[0].shape[0] for c in cropped_poses)
    normalized = []
    for crop_rgb, crop_mask in cropped_poses:
        h, w = crop_rgb.shape[:2]
        if h < max_h:
            pad_top = (max_h - h) // 2
            pad_bot = max_h - h - pad_top
            crop_rgb = np.pad(crop_rgb, ((pad_top, pad_bot), (0, 0), (0, 0)),
                              mode='constant', constant_values=255)
            crop_mask = np.pad(crop_mask, ((pad_top, pad_bot), (0, 0)),
                               mode='constant', constant_values=False)
        normalized.append((crop_rgb, crop_mask))

    avg_w = sum(c[0].shape[1] for c in normalized) // n_poses
    gap = max(int(avg_w * gap_fraction), 4)
    total_w = sum(c[0].shape[1] for c in normalized) + gap * (n_poses - 1)

    composite = np.full((max_h, total_w, 3), 255, dtype=np.uint8)
    x_cursor = 0
    for pose_i, (crop_rgb, crop_mask) in enumerate(normalized):
        h, w = crop_rgb.shape[:2]
        t = pose_i / max(n_poses - 1, 1)
        tint = tint_start * (1 - t) + tint_end * t

        body_pixels = crop_rgb[crop_mask].astype(np.float32) / 255.0
        luminance = 1.0 - np.mean(body_pixels, axis=-1, keepdims=True)
        colored = luminance * tint[None, :] + (1.0 - luminance) * 1.0
        tinted_crop = np.full_like(crop_rgb, 255)
        tinted_crop[crop_mask] = np.clip(colored * 255.0, 0, 255).astype(np.uint8)

        composite[:h, x_cursor:x_cursor + w] = tinted_crop
        x_cursor += w + gap

    return composite[:, :x_cursor - gap]


def render_strip(
    spec: PanelSpec,
    filmstrip_cfg: FilmstripConfig,
    gap_fraction: float = 0.3,
) -> np.ndarray:
    qposes = load_qpos(spec)
    n_total = min(spec.n_frames, qposes.shape[0])
    frame_indices = list(range(0, n_total, spec.sample_every))

    is_walker = spec.dataset in ("walker", "walker_long")

    if is_walker:
        fovy_rad = math.radians(45.0)
        view_height = 2.0
        distance = (view_height / 2.0) / math.tan(fovy_rad / 2.0) * 1.05

        def walker_lookat(qpos):
            return [qpos[1], 0.0, 0.65]

        cropped = _render_poses_individually(
            qposes, frame_indices, spec.xml_path,
            camera_elevation=-10.0,
            camera_azimuth=90.0,
            camera_distance=distance,
            camera_lookat_fn=walker_lookat,
            render_h=720, render_w=400,
        )
    else:
        fovy_rad = math.radians(45.0)
        view_height = 0.35
        distance = (view_height / 2.0) / math.tan(fovy_rad / 2.0) * 1.3

        def rodent_lookat(qpos):
            return [qpos[0], qpos[1], qpos[2] + 0.02]

        cropped = _render_poses_individually(
            qposes, frame_indices, spec.xml_path,
            camera_elevation=filmstrip_cfg.camera_elevation,
            camera_azimuth=filmstrip_cfg.camera_azimuth,
            camera_distance=distance,
            camera_lookat_fn=rodent_lookat,
            render_h=800, render_w=800,
        )

    return _tile_cropped_poses(cropped, gap_fraction=gap_fraction)


def create_complexity_figure(
    walker_strip: np.ndarray,
    rodent_strips: list[np.ndarray],
    output_path: str,
    target_height: int = 400,
    dpi: int = 300,
    walker_label: str = "",
    rodent_label: str = "",
) -> None:
    from matplotlib.patches import FancyBboxPatch

    def _resize_to_width(img: np.ndarray, w: int) -> np.ndarray:
        oh, ow = img.shape[:2]
        scale = w / ow
        nh = int(oh * scale)
        return np.array(Image.fromarray(img).resize((w, nh), Image.LANCZOS))

    def _replace_white_with_color(img: np.ndarray, color: tuple, thresh: int = 245) -> np.ndarray:
        out = img.copy()
        white_mask = np.all(img >= thresh, axis=-1)
        out[white_mask] = color
        return out

    box_color = "#e8e8e8"
    box_edge = "#b0b0b0"

    n_rodent = len(rodent_strips)
    n_panels = 1 + n_rodent
    arrow_w_in = 0.6
    panel_w_in = 3.0
    bx_pad_in = 0.04
    by_pad_in = 0.04
    label_h_in = 0.28
    margin = 0.05

    fig_w = n_panels * (panel_w_in + 2 * bx_pad_in) + n_rodent * arrow_w_in + 2 * margin
    target_px_w = int(panel_w_in * dpi)

    box_rgb = tuple(int(box_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    walker_r = _replace_white_with_color(_resize_to_width(walker_strip, target_px_w), box_rgb)
    rodent_rs = [_replace_white_with_color(_resize_to_width(r, target_px_w), box_rgb) for r in rodent_strips]

    all_imgs = [walker_r] + rodent_rs
    max_img_h_in = max(img.shape[0] / dpi for img in all_imgs)
    box_h = max_img_h_in + 2 * by_pad_in + label_h_in
    fig_h = box_h + 2 * margin
    label_font = 8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_axis_off()
    ax.set_aspect("equal")

    def _place_boxed_panel(img_arr, x_left, label_text):
        iw_in = img_arr.shape[1] / dpi
        ih_in = img_arr.shape[0] / dpi
        box_w = iw_in + 2 * bx_pad_in
        box_y = (fig_h - box_h) / 2

        rect = FancyBboxPatch(
            (x_left, box_y), box_w, box_h,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor=box_color, edgecolor=box_edge, linewidth=1.0,
            zorder=0,
        )
        ax.add_patch(rect)

        img_x = x_left + bx_pad_in
        img_y = box_y + by_pad_in + label_h_in
        img_top = img_y + ih_in
        center_y = box_y + by_pad_in + label_h_in + max_img_h_in / 2
        img_y_centered = center_y - ih_in / 2
        extent = [img_x, img_x + iw_in, img_y_centered, img_y_centered + ih_in]
        ax.imshow(img_arr, extent=extent, aspect="auto", zorder=1)

        if label_text:
            ax.text(
                x_left + box_w / 2,
                box_y + by_pad_in + label_h_in * 0.5,
                label_text,
                ha="center", va="center",
                fontsize=label_font, fontfamily="sans-serif",
                fontweight="bold", color="#333333",
                zorder=2,
            )

        return x_left + box_w

    x = margin
    x = _place_boxed_panel(walker_r, x, walker_label)

    for i, rimg in enumerate(rodent_rs):
        mid_y = fig_h / 2
        ax.annotate(
            "", xy=(x + arrow_w_in - 0.08, mid_y),
            xytext=(x + 0.08, mid_y),
            arrowprops=dict(
                arrowstyle="-|>", color="black",
                lw=2.8, mutation_scale=22,
            ),
            zorder=3,
        )
        x += arrow_w_in
        x = _place_boxed_panel(rimg, x, rodent_label)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white",
                pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Figure 1E: Complexity Arrow")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent.resolve()
    default_config = project_root / "configs" / "demos" / "illustrative" / "demo_bodies.yaml"

    if args.config:
        cfg = load_config(args.config)
    elif default_config.exists():
        print(f"Loading config from {default_config}")
        cfg = load_config(str(default_config))
    else:
        cfg = Config()

    output_dir = Path(resolve_path(cfg.output_dir, project_root))
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg.walker.rollout_path = resolve_path(cfg.walker.rollout_path, project_root)
    cfg.walker.xml_path = resolve_path(cfg.walker.xml_path, project_root)
    for panel in cfg.rodent_panels:
        panel.rollout_path = resolve_path(panel.rollout_path, project_root)
        panel.xml_path = resolve_path(panel.xml_path, project_root)

    print("=" * 60)
    print("Figure 1E: Increasing Complexity (Walker → Rodent)")
    print("=" * 60)

    print(f"\nRendering walker (clip {cfg.walker.clip_id})...")
    walker_strip = render_strip(cfg.walker, cfg.filmstrip, cfg.gap_fraction)
    print(f"  Strip shape: {walker_strip.shape}")

    rodent_strips = []
    for i, panel in enumerate(cfg.rodent_panels):
        print(f"\nRendering rodent panel {i + 1} (clip {panel.clip_id})...")
        strip = render_strip(panel, cfg.filmstrip, cfg.gap_fraction)
        rodent_strips.append(strip)
        print(f"  Strip shape: {strip.shape}")

    ext = cfg.output_format
    out_path = str(output_dir / f"complexity_arrow.{ext}")
    print(f"\nAssembling figure...")
    create_complexity_figure(
        walker_strip, rodent_strips, out_path,
        target_height=cfg.target_strip_height,
        dpi=cfg.dpi,
        walker_label=cfg.walker_label,
        rodent_label=cfg.rodent_label,
    )

    print(f"\nDone. Output: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
