"""Render a clean density manifold for use as a favicon.

Produces a floating 3D surface with transparent background, no axes, no box.
Uses the two-basin landscape from demo_saddle.py (figure 1A).

Usage:
    python3 -m demos.illustrative.demo_favicon configs/demos/illustrative/demo_favicon.yaml
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


@dataclass
class FaviconConfig:
    output_dir: str = "outputs/favicon"
    surface_fn: str = "two_basin"
    elev: float = 30
    azim: float = -60
    resolution: int = 200
    sizes: list[int] | None = None
    cmap: str = "coolwarm"
    alpha: float = 0.9

    def __post_init__(self):
        if self.sizes is None:
            self.sizes = [512, 64, 32]


SURFACE_FNS = {}


def density_basin_surface_two(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    base = np.zeros_like(x)
    basin_a = -1.2 * np.exp(-((x + 1.5)**2 + y**2) / 0.5)
    basin_b = -1.2 * np.exp(-((x - 1.5)**2 + y**2) / 0.5)
    ridge = 0.6 * np.exp(-(x**2) / 0.4) * np.exp(-(y**2) / 3.0)
    pass_upper = -0.35 * np.exp(-(x**2) / 0.6) * np.exp(-((y - 0.9)**2) / 0.3)
    pass_lower = -0.35 * np.exp(-(x**2) / 0.6) * np.exp(-((y + 0.9)**2) / 0.3)
    return base + basin_a + basin_b + ridge + pass_upper + pass_lower


SURFACE_FNS["two_basin"] = density_basin_surface_two


def density_basin_surface_multi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    base = np.zeros_like(x)
    ba = -1.2 * np.exp(-((x + 2)**2 + y**2) / 0.5)
    bb = -1.2 * np.exp(-((x - 1)**2 + (y - 1.5)**2) / 0.5)
    bc = -1.2 * np.exp(-((x - 1)**2 + (y + 1.5)**2) / 0.5)
    ridge_ab = 0.5 * np.exp(-((x + 0.5)**2) / 0.5) * np.exp(-((y - 0.75)**2) / 1.5)
    pass_ab_1 = -0.35 * np.exp(-((x + 0.3)**2 + (y - 1.4)**2) / 0.4)
    pass_ab_2 = -0.35 * np.exp(-((x + 0.3)**2 + (y - 0.1)**2) / 0.4)
    ridge_ac = 0.6 * np.exp(-((x + 0.5)**2) / 0.5) * np.exp(-((y + 0.75)**2) / 1.5)
    pass_ac = -0.4 * np.exp(-((x + 0.3)**2 + (y + 0.8)**2) / 0.4)
    wall_ac_lower = 0.4 * np.exp(-((x + 0.3)**2 + (y + 1.6)**2) / 0.4)
    return base + ba + bb + bc + ridge_ab + pass_ab_1 + pass_ab_2 + ridge_ac + pass_ac + wall_ac_lower


SURFACE_FNS["multi_basin"] = density_basin_surface_multi


def render_favicon(cfg: FaviconConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    surface_fn = SURFACE_FNS[cfg.surface_fn]

    if cfg.surface_fn == "multi_basin":
        lim_x, lim_y = (-3.5, 2.5), (-3, 3)
    else:
        lim_x, lim_y = (-3, 3), (-2.5, 2.5)

    x = np.linspace(lim_x[0], lim_x[1], cfg.resolution)
    y = np.linspace(lim_y[0], lim_y[1], cfg.resolution)
    X, Y = np.meshgrid(x, y)
    Z = surface_fn(X, Y)

    fig = plt.figure(figsize=(6, 6), dpi=300)
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        X, Y, Z, cmap=cfg.cmap, alpha=cfg.alpha,
        linewidth=0, antialiased=True,
    )

    ax.set_axis_off()
    ax.set_facecolor((0, 0, 0, 0))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.view_init(elev=cfg.elev, azim=cfg.azim)

    ax.set_xlim(lim_x)
    ax.set_ylim(lim_y)

    full_path = output_dir / "favicon_full.png"
    fig.savefig(full_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

    from PIL import Image

    img = Image.open(full_path).convert("RGBA")
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    side = max(img.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    offset = ((side - img.size[0]) // 2, (side - img.size[1]) // 2)
    square.paste(img, offset)

    for size in cfg.sizes:
        resized = square.resize((size, size), Image.LANCZOS)
        resized.save(output_dir / f"favicon_{size}.png")

    favicon_32 = square.resize((32, 32), Image.LANCZOS)
    favicon_32.save(output_dir / "favicon.ico")

    print(f"Saved favicon outputs to {output_dir}/")
    for size in cfg.sizes:
        print(f"  favicon_{size}.png")
    print(f"  favicon.ico")


def load_config(config_path: str) -> FaviconConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return FaviconConfig(**raw)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m demos.illustrative.demo_favicon <config.yaml>")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    render_favicon(cfg)


if __name__ == "__main__":
    main()
