#!/usr/bin/env python3
"""Build paper/bian_NeurReps_2026/figures/figure3.png from the transition_projections demo
outputs (manuscript Figure 3: DM-cycles).

Main figure = vertical stack of (A) the intention-only t-SNE panel and
(B) the intention-only DM-path filmstrip. Both source PNGs are produced
directly by demos.demo_transition_projections (with stripped suptitles).

Full 3-feature-space versions are copied to paper/bian_NeurReps_2026/appendix_figures/.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]

PANEL_INT_SRC = REPO / "outputs/demos/illustrative/transition_projections/transition_panel_tsne_intention.png"
FILM_INT_SRC = REPO / "outputs/demos/illustrative/transition_projections/transition_filmstrips_intention.png"
PANEL_FULL_SRC = REPO / "outputs/demos/illustrative/transition_projections/transition_panel_tsne.png"
FILM_FULL_SRC = REPO / "outputs/demos/illustrative/transition_projections/transition_filmstrips.png"

FIG2_OUT = REPO / "paper/bian_NeurReps_2026/figures/figure3.png"
APPENDIX_PANEL = REPO / "paper/bian_NeurReps_2026/appendix_figures/appendix_transition_panel_tsne.png"
APPENDIX_FILM = REPO / "paper/bian_NeurReps_2026/appendix_figures/appendix_transition_filmstrips.png"


def fit_width(img: Image.Image, target_w: int) -> Image.Image:
    if img.size[0] == target_w:
        return img
    scale = target_w / img.size[0]
    new_h = int(img.size[1] * scale)
    return img.resize((target_w, new_h), Image.LANCZOS)


def add_label(img: Image.Image, label: str, pad: int = 38,
              font_size: int = 28) -> Image.Image:
    out = Image.new("RGB", (img.size[0], img.size[1] + pad), color=(255, 255, 255))
    out.paste(img, (0, pad))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 4), label, fill=(0, 0, 0), font=font)
    return out


def main() -> None:
    for src in (PANEL_INT_SRC, FILM_INT_SRC, PANEL_FULL_SRC, FILM_FULL_SRC):
        if not src.exists():
            raise FileNotFoundError(
                f"{src} missing. Run "
                "`python -m demos.demo_transition_projections "
                "--config configs/demos/demo_transition_projections.yaml` first."
            )

    panel = Image.open(PANEL_INT_SRC).convert("RGB")
    film = Image.open(FILM_INT_SRC).convert("RGB")

    target_w = max(panel.size[0], film.size[0])
    panel = fit_width(panel, target_w)
    film = fit_width(film, target_w)

    panel_lab = add_label(panel, "(A)", pad=38, font_size=28)
    film_lab = add_label(film, "(B)", pad=38, font_size=28)

    gap = 12
    out_h = panel_lab.size[1] + gap + film_lab.size[1]
    out = Image.new("RGB", (target_w, out_h), color=(255, 255, 255))
    out.paste(panel_lab, (0, 0))
    out.paste(film_lab, (0, panel_lab.size[1] + gap))

    FIG2_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.save(FIG2_OUT, dpi=(220, 220))
    print(f"wrote {FIG2_OUT} (size={out.size})")

    APPENDIX_PANEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PANEL_FULL_SRC, APPENDIX_PANEL)  # content only; copy2 chmod fails on root-owned dest
    shutil.copyfile(FILM_FULL_SRC, APPENDIX_FILM)
    print(f"copied  {APPENDIX_PANEL}")
    print(f"copied  {APPENDIX_FILM}")


if __name__ == "__main__":
    main()
