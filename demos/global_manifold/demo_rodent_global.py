#!/usr/bin/env python3
"""Global-manifold transition demo — rodent.

Pools rodent clips whose GT labels contain configured A↔B transitions, runs DM
on the pooled point cloud per feature space (intention / qpos / ekp), and
produces a projection overlay figure + a cross-clip filmstrip figure.

Usage:
    python -m demos.global_manifold.demo_rodent_global
    python -m demos.global_manifold.demo_rodent_global --config configs/demos/global_manifold/rodent.yaml
    python -m demos.global_manifold.demo_rodent_global --skip-filmstrip
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demos.global_manifold.common import (
    build_frame_labels,
    load_config,
    process_feature_space,
    render_projection_grid,
    select_clips_for_transitions,
)
from demos.global_manifold.filmstrip import render_filmstrip_grid

DEFAULT_CONFIG = "configs/demos/global_manifold/rodent.yaml"


def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return str(REPO_ROOT / p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rodent global-manifold transition demo")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-filmstrip", action="store_true")
    parser.add_argument("--only-feature-space", type=str, default=None,
                        help="Run only one feature space (intention|qpos|ekp)")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Cap number of clips (debug)")
    args = parser.parse_args()

    cfg_path = _resolve_path(args.config)
    print(f"Loading config: {cfg_path}")
    cfg = load_config(cfg_path)

    cfg.data_path = _resolve_path(cfg.data_path)
    cfg.labels_csv = _resolve_path(cfg.labels_csv)
    cfg.xml_path = _resolve_path(cfg.xml_path) if cfg.xml_path else ""
    cfg.stac_data_path = _resolve_path(cfg.stac_data_path) if cfg.stac_data_path else ""
    cfg.output_dir = _resolve_path(cfg.output_dir)
    cfg.cache_dir = _resolve_path(cfg.cache_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)

    if args.only_feature_space is not None:
        cfg.feature_spaces = [fs for fs in cfg.feature_spaces if fs.dm_key == args.only_feature_space]
        if not cfg.feature_spaces:
            raise SystemExit(f"no feature space matched --only-feature-space={args.only_feature_space}")

    print("=" * 60)
    print(f"Rodent global-manifold demo")
    print(f"  transitions: {cfg.transition_types}")
    print(f"  feature spaces: {[fs.dm_key for fs in cfg.feature_spaces]}")
    print(f"  output: {cfg.output_dir}")
    print("=" * 60)

    print("\n[1/4] selecting clips")
    clip_ids = select_clips_for_transitions(
        cfg.labels_csv, cfg.transition_types, max_clips=args.max_clips,
    )
    clip_ids_sorted = sorted(clip_ids)
    print(f"  selected {len(clip_ids_sorted)} rodent clips")

    print("\n[2/4] building per-frame GT labels")
    frame_labels = build_frame_labels(cfg.labels_csv, selected_clips=set(clip_ids_sorted))

    print("\n[3/4] per-feature-space pipeline")
    results = []
    for fs in cfg.feature_spaces:
        print(f"\n--- feature space: {fs.dm_key} ---")
        res = process_feature_space(cfg, fs, frame_labels, clip_ids_sorted)
        results.append(res)

    print("\n[4/4] rendering figures")
    proj_pdf = os.path.join(cfg.output_dir, "projection_grid.pdf")
    render_projection_grid(cfg, results, frame_labels, proj_pdf)

    if not args.skip_filmstrip:
        film_pdf = os.path.join(cfg.output_dir, "filmstrip_grid.pdf")
        render_filmstrip_grid(cfg, results, film_pdf)

    print("\nDone.")


if __name__ == "__main__":
    main()
