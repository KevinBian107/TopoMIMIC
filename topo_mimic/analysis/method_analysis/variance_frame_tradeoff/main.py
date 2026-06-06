"""Variance-frame tradeoff experiment: show DM-paths and DM-cycles are
meaningful behavioral summaries (capture more qpos pose variance per frame than
random sub-trajectories of equal length), over all clips and feature spaces.

Usage:
    python -m topo_mimic.analysis.method_analysis.variance_frame_tradeoff.main \
        configs/analysis/method_analysis/variance_frame_tradeoff/variance_frame_tradeoff.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .analysis import compute_tradeoff
from .config import load_config
from .plotting import render, render_appendix


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Computing variance-frame tradeoff over all clips...")
    df = compute_tradeoff(cfg)
    if df.empty:
        raise RuntimeError("no primitives found; check extraction_dir paths in config")

    csv = out_dir / "variance_frame_tradeoff.csv"
    df.to_csv(csv, index=False)
    print(f"Wrote {csv}  ({len(df)} primitives)")

    # summary: median efficiency (captured / matched-random) and fraction > 1
    print("\nMeaningfulness summary (efficiency = captured variance / matched random window):")
    for sp in df["space"].unique():
        for typ in ("DM-cycle", "DM-path"):
            t = df[(df["space"] == sp) & (df["type"] == typ)]["efficiency"].dropna()
            if len(t):
                print(f"  {sp:10s} {typ:9s}  median={np.median(t):.2f}  "
                      f"frac>1={(t > 1).mean():.2f}  n={len(t)}")

    png = render(df, out_dir)
    print(f"\nWrote figure {png}")
    png_app = render_appendix(df, out_dir)
    print(f"Wrote appendix figure {png_app}")


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        raise SystemExit(
            "usage: python -m topo_mimic.analysis.method_analysis.variance_frame_tradeoff.main <config.yaml>")
    run(args[0])


if __name__ == "__main__":
    main()
