"""Visualization for scale-sweep.

Two artifacts, both dev-only:
  - discovery_eyeball.png — 50x~20 matrix of checks/crosses, auto-generated matplotlib,
    no polish. Readable at a glance so Kaiwen can spot patterns.
  - trace PNGs — per-variant, per-exemplar-clip. 3 horizontal color strips: GT behavior
    on top, predicted cluster/node id in middle, time axis at bottom. 5 exemplar clips
    per variant, rendered only when data is available.

Paper artifacts are pivoted from discovery_long.csv, not from these PNGs.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

matplotlib.use("Agg")

from topo_mimic.experiment.auto_label import BehaviorSegment

log = logging.getLogger(__name__)


def save_discovery_eyeball_png(
    binary_wide: pd.DataFrame,
    out_path: Path,
    method_blocks: list[tuple[str, list[str], str]] | None = None,
) -> None:
    """Save a simple check/cross heatmap.

    Args:
        binary_wide: rows=transitions, columns=variant_ids, values in {0, 1}.
        out_path: where to save the PNG.
        method_blocks: optional list of (block_name, variant_id_list, color) for vertical
            separators. If None, columns are shown in their existing order with no banding.
    """
    n_trans, n_var = binary_wide.shape
    fig_w = max(12.0, n_var * 0.3)
    fig_h = max(6.0, n_trans * 0.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = ListedColormap(["#f4b3b3", "#b3e5b3"])
    ax.imshow(binary_wide.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(n_trans):
        for j in range(n_var):
            val = int(binary_wide.values[i, j])
            ax.text(j, i, "\u2713" if val else "\u00d7",
                    ha="center", va="center", fontsize=8, color="black")

    ax.set_xticks(range(n_var))
    ax.set_xticklabels(binary_wide.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(n_trans))
    ax.set_yticklabels(binary_wide.index, fontsize=8)
    ax.set_xlabel("variant")
    ax.set_ylabel("real transition")

    if method_blocks:
        col_to_idx = {c: i for i, c in enumerate(binary_wide.columns)}
        for block_name, variant_ids, color in method_blocks:
            indices = sorted(col_to_idx[v] for v in variant_ids if v in col_to_idx)
            if not indices:
                continue
            left, right = indices[0], indices[-1]
            ax.axvline(left - 0.5, color="black", lw=1.0)
            ax.axvline(right + 0.5, color="black", lw=1.0)
            ax.add_patch(Rectangle(
                (left - 0.5, -0.7),
                right - left + 1, 0.5,
                facecolor=color, edgecolor="black", lw=0.5, clip_on=False,
            ))
            ax.text((left + right) / 2, -1.0, block_name,
                    ha="center", va="bottom", fontsize=8, color="black")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[plots] wrote discovery_eyeball.png ({n_trans}x{n_var}) -> {out_path}")


def _pick_exemplar_clips(
    clip_ids: list[int],
    segments_by_clip: dict[int, list[BehaviorSegment]],
    n: int,
) -> list[int]:
    """Pick n exemplar clips deterministically.

    Selects clips with diverse GT structure (multi-transition, grooming, rare labels)
    when possible. Falls back to the first n clips otherwise.
    """
    scored: list[tuple[int, int, int]] = []
    for cid in clip_ids:
        segs = segments_by_clip.get(cid, [])
        if not segs:
            continue
        distinct_behaviors = len({s.behavior for s in segs})
        n_segments = len(segs)
        scored.append((cid, n_segments, distinct_behaviors))

    scored.sort(key=lambda t: (-t[1], -t[2], t[0]))
    picks = [t[0] for t in scored[:n]]
    while len(picks) < n and len(clip_ids) > len(picks):
        for cid in clip_ids:
            if cid not in picks:
                picks.append(cid)
                if len(picks) >= n:
                    break
    return picks[:n]


_TAB20 = plt.get_cmap("tab20").colors


def _behavior_color_map(all_behaviors: list[str]) -> dict[str, tuple]:
    ordered = sorted(set(all_behaviors))
    return {b: _TAB20[i % 20] for i, b in enumerate(ordered)}


def save_trace_png(
    clip_id: int,
    n_frames: int,
    gt_segments: list[BehaviorSegment],
    predicted_labels: np.ndarray,
    out_path: Path,
    title: str,
    behavior_colors: dict[str, tuple],
) -> None:
    """3-row color-strip: GT | predicted | time axis."""
    assert predicted_labels.ndim == 1
    fig, axes = plt.subplots(3, 1, figsize=(12, 2.5),
                             gridspec_kw={"height_ratios": [1, 1, 0.2]})

    for seg in gt_segments:
        color = behavior_colors.get(seg.behavior, (0.5, 0.5, 0.5))
        axes[0].axvspan(seg.start, min(seg.end, n_frames), color=color, alpha=0.85)
    axes[0].set_xlim(0, n_frames)
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[0].set_ylabel("GT", rotation=0, labelpad=35, va="center", fontsize=9)

    runs = _contiguous_runs(predicted_labels)
    for start, end, sid in runs:
        if sid == -1:
            color = (0.85, 0.85, 0.85)
        else:
            color = _TAB20[int(sid) % 20]
        axes[1].axvspan(start, end, color=color, alpha=0.85)
    axes[1].set_xlim(0, n_frames)
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_ylabel("Pred", rotation=0, labelpad=35, va="center", fontsize=9)

    axes[2].set_xlim(0, n_frames)
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    axes[2].set_xticks(np.linspace(0, n_frames, 11, dtype=int))
    axes[2].set_xlabel("frame")
    for seg in gt_segments[1:]:
        axes[2].axvline(seg.start, color="black", lw=0.8, alpha=0.5)

    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _contiguous_runs(arr: np.ndarray) -> list[tuple[int, int, int]]:
    """Yield (start, end, value) triples for contiguous same-value runs."""
    if arr.size == 0:
        return []
    runs: list[tuple[int, int, int]] = []
    start = 0
    prev = int(arr[0])
    for i in range(1, len(arr)):
        cur = int(arr[i])
        if cur != prev:
            runs.append((start, i, prev))
            start = i
            prev = cur
    runs.append((start, len(arr), prev))
    return runs


def generate_trace_pngs_for_variant(
    variant_id: str,
    labels_list_path: str | Path,
    clip_ids: list[int],
    exemplar_clips: list[int],
    segments_by_clip: dict[int, list[BehaviorSegment]],
    output_dir: Path,
    behavior_colors: dict[str, tuple],
    n_frames_per_clip: int,
) -> list[Path]:
    """Render one PNG per exemplar clip for this variant."""
    labels_path = Path(labels_list_path)
    if not labels_path.exists():
        log.warning(f"[plots] labels_list not found for {variant_id}: {labels_path}")
        return []
    with labels_path.open("rb") as f:
        labels_list = pickle.load(f)

    cid_to_idx = {cid: i for i, cid in enumerate(clip_ids)}
    out_paths: list[Path] = []

    for cid in exemplar_clips:
        if cid not in cid_to_idx:
            continue
        per_clip = np.asarray(labels_list[cid_to_idx[cid]])
        png_path = output_dir / variant_id / f"clip_{cid}.png"
        if png_path.exists():
            out_paths.append(png_path)
            continue
        save_trace_png(
            clip_id=cid,
            n_frames=n_frames_per_clip,
            gt_segments=segments_by_clip.get(cid, []),
            predicted_labels=per_clip,
            out_path=png_path,
            title=f"{variant_id} | clip_{cid}",
            behavior_colors=behavior_colors,
        )
        out_paths.append(png_path)
    return out_paths
