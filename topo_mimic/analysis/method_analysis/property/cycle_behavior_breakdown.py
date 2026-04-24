"""Clip-level behavior signature breakdown across cycle-count categories.

For each clip, compute the ordered, dedup'd sequence of behavior labels from
rodent_labels.csv (direction labels ignored). Each clip is categorized as
either a SINGLETON (one behavior throughout) or a TRANSITION-CONTAINING
clip with one or more adjacent "A -> B" transitions. A clip with three or
more distinct consecutive behaviors contributes to multiple transitions
(binary containment).

Hypothesis: no-cycle clips are mostly singletons; many-cycle clips are
dominated by transitions like "Walk -> Immobile". This script tabulates it.

Outputs per category (no_cycle, many_cycle):
  - <cat>_signature_table.csv  — rows: singleton/transition types, cols: clip count + %
  - <cat>_signature_table.png  — bar chart of the same

Usage:
    python -m topo_mimic.analysis.method_analysis.property.cycle_behavior_breakdown \
        --dm_results_dir outputs/extraction/dm_pipeline_intention \
        --labels_csv behavior_labels/rodent_labels.csv \
        --output_dir outputs/analysis/method_analysis/cycle_behavior_breakdown \
        --many_cycle_threshold 2
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from topo_mimic.experiment.auto_label import (
    BEHAVIOR_LABELS,
    BehaviorSegment,
    load_behavior_segments,
    split_segments_by_kind,
)


NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}

CATEGORY_COLORS = {
    "no_cycle": "#C44E52",
    "many_cycle": "#4878CF",
}


def categorize_clips(
    dm_results_dir: str,
    many_cycle_threshold: int,
) -> tuple[list[int], list[int], list[int]]:
    """Return (no_cycle_clips, many_cycle_clips, one_cycle_clips)."""
    cycles_path = os.path.join(dm_results_dir, "all_cycle_dm.pkl")
    with open(cycles_path, "rb") as f:
        all_cycle_dm = pickle.load(f)

    no_cycle, one_cycle, many_cycle = [], [], []
    for clip_id, cycles in enumerate(all_cycle_dm):
        n = len(cycles)
        if n == 0:
            no_cycle.append(clip_id)
        elif n >= many_cycle_threshold:
            many_cycle.append(clip_id)
        else:
            one_cycle.append(clip_id)
    return no_cycle, many_cycle, one_cycle


def clip_behavior_runs(
    clip_segments: list[BehaviorSegment],
) -> list[tuple[str, int]]:
    """Return the ordered runs of the clip as (behavior, frames).

    Segments are sorted by start_frame; consecutive segments sharing the same
    behavior are merged so [Immobile (50), Immobile (30), Walk (20), Walk (10),
    Immobile (100)] -> [(Immobile, 80), (Walk, 30), (Immobile, 100)].
    Only BEHAVIOR_LABELS are kept.
    """
    segs = [s for s in clip_segments if s.behavior in BEHAVIOR_LABELS]
    if not segs:
        return []
    segs_sorted = sorted(segs, key=lambda s: s.start)
    runs: list[tuple[str, int]] = [(segs_sorted[0].behavior,
                                    max(0, segs_sorted[0].end - segs_sorted[0].start))]
    for s in segs_sorted[1:]:
        n = max(0, s.end - s.start)
        if s.behavior == runs[-1][0]:
            runs[-1] = (runs[-1][0], runs[-1][1] + n)
        else:
            runs.append((s.behavior, n))
    return runs


def summarize_clip(
    runs: list[tuple[str, int]],
) -> tuple[str, list[tuple[str, int]]]:
    """Classify a clip by its run sequence, attributing frame counts to each signature.

    Returns (kind, signature_frames):
      - kind == "empty" when no labels.
      - kind == "singleton" when only one distinct behavior. signature_frames =
        [(behavior, total_frames_of_that_behavior_in_clip)].
      - kind == "transition" when 2+ distinct behaviors. signature_frames has
        one entry per adjacent (A, B) pair: the tuple ("A -> B", frames_of_A + frames_of_B).
        A run that participates in two adjacencies (e.g. Walk in [Immobile, Walk, Rear])
        contributes its frames to BOTH.
    """
    if not runs:
        return "empty", []
    if len(runs) == 1:
        b, f = runs[0]
        return "singleton", [(b, int(f))]
    sig_frames: list[tuple[str, int]] = []
    for (a_b, a_f), (b_b, b_f) in zip(runs[:-1], runs[1:]):
        sig_frames.append((f"{a_b} -> {b_b}", int(a_f) + int(b_f)))
    return "transition", sig_frames


def tally_category(
    clip_ids: list[int],
    behavior_segments: dict[int, list[BehaviorSegment]],
) -> tuple[pd.DataFrame, dict]:
    """For one category, weight each signature by the frame count it covers
    per clip (not by 1 per clip).

    Frame attribution:
      - Singleton "X": frames = total frames of X in that clip.
      - Transition "A -> B": frames = frames_of_A_run + frames_of_B_run for each
        adjacency; a run that participates in two adjacencies contributes its
        frames to both transitions.
    """
    singleton_frames: Counter = Counter()
    transition_frames: Counter = Counter()
    singleton_clip_count: Counter = Counter()
    transition_clip_count: Counter = Counter()
    n_singleton_clips = 0
    n_transition_clips = 0
    n_empty_clips = 0
    total_singleton_frames = 0
    total_transition_frames = 0

    for cid in clip_ids:
        segs = behavior_segments.get(int(cid), [])
        runs = clip_behavior_runs(segs)
        kind, sig_frames = summarize_clip(runs)
        if kind == "empty":
            n_empty_clips += 1
            continue
        if kind == "singleton":
            sig, f = sig_frames[0]
            singleton_frames[sig] += int(f)
            singleton_clip_count[sig] += 1
            n_singleton_clips += 1
            total_singleton_frames += int(f)
        else:
            n_transition_clips += 1
            seen: set[str] = set()
            clip_transition_frames = 0
            for sig, f in sig_frames:
                transition_frames[sig] += int(f)
                clip_transition_frames += int(f)
                if sig not in seen:
                    transition_clip_count[sig] += 1
                    seen.add(sig)
            total_transition_frames += clip_transition_frames

    total_frames_all = total_singleton_frames + total_transition_frames
    denom_frames = max(total_frames_all, 1)

    rows: list[dict] = []
    for b in sorted(singleton_frames.keys()):
        f = singleton_frames[b]
        rows.append({
            "kind": "singleton",
            "signature": b,
            "clip_count": int(singleton_clip_count[b]),
            "frame_count": int(f),
            "pct_of_labeled_frames": round(100.0 * f / denom_frames, 3),
        })
    for t, f in sorted(transition_frames.items(), key=lambda kv: (-kv[1], kv[0])):
        rows.append({
            "kind": "transition",
            "signature": t,
            "clip_count": int(transition_clip_count[t]),
            "frame_count": int(f),
            "pct_of_labeled_frames": round(100.0 * f / denom_frames, 3),
        })

    df = pd.DataFrame(rows)

    stats = {
        "n_clips_total": len(clip_ids),
        "n_clips_with_labels": n_singleton_clips + n_transition_clips,
        "n_clips_unlabeled": n_empty_clips,
        "n_singleton_clips": n_singleton_clips,
        "n_transition_clips": n_transition_clips,
        "total_singleton_frames": int(total_singleton_frames),
        "total_transition_frames": int(total_transition_frames),
        "total_labeled_frames": int(total_frames_all),
        "pct_frames_singleton": round(100.0 * total_singleton_frames / denom_frames, 2),
        "pct_frames_transition": round(100.0 * total_transition_frames / denom_frames, 2),
    }
    return df, stats


def plot_category_table(
    df: pd.DataFrame,
    stats: dict,
    category: str,
    output_png: str,
    color: str,
) -> None:
    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(max(6.4, 0.35 * len(df) + 2.0), 4.2))

        kinds = df["kind"].tolist()
        sigs = df["signature"].tolist()
        frames = df["frame_count"].to_numpy()
        pct = df["pct_of_labeled_frames"].to_numpy()

        x = np.arange(len(df))
        bar_colors = [color if k == "transition" else "#888888" for k in kinds]
        ax.bar(x, pct, color=bar_colors, edgecolor="none")
        for xi, (fc, p) in enumerate(zip(frames, pct)):
            ax.text(xi, p + 0.5, f"{p:.1f}%\n({fc:,}f)", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(sigs, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("% of labeled frames in category")
        ax.set_title(
            f"{category}: clip-level signature breakdown (frame-weighted)\n"
            f"singleton frames={stats['total_singleton_frames']:,} "
            f"({stats['pct_frames_singleton']:.1f}%), "
            f"transition frames={stats['total_transition_frames']:,} "
            f"({stats['pct_frames_transition']:.1f}%)",
            fontweight="bold",
        )
        ax.set_ylim(0, max(pct.max() * 1.25, 10.0))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        from matplotlib.patches import Patch
        legend = [
            Patch(facecolor="#888888", label="singleton (only one behavior)"),
            Patch(facecolor=color, label="transition (adjacent A -> B)"),
        ]
        ax.legend(handles=legend, fontsize=7, frameon=True, edgecolor="#cccccc")

        plt.tight_layout()
        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Clip-level behavior signature by cycle category.")
    parser.add_argument("--dm_results_dir", default="outputs/extraction/dm_pipeline_intention")
    parser.add_argument("--labels_csv", default="behavior_labels/rodent_labels.csv")
    parser.add_argument("--output_dir", default="outputs/analysis/method_analysis/cycle_behavior_breakdown")
    parser.add_argument("--many_cycle_threshold", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = Path(args.output_dir)

    no_cycle, many_cycle, one_cycle = categorize_clips(
        args.dm_results_dir, args.many_cycle_threshold,
    )
    print(f"no_cycle clips:   {len(no_cycle)}")
    print(f"one_cycle clips:  {len(one_cycle)}")
    print(f"many_cycle clips: {len(many_cycle)} (threshold >= {args.many_cycle_threshold})")

    raw = load_behavior_segments(args.labels_csv)
    behav, _dirs = split_segments_by_kind(raw)

    summary_rows = []
    for name, clip_ids in [("no_cycle", no_cycle), ("many_cycle", many_cycle)]:
        df, stats = tally_category(clip_ids, behav)
        print(f"\n=== {name} ({len(clip_ids)} clips) ===")
        print(f"  singleton frames: {stats['total_singleton_frames']:,} "
              f"({stats['pct_frames_singleton']:.1f}%)")
        print(f"  transition frames: {stats['total_transition_frames']:,} "
              f"({stats['pct_frames_transition']:.1f}%)")
        print(df.to_string(index=False))

        csv_path = out_dir / f"{name}_signature_table.csv"
        df.to_csv(csv_path, index=False)
        png_path = out_dir / f"{name}_signature_table.png"
        plot_category_table(df, stats, name, str(png_path), CATEGORY_COLORS[name])
        print(f"wrote {csv_path}")
        print(f"wrote {png_path}")

        summary_rows.append({
            "category": name,
            "n_clips": stats["n_clips_total"],
            "n_labeled_clips": stats["n_clips_with_labels"],
            "total_singleton_frames": stats["total_singleton_frames"],
            "pct_frames_singleton": stats["pct_frames_singleton"],
            "total_transition_frames": stats["total_transition_frames"],
            "pct_frames_transition": stats["pct_frames_transition"],
            "total_labeled_frames": stats["total_labeled_frames"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "category_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nwrote {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
