"""
Analyze how cycle detection changes across persistence values for individual clips.

Shows the relationship between persistence threshold and:
- Longest cycle length (number of nodes)
- Total number of cycles detected
- Cycle coverage (fraction of points in any cycle)

This demonstrates the "shortcut fragmentation" effect where low persistence
creates extra edges that break global cycles into smaller pieces.

Usage:
    python cycle_persistence_analysis.py
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import pickle as pkl
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from tqdm import tqdm


K_NEIGHBORS = 5
NUM_EXAMPLE_CLIPS = 3


@dataclass
class Config:
    dm_hypertune_dir: str = "PCD-Graph-Recon-DM/out/dm_hypertune"
    output_dir: str = "out/cycle_persistence_analysis"
    dpi: int = 150


def find_csv_file(dm_hypertune_dir: str) -> str:
    dm_dir = Path(dm_hypertune_dir)
    csv_files = list(dm_dir.glob("tuning_results_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No tuning_results_*.csv found in {dm_hypertune_dir}")
    return str(sorted(csv_files)[-1])


def load_results(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def find_interesting_clips(df: pd.DataFrame, k: int, n_clips: int = 3) -> list[dict]:
    """
    Find clips that show interesting variation in cycle length across persistence.

    Criteria:
    - Has cycles at all persistence values
    - Shows large variation in longest_cycle_length across persistence
    """
    df_k = df[df["k_neighbors"] == k].copy()
    pers_values = sorted(df_k["pers_delta"].unique())
    n_pers = len(pers_values)

    clip_stats = []
    for clip_id in df_k["clip_id"].unique():
        clip_data = df_k[df_k["clip_id"] == clip_id].sort_values("pers_delta")

        if len(clip_data) != n_pers:
            continue
        if not (clip_data["num_cycles"] > 0).all():
            continue

        lengths = clip_data["longest_cycle_length"].values
        num_cycles = clip_data["num_cycles"].values

        length_range = lengths.max() - lengths.min()
        length_ratio = lengths.max() / max(lengths.min(), 1)
        correlation = np.corrcoef(pers_values, lengths)[0, 1] if len(lengths) > 1 else 0

        clip_stats.append({
            "clip_id": int(clip_id),
            "length_range": length_range,
            "length_ratio": length_ratio,
            "correlation": correlation,
            "min_length": lengths.min(),
            "max_length": lengths.max(),
            "lengths": lengths,
            "num_cycles": num_cycles,
        })

    clip_stats.sort(key=lambda x: (-x["correlation"], -x["length_range"]))

    return clip_stats[:n_clips]


def get_run_dir(dm_hypertune_dir: str, k: int, pers_delta: float) -> str | None:
    dm_dir = Path(dm_hypertune_dir)
    for run_dir in dm_dir.iterdir():
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if f"_k{k}_pd" in name:
            pd_str = name.split("_pd")[-1]
            try:
                pd_val = float(pd_str)
                if abs(pd_val - pers_delta) < 0.01:
                    return str(run_dir)
            except ValueError:
                continue
    return None


def load_cycle_data(artifacts_path: str) -> dict:
    """Load all cycle information from artifacts."""
    if not os.path.exists(artifacts_path):
        return {"cycle_inds": [], "pts": None}

    with open(artifacts_path, "rb") as f:
        data = pkl.load(f)

    cycle_inds = data.get("cycle_inds", [])

    features_dir = Path(artifacts_path).parent
    features_path = features_dir / "features.txt"

    pts = None
    if features_path.exists():
        pts = np.loadtxt(str(features_path))

    return {"cycle_inds": cycle_inds, "pts": pts}


def plot_pca_with_cycles(
    ax: plt.Axes,
    pts: np.ndarray,
    cycle_inds: list,
    title: str = "",
) -> None:
    """Plot PCA with all cycles highlighted."""
    if pts is None or len(pts) < 2:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        return

    pca = PCA(n_components=2)
    pts_2d = pca.fit_transform(pts)

    ax.scatter(pts_2d[:, 0], pts_2d[:, 1], c="lightgray", s=2, alpha=0.3, zorder=1)

    if cycle_inds:
        longest_cycle = max(cycle_inds, key=len)
        longest_cycle = np.array(longest_cycle, dtype=int)
        valid_inds = longest_cycle[longest_cycle < len(pts_2d)]

        if len(valid_inds) > 0:
            cycle_pts = pts_2d[valid_inds]
            colors = np.arange(len(cycle_pts))
            ax.scatter(cycle_pts[:, 0], cycle_pts[:, 1], c=colors, cmap="viridis", s=10, zorder=3)
            ax.plot(cycle_pts[:, 0], cycle_pts[:, 1], c="red", linewidth=0.5, alpha=0.5, zorder=2)

    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_aspect("equal", adjustable="box")


def create_analysis_figure(
    df: pd.DataFrame,
    k: int,
    clip_info: dict,
    pers_values: list[float],
    cfg: Config,
) -> plt.Figure:
    """Create a comprehensive analysis figure for one clip."""
    clip_id = clip_info["clip_id"]
    n_pers = len(pers_values)

    fig = plt.figure(figsize=(20, 10))

    gs = GridSpec(2, n_pers, figure=fig, height_ratios=[2.5, 1], hspace=0.3, wspace=0.2)

    fig.suptitle(
        f"Cycle Detection vs Persistence Threshold  |  "
        f"Clip {clip_id}, k={k}  |  Longest cycle: {int(clip_info['min_length'])} → {int(clip_info['max_length'])} nodes",
        fontsize=14, fontweight="bold"
    )

    all_cycle_data = []
    for col_idx, pers_delta in enumerate(pers_values):
        run_dir = get_run_dir(cfg.dm_hypertune_dir, k, pers_delta)
        if run_dir is None:
            all_cycle_data.append({"cycle_inds": [], "pts": None})
            continue

        clip_dir = os.path.join(run_dir, f"clip_{clip_id:03d}")
        artifacts_path = os.path.join(clip_dir, "artifacts.pkl")
        cycle_data = load_cycle_data(artifacts_path)
        all_cycle_data.append(cycle_data)

    for col_idx, pers_delta in enumerate(pers_values):
        ax = fig.add_subplot(gs[0, col_idx])
        cycle_data = all_cycle_data[col_idx]

        n_cycles = len(cycle_data["cycle_inds"])
        longest_len = max(len(c) for c in cycle_data["cycle_inds"]) if cycle_data["cycle_inds"] else 0

        title = f"δ={pers_delta:.1f}\n{n_cycles} cycles, longest={longest_len}"
        plot_pca_with_cycles(ax, cycle_data["pts"], cycle_data["cycle_inds"], title)

    ax_lengths = fig.add_subplot(gs[1, :5])
    ax_lengths.plot(pers_values, clip_info["lengths"], "o-", color="blue", linewidth=2, markersize=8)
    ax_lengths.fill_between(pers_values, clip_info["lengths"], alpha=0.3)
    ax_lengths.set_xlabel("Persistence Threshold (δ)", fontsize=10)
    ax_lengths.set_ylabel("Longest Cycle Length (nodes)", fontsize=10)
    ax_lengths.set_title("Longest Cycle Length vs Persistence", fontsize=10, fontweight="bold")
    ax_lengths.grid(True, alpha=0.3)
    ax_lengths.set_xlim(0, 1.1)

    for i, (pd, length) in enumerate(zip(pers_values, clip_info["lengths"])):
        ax_lengths.annotate(f"{int(length)}", (pd, length), textcoords="offset points",
                           xytext=(0, 8), ha="center", fontsize=8)

    ax_ncycles = fig.add_subplot(gs[1, 5:])
    ax_ncycles.plot(pers_values, clip_info["num_cycles"], "o-", color="green", linewidth=2, markersize=8)
    ax_ncycles.fill_between(pers_values, clip_info["num_cycles"], alpha=0.3, color="green")
    ax_ncycles.set_xlabel("Persistence Threshold (δ)", fontsize=10)
    ax_ncycles.set_ylabel("Number of Cycles", fontsize=10)
    ax_ncycles.set_title("Number of Cycles vs Persistence", fontsize=10, fontweight="bold")
    ax_ncycles.grid(True, alpha=0.3)
    ax_ncycles.set_xlim(0, 1.1)

    for i, (pd, nc) in enumerate(zip(pers_values, clip_info["num_cycles"])):
        ax_ncycles.annotate(f"{int(nc)}", (pd, nc), textcoords="offset points",
                           xytext=(0, 8), ha="center", fontsize=8)

    return fig


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print("Cycle-Persistence Analysis")
    print("=" * 60)
    print(f"K_NEIGHBORS: {K_NEIGHBORS}")
    print(f"NUM_EXAMPLE_CLIPS: {NUM_EXAMPLE_CLIPS}")
    print()

    csv_path = find_csv_file(cfg.dm_hypertune_dir)
    print(f"Using results: {csv_path}")

    df = load_results(csv_path)
    pers_values = sorted(df["pers_delta"].unique())

    print(f"\nFinding clips with interesting persistence-cycle relationships...")
    interesting_clips = find_interesting_clips(df, K_NEIGHBORS, NUM_EXAMPLE_CLIPS)

    if not interesting_clips:
        print("No suitable clips found!")
        return

    print(f"\nFound {len(interesting_clips)} interesting clips:")
    for clip_info in interesting_clips:
        print(f"  Clip {clip_info['clip_id']}: "
              f"length {int(clip_info['min_length'])} → {int(clip_info['max_length'])} "
              f"(correlation: {clip_info['correlation']:.2f})")

    for i, clip_info in enumerate(interesting_clips):
        clip_id = clip_info["clip_id"]
        print(f"\n=== Creating figure for Clip {clip_id} ({i+1}/{len(interesting_clips)}) ===")

        fig = create_analysis_figure(df, K_NEIGHBORS, clip_info, pers_values, cfg)

        output_path = os.path.join(cfg.output_dir, f"clip_{clip_id}_k{K_NEIGHBORS}_analysis.png")
        fig.savefig(output_path, dpi=cfg.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {output_path}")

    print("\n" + "=" * 60)
    print("Creating summary comparison...")

    fig_summary, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(interesting_clips)))

    for idx, clip_info in enumerate(interesting_clips):
        clip_id = clip_info["clip_id"]
        axes[0].plot(pers_values, clip_info["lengths"], "o-",
                    color=colors[idx], linewidth=2, markersize=6,
                    label=f"Clip {clip_id}")
        axes[1].plot(pers_values, clip_info["num_cycles"], "o-",
                    color=colors[idx], linewidth=2, markersize=6,
                    label=f"Clip {clip_id}")

    axes[0].set_xlabel("Persistence Threshold (δ)", fontsize=11)
    axes[0].set_ylabel("Longest Cycle Length (nodes)", fontsize=11)
    axes[0].set_title("Longest Cycle Length vs Persistence", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Persistence Threshold (δ)", fontsize=11)
    axes[1].set_ylabel("Number of Cycles", fontsize=11)
    axes[1].set_title("Number of Cycles vs Persistence", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig_summary.suptitle(f"Cycle Detection vs Persistence (k={K_NEIGHBORS})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    summary_path = os.path.join(cfg.output_dir, f"summary_k{K_NEIGHBORS}.png")
    fig_summary.savefig(summary_path, dpi=cfg.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig_summary)
    print(f"Saved: {summary_path}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Figures saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
