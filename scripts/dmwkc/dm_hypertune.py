"""
Hyperparameter tuning for Discrete Morse graph extraction.

Refactored from dm_latent_subspace_hypertune.py.
Provides grid search over k_neighbors and pers_delta with optional
auto-estimation of Takens parameters.
"""

import os
import sys
import json
import functools
import multiprocessing as mp
import pickle as pkl
import yaml
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid

from vnl_tda.utils.load_utils import get_aggregate_data
from vnl_tda.pipelines.tda.skeletonization.skeletonization import extract_skeleton_pcd_dm
from vnl_tda.pipelines.tda.skeletonization.preprocess import estimate_delay_embedding_params


@dataclass
class DataConfig:
    dm_key: str = "intention"
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    num_clips_total: int = 800
    num_clips_eval: int = 10


@dataclass
class TakensConfig:
    auto_estimate: bool = True
    m: int = 3
    tau: int = 6
    max_tau: int = 200
    bins: int = 64
    m_max: int = 15
    R_tol: float = 15.0
    A_tol: float = 2.0
    fnn_thr: float = 0.05
    theiler: str | int | None = "auto"
    max_points: int = 20000


@dataclass
class DMConfig:
    epsilon: float = 0.99


@dataclass
class ParamGridConfig:
    k_neighbors: list[int] = field(default_factory=lambda: [3, 5, 7])
    pers_delta: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75])


@dataclass
class HypertuneConfig:
    output_dir: str = "out/dm_hypertune"
    data: DataConfig = None
    takens: TakensConfig = None
    dm: DMConfig = None
    param_grid: ParamGridConfig = None
    save_plots: bool = True

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.takens is None:
            self.takens = TakensConfig()
        if self.dm is None:
            self.dm = DMConfig()
        if self.param_grid is None:
            self.param_grid = ParamGridConfig()


def load_config(config_path: str) -> HypertuneConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = DataConfig(**cfg.get("data", {}))

    takens_raw = cfg.get("takens", {})
    theiler_val = takens_raw.get("theiler", "auto")
    if theiler_val == "auto":
        theiler_val = "auto"
    elif theiler_val is not None:
        theiler_val = int(theiler_val)
    takens_raw["theiler"] = theiler_val
    takens_cfg = TakensConfig(**takens_raw)

    dm_cfg = DMConfig(**cfg.get("dm", {}))

    param_grid_raw = cfg.get("param_grid", {})
    param_grid_cfg = ParamGridConfig(
        k_neighbors=param_grid_raw.get("k_neighbors", [3, 5, 7]),
        pers_delta=param_grid_raw.get("pers_delta", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75]),
    )

    return HypertuneConfig(
        output_dir=cfg.get("output_dir", "out/dm_hypertune"),
        data=data_cfg,
        takens=takens_cfg,
        dm=dm_cfg,
        param_grid=param_grid_cfg,
        save_plots=cfg.get("save_plots", True),
    )


def load_data(config: DataConfig) -> np.ndarray:
    work_fn = functools.partial(
        get_aggregate_data, f"/{config.dm_key}", [], path=config.rodent_rollout_dir
    )
    with mp.Pool(processes=mp.cpu_count()) as pool:
        data_list = list(tqdm(pool.imap(work_fn, range(config.num_clips_total)), total=config.num_clips_total, desc="Loading data"))

    first_clip = data_list[0]
    if first_clip.ndim == 2:
        n_frames, n_dim = first_clip.shape
    else:
        total_size = first_clip.size
        n_dim = total_size // 500
        n_frames = 500

    data = np.array([
        clip.reshape(-1, n_dim) if clip.ndim != 2 else clip
        for clip in data_list
    ])

    min_frames = min(d.shape[0] for d in data)
    data = data[:, :min_frames, :]

    print(f"Loaded: {data.shape}")
    return data


def takens_embedding(X: np.ndarray, m: int, tau: int) -> np.ndarray:
    n_frames, D = X.shape
    start = (m - 1) * tau
    idxs = np.arange(start, n_frames)
    pts = np.stack([X[[j - a * tau for a in range(m)]].reshape(-1) for j in idxs], axis=0)
    return pts


def evaluate_skeleton_quality(
    cycle_inds: list,
    G_jc,
    pts: np.ndarray,
) -> dict:
    import networkx as nx

    metrics = {
        "num_cycles": len(cycle_inds),
        "num_edges": G_jc.number_of_edges(),
        "num_nodes": G_jc.number_of_nodes(),
    }

    if len(cycle_inds) > 0:
        cycle_lengths = [len(c) for c in cycle_inds]
        metrics["mean_cycle_length"] = float(np.mean(cycle_lengths))
        metrics["std_cycle_length"] = float(np.std(cycle_lengths))
        metrics["min_cycle_length"] = int(np.min(cycle_lengths))
        metrics["max_cycle_length"] = int(np.max(cycle_lengths))

        all_cycle_points = set()
        for cycle in cycle_inds:
            all_cycle_points.update(cycle)
        metrics["cycle_coverage"] = float(len(all_cycle_points) / len(pts))
    else:
        metrics["mean_cycle_length"] = 0.0
        metrics["std_cycle_length"] = 0.0
        metrics["min_cycle_length"] = 0
        metrics["max_cycle_length"] = 0
        metrics["cycle_coverage"] = 0.0

    if G_jc.number_of_nodes() > 0:
        metrics["is_connected"] = bool(nx.is_connected(G_jc))
        metrics["num_components"] = int(nx.number_connected_components(G_jc))
        degrees = [d for _, d in G_jc.degree()]
        metrics["mean_degree"] = float(np.mean(degrees))
        metrics["std_degree"] = float(np.std(degrees))
        metrics["max_degree"] = int(np.max(degrees)) if len(degrees) else 0
    else:
        metrics["is_connected"] = False
        metrics["num_components"] = 0
        metrics["mean_degree"] = 0.0
        metrics["std_degree"] = 0.0
        metrics["max_degree"] = 0

    metrics["edge_cycle_ratio"] = (
        float(metrics["num_edges"] / metrics["num_cycles"])
        if metrics["num_cycles"] > 0
        else 0.0
    )

    return metrics


def plot_longest_cycle_pca(
    pts: np.ndarray,
    cycle_inds: list[np.ndarray],
    save_path: str,
    figsize: tuple[int, int] = (8, 8),
) -> dict:
    """
    Project point cloud to 2D PCA and highlight the longest cycle's traversal.

    Args:
        pts: Point cloud array (N, D)
        cycle_inds: List of cycle index arrays
        save_path: Path to save the figure
        figsize: Figure size

    Returns:
        Dict with longest_cycle_length and pca variance info
    """
    if len(cycle_inds) == 0:
        return {"longest_cycle_length": 0, "pca_var_1": 0.0, "pca_var_2": 0.0}

    longest_cycle = max(cycle_inds, key=len)
    longest_len = len(longest_cycle)

    pca = PCA(n_components=2)
    pts_2d = pca.fit_transform(pts)
    var_explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        pts_2d[:, 0], pts_2d[:, 1],
        c="gray", s=5, alpha=0.3, label="All points"
    )

    cycle_pts = pts_2d[longest_cycle]
    ax.plot(
        cycle_pts[:, 0], cycle_pts[:, 1],
        c="red", linewidth=1.5, alpha=0.8, label=f"Longest cycle (n={longest_len})"
    )
    ax.scatter(
        cycle_pts[:, 0], cycle_pts[:, 1],
        c=np.arange(len(cycle_pts)), cmap="coolwarm", s=20, zorder=5
    )

    ax.scatter(cycle_pts[0, 0], cycle_pts[0, 1], c="green", s=100, marker="o", zorder=10, label="Start")
    ax.scatter(cycle_pts[-1, 0], cycle_pts[-1, 1], c="blue", s=100, marker="x", zorder=10, label="End")

    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% var)")
    ax.set_title(f"Longest Cycle Traversal (length={longest_len})")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {
        "longest_cycle_length": longest_len,
        "pca_var_1": float(var_explained[0]),
        "pca_var_2": float(var_explained[1]),
    }


def estimate_takens_params(
    config: HypertuneConfig,
    save_plot: bool = True,
) -> tuple[int, int, dict]:
    print("=== Auto-estimating Takens parameters (AMI + FNN) ===")

    tau_star, m_star, diagnostics = estimate_delay_embedding_params(
        rodent_rollout_dir=config.data.rodent_rollout_dir,
        get_aggregate_data_fn=get_aggregate_data,
        num_clips=config.data.num_clips_total,
        intention_key=f"/{config.data.dm_key}",
        clip_indices=range(config.data.num_clips_total),
        max_tau=config.takens.max_tau,
        bins=config.takens.bins,
        m_max=config.takens.m_max,
        R_tol=config.takens.R_tol,
        A_tol=config.takens.A_tol,
        fnn_thr=config.takens.fnn_thr,
        theiler=config.takens.theiler,
        max_points=config.takens.max_points,
        plot=False,
    )

    print(f"Estimated: tau* = {tau_star}, m* = {m_star}")

    if save_plot:
        fig = plt.figure(figsize=(12, 4.5))
        gs = fig.add_gridspec(1, 2, wspace=0.25)

        ax1 = fig.add_subplot(gs[0, 0])
        taus = np.arange(1, config.takens.max_tau + 1)
        I_mean = diagnostics["I_mean"]
        I_se = diagnostics["I_se"]
        ax1.plot(taus, I_mean, label="mean AMI")
        if np.all(np.isfinite(I_se)):
            ax1.fill_between(taus, I_mean - I_se, I_mean + I_se, alpha=0.2, linewidth=0)
        ax1.axvline(tau_star, ls="--", color="red", label=f"tau* = {tau_star}")
        ax1.set_xlabel("Delay tau")
        ax1.set_ylabel("AMI")
        ax1.set_title("Average Mutual Information")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ms = diagnostics["ms"]
        fnn_mean = diagnostics["fnn_mean"]
        fnn_se = diagnostics["fnn_se"]
        ax2.plot(ms, fnn_mean, marker="o", label="mean FNN")
        if np.all(np.isfinite(fnn_se)):
            ax2.fill_between(ms, fnn_mean - fnn_se, fnn_mean + fnn_se, alpha=0.2, linewidth=0)
        ax2.axhline(config.takens.fnn_thr, ls=":", color="gray", label=f"threshold = {config.takens.fnn_thr}")
        ax2.axvline(m_star, ls="--", color="red", label=f"m* = {m_star}")
        ax2.set_xlabel("Embedding dimension m")
        ax2.set_ylabel("Fraction of false neighbors")
        ax2.set_title(f"FNN at tau = {tau_star}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Delay-Embedding Parameter Selection (AMI -> tau*, FNN -> m*)")
        plt.tight_layout()

        plot_path = os.path.join(config.output_dir, "ami_fnn_estimation.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved AMI/FNN plot to {plot_path}")
        plt.close(fig)

    return tau_star, m_star, diagnostics


def run_grid_search(
    data: np.ndarray,
    config: HypertuneConfig,
) -> tuple[pd.DataFrame, dict | None, str]:
    os.makedirs(config.output_dir, exist_ok=True)

    selected = data[:config.data.num_clips_eval]
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    param_dict = {
        "k_neighbors": config.param_grid.k_neighbors,
        "pers_delta": config.param_grid.pers_delta,
    }
    grid = list(ParameterGrid(param_dict))

    results = []
    for i, params in enumerate(tqdm(grid, desc="Grid search")):
        k_neighbors = int(params["k_neighbors"])
        pers_delta = float(params["pers_delta"])
        run_dir = os.path.join(
            config.output_dir,
            f"run_{i:03d}_m{config.takens.m}_tau{config.takens.tau}_k{k_neighbors}_pd{pers_delta:.3f}",
        )
        os.makedirs(run_dir, exist_ok=True)

        for clip_idx, clip in enumerate(selected):
            clip_dir = os.path.join(run_dir, f"clip_{clip_idx:03d}")
            os.makedirs(clip_dir, exist_ok=True)
            try:
                pts = takens_embedding(clip, m=config.takens.m, tau=config.takens.tau)
                cycle_inds, G_jc = extract_skeleton_pcd_dm(
                    pts,
                    k=k_neighbors,
                    pers_delta=pers_delta,
                    epsilon=config.dm.epsilon,
                    output_dir=clip_dir,
                    cleanup=False,
                )
                cycle_inds = [np.array(inds, dtype=int) for inds in cycle_inds]
                metrics = evaluate_skeleton_quality(cycle_inds, G_jc, pts)

                pca_plot_path = os.path.join(clip_dir, "longest_cycle_pca.png")
                pca_metrics = plot_longest_cycle_pca(pts, cycle_inds, pca_plot_path)
                metrics.update(pca_metrics)

                metrics.update({
                    "m": config.takens.m,
                    "tau": config.takens.tau,
                    "k_neighbors": k_neighbors,
                    "pers_delta": pers_delta,
                })
                metrics["run_id"] = i
                metrics["clip_id"] = clip_idx
                metrics["output_dir"] = clip_dir
                metrics["success"] = True
                with open(os.path.join(clip_dir, "artifacts.pkl"), "wb") as f:
                    pkl.dump(
                        {"cycle_inds": cycle_inds, "G_jc": G_jc, "pts_shape": pts.shape}, f
                    )
            except Exception as e:
                print(f"  clip {clip_idx} failed: {e}")
                metrics = {
                    "m": config.takens.m,
                    "tau": config.takens.tau,
                    "k_neighbors": k_neighbors,
                    "pers_delta": pers_delta,
                    "run_id": i,
                    "clip_id": clip_idx,
                    "output_dir": clip_dir,
                    "success": False,
                    "error": str(e),
                    "longest_cycle_length": 0,
                    "pca_var_1": 0.0,
                    "pca_var_2": 0.0,
                }
            results.append(metrics)

    df = pd.DataFrame(results)
    ok = df[df["success"] == True].copy()

    best = None
    if len(ok) > 0:
        agg = ok.groupby(["k_neighbors", "pers_delta"], as_index=False).agg(
            num_cycles_mean=("num_cycles", "mean"),
            cycle_coverage_mean=("cycle_coverage", "mean"),
            num_edges_mean=("num_edges", "mean"),
            mean_cycle_length_mean=("mean_cycle_length", "mean"),
            num_components_mean=("num_components", "mean"),
            edge_cycle_ratio_mean=("edge_cycle_ratio", "mean"),
            longest_cycle_length_mean=("longest_cycle_length", "mean"),
            pca_var_1_mean=("pca_var_1", "mean"),
            pca_var_2_mean=("pca_var_2", "mean"),
        )
        best_row = agg.loc[agg["cycle_coverage_mean"].idxmax()]
        best = {
            "m": config.takens.m,
            "tau": config.takens.tau,
            "k_neighbors": int(best_row["k_neighbors"]),
            "pers_delta": float(best_row["pers_delta"]),
            "cycle_coverage": float(best_row["cycle_coverage_mean"]),
        }

    run_base = os.path.join(config.output_dir, f"tuning_results_{run_ts}")
    df.to_csv(run_base + ".csv", index=False)
    with open(run_base + ".pkl", "wb") as f:
        pkl.dump({"results_df": df, "best": best, "config": config}, f)
    if best is not None:
        with open(os.path.join(config.output_dir, f"best_params_{run_ts}.json"), "w") as f:
            json.dump(best, f, indent=2)

    return df, best, run_ts


def plot_tuning_results(
    results_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    df = results_df[results_df["success"] == True].copy()

    metrics = [
        ("num_cycles", "Number of Cycles"),
        ("cycle_coverage", "Cycle Coverage"),
        ("num_edges", "Number of Edges"),
        ("mean_cycle_length", "Mean Cycle Length"),
        ("num_components", "Number of Components"),
        ("edge_cycle_ratio", "Edge/Cycle Ratio"),
    ]

    unique_k = sorted(df["k_neighbors"].unique())
    palette = sns.color_palette("tab10", len(unique_k))
    cmap = {k: palette[i] for i, k in enumerate(unique_k)}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (mname, title) in zip(axes.flat, metrics):
        for k in unique_k:
            sub = df[df["k_neighbors"] == k]
            g = (
                sub.groupby("pers_delta")[mname]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("pers_delta")
            )
            x = g["pers_delta"].values
            y = g["mean"].values
            yerr = g["std"].values
            ax.errorbar(
                x, y, yerr=yerr, fmt="o-", alpha=0.9, label=f"k={k}", color=cmap[k], capsize=3
            )
        ax.set_xlabel("pers_delta")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    handles, labels = axes.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="k", loc="upper right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
        plt.close(fig)

    return fig


def run_hypertune(config: HypertuneConfig) -> tuple[pd.DataFrame, dict | None]:
    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("DM Hyperparameter Tuning")
    print("=" * 60)

    if config.takens.auto_estimate:
        print("\n=== Stage 1: Auto-estimating Takens parameters ===")
        tau_star, m_star, diagnostics = estimate_takens_params(
            config,
            save_plot=config.save_plots,
        )
        config.takens.tau = tau_star
        config.takens.m = m_star

        diag_path = os.path.join(config.output_dir, "ami_fnn_diagnostics.pkl")
        with open(diag_path, "wb") as f:
            pkl.dump({"tau_star": tau_star, "m_star": m_star, "diagnostics": diagnostics}, f)
        print(f"Saved diagnostics to {diag_path}")
    else:
        print("\n=== Stage 1: Using fixed Takens parameters ===")
        print(f"tau = {config.takens.tau}, m = {config.takens.m}")

    print(f"\n=== Stage 2: Loading data ===")
    data = load_data(config.data)

    print(f"\n=== Stage 3: Grid search (m={config.takens.m}, tau={config.takens.tau}) ===")
    print(f"Evaluating on {config.data.num_clips_eval} clips")
    print(f"k_neighbors: {config.param_grid.k_neighbors}")
    print(f"pers_delta: {config.param_grid.pers_delta}")

    results_df, best_params, ts = run_grid_search(data, config)

    if config.save_plots:
        print("\n=== Stage 4: Visualization ===")
        viz_path = os.path.join(config.output_dir, f"tuning_results_{ts}.png")
        plot_tuning_results(results_df, viz_path)

    print("\n" + "=" * 50)
    print("TUNING COMPLETE")
    print("=" * 50)

    if best_params:
        print(
            f"Best: m={int(best_params['m'])} tau={int(best_params['tau'])} "
            f"k={int(best_params['k_neighbors'])} pers_delta={float(best_params['pers_delta']):.3f}"
        )
        if "cycle_coverage" in best_params:
            print(f"Avg Cycle Coverage: {best_params['cycle_coverage']:.3f}")
        print(f"Artifacts in: {config.output_dir}")

    return results_df, best_params


def main(config_path: str = "configs/dmwkc/dm_hypertune.yaml"):
    if os.path.exists(config_path):
        cfg = load_config(config_path)
    else:
        cfg = HypertuneConfig()

    run_hypertune(cfg)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dmwkc/dm_hypertune.yaml"
    main(config_path)
