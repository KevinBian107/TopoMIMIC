import os
import sys
import json
import numpy as np
import pandas as pd
import multiprocessing as mp
import functools
import pickle as pkl
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

from vnl_tda.utils.load_utils import get_aggregate_data
from vnl_tda.pipelines.tda.skeletonization.skeletonization import extract_skeleton_pcd_dm


@dataclass
class TakensConfig:
    m: int = 3
    tau: int = 6


@dataclass
class DMConfig:
    epsilon: float = 0.99


@dataclass
class ScoringConfig:
    cycle_coverage_weight: float = 0.3
    num_cycles_weight: float = 0.2
    connectivity_weight: float = 0.2
    mean_cycle_length_weight: float = 0.15
    cycle_length_std_weight: float = 0.15


@dataclass
class VisualizationConfig:
    save_plots: bool = True
    plot_by_k: bool = True
    plot_by_clip: bool = True


@dataclass
class HypertuneConfig:
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    output_dir: str = "dm_tuning_takens_m3_tau6"
    num_clips_total: int = 800
    num_clips_eval: int = 10
    takens: TakensConfig = None
    dm: DMConfig = None
    param_grid: dict = None
    scoring: ScoringConfig = None
    visualization: VisualizationConfig = None

    def __post_init__(self):
        if self.takens is None:
            self.takens = TakensConfig()
        if self.dm is None:
            self.dm = DMConfig()
        if self.param_grid is None:
            self.param_grid = {
                "k_neighbors": [3, 5, 7],
                "pers_delta": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75],
            }
        if self.scoring is None:
            self.scoring = ScoringConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()


def load_config(config_path: str) -> HypertuneConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    takens_cfg = TakensConfig(**cfg.get("takens", {}))
    dm_cfg = DMConfig(**cfg.get("dm", {}))
    scoring_cfg = ScoringConfig(**cfg.get("scoring", {}))
    viz_cfg = VisualizationConfig(**cfg.get("visualization", {}))

    return HypertuneConfig(
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", HypertuneConfig.rodent_rollout_dir),
        output_dir=cfg.get("output_dir", HypertuneConfig.output_dir),
        num_clips_total=cfg.get("num_clips_total", HypertuneConfig.num_clips_total),
        num_clips_eval=cfg.get("num_clips_eval", HypertuneConfig.num_clips_eval),
        takens=takens_cfg,
        dm=dm_cfg,
        param_grid=cfg.get("param_grid", None),
        scoring=scoring_cfg,
        visualization=viz_cfg,
    )


def takens_points(X: np.ndarray, m: int, tau: int) -> np.ndarray:
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

    metrics = {}
    metrics["num_cycles"] = len(cycle_inds)
    metrics["num_edges"] = G_jc.number_of_edges()
    metrics["num_nodes"] = G_jc.number_of_nodes()

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


def compute_score(
    df: pd.DataFrame,
    scoring_cfg: ScoringConfig,
) -> pd.DataFrame:
    ok = df[df["success"] == True].copy()
    if len(ok) == 0:
        return ok

    nc_max = max(1, int(ok["num_cycles"].max()))
    comp_max = max(1, int(ok["num_components"].max()))
    mcl_max = float(max(1e-9, ok["mean_cycle_length"].replace(0, np.nan).max()))
    scl_max = float(max(1e-9, ok["std_cycle_length"].replace(0, np.nan).max()))

    ok["score"] = (
        scoring_cfg.cycle_coverage_weight * ok["cycle_coverage"].fillna(0)
        + scoring_cfg.num_cycles_weight * (ok["num_cycles"] / nc_max)
        + scoring_cfg.connectivity_weight * (1 - ok["num_components"] / comp_max)
        + scoring_cfg.mean_cycle_length_weight * (ok["mean_cycle_length"] / mcl_max)
        + scoring_cfg.cycle_length_std_weight * (1 - ok["std_cycle_length"] / scl_max)
    )
    return ok


def grid_search(
    int_rodent: np.ndarray,
    cfg: HypertuneConfig,
) -> tuple[pd.DataFrame, dict | None, str]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    selected = int_rodent[: cfg.num_clips_eval]
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid = list(ParameterGrid(cfg.param_grid))

    results = []
    for i, params in enumerate(tqdm(grid, desc="Grid search")):
        k_neighbors = int(params["k_neighbors"])
        pers_delta = float(params["pers_delta"])
        run_dir = os.path.join(
            cfg.output_dir,
            f"run_{i:03d}_m{cfg.takens.m}_tau{cfg.takens.tau}_k{k_neighbors}_pd{pers_delta:.3f}",
        )
        os.makedirs(run_dir, exist_ok=True)

        for clip_idx, clip in enumerate(selected):
            clip_dir = os.path.join(run_dir, f"clip_{clip_idx:03d}")
            os.makedirs(clip_dir, exist_ok=True)
            try:
                pts = takens_points(clip, m=cfg.takens.m, tau=cfg.takens.tau)
                cycle_inds, G_jc = extract_skeleton_pcd_dm(
                    pts,
                    k=k_neighbors,
                    pers_delta=pers_delta,
                    epsilon=cfg.dm.epsilon,
                    output_dir=clip_dir,
                    cleanup=False,
                )
                cycle_inds = [np.array(inds, dtype=int) for inds in cycle_inds]
                metrics = evaluate_skeleton_quality(cycle_inds, G_jc, pts)
                metrics.update({
                    "m": cfg.takens.m,
                    "tau": cfg.takens.tau,
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
                    "m": cfg.takens.m,
                    "tau": cfg.takens.tau,
                    "k_neighbors": k_neighbors,
                    "pers_delta": pers_delta,
                    "run_id": i,
                    "clip_id": clip_idx,
                    "output_dir": clip_dir,
                    "success": False,
                    "error": str(e),
                }
            results.append(metrics)

    df = pd.DataFrame(results)
    ok = compute_score(df, cfg.scoring)

    best = None
    if len(ok) > 0:
        agg = ok.groupby(["k_neighbors", "pers_delta"], as_index=False).agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            num_cycles_mean=("num_cycles", "mean"),
            cycle_coverage_mean=("cycle_coverage", "mean"),
            num_edges_mean=("num_edges", "mean"),
            mean_cycle_length_mean=("mean_cycle_length", "mean"),
            num_components_mean=("num_components", "mean"),
            edge_cycle_ratio_mean=("edge_cycle_ratio", "mean"),
        )
        best_row = agg.loc[agg["score_mean"].idxmax()]
        best = {
            "m": cfg.takens.m,
            "tau": cfg.takens.tau,
            "k_neighbors": int(best_row["k_neighbors"]),
            "pers_delta": float(best_row["pers_delta"]),
            "score": float(best_row["score_mean"]),
        }

    run_base = os.path.join(cfg.output_dir, f"tuning_results_{run_ts}")
    df.to_csv(run_base + ".csv", index=False)
    with open(run_base + ".pkl", "wb") as f:
        pkl.dump({"results_df": df, "best": best, "config": cfg}, f)
    if best is not None:
        with open(os.path.join(cfg.output_dir, f"best_params_{run_ts}.json"), "w") as f:
            json.dump(best, f, indent=2)

    return df, best, run_ts


def plot_tuning_by_k(
    results_df: pd.DataFrame,
    save_path: str | None = None,
    metrics: list | None = None,
) -> plt.Figure:
    df = results_df[results_df["success"] == True].copy()
    if metrics is None:
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


def plot_tuning_by_clip(
    results_df: pd.DataFrame,
    save_path: str | None = None,
    metrics: list | None = None,
) -> plt.Figure:
    df = results_df[results_df["success"] == True].copy()
    if metrics is None:
        metrics = [
            ("num_cycles", "Number of Cycles"),
            ("cycle_coverage", "Cycle Coverage"),
            ("num_edges", "Number of Edges"),
            ("mean_cycle_length", "Mean Cycle Length"),
            ("num_components", "Number of Components"),
            ("edge_cycle_ratio", "Edge/Cycle Ratio"),
        ]

    unique_clips = sorted(df["clip_id"].unique())
    unique_k = sorted(df["k_neighbors"].unique())
    palette = sns.color_palette("tab10", len(unique_clips))
    color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(unique_clips)}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (metric, title) in zip(axes.flat, metrics):
        labeled = set()
        for cid in unique_clips:
            for k in unique_k:
                subset = df[(df["clip_id"] == cid) & (df["k_neighbors"] == k)].sort_values(
                    "pers_delta"
                )
                if subset.empty:
                    continue
                x = subset["pers_delta"].values
                y = subset[metric].values
                lbl = f"clip {cid}" if cid not in labeled else None
                ax.plot(x, y, "o-", color=color_map[cid], alpha=0.9, label=lbl)
                if lbl is not None:
                    labeled.add(cid)
        ax.set_xlabel("pers_delta")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    handles, labels = axes.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="clip", loc="upper right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
        plt.close(fig)
    return fig


def load_latest_results(output_dir: str) -> tuple[pd.DataFrame, dict | None]:
    files = [
        f for f in os.listdir(output_dir)
        if f.startswith("tuning_results_") and f.endswith(".pkl")
    ]
    if not files:
        raise FileNotFoundError(f"No tuning results found in {output_dir}")

    latest_file = sorted(files)[-1]
    print(f"Loading: {latest_file}")

    with open(os.path.join(output_dir, latest_file), "rb") as f:
        data = pkl.load(f)

    return data["results_df"], data.get("best")


def main(config_path: str = "configs/dmwkc/dm_latent_subspace_hypertune.yaml"):
    cfg = load_config(config_path)

    sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=== Stage 1: Loading intention data ===")
    work = functools.partial(get_aggregate_data, "/intention", [], path=cfg.rodent_rollout_dir)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        arrs = list(tqdm(pool.imap(work, range(cfg.num_clips_total)), total=cfg.num_clips_total))
    int_rodent = np.vstack(arrs)
    int_rodent = int_rodent.reshape(cfg.num_clips_total, 499, 16)
    print(f"Loaded intention: {int_rodent.shape}")

    print("\n=== Stage 2: Grid search ===")
    print(f"Evaluating on {cfg.num_clips_eval} clips")
    print(f"Parameter grid: {cfg.param_grid}")

    results_df, best_params, ts = grid_search(int_rodent, cfg)

    print("\n=== Stage 3: Visualization ===")
    if cfg.visualization.save_plots:
        if cfg.visualization.plot_by_k:
            viz_path_k = os.path.join(cfg.output_dir, f"tuning_by_k_{ts}.png")
            plot_tuning_by_k(results_df, viz_path_k)

        if cfg.visualization.plot_by_clip:
            viz_path_clip = os.path.join(cfg.output_dir, f"tuning_by_clip_{ts}.png")
            plot_tuning_by_clip(results_df, viz_path_clip)

    print("\n" + "=" * 50)
    print("TUNING COMPLETE")
    print("=" * 50)
    if best_params:
        print(
            f"Best: m={int(best_params['m'])} tau={int(best_params['tau'])} "
            f"k={int(best_params['k_neighbors'])} pers_delta={float(best_params['pers_delta']):.3f}"
        )
        if "score" in best_params:
            print(f"Avg Score: {best_params['score']:.3f}")
        print(f"Artifacts in: {cfg.output_dir}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dmwkc/dm_latent_subspace_hypertune.yaml"
    main(config_path)
