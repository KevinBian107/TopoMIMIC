import os
import sys
import json
import numpy as np
import pandas as pd
import multiprocessing as mp
import functools
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import ParameterGrid
from vnl_tda.utils.load_utils import get_aggregate_data
from vnl_tda.pipelines.tda.skeletonization.skeletonization import extract_skeleton_pcd_dm

M_FIXED = 3
TAU_FIXED = 6

def evaluate_skeleton_quality(cycle_inds, G_jc, pts):
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
        import networkx as nx
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
    metrics["edge_cycle_ratio"] = float(metrics["num_edges"] / metrics["num_cycles"]) if metrics["num_cycles"] > 0 else 0.0
    return metrics

def takens_points(X, m=M_FIXED, tau=TAU_FIXED):
    n_frames, D = X.shape
    start = (m - 1) * tau
    idxs = np.arange(start, n_frames)
    pts = np.stack([X[[j - a * tau for a in range(m)]].reshape(-1) for j in idxs], axis=0)
    return pts

def grid_search(int_rodent, param_grid, epsilon, num_clips=10, outdir="dm_tuning_k", verbose=True):
    os.makedirs(outdir, exist_ok=True)
    selected = int_rodent[:num_clips]
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid = list(ParameterGrid(param_grid))
    results = []
    for i, params in enumerate(grid):
        k_neighbors = int(params["k_neighbors"])
        pers_delta = float(params["pers_delta"])
        run_dir = os.path.join(outdir, f"run_{i:03d}_m{M_FIXED}_tau{TAU_FIXED}_k{k_neighbors}_pd{pers_delta:.3f}")
        os.makedirs(run_dir, exist_ok=True)
        if verbose:
            print(f"[{i+1}/{len(grid)}] k={k_neighbors} pd={pers_delta:.3f}")
        for clip_idx, clip in enumerate(selected):
            clip_dir = os.path.join(run_dir, f"clip_{clip_idx:03d}")
            os.makedirs(clip_dir, exist_ok=True)
            try:
                pts = takens_points(clip, m=M_FIXED, tau=TAU_FIXED)
                cycle_inds, G_jc = extract_skeleton_pcd_dm(
                    pts, k=k_neighbors, pers_delta=pers_delta, epsilon=epsilon, output_dir=clip_dir, cleanup=False
                )
                cycle_inds = [np.array(inds, dtype=int) for inds in cycle_inds]
                metrics = evaluate_skeleton_quality(cycle_inds, G_jc, pts)
                metrics.update({"m": M_FIXED, "tau": TAU_FIXED, "k_neighbors": k_neighbors, "pers_delta": pers_delta})
                metrics["run_id"] = i
                metrics["clip_id"] = clip_idx
                metrics["output_dir"] = clip_dir
                metrics["success"] = True
                with open(os.path.join(clip_dir, "artifacts.pkl"), "wb") as f:
                    pkl.dump({"cycle_inds": cycle_inds, "G_jc": G_jc, "pts_shape": pts.shape}, f)
            except Exception as e:
                if verbose:
                    print(f"  clip {clip_idx} failed: {e}")
                metrics = {"m": M_FIXED, "tau": TAU_FIXED, "k_neighbors": k_neighbors, "pers_delta": pers_delta, "run_id": i, "clip_id": clip_idx, "output_dir": clip_dir, "success": False, "error": str(e)}
            results.append(metrics)
    df = pd.DataFrame(results)
    ok = df[df["success"] == True].copy()
    if len(ok) > 0:
        nc_max   = max(1,        int(ok["num_cycles"].max()))
        comp_max = max(1,        int(ok["num_components"].max()))
        mcl_max  = float(max(1e-9, ok["mean_cycle_length"].replace(0, np.nan).max()))
        scl_max  = float(max(1e-9, ok["std_cycle_length"].replace(0, np.nan).max()))
        ok["score"] = (
            0.3  * ok["cycle_coverage"].fillna(0)
            + 0.2 * (ok["num_cycles"] / nc_max)
            + 0.2 * (1 - ok["num_components"] / comp_max)
            + 0.15 * (ok["mean_cycle_length"] / mcl_max)
            + 0.15 * (1 - ok["std_cycle_length"] / scl_max)
        )
        agg = ok.groupby(["k_neighbors","pers_delta"], as_index=False).agg(
            score_mean=("score","mean"),
            score_std =("score","std"),
            num_cycles_mean=("num_cycles","mean"),
            cycle_coverage_mean=("cycle_coverage","mean"),
            num_edges_mean=("num_edges","mean"),
            mean_cycle_length_mean=("mean_cycle_length","mean"),
            num_components_mean=("num_components","mean"),
            edge_cycle_ratio_mean=("edge_cycle_ratio","mean")
        )
        best_row = agg.loc[agg["score_mean"].idxmax()]
        best = {
            "m": M_FIXED,
            "tau": TAU_FIXED,
            "k_neighbors": int(best_row["k_neighbors"]),
            "pers_delta": float(best_row["pers_delta"]),
            "score": float(best_row["score_mean"])
        }
    else:
        best = None
    run_base = os.path.join(outdir, f"tuning_results_{run_ts}")
    df.to_csv(run_base + ".csv", index=False)
    with open(run_base + ".pkl", "wb") as f:
        pkl.dump({"results_df": df, "best": best}, f)
    if best is not None:
        with open(os.path.join(outdir, f"best_params_{run_ts}.json"), "w") as f:
            json.dump(best, f, indent=2)
    return df, best, run_ts

def visualize_tuning_by_k(results_df, save_path=None, metrics=None):
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
            g = sub.groupby("pers_delta")[mname].agg(["mean","std"]).reset_index().sort_values("pers_delta")
            x = g["pers_delta"].values
            y = g["mean"].values
            yerr = g["std"].values
            ax.errorbar(x, y, yerr=yerr, fmt="o-", alpha=0.9, label=f"k={k}", color=cmap[k], capsize=3)
        ax.set_xlabel("pers_delta")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="k", loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig

def load_intention(num_clips=800, path="/root/vast/kaiwen/TopoVNL/data/rodent_ar"):
    work = functools.partial(get_aggregate_data, "/intention", [], path=path)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        arrs = list(
            mp.pool.Pool.imap_unordered(pool, work, range(num_clips))
            if hasattr(mp.pool, "Pool") else []
        )
    if not arrs:
        with mp.Pool(processes=mp.cpu_count()) as pool2:
            arrs = list(pool2.imap(work, range(num_clips)))
    X = np.vstack(arrs)
    X = X.reshape(num_clips, 499, 16)
    return X

def main():
    sys.path.append("/root/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")
    outdir = "dm_tuning_takens_m3_tau6_10clips"
    os.makedirs(outdir, exist_ok=True)
    int_rodent = load_intention(num_clips=800, path="/root/vast/kaiwen/TopoVNL/data/rodent_ar")
    base_epsilon = 0.99
    param_grid = {
        "pers_delta": np.linspace(0.1, 0.75, 10),
        "k_neighbors": [3],
    }
    print("Starting per-clip hyperparameter tuning (Takens m=3, tau=6) and averaging across clips...")
    results_df, best_params, ts = grid_search(
        int_rodent=int_rodent,
        param_grid=param_grid,
        epsilon=base_epsilon,
        num_clips=10,
        outdir=outdir,
        verbose=True,
    )
    viz_path = os.path.join(outdir, f"tuning_viz_{ts}.png")
    visualize_tuning_by_k(results_df, viz_path)
    print("=" * 50)
    print("TUNING COMPLETE")
    print("=" * 50)
    if best_params:
        print(f"Best: m={int(best_params['m'])} tau={int(best_params['tau'])} k={int(best_params['k_neighbors'])} pers_delta={float(best_params['pers_delta']):.3f}")
        if "score" in best_params:
            print(f"Avg Score: {best_params['score']:.3f}")
        print(f"Artifacts in: {outdir}")

if __name__ == "__main__":
    main()