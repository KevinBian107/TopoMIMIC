"""
DM Dimensionality Reduction Effect Analysis.

Empirically tests why DM finds cycles in PCA-reduced space but not in raw high-D space.

Hypothesis (from TopoMimic_Proof.pdf, Corollary 1 & 2):
- DM detects cycles when multiple basins are "sufficiently close" to form crossover routes
- In high-D space, curse of dimensionality spreads points apart → basins too far → tree-like → H₁=0
- PCA compression brings basins into spatial proximity → crossover routes → H₁≠0 → cycles

Metrics computed:
1. k-NN distances (neighborhood density)
2. Basin proximity via local density minima detection
3. Crossover route counting between basins in k-NN graph
4. Local Betti numbers (β₁) via persistent homology

Usage:
    python scripts/dm_dim_reduction_effect.py [config.yaml]
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import pickle as pkl
import yaml
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MeanShift
import networkx as nx
from tqdm import tqdm

try:
    import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("Warning: ripser not available, skipping local PH computation")


@dataclass
class AnalysisConfig:
    dm_key: str = "ekp"
    pca_dims_list: list[int] = field(default_factory=lambda: [0, 5, 10, 15, 20, 30])
    k_neighbors: int = 10
    n_sample_points: int = 50
    neighborhood_radius_percentile: float = 10.0
    rips_max_dim: int = 1
    persistence_threshold: float = 0.1
    output_dir: str = "out/dm_dim_reduction_effect"
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar.h5"
    split: str = "train"
    xml_path: str = "topo_mimic/dmwk/xml/rodent.xml"
    stac_data_path: str = "data/rodent_reference_clips.h5"
    takens_m: int = 3
    takens_tau: int = 6
    max_clips: int = 10
    dpi: int = 150


def load_config(config_path: str) -> AnalysisConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "pca_dims_list" in cfg and isinstance(cfg["pca_dims_list"], list):
        cfg["pca_dims_list"] = [int(x) for x in cfg["pca_dims_list"]]

    return AnalysisConfig(**cfg)


def load_data(config: AnalysisConfig) -> tuple[np.ndarray, list[int]]:
    from topo_mimic.dmwk.dm_pipeline import DataConfig, load_data as dm_load_data

    data_cfg = DataConfig(
        dm_key=config.dm_key,
        rodent_rollout_dir=config.rodent_rollout_dir,
        split=config.split,
        pca_dims=None,
        xml_path=config.xml_path,
        stac_data_path=config.stac_data_path,
    )

    data, clip_indices = dm_load_data(data_cfg, output_dir=config.output_dir)

    if config.max_clips > 0 and len(data) > config.max_clips:
        data = data[:config.max_clips]
        clip_indices = clip_indices[:config.max_clips]

    return data, clip_indices


def takens_embedding(X: np.ndarray, m: int, tau: int) -> np.ndarray:
    n_frames, D = X.shape
    start = (m - 1) * tau
    idxs = np.arange(start, n_frames)
    pts = np.stack([X[[j - a * tau for a in range(m)]].reshape(-1) for j in idxs], axis=0)
    return pts


def apply_pca(pts: np.ndarray, n_dims: int) -> tuple[np.ndarray, float]:
    if n_dims <= 0 or n_dims >= pts.shape[1]:
        return pts, 1.0

    pca = PCA(n_components=n_dims)
    pts_reduced = pca.fit_transform(pts)
    var_explained = pca.explained_variance_ratio_.sum()
    return pts_reduced, var_explained


def compute_knn_distances(pts: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(pts)
    distances, _ = nn.kneighbors(pts)
    return distances[:, 1:].mean(axis=1)


def estimate_diameter(pts: np.ndarray, n_samples: int = 200) -> float:
    n = len(pts)
    if n <= n_samples:
        idx = np.arange(n)
    else:
        idx = np.random.choice(n, n_samples, replace=False)

    sample_pts = pts[idx]
    dists = np.linalg.norm(sample_pts[:, None, :] - sample_pts[None, :, :], axis=2)
    return float(dists.max())


def compute_normalized_knn_distances(pts: np.ndarray, k: int) -> dict:
    knn_dists = compute_knn_distances(pts, k)
    diameter = estimate_diameter(pts)

    return {
        "mean_knn_dist": float(knn_dists.mean()),
        "median_knn_dist": float(np.median(knn_dists)),
        "std_knn_dist": float(knn_dists.std()),
        "normalized_mean": float(knn_dists.mean() / diameter) if diameter > 0 else 0,
        "diameter": float(diameter),
        "knn_dists": knn_dists,
    }


def find_local_minima_meanshift(
    pts: np.ndarray,
    neighborhood_mask: np.ndarray,
    bandwidth: float | None = None,
) -> list[int]:
    import warnings

    local_pts = pts[neighborhood_mask]
    if len(local_pts) < 3:
        return []

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=False)
            ms.fit(local_pts)
        cluster_centers = ms.cluster_centers_

        minima_indices = []
        for center in cluster_centers:
            dists = np.linalg.norm(pts[neighborhood_mask] - center, axis=1)
            local_min_idx = np.argmin(dists)
            global_idx = np.where(neighborhood_mask)[0][local_min_idx]
            minima_indices.append(global_idx)

        return list(set(minima_indices))
    except Exception:
        return []


def count_crossover_routes(
    G: nx.Graph,
    source: int,
    target: int,
    max_routes: int = 10,
) -> int:
    if not G.has_node(source) or not G.has_node(target):
        return 0

    try:
        paths = list(nx.node_disjoint_paths(G, source, target))
        return min(len(paths), max_routes)
    except (nx.NetworkXNoPath, nx.NetworkXError):
        return 0


def compute_local_betti(
    pts: np.ndarray,
    neighborhood_mask: np.ndarray,
    max_dim: int = 1,
    persistence_threshold: float = 0.1,
    min_points: int = 20,
) -> dict:
    import warnings

    if not HAS_RIPSER:
        return {"beta_0": 0, "beta_1": 0, "significant_h1": 0}

    local_pts = pts[neighborhood_mask]
    n_local = len(local_pts)
    n_dims = local_pts.shape[1] if local_pts.ndim == 2 else 1

    if n_local < min_points or n_local < n_dims:
        return {"beta_0": n_local, "beta_1": 0, "significant_h1": 0}

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            result = ripser.ripser(local_pts, maxdim=max_dim)
        diagrams = result["dgms"]

        h0 = diagrams[0]
        beta_0 = np.sum(np.isinf(h0[:, 1]))

        beta_1 = 0
        significant_h1 = 0
        if len(diagrams) > 1:
            h1 = diagrams[1]
            if len(h1) > 0:
                persistences = h1[:, 1] - h1[:, 0]
                persistences = persistences[np.isfinite(persistences)]
                beta_1 = len(persistences)
                diameter = np.max(local_pts) - np.min(local_pts)
                if diameter > 0:
                    significant_h1 = np.sum(persistences / diameter > persistence_threshold)

        return {
            "beta_0": int(beta_0),
            "beta_1": int(beta_1),
            "significant_h1": int(significant_h1),
        }
    except Exception:
        return {"beta_0": 0, "beta_1": 0, "significant_h1": 0}


def estimate_neighborhood_radius(pts: np.ndarray, percentile: float, n_samples: int = 300) -> float:
    n = len(pts)
    if n <= n_samples:
        idx = np.arange(n)
    else:
        idx = np.random.choice(n, n_samples, replace=False)

    sample_pts = pts[idx]
    dists = np.linalg.norm(sample_pts[:, None, :] - sample_pts[None, :, :], axis=2)
    dists_flat = dists[dists > 0]
    return float(np.percentile(dists_flat, percentile))


def analyze_clip_at_pca_dim_fast(
    pts_work: np.ndarray,
    config: AnalysisConfig,
    var_explained: float,
) -> dict:
    """Analyze clip with already PCA-transformed points."""
    knn_stats = compute_normalized_knn_distances(pts_work, config.k_neighbors)

    n_pts = len(pts_work)
    sample_indices = np.random.choice(n_pts, min(config.n_sample_points, n_pts), replace=False)

    radius = estimate_neighborhood_radius(pts_work, config.neighborhood_radius_percentile)

    nn = NearestNeighbors(n_neighbors=config.k_neighbors + 1, algorithm="auto")
    nn.fit(pts_work)

    basin_counts = []
    basin_distances = []
    crossover_counts = []
    betti_numbers = []

    knn_graph = nx.Graph()
    knn_graph.add_nodes_from(range(n_pts))
    _, indices = nn.kneighbors(pts_work)
    for i in range(n_pts):
        for j in indices[i, 1:]:
            knn_graph.add_edge(i, j)

    for center_idx in sample_indices:
        center = pts_work[center_idx]
        dists_to_center = np.linalg.norm(pts_work - center, axis=1)
        neighborhood_mask = dists_to_center < radius

        if neighborhood_mask.sum() < 5:
            continue

        local_knn_dists = knn_stats["knn_dists"][neighborhood_mask]
        bandwidth = np.median(local_knn_dists) if len(local_knn_dists) > 0 else None
        minima = find_local_minima_meanshift(pts_work, neighborhood_mask, bandwidth)

        basin_counts.append(len(minima))

        if len(minima) >= 2:
            for i in range(len(minima)):
                for j in range(i + 1, len(minima)):
                    dist = np.linalg.norm(pts_work[minima[i]] - pts_work[minima[j]])
                    basin_distances.append(dist)

                    routes = count_crossover_routes(knn_graph, minima[i], minima[j])
                    crossover_counts.append(routes)

        betti = compute_local_betti(
            pts_work, neighborhood_mask,
            max_dim=config.rips_max_dim,
            persistence_threshold=config.persistence_threshold,
        )
        betti_numbers.append(betti)

    return {
        "var_explained": float(var_explained),
        "knn_mean": knn_stats["mean_knn_dist"],
        "knn_median": knn_stats["median_knn_dist"],
        "knn_normalized": knn_stats["normalized_mean"],
        "diameter": knn_stats["diameter"],
        "basin_count_mean": float(np.mean(basin_counts)) if basin_counts else 0,
        "basin_count_std": float(np.std(basin_counts)) if basin_counts else 0,
        "basin_dist_mean": float(np.mean(basin_distances)) if basin_distances else 0,
        "basin_dist_normalized": float(np.mean(basin_distances) / knn_stats["diameter"]) if basin_distances and knn_stats["diameter"] > 0 else 0,
        "crossover_mean": float(np.mean(crossover_counts)) if crossover_counts else 0,
        "crossover_multi_route_frac": float(np.mean([c >= 2 for c in crossover_counts])) if crossover_counts else 0,
        "beta1_mean": float(np.mean([b["beta_1"] for b in betti_numbers])) if betti_numbers else 0,
        "significant_h1_mean": float(np.mean([b["significant_h1"] for b in betti_numbers])) if betti_numbers else 0,
        "h1_positive_frac": float(np.mean([b["beta_1"] > 0 for b in betti_numbers])) if betti_numbers else 0,
    }


def analyze_clip_at_pca_dim(
    pts: np.ndarray,
    config: AnalysisConfig,
    pca_dim: int,
) -> dict:
    """Legacy function - applies PCA per-clip (slower)."""
    pts_work, var_explained = apply_pca(pts, pca_dim)

    knn_stats = compute_normalized_knn_distances(pts_work, config.k_neighbors)

    n_pts = len(pts_work)
    sample_indices = np.random.choice(n_pts, min(config.n_sample_points, n_pts), replace=False)

    radius = estimate_neighborhood_radius(pts_work, config.neighborhood_radius_percentile)

    nn = NearestNeighbors(n_neighbors=config.k_neighbors + 1, algorithm="auto")
    nn.fit(pts_work)

    basin_counts = []
    basin_distances = []
    crossover_counts = []
    betti_numbers = []

    knn_graph = nx.Graph()
    knn_graph.add_nodes_from(range(n_pts))
    _, indices = nn.kneighbors(pts_work)
    for i in range(n_pts):
        for j in indices[i, 1:]:
            knn_graph.add_edge(i, j)

    for center_idx in sample_indices:
        center = pts_work[center_idx]
        dists_to_center = np.linalg.norm(pts_work - center, axis=1)
        neighborhood_mask = dists_to_center < radius

        if neighborhood_mask.sum() < 5:
            continue

        local_knn_dists = knn_stats["knn_dists"][neighborhood_mask]
        bandwidth = np.median(local_knn_dists) if len(local_knn_dists) > 0 else None
        minima = find_local_minima_meanshift(pts_work, neighborhood_mask, bandwidth)

        basin_counts.append(len(minima))

        if len(minima) >= 2:
            for i in range(len(minima)):
                for j in range(i + 1, len(minima)):
                    dist = np.linalg.norm(pts_work[minima[i]] - pts_work[minima[j]])
                    basin_distances.append(dist)

                    routes = count_crossover_routes(knn_graph, minima[i], minima[j])
                    crossover_counts.append(routes)

        betti = compute_local_betti(
            pts_work, neighborhood_mask,
            max_dim=config.rips_max_dim,
            persistence_threshold=config.persistence_threshold,
        )
        betti_numbers.append(betti)

    result = {
        "pca_dim": pca_dim if pca_dim > 0 else pts.shape[1],
        "actual_dim": pts_work.shape[1],
        "var_explained": float(var_explained),
        "knn_mean": knn_stats["mean_knn_dist"],
        "knn_median": knn_stats["median_knn_dist"],
        "knn_normalized": knn_stats["normalized_mean"],
        "diameter": knn_stats["diameter"],
        "basin_count_mean": float(np.mean(basin_counts)) if basin_counts else 0,
        "basin_count_std": float(np.std(basin_counts)) if basin_counts else 0,
        "basin_dist_mean": float(np.mean(basin_distances)) if basin_distances else 0,
        "basin_dist_normalized": float(np.mean(basin_distances) / knn_stats["diameter"]) if basin_distances and knn_stats["diameter"] > 0 else 0,
        "crossover_mean": float(np.mean(crossover_counts)) if crossover_counts else 0,
        "crossover_multi_route_frac": float(np.mean([c >= 2 for c in crossover_counts])) if crossover_counts else 0,
        "beta1_mean": float(np.mean([b["beta_1"] for b in betti_numbers])) if betti_numbers else 0,
        "significant_h1_mean": float(np.mean([b["significant_h1"] for b in betti_numbers])) if betti_numbers else 0,
        "h1_positive_frac": float(np.mean([b["beta_1"] > 0 for b in betti_numbers])) if betti_numbers else 0,
    }

    return result


def analyze_all_clips(
    data: np.ndarray,
    config: AnalysisConfig,
) -> dict:
    results = defaultdict(list)

    print("Applying Takens embedding to all clips...")
    all_pts = []
    for clip_id in range(len(data)):
        pts = takens_embedding(data[clip_id], m=config.takens_m, tau=config.takens_tau)
        all_pts.append(pts)

    all_pts_flat = np.vstack(all_pts)
    clip_lengths = [len(p) for p in all_pts]
    print(f"Total points: {len(all_pts_flat)}, dim: {all_pts_flat.shape[1]}")

    for pca_dim in config.pca_dims_list:
        dim_label = f"pca_{pca_dim}" if pca_dim > 0 else "raw"
        print(f"\n=== Analyzing {dim_label} ===")

        if pca_dim > 0 and pca_dim < all_pts_flat.shape[1]:
            print(f"  Fitting PCA({pca_dim}) on all data...")
            pca = PCA(n_components=pca_dim)
            all_pts_reduced = pca.fit_transform(all_pts_flat)
            var_explained = pca.explained_variance_ratio_.sum()
            print(f"  Variance explained: {var_explained:.3f}")

            pts_per_clip = []
            start = 0
            for length in clip_lengths:
                pts_per_clip.append(all_pts_reduced[start:start + length])
                start += length
        else:
            pts_per_clip = all_pts
            var_explained = 1.0

        clip_results = []
        for clip_id in tqdm(range(len(data)), desc=f"Clips ({dim_label})"):
            pts_work = pts_per_clip[clip_id]

            clip_result = analyze_clip_at_pca_dim_fast(pts_work, config, var_explained)
            clip_result["clip_id"] = clip_id
            clip_result["pca_dim"] = pca_dim if pca_dim > 0 else all_pts[0].shape[1]
            clip_result["actual_dim"] = pts_work.shape[1]
            clip_results.append(clip_result)

        results[dim_label] = clip_results

        means = {
            k: np.mean([r[k] for r in clip_results])
            for k in clip_results[0].keys()
            if isinstance(clip_results[0][k], (int, float))
        }
        print(f"  Mean k-NN (normalized): {means['knn_normalized']:.4f}")
        print(f"  Mean basin count: {means['basin_count_mean']:.2f}")
        print(f"  Mean crossover routes: {means['crossover_mean']:.2f}")
        print(f"  Multi-route fraction: {means['crossover_multi_route_frac']:.3f}")
        print(f"  Mean β₁: {means['beta1_mean']:.2f}")
        print(f"  H₁ positive fraction: {means['h1_positive_frac']:.3f}")

    return dict(results)


def aggregate_results(results: dict) -> dict:
    summary = {}

    for dim_label, clip_results in results.items():
        if not clip_results:
            continue

        numeric_keys = [
            k for k in clip_results[0].keys()
            if isinstance(clip_results[0][k], (int, float)) and k != "clip_id"
        ]

        summary[dim_label] = {
            "n_clips": len(clip_results),
            "actual_dim": clip_results[0]["actual_dim"],
        }

        for k in numeric_keys:
            values = [r[k] for r in clip_results]
            summary[dim_label][f"{k}_mean"] = float(np.mean(values))
            summary[dim_label][f"{k}_std"] = float(np.std(values))

    return summary


def create_per_dimension_boxplots(results: dict, config: AnalysisConfig) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    dim_labels = []
    knn_data = []
    crossover_data = []
    beta1_data = []

    for pca_dim in config.pca_dims_list:
        dim_label = f"pca_{pca_dim}" if pca_dim > 0 else "raw"
        if dim_label not in results:
            continue

        clip_results = results[dim_label]
        actual_dim = clip_results[0]["actual_dim"] if clip_results else pca_dim
        dim_labels.append(f"{actual_dim}D")

        knn_data.append([r["knn_normalized"] for r in clip_results])
        crossover_data.append([r["crossover_mean"] for r in clip_results])
        beta1_data.append([r["beta1_mean"] for r in clip_results])

    axes[0, 0].boxplot(knn_data, labels=dim_labels)
    axes[0, 0].set_xlabel("Dimensionality", fontsize=11)
    axes[0, 0].set_ylabel("Normalized k-NN Distance", fontsize=11)
    axes[0, 0].set_title("k-NN Distance Distribution", fontsize=12, fontweight="bold")
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].boxplot(crossover_data, labels=dim_labels)
    axes[0, 1].set_xlabel("Dimensionality", fontsize=11)
    axes[0, 1].set_ylabel("Crossover Routes", fontsize=11)
    axes[0, 1].set_title("Crossover Route Distribution", fontsize=12, fontweight="bold")
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[1, 0].boxplot(beta1_data, labels=dim_labels)
    axes[1, 0].set_xlabel("Dimensionality", fontsize=11)
    axes[1, 0].set_ylabel("Local β₁", fontsize=11)
    axes[1, 0].set_title("Local Betti Number Distribution", fontsize=12, fontweight="bold")
    axes[1, 0].tick_params(axis="x", rotation=45)

    axes[1, 1].axis("off")

    summary = aggregate_results(results)
    text_lines = ["Summary Statistics:\n"]
    for dim_label in list(summary.keys())[:4]:
        s = summary[dim_label]
        text_lines.append(f"{dim_label}:")
        text_lines.append(f"  k-NN norm: {s['knn_normalized_mean']:.4f} ± {s['knn_normalized_std']:.4f}")
        text_lines.append(f"  Crossover: {s['crossover_mean_mean']:.2f} ± {s['crossover_mean_std']:.2f}")
        text_lines.append(f"  β₁: {s['beta1_mean_mean']:.2f} ± {s['beta1_mean_std']:.2f}")
        text_lines.append(f"  H₁ positive: {s['h1_positive_frac_mean']:.1%}")
        text_lines.append("")

    axes[1, 1].text(0.1, 0.9, "\n".join(text_lines), transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment="top", fontfamily="monospace")

    fig.suptitle(
        f"DM Dim Reduction Effect: Per-Clip Distributions ({config.dm_key})",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    return fig


def run_analysis(config: AnalysisConfig) -> dict:
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("DM Dimensionality Reduction Effect Analysis")
    print("=" * 60)
    print(f"Feature space: {config.dm_key}")
    print(f"PCA dimensions to test: {config.pca_dims_list}")
    print(f"k-neighbors: {config.k_neighbors}")
    print(f"Sample points per clip: {config.n_sample_points}")
    print()

    config_path = os.path.join(config.output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump({
            k: v for k, v in config.__dict__.items()
            if not k.startswith("_")
        }, f)
    print(f"Saved config to {config_path}")

    print("\n=== Loading data ===")
    data, clip_indices = load_data(config)
    print(f"Loaded {len(data)} clips, shape: {data.shape}")

    print("\n=== Running analysis ===")
    results = analyze_all_clips(data, config)

    results_path = os.path.join(config.output_dir, "results.pkl")
    with open(results_path, "wb") as f:
        pkl.dump(results, f)
    print(f"\nSaved results to {results_path}")

    summary = aggregate_results(results)
    summary_path = os.path.join(config.output_dir, "summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    print(f"Saved summary to {summary_path}")

    print("\n=== Creating visualizations ===")

    fig_boxplots = create_per_dimension_boxplots(results, config)
    boxplot_path = os.path.join(config.output_dir, "distributions.png")
    fig_boxplots.savefig(boxplot_path, dpi=config.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig_boxplots)
    print(f"Saved: {boxplot_path}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {config.output_dir}")

    return results


def main(config_path: str | None = None):
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = AnalysisConfig()

    run_analysis(config)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
