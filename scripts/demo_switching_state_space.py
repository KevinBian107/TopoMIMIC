#!/usr/bin/env python3
"""
NASCAR Racetrack: DM-G vs SLDS Comparison
==========================================
Demonstrates topological analysis on the NASCAR racetrack SLDS example.

The NASCAR system (from Linderman et al. 2017 rSLDS paper) is a classic
switching linear dynamical system with:
- 4 discrete states (track segments: top, right, bottom, left)
- Different linear dynamics in each segment
- Clear transitions between states at segment boundaries

We compare two approaches for discovering structure:
1. SLDS fitting - fit a switching linear model to recover states
2. DM-G - topological graph analysis to find cycles/structure

Usage:
    python scripts/demo_switching_state_space.py
    python scripts/demo_switching_state_space.py --obs_noise 0.3
    python scripts/demo_switching_state_space.py --subsample 500
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from pathlib import Path
from dataclasses import dataclass, field
import yaml

sys.path.insert(0, "/home/jovyan/vast/kaiwen/TopoVNL")
sys.path.append("/home/jovyan/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")

from topo_mimic.dmwk.skeletonization.skeletonization import extract_skeleton_pcd_dm


@dataclass
class NASCARConfig:
    n_timesteps: int = 2000
    num_states: int = 4
    noise_std: float = 0.1
    obs_noise_std: float = 0.3
    seed: int = 42


@dataclass
class SLDSFitConfig:
    num_states: int = 4
    latent_dim: int = 2
    full_iters: int = 100


@dataclass
class MorseConfig:
    k_neighbors: int = 10
    pers_delta: float = 0.25
    epsilon: float = 0.99
    subsample: int = 1000


@dataclass
class VisualizationConfig:
    figsize: tuple[int, int] = (16, 8)
    dpi: int = 150
    trajectory_linewidth: float = 1.0
    trajectory_alpha: float = 0.8
    cycle_color: str = "purple"
    marker_size: int = 30
    state_colors: tuple[str, ...] = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


@dataclass
class Config:
    output_dir: str = "out/switching_state_space_demo"
    nascar: NASCARConfig = field(default_factory=NASCARConfig)
    slds_fit: SLDSFitConfig = field(default_factory=SLDSFitConfig)
    dm: MorseConfig = field(default_factory=MorseConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    nascar_dict = cfg_dict.get("nascar", {})
    nascar_cfg = NASCARConfig(
        n_timesteps=nascar_dict.get("n_timesteps", 2000),
        num_states=nascar_dict.get("num_states", 4),
        noise_std=nascar_dict.get("noise_std", 0.1),
        obs_noise_std=nascar_dict.get("obs_noise_std", 0.3),
        seed=nascar_dict.get("seed", 42),
    )

    slds_dict = cfg_dict.get("slds_fit", {})
    slds_cfg = SLDSFitConfig(
        num_states=slds_dict.get("num_states", 4),
        latent_dim=slds_dict.get("latent_dim", 2),
        full_iters=slds_dict.get("full_iters", 100),
    )

    dm_dict = cfg_dict.get("dm", {})
    dm_cfg = MorseConfig(
        k_neighbors=dm_dict.get("k_neighbors", 10),
        pers_delta=dm_dict.get("pers_delta", 0.25),
        epsilon=dm_dict.get("epsilon", 0.99),
        subsample=dm_dict.get("subsample", 1000),
    )

    vis_dict = cfg_dict.get("visualization", {})
    vis_cfg = VisualizationConfig(
        figsize=tuple(vis_dict.get("figsize", [16, 8])),
        dpi=vis_dict.get("dpi", 150),
        trajectory_linewidth=vis_dict.get("trajectory_linewidth", 1.0),
        trajectory_alpha=vis_dict.get("trajectory_alpha", 0.8),
        cycle_color=vis_dict.get("cycle_color", "purple"),
        marker_size=vis_dict.get("marker_size", 30),
        state_colors=tuple(vis_dict.get("state_colors", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])),
    )

    return Config(
        output_dir=cfg_dict.get("output_dir", "out/switching_state_space_demo"),
        nascar=nascar_cfg,
        slds_fit=slds_cfg,
        dm=dm_cfg,
        visualization=vis_cfg,
    )


# =============================================================================
# NASCAR Racetrack System
# =============================================================================

def make_nascar_dynamics():
    """
    Create 3D dynamics matrices for the NASCAR racetrack.

    Creates an elliptical (oval) trajectory like a real racetrack,
    with state-dependent Z oscillation.

    Returns:
        A_matrices: list of 4 (3x3) dynamics matrices
        b_vectors: list of 4 (3,) bias vectors
    """
    # Ellipse parameters: wider in X, narrower in Y
    a = 2.0  # Semi-major axis (X direction)
    b = 1.0  # Semi-minor axis (Y direction)

    rot_speed = 0.06
    theta = rot_speed

    # Rotation with ellipse scaling: scale, rotate, unscale
    # This creates rotation along an ellipse
    S = np.array([[1/a, 0], [0, 1/b]])  # Scale to circle
    S_inv = np.array([[a, 0], [0, b]])  # Scale back to ellipse
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    R2d = S_inv @ R @ S  # Elliptical rotation

    A_matrices = []
    b_vectors = []

    # Z has different behavior in each state
    z_biases = [0.02, 0.01, -0.01, -0.02]

    for i in range(4):
        A = np.zeros((3, 3))
        A[:2, :2] = R2d
        A[2, 2] = 0.9
        A_matrices.append(A)
        b_vectors.append(np.array([0.0, 0.0, z_biases[i]]))

    return A_matrices, b_vectors


def get_state_from_position(x: np.ndarray) -> int:
    """
    Determine which track segment based on position.
    Uses angle from origin to determine state.
    """
    angle = np.arctan2(x[1], x[0])  # Range: [-pi, pi]

    # Divide into 4 quadrants
    if angle >= np.pi/4 and angle < 3*np.pi/4:
        return 0  # Top
    elif angle >= 3*np.pi/4 or angle < -3*np.pi/4:
        return 1  # Left
    elif angle >= -3*np.pi/4 and angle < -np.pi/4:
        return 2  # Bottom
    else:
        return 3  # Right


def generate_nascar_trajectory(
    cfg: NASCARConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 3D NASCAR racetrack trajectory (elliptical).

    Returns:
        latent: (T, 3) true trajectory
        observations: (T, 3) noisy observations
        true_states: (T,) ground truth state assignments (0-3)
    """
    rng = np.random.default_rng(cfg.seed)

    A_matrices, b_vectors = make_nascar_dynamics()

    # Start at right side of elliptical track (a=2.0 is semi-major axis)
    x = np.array([2.0, 0.0, 0.0])

    latent = np.zeros((cfg.n_timesteps, 3))
    true_states = np.zeros(cfg.n_timesteps, dtype=int)

    for t in range(cfg.n_timesteps):
        latent[t] = x

        # Determine state from angle (normalized for ellipse)
        state = get_state_from_position(x[:2])
        true_states[t] = state

        A = A_matrices[state]
        b = b_vectors[state]

        x = A @ x + b + rng.normal(0, cfg.noise_std, 3)

    observations = latent + rng.normal(0, cfg.obs_noise_std, latent.shape)

    return latent, observations, true_states


# =============================================================================
# SLDS Fitting (using dynamax)
# =============================================================================

def fit_slds_to_observations(
    observations: np.ndarray,
    cfg: SLDSFitConfig,
    output_dir: str,
    seed: int = 42,
) -> tuple[np.ndarray | None, dict]:
    """
    Fit HMM to observations using dynamax.
    Uses GaussianHMM to cluster trajectory into discrete states.
    """
    try:
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        from dynamax.hidden_markov_model import GaussianHMM
        jax.config.update("jax_enable_x64", True)
    except ImportError:
        print("  WARNING: dynamax not available, skipping SLDS fitting")
        return None, {}

    print(f"  Fitting HMM (dynamax) to observations...")
    print(f"    Shape: {observations.shape}, num_states={cfg.num_states}")

    T, D = observations.shape
    num_states = cfg.num_states

    # Initialize the model
    key = jr.PRNGKey(seed)
    model = GaussianHMM(
        num_states=num_states,
        emission_dim=D,
    )

    # Initialize parameters
    key, subkey = jr.split(key)
    params, props = model.initialize(subkey, emissions=jnp.array(observations))

    # Fit using EM
    print(f"    Running EM ({cfg.full_iters} iterations)...")
    params, lls = model.fit_em(
        params, props,
        emissions=jnp.array(observations),
        num_iters=cfg.full_iters,
    )

    # Infer most likely state sequence
    inferred_states = model.most_likely_states(params, jnp.array(observations))
    inferred_states = np.array(inferred_states)

    metrics = {
        "n_active_states": len(np.unique(inferred_states)),
        "state_distribution": np.bincount(inferred_states, minlength=num_states).tolist(),
        "final_ll": float(lls[-1]),
    }

    print(f"    Active states: {metrics['n_active_states']}")
    print(f"    Final log-likelihood: {metrics['final_ll']:.2f}")

    return inferred_states, metrics


def compute_state_agreement(true_states: np.ndarray, pred_states: np.ndarray) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    return {
        "ari": adjusted_rand_score(true_states, pred_states),
        "nmi": normalized_mutual_info_score(true_states, pred_states),
    }


# =============================================================================
# DM-G Analysis
# =============================================================================

def run_dm_analysis(
    points: np.ndarray,
    cfg: MorseConfig,
    output_dir: str,
) -> tuple[list[np.ndarray], nx.Graph, np.ndarray]:
    """Run DM-G analysis with subsampling."""
    exp_dir = os.path.abspath(os.path.join(output_dir, "dm_analysis"))
    os.makedirs(exp_dir, exist_ok=True)

    n_points = len(points)
    if cfg.subsample > 0 and n_points > cfg.subsample:
        step = n_points // cfg.subsample
        subsample_idx = np.arange(0, n_points, step)[:cfg.subsample]
        points_sub = points[subsample_idx]
        print(f"  Subsampled: {n_points} -> {len(points_sub)} points (step={step})")
    else:
        subsample_idx = np.arange(n_points)
        points_sub = points

    print(f"  Running DM-G analysis...")
    print(f"    Points: {len(points_sub)}, k={cfg.k_neighbors}, pers_delta={cfg.pers_delta}")

    cycles, G_dm = extract_skeleton_pcd_dm(
        points=points_sub,
        k=cfg.k_neighbors,
        pers_delta=cfg.pers_delta,
        epsilon=cfg.epsilon,
        output_dir=exp_dir,
        cleanup=False,
    )

    print(f"    DM graph: {G_dm.number_of_nodes()} nodes, {G_dm.number_of_edges()} edges")
    print(f"    Cycles found: {len(cycles)}")

    if cycles:
        cycles = sorted(cycles, key=len, reverse=True)
        print(f"    Longest cycle: {len(cycles[0])} vertices")

    return cycles, G_dm, subsample_idx


# =============================================================================
# Visualization
# =============================================================================

def plot_trajectory_colored_2d(
    ax,
    trajectory: np.ndarray,
    states: np.ndarray,
    vis_cfg: VisualizationConfig,
    title: str = "",
):
    """Plot X-Y trajectory colored by discrete state."""
    colors = vis_cfg.state_colors

    for i in range(len(trajectory) - 1):
        ax.plot(
            trajectory[i:i+2, 0],
            trajectory[i:i+2, 1],
            color=colors[states[i] % len(colors)],
            linewidth=vis_cfg.trajectory_linewidth,
            alpha=vis_cfg.trajectory_alpha,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold")

    from matplotlib.patches import Patch
    state_names = ["Top", "Left", "Bottom", "Right"]
    legend_elements = [Patch(facecolor=colors[i], label=state_names[i])
                       for i in range(min(4, len(colors)))]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)


def plot_trajectory_colored_3d(
    ax,
    trajectory: np.ndarray,
    states: np.ndarray,
    vis_cfg: VisualizationConfig,
    title: str = "",
):
    """Plot 3D trajectory colored by discrete state."""
    colors = vis_cfg.state_colors

    for i in range(len(trajectory) - 1):
        ax.plot(
            trajectory[i:i+2, 0],
            trajectory[i:i+2, 1],
            trajectory[i:i+2, 2],
            color=colors[states[i] % len(colors)],
            linewidth=vis_cfg.trajectory_linewidth,
            alpha=vis_cfg.trajectory_alpha,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=11, fontweight="bold")


def plot_dm_longest_loop_2d(
    ax,
    trajectory: np.ndarray,
    subsample_idx: np.ndarray,
    cycles: list[np.ndarray],
    vis_cfg: VisualizationConfig,
    title: str = "",
):
    """Plot the longest DM-G loop (X-Y view) - points only."""
    traj_sub = trajectory[subsample_idx]

    ax.plot(
        traj_sub[:, 0], traj_sub[:, 1],
        color="gray", linewidth=0.5, alpha=0.3,
    )

    if cycles and len(cycles) > 0:
        cycle = cycles[0]
        valid_cycle = [idx for idx in cycle if idx < len(traj_sub)]
        if len(valid_cycle) > 1:
            cycle_points = traj_sub[valid_cycle]
            ax.scatter(
                cycle_points[:, 0], cycle_points[:, 1],
                c=vis_cfg.cycle_color, s=30, marker="o",
                zorder=50, alpha=0.9,
                label=f"DM-G loop ({len(valid_cycle)} pts)",
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold")
    if cycles and len(cycles) > 0:
        ax.legend(loc="upper right", fontsize=8)


def plot_dm_longest_loop_3d(
    ax,
    trajectory: np.ndarray,
    subsample_idx: np.ndarray,
    cycles: list[np.ndarray],
    vis_cfg: VisualizationConfig,
    title: str = "",
):
    """Plot the longest DM-G loop in 3D - points only."""
    traj_sub = trajectory[subsample_idx]

    ax.plot(
        traj_sub[:, 0], traj_sub[:, 1], traj_sub[:, 2],
        color="gray", linewidth=0.5, alpha=0.3,
    )

    if cycles and len(cycles) > 0:
        cycle = cycles[0]
        valid_cycle = [idx for idx in cycle if idx < len(traj_sub)]
        if len(valid_cycle) > 1:
            cycle_points = traj_sub[valid_cycle]
            ax.scatter(
                cycle_points[:, 0], cycle_points[:, 1], cycle_points[:, 2],
                c=vis_cfg.cycle_color, s=30, marker="o",
                zorder=50, alpha=0.9,
                label=f"DM-G loop ({len(valid_cycle)} pts)",
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=11, fontweight="bold")
    if cycles and len(cycles) > 0:
        ax.legend(loc="upper right", fontsize=8)


def plot_summary_figure(
    latent: np.ndarray,
    observations: np.ndarray,
    true_states: np.ndarray,
    slds_states: np.ndarray | None,
    cycles: list[np.ndarray],
    G_dm: nx.Graph,
    subsample_idx: np.ndarray,
    vis_cfg: VisualizationConfig,
    output_path: str,
):
    """Create summary figure comparing ground truth, SLDS, and DM-G."""
    fig = plt.figure(figsize=(18, 10))

    # Row 1: 3D views
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    plot_trajectory_colored_3d(ax1, latent, true_states, vis_cfg, "Ground Truth (3D)")
    ax1.view_init(elev=20, azim=45)

    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    if slds_states is not None:
        plot_trajectory_colored_3d(ax2, observations, slds_states, vis_cfg, "SLDS Inferred (3D)")
    else:
        plot_trajectory_colored_3d(ax2, observations, true_states, vis_cfg, "Observations (3D)")
    ax2.view_init(elev=20, azim=45)

    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    plot_dm_longest_loop_3d(ax3, observations, subsample_idx, cycles, vis_cfg, "DM-G Loop (3D)")
    ax3.view_init(elev=20, azim=45)

    # Row 2: X-Y views (top-down racetrack)
    ax4 = fig.add_subplot(2, 3, 4)
    plot_trajectory_colored_2d(ax4, latent, true_states, vis_cfg, "Ground Truth (X-Y)")

    ax5 = fig.add_subplot(2, 3, 5)
    if slds_states is not None:
        plot_trajectory_colored_2d(ax5, observations, slds_states, vis_cfg, "SLDS Inferred (X-Y)")
    else:
        plot_trajectory_colored_2d(ax5, observations, true_states, vis_cfg, "Observations (X-Y)")

    ax6 = fig.add_subplot(2, 3, 6)
    plot_dm_longest_loop_2d(ax6, observations, subsample_idx, cycles, vis_cfg, "DM-G Loop (X-Y)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=vis_cfg.dpi, bbox_inches="tight")
    print(f"  Saved: {os.path.basename(output_path)}")
    plt.close()


def plot_state_timeline(
    true_states: np.ndarray,
    slds_states: np.ndarray | None,
    vis_cfg: VisualizationConfig,
    output_path: str,
):
    """Plot state timeline comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)

    t = np.arange(len(true_states))
    colors = vis_cfg.state_colors

    # Ground truth
    for i in range(len(t) - 1):
        axes[0].axvspan(t[i], t[i+1], color=colors[true_states[i] % len(colors)], alpha=0.7)
    axes[0].set_ylabel("Ground Truth")
    axes[0].set_yticks([])

    # SLDS
    if slds_states is not None:
        for i in range(len(t) - 1):
            axes[1].axvspan(t[i], t[i+1], color=colors[slds_states[i] % len(colors)], alpha=0.7)
        axes[1].set_ylabel("SLDS Inferred")
    else:
        axes[1].set_ylabel("(No SLDS)")
    axes[1].set_yticks([])
    axes[1].set_xlabel("Time")

    plt.tight_layout()
    plt.savefig(output_path, dpi=vis_cfg.dpi, bbox_inches="tight")
    print(f"  Saved: {os.path.basename(output_path)}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NASCAR Racetrack DM-G vs SLDS Demo")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--n_timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--obs_noise", type=float, default=None)
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--pers_delta", type=float, default=None)
    parser.add_argument("--skip_slds", action="store_true")
    args = parser.parse_args()

    # Load config
    default_config_path = Path(__file__).parent.parent / "configs" / "scripts" / "demo_switching_state_space.yaml"

    if args.config:
        cfg = load_config(args.config)
    elif default_config_path.exists():
        print(f"Loading config from {default_config_path}")
        cfg = load_config(str(default_config_path))
    else:
        cfg = Config()

    # CLI overrides
    if args.n_timesteps is not None:
        cfg.nascar.n_timesteps = args.n_timesteps
    if args.seed is not None:
        cfg.nascar.seed = args.seed
    if args.obs_noise is not None:
        cfg.nascar.obs_noise_std = args.obs_noise
    if args.subsample is not None:
        cfg.dm.subsample = args.subsample
    if args.k is not None:
        cfg.dm.k_neighbors = args.k
    if args.pers_delta is not None:
        cfg.dm.pers_delta = args.pers_delta

    # Output directory
    if os.path.isabs(cfg.output_dir):
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / cfg.output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NASCAR Racetrack: DM-G vs SLDS Comparison")
    print("=" * 70)
    print(f"  Output: {output_dir}")
    print(f"  NASCAR: {cfg.nascar.n_timesteps} timesteps, noise={cfg.nascar.obs_noise_std}")
    print(f"  DM-G: k={cfg.dm.k_neighbors}, pers_delta={cfg.dm.pers_delta}, subsample={cfg.dm.subsample}")

    # Generate NASCAR trajectory
    print("\n[1/4] Generating NASCAR trajectory...")
    latent, observations, true_states = generate_nascar_trajectory(cfg.nascar)
    n_transitions = np.sum(np.diff(true_states) != 0)
    print(f"  Shape: {latent.shape}")
    print(f"  State distribution: {np.bincount(true_states)}")
    print(f"  State transitions: {n_transitions}")

    # Save data
    np.save(output_dir / "latent.npy", latent)
    np.save(output_dir / "observations.npy", observations)
    np.save(output_dir / "true_states.npy", true_states)

    # Fit SLDS (or load existing)
    slds_states = None
    slds_metrics = {}
    slds_states_path = output_dir / "slds_states.npy"

    if not args.skip_slds:
        if slds_states_path.exists():
            print("\n[2/4] Loading existing SLDS states...")
            slds_states = np.load(slds_states_path)
            print(f"  Loaded: {slds_states_path}")
            agreement = compute_state_agreement(true_states, slds_states)
            slds_metrics = {
                "n_active_states": len(np.unique(slds_states)),
                **agreement,
            }
            print(f"  SLDS ARI: {agreement['ari']:.3f}")
            print(f"  SLDS NMI: {agreement['nmi']:.3f}")
        else:
            print("\n[2/4] Fitting SLDS to observations...")
            slds_states, slds_metrics = fit_slds_to_observations(
                observations, cfg.slds_fit, str(output_dir), cfg.nascar.seed
            )
            if slds_states is not None:
                agreement = compute_state_agreement(true_states, slds_states)
                slds_metrics.update(agreement)
                print(f"  SLDS ARI: {agreement['ari']:.3f}")
                print(f"  SLDS NMI: {agreement['nmi']:.3f}")
                np.save(slds_states_path, slds_states)
    else:
        print("\n[2/4] Skipping SLDS fitting (--skip_slds)")

    # Run DM-G
    print("\n[3/4] Running DM-G analysis...")
    cycles, G_dm, subsample_idx = run_dm_analysis(observations, cfg.dm, str(output_dir))

    # Visualize
    print("\n[4/4] Creating visualizations...")

    plot_summary_figure(
        latent, observations, true_states, slds_states,
        cycles, G_dm, subsample_idx,
        cfg.visualization,
        str(output_dir / "summary.png"),
    )

    plot_state_timeline(
        true_states, slds_states,
        cfg.visualization,
        str(output_dir / "state_timeline.png"),
    )

    # Save metrics
    metrics = {
        "nascar": {
            "n_timesteps": cfg.nascar.n_timesteps,
            "obs_noise_std": cfg.nascar.obs_noise_std,
            "n_transitions": int(n_transitions),
        },
        "dm": {
            "subsample": cfg.dm.subsample,
            "n_cycles": len(cycles),
            "longest_cycle": len(cycles[0]) if cycles else 0,
            "graph_nodes": G_dm.number_of_nodes(),
            "graph_edges": G_dm.number_of_edges(),
        },
    }
    if slds_metrics:
        metrics["slds"] = slds_metrics

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"  NASCAR state transitions: {n_transitions}")
    if slds_states is not None:
        print(f"  SLDS: ARI={slds_metrics['ari']:.3f}, NMI={slds_metrics['nmi']:.3f}")
    print(f"  DM-G: {len(cycles)} cycles, {G_dm.number_of_nodes()} nodes, {G_dm.number_of_edges()} edges")
    print(f"\nOutput: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
