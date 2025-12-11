import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import numpy as np
import pickle as pkl
import yaml
import h5py
from dataclasses import dataclass, field
from tqdm import tqdm

import matplotlib.pyplot as plt

from vnl_tda.utils.load_utils import load_from_h5py
from vnl_tda.pipelines.tda.skeletonization.plotting import (
    plot_cluster_subspace_mds,
    _compare_clusters_axis_motion_with_union,
)


QPOS_BODY_NAMES = [
    "world", "floor", "walker", "torso", "vertebra_1", "vertebra_2", "vertebra_3",
    "vertebra_4", "vertebra_5", "vertebra_6", "pelvis", "upper_leg_L", "lower_leg_L",
    "foot_L", "toe_L", "upper_leg_R", "lower_leg_R", "foot_R", "toe_R",
    "vertebra_C1", "vertebra_C2", "vertebra_C3", "vertebra_C4", "vertebra_C5",
    "vertebra_C6", "vertebra_C7", "vertebra_C8", "vertebra_C9", "vertebra_C10",
    "vertebra_C11", "vertebra_C12", "vertebra_C13", "vertebra_C14", "vertebra_C15",
    "vertebra_C16", "vertebra_C17", "vertebra_C18", "vertebra_C19", "vertebra_C20",
    "vertebra_C21", "vertebra_C22", "vertebra_C23", "vertebra_C24", "vertebra_C25",
    "vertebra_C26", "vertebra_C27", "vertebra_C28", "vertebra_C29", "vertebra_C30",
    "vertebra_cervical_5", "vertebra_cervical_4", "vertebra_cervical_3",
    "vertebra_cervical_2", "vertebra_cervical_1", "vertebra_axis", "vertebra_atlant",
    "skull", "jaw", "scapula_L", "upper_arm_L", "lower_arm_L", "hand_L", "finger_L",
    "scapula_R", "upper_arm_R", "lower_arm_R", "hand_R", "finger_R",
]



@dataclass
class ClusterConfig:
    results_dir: str = "out_subspace_dunn"
    n_clusters: int = 13
    cluster_labels: dict[int, str] = field(default_factory=dict)


@dataclass
class AnalysisConfig:
    rodent_rollout_dir: str = "/home/jovyan/vast/kaiwen/TopoVNL/data/rodent_ar"
    behavior_csv: str = "/home/jovyan/vast/kaiwen/TopoVNL/behavior_book.csv"
    output_dir: str = "analysis_output"
    cluster: ClusterConfig = None

    mds_k_cycles: int = 50
    mds_s_top: int = 10
    mds_label_unions: tuple[str, ...] = ("walk", "rear", "groom", "turn", "wait")

    axis_motion_max_loops: int = 200

    pose_variance_body_parts: list[str] = None

    def __post_init__(self):
        if self.cluster is None:
            self.cluster = ClusterConfig()


def load_config(config_path: str) -> AnalysisConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cluster_dict = cfg.get("cluster", {})
    if "cluster_labels" in cluster_dict:
        cluster_dict["cluster_labels"] = {
            int(k): v for k, v in cluster_dict["cluster_labels"].items()
        }
    cluster_cfg = ClusterConfig(**cluster_dict)

    mds_unions = cfg.get("mds_label_unions", ("walk", "rear", "groom", "turn", "wait"))
    if isinstance(mds_unions, list):
        mds_unions = tuple(mds_unions)

    return AnalysisConfig(
        rodent_rollout_dir=cfg.get("rodent_rollout_dir", AnalysisConfig.rodent_rollout_dir),
        behavior_csv=cfg.get("behavior_csv", AnalysisConfig.behavior_csv),
        output_dir=cfg.get("output_dir", AnalysisConfig.output_dir),
        cluster=cluster_cfg,
        mds_k_cycles=cfg.get("mds_k_cycles", 50),
        mds_s_top=cfg.get("mds_s_top", 10),
        mds_label_unions=mds_unions,
        axis_motion_max_loops=cfg.get("axis_motion_max_loops", 200),
        pose_variance_body_parts=cfg.get("pose_variance_body_parts", None),
    )


def load_cluster_dict(results_dir: str) -> dict:
    cluster_dict_path = os.path.join(results_dir, "clustering", "cluster_dict.pkl")
    if not os.path.exists(cluster_dict_path):
        cluster_dict_path = os.path.join(results_dir, "cluster_dict.pkl")
    with open(cluster_dict_path, "rb") as f:
        return pkl.load(f)


def load_qpos(clip_id: int, rodent_rollout_dir: str) -> np.ndarray:
    with h5py.File(f"{rodent_rollout_dir}/clip_{int(clip_id)}.h5", "r") as f:
        roll = load_from_h5py(f)
    return np.asarray(roll["qpos"])


def plot_pose_variance_heatmap(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    rodent_rollout_dir: str,
    output_dir: str,
    body_parts: list[str] = None,
    max_loops: int = 200,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """
    Plot heatmap of joint angle variance per cluster.

    Rows = clusters, Columns = body parts.
    Color intensity = variance of that joint in that cluster.
    """
    os.makedirs(output_dir, exist_ok=True)

    if body_parts is None:
        body_parts = [
            "torso", "pelvis",
            "upper_leg_L", "lower_leg_L", "foot_L",
            "upper_leg_R", "lower_leg_R", "foot_R",
            "skull", "jaw",
            "upper_arm_L", "lower_arm_L",
            "upper_arm_R", "lower_arm_R",
        ]

    body_indices = []
    valid_body_parts = []
    for bp in body_parts:
        if bp in QPOS_BODY_NAMES:
            body_indices.append(QPOS_BODY_NAMES.index(bp))
            valid_body_parts.append(bp)

    cluster_ids = sorted(cluster_dict.keys())
    cluster_names_raw = [cluster_labels.get(cid, f"cluster_{cid}") for cid in cluster_ids]

    sorted_pairs = sorted(zip(cluster_ids, cluster_names_raw), key=lambda x: x[1])
    cluster_ids = [p[0] for p in sorted_pairs]

    n_clusters = len(cluster_ids)
    n_bodies = len(valid_body_parts)

    variance_matrix = np.zeros((n_clusters, n_bodies))

    for row_idx, cid in enumerate(tqdm(cluster_ids, desc="Computing variance")):
        entries = cluster_dict[cid]

        all_qpos = []
        for i, (clip_id, frames) in enumerate(entries):
            if i >= max_loops:
                break

            qpos = load_qpos(clip_id, rodent_rollout_dir)
            frames = np.asarray(frames, dtype=int)
            frames = frames[frames < len(qpos)]

            if len(frames) < 2:
                continue

            for body_idx in body_indices:
                if body_idx < qpos.shape[1]:
                    all_qpos.append(qpos[frames, body_idx])

        if not all_qpos:
            continue

        all_qpos_by_body = {bp: [] for bp in valid_body_parts}
        for i, (clip_id, frames) in enumerate(entries):
            if i >= max_loops:
                break

            qpos = load_qpos(clip_id, rodent_rollout_dir)
            frames = np.asarray(frames, dtype=int)
            frames = frames[frames < len(qpos)]

            if len(frames) < 2:
                continue

            for col_idx, (bp, body_idx) in enumerate(zip(valid_body_parts, body_indices)):
                if body_idx < qpos.shape[1]:
                    all_qpos_by_body[bp].append(qpos[frames, body_idx])

        for col_idx, bp in enumerate(valid_body_parts):
            if all_qpos_by_body[bp]:
                combined = np.concatenate(all_qpos_by_body[bp])
                variance_matrix[row_idx, col_idx] = np.var(combined)

    cluster_names = [cluster_labels.get(cid, f"cluster_{cid}").replace("_", " ") for cid in cluster_ids]
    body_labels = [bp.replace("_", " ") for bp in valid_body_parts]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(variance_matrix, aspect="auto", cmap="plasma")

    ax.set_xticks(np.arange(n_bodies))
    ax.set_xticklabels(body_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(n_clusters))
    ax.set_yticklabels(cluster_names, fontsize=10)

    ax.set_xlabel("Body Part", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)
    ax.set_title("Pose Variance by Cluster", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Variance", fontsize=11)

    plt.tight_layout()

    save_path = os.path.join(output_dir, "pose_variance_heatmap.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved pose variance heatmap to {save_path}")


def run_mds_analysis(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    cfg: AnalysisConfig,
) -> None:
    print("\n=== Running MDS Subspace Analysis ===")
    target_clusters = list(cluster_labels.values())

    save_path = os.path.join(cfg.output_dir, "cluster_subspace_mds.pdf")

    plot_cluster_subspace_mds(
        cluster_cycles_by_id=cluster_dict,
        cluster_labels=cluster_labels,
        target_clusters=target_clusters,
        rodent_rollout_dir=cfg.rodent_rollout_dir,
        csv_path=cfg.behavior_csv,
        label_unions=cfg.mds_label_unions,
        k_cycles=cfg.mds_k_cycles,
        s_top=cfg.mds_s_top,
        pad=0,
        state_key="qpos",
        sample_strategy="shortest",
        seed=0,
        plot=True,
        figsize=(8.5, 6.5),
        save_path=save_path,
    )
    print(f"Saved MDS plot to {save_path}")


def run_axis_motion_analysis(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    cfg: AnalysisConfig,
) -> None:
    print("\n=== Running Axis Motion Analysis ===")
    target_clusters = list(cluster_labels.values())

    out_pdf = os.path.join(cfg.output_dir, "cluster_axis_motion.pdf")

    _compare_clusters_axis_motion_with_union(
        cluster_dict=cluster_dict,
        cluster_labels=cluster_labels,
        target_cluster_names=target_clusters,
        rodent_rollout_dir=cfg.rodent_rollout_dir,
        max_loops=cfg.axis_motion_max_loops,
        out_pdf=out_pdf,
        figsize=(12, 7),
    )
    print(f"Saved axis motion plot to {out_pdf}")


def run_pose_variance_analysis(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    cfg: AnalysisConfig,
) -> None:
    print("\n=== Running Pose Variance Analysis ===")

    plot_pose_variance_heatmap(
        cluster_dict=cluster_dict,
        cluster_labels=cluster_labels,
        rodent_rollout_dir=cfg.rodent_rollout_dir,
        output_dir=cfg.output_dir,
        body_parts=cfg.pose_variance_body_parts,
        max_loops=200,
        figsize=(12, 8),
    )


def main(config_path: str = "configs/analysis.yaml"):
    cfg = load_config(config_path)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"=== Loading clusters from {cfg.cluster.results_dir} ===")
    cluster_dict = load_cluster_dict(cfg.cluster.results_dir)

    if not cfg.cluster.cluster_labels:
        cfg.cluster.cluster_labels = {i: f"cluster_{i}" for i in cluster_dict.keys()}

    print(f"Loaded {len(cluster_dict)} clusters:")
    for cid, entries in sorted(cluster_dict.items()):
        label = cfg.cluster.cluster_labels.get(cid, f"cluster_{cid}")
        print(f"  Cluster {cid} ({label}): {len(entries)} samples")

    run_mds_analysis(cluster_dict, cfg.cluster.cluster_labels, cfg)
    run_axis_motion_analysis(cluster_dict, cfg.cluster.cluster_labels, cfg)
    run_pose_variance_analysis(cluster_dict, cfg.cluster.cluster_labels, cfg)

    print("\n=== Analysis complete! ===")
    print(f"Results saved to {cfg.output_dir}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/analysis.yaml"
    main(config_path)
