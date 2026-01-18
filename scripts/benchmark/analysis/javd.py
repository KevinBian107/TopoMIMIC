"""Joint Angle Variance Difference (JAVD) analysis."""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from .behavior_parsing import parse_cluster_label
from .ground_truth import find_gt_transition_frames, TransitionSegment
from .data_loading import load_qpos, QPOS_BODY_NAMES, get_body_part_indices


def compute_variance_ratio_similarity(
    gt_variance: dict[str, float],
    disc_variance: dict[str, float],
    joints: list[str],
) -> float:
    """
    Compute cosine similarity between normalized variance profiles.

    Normalizes each profile to sum to 1, then computes cosine similarity.
    This measures whether the relative proportions (ratios) between joints
    are preserved, regardless of absolute magnitude differences.

    Example:
        GT:   [0.05, 0.03, 0.02] → normalized: [0.50, 0.30, 0.20]
        Disc: [0.01, 0.006, 0.004] → normalized: [0.50, 0.30, 0.20]
        Cosine similarity = 1.0 (perfect ratio preservation)

    Args:
        gt_variance: Dict mapping joint name to GT variance
        disc_variance: Dict mapping joint name to discovered variance
        joints: List of joint names to compare

    Returns:
        Cosine similarity (0 to 1), or NaN if insufficient data
    """
    if len(joints) < 2:
        return np.nan

    gt_values = np.array([gt_variance.get(j, 0.0) for j in joints])
    disc_values = np.array([disc_variance.get(j, 0.0) for j in joints])

    gt_sum = gt_values.sum()
    disc_sum = disc_values.sum()

    if gt_sum == 0 or disc_sum == 0:
        return np.nan

    gt_norm = gt_values / gt_sum
    disc_norm = disc_values / disc_sum

    dot_product = np.dot(gt_norm, disc_norm)
    gt_mag = np.linalg.norm(gt_norm)
    disc_mag = np.linalg.norm(disc_norm)

    if gt_mag == 0 or disc_mag == 0:
        return np.nan

    return dot_product / (gt_mag * disc_mag)


def compute_frames_variance(
    frames_list: list[tuple[int, np.ndarray]],
    rodent_rollout_dir: str,
    body_parts: list[str],
    max_loops: int = 200,
) -> dict[str, float]:
    """
    Compute variance for each body part from discovered cluster frames.

    Args:
        frames_list: List of (clip_id, frame_indices) tuples
        rodent_rollout_dir: Path to rollout data
        body_parts: List of body part names to compute variance for
        max_loops: Maximum number of entries to process

    Returns:
        Dict mapping body part name to variance value
    """
    body_indices, valid_body_parts = get_body_part_indices(body_parts)

    all_qpos_by_body = {bp: [] for bp in valid_body_parts}

    for i, (clip_id, frames) in enumerate(frames_list):
        if i >= max_loops:
            break

        try:
            qpos = load_qpos(clip_id, rodent_rollout_dir)
        except Exception:
            continue

        frames = np.asarray(frames, dtype=int)
        frames = frames[frames < len(qpos)]

        if len(frames) < 1:
            continue

        for bp, body_idx in zip(valid_body_parts, body_indices):
            if body_idx < qpos.shape[1]:
                all_qpos_by_body[bp].append(qpos[frames, body_idx])

    variance_dict = {}
    for bp in valid_body_parts:
        if all_qpos_by_body[bp]:
            combined = np.concatenate(all_qpos_by_body[bp])
            variance_dict[bp] = float(np.var(combined))
        else:
            variance_dict[bp] = 0.0

    return variance_dict


def compute_gt_frames_variance(
    gt_transitions: list[TransitionSegment],
    rodent_rollout_dir: str,
    body_parts: list[str],
) -> dict[str, float]:
    """
    Compute variance for each body part from GT transition frames.

    Args:
        gt_transitions: List of TransitionSegment objects
        rodent_rollout_dir: Path to rollout data
        body_parts: List of body part names to compute variance for

    Returns:
        Dict mapping body part name to variance value
    """
    body_indices, valid_body_parts = get_body_part_indices(body_parts)

    all_qpos_by_body = {bp: [] for bp in valid_body_parts}
    qpos_cache = {}

    for t in gt_transitions:
        clip_id = t.clip_id
        if clip_id not in qpos_cache:
            try:
                qpos_cache[clip_id] = load_qpos(clip_id, rodent_rollout_dir)
            except Exception:
                continue

        qpos = qpos_cache[clip_id]

        for frame in t.all_frames:
            if frame >= len(qpos):
                continue

            for bp, body_idx in zip(valid_body_parts, body_indices):
                if body_idx < qpos.shape[1]:
                    all_qpos_by_body[bp].append(qpos[frame, body_idx])

    variance_dict = {}
    for bp in valid_body_parts:
        if all_qpos_by_body[bp]:
            values = np.array(all_qpos_by_body[bp])
            variance_dict[bp] = float(np.var(values))
        else:
            variance_dict[bp] = 0.0

    return variance_dict


def compute_javd_table(
    cluster_dict: dict,
    cluster_labels: dict[int, str],
    behavior_labels_csv: str,
    rodent_rollout_dir: str,
    output_dir: str,
    body_parts: list[str],
    top_k: int = 5,
    max_loops: int = 200,
    method_name: str = "",
    seed: int = 0,
    short_threshold: int = 10,
    max_gap: int = 10,
    merge_rear: bool = True,
    render_gt: bool = False,
    gt_render_max_samples: int = 5,
    gt_render_framerate: int = 30,
    gt_render_target_size: tuple[int, int] = (320, 240),
) -> pd.DataFrame | None:
    """
    Compute JAVD (Joint Angle Variance Difference) table for transition clusters.

    Compares variance in joint angles between discovered transition clusters
    and ground truth transition frames. For each cluster, identifies the top K
    highest-variance joints from GT and computes the variance difference.

    Args:
        cluster_dict: Dict mapping cluster_id to list of (clip_id, frames)
        cluster_labels: Dict mapping cluster_id to label string
        behavior_labels_csv: Path to behavior labels CSV file
        rodent_rollout_dir: Path to rollout data
        output_dir: Directory to save output files
        body_parts: List of body part names to analyze
        top_k: Number of top variance joints to consider per cluster
        max_loops: Maximum entries to process per cluster
        method_name: Name of the method for output filenames
        seed: Seed value for output filenames
        short_threshold: Max frames for intermediate behavior bridging
        max_gap: Max gap between consecutive segments
        merge_rear: Whether to merge Supported/Unsupported Rear
        render_gt: Whether to render GT transition videos
        gt_render_max_samples: Max samples per transition for rendering
        gt_render_framerate: Video framerate for GT rendering
        gt_render_target_size: Target tile size for GT rendering

    Returns:
        DataFrame with JAVD table, or None if no transition clusters found
    """
    os.makedirs(output_dir, exist_ok=True)

    transition_clusters = []
    for cid, label in cluster_labels.items():
        parsed = parse_cluster_label(label, merge_rear=merge_rear)
        if parsed is None:
            continue
        if parsed.comparison_type != "transition":
            continue
        if len(parsed.primitives) < 2:
            continue

        transition_clusters.append({
            "cid": cid,
            "label": label,
            "parsed": parsed,
            "primitives": parsed.primitives,
        })

    if not transition_clusters:
        print("    No transition clusters found, skipping JAVD analysis")
        return None

    print(f"    Found {len(transition_clusters)} transition clusters for JAVD analysis")

    javd_data = {}
    all_top_joints = set()
    all_gt_transitions_for_render = []

    for tc in tqdm(transition_clusters, desc="    Computing JAVD"):
        cid = tc["cid"]
        primitives = tc["primitives"]
        gt_label = f"{primitives[0]} <-> {primitives[1]}"

        entries = cluster_dict.get(cid, [])
        if not entries:
            continue

        gt_transitions = find_gt_transition_frames(
            behavior_labels_csv,
            primitives,
            short_threshold=short_threshold,
            max_gap=max_gap,
            merge_rear=merge_rear,
        )
        if not gt_transitions:
            continue

        all_gt_transitions_for_render.extend(gt_transitions)

        n_discovered_frames = sum(len(frames) for _, frames in entries[:max_loops])
        n_gt_frames = sum(len(t.all_frames) for t in gt_transitions)

        discovered_variance = compute_frames_variance(
            entries, rodent_rollout_dir, body_parts, max_loops
        )
        gt_variance = compute_gt_frames_variance(
            gt_transitions, rodent_rollout_dir, body_parts
        )

        sorted_gt_joints = sorted(
            gt_variance.items(), key=lambda x: x[1], reverse=True
        )
        top_joints = [joint for joint, _ in sorted_gt_joints[:top_k] if gt_variance[joint] > 0]

        javd_for_cluster = {}
        gt_var_for_cluster = {}
        disc_var_for_cluster = {}
        for joint in top_joints:
            var_disc = discovered_variance.get(joint, 0.0)
            var_gt = gt_variance.get(joint, 0.0)
            javd_for_cluster[joint] = abs(var_disc - var_gt)
            gt_var_for_cluster[joint] = var_gt
            disc_var_for_cluster[joint] = var_disc
            all_top_joints.add(joint)

        ratio_sim = compute_variance_ratio_similarity(
            gt_variance, discovered_variance, top_joints
        )

        javd_data[gt_label] = {
            "javd": javd_for_cluster,
            "gt_var": gt_var_for_cluster,
            "disc_var": disc_var_for_cluster,
            "ratio_similarity": ratio_sim,
            "top_joints": top_joints,
            "cid": cid,
            "n_discovered_frames": n_discovered_frames,
            "n_gt_frames": n_gt_frames,
        }

    if not javd_data:
        print("    No valid JAVD data computed")
        return None

    all_joints_list = sorted(all_top_joints)
    cluster_labels_list = sorted(javd_data.keys())

    def build_variance_table(data_key: str) -> pd.DataFrame:
        table_data = []
        for joint in all_joints_list:
            row = {"Joint": joint}
            for cluster_name in cluster_labels_list:
                cluster_info = javd_data[cluster_name]
                if joint in cluster_info[data_key]:
                    row[cluster_name] = cluster_info[data_key][joint]
                else:
                    row[cluster_name] = np.nan
            table_data.append(row)

        df = pd.DataFrame(table_data)
        df = df.set_index("Joint")

        row_means = df.mean(axis=1, skipna=True)
        df["Average"] = row_means

        col_means = df.mean(axis=0, skipna=True)
        avg_row = pd.DataFrame([col_means], index=["Average"])
        df = pd.concat([df, avg_row])
        return df

    df_javd = build_variance_table("javd")
    df_gt = build_variance_table("gt_var")
    df_disc = build_variance_table("disc_var")

    ratio_sim_data = []
    for cluster_name in cluster_labels_list:
        ratio_sim_data.append({
            "Transition": cluster_name,
            "Ratio Similarity": javd_data[cluster_name]["ratio_similarity"],
        })
    df_ratio_sim = pd.DataFrame(ratio_sim_data)

    prefix = f"{method_name}_seed{seed}" if method_name else f"seed{seed}"

    javd_output_dir = os.path.join(output_dir, "javd")
    os.makedirs(javd_output_dir, exist_ok=True)

    javd_path = os.path.join(javd_output_dir, f"{prefix}_javd_table.csv")
    df_javd.to_csv(javd_path, float_format="%.6f")
    print(f"    Saved JAVD (absolute difference) table to {javd_path}")

    gt_path = os.path.join(javd_output_dir, f"{prefix}_gt_variance_table.csv")
    df_gt.to_csv(gt_path, float_format="%.6f")
    print(f"    Saved GT variance table to {gt_path}")

    disc_path = os.path.join(javd_output_dir, f"{prefix}_discovered_variance_table.csv")
    df_disc.to_csv(disc_path, float_format="%.6f")
    print(f"    Saved Discovered variance table to {disc_path}")

    ratio_sim_path = os.path.join(javd_output_dir, f"{prefix}_ratio_similarity.csv")
    df_ratio_sim.to_csv(ratio_sim_path, index=False, float_format="%.4f")
    print(f"    Saved variance ratio similarity to {ratio_sim_path}")

    sample_counts = []
    for cluster_name in cluster_labels_list:
        info = javd_data[cluster_name]
        sample_counts.append({
            "Transition": cluster_name,
            "GT Frames": info["n_gt_frames"],
            "Discovered Frames": info["n_discovered_frames"],
            "Cluster ID": info["cid"],
        })
    df_counts = pd.DataFrame(sample_counts)

    counts_path = os.path.join(javd_output_dir, f"{prefix}_sample_counts.csv")
    df_counts.to_csv(counts_path, index=False)
    print(f"    Saved sample counts to {counts_path}")

    print(f"\n    === Sample Counts ({method_name} seed {seed}) ===")
    print(df_counts.to_string(index=False))
    print()

    low_sample_transitions = df_counts[
        (df_counts["GT Frames"] < 50) | (df_counts["Discovered Frames"] < 50)
    ]["Transition"].tolist()
    if low_sample_transitions:
        print(f"    WARNING: Low sample count (<50 frames) for: {low_sample_transitions}")
        print()

    print(f"\n    === JAVD Table (|Discovered - GT|) ({method_name} seed {seed}) ===")
    print(df_javd.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else ""))

    print(f"\n    === GT Variance Table ({method_name} seed {seed}) ===")
    print(df_gt.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else ""))

    print(f"\n    === Discovered Variance Table ({method_name} seed {seed}) ===")
    print(df_disc.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else ""))

    print(f"\n    === Variance Ratio Similarity ({method_name} seed {seed}) ===")
    print("    Cosine similarity of normalized variance profiles (1.0 = identical ratios)")
    print("    Comparing: Discovered variance proportions vs GT variance proportions")
    print("    (Normalized by sum, so [0.05, 0.03, 0.02] and [0.01, 0.006, 0.004] both become [0.5, 0.3, 0.2])")
    print(df_ratio_sim.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else ""))
    print()

    overall_javd = df_javd.loc["Average", "Average"]
    overall_gt = df_gt.loc["Average", "Average"]
    overall_disc = df_disc.loc["Average", "Average"]
    overall_ratio_sim = df_ratio_sim["Ratio Similarity"].mean()
    print(f"    Overall Mean JAVD (absolute): {overall_javd:.6f}")
    print(f"    Overall Mean GT Variance: {overall_gt:.6f}")
    print(f"    Overall Mean Discovered Variance: {overall_disc:.6f}")
    print(f"    Overall Mean Ratio Similarity: {overall_ratio_sim:.4f} (1.0 = perfect ratio preservation)")

    if render_gt and all_gt_transitions_for_render:
        from .visualization import render_gt_transition_grid

        print(f"\n    Rendering GT transition videos...")
        render_gt_transition_grid(
            gt_transitions=all_gt_transitions_for_render,
            rodent_rollout_dir=rodent_rollout_dir,
            output_dir=output_dir,
            max_samples=gt_render_max_samples,
            framerate=gt_render_framerate,
            target_size=gt_render_target_size,
            method_name=method_name,
            seed=seed,
        )

    return df_javd
