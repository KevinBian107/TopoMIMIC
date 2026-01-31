"""Stationary clip analysis for head/arm joint variance within DM cycles."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import GranularityConfig
from .dm_multi_persistence import DMExtractionResult
from .utils import compute_root_displacement, resolve_joint_indices


@dataclass
class StationaryClipInfo:
    clip_idx: int
    root_displacement: float
    n_frames: int


@dataclass
class JointVarianceResult:
    pers_delta: float
    joint_name: str
    variances: list[float]
    mean_variance: float
    std_variance: float
    n_samples: int


@dataclass
class CycleSurvivalResult:
    """Results for cycle survival analysis across persistence levels."""
    transition: tuple[float, float]  # (pers_low, pers_high)
    joint_name: str
    lost_variances: list[float]      # Variances of cycles lost at this transition
    survived_variances: list[float]  # Variances of cycles that survived


@dataclass
class CycleLifetimeResult:
    """Results for cycle lifetime analysis."""
    joint_name: str
    max_persistence: list[float]     # Max persistence where each cycle survived
    variances: list[float]           # Corresponding joint variances


@dataclass
class StationaryAnalysisOutput:
    stationary_clips: list[StationaryClipInfo]
    head_joint_results: dict[float, list[JointVarianceResult]]
    arm_joint_results: dict[float, list[JointVarianceResult]]
    survival_results: list[CycleSurvivalResult] | None = None
    lifetime_results: list[CycleLifetimeResult] | None = None


def identify_stationary_clips(
    qpos_data: np.ndarray,
    config: GranularityConfig,
    clip_indices: list[int],
) -> list[StationaryClipInfo]:
    """
    Identify clips with low root displacement (stationary behavior).

    Args:
        qpos_data: (n_clips, n_frames, D) qpos array
        config: Configuration object
        clip_indices: List of clip indices

    Returns:
        List of StationaryClipInfo for clips below displacement threshold
    """
    stationary_clips = []
    threshold = config.stationary.root_displacement_threshold

    for local_idx in range(len(qpos_data)):
        qpos_clip = qpos_data[local_idx]
        displacement = compute_root_displacement(qpos_clip, config.joints.root_indices)

        if displacement < threshold:
            stationary_clips.append(StationaryClipInfo(
                clip_idx=local_idx,
                root_displacement=displacement,
                n_frames=qpos_clip.shape[0],
            ))

    print(f"  Found {len(stationary_clips)} stationary clips "
          f"(threshold={threshold}, total={len(qpos_data)})")

    return stationary_clips


def extract_cycle_frame_indices(
    dm_result: DMExtractionResult,
    clip_idx: int,
    takens_offset: int,
    only_max: bool = False,
) -> list[np.ndarray]:
    """
    Get frame indices from DM cycles for a specific clip.

    Args:
        dm_result: DM extraction result
        clip_idx: Local clip index
        takens_offset: Offset due to Takens embedding (m-1)*tau
        only_max: If True, only return the longest cycle

    Returns:
        List of frame index arrays (one per cycle, or just the longest if only_max)
    """
    if clip_idx >= len(dm_result.all_cycle_dm):
        return []

    cycles = dm_result.all_cycle_dm[clip_idx]
    if not cycles:
        return []

    if only_max:
        cycles = [max(cycles, key=len)]

    frame_indices_list = []

    for cycle in cycles:
        if len(cycle) < 2:
            continue
        cycle_arr = np.asarray(cycle, dtype=int)
        frame_indices = cycle_arr + takens_offset
        frame_indices_list.append(frame_indices)

    return frame_indices_list


def compute_joint_variance_in_cycles(
    qpos_clip: np.ndarray,
    frame_indices_list: list[np.ndarray],
    joint_idx: int,
    skip_root_dims: int,
) -> list[float]:
    """
    Compute variance of a joint within DM cycles.

    Args:
        qpos_clip: (T, D) qpos for one clip
        frame_indices_list: List of frame index arrays from cycles
        joint_idx: Index of joint (after skip_root_dims)
        skip_root_dims: Number of root dims to skip

    Returns:
        List of variance values (one per cycle)
    """
    variances = []
    full_joint_idx = skip_root_dims + joint_idx
    n_frames = qpos_clip.shape[0]

    for frame_indices in frame_indices_list:
        valid_frames = frame_indices[(frame_indices >= 0) & (frame_indices < n_frames)]
        if len(valid_frames) < 3:
            continue

        joint_values = qpos_clip[valid_frames, full_joint_idx]
        var = float(np.var(joint_values))
        variances.append(var)

    return variances


def analyze_joints_in_cycles(
    stationary_clips: list[StationaryClipInfo],
    dm_result: DMExtractionResult,
    qpos_data: np.ndarray,
    joint_indices: list[int],
    joint_names: list[str],
    config: GranularityConfig,
) -> list[JointVarianceResult]:
    """
    Analyze joint variance within DM cycles for stationary clips.

    Args:
        stationary_clips: List of stationary clip info
        dm_result: DM extraction result for a persistence value
        qpos_data: (n_clips, n_frames, D) qpos array
        joint_indices: List of joint indices to analyze
        joint_names: Names of joints
        config: Configuration object

    Returns:
        List of JointVarianceResult (one per joint)
    """
    takens_offset = (config.takens.m - 1) * config.takens.tau
    results = []

    for joint_idx, joint_name in zip(joint_indices, joint_names):
        all_variances = []

        for clip_info in stationary_clips:
            local_idx = clip_info.clip_idx
            qpos_clip = qpos_data[local_idx]

            frame_indices_list = extract_cycle_frame_indices(
                dm_result, local_idx, takens_offset,
                only_max=config.clustering.only_max,
            )

            variances = compute_joint_variance_in_cycles(
                qpos_clip, frame_indices_list, joint_idx, config.joints.skip_root_dims
            )
            all_variances.extend(variances)

        if all_variances:
            results.append(JointVarianceResult(
                pers_delta=dm_result.pers_delta,
                joint_name=joint_name,
                variances=all_variances,
                mean_variance=float(np.mean(all_variances)),
                std_variance=float(np.std(all_variances)),
                n_samples=len(all_variances),
            ))
        else:
            results.append(JointVarianceResult(
                pers_delta=dm_result.pers_delta,
                joint_name=joint_name,
                variances=[],
                mean_variance=0.0,
                std_variance=0.0,
                n_samples=0,
            ))

    return results


def analyze_cycle_survival(
    stationary_clips: list[StationaryClipInfo],
    dm_results: dict[float, DMExtractionResult],
    qpos_data: np.ndarray,
    joint_indices: list[int],
    joint_names: list[str],
    config: GranularityConfig,
) -> list[CycleSurvivalResult]:
    """
    Analyze which cycles survive vs get lost at each persistence transition.

    Args:
        stationary_clips: List of stationary clip info
        dm_results: Dict mapping pers_delta to DMExtractionResult
        qpos_data: (n_clips, n_frames, D) qpos array
        joint_indices: List of joint indices to analyze
        joint_names: Names of joints
        config: Configuration object

    Returns:
        List of CycleSurvivalResult for each transition and joint
    """
    pers_values = sorted(dm_results.keys())
    if len(pers_values) < 2:
        return []

    takens_offset = (config.takens.m - 1) * config.takens.tau
    results = []

    clips_with_cycles = {}
    for pers_delta in pers_values:
        dm_result = dm_results[pers_delta]
        clips_with_cycles[pers_delta] = set()

        for clip_info in stationary_clips:
            local_idx = clip_info.clip_idx
            if local_idx < len(dm_result.all_cycle_dm):
                cycles = dm_result.all_cycle_dm[local_idx]
                if cycles and len(cycles) > 0:
                    clips_with_cycles[pers_delta].add(local_idx)

    for i in range(len(pers_values) - 1):
        pers_low = pers_values[i]
        pers_high = pers_values[i + 1]

        clips_at_low = clips_with_cycles[pers_low]
        clips_at_high = clips_with_cycles[pers_high]

        lost_clips = clips_at_low - clips_at_high
        survived_clips = clips_at_low & clips_at_high

        dm_result_low = dm_results[pers_low]

        for joint_idx, joint_name in zip(joint_indices, joint_names):
            lost_variances = []
            survived_variances = []

            for local_idx in lost_clips:
                qpos_clip = qpos_data[local_idx]
                frame_indices_list = extract_cycle_frame_indices(
                    dm_result_low, local_idx, takens_offset,
                    only_max=config.clustering.only_max,
                )
                variances = compute_joint_variance_in_cycles(
                    qpos_clip, frame_indices_list, joint_idx, config.joints.skip_root_dims
                )
                lost_variances.extend(variances)

            for local_idx in survived_clips:
                qpos_clip = qpos_data[local_idx]
                frame_indices_list = extract_cycle_frame_indices(
                    dm_result_low, local_idx, takens_offset,
                    only_max=config.clustering.only_max,
                )
                variances = compute_joint_variance_in_cycles(
                    qpos_clip, frame_indices_list, joint_idx, config.joints.skip_root_dims
                )
                survived_variances.extend(variances)

            results.append(CycleSurvivalResult(
                transition=(pers_low, pers_high),
                joint_name=joint_name,
                lost_variances=lost_variances,
                survived_variances=survived_variances,
            ))

    return results


def analyze_cycle_lifetime(
    stationary_clips: list[StationaryClipInfo],
    dm_results: dict[float, DMExtractionResult],
    qpos_data: np.ndarray,
    joint_indices: list[int],
    joint_names: list[str],
    config: GranularityConfig,
) -> list[CycleLifetimeResult]:
    """
    Analyze the maximum persistence level where each cycle survives.

    Args:
        stationary_clips: List of stationary clip info
        dm_results: Dict mapping pers_delta to DMExtractionResult
        qpos_data: (n_clips, n_frames, D) qpos array
        joint_indices: List of joint indices to analyze
        joint_names: Names of joints
        config: Configuration object

    Returns:
        List of CycleLifetimeResult for each joint
    """
    pers_values = sorted(dm_results.keys())
    if not pers_values:
        return []

    takens_offset = (config.takens.m - 1) * config.takens.tau

    clip_max_pers = {}
    for clip_info in stationary_clips:
        local_idx = clip_info.clip_idx
        max_pers = None

        for pers_delta in pers_values:
            dm_result = dm_results[pers_delta]
            if local_idx < len(dm_result.all_cycle_dm):
                cycles = dm_result.all_cycle_dm[local_idx]
                if cycles and len(cycles) > 0:
                    max_pers = pers_delta

        if max_pers is not None:
            clip_max_pers[local_idx] = max_pers

    results = []
    min_pers = pers_values[0]
    dm_result_min = dm_results[min_pers]

    for joint_idx, joint_name in zip(joint_indices, joint_names):
        max_persistence_list = []
        variances_list = []

        for local_idx, max_pers in clip_max_pers.items():
            qpos_clip = qpos_data[local_idx]
            frame_indices_list = extract_cycle_frame_indices(
                dm_result_min, local_idx, takens_offset,
                only_max=config.clustering.only_max,
            )
            variances = compute_joint_variance_in_cycles(
                qpos_clip, frame_indices_list, joint_idx, config.joints.skip_root_dims
            )

            for var in variances:
                max_persistence_list.append(max_pers)
                variances_list.append(var)

        results.append(CycleLifetimeResult(
            joint_name=joint_name,
            max_persistence=max_persistence_list,
            variances=variances_list,
        ))

    return results


def run_stationary_analysis(
    qpos_data: np.ndarray,
    dm_results: dict[float, DMExtractionResult],
    config: GranularityConfig,
    clip_indices: list[int],
) -> StationaryAnalysisOutput:
    """
    Run full stationary clip analysis across persistence values.

    Args:
        qpos_data: (n_clips, n_frames, D) qpos array
        dm_results: Dict mapping pers_delta to DMExtractionResult
        config: Configuration object
        clip_indices: List of clip indices

    Returns:
        StationaryAnalysisOutput with all results
    """
    print("\n=== Stationary Analysis ===")

    stationary_clips = identify_stationary_clips(qpos_data, config, clip_indices)

    head_indices, head_names = resolve_joint_indices(
        config.joints.head_joints, config.joints.skip_root_dims
    )
    arm_indices, arm_names = resolve_joint_indices(
        config.joints.arm_joints, config.joints.skip_root_dims
    )

    print(f"  Head joints: {head_names}")
    print(f"  Arm joints: {arm_names}")

    head_joint_results = {}
    arm_joint_results = {}

    for pers_delta, dm_result in tqdm(dm_results.items(), desc="Analyzing joints"):
        head_results = analyze_joints_in_cycles(
            stationary_clips, dm_result, qpos_data,
            head_indices, head_names, config
        )
        head_joint_results[pers_delta] = head_results

        arm_results = analyze_joints_in_cycles(
            stationary_clips, dm_result, qpos_data,
            arm_indices, arm_names, config
        )
        arm_joint_results[pers_delta] = arm_results

    output_dir = Path(config.output_dir) / "stationary"
    output_dir.mkdir(parents=True, exist_ok=True)

    clips_df = pd.DataFrame([
        {
            "clip_idx": c.clip_idx,
            "abs_clip_idx": clip_indices[c.clip_idx],
            "root_displacement": c.root_displacement,
            "n_frames": c.n_frames,
        }
        for c in stationary_clips
    ])
    clips_df.to_csv(output_dir / "stationary_clips.csv", index=False)
    print(f"  Saved stationary clips to {output_dir / 'stationary_clips.csv'}")

    all_joint_indices = head_indices + arm_indices
    all_joint_names = head_names + arm_names

    print("  Analyzing cycle survival across persistence levels...")
    survival_results = analyze_cycle_survival(
        stationary_clips, dm_results, qpos_data,
        all_joint_indices, all_joint_names, config
    )

    print("  Analyzing cycle lifetime...")
    lifetime_results = analyze_cycle_lifetime(
        stationary_clips, dm_results, qpos_data,
        all_joint_indices, all_joint_names, config
    )

    return StationaryAnalysisOutput(
        stationary_clips=stationary_clips,
        head_joint_results=head_joint_results,
        arm_joint_results=arm_joint_results,
        survival_results=survival_results,
        lifetime_results=lifetime_results,
    )
