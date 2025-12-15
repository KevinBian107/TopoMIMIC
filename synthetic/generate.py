"""
Synthetic behavioral state space generator.

Models a behavioral manifold with:
- Main clusters representing distinct behaviors (walk, rear, groom)
- Sub-clusters within each main behavior (variations)
- Transition regions connecting behaviors

Each synthetic clip is a trajectory that:
1. Dwells in sub-clusters for variable durations
2. Transitions between sub-clusters (within or across behaviors)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal
import os
import pickle as pkl


@dataclass
class ClusterSpec:
    """Specification for a single cluster."""
    name: str
    behavior: str
    center: np.ndarray
    cov_scale: float = 0.3
    weight: float = 1.0


@dataclass
class TransitionSpec:
    """Specification for transition between clusters."""
    from_cluster: str
    to_cluster: str
    probability: float = 0.1


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    n_clips: int = 500
    frames_per_clip: int = 500
    latent_dim: int = 16

    behaviors: list[str] = field(default_factory=lambda: ["walk", "rear", "groom"])
    subclusters_per_behavior: int = 3

    behavior_separation: float = 2.0
    subcluster_separation: float = 0.8
    cluster_std: float = 0.6

    dwell_time_mean: int = 50
    dwell_time_std: int = 20
    dwell_time_min: int = 10

    transition_smoothness: int = 10

    intra_behavior_transition_prob: float = 0.7
    inter_behavior_transition_prob: float = 0.3

    noise_scale: float = 0.3
    global_noise_scale: float = 1.5

    random_seed: int = 42
    output_dir: str = "data/synthetic"


@dataclass
class SyntheticClip:
    """A single synthetic clip with ground truth labels."""
    latents: np.ndarray
    cluster_labels: np.ndarray
    behavior_labels: np.ndarray
    subcluster_labels: np.ndarray
    clip_id: int


@dataclass
class SyntheticDataset:
    """Complete synthetic dataset."""
    clips: list[SyntheticClip]
    cluster_specs: list[ClusterSpec]
    config: SyntheticConfig

    @property
    def all_latents(self) -> np.ndarray:
        return np.vstack([c.latents for c in self.clips])

    @property
    def all_cluster_labels(self) -> np.ndarray:
        return np.concatenate([c.cluster_labels for c in self.clips])

    @property
    def all_behavior_labels(self) -> np.ndarray:
        return np.concatenate([c.behavior_labels for c in self.clips])


def generate_cluster_centers(
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> list[ClusterSpec]:
    """
    Generate cluster centers in latent space.

    Structure is spread across ALL dimensions to avoid trivial PCA separation.
    Uses random orthogonal directions for behavior separation.
    """
    clusters = []
    n_behaviors = len(config.behaviors)
    dim = config.latent_dim

    behavior_directions = rng.standard_normal((n_behaviors, dim))
    behavior_directions = behavior_directions / np.linalg.norm(behavior_directions, axis=1, keepdims=True)

    for b_idx, behavior in enumerate(config.behaviors):
        behavior_center = behavior_directions[b_idx] * config.behavior_separation

        subcluster_directions = rng.standard_normal((config.subclusters_per_behavior, dim))
        subcluster_directions = subcluster_directions / np.linalg.norm(subcluster_directions, axis=1, keepdims=True)

        for sub_idx in range(config.subclusters_per_behavior):
            sub_offset = subcluster_directions[sub_idx] * config.subcluster_separation
            center = behavior_center + sub_offset

            clusters.append(ClusterSpec(
                name=f"{behavior}_{sub_idx}",
                behavior=behavior,
                center=center,
                cov_scale=config.cluster_std,
                weight=1.0 / config.subclusters_per_behavior,
            ))

    return clusters


def sample_point_in_cluster(
    cluster: ClusterSpec,
    rng: np.random.Generator,
    noise_scale: float = 0.1,
) -> np.ndarray:
    """Sample a point from a cluster distribution."""
    dim = len(cluster.center)
    noise = rng.normal(0, cluster.cov_scale, dim)
    extra_noise = rng.normal(0, noise_scale, dim)
    return cluster.center + noise + extra_noise


def smooth_transition(
    start: np.ndarray,
    end: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
    noise_scale: float = 0.1,
) -> np.ndarray:
    """Generate smooth transition between two points."""
    t = np.linspace(0, 1, n_steps).reshape(-1, 1)
    t_smooth = 3 * t**2 - 2 * t**3

    path = start + t_smooth * (end - start)
    noise = rng.normal(0, noise_scale, path.shape)

    return path + noise


def generate_clip_trajectory(
    clusters: list[ClusterSpec],
    config: SyntheticConfig,
    rng: np.random.Generator,
    clip_id: int,
) -> SyntheticClip:
    """
    Generate a single clip trajectory through the state space.

    The trajectory:
    1. Starts in a random cluster
    2. Dwells for a random duration
    3. Transitions to a sub-cluster in a DIFFERENT behavior (cross-behavior transitions)
    4. Repeats until clip is filled
    """
    latents = []
    cluster_labels = []
    behavior_labels = []
    subcluster_labels = []

    cluster_name_to_idx = {c.name: i for i, c in enumerate(clusters)}
    behavior_to_clusters = {}
    for i, c in enumerate(clusters):
        if c.behavior not in behavior_to_clusters:
            behavior_to_clusters[c.behavior] = []
        behavior_to_clusters[c.behavior].append(i)

    current_cluster_idx = rng.integers(0, len(clusters))
    current_cluster = clusters[current_cluster_idx]
    current_point = sample_point_in_cluster(current_cluster, rng, config.noise_scale)

    frames_generated = 0

    while frames_generated < config.frames_per_clip:
        dwell_time = max(
            config.dwell_time_min,
            int(rng.normal(config.dwell_time_mean, config.dwell_time_std))
        )
        dwell_time = min(dwell_time, config.frames_per_clip - frames_generated)

        for _ in range(dwell_time):
            point = sample_point_in_cluster(current_cluster, rng, config.noise_scale)
            point = 0.3 * current_point + 0.7 * point
            current_point = point

            latents.append(point)
            cluster_labels.append(current_cluster_idx)
            behavior_labels.append(config.behaviors.index(current_cluster.behavior))
            subcluster_labels.append(int(current_cluster.name.split('_')[1]))

            frames_generated += 1
            if frames_generated >= config.frames_per_clip:
                break

        if frames_generated >= config.frames_per_clip:
            break

        # Always transition to a DIFFERENT behavior (cross macro-cluster)
        other_behaviors = [b for b in config.behaviors if b != current_cluster.behavior]
        if other_behaviors:
            next_behavior = rng.choice(other_behaviors)
            next_cluster_idx = rng.choice(behavior_to_clusters[next_behavior])
        else:
            next_cluster_idx = current_cluster_idx

        next_cluster = clusters[next_cluster_idx]
        next_point = sample_point_in_cluster(next_cluster, rng, config.noise_scale)

        transition_frames = min(config.transition_smoothness, config.frames_per_clip - frames_generated)
        if transition_frames > 1:
            transition_path = smooth_transition(
                current_point, next_point, transition_frames, rng, config.noise_scale
            )

            for t_idx, point in enumerate(transition_path):
                progress = t_idx / len(transition_path)
                if progress < 0.5:
                    label_cluster = current_cluster_idx
                    label_behavior = config.behaviors.index(current_cluster.behavior)
                    label_sub = int(current_cluster.name.split('_')[1])
                else:
                    label_cluster = next_cluster_idx
                    label_behavior = config.behaviors.index(next_cluster.behavior)
                    label_sub = int(next_cluster.name.split('_')[1])

                latents.append(point)
                cluster_labels.append(label_cluster)
                behavior_labels.append(label_behavior)
                subcluster_labels.append(label_sub)

                frames_generated += 1
                if frames_generated >= config.frames_per_clip:
                    break

        current_cluster_idx = next_cluster_idx
        current_cluster = next_cluster
        current_point = next_point

    return SyntheticClip(
        latents=np.array(latents),
        cluster_labels=np.array(cluster_labels),
        behavior_labels=np.array(behavior_labels),
        subcluster_labels=np.array(subcluster_labels),
        clip_id=clip_id,
    )


def generate_synthetic_dataset(config: SyntheticConfig) -> SyntheticDataset:
    """Generate complete synthetic dataset."""
    rng = np.random.default_rng(config.random_seed)

    clusters = generate_cluster_centers(config, rng)

    clips = []
    for clip_id in range(config.n_clips):
        clip = generate_clip_trajectory(clusters, config, rng, clip_id)
        clips.append(clip)

    if config.global_noise_scale > 0:
        all_latents = np.vstack([c.latents for c in clips])
        global_noise = rng.standard_normal(all_latents.shape) * config.global_noise_scale

        idx = 0
        for clip in clips:
            n_frames = len(clip.latents)
            clip.latents = clip.latents + global_noise[idx:idx + n_frames]
            idx += n_frames

    return SyntheticDataset(
        clips=clips,
        cluster_specs=clusters,
        config=config,
    )


def save_synthetic_dataset(dataset: SyntheticDataset, output_dir: str) -> None:
    """Save synthetic dataset to disk."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "dataset.pkl"), "wb") as f:
        pkl.dump(dataset, f)

    np.save(os.path.join(output_dir, "all_latents.npy"), dataset.all_latents)
    np.save(os.path.join(output_dir, "all_cluster_labels.npy"), dataset.all_cluster_labels)
    np.save(os.path.join(output_dir, "all_behavior_labels.npy"), dataset.all_behavior_labels)

    cluster_info = []
    for c in dataset.cluster_specs:
        cluster_info.append({
            "name": c.name,
            "behavior": c.behavior,
            "center": c.center.tolist(),
        })

    import json
    with open(os.path.join(output_dir, "cluster_info.json"), "w") as f:
        json.dump(cluster_info, f, indent=2)

    behavior_to_clips = {b: [] for b in dataset.config.behaviors}
    for clip in dataset.clips:
        dominant_behavior = np.bincount(clip.behavior_labels).argmax()
        behavior_name = dataset.config.behaviors[dominant_behavior]
        behavior_to_clips[behavior_name].append(clip.clip_id)

    with open(os.path.join(output_dir, "behavior_to_clips.json"), "w") as f:
        json.dump(behavior_to_clips, f, indent=2)

    print(f"Saved synthetic dataset to {output_dir}")
    print(f"  Clips: {len(dataset.clips)}")
    print(f"  Frames per clip: {dataset.config.frames_per_clip}")
    print(f"  Total frames: {len(dataset.all_latents)}")
    print(f"  Clusters: {len(dataset.cluster_specs)}")
    print(f"  Behaviors: {dataset.config.behaviors}")


def load_synthetic_dataset(output_dir: str) -> SyntheticDataset:
    """Load synthetic dataset from disk."""
    with open(os.path.join(output_dir, "dataset.pkl"), "rb") as f:
        return pkl.load(f)
