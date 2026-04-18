"""
Unified method implementations for behavioral clustering and segmentation.

Combines clustering methods (KMeans variants, DBSCAN, Spectral) and
state-space methods (AR-HMM, SLDS) into a single interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

try:
    import ot
    HAS_OT = True
except ImportError:
    HAS_OT = False

try:
    import keypoint_moseq as kpms
    import jax
    jax.config.update("jax_enable_x64", True)
    HAS_KPMS = True
except ImportError:
    HAS_KPMS = False

try:
    import torch
    from topo_mimic.benchmark.hyperparam_sweep.custom_vae.model import RNNVAE
    from topo_mimic.benchmark.hyperparam_sweep.custom_vae.trainer import VAETrainer
    from topo_mimic.benchmark.hyperparam_sweep.custom_vae.config import VAEHyperparams
    from topo_mimic.benchmark.hyperparam_sweep.custom_vae.segmentation import segment_latent
    HAS_CUSTOM_VAE = True
except ImportError:
    HAS_CUSTOM_VAE = False


class MethodType(Enum):
    CLUSTERING = "clustering"
    STATE_SPACE = "state_space"


@dataclass
class MethodConfig:
    name: str
    method_type: MethodType
    enabled: bool = True
    params: dict = field(default_factory=dict)


@dataclass
class ClusteringResult:
    labels: np.ndarray
    n_clusters: int
    method_name: str
    seed: int
    extra: dict = field(default_factory=dict)


@dataclass
class KPMSResult:
    labels: list[np.ndarray]
    n_states: int
    method_name: str
    model: Any = None
    results: dict = None
    data: dict = None
    metadata: tuple = None
    original: np.ndarray = None
    reconstructed: np.ndarray = None
    reconstruction_mse: float = None
    transition_cluster_dict: dict = None
    transition_stats: dict = None


@dataclass
class SustainedSegment:
    syllable_id: int
    start_frame: int
    end_frame: int
    duration: int


@dataclass
class TransitionInstance:
    clip_id: int
    from_syllable: int
    to_syllable: int
    boundary_frame: int
    frame_indices: np.ndarray


def find_sustained_segments(
    syllables: np.ndarray,
    min_duration: int = 50,
) -> list[SustainedSegment]:
    """
    Find contiguous runs of the same syllable lasting >= min_duration frames.

    Args:
        syllables: 1D array of syllable IDs for a single clip
        min_duration: Minimum frames for a segment to be considered sustained

    Returns:
        List of SustainedSegment objects
    """
    if len(syllables) == 0:
        return []

    segments = []
    current_syll = syllables[0]
    start_frame = 0

    for i in range(1, len(syllables)):
        if syllables[i] != current_syll:
            duration = i - start_frame
            if duration >= min_duration:
                segments.append(SustainedSegment(
                    syllable_id=int(current_syll),
                    start_frame=start_frame,
                    end_frame=i,
                    duration=duration,
                ))
            current_syll = syllables[i]
            start_frame = i

    duration = len(syllables) - start_frame
    if duration >= min_duration:
        segments.append(SustainedSegment(
            syllable_id=int(current_syll),
            start_frame=start_frame,
            end_frame=len(syllables),
            duration=duration,
        ))

    return segments


def extract_sustained_transitions(
    labels_list: list[np.ndarray],
    min_duration: int = 50,
    top_k: int = 10,
    min_instances: int = 3,
) -> tuple[dict[int, list[tuple[int, np.ndarray]]], dict]:
    """
    Extract transition clusters from KPMS syllable sequences.

    Args:
        labels_list: List of syllable arrays, one per clip
        min_duration: Minimum frames for a syllable to be sustained
        top_k: Number of top transition types to keep as clusters
        min_instances: Minimum instances required for a transition type

    Returns:
        cluster_dict: Dict mapping cluster_id to list of (clip_id, frame_indices)
        stats: Statistics dictionary
    """
    all_transitions: list[TransitionInstance] = []
    total_sustained_segments = 0
    segments_per_syllable: dict[int, int] = {}
    durations: list[int] = []
    total_frames = 0
    sustained_frames = 0

    for clip_id, syllables in enumerate(labels_list):
        syllables = np.asarray(syllables)
        total_frames += len(syllables)

        segments = find_sustained_segments(syllables, min_duration)
        total_sustained_segments += len(segments)

        for seg in segments:
            segments_per_syllable[seg.syllable_id] = segments_per_syllable.get(seg.syllable_id, 0) + 1
            durations.append(seg.duration)
            sustained_frames += seg.duration

        for i in range(len(segments) - 1):
            seg_a = segments[i]
            seg_b = segments[i + 1]

            if seg_a.syllable_id == seg_b.syllable_id:
                continue

            frame_indices = np.arange(seg_a.start_frame, seg_b.end_frame)

            all_transitions.append(TransitionInstance(
                clip_id=clip_id,
                from_syllable=seg_a.syllable_id,
                to_syllable=seg_b.syllable_id,
                boundary_frame=seg_a.end_frame,
                frame_indices=frame_indices,
            ))

    transition_groups: dict[tuple[int, int], list[TransitionInstance]] = {}
    for t in all_transitions:
        key = (t.from_syllable, t.to_syllable)
        if key not in transition_groups:
            transition_groups[key] = []
        transition_groups[key].append(t)

    sorted_transitions = sorted(
        transition_groups.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )

    filtered_transitions = [
        (key, instances) for key, instances in sorted_transitions
        if len(instances) >= min_instances
    ]

    top_transitions = filtered_transitions[:top_k]

    cluster_dict: dict[int, list[tuple[int, np.ndarray]]] = {}
    transition_labels: dict[int, str] = {}

    for cluster_id, (key, instances) in enumerate(top_transitions):
        from_syll, to_syll = key
        cluster_dict[cluster_id] = [
            (t.clip_id, t.frame_indices) for t in instances
        ]
        transition_labels[cluster_id] = f"S{from_syll}->S{to_syll}"

    stats = {
        "n_clips": len(labels_list),
        "n_sustained_segments": total_sustained_segments,
        "segments_per_syllable": segments_per_syllable,
        "duration_mean": float(np.mean(durations)) if durations else 0.0,
        "duration_std": float(np.std(durations)) if durations else 0.0,
        "duration_min": int(np.min(durations)) if durations else 0,
        "duration_max": int(np.max(durations)) if durations else 0,
        "coverage": sustained_frames / total_frames if total_frames > 0 else 0.0,
        "n_transition_types": len(transition_groups),
        "n_transition_instances": len(all_transitions),
        "top_k_transitions": [
            {
                "rank": i + 1,
                "from_syllable": key[0],
                "to_syllable": key[1],
                "count": len(instances),
                "n_clips": len(set(t.clip_id for t in instances)),
            }
            for i, (key, instances) in enumerate(top_transitions)
        ],
        "transition_labels": transition_labels,
        "min_duration": min_duration,
        "top_k": top_k,
        "min_instances": min_instances,
    }

    return cluster_dict, stats


def extract_each_syllable_as_cluster(
    labels_list: list[np.ndarray],
    min_total_frames: int = 0,
) -> tuple[dict[int, list[tuple[int, np.ndarray]]], dict]:
    """
    Treat each unique syllable (or VAE cluster id) as its own cluster.

    Contrast with ``extract_sustained_transitions`` which groups (from, to)
    syllable pairs into transition clusters. This function is a flat partition:
    for each unique label in ``labels_list``, collect (clip_id, frame_indices)
    where that label is active.

    Args:
        labels_list: list of 1D arrays, one per clip, holding frame-level
            syllable or cluster ids.
        min_total_frames: drop syllables whose total frame count across all
            clips is below this threshold. Default 0 (keep all).

    Returns:
        cluster_dict: {cluster_id -> [(clip_id, frame_indices), ...]} where
            cluster_id is a 0-indexed label ordered by descending total frames
            across all clips.
        stats: dict with n_clusters, n_clips, total_frames, syllable_id_map
            (cluster_id -> original syllable id), syllable_totals_original.
    """
    per_syllable_members: dict[int, list[tuple[int, np.ndarray]]] = {}
    syllable_totals: dict[int, int] = {}
    total_frames = 0

    for clip_id, syllables in enumerate(labels_list):
        syllables = np.asarray(syllables)
        total_frames += int(syllables.size)
        if syllables.size == 0:
            continue
        for syll_id in np.unique(syllables):
            syll_int = int(syll_id)
            frame_indices = np.where(syllables == syll_id)[0].astype(int)
            if frame_indices.size == 0:
                continue
            if syll_int not in per_syllable_members:
                per_syllable_members[syll_int] = []
            per_syllable_members[syll_int].append((clip_id, frame_indices))
            syllable_totals[syll_int] = syllable_totals.get(syll_int, 0) + int(frame_indices.size)

    if min_total_frames > 0:
        per_syllable_members = {
            s: m for s, m in per_syllable_members.items()
            if syllable_totals.get(s, 0) >= min_total_frames
        }

    sorted_syllables = sorted(per_syllable_members.keys(), key=lambda s: -syllable_totals[s])
    cluster_dict: dict[int, list[tuple[int, np.ndarray]]] = {}
    syllable_id_map: dict[int, int] = {}
    for cluster_id, syll_id in enumerate(sorted_syllables):
        cluster_dict[cluster_id] = per_syllable_members[syll_id]
        syllable_id_map[cluster_id] = syll_id

    stats = {
        "n_clips": len(labels_list),
        "n_clusters": len(cluster_dict),
        "total_frames": total_frames,
        "min_total_frames": min_total_frames,
        "syllable_id_map": syllable_id_map,
        "syllable_totals_original": syllable_totals,
    }
    return cluster_dict, stats


def print_transition_stats(stats: dict) -> None:
    """Print formatted cluster / transition statistics.

    Supports both the new syllable-as-cluster stats schema and the legacy
    sustained-transition schema.
    """
    print(f"\n{'='*60}")
    if "syllable_id_map" in stats:
        print("Syllable-as-Cluster Analysis")
        print(f"{'='*60}")
        print(f"n_clips: {stats['n_clips']}")
        print(f"n_clusters (= unique syllables kept): {stats['n_clusters']}")
        print(f"total_frames across all clips: {stats['total_frames']}")
        print(f"min_total_frames threshold: {stats['min_total_frames']}")
        totals = stats.get("syllable_totals_original", {})
        syll_map = stats.get("syllable_id_map", {})
        if syll_map:
            print(f"\nTop 15 clusters by total frames:")
            print(f"  {'Cluster':<10} {'Orig Syll':<12} {'Total Frames':<15}")
            print(f"  {'-'*37}")
            for cluster_id in sorted(syll_map.keys())[:15]:
                orig = syll_map[cluster_id]
                total = totals.get(orig, 0)
                print(f"  {cluster_id:<10} {orig:<12} {total:<15}")
        return

    print("Sustained Transition Analysis (legacy)")
    print(f"{'='*60}")
    print(f"Parameters:")
    print(f"  min_sustained_duration: {stats['min_duration']} frames")
    print(f"  n_clips: {stats['n_clips']}")

    print(f"\n{'='*60}")
    print("Sustained Segment Statistics")
    print(f"{'='*60}")
    print(f"Total sustained segments: {stats['n_sustained_segments']}")
    print(f"Duration: mean={stats['duration_mean']:.1f}, std={stats['duration_std']:.1f}, "
          f"min={stats['duration_min']}, max={stats['duration_max']}")
    print(f"Coverage: {stats['coverage']*100:.1f}% of frames in sustained segments")
    print(f"\nSegments per syllable:")
    for syll_id, count in sorted(stats['segments_per_syllable'].items()):
        print(f"  Syllable {syll_id}: {count} segments")

    print(f"\n{'='*60}")
    print("Transition Statistics")
    print(f"{'='*60}")
    print(f"Total unique transition types: {stats['n_transition_types']}")
    print(f"Total transition instances: {stats['n_transition_instances']}")

    print(f"\nTop {len(stats['top_k_transitions'])} Transitions (selected as clusters):")
    print(f"  {'Rank':<6} {'From->To':<12} {'Count':<8} {'Clips':<8}")
    print(f"  {'-'*34}")
    for t in stats['top_k_transitions']:
        label = f"S{t['from_syllable']}->S{t['to_syllable']}"
        print(f"  {t['rank']:<6} {label:<12} {t['count']:<8} {t['n_clips']:<8}")


class BaseMethod(ABC):
    def __init__(self, name: str, params: dict = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def method_type(self) -> MethodType:
        pass


class W2DMCIntKMeans(BaseMethod):
    """
    Wasserstein-2 based clustering on DM cycles (intention space) using MDS + KMeans.
    Requires precomputed W2 distance matrix passed via params.
    """
    def __init__(self, params: dict = None):
        super().__init__("w2_dmc_int_kmeans", params)
        self.mds_components = self.params.get("mds_components", 3)

    @property
    def method_type(self) -> MethodType:
        return MethodType.CLUSTERING

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        distance_matrix = self.params.get("distance_matrix")
        if distance_matrix is None:
            raise ValueError("W2DMKMeans requires 'distance_matrix' in params")

        mds = MDS(
            n_components=self.mds_components,
            dissimilarity='precomputed',
            random_state=seed,
            normalized_stress='auto',
        )
        X_mds = mds.fit_transform(distance_matrix)
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
        return km.fit_predict(X_mds)


class W2UnifKMeans(BaseMethod):
    """
    Baseline: W2 clustering on uniformly sampled frames (not DM cycles).

    Uses one window per clip with frames uniformly downsampled to target_len.
    """
    def __init__(self, params: dict = None):
        super().__init__("w2_unif_kmeans", params)
        self.mds_components = self.params.get("mds_components", 3)

    @property
    def method_type(self) -> MethodType:
        return MethodType.CLUSTERING

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        distance_matrix = self.params.get("distance_matrix")
        if distance_matrix is None:
            raise ValueError("W2UnifKMeans requires 'distance_matrix' in params")

        mds = MDS(
            n_components=self.mds_components,
            dissimilarity='precomputed',
            random_state=seed,
            normalized_stress='auto',
        )
        X_mds = mds.fit_transform(distance_matrix)
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
        return km.fit_predict(X_mds)


class W2FullKMeans(BaseMethod):
    """
    Baseline: W2 clustering on all frames per clip (no downsampling).

    Uses one window per clip with every frame included.
    """
    def __init__(self, params: dict = None):
        super().__init__("w2_full_kmeans", params)
        self.mds_components = self.params.get("mds_components", 3)

    @property
    def method_type(self) -> MethodType:
        return MethodType.CLUSTERING

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        distance_matrix = self.params.get("distance_matrix")
        if distance_matrix is None:
            raise ValueError("W2FullKMeans requires 'distance_matrix' in params")

        mds = MDS(
            n_components=self.mds_components,
            dissimilarity='precomputed',
            random_state=seed,
            normalized_stress='auto',
        )
        X_mds = mds.fit_transform(distance_matrix)
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
        return km.fit_predict(X_mds)


class GWDMCIntKMeans(BaseMethod):
    """
    Gromov-Wasserstein based clustering on DM cycles (intention space) using MDS + KMeans.
    Requires precomputed GW distance matrix passed via params.
    """
    def __init__(self, params: dict = None):
        super().__init__("gw_dmc_int_kmeans", params)
        self.mds_components = self.params.get("mds_components", 3)

    @property
    def method_type(self) -> MethodType:
        return MethodType.CLUSTERING

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        distance_matrix = self.params.get("distance_matrix")
        if distance_matrix is None:
            raise ValueError("GWDMKMeans requires 'distance_matrix' in params")

        mds = MDS(
            n_components=self.mds_components,
            dissimilarity='precomputed',
            random_state=seed,
            normalized_stress='auto',
        )
        X_mds = mds.fit_transform(distance_matrix)
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
        return km.fit_predict(X_mds)


class KPMSIntentionMethod(BaseMethod):
    """
    Keypoint-MoSeq based state-space method for intention data.

    Supports both AR-HMM and SLDS model types via the model_type parameter.
    Default hyperparameters are tuned for intention data (from hyperparam sweep):
    - kappa=1.35e4, latent_dim=10, num_states=15, model_type=slds
    - Achieves intention reconstruction MSE ~0.57
    """
    def __init__(self, params: dict = None):
        super().__init__("kpms_intention", params)
        self.model_type = self.params.get("model_type", "slds")
        self.latent_dim = self.params.get("latent_dim", 10)
        self.kappa = self.params.get("kappa", 1.35e4)
        self.ar_iters = self.params.get("ar_iters", 100)
        self.full_iters = self.params.get("full_iters", 300)

    @property
    def method_type(self) -> MethodType:
        return MethodType.STATE_SPACE

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        raise NotImplementedError("Use fit() for state-space methods")

    def fit(
        self,
        data: np.ndarray,
        n_states: int,
        project_dir: str,
        seed: int = 0,
    ) -> KPMSResult:
        if not HAS_KPMS:
            raise ImportError("keypoint-moseq not installed")

        import shutil
        from pathlib import Path

        project_path = Path(project_dir)
        if project_path.exists():
            shutil.rmtree(project_path)
        project_path.mkdir(parents=True, exist_ok=True)

        n_clips, n_frames, n_dim = data.shape
        coordinates = {}
        confidences = {}

        for i in range(n_clips):
            coords = data[i][:, :, np.newaxis].astype(np.float64)
            coordinates[f"recording_{i:04d}"] = coords
            confidences[f"recording_{i:04d}"] = np.ones((n_frames, n_dim), dtype=np.float64)

        kpms.setup_project(
            project_dir,
            fix_heading=True,
            bodyparts=['placeholder'],
            skeleton=[],
            fps=30,
            overwrite=True,
        )

        kpms.update_config(
            project_dir,
            fix_heading=False,
            bodyparts=[f'dim_{i}' for i in range(n_dim)],
            use_bodyparts=[f'dim_{i}' for i in range(n_dim)],
            anterior_bodyparts=['dim_0'],
            posterior_bodyparts=[f'dim_{n_dim-1}'],
            verbose=False,
            latent_dim=self.latent_dim,
            num_states=n_states,
            kappa=self.kappa,
        )

        config = kpms.load_config(project_dir)
        int_data, int_metadata = kpms.format_data(coordinates, confidences, **config)

        model_name = f"{self.model_type}_states{n_states}"
        model = kpms.init_model(
            data=int_data,
            metadata=int_metadata,
            states=None,
            **config,
            seed=jax.random.PRNGKey(seed + n_states),
        )
        model = kpms.update_hypparams(model, kappa=self.kappa)

        model = kpms.fit_model(
            model=model,
            data=int_data,
            metadata=int_metadata,
            project_dir=project_dir,
            model_name=model_name,
            ar_only=True,
            num_iters=self.ar_iters,
            num_states=n_states,
            generate_progress_plots=False,
        )[0]

        if self.model_type == "slds":
            model = kpms.update_hypparams(model, kappa=self.kappa)
            kpms.fit_model(
                model=model,
                data=int_data,
                metadata=int_metadata,
                project_dir=project_dir,
                model_name=model_name,
                ar_only=False,
                start_iter=self.ar_iters,
                num_iters=self.full_iters,
                generate_progress_plots=False,
            )

        kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
        model_ckpt, _, _, _ = kpms.load_checkpoint(project_dir, model_name)
        model_ckpt['project_dir'] = project_dir
        model_ckpt['model_name'] = model_name

        results = kpms.extract_results(model_ckpt, int_metadata, project_dir, model_name)

        rec_names = sorted(results.keys())
        labels_list = [results[rn]['syllable'] for rn in rec_names]

        cluster_dict, stats = extract_each_syllable_as_cluster(
            labels_list,
            min_total_frames=self.params.get("min_syllable_total_frames", 0),
        )

        print_transition_stats(stats)

        original_data = None
        reconstructed_data = None
        reconstruction_mse = None
        try:
            from jax_moseq.models.keypoint_slds import estimate_coordinates
            from jax_moseq.utils import unbatch
            import jax.numpy as jnp

            Y = int_data.get("Y")
            if Y is not None:
                if isinstance(Y, dict):
                    keys = sorted(Y.keys())
                    original_data = np.stack([np.array(Y[k]) for k in keys], axis=0)
                else:
                    original_data = np.array(Y)
                if original_data.ndim > 3:
                    original_data = original_data.squeeze(-1)

            if 'states' in model_ckpt and 'params' in model_ckpt:
                Y_est = estimate_coordinates(
                    jnp.array(model_ckpt['states']['x']),
                    jnp.array(model_ckpt['states']['v']),
                    jnp.array(model_ckpt['states']['h']),
                    jnp.array(model_ckpt['params']['Cd'])
                )
                coords_est = unbatch(Y_est, *int_metadata)
                rec_keys = sorted(coords_est.keys())
                reconstructed_data = np.stack(
                    [np.array(coords_est[k]).squeeze(-1) for k in rec_keys], axis=0
                )

                if original_data is not None:
                    min_len = min(original_data.shape[1], reconstructed_data.shape[1])
                    reconstruction_mse = float(np.mean(
                        (original_data[:, :min_len] - reconstructed_data[:, :min_len]) ** 2
                    ))
                    print(f"Intention Reconstruction MSE: {reconstruction_mse:.6f}")
        except Exception as e:
            print(f"Warning: Reconstruction computation failed: {e}")

        return KPMSResult(
            labels=labels_list,
            n_states=n_states,
            method_name=self.name,
            model=model_ckpt,
            results=results,
            data=int_data,
            metadata=int_metadata,
            original=original_data,
            reconstructed=reconstructed_data,
            reconstruction_mse=reconstruction_mse,
            transition_cluster_dict=cluster_dict,
            transition_stats=stats,
        )


class KPMSKeypointMethod(BaseMethod):
    """
    Keypoint-MoSeq based state-space method for keypoint data.

    Supports both AR-HMM and SLDS model types via the model_type parameter.
    Default hyperparameters are tuned for keypoint data (from hyperparam sweep):
    - kappa=1.09e6, latent_dim=15, num_states=10, model_type=arhmm
    - Achieves keypoint reconstruction MSE ~0.002
    """
    def __init__(self, params: dict = None):
        super().__init__("kpms_keypoint", params)
        self.model_type = self.params.get("model_type", "arhmm")
        self.latent_dim = self.params.get("latent_dim", 15)
        self.kappa = self.params.get("kappa", 1.09e6)
        self.ar_iters = self.params.get("ar_iters", 100)
        self.full_iters = self.params.get("full_iters", 300)

    @property
    def method_type(self) -> MethodType:
        return MethodType.STATE_SPACE

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        raise NotImplementedError("Use fit() for state-space methods")

    def fit(
        self,
        data: np.ndarray,
        n_states: int,
        project_dir: str,
        seed: int = 0,
        kp_names: list[str] | None = None,
    ) -> KPMSResult:
        if not HAS_KPMS:
            raise ImportError("keypoint-moseq not installed")

        import shutil
        from pathlib import Path

        project_path = Path(project_dir)
        if project_path.exists():
            shutil.rmtree(project_path)
        project_path.mkdir(parents=True, exist_ok=True)

        n_clips, n_frames, n_kp, n_dim = data.shape
        coordinates = {}
        confidences = {}

        for i in range(n_clips):
            coordinates[f"recording_{i:04d}"] = data[i].astype(np.float64)
            confidences[f"recording_{i:04d}"] = np.ones((n_frames, n_kp), dtype=np.float64)

        if kp_names is None:
            kp_names = [f"kp_{i}" for i in range(n_kp)]

        kpms.setup_project(
            project_dir,
            fix_heading=True,
            bodyparts=kp_names,
            skeleton=[],
            fps=30,
            overwrite=True,
        )

        kpms.update_config(
            project_dir,
            fix_heading=True,
            bodyparts=kp_names,
            use_bodyparts=kp_names,
            anterior_bodyparts=[kp_names[0]],
            posterior_bodyparts=[kp_names[-1]],
            verbose=False,
            latent_dim=self.latent_dim,
            num_states=n_states,
            kappa=self.kappa,
        )

        config = kpms.load_config(project_dir)
        kp_data, kp_metadata = kpms.format_data(coordinates, confidences, **config)

        model_name = f"{self.model_type}_states{n_states}"
        model = kpms.init_model(
            data=kp_data,
            metadata=kp_metadata,
            states=None,
            **config,
            seed=jax.random.PRNGKey(seed + n_states),
        )
        model = kpms.update_hypparams(model, kappa=self.kappa)

        model = kpms.fit_model(
            model=model,
            data=kp_data,
            metadata=kp_metadata,
            project_dir=project_dir,
            model_name=model_name,
            ar_only=True,
            num_iters=self.ar_iters,
            num_states=n_states,
            generate_progress_plots=False,
        )[0]

        if self.model_type == "slds":
            model = kpms.update_hypparams(model, kappa=self.kappa)
            kpms.fit_model(
                model=model,
                data=kp_data,
                metadata=kp_metadata,
                project_dir=project_dir,
                model_name=model_name,
                ar_only=False,
                start_iter=self.ar_iters,
                num_iters=self.full_iters,
                generate_progress_plots=False,
            )

        kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
        model_ckpt, _, _, _ = kpms.load_checkpoint(project_dir, model_name)
        model_ckpt['project_dir'] = project_dir
        model_ckpt['model_name'] = model_name

        results = kpms.extract_results(model_ckpt, kp_metadata, project_dir, model_name)

        rec_names = sorted(results.keys())
        labels_list = [results[rn]['syllable'] for rn in rec_names]

        cluster_dict, stats = extract_each_syllable_as_cluster(
            labels_list,
            min_total_frames=self.params.get("min_syllable_total_frames", 0),
        )

        print_transition_stats(stats)

        original_data = None
        reconstructed_data = None
        reconstruction_mse = None
        try:
            from jax_moseq.models.keypoint_slds import estimate_coordinates
            from jax_moseq.utils import unbatch
            import jax.numpy as jnp

            Y = kp_data.get("Y")
            if Y is not None:
                if isinstance(Y, dict):
                    keys = sorted(Y.keys())
                    original_data = np.stack([np.array(Y[k]) for k in keys], axis=0)
                else:
                    original_data = np.array(Y)
                if original_data.ndim == 4:
                    n_clips_y, n_frames_y, n_kp_y, n_dim_y = original_data.shape
                    original_data = original_data.reshape(n_clips_y, n_frames_y, n_kp_y * n_dim_y)

            if 'states' in model_ckpt and 'params' in model_ckpt:
                Y_est = estimate_coordinates(
                    jnp.array(model_ckpt['states']['x']),
                    jnp.array(model_ckpt['states']['v']),
                    jnp.array(model_ckpt['states']['h']),
                    jnp.array(model_ckpt['params']['Cd'])
                )
                coords_est = unbatch(Y_est, *kp_metadata)
                rec_keys = sorted(coords_est.keys())
                recon_list = []
                for k in rec_keys:
                    arr = np.array(coords_est[k])
                    if arr.ndim == 3 and arr.shape[-1] == 1:
                        arr = arr.squeeze(-1)
                    elif arr.ndim == 4:
                        arr = arr.reshape(arr.shape[0], -1)
                    recon_list.append(arr)
                reconstructed_data = np.stack(recon_list, axis=0)
                if reconstructed_data.ndim == 4:
                    n_clips_r, n_frames_r, n_kp_r, n_dim_r = reconstructed_data.shape
                    reconstructed_data = reconstructed_data.reshape(n_clips_r, n_frames_r, n_kp_r * n_dim_r)

                if original_data is not None:
                    min_len = min(original_data.shape[1], reconstructed_data.shape[1])
                    reconstruction_mse = float(np.mean(
                        (original_data[:, :min_len] - reconstructed_data[:, :min_len]) ** 2
                    ))
                    print(f"Keypoint Reconstruction MSE: {reconstruction_mse:.6f}")
        except Exception as e:
            print(f"Warning: Reconstruction computation failed: {e}")

        return KPMSResult(
            labels=labels_list,
            n_states=n_states,
            method_name=self.name,
            model=model_ckpt,
            results=results,
            data=kp_data,
            metadata=kp_metadata,
            original=original_data,
            reconstructed=reconstructed_data,
            reconstruction_mse=reconstruction_mse,
            transition_cluster_dict=cluster_dict,
            transition_stats=stats,
        )


class CustomVAEMethod(BaseMethod):
    """
    Custom RNN-VAE based state-space method for intention data.

    Uses an RNN-VAE to encode behavioral sequences, then segments the latent
    space using KMeans clustering. Extracts sustained transitions similar to KPMS.

    Default hyperparameters are tuned from wandb sweep (run 0etcc386):
    - latent_dim=30, time_window=30, hidden_dim=512, beta=8.51
    - segmentation_algo=kmeans, learning_rate=0.000245
    - Achieves intention reconstruction MSE ~2.71e-05
    """
    def __init__(self, params: dict = None):
        super().__init__("custom_vae", params)
        self.latent_dim = self.params.get("latent_dim", 30)
        self.time_window = self.params.get("time_window", 30)
        self.hidden_dim = self.params.get("hidden_dim", 512)
        self.beta = self.params.get("beta", 8.51)
        self.learning_rate = self.params.get("learning_rate", 0.000245)
        self.batch_size = self.params.get("batch_size", 256)
        self.max_epochs = self.params.get("max_epochs", 200)
        self.segmentation_algo = self.params.get("segmentation_algo", "kmeans")

    @property
    def method_type(self) -> MethodType:
        return MethodType.STATE_SPACE

    def fit_predict(
        self,
        X: np.ndarray,
        n_clusters: int,
        seed: int = 0,
    ) -> np.ndarray:
        raise NotImplementedError("Use fit() for state-space methods")

    def fit(
        self,
        data: np.ndarray,
        n_states: int,
        project_dir: str,
        seed: int = 0,
    ) -> KPMSResult:
        if not HAS_CUSTOM_VAE:
            raise ImportError("Custom VAE dependencies not installed (torch, etc.)")

        import os
        from pathlib import Path

        project_path = Path(project_dir)
        project_path.mkdir(parents=True, exist_ok=True)

        n_clips, n_frames, n_dim = data.shape
        print(f"Training CustomVAE on {n_clips} clips, {n_frames} frames, {n_dim} dims")

        hyperparams = VAEHyperparams(
            latent_dim=self.latent_dim,
            time_window=self.time_window,
            hidden_dim=self.hidden_dim,
            beta=self.beta,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
        )

        model = RNNVAE(
            input_dim=n_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
        )

        torch.manual_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = VAETrainer(model, hyperparams, device=device)

        train_result = trainer.train(
            data,
            wandb_logger=None,
            modality_name="intention",
            val_ratio=0.1,
            seed=seed,
        )

        print(f"VAE training completed at epoch {train_result.final_epoch}")

        latent_embeddings = trainer.encode_per_frame(data)
        print(f"Per-frame latent embeddings shape: {latent_embeddings.shape}")

        seg_result = segment_latent(
            latent_embeddings,
            n_clusters=n_states,
            method=self.segmentation_algo,
            random_state=seed,
        )

        labels_list = seg_result.labels_list
        print(f"Segmentation complete: {n_states} clusters using {self.segmentation_algo}")

        cluster_dict, stats = extract_each_syllable_as_cluster(
            labels_list,
            min_total_frames=self.params.get("min_syllable_total_frames", 0),
        )

        print_transition_stats(stats)

        original_data = data.copy()
        reconstructed_data = trainer.reconstruct(data)
        reconstruction_mse = float(np.mean((original_data - reconstructed_data) ** 2))
        print(f"Intention Reconstruction MSE: {reconstruction_mse:.6f}")

        model_path = os.path.join(project_dir, "vae_model.pt")
        torch.save(train_result.model.state_dict(), model_path)
        print(f"Saved VAE model to {model_path}")

        model_info = {
            "hyperparams": {
                "latent_dim": self.latent_dim,
                "time_window": self.time_window,
                "hidden_dim": self.hidden_dim,
                "beta": self.beta,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "segmentation_algo": self.segmentation_algo,
            },
            "final_epoch": train_result.final_epoch,
            "train_losses": train_result.train_losses,
            "val_losses": train_result.val_losses,
            "model_path": model_path,
            "project_dir": project_dir,
        }

        return KPMSResult(
            labels=labels_list,
            n_states=n_states,
            method_name=self.name,
            model=model_info,
            results=None,
            data={"latent_embeddings": latent_embeddings},
            metadata=None,
            original=original_data,
            reconstructed=reconstructed_data,
            reconstruction_mse=reconstruction_mse,
            transition_cluster_dict=cluster_dict,
            transition_stats=stats,
        )


class W2DMCQposKMeans(W2DMCIntKMeans):
    def __init__(self, params: dict = None):
        BaseMethod.__init__(self, "w2_dmc_qpos_kmeans", params)
        self.mds_components = self.params.get("mds_components", 3)


class W2DMCEkpKMeans(W2DMCIntKMeans):
    def __init__(self, params: dict = None):
        BaseMethod.__init__(self, "w2_dmc_ekp_kmeans", params)
        self.mds_components = self.params.get("mds_components", 3)


class W2DMPIntKMeans(W2DMCIntKMeans):
    def __init__(self, params: dict = None):
        BaseMethod.__init__(self, "w2_dmp_int_kmeans", params)
        self.mds_components = self.params.get("mds_components", 3)


class W2DMPQposKMeans(W2DMCIntKMeans):
    def __init__(self, params: dict = None):
        BaseMethod.__init__(self, "w2_dmp_qpos_kmeans", params)
        self.mds_components = self.params.get("mds_components", 3)


class W2DMPEkpKMeans(W2DMCIntKMeans):
    def __init__(self, params: dict = None):
        BaseMethod.__init__(self, "w2_dmp_ekp_kmeans", params)
        self.mds_components = self.params.get("mds_components", 3)


METHOD_REGISTRY: dict[str, type[BaseMethod]] = {
    "w2_dmc_int_kmeans": W2DMCIntKMeans,
    "w2_dmc_qpos_kmeans": W2DMCQposKMeans,
    "w2_dmc_ekp_kmeans": W2DMCEkpKMeans,
    "w2_dmp_int_kmeans": W2DMPIntKMeans,
    "w2_dmp_qpos_kmeans": W2DMPQposKMeans,
    "w2_dmp_ekp_kmeans": W2DMPEkpKMeans,
    "w2_unif_kmeans": W2UnifKMeans,
    "w2_full_kmeans": W2FullKMeans,
    "gw_dmc_int_kmeans": GWDMCIntKMeans,
    "kpms_intention": KPMSIntentionMethod,
    "kpms_keypoint": KPMSKeypointMethod,
    "custom_vae": CustomVAEMethod,
}


def _register_mapper_methods() -> None:
    """Lazy-register Mapper methods. Called the first time get_method() is invoked.

    Done lazily because the mapper_method module imports names from this module
    (BaseMethod, KPMSResult, etc.), which would cause a partial-import error if
    done at module import time.
    """
    if "mapper_int" in METHOD_REGISTRY:
        return
    try:
        from topo_mimic.benchmark.mapper.mapper_method import (
            MapperIntMethod,
            MapperQposMethod,
            MapperEKPMethod,
        )
    except ImportError as exc:
        print(f"[METHOD_REGISTRY] Mapper methods not registered: {exc}")
        return
    METHOD_REGISTRY["mapper_int"] = MapperIntMethod
    METHOD_REGISTRY["mapper_qpos"] = MapperQposMethod
    METHOD_REGISTRY["mapper_ekp"] = MapperEKPMethod


def get_method(name: str, params: dict = None) -> BaseMethod:
    _register_mapper_methods()
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[name](params)


def get_clustering_method(name: str, params: dict = None, seed: int = 0) -> BaseMethod:
    method = get_method(name, params)
    if method.method_type != MethodType.CLUSTERING:
        raise ValueError(f"{name} is not a clustering method")
    return method


def get_kpms_method(name: str, params: dict = None) -> BaseMethod:
    method = get_method(name, params)
    if method.method_type != MethodType.STATE_SPACE:
        raise ValueError(f"{name} is not a state-space method")
    return method


def list_available_methods() -> dict[str, list[str]]:
    clustering = []
    state_space = []

    for name, cls in METHOD_REGISTRY.items():
        instance = cls()
        if instance.method_type == MethodType.CLUSTERING:
            clustering.append(name)
        else:
            state_space.append(name)

    return {
        "clustering": clustering,
        "state_space": state_space,
    }
