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
    reconstructed: np.ndarray = None
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


def print_transition_stats(stats: dict) -> None:
    """Print formatted transition statistics."""
    print(f"\n{'='*60}")
    print("KPMS Sustained Transition Analysis")
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


class W2DMKMeans(BaseMethod):
    """
    Wasserstein-2 based clustering on DM cycles using MDS embedding + KMeans.
    Requires precomputed W2 distance matrix passed via params.
    """
    def __init__(self, params: dict = None):
        super().__init__("w2_dm_kmeans", params)
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

    Same algorithm as W2DMKMeans but uses windows from all frames
    instead of topologically-discovered cycles. This serves as a
    baseline to evaluate whether topology-based cycle detection
    improves clustering quality.
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


class GWDMKMeans(BaseMethod):
    """
    Gromov-Wasserstein based clustering on DM cycles using MDS embedding + KMeans.
    Requires precomputed GW distance matrix passed via params.
    """
    def __init__(self, params: dict = None):
        super().__init__("gw_dm_kmeans", params)
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
    """
    def __init__(self, params: dict = None):
        super().__init__("kpms_intention", params)
        self.model_type = self.params.get("model_type", "arhmm")
        self.latent_dim = self.params.get("latent_dim", 10)
        self.kappa = self.params.get("kappa", 1e5)
        self.ar_iters = self.params.get("ar_iters", 50)
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

        cluster_dict, stats = extract_sustained_transitions(
            labels_list,
            min_duration=self.params.get("min_sustained_duration", 50),
            top_k=self.params.get("top_k_transitions", 10),
            min_instances=self.params.get("min_instances", 3),
        )

        print_transition_stats(stats)

        return KPMSResult(
            labels=labels_list,
            n_states=n_states,
            method_name=self.name,
            model=model_ckpt,
            results=results,
            transition_cluster_dict=cluster_dict,
            transition_stats=stats,
        )


METHOD_REGISTRY: dict[str, type[BaseMethod]] = {
    "w2_dm_kmeans": W2DMKMeans,
    "w2_unif_kmeans": W2UnifKMeans,
    "gw_dm_kmeans": GWDMKMeans,
    "kpms_intention": KPMSIntentionMethod,
}


def get_method(name: str, params: dict = None) -> BaseMethod:
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
