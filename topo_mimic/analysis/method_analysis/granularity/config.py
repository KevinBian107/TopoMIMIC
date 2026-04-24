"""Configuration dataclasses for granularity analysis."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dm_key: str = "intention"
    qpos_key: str = "qpos"
    rodent_rollout_dir: str = ""
    split: str = "train"
    pca_dims: int | None = None
    xml_path: str = "topo_mimic/extraction/xml/rodent.xml"
    stac_data_path: str = "data/rodent_reference_clips.h5"


@dataclass
class TakensConfig:
    m: int = 3
    tau: int = 6


@dataclass
class DMConfig:
    k_neighbors: int = 3
    epsilon: float = 0.99
    use_dm_path: bool = False
    segment_gap_limit: int = 50
    min_segment_len: int = 50


@dataclass
class MultiPersistenceConfig:
    enabled: bool = True
    pers_delta_values: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0])


@dataclass
class JointConfig:
    head_joints: list[str] = field(default_factory=lambda: ["atlas", "vertebra_axis_twist"])
    arm_joints: list[str] = field(default_factory=lambda: ["elbow_L", "elbow_R", "wrist_L", "wrist_R"])
    root_indices: list[int] = field(default_factory=lambda: [0, 1, 2])
    skip_root_dims: int = 7


@dataclass
class StationaryConfig:
    enabled: bool = True
    root_displacement_threshold: float = 0.5


@dataclass
class ClusteringConfig:
    target_len: int = 70
    only_max: bool = True
    mds_components: int = 3


@dataclass
class RedundancyConfig:
    enabled: bool = True
    n_clusters_range: list[int] = field(default_factory=lambda: [5, 10, 15, 20, 25])
    dtw_threshold: float = 0.1
    dtw_percentile: float = 10.0
    threshold_method: str = "both"
    top_k_similar_pairs: int = 5
    max_dtw_comparisons: int = 100
    cache_clustering: bool = True
    use_cached_clustering: bool = False
    cache_dtw_matrix: bool = True
    use_cached_dtw: bool = False
    seed: int = 42


@dataclass
class RenderingConfig:
    enabled: bool = True
    pad: int = 0
    max_samples: int = 5
    framerate: int = 30
    target_size: tuple[int, int] = (320, 240)


@dataclass
class PlottingConfig:
    figsize: tuple[float, float] = (4.0, 3.5)
    dpi: int = 300


@dataclass
class GranularityConfig:
    output_dir: str = "outputs/granularity"
    data: DataConfig = field(default_factory=DataConfig)
    takens: TakensConfig = field(default_factory=TakensConfig)
    dm: DMConfig = field(default_factory=DMConfig)
    multi_persistence: MultiPersistenceConfig = field(default_factory=MultiPersistenceConfig)
    joints: JointConfig = field(default_factory=JointConfig)
    stationary: StationaryConfig = field(default_factory=StationaryConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    redundancy: RedundancyConfig = field(default_factory=RedundancyConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)


def load_config(config_path: str | Path) -> GranularityConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = DataConfig(**cfg.get("data", {}))
    takens_cfg = TakensConfig(**cfg.get("takens", {}))
    dm_cfg = DMConfig(**cfg.get("dm", {}))
    multi_pers_cfg = MultiPersistenceConfig(**cfg.get("multi_persistence", {}))
    joints_cfg = JointConfig(**cfg.get("joints", {}))
    stationary_cfg = StationaryConfig(**cfg.get("stationary", {}))
    clustering_cfg = ClusteringConfig(**cfg.get("clustering", {}))
    redundancy_cfg = RedundancyConfig(**cfg.get("redundancy", {}))

    rendering_dict = cfg.get("rendering", {})
    if "target_size" in rendering_dict:
        rendering_dict["target_size"] = tuple(rendering_dict["target_size"])
    rendering_cfg = RenderingConfig(**rendering_dict)

    plotting_dict = cfg.get("plotting", {})
    if "figsize" in plotting_dict:
        plotting_dict["figsize"] = tuple(plotting_dict["figsize"])
    plotting_cfg = PlottingConfig(**plotting_dict)

    return GranularityConfig(
        output_dir=cfg.get("output_dir", "outputs/granularity"),
        data=data_cfg,
        takens=takens_cfg,
        dm=dm_cfg,
        multi_persistence=multi_pers_cfg,
        joints=joints_cfg,
        stationary=stationary_cfg,
        clustering=clustering_cfg,
        redundancy=redundancy_cfg,
        rendering=rendering_cfg,
        plotting=plotting_cfg,
    )
