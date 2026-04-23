"""Scale-sweep configuration dataclasses.

Mirrors the dataclass-of-dataclasses pattern used by `topo_mimic/dmwk/dm_pipeline.py`
and `topo_mimic/benchmark/experiment_transitions.py`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataCfg:
    dataset: str = "rodent"
    rodent_rollout_dir: str = "data/rodent_ar.h5"
    behavior_labels_csv: str = "behavior_labels/rodent_labels.csv"
    start_clip: int = 0
    end_clip: int = 842  # full rodent_labels.csv coverage (clips 0..841)
    xml_path: str = "topo_mimic/dmwk/xml/rodent.xml"
    stac_data_path: str = "data/rodent_reference_clips.h5"
    respect_skip_clips: bool = True


@dataclass
class KPMSSelectionCfg:
    criterion: str = "eml"
    min_usage_ratio: float = 0.8


@dataclass
class KPMSCfg:
    enabled: bool = True
    kappa_grid: list[float] = field(default_factory=lambda: [1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7])
    num_states_grid: list[int] = field(default_factory=lambda: [5, 10, 15, 20, 30])
    seeds: list[int] = field(default_factory=lambda: [0])
    ar_iters: int = 50
    full_iters: int = 300
    latent_dim: int = 10
    model_type: str = "arhmm"
    selection: KPMSSelectionCfg = field(default_factory=KPMSSelectionCfg)


@dataclass
class MapperLensSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MapperTakensCfg:
    m: int = 3
    tau: int = 6


@dataclass
class MapperCfg:
    enabled: bool = True
    feature_spaces: list[str] = field(default_factory=lambda: ["intention", "qpos", "ekp"])
    lens_grid: list[MapperLensSpec] = field(default_factory=lambda: [
        MapperLensSpec(name="knn_density", params={"knn_k": 3}),
        MapperLensSpec(name="pca_1d"),
        MapperLensSpec(name="eccentricity", params={"knn_k": 3}),
    ])
    n_intervals_grid: list[int] = field(default_factory=lambda: [5, 10, 15, 20, 25])
    overlap_frac: float = 0.5
    dbscan_eps: str | float = "auto"
    dbscan_min_samples: int = 5
    takens_per_feature_space: dict[str, MapperTakensCfg] = field(default_factory=lambda: {
        "intention": MapperTakensCfg(m=3, tau=6),
        "qpos": MapperTakensCfg(m=3, tau=9),
        "ekp": MapperTakensCfg(m=3, tau=8),
    })
    min_total_frames: int = 500


@dataclass
class DiscoveryCfg:
    min_member_clips: int = 3
    min_majority_fraction: float = 0.50
    min_total_frames: int = 500
    second_behavior_threshold: float = 0.20
    groom_singleton: bool = True
    dm_wkc_reference_path: str | None = None


@dataclass
class TracePngCfg:
    enabled: bool = True
    n_exemplar_clips: int = 5


@dataclass
class VisualizationCfg:
    discovery_eyeball_png: bool = True
    trace_png: TracePngCfg = field(default_factory=TracePngCfg)


@dataclass
class IoCfg:
    output_dir: str = "outputs/scale_sweep/v1"
    log_level: str = "info"
    save_checkpoints: bool = True


@dataclass
class ScaleSweepConfig:
    io: IoCfg = field(default_factory=IoCfg)
    data: DataCfg = field(default_factory=DataCfg)
    kpms: KPMSCfg = field(default_factory=KPMSCfg)
    mapper: MapperCfg = field(default_factory=MapperCfg)
    discovery: DiscoveryCfg = field(default_factory=DiscoveryCfg)
    visualization: VisualizationCfg = field(default_factory=VisualizationCfg)

    def __post_init__(self) -> None:
        if self.data.start_clip >= self.data.end_clip:
            raise ValueError(
                f"data.start_clip ({self.data.start_clip}) must be < data.end_clip ({self.data.end_clip})"
            )
        self.kpms.kappa_grid = [float(k) for k in self.kpms.kappa_grid]
        for spec in self.mapper.lens_grid:
            if spec.name not in {"knn_density", "pca_1d", "eccentricity"}:
                raise ValueError(f"Unknown lens name: {spec.name!r}")
        for fs in self.mapper.feature_spaces:
            if fs not in self.mapper.takens_per_feature_space:
                raise ValueError(
                    f"feature_space {fs!r} has no takens config; add to mapper.takens_per_feature_space"
                )

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:8]


def _instantiate(cls, raw: dict[str, Any]):
    if raw is None:
        return cls()
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in raw.items():
        if k not in field_types:
            continue
        kwargs[k] = v
    return cls(**kwargs)


def load_config(path: str | Path) -> ScaleSweepConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}

    kpms_raw = dict(raw.get("kpms", {}) or {})
    sel = kpms_raw.pop("selection", None)
    if sel is not None:
        kpms_raw["selection"] = KPMSSelectionCfg(**sel)
    kpms = KPMSCfg(**kpms_raw) if kpms_raw else KPMSCfg()

    mapper_raw = dict(raw.get("mapper", {}) or {})
    lens_raw = mapper_raw.pop("lens_grid", None)
    if lens_raw is not None:
        mapper_raw["lens_grid"] = [MapperLensSpec(**spec) for spec in lens_raw]
    takens_raw = mapper_raw.pop("takens_per_feature_space", None)
    if takens_raw is not None:
        mapper_raw["takens_per_feature_space"] = {
            fs: MapperTakensCfg(**t) for fs, t in takens_raw.items()
        }
    mapper = MapperCfg(**mapper_raw) if mapper_raw else MapperCfg()

    viz_raw = dict(raw.get("visualization", {}) or {})
    trace_raw = viz_raw.pop("trace_png", None)
    if trace_raw is not None:
        viz_raw["trace_png"] = TracePngCfg(**trace_raw)
    viz = VisualizationCfg(**viz_raw) if viz_raw else VisualizationCfg()

    return ScaleSweepConfig(
        io=_instantiate(IoCfg, raw.get("io")),
        data=_instantiate(DataCfg, raw.get("data")),
        kpms=kpms,
        mapper=mapper,
        discovery=_instantiate(DiscoveryCfg, raw.get("discovery")),
        visualization=viz,
    )
