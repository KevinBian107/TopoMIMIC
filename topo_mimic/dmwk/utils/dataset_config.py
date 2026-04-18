"""Dataset preset registry for multi-dataset support."""

from dataclasses import dataclass, field


@dataclass
class DatasetPreset:
    total_clips: int
    test_size: int
    default_data_path: str
    labels_csv: str
    supported_dm_keys: list[str]
    qpos_dims: int
    root_strip_dims: int
    xml_path: str = ""
    stac_data_path: str = ""
    skip_clips: tuple[int, ...] = ()


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "rodent": DatasetPreset(
        total_clips=842,
        test_size=200,
        default_data_path="data/rodent_ar.h5",
        labels_csv="behavior_labels/rodent.csv",
        supported_dm_keys=["intention", "qpos", "ekp"],
        qpos_dims=74,
        root_strip_dims=7,
        xml_path="topo_mimic/dmwk/xml/rodent.xml",
        stac_data_path="data/rodent_reference_clips.h5",
        skip_clips=(80, 187, 227, 246, 267, 392, 434, 456, 475, 491, 562, 610, 685),
    ),
    "walker_short": DatasetPreset(
        total_clips=600,
        test_size=0,
        default_data_path="data/walker_short.h5",
        labels_csv="behavior_labels/walker_short.csv",
        supported_dm_keys=["intention", "qpos"],
        qpos_dims=9,
        root_strip_dims=0,
        xml_path="topo_mimic/dmwk/xml/walker.xml",
        stac_data_path="",
        skip_clips=(),
    ),
    "walker_long": DatasetPreset(
        total_clips=10,
        test_size=0,
        default_data_path="data/walker_long.h5",
        labels_csv="behavior_labels/walker_long.csv",
        supported_dm_keys=["intention", "qpos"],
        qpos_dims=9,
        root_strip_dims=0,
        xml_path="topo_mimic/dmwk/xml/walker.xml",
        stac_data_path="",
        skip_clips=(),
    ),
}


def get_dataset_preset(name: str) -> DatasetPreset:
    if name not in DATASET_PRESETS:
        raise ValueError(
            f"Unknown dataset: {name!r}. Available: {list(DATASET_PRESETS.keys())}"
        )
    return DATASET_PRESETS[name]
