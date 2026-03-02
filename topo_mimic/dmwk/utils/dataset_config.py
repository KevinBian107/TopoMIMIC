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


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "rodent": DatasetPreset(
        total_clips=842,
        test_size=200,
        default_data_path="data/rodent_ar.h5",
        labels_csv="data/behavior_labels.csv",
        supported_dm_keys=["intention", "qpos", "ekp"],
        qpos_dims=74,
        root_strip_dims=7,
    ),
    "walker": DatasetPreset(
        total_clips=2000,
        test_size=400,
        default_data_path="data/walker.h5",
        labels_csv="walker_behavior_labels.csv",
        supported_dm_keys=["intention", "qpos"],
        qpos_dims=9,
        root_strip_dims=0,
    ),
}


def get_dataset_preset(name: str) -> DatasetPreset:
    if name not in DATASET_PRESETS:
        raise ValueError(
            f"Unknown dataset: {name!r}. Available: {list(DATASET_PRESETS.keys())}"
        )
    return DATASET_PRESETS[name]
