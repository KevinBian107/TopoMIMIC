# Automatic Label Assignment

Automatically recommend labels for discovered behavioral clusters by comparing them to ground truth transitions using Wasserstein-2 distance.

## Overview

The labeling module sits between the experiment pipeline (which discovers clusters) and the analysis pipeline (which evaluates clusters against ground truth). It computes W2 distances between discovered clusters and GT transition types, then recommends the best-matching labels for each cluster.

```
experiment.py → labeling/ → analysis/
    ↓              ↓            ↓
 clusters    recommendations  evaluation
```

## Pipeline Stages

### Stage 1: GT Transition Extraction

Extracts ground truth transitions from `behavior_labels.csv` and normalizes them to a fixed length (`target_len`). Short intermediate behaviors are bridged (e.g., `Rear → Walking → Orienting(short) → Walking` becomes `Rear → Walking`).

### Stage 2: W2 Distance Computation

For each discovered cluster, computes the average W2 (Earth Mover's) distance to each GT transition type. Uses representative sampling for efficiency:
- Sample up to 20 cycles per cluster
- Sample up to 10 GT transitions per type
- Normalize all trajectories to `target_len` before comparison

### Stage 3: Label Recommendation

Ranks GT transition types by distance for each cluster. The closest match is the recommended label.

### Stage 4: GUI Assignment (Optional)

Gradio-based interface for reviewing recommendations and manually assigning labels with video playback.

## Usage

### Command Line

```bash
python -m scripts.benchmark.labeling.main configs/benchmark/labeling.yaml
```

### Configuration

Key settings in `configs/benchmark/labeling.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `w2_kmeans` | Experiment method to label |
| `seed` | `0` | Random seed |
| `gt_extraction.target_len` | `100` | Normalize transitions to this length |
| `gt_extraction.short_threshold` | `10` | Max frames to bridge intermediate behaviors |
| `distance.max_cycles_per_cluster` | `20` | Cycles sampled per cluster |
| `distance.max_gt_samples` | `10` | GT samples per transition type |
| `recommendations.top_k` | `5` | Number of recommendations per cluster |

### Launching the GUI

Set `launch_gui: true` in the config, or launch standalone:

```python
from scripts.benchmark.labeling.gui import launch_gui_standalone

launch_gui_standalone(
    experiment_dir="out/benchmark_experiment",
    method="w2_kmeans",
    seed=0,
)
```

## Output Files

```
out/benchmark_experiment/{method}/
├── labeling/
│   ├── recommendations_seed_{seed}.csv   # Ranked recommendations
│   └── gt_statistics.csv                  # GT transition statistics
└── labels_seed_{seed}.yaml                # Updated labels (from GUI)
```

### Recommendations CSV Format

| Column | Description |
|--------|-------------|
| `cluster_id` | Cluster identifier |
| `rank` | Recommendation rank (1 = best) |
| `label` | GT transition type (e.g., "Rear <-> Walking") |
| `distance` | W2 distance to cluster |

### GT Statistics CSV Format

| Column | Description |
|--------|-------------|
| `transition_type` | e.g., "Immobility <-> Walking" |
| `count` | Number of GT instances |
| `avg_frames` | Average original length |
| `n_upsampled` | Count normalized by upsampling |
| `n_downsampled` | Count normalized by downsampling |

## Integration with Analysis

The analysis pipeline automatically loads labels from `labels_seed_{seed}.yaml`. After using the labeling module (either accepting recommendations or manually assigning via GUI), run the analysis:

```bash
python -m scripts.benchmark.analysis.main configs/benchmark/analysis.yaml
```

## Transition Type Format

GT transition types use the format `"Behavior1 <-> Behavior2"` with behaviors sorted alphabetically:
- `"Immobility <-> Walking"`
- `"Rear <-> Walking"`
- `"Grooming <-> Immobility"`

This ensures consistent matching regardless of transition direction.
