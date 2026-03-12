# Walker Generation Module

Config-driven pipeline for generating walker long-sequence clips in TopoVNL format.

## Overview

Replaces the 3-step manual process:
1. `walker/scripts/generate_topo_dataset.py` (Step 1 policy → raw clips)
2. `walker/scripts/extract_imitation_activations.py` (Step 2 policy → activations)
3. `scripts/convert_walker_data.py` (legacy → unified format)

With a single command:
```bash
python3 -m walker_generation configs/walker_generation/path_sequence.yaml
```

## Key Features

- **Multi-transition clips**: Each clip contains multiple behavior transitions
- **Configurable pools**: Source and target behavior pools for random mode selection
- **Variable transitions**: Optional variable-length transition blending
- **Direct output**: Writes `/intention` and `/qpos_rollout` to H5 + behavior labels CSV
- **Single-pass extraction**: All clips processed through Step 2 in one batched pass

## Pipeline Flow

```
Config YAML
    │
    ▼
Generate long-sequence clips:
    │
    ├─ Build schedule with N transitions from behavior pools
    ├─ Generate rollouts via Step 1 policy (batched vmap)
    └─ Collect full-length clips with segment metadata
    │
    ▼
Extract activations via Step 2 policy (batched vmap)
    │
    ▼
Write unified output (H5 + CSV)
```

## Config Structure

```yaml
long_sequence:
  n_clips: 50
  clip_length: 500
  n_transitions: 5
  source_pool:
    - walk_slow
    - walk_backward
  target_pool:
    - hop
    - tiptoe
  transition_length: 25
  variable_transition_length: true
  transition_length_min: 15
  transition_length_max: 30

checkpoints:
  step1_path: data/checkpoints/walker_multi_behavior/260215_202424
  step2_path: data/checkpoints/walker_imitation/260224_030923

schedule:
  rollout_frames: 1500
  warmup_frames: 200
  warmup_transition_frames: 50

rollout:
  batch_size: 10
  extraction_batch_size: 5
  seed: 0

output:
  h5_path: data/walker_long.h5
  csv_path: walker_long_behavior_labels.csv
  demo_clip_video: outputs/walker_long_demo/demo_clip.mp4
```

## Output Format

**H5 file** (`data/walker_long.h5`):
- `/intention`: `(N, T, 16)` — Step 2 latent mean embeddings
- `/qpos_rollout`: `(N, T, 9)` — Agent joint positions

**CSV file** (`walker_long_behavior_labels.csv`):
- `Clip`: Clip index
- `Behavior`: Mode name (e.g. "walk_slow", "hop")
- `Start_Frame`: Segment start frame within clip
- `End_Frame`: Segment end frame within clip

## Available Behavior Modes

| Index | Name |
|-------|------|
| 0 | stand |
| 1 | walk_slow |
| 2 | run |
| 3 | knee_down |
| 4 | walk_backward |
| 5 | hop |
| 6 | walk_fast |
| 7 | tiptoe |

## Requirements

- Walker environment packages: `vnl_playground`, `track_mjx`
- GPU with sufficient VRAM for batched rollouts
- Step 1 + Step 2 trained checkpoints

## Module Structure

```
walker_generation/
├── __init__.py      # Package init, exposes run_generation()
├── __main__.py      # CLI entry point
├── config.py        # Dataclasses + YAML loader
├── generate.py      # Core pipeline logic
└── output.py        # H5/CSV writer
```
