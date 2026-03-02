# Walker Generation Module

Config-driven pipeline for generating walker transition clips in TopoVNL format.

## Overview

Replaces the 3-step manual process:
1. `walker/scripts/generate_topo_dataset.py` (Step 1 policy → raw clips)
2. `walker/scripts/extract_imitation_activations.py` (Step 2 policy → activations)
3. `scripts/convert_walker_data.py` (legacy → unified format)

With a single command:
```bash
python3 -m walker_generation configs/walker_generation/default.yaml
```

## Key Features

- **Per-transition-pair control**: Specify exact `(src, dst, n_clips)` tuples
- **Targeted scheduling**: Restricts `allowed_modes` to `[src, dst]` for each pair
- **Direct output**: Writes `/intention` and `/qpos_rollout` to H5 + behavior labels CSV
- **Single-pass extraction**: All clips processed through Step 2 in one batched pass

## Pipeline Flow

```
Config YAML
    │
    ▼
For each TransitionSpec(src, dst, n_clips):
    │
    ├─ Restrict allowed_modes to [src_idx, dst_idx]
    ├─ Generate rollouts via Step 1 policy (batched vmap)
    ├─ Crop clips around transitions
    └─ Filter for exact src→dst direction
    │
    ▼
Collect ALL clips from ALL pairs
    │
    ▼
Extract activations via Step 2 policy (batched vmap)
    │
    ▼
Write unified output (H5 + CSV)
```

## Config Structure

```yaml
transitions:
  - src: walk_slow      # Source behavior mode name
    dst: hop            # Destination behavior mode name
    n_clips: 300        # Number of clips for this pair

checkpoints:
  step1_path: walker/checkpoints/walker_multi_behavior/260215_202424
  step2_path: walker/checkpoints/walker_imitation/260224_030923

schedule:
  clip_length: 500          # Frames per output clip
  rollout_frames: 2000      # Frames per rollout (before cropping)
  warmup_frames: 200        # Standing warmup (not recorded)
  mode_duration_min: 250    # Min frames per behavior segment
  transition_mean: 100      # Mean transition blend duration

rollout:
  batch_size: 200           # Step 1 vmap batch size
  extraction_batch_size: 100 # Step 2 vmap batch size
  seed: 0

output:
  h5_path: data/walker.h5
  csv_path: walker_behavior_labels.csv
```

## Output Format

**H5 file** (`data/walker.h5`):
- `/intention`: `(N, T, 16)` — Step 2 latent mean embeddings
- `/qpos_rollout`: `(N, T, 9)` — Agent joint positions

**CSV file** (`walker_behavior_labels.csv`):
- `Clip`: Clip index
- `Behavior`: Mode name (e.g. "walk_slow", "hop")
- `Start_Frame`: Segment start frame within clip
- `End_Frame`: Segment end frame within clip (-1 = end of clip)

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
