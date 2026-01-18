# Transition Mapping Documentation

This document explains how cluster labels are mapped to ground truth data for the transition cluster comparison analysis.

## Behavior Types

### Primitive Behaviors
These are mutually exclusive base behaviors - a frame can only have ONE primitive behavior at a time:
- Walking
- Immobility
- Orienting
- Grooming
- Rear (merged from Supported Rear and Unsupported Rear)

**Note:** By default, `Supported Rear` and `Unsupported Rear` are merged into a single `Rear` category. This can be disabled via configuration.

### Turn Direction (Flag Variable)
Turn direction is a **flag variable** that modifies each behavior segment:
- **Left (L)**: Segment has Left Turn overlay
- **Right (R)**: Segment has Right Turn overlay
- **Straight (S)**: No turn overlay

Turn direction is determined by checking if Left Turn or Right Turn labels overlap with the behavior segment. If both overlap, the turn closest to the transition point is used.

## Short Intermediate Behavior Bridging

When analyzing transitions, short intermediate behaviors are bridged to create cleaner transition segments.

### How It Works

If a sequence looks like:
```
Rear (100 frames) → Walk (50 frames) → Orienting (8 frames) → Walk (60 frames)
```

And `Orienting` is below the threshold (default: 10 frames), the entire sequence becomes:
```
Rear → Walk (single transition)
```

The short `Orienting` segment is bridged, and the two `Walk` segments are merged.

### Configuration

```yaml
short_behavior_threshold: 10  # Max frames for intermediate behavior to be bridged
```

## Transition Variants (3x3 Matrix)

Each transition can have up to 9 variants based on turn directions:

| From \ To | S-Walk | L-Walk | R-Walk |
|-----------|--------|--------|--------|
| S-Rear    | S-Rear → S-Walk | S-Rear → L-Walk | S-Rear → R-Walk |
| L-Rear    | L-Rear → S-Walk | L-Rear → L-Walk | L-Rear → R-Walk |
| R-Rear    | R-Rear → S-Walk | R-Rear → L-Walk | R-Rear → R-Walk |

### Turn Direction Determination

For each segment in a transition:

1. Find all Left Turn / Right Turn labels that overlap with the segment
2. If no overlaps → **Straight**
3. If one type overlaps → use that direction
4. If both types overlap → use the turn **closest to the transition point**

The transition point is the frame where the "from" behavior ends.

## Label Parsing Rules

The analysis parses cluster labels by splitting on whitespace and categorizing each word.

### Case 1: Transition (2+ Primitives)
**Example labels:** `"Walking Immobility"`, `"Walking Rear Left_Turn"`

When a label contains **2 or more primitive behaviors**, it's treated as a **transition**.

| Label | Primitives | Turn Context | Ground Truth Lookup |
|-------|------------|--------------|---------------------|
| `Walking Immobility` | [Walking, Immobility] | - | All Walking <-> Immobility transitions |
| `Walking Rear Left_Turn` | [Walking, Rear] | Left Turn | Walking <-> Rear transitions (turn context noted) |

### Case 2: Intersection (1 Primitive + 1+ Turn)
**Example labels:** `"Walking Left_Turn"`, `"Orienting Right_Turn"`

When a label contains **exactly 1 primitive** and **1+ turn directions**, it's treated as an **intersection**.

| Label | Primitives | Turn | Ground Truth Lookup |
|-------|------------|------|---------------------|
| `Walking Left_Turn` | [Walking] | Left Turn | Frames where Walking AND Left Turn are labeled |
| `Orienting Right_Turn` | [Orienting] | Right Turn | Frames where Orienting AND Right Turn are labeled |

### Case 3: Single Behavior (Skipped)
**Example labels:** `"Walking"`, `"Left_Turn"`

Labels with only a single behavior are **skipped** in the comparison analysis.

## Transition Ranking Analysis

The analysis outputs a ranking of transition variants by frequency in the ground truth data.

### Output Files

1. **`transition_variants_ranking.csv`**: Full ranking with turn directions
   ```csv
   Rank,From_Behavior,To_Behavior,From_Turn,To_Turn,Count,Formatted,Base_Transition
   1,Immobility,Walking,straight,straight,45,S-Immobility -> S-Walking,Immobility <-> Walking
   2,Immobility,Walking,straight,left,23,S-Immobility -> L-Walking,Immobility <-> Walking
   ...
   ```

2. **`base_transition_counts.csv`**: Counts ignoring turn directions
   ```csv
   Rank,Behavior_1,Behavior_2,Transition,Count
   1,Immobility,Walking,Immobility <-> Walking,120
   2,Orienting,Walking,Orienting <-> Walking,85
   ...
   ```

3. **`transition_matrix.csv`**: Matrix format (behaviors × behaviors)

4. **`transition_matrix_with_turns.csv`**: Matrix with turn variants expanded

## Cluster Grouping

Multiple clusters with the **same ground truth key** are grouped together:

- Clusters labeled `"Walking Immobility"` and `"Immobility Walking"` → grouped (same transition)
- Clusters labeled `"Walking Left_Turn"` and `"Left_Turn Walking"` → grouped (same intersection)
- Clusters labeled `"Walking Left_Turn"` and `"Walking Right_Turn"` → NOT grouped (different turn)

## Behavior Groups

Transitions are organized into groups for visualization:

| Group | Behaviors Included |
|-------|-------------------|
| Immobility | Immobility |
| Walking | Walking |
| Rear | Rear (includes Supported Rear, Unsupported Rear) |
| Orienting | Orienting |
| Grooming | Grooming |

A transition is assigned to the **first matching group** based on its behaviors.

## Configuration Options

```yaml
# Short behavior bridging
short_behavior_threshold: 10  # Max frames for intermediate behavior bridging
max_gap_between_segments: 10  # Max gap between consecutive segments

# Rear merging
merge_rear: true  # Merge Supported/Unsupported Rear into Rear

# Analysis flags
run_transition_ranking: true  # Generate transition variant rankings
run_transition_comparison: true  # Generate comparison plots
```

## Output

### Transition Ranking Output
```
analysis_output/
└── transition_ranking/
    ├── transition_variants_ranking.csv
    ├── base_transition_counts.csv
    ├── transition_matrix.csv
    └── transition_matrix_with_turns.csv
```

### Per-Method Output
```
analysis_output/
└── {method}/
    └── seed_{N}/
        ├── {method}_seed{N}_transitions_intention_{BehaviorGroup}.pdf
        ├── {method}_seed{N}_transitions_qpos_{BehaviorGroup}.pdf
        └── {method}_seed{N}_variance_heatmap.pdf
```

## Module Structure

The analysis code is organized in `topo_mimic/benchmark/analysis/`:

```
analysis/
├── __init__.py           # Package exports
├── config.py             # Configuration dataclasses
├── behavior_parsing.py   # Behavior normalization, turn direction enum
├── ground_truth.py       # GT extraction with bridging and turns
├── transition_ranking.py # Variant counting and ranking
├── data_loading.py       # Data loading utilities
├── visualization.py      # Plotting functions
└── main.py              # Main analysis runner
```

## Examples

### Example 1: Short Behavior Bridging
```
Input segments (same clip):
  Rear: frames 0-100
  Walk: frames 101-150
  Orienting: frames 151-158 (8 frames, below threshold)
  Walk: frames 159-220

After bridging:
  Rear: frames 0-100
  Walk: frames 101-220 (merged)

Result: One Rear <-> Walk transition
```

### Example 2: Turn Direction Assignment
```
Segment: Rear, frames 50-150
Turn overlaps:
  - Right Turn: frames 80-120 (overlaps)
  - Left Turn: frames 200-250 (no overlap)

Transition point: frame 150 (end of Rear)
Result: Right Turn (closest to transition point)
```

### Example 3: Transition Variant
```
From: Rear (frames 50-150) with Right Turn overlay → R-Rear
To: Walk (frames 151-250) with no turn overlay → S-Walk

Variant: R-Rear → S-Walk
```

### Example 4: Full Transition Sequence
```
Ground truth labels:
  clip_642.mp4:
    - Supported Rear: 0-100, Right Turn: 50-100
    - Walking: 101-200
    - Orienting: 201-208 (8 frames)
    - Walking: 209-300, Left Turn: 250-300
    - Immobility: 301-400

After processing (merge_rear=True, short_threshold=10):
  - Rear (R): 0-100 → Walking (S): 101-300 (Orienting bridged)
  - Walking (L): 101-300 → Immobility (S): 301-400

Transitions found:
  1. R-Rear → S-Walking
  2. L-Walking → S-Immobility
```
