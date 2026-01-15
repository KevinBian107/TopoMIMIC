# Transition Mapping Documentation

This document explains how cluster labels are mapped to ground truth data for the transition cluster comparison analysis.

## Behavior Types

### Primitive Behaviors
These are mutually exclusive base behaviors - a frame can only have ONE primitive behavior at a time (in terms of the animal's primary action):
- Walking
- Immobility
- Orienting
- Grooming
- Supported Rear
- Unsupported Rear

### Compositional Behaviors
These are overlays that can co-occur with primitive behaviors:
- Left Turn
- Right Turn

## Label Parsing Rules

The analysis parses cluster labels by splitting on whitespace and categorizing each word as primitive or compositional.

### Case 1: Transition (2+ Primitives)
**Example labels:** `"Walking Immobility"`, `"Walking Supported_Rear Left_Turn"`

When a label contains **2 or more primitive behaviors**, it's treated as a **transition**.

| Label | Primitives | Compositional | Ground Truth Lookup |
|-------|------------|---------------|---------------------|
| `Walking Immobility` | [Walking, Immobility] | [] | Frames near Walking <-> Immobility transitions |
| `Walking Supported_Rear Left_Turn` | [Walking, Supported Rear] | [Left Turn] | Frames near Walking <-> Supported Rear transitions |

**Note:** When there are 2+ primitives plus compositional behaviors, only the first two primitives are used for transition lookup. The compositional behavior indicates the context but doesn't change the ground truth search.

### Case 2: Intersection (1 Primitive + 1+ Compositional)
**Example labels:** `"Walking Left_Turn"`, `"Orienting Right_Turn"`

When a label contains **exactly 1 primitive** and **1+ compositional behaviors**, it's treated as an **intersection**.

| Label | Primitives | Compositional | Ground Truth Lookup |
|-------|------------|---------------|---------------------|
| `Walking Left_Turn` | [Walking] | [Left Turn] | Frames where BOTH Walking AND Left Turn are labeled |
| `Orienting Right_Turn` | [Orienting] | [Right Turn] | Frames where BOTH Orienting AND Right Turn are labeled |

### Case 3: Single Behavior (Skipped)
**Example labels:** `"Walking"`, `"Left_Turn"`

Labels with only a single behavior (primitive or compositional) are **skipped** in the comparison analysis since there's no transition or intersection to compare.

## Cluster Grouping

Multiple clusters with the **same ground truth key** are grouped together:

- Clusters labeled `"Walking Immobility"` and `"Immobility Walking"` → grouped (same transition)
- Clusters labeled `"Walking Left_Turn"` and `"Left_Turn Walking"` → grouped (same intersection)
- Clusters labeled `"Walking Left_Turn"` and `"Walking Right_Turn"` → NOT grouped (different compositional)

## Examples

### Example 1: Transition Cluster
```
Cluster Label: "Walking Supported_Rear Left_Turn"
Parsed:
  - primitives: [Walking, Supported Rear]
  - compositional: [Left Turn]
  - comparison_type: transition

Ground Truth: Find all frames near transitions between Walking and Supported Rear
(The Left Turn context is noted but doesn't affect GT lookup)
```

### Example 2: Intersection Cluster
```
Cluster Label: "Walking Left_Turn"
Parsed:
  - primitives: [Walking]
  - compositional: [Left Turn]
  - comparison_type: intersection

Ground Truth: Find all frames where BOTH Walking AND Left Turn labels are present
```

### Example 3: Multiple Same-Type Clusters
```
Cluster 0: "Walking Immobility"
Cluster 3: "Immobility Walking"

Both parse to: transition(Walking, Immobility)
→ Grouped together for comparison
→ Combined frames from clusters [0, 3] vs Ground Truth Walking <-> Immobility transitions
```

## Behavior Groups

Transitions are organized into groups for visualization based on the behaviors involved:

| Group | Behaviors Included |
|-------|-------------------|
| Immobility | Immobility |
| Walking | Walking |
| Rear | Supported Rear, Unsupported Rear |
| Orienting | Orienting |
| Grooming | Grooming |

A transition is assigned to the **first matching group** based on its primitive behaviors. For example:
- `Walking <-> Immobility` → Immobility group (Immobility checked first)
- `Walking <-> Supported Rear` → Walking group
- `Supported Rear <-> Unsupported Rear` → Rear group

## Output

Transitions are grouped by behavior type, with each group generating figures for **both feature types**:
- **Intention features:** Latent space representations from the policy network
- **Qpos features:** Joint position data from MuJoCo

Each figure has:
- **Left panel:** All discovered cluster frames for that behavior group
- **Right panel:** All corresponding ground truth frames

PCA is computed **once on all frames per feature type**, ensuring consistent projections within each feature space.

Files are named:
- `{method}_seed{N}_transitions_{feature}_{BehaviorGroup}.pdf`

Where `{feature}` is either `intention` or `qpos`.

Example outputs:
- `gw_kmeans_seed0_transitions_intention_Immobility.pdf` - Immobility transitions in intention space
- `gw_kmeans_seed0_transitions_qpos_Immobility.pdf` - Immobility transitions in qpos space
- `gw_kmeans_seed0_transitions_intention_Walking.pdf` - Walking transitions in intention space
- `gw_kmeans_seed0_transitions_qpos_Walking.pdf` - Walking transitions in qpos space
