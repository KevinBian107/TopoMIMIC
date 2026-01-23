# DTW (Dynamic Time Warping) Metric

## Overview

DTW measures how similar two movement sequences are by finding the optimal temporal alignment between them. Unlike frame-by-frame comparison, DTW handles sequences of different lengths and accounts for variations in movement speed. It is a **temporal-spatial** alignment verification.

## Mathematical Formulation

Given two sequences:
- Sequence A: `[a₁, a₂, ..., aₙ]` where each `aᵢ ∈ ℝ⁶⁷` (joint angles)
- Sequence B: `[b₁, b₂, ..., bₘ]` where each `bⱼ ∈ ℝ⁶⁷`

### Step 1: Cost Matrix

Compute pairwise Euclidean distances:

```
cost(i, j) = ‖aᵢ - bⱼ‖₂
```

### Step 2: Dynamic Programming

Build the cumulative distance matrix D:

```
D(0, 0) = 0
D(i, j) = cost(i, j) + min(
    D(i-1, j),      # insertion (stretch A)
    D(i, j-1),      # deletion (stretch B)
    D(i-1, j-1)     # match (advance both)
)
```

### Step 3: Normalization

```
DTW_distance = D(n, m) / (n + m)
```

Dividing by `(n + m)` normalizes for sequence length, making distances comparable across transitions of different durations.

## Intuition in Behavioral Context

### What DTW Captures

DTW compares the **temporal dynamics** of movements, not just static poses:

```
Discovered Sequence:    ●───●───●───●───●
                            ╲   ╲
GT Sequence:            ●───●───●───●───●───●
                        ↑
                        Optimal alignment found
```

If a rodent performs Walking→Immobility faster in one instance than another, DTW aligns the "slowing down" phases together rather than comparing misaligned frames.

### Why This Matters for Transition Clusters

Behavioral transitions have characteristic movement patterns:
- **Walking→Immobility**: Gradual deceleration, weight shifting
- **Immobility→Rear**: Weight transfer to hindlimbs, forelimb lift
- **Walking with Left Turn**: Asymmetric leg movements, body rotation

DTW asks: "Does this discovered cluster's movement pattern match how the GT transition actually unfolds?"

### Interpreting Distance Values

The DTW distance is in units of **radians** (average joint angle difference per frame):

| DTW Value | Interpretation |
|-----------|----------------|
| 0.5 - 0.8 | Strong match - movement dynamics closely follow GT |
| 0.8 - 1.1 | Moderate match - similar overall pattern with some deviation |
| 1.1 - 1.5 | Weak match - recognizable but with significant differences |
| > 1.5 | Poor match - fundamentally different movement patterns |

### Turn Direction Filtering

When a discovered cluster has a turn direction (left/right):
- **Left**: Compare only against GT transitions with left turns
- **Right**: Compare only against GT transitions with right turns
- **Non-deterministic/Straight**: Compare against all GT transitions of that type

This ensures fair comparison - a "Walking→Immobility [left]" cluster is compared against GT transitions that also involve left turns.

## Example

```
Discovered: Walking→Immobility [left], Clusters C0, C3
    ↓
Extract sequences from C0 and C3 (combined)
    ↓
Find GT Walking↔Immobility transitions with left turns
    ↓
Compute pairwise DTW between discovered and GT sequences
    ↓
Report: mean DTW = 0.78 ± 0.12
    ↓
Interpretation: Good match - discovered cluster captures
                the left-turning walk-to-stop pattern well
```

## Comparison Table

The final comparison table shows DTW across methods:

```
Label                    Turn    method_A    method_B
Immobility↔Walking       left    0.7882      0.9856
Immobility↔Walking       right   0.9519      N/A
```

- **Lower values**: Method better captures that transition type
- **N/A**: Method did not discover that transition type
- **Method averages**: Overall performance across all discovered transitions
