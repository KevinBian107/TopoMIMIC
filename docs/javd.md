# JAVD: Joint Angle Variance Difference

A benchmark metric for evaluating unsupervised behavioral transition discovery against ground truth annotations.

## Overview

JAVD measures how well unsupervised clustering methods capture the joint angle variance patterns present in ground truth behavioral transitions. Rather than using simple accuracy (which only provides a boolean match), JAVD provides a continuous measure of how similar the variance profiles are between discovered and ground truth transition clusters.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         JAVD Pipeline                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Discovered Clusters              Ground Truth Segments             │
│   (from unsupervised              (from behavior_labels.csv)        │
│    clustering)                                                       │
│         │                                   │                        │
│         ▼                                   ▼                        │
│   ┌───────────┐                      ┌───────────────┐              │
│   │  Cluster  │                      │  Find         │              │
│   │  Frames   │                      │  Consecutive  │              │
│   │           │                      │  Transitions  │              │
│   └─────┬─────┘                      └───────┬───────┘              │
│         │                                    │                       │
│         ▼                                    ▼                       │
│   ┌───────────┐                      ┌───────────────┐              │
│   │  Compute  │                      │  Compute      │              │
│   │  Variance │                      │  Variance     │              │
│   │  per Joint│                      │  per Joint    │              │
│   └─────┬─────┘                      └───────┬───────┘              │
│         │                                    │                       │
│         │         ┌─────────────┐            │                       │
│         └────────►│   JAVD =    │◄───────────┘                       │
│                   │ |Var_d - Var_gt|                                 │
│                   └─────────────┘                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Why JAVD?

Traditional accuracy metrics for behavioral clustering provide only boolean results: a cluster either matches a behavior or it doesn't. This loses important information about *how well* the cluster captures the behavioral dynamics.

| Metric | What It Measures | Limitation |
|--------|------------------|------------|
| Accuracy | Does cluster match GT label? | Binary - no nuance |
| NMI/ARI | Cluster assignment agreement | Global - misses joint-level detail |
| **JAVD** | Joint variance similarity | Continuous, interpretable, joint-specific |

JAVD answers: "For the joints that matter most in this transition, how similar is the motion variance between what we discovered and ground truth?"

## Key Design Decisions

### 1. Max Variance Joint Selection: GT-Based

**Decision**: Select top-K variance joints based on **ground truth** variance, not discovered cluster variance.

**Rationale**: Ground truth represents the "correct" answer. We want to evaluate whether discovered clusters capture variance in the joints that *should* have high variance during this transition, not joints that happen to have high variance in the (potentially incorrect) discovered cluster.

```
Example: "Walking <-> Immobility" transition

GT Variance (ranked):           Discovered Variance:
  1. lower_leg_R: 0.0892          lower_leg_R: 0.0654
  2. lower_leg_L: 0.0845          skull: 0.0923        ← High but irrelevant
  3. foot_R: 0.0712               lower_leg_L: 0.0701
  4. foot_L: 0.0698               foot_R: 0.0589
  5. upper_leg_R: 0.0534          torso: 0.0445

We evaluate on GT's top 5 joints (leg/foot movement),
not skull which happens to be noisy in discovered cluster.
```

### 2. Per-Cluster Top K (Not Global)

**Decision**: Each transition cluster has its own top-K joints from GT.

**Rationale**: Different transitions involve different body parts. A "Walking <-> Immobility" transition primarily involves legs, while "Grooming <-> Orienting" involves head and arms. Using a global top-K would bias toward transitions with universally high-variance joints.

```
Transition A: "Walking <-> Immobility"     Transition B: "Grooming <-> Orienting"
┌─────────────────────────────┐            ┌─────────────────────────────┐
│ Top 5 from GT:              │            │ Top 5 from GT:              │
│   • lower_leg_R             │            │   • skull                   │
│   • lower_leg_L             │            │   • jaw                     │
│   • foot_R                  │            │   • upper_arm_L             │
│   • foot_L                  │            │   • upper_arm_R             │
│   • upper_leg_R             │            │   • lower_arm_L             │
└─────────────────────────────┘            └─────────────────────────────┘

Different joints for different transitions → per-cluster top-K
```

### 3. Table Structure with Sparse Entries

**Decision**: Table has joints as rows, clusters as columns, with empty cells where a joint isn't in that cluster's top-K.

**Rationale**: This preserves the per-cluster nature of joint selection while allowing cross-cluster comparison where joints overlap.

```
                    │ Immo↔Walk │ Walk↔Rear │ Immo↔Orient │ Average
────────────────────┼───────────┼───────────┼─────────────┼─────────
lower_leg_R         │   0.0238  │   0.0156  │             │  0.0197
lower_leg_L         │   0.0201  │   0.0189  │             │  0.0195
foot_R              │   0.0123  │           │             │  0.0123
skull               │           │           │   0.0067    │  0.0067
upper_arm_L         │           │           │   0.0089    │  0.0089
────────────────────┼───────────┼───────────┼─────────────┼─────────
Average             │   0.0187  │   0.0173  │   0.0078    │  0.0146

Empty cells = joint not in top-K for that transition
```

### 4. Dual Averaging (Row + Column)

**Decision**: Compute both per-joint averages (across clusters) and per-cluster averages (across joints), plus overall mean.

**Rationale**:
- **Per-cluster average**: "How well does this discovered cluster match GT overall?"
- **Per-joint average**: "How well do we capture this joint's dynamics across all transitions?"
- **Overall average**: Single summary statistic for method comparison

### 5. Multiple Output Tables

**Decision**: Output four related tables, not just JAVD.

| Table | Formula | Purpose |
|-------|---------|---------|
| JAVD | \|Var_d - Var_gt\| | Absolute difference |
| JAVD Normalized | \|Var_d - Var_gt\| / Var_gt | Relative difference (%) |
| GT Variance | Var_gt | Reference values |
| Discovered Variance | Var_d | What we found |

**Rationale**: The absolute JAVD value is hard to interpret without context. A JAVD of 0.05 could be small (if GT variance is 0.5) or large (if GT variance is 0.001). The normalized table and raw variance tables provide this context.

```
Interpreting Normalized JAVD:

  0.0  = Perfect match (Var_d = Var_gt)
  0.25 = 25% different from GT
  1.0  = 100% different (Var_d is 0 or 2× Var_gt)
  2.0  = 200% different (Var_d is 3× Var_gt or negative impossible)
  >1.0 = Discovered variance is very different from GT
```

## Transition Definition

### What Constitutes a Transition?

A transition is defined as two consecutive behavior segments where different primitive behaviors meet.

```
Behavior Labels CSV structure:

  clip_643.mp4, Immobility, 0, 383      ←─┐
  clip_643.mp4, Walking, 383, 470         │ Transition: Immobility <-> Walking
  clip_643.mp4, Immobility, 471, 497    ←─┘ (and another one here)

Timeline:
  Frame:   0        383      470   497
           │─────────┼────────┼─────│
           │Immobility│Walking │Immob│
           │         ↑        ↑     │
           │    Transition   Transition
```

### Bidirectional Equivalence

**Decision**: Treat both directions as the same transition type.

```
Walking → Immobility  ═══════════════╗
                                     ╠══► Same: "Walking <-> Immobility"
Immobility → Walking  ═══════════════╝
```

**Rationale**: The behavioral dynamics at the boundary are similar regardless of direction. Both capture the moment when locomotion starts or stops.

### Segment Adjacency Requirement

**Decision**: Segments must be within `max_gap=10` frames to count as a transition.

```
Valid Transition (gap = 2):
  │──Immobility──│··│──Walking──│
  Frame:        383 385

Invalid (gap = 50):
  │──Immobility──│·················│──Walking──│
  Frame:        300               350
                    (50 frame gap - not a direct transition)
```

**Rationale**: Large gaps suggest something happened between behaviors (possibly unlabeled activity). We want clean transitions where one behavior directly follows another.

## Segment Balancing

### The Problem

Ground truth segments can have vastly different lengths, causing one behavior to dominate the variance calculation.

```
Unbalanced Example:
  Walking:    frames 100-150  (50 frames)
  Grooming:   frames 151-350  (200 frames)

  Total: 250 frames, but Grooming has 4× more influence on variance
```

### The Solution

**Decision**: Truncate the longer segment to match the shorter one's length, keeping frames adjacent to the transition boundary.

```
Before Balancing:
  ┌─────────────────────────────────────────────────────────────┐
  │  Walking (50 frames)  │        Grooming (200 frames)        │
  │  frames 100-150       │        frames 151-350               │
  └───────────────────────┴─────────────────────────────────────┘
                          ↑
                     Boundary (frame 150/151)

After Balancing:
  ┌───────────────────────┬───────────────────────┐
  │  Walking (50 frames)  │  Grooming (50 frames) │
  │  frames 100-150       │  frames 151-200       │
  └───────────────────────┴───────────────────────┘
                          ↑
                     Keep frames adjacent to boundary
```

**Key Principle**: Always keep the frames closest to the transition boundary, as these are most relevant to the transition dynamics.

```
Segment A (comes first):  Keep LAST N frames   (adjacent to boundary)
Segment B (comes second): Keep FIRST N frames  (adjacent to boundary)

Where N = min(length_A, length_B)
```

### Why Balance?

1. **Equal representation**: Both behaviors contribute equally to variance
2. **Transition focus**: Frames near the boundary capture actual transition dynamics
3. **Comparability**: Discovered clusters (from DM cycles) typically have balanced frame counts
4. **Fairness**: Prevents long segments from overwhelming short ones in variance calculation

## Sample Count Diagnostics

### The Problem

Variance estimates are unreliable with few samples. High normalized JAVD values (e.g., 19.0 or 45.0) often indicate sparse data rather than true mismatch.

### The Solution

**Decision**: Output sample counts alongside JAVD tables with warnings for low counts.

```
Sample Counts Table:
┌──────────────────────────────┬───────────┬───────────────────┬────────────┐
│ Transition                   │ GT Frames │ Discovered Frames │ Cluster ID │
├──────────────────────────────┼───────────┼───────────────────┼────────────┤
│ Immobility <-> Walking       │    847    │       234         │     2      │
│ Walking <-> Supported Rear   │     32    │        89         │     8      │ ← WARNING
│ Immobility <-> Orienting     │    156    │       412         │     3      │
└──────────────────────────────┴───────────┴───────────────────┴────────────┘

WARNING: Low sample count (<50 frames) for: ['Walking <-> Supported Rear']
```

**Guideline**: Trust JAVD values only when both GT and Discovered have ≥50-100 frames.

## Interpretation Guide

### Reading the JAVD Table

```
                    │ Immo↔Walk │ Walk↔Rear │ Average
────────────────────┼───────────┼───────────┼─────────
lower_leg_R         │   0.0238  │   0.0156  │  0.0197
foot_R              │   0.0123  │           │  0.0123
────────────────────┼───────────┼───────────┼─────────
Average             │   0.0181  │   0.0156  │  0.0168
```

**Reading by column**: "For Immo↔Walk transition, our discovered cluster has JAVD of 0.0238 for lower_leg_R and 0.0123 for foot_R, averaging 0.0181 across joints."

**Reading by row**: "For lower_leg_R joint, average JAVD across transitions is 0.0197."

### Reading Normalized JAVD

```
Normalized JAVD = |Var_discovered - Var_GT| / Var_GT

Value    Interpretation
─────────────────────────────────────────────────────
0.0      Perfect match
0.1      10% different - excellent
0.25     25% different - good
0.5      50% different - moderate mismatch
1.0      100% different - significant mismatch
>2.0     Very poor match - check sample counts
```

### Common Patterns

**Pattern 1: Uniformly low JAVD**
```
All values < 0.3 normalized
→ Method captures transition dynamics well
```

**Pattern 2: High JAVD for specific transitions**
```
Immo↔Walk: 0.15 (good)
Walk↔Rear: 2.34 (poor)
→ Check sample counts for Walk↔Rear
→ Or: this transition type is inherently harder to capture
```

**Pattern 3: High JAVD for specific joints**
```
lower_leg: 0.12 (good across transitions)
skull: 1.89 (poor across transitions)
→ Method may not capture head movements well
→ Or: skull variance is noisy/unreliable
```

## Visualization: GT Transition Rendering

To complement numerical metrics, GT transitions can be rendered as grid videos for visual inspection.

```
Video Grid Layout:
┌─────────────────────────────────────────────────────────┐
│              GT Transitions (w2_kmeans seed 0)          │
├─────────────┬─────────────┬─────────────┬───────────────┤
│ Sample 1    │ Sample 2    │ Sample 3    │ Sample 4      │
├─────────────┼─────────────┼─────────────┼───────────────┤
│ Immo↔Walk   │ Immo↔Walk   │ Immo↔Walk   │ Immo↔Walk     │  Row 1
│ (clip 643)  │ (clip 645)  │ (clip 712)  │ (clip 801)    │
├─────────────┼─────────────┼─────────────┼───────────────┤
│ Walk↔Rear   │ Walk↔Rear   │ Walk↔Rear   │               │  Row 2
│ (clip 647)  │ (clip 650)  │ (clip 689)  │               │
├─────────────┼─────────────┼─────────────┼───────────────┤
│ Immo↔Orient │ Immo↔Orient │             │               │  Row 3
│ (clip 648)  │ (clip 649)  │             │               │
└─────────────┴─────────────┴─────────────┴───────────────┘

Each tile plays the balanced transition segment as a video loop.
```

**Purpose**: Visual verification that GT transitions are correctly identified and balanced.

## Summary

| Design Choice | Decision | Rationale |
|---------------|----------|-----------|
| Max variance source | GT-based | Evaluate against "correct" joints |
| Top-K scope | Per-cluster | Different transitions → different joints |
| Averaging | Row + Column + Overall | Multiple perspectives on performance |
| Output tables | 4 tables | Context for interpretation |
| Transition direction | Bidirectional equivalence | Similar dynamics both ways |
| Segment adjacency | max_gap = 10 frames | Clean direct transitions only |
| Segment balancing | Truncate to min length | Equal representation, boundary focus |
| Sample counts | Diagnostic output | Identify unreliable estimates |
