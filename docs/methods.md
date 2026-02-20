# TopoVNL Methods: A Procedural Guide

This document explains how the main methods work procedurally in TopoVNL. The key insight is that **traversal extraction via Discrete Morse theory** produces topologically meaningful cycles, which are then compared using **Wasserstein distance** for clustering.

---

## Overview: Two-Stage Pipeline

```
                     TopoVNL Two-Stage Pipeline

   STAGE 1: Traversal Extraction (DM)
   ══════════════════════════════════

   ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
   │   Load     │───▶│   Takens   │───▶│     DM     │───▶│  Extract   │
   │   Data     │    │  Embedding │    │   Graph    │    │  Cycles    │
   └────────────┘    └────────────┘    └────────────┘    └────────────┘

                              │
                              ▼
                     ┌────────────────┐
                     │  DM Cycles     │
                     │  (per clip)    │
                     └────────┬───────┘
                              │
                              ▼
   STAGE 2: Distance-Based Clustering
   ═══════════════════════════════════

              ┌────────────────────┐
              │   W2/GW Distance   │
              │   (Optimal Transp) │
              └────────────────────┘
                        │
                        ▼
              ┌────────────────────┐
              │   MDS + KMeans     │
              │   Clustering       │
              └────────────────────┘

Figure 1: The two-stage pipeline. Stage 1 extracts topologically
meaningful cycles via Discrete Morse theory. Stage 2 clusters
cycles using W2 or GW distance with MDS embedding.
```

---

## Stage 1: Traversal Extraction

This stage is **identical** for all downstream methods. The goal is to extract meaningful cycle (traversal) structures from each clip's intention trajectory.

### Step 1.1: Data Loading

**Input**: Mimic-MJX rollout files in HDF5 format.

Each clip contains:
- `/intention`: Latent behavioral encoding with shape $(T, 16)$ where $T \approx 499$ frames
- `/qpos`: Joint positions with shape $(T, 69)$
- `/ctrl`: Control signals
- `/obs`: Observations

**Output**: Stacked intention array with shape $(N_{\text{clips}}, T, D)$ where $D = 16$.

```
                           Data Loading

    ┌───────────────────────────────────────────────────────────────────┐
    │                    rodent_ar.h5                                   │
    │   ┌─────────────────────────────────────────────────────────────┐ │
    │   │  /intention    (842, 499, 16)   ← Latent embedding          │ │
    │   │  /qpos         (842, 500, 69)   ← Joint positions           │ │
    │   │  /ctrl         (842, 499, 69)   ← Control signals           │ │
    │   │  /obs          (842, 499, ...)  ← Observations              │ │
    │   └─────────────────────────────────────────────────────────────┘ │
    └───────────────────────────────────────────────────────────────────┘

Figure 2: HDF5 structure of consolidated rollout data. The intention
array is the primary input for topological analysis, while qpos is
used for evaluation and visualization.
```

### Step 1.2: Takens Delay Embedding

The Takens embedding reconstructs attractor dynamics by stacking time-delayed copies of the signal. This is based on Takens' theorem, which guarantees that for generic smooth dynamical systems, the delay embedding preserves the topology of the original attractor.

**Parameters**:
- $m$ = Embedding dimension (number of delays), typically 3
- $\tau$ = Time delay in frames, typically 6

These are found via:
- **Average Mutual Information (AMI)** for $\tau$: Find first local minimum
- **False Nearest Neighbors (FNN)** for $m$: Find where FNN ratio drops below threshold

**Transform**: For each time $t \geq (m-1)\tau$:

$$\tilde{X}_t = \begin{bmatrix} X_t \\ X_{t-\tau} \\ X_{t-2\tau} \\ \vdots \\ X_{t-(m-1)\tau} \end{bmatrix}$$

This maps $(T, D) \to (T', m \cdot D)$ where $T' = T - (m-1)\tau$.

```
                      Takens Delay Embedding

    Original Signal (16-dim intention at each frame):

    Time:    t₀    t₁    t₂    ...   t₆    ...   t₁₂   ...
    Value:   X₀    X₁    X₂    ...   X₆    ...   X₁₂   ...

                               │
                               │  Takens transform with m=3, τ=6
                               ▼

    Embedded Signal (48-dim vector at each valid frame):

    For t = 12:  X̃₁₂ = [ X₁₂ | X₆ | X₀ ]
                        └─┬─┘ └─┬─┘ └─┬─┘
                        (16)  (16)  (16)  = 48 dimensions

Figure 3: Takens delay embedding concatenates the current frame with
time-delayed copies. With m=3 and τ=6, each embedded point captures
the state at t, t-6, and t-12, creating a 48-dimensional representation.
```

**Why Takens embedding?**
1. Reconstructs attractor geometry from low-dimensional observations
2. Makes the point cloud "thicker" for better topological analysis
3. Captures local dynamics and temporal structure

### Step 1.3: Discrete Morse Graph Extraction

This is the core TDA step that extracts the skeleton structure from the embedded point cloud.

#### Substep A: Sparse Weighted Rips Filtration

Given point cloud $\tilde{X} \in \mathbb{R}^{N \times 48}$:

1. **Compute pairwise distances** between all points using Euclidean metric

2. **Compute vertex weights** (inverse local density):
   $$w(v) = \sqrt{\frac{1}{k} \sum_{i=1}^{k} d(v, \text{NBR}_i)^2}$$
   where $\text{NBR}_i$ is the $i$-th nearest neighbor. Low density regions get high weights.

3. **Build sparse Rips filtration** with sparsity parameter $\epsilon$ (typically 0.99)

```
                 Sparse Weighted Rips Filtration

    Point Cloud                    Neighborhood Graph

         ●                              ●
        /│\                            /│\
       / │ \                          / │ \
      ●  ●  ●                        ●──●──●
       \ │ /                          \ │ /
        \│/                            \│/
         ●                              ●

    Each vertex v gets weight w(v) based on local density.
    Low density (sparse) → High weight
    High density (dense) → Low weight

Figure 4: The sparse weighted Rips filtration builds a simplicial
complex from the point cloud. Vertex weights reflect inverse density,
so sparse regions (high weight) appear later in the filtration.
```

#### Substep B: Persistence Computation

The persistence algorithm tracks topological features (connected components, loops, voids) as the filtration parameter increases.

For each feature, we record:
- **Birth time**: When the feature first appears
- **Death time**: When the feature merges with an older feature
- **Persistence**: $\text{death} - \text{birth}$ (lifespan)

```
                      Persistence Diagram

    death
      ▲
      │    ●
      │        ●  ← Long-lived features (topologically significant)
      │  ●         ●
      │    ●  ●       ●
      │  ●   ●●  ●     ● ← Short-lived features (noise)
      └──────────────────▶ birth

    Points far from the diagonal have high persistence
    and represent genuine topological features.

Figure 5: Persistence diagram showing birth-death pairs. Features
with high persistence (far from diagonal) are topologically significant,
while short-lived features near the diagonal are typically noise.
```

#### Substep C: Graph Reconstruction

Apply persistence threshold $\delta$ to filter edges:

**Keep edge $e$ if**: $\text{persistence}(e) > \delta$

This simplifies the complex filtration to a skeleton graph containing only the most persistent (topologically significant) edges.

```
                      Graph Reconstruction

    Before thresholding:              After thresholding:

         ●─●─●─●─●                         ●───●───●
        /│\│/│\│\│                           \   /
       ● │ ● │ ● ●         ───▶               ●
        \│/│\│/│/│                           / \
         ●─●─●─●─●                         ●───●

    Complex simplicial complex        Simplified skeleton graph

Figure 6: The persistence threshold δ filters out low-persistence
edges, leaving only the topologically significant skeleton that
captures the essential structure of the behavioral manifold.
```

### Step 1.4: Cycle Extraction

Extract cycles from the DM skeleton graph $G = (V, E)$.

**Method**: Minimum Cycle Basis
1. Build minimum spanning tree $T$ of $G$
2. For each non-tree edge $(u, v)$ (back edge):
   - Find shortest path $P(u, v)$ in $T$
   - Cycle = $P(u, v) \cup \{(u, v)\}$
3. Apply `minimum_cycle_basis()` for independent cycles

```
                        Cycle Extraction

    Skeleton Graph G          MST T              Cycles

         ●                     ●                 Cycle 1:
        /│\                   / \                ●───●
       / │ \                 /   \                 \ /
      ●──●──●     ───▶     ●─────●     ───▶        ●
       \ │ /                 \ │
        \│/                   \│                 Cycle 2:
         ●                     ●                 (if exists)

    Non-tree edges (back edges) close cycles.
    Each back edge defines a fundamental cycle.

Figure 7: Cycle extraction via MST. After building the minimum
spanning tree, each non-tree edge (back edge) creates a cycle
when combined with the unique tree path between its endpoints.
```

**Vertex-to-Frame Mapping**: DM vertex indices must be mapped back to original frame indices. Vertices are re-ordered by weight during the DM algorithm, so this reverse mapping recovers the temporal information.

**Output per clip**:
```
all_cycle_dm[clip_id] = [
    cycle_0: array of frame indices,
    cycle_1: array of frame indices,
    ...
]
```

### End of Stage 1: What We Have

For each of $N$ clips, we now have a list of cycles, where each cycle is an array of frame indices representing topologically significant traversals in the intention manifold.

**Key insight**: The cycles are NOT individual behaviors. They are **"behavioral highways"** - the repeated traversal patterns that any intention trajectory must cross multiple times. Think of them as the main routes connecting different behavioral states.

---

## Stage 2: Feature Comparison (Two Approaches)

After extracting cycles, we need to compare them across clips for clustering. There are two main approaches.

### Option A: W2 Distance Directly (Primary Method)

**Core Idea**: Compare cycle windows as **distributions**, not vectors.

#### Step A1: Window Preparation

For each cycle:
1. **Optional padding**: Expand frame indices by $\pm \text{pad}$ frames
2. **Uniform downsample** to `target_len` (e.g., 100 frames) for fair comparison
3. **Extract intention vectors** for these frames

Result: Each cycle becomes a point cloud of shape $(\text{target\_len}, 16)$ in intention space.

#### Step A2: W2 Distance Computation

The Wasserstein-2 (Earth Mover's) distance measures the minimum cost to transport one distribution to another.

Given two point sets $X = \{x_1, \ldots, x_n\}$ and $Y = \{y_1, \ldots, y_m\}$ with uniform weights:

$$W_2(X, Y) = \min_{\pi} \sum_{i,j} \pi_{ij} \cdot \|x_i - y_j\|^2$$

subject to:
- $\sum_j \pi_{ij} = \frac{1}{n}$ (mass leaving $x_i$)
- $\sum_i \pi_{ij} = \frac{1}{m}$ (mass arriving at $y_j$)

where $\pi$ is the optimal transport plan.

```
                     W2 Distance Computation

    Cycle X (in intention space):    Cycle Y (in intention space):

           ●                                ○
          /│\                              /│\
         ● ● ●                            ○ ○ ○
          \│/                              \│/
           ●                                ○

    W2 finds optimal matching:

           ●─────────────────────○   (match point 1)
          /│\                   /│\
         ●─┼─●─────────────────○─┼─○  (match points 2,3,4)
          \│/                   \│/
           ●─────────────────────○   (match point 5)

    Cost = Σ (transport distance × mass moved)

Figure 8: W2 distance finds the optimal transport plan π that
minimizes total cost of moving mass from X to Y. The cost matrix
uses pairwise Euclidean distances in intention space.
```

**Why W2 for behavior comparison?**
- Handles distributions with different supports
- Captures geometric differences in intention space
- Robust to small temporal shifts within cycles
- Naturally handles multi-dimensional intention vectors

#### Step A3: Distance Matrix Construction

Compute pairwise W2 distances between all $M$ cycle windows:

$$D_{ij} = W_2(\text{window}_i, \text{window}_j)$$

Result: Symmetric $M \times M$ distance matrix $D$.

#### Step A4: MDS Embedding + KMeans

1. **Normalize** distance matrix:
   $$\hat{D}_{ij} = \frac{D_{ij}}{\sqrt{\text{median}(D_{i,:}) \cdot \text{median}(D_{:,j})}}$$

2. **Multi-Dimensional Scaling (MDS)**: Find embedding $X \in \mathbb{R}^{M \times k}$ such that:
   $$\|x_i - x_j\| \approx \hat{D}_{ij}$$

3. **KMeans** on embedded points:
   $$\text{labels} = \text{KMeans}(n\_clusters=K).\text{fit\_predict}(X)$$

```
                    MDS Embedding + KMeans

    Distance Matrix D              MDS Embedding (3D)

    ┌─────────────┐                     ●
    │ 0  3  5  2  │                    / \
    │ 3  0  4  6  │      ───▶        ●───●
    │ 5  4  0  3  │                    \ /
    │ 2  6  3  0  │                     ●
    └─────────────┘
                                    KMeans clusters
    Precomputed W2 distances        in embedded space

Figure 9: MDS projects the W2 distance matrix into a low-dimensional
Euclidean space where distances are approximately preserved. KMeans
then clusters in this embedded space.
```

### Baseline: Uniform Sampling (W2 Unif)

As a baseline, we also provide `w2_unif_kmeans` which uses the same W2 + MDS + KMeans pipeline but on **uniformly sampled windows** from all frames instead of topologically-discovered cycles. This helps evaluate whether the DM-based cycle extraction improves clustering quality.

---

## Complete Pipeline Summary

```
                        Complete Pipeline

    INPUT: Mimic-MJX rollouts (N_clips × T × D_intention)
                               │
                               ▼
    STEP 1: TAKENS DELAY EMBEDDING
    Parameters: m=3, τ=6 (found via AMI + FNN)
    Transform: X_t → [X_t, X_{t-τ}, X_{t-2τ}]
    Output: (N_clips × T' × 48) embedded trajectories
                               │
                               ▼
    STEP 2: DISCRETE MORSE GRAPH EXTRACTION (per clip)
    2a. Build sparse weighted Rips filtration (k=3, ε=0.99)
    2b. Compute persistence via PHAT
    2c. Graph reconstruction with threshold δ=1.0
    Output: Skeleton graph G per clip
                               │
                               ▼
    STEP 3: CYCLE EXTRACTION
    Find minimum cycle basis
    Map vertex IDs → frame indices
    Optionally keep only longest cycle (only_max=True)
    Output: all_cycle_dm[clip] = [cycle_frames_1, ...]
                               │
                               ▼
    STEP 4: W2/GW DISTANCE CLUSTERING
    1. Uniform sample cycles to target_len
    2. Compute W2 or GW distance matrix
    3. Normalize distance matrix
    4. MDS embedding to 3D
    5. KMeans clustering
                               │
                               ▼
    STEP 5: CLUSTER ASSIGNMENT
    cluster_dict = {
        0: [(clip_5, frames), (clip_12, frames), ...],
        1: [(clip_3, frames), (clip_7, frames), ...],
        ...
    }
                               │
                               ▼
    STEP 6: ANALYSIS & VISUALIZATION
    • Render cluster videos (MuJoCo)
    • MDS subspace projections
    • Transition comparisons
    • labels.yaml for manual annotation

Figure 10: Complete pipeline from raw data to behavioral clusters.
DM cycles are compared via W2 or GW distance for clustering.
```

---

## Key Parameters Reference

| Stage | Parameter | Typical Value | Description |
|-------|-----------|---------------|-------------|
| Takens | $m$ | 3 | Embedding dimension |
| Takens | $\tau$ | 6 | Time delay (frames) |
| DM | `k_neighbors` | 3 | kNN for density weights |
| DM | $\epsilon$ | 0.99 | Rips sparsification |
| DM | $\delta$ | 1.0 | Persistence threshold |
| Clustering | `target_len` | 100 | Uniform sampling length |
| Clustering | `n_clusters` | 10 | Number of clusters |
| Clustering | `only_max` | True | Keep only longest cycle |
| MDS | `n_components` | 3 | MDS embedding dimension |

---

## Interpretation of Results

**What do the clusters represent?**

Cycles are NOT individual behaviors. They are **"behavioral highways"** - the repeated traversal patterns that any intention trajectory must cross multiple times.

```
                    Behavioral Highway Interpretation

                       ●───●
                      /     \
              REAR   ●───●───●   WALK
                      \     /
                       ●───●
                         │
                       GROOM

    Cycles capture the TRANSITIONS between behavioral states,
    not the states themselves. The skeleton represents the
    "roads" that connect different behavioral attractors.

Figure 11: Cycles represent transition paths between behavioral
states, not the states themselves. Clustering groups clips with
similar transition patterns.
```

**Cluster labels** (manually assigned after visualization):
- "walk_turn": Cycles dominated by locomotion + direction changes
- "rear_walk": Cycles transitioning between rear and walk
- "groom": Cycles with stationary grooming patterns
- "walk_turn_stay": Mixed locomotion with pauses

**Analysis outputs**:
1. **MDS projections**: How clusters relate in subspace
2. **Transition comparisons**: Discovered vs ground truth transitions
3. **Transition graph**: W2-based connectivity between clusters

---

## Running the Methods

**Step 1: Extract DM Cycles**
```bash
python -m topo_mimic.dmwk.dm_pipeline configs/dmwk/dm_pipeline.yaml
```
Output: `outputs/dm_pipeline_ar/{all_cycle_dm.pkl, all_G_sm.pkl}`

**Step 2: Run W2/GW + KMeans Clustering**
```bash
python -m topo_mimic.benchmark.experiment configs/benchmark/experiment.yaml
```
Enable methods in config:
- `w2_dm_kmeans`: W2 distance on DM cycles
- `gw_dm_kmeans`: Gromov-Wasserstein on DM cycles
- `w2_unif_kmeans`: W2 on uniform samples (baseline)

**Step 3: Analyze Results**
```bash
python -m topo_mimic.benchmark.analysis.main configs/benchmark/analysis.yaml
```
Output: Transition comparisons, MDS plots in `outputs/analysis_output/`

---

## Granularity Analysis

The granularity analysis module examines how persistence threshold affects behavior discrimination and cluster redundancy.

### Purpose

Different persistence thresholds capture different levels of behavioral detail:
- **Low δ (e.g., 0.1)**: Captures fine-grained movements (orienting, grooming)
- **High δ (e.g., 1.0)**: Captures only gross movements (locomotion)

The granularity analysis quantifies this relationship through:
1. **Stationary clip analysis**: Joint variance within DM cycles
2. **Redundancy analysis**: How many clusters become redundant at different k values

### Pipeline Overview

```
                  Granularity Analysis Pipeline

    ┌────────────────────────────────────────────────────────┐
    │  Multi-Persistence DM Extraction                       │
    │  δ ∈ {0.1, 0.25, 0.5, 0.75, 1.0}                       │
    └────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │ Stationary Clip  │            │   Redundancy     │
    │    Analysis      │            │    Analysis      │
    │                  │            │                  │
    │ • Head joints    │            │ • W2 clustering  │
    │ • Arm joints     │            │ • DTW distances  │
    │ • Variance in    │            │ • Union-Find     │
    │   cycles         │            │   merging        │
    └──────────────────┘            └──────────────────┘
              │                               │
              ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │ Joint Variance   │            │ Unique Clusters  │
    │   Boxplots       │            │     Curves       │
    └──────────────────┘            └──────────────────┘

Figure 12: Granularity analysis pipeline. Multi-persistence DM extraction
feeds into two analysis branches: stationary clip analysis and redundancy
analysis.
```

### Stationary Clip Analysis

For clips with low root displacement (stationary behavior):
1. Extract DM cycles at each persistence threshold
2. Compute joint variance within cycles for head and arm joints
3. Plot boxplots showing how variance changes with persistence

**Expected result**: Lower persistence captures more head movement variance (orienting behavior).

### Redundancy Analysis

For each (pers_delta, n_clusters) configuration:
1. Build windows from DM cycles
2. Compute W2 distance matrix and cluster with MDS + KMeans
3. Compute pairwise DTW between cluster representatives
4. Identify redundant pairs using threshold or percentile
5. Count unique clusters after merging redundant pairs

**Expected result**: Higher n_clusters leads to a plateau in unique clusters as redundancy increases.

### Running Granularity Analysis

```bash
python -m topo_mimic.granularity.main configs/granularity/granularity.yaml
```

Output directory structure:
```
out/granularity/
├── dm_cache/
│   └── pers_0.10/dm_result.pkl
├── stationary/
│   ├── stationary_clips.csv
│   └── plots/
├── redundancy/
│   ├── pers_0.10/n_clusters_10/
│   ├── plots/
│   └── summary.csv
└── summary.yaml
```

---

*Document generated for TopoVNL methods documentation.*
