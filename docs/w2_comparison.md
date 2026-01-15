# Wasserstein-2 & Gromov-Wasserstein

This document compares Gromov-Wasserstein (GW) and Wasserstein-2 (W2) distances for comparing behavioral cycles extracted from the Discrete Morse pipeline. It covers theoretical properties, practical limitations, and empirical findings from our behavioral analysis.

---

## Theoretical Background: The Speed Variation Problem

**The Original Motivation for GW**: Behaviors in Mimic-MJX rollouts can occur at different speeds. When comparing DM cycles using W2, the same behavior performed at different speeds could potentially be misclassified.

```
                     The Speed Variation Problem

    Clip A: Slow Rear (80 frames)        Clip B: Fast Rear (30 frames)

    torso                                torso
    height                               height
      │      ╭──╮                          │    ╭╮
      │     /    \                         │   /  \
      │    /      \                        │  /    \
      │   /        \                       │ /      \
      │  /          \                      │/        \
      └────────────────→ time              └──────────→ time

    SAME BEHAVIOR, DIFFERENT TEMPORAL DENSITY

Figure 1: The same "rear" behavior can span different numbers of frames
depending on how quickly the rodent performs it. After uniform downsampling,
frame-by-frame comparison (W2) sees these as different behaviors.
```

**Theoretical expectation**:
- **W2**: HIGH distance → Different clusters ❌
- **GW**: LOW distance → Same cluster ✅

**However**, this theoretical advantage is overshadowed by GW's inability to distinguish different behaviors with similar temporal dynamics (see Characteristics 5).

---

## Why This Matters for TopoVNL

### Current Pipeline Limitation

The Discrete Morse extraction successfully identifies topologically significant cycles. However, W2 comparison can lose this structural information.

```
                  Current W2 Pipeline Problem

    DM Cycle Extraction              W2 Distance Computation
    ───────────────────              ───────────────────────

    Clip 123:                        After uniform downsampling:
    Cycle frames: [10,15,20,25,...]
         │                                  Clip 123      Clip 456
    Uniform sample to 70 frames     Frame 35: rear-peak  walk-phase
         │                          Frame 36: rear-down   walk-phase
    Extract intention vectors                    ↑             ↑
         │                                       └─────────────┘
    Compare via W2                            POINT MISMATCH!

    The problem: W2 compares POINT-TO-POINT

    Frame 35 in Clip A  ←──compare──→  Frame 35 in Clip B

    But frame 35 in different clips may represent
    COMPLETELY DIFFERENT phases of the same behavior!

Figure 2: W2 compares frames at the same index position after downsampling.
If two clips perform the same behavior at different speeds, corresponding
frame indices will be at different phases of the behavior.
```

---

## Mathematical Difference: What Each Distance Measures

### Wasserstein-2 (W2): Point Positions

W2 answers: *"How much work to move mass from distribution X to Y?"*

Given trajectories $X = \{x_1, \ldots, x_n\}$ and $Y = \{y_1, \ldots, y_m\}$ with uniform weights:

$$W_2(X, Y) = \min_{\pi} \sum_{i,j} \pi_{ij} \cdot \|x_i - y_j\|^2$$

subject to marginal constraints on $\pi$.

**Key property**: Requires X and Y to be in the **same coordinate space**. Compares **where points are** directly.

```
                    Wasserstein-2 Distance

    Intention Space (2D projection)

         Cycle X               Cycle Y

           ●                     ○
          /│\                   /│\
         ● ● ●       vs.       ○ ○ ○
          \│/                   \│/
           ●                     ○

    W2 finds the optimal transport plan π:

           ●─────────────────────○   (match point 1)
          /│\                   /│\
         ●─┼─●───────────────○─┼─○   (match points 2,3,4)
          \│/                   \│/
           ●─────────────────────○   (match point 5)

    Cost = Σ πᵢⱼ · ‖xᵢ - yⱼ‖²

Figure 3: W2 finds an optimal matching between point sets that
minimizes total transport cost. The cost depends on direct
Euclidean distances between matched points.
```

### Gromov-Wasserstein (GW): Internal Relationships

GW answers: *"How similar are the internal structures of X and Y?"*

$$GW(X, Y) = \min_{\pi} \sum_{i,j,k,l} \pi_{ik} \cdot \pi_{jl} \cdot |d(x_i, x_j) - d(y_k, y_l)|^2$$

**Key property**: Compares **pairwise distance matrices**, not point positions. X and Y can be in **different spaces**.

```
                   Gromov-Wasserstein Distance

    Instead of comparing point positions directly,
    GW compares PAIRWISE DISTANCE MATRICES:

    Cycle X:          Internal distances (Cₓ):

        ●              │ x₁  x₂  x₃  x₄  x₅
       /│\          ───┼────────────────────
      ● ● ●         x₁ │  0   2   3   2   4
       \│/          x₂ │  2   0   1   2   2
        ●           x₃ │  3   1   0   1   3
                    x₄ │  2   2   1   0   2
                    x₅ │  4   2   3   2   0

    Cycle Y:          Internal distances (Cᵧ):

        ○              │ y₁  y₂  y₃  y₄  y₅
       /│\          ───┼────────────────────
      ○ ○ ○         y₁ │  0   2   3   2   4   ← SAME STRUCTURE!
       \│/          y₂ │  2   0   1   2   2
        ○           y₃ │  3   1   0   1   3
                    y₄ │  2   2   1   0   2
                    y₅ │  4   2   3   2   0

Figure 4: GW compares how points relate to each other within each
point cloud. If the internal distance matrices match, GW reports
low distance regardless of where the point clouds are located.
```

**GW Matching Criterion**: *"If I match $x_i \to y_k$ and $x_j \to y_l$, were $x_i$ and $x_j$ as far apart as $y_k$ and $y_l$?"*

---

## Characteristics 1: Different Behavioral Speeds

### The Problem with W2

When the same behavior occurs at different speeds, uniform downsampling creates misaligned comparisons.

```
                Characteristics: Same Behavior at Different Speeds

    Clip A (slow rear):               Clip B (fast rear):

    torso height                      torso height
         │      ●                          │    ●
         │     / \                         │   / \
         │    /   \     ← peak             │  /   \  ← same peak
         │   /     \                       │ /     \
         │  /       \                      │/       \
         └●──────────●→ time               └●────────●→ time
          80 frames                         30 frames

    After uniform downsampling to target_len = 70:

    Clip A:                           Clip B:
    Frame 35: Still rising            Frame 35: Already descending!

    W2 compares frame 35 to frame 35:
    "rising phase" vs "descending phase" = HIGH DISTANCE!

Figure 5: After uniform downsampling, frame indices no longer correspond
to the same phase of behavior. W2's point-by-point comparison sees
large differences even for identical behaviors at different speeds.
```

### How GW Solves It

GW matches based on structural relationships, not frame positions.

```
               GW Solution: Structure-Based Matching

    Clip A internal structure:       Clip B internal structure:

    "Peak frame is FAR from          "Peak frame is FAR from
     start and end frames"            start and end frames"

    start ●────────────● peak         start ○────────────○ peak
           \          /                      \          /
            \        /                        \        /
             \      /                          \      /
              ● end                             ○ end

    Both have the SAME structural pattern:
    • Peak is maximally distant from start and end
    • Start and end are close to each other
    • Triangular relationship preserved

    GW naturally matches:
    • Start → Start  (both far from peak)
    • Peak → Peak    (both far from endpoints)
    • End → End      (both far from peak)

    Result: LOW DISTANCE (same structure) ✅

Figure 6: GW recognizes that both clips have the same internal
structure: a peak state that is far from both start and end states.
This structural similarity is preserved regardless of temporal speed.
```

---

## Characteristics 2: Phase-Shifted Behaviors

### The Problem with W2

When the same cyclic behavior starts at different phases, W2 sees complete misalignment.

```
                 Characteristics: Phase-Shifted Behaviors

    Clip A: Walk starting at heel-strike

    leg angle    ╭─╮     ╭─╮     ╭─╮
         │      /   \   /   \   /   \
         │     /     \ /     \ /     \
         │    /       X       X       \
         │   /       / \     / \       \
         └──●───────────────────────────→ time
            ↑
         heel-strike (frame 0)

    Clip B: Walk starting at mid-swing

    leg angle        ╭─╮     ╭─╮     ╭─
         │          /   \   /   \   /
         │         /     \ /     \ /
         │        /       X       X
         │       /       / \     / \
         └──────●───────────────────────→ time
                ↑
             mid-swing (frame 0)

    W2 Comparison (frame-by-frame):
    Frame 0:  Clip A = heel-strike    Clip B = mid-swing
    Frame 25: Clip A = mid-swing      Clip B = heel-strike

    Every frame is MISALIGNED!
    W2 Result: HIGH distance ❌

Figure 7: The same gait cycle starting at different phases results
in complete misalignment when compared frame-by-frame. W2 reports
high distance for what is actually identical cyclic behavior.
```

### How GW Solves It

GW recognizes that both trajectories form the same loop structure.

```
               GW Solution: Cycle-Invariant Matching

    Intention space (2D projection of gait cycle):

    Clip A:                        Clip B:

         ●₀ ← start (heel-strike)        ○₂₅ ← start (mid-swing)
        /  \                            /  \
       /    \                          /    \
     ●₇₀    ●₂₅ ← mid-swing          ○₀    ○₅₀
       \    /                    heel→ \    /
        \  /                     strike \  /
         ●₅₀                             ○₇₀

    SAME LOOP SHAPE, different starting point!

    Both have:
    • Points arranged in a LOOP
    • Each point has TWO neighbors (before and after)
    • Opposite points are FAR apart in the cycle

    GW can match either way:
    Option 1: ●₀ → ○₂₅, ●₂₅ → ○₅₀, etc. (shifted match)
    Option 2: ●₀ → ○₀, ●₂₅ → ○₂₅, etc. (aligned match)

    Either way: LOW DISTANCE (same loop structure) ✅

Figure 8: GW recognizes that both point clouds form the same loop
shape in intention space. The starting point of the loop doesn't
affect the internal structure comparison.
```

---

## Characteristics 3: Scale and Intensity Variations

### The Problem with W2

Different animals may perform the same behavior with different amplitudes.

```
              Characteristics: Same Behavior at Different Intensities

    Clip A: Full rear (torso lifted high)

    intention[z]
         │    ●
         │   / \
         │  /   \     ← peak at 0.9
         │ /     \
         │/       \
         ●─────────●──→ time

    Clip B: Partial rear (torso only slightly lifted)

    intention[z]
         │
         │   ○
         │  / \      ← peak at 0.4 (same behavior, less intense)
         │ /   \
         ○─────────○──→ time

    W2 Comparison in intention space:

              ●  ← Clip A peak
             /|\
            / | \    BIG GAP between trajectories
           /  |  \
          /   |   \
         ○────┼────○  ← Clip B (smaller loop)
        /     |     \
       ●──────┼──────●

    W2 measures ABSOLUTE distances between points
    Large gap → HIGH distance → Different clusters ❌

Figure 9: W2 measures absolute positions in intention space. Different
amplitudes of the same behavior result in large point-wise distances,
even though the structural pattern is identical.
```

### How GW Solves It

GW compares normalized internal distances, achieving scale invariance.

```
                 GW Solution: Scale Invariance

    Clip A (large amplitude):       Clip B (small amplitude):

         ●  peak                          ○  peak
        / \                              / \
       /   \                            /   \
      /     \                          /     \
     ●       ●                        ○       ○
    start   end                      start   end

    Internal distances (normalized by max distance):

    Clip A:                          Clip B:
         start  peak   end                start  peak   end
    start [  0    1.0   0.2 ]        start [  0    1.0   0.2 ]
    peak  [ 1.0    0    1.0 ]        peak  [ 1.0    0    1.0 ]
    end   [ 0.2   1.0    0  ]        end   [ 0.2   1.0    0  ]

    IDENTICAL NORMALIZED STRUCTURE!

    Both behaviors have:
    • Peak is MAXIMALLY far from both start and end
    • Start and end are CLOSE to each other
    • Same triangular relationship pattern

    GW Result: LOW distance ✅

Figure 10: After normalizing internal distances, both behaviors
show identical structure. GW naturally achieves scale invariance
by comparing relative distances rather than absolute positions.
```

---

## Characteristics 4: When GW Fails - Directional Behaviors

### A Limitation of Pure GW

GW compares only internal structure, so it cannot distinguish behaviors that differ only in direction.

```
            GW Failure Case: Direction Matters

    Clip A: Walk forward

    position (x)
         │                    ●
         │               ●
         │          ●
         │     ●
         │●
         └────────────────────→ time
         0    1    2    3    4

    Clip B: Walk backward

    position (x)
         │○
         │     ○
         │          ○
         │               ○
         │                    ○
         └────────────────────→ time
         0    1    2    3    4

    Internal distance matrices are IDENTICAL:

    Clip A:                    Clip B:
         t₀   t₁   t₂   t₃   t₄        t₀   t₁   t₂   t₃   t₄
    t₀ [  0    1    2    3    4 ]  t₀ [  0    1    2    3    4 ]
    t₁ [  1    0    1    2    3 ]  t₁ [  1    0    1    2    3 ]
    t₂ [  2    1    0    1    2 ]  t₂ [  2    1    0    1    2 ]
    t₃ [  3    2    1    0    1 ]  t₃ [  3    2    1    0    1 ]
    t₄ [  4    3    2    1    0 ]  t₄ [  4    3    2    1    0 ]

    GW Result: ZERO distance (identical structure) ❌

    The problem: GW is BLIND to absolute direction/position

Figure 11: Linear trajectories in opposite directions have identical
internal distance matrices. GW cannot distinguish them because it
only compares relative point relationships, not absolute positions.
```

---

## Characteristics 5: GW Clusters by Temporal Dynamics, Not Behavioral Content

### The Critical Empirical Finding

In practice, GW clustering produces clusters that group behaviors by their **temporal pace/rhythm** rather than by **what behavior is actually happening**. This is a fundamental limitation when the goal is behavioral content clustering.

```
                The Core Problem: What GW Actually Captures

    GW compares INTERNAL DISTANCE MATRICES:

    Ci[j,k] = ||frame_j - frame_k||  (distance between frames j and k)

    For consecutive frames, this approximates VELOCITY:

    Ci[j, j+1] ≈ ||velocity at frame j|| × Δt

    Therefore, the structure matrix encodes the VELOCITY PROFILE
    (how fast the trajectory moves through intention space over time)

    ═══════════════════════════════════════════════════════════════

    GW clusters by: HOW the behavior unfolds (pace/rhythm)
    W2 clusters by: WHAT behavior is happening (semantic content)

    These are ORTHOGONAL dimensions of the data!

Figure 15: The structure matrix Ci captures pairwise distances between
frames, which encodes the velocity/pace profile of the trajectory rather
than its absolute position in intention space.
```

### Why This Happens: The Velocity Profile Interpretation

Consider two completely different behavioral sequences that happen to have the same temporal rhythm:

```
            Same Pace Pattern, Different Behaviors

    ═══════════════════════════════════════════════════════════════
    Sequence A: rear → walk → rear
    ═══════════════════════════════════════════════════════════════

    Intention Space Position:           Velocity Profile:

         │    REAR zone                      │    ╱╲
         │  ●────────●                       │   ╱  ╲
         │ ╱          ╲                      │  ╱    ╲
         │╱            ╲                     │ ╱      ╲
         ●──────────────●                    │╱        ╲
            WALK zone                        └──────────→ time
                                              slow fast slow

    ═══════════════════════════════════════════════════════════════
    Sequence B: groom → run → stand
    ═══════════════════════════════════════════════════════════════

    Intention Space Position:           Velocity Profile:

         │                                   │    ╱╲
         │      ○ GROOM                      │   ╱  ╲
         │     ╱                             │  ╱    ╲
         │    ╱                              │ ╱      ╲
         │   ○─────○ RUN→STAND               │╱        ╲
         │                                   └──────────→ time
                                              slow fast slow

    ═══════════════════════════════════════════════════════════════

    DIFFERENT positions in intention space (different behaviors)
    SAME velocity profile (slow-fast-slow rhythm)

    W2 sees: Different positions → DIFFERENT clusters ✓
    GW sees: Same pace pattern → SAME cluster ✗

Figure 16: Two behavioral sequences with identical temporal dynamics
(slow-fast-slow) but completely different behavioral content. GW groups
them together based on their shared rhythm, ignoring the semantic content.
```

### GW Clusters = Temporal Dynamics Clusters

This explains what GW clustering actually produces:

```
    ╔═══════════════════════════════════════════════════════════════╗
    ║         What GW Clusters Actually Represent                   ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║  GW Cluster 1: "slow → fast → slow" rhythm                    ║
    ║  ┌─────────────────────────────────────────────────────────┐  ║
    ║  │  • rear → walk → rear                                   │  ║
    ║  │  • rear → groom → walk                                  │  ║
    ║  │  • groom → run → groom                                  │  ║
    ║  │  • stand → jump → stand                                 │  ║
    ║  │  → Any sequence with "accelerate then decelerate"       │  ║
    ║  └─────────────────────────────────────────────────────────┘  ║
    ║                                                               ║
    ║  GW Cluster 2: "slow → slow → fast" rhythm                    ║
    ║  ┌─────────────────────────────────────────────────────────┐  ║
    ║  │  • rear → rear → walk                                   │  ║
    ║  │  • groom → groom → run                                  │  ║
    ║  │  • stand → stand → jump                                 │  ║
    ║  │  → Any sequence with "gradual buildup then burst"       │  ║
    ║  └─────────────────────────────────────────────────────────┘  ║
    ║                                                               ║
    ║  GW Cluster 3: "fast → slow → fast" rhythm                    ║
    ║  ┌─────────────────────────────────────────────────────────┐  ║
    ║  │  • walk → stand → walk                                  │  ║
    ║  │  • run → pause → run                                    │  ║
    ║  │  • jump → land → jump                                   │  ║
    ║  │  → Any sequence with "active-pause-active" pattern      │  ║
    ║  └─────────────────────────────────────────────────────────┘  ║
    ║                                                               ║
    ║  GW Cluster 4: "steady pace" rhythm                           ║
    ║  ┌─────────────────────────────────────────────────────────┐  ║
    ║  │  • continuous walking                                   │  ║
    ║  │  • continuous grooming                                  │  ║
    ║  │  • continuous rearing                                   │  ║
    ║  │  → Any sequence with constant velocity                  │  ║
    ║  └─────────────────────────────────────────────────────────┘  ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝

Figure 17: GW clusters group behavioral sequences by their temporal
dynamics pattern (pace/rhythm) rather than by semantic content. Each
cluster contains diverse behaviors that share a common velocity profile.
```

### The Two Orthogonal Dimensions

The data has two independent dimensions that can be clustered:

```
                Behavioral Content vs Temporal Dynamics

                         WHAT behavior
                         (semantic content)
                              ▲
                              │
                   rear       │        walk
                      ●───────┼────────●
                      │       │        │
                      │       │        │
        slow─fast─slow│       │        │fast─slow─fast
    ──────────────────┼───────┼────────┼──────────────────▶
                      │       │        │      HOW it unfolds
                      │       │        │    (temporal dynamics)
                      ●───────┼────────●
                   groom      │        run
                              │


    W2 clusters along the VERTICAL axis → Groups by behavioral content
    GW clusters along the HORIZONTAL axis → Groups by temporal dynamics

    These dimensions are ORTHOGONAL:
    • A behavior can be "rear" with "slow-fast-slow" dynamics
    • Or "rear" with "fast-slow-fast" dynamics
    • Or "walk" with "slow-fast-slow" dynamics
    • etc.

Figure 18: W2 and GW capture orthogonal aspects of the behavioral data.
W2 measures where the trajectory goes (semantic content), while GW
measures how fast it moves (temporal dynamics).
```

### Mathematical Explanation

The structure matrix encodes the velocity profile:

```
    Structure Matrix as Velocity Encoder

    For a trajectory X = [x₁, x₂, ..., xₙ] in intention space:

    Structure matrix: Ci[j,k] = ||xⱼ - xₖ||

    For consecutive frames:

        Ci[j, j+1] = ||xⱼ₊₁ - xⱼ|| ≈ ||v(t)|| × Δt

        where v(t) is the velocity in intention space

    ─────────────────────────────────────────────────────────────

    The PATTERN of values in Ci encodes:

    • Small consecutive distances → slow movement (low velocity)
    • Large consecutive distances → fast movement (high velocity)
    • Pattern of small/large → rhythm/pace signature

    ─────────────────────────────────────────────────────────────

    When GW compares two structure matrices:

    Cᵢ ≈ Cⱼ  ⟺  "Trajectories have similar velocity profiles"

    NOT: "Trajectories go to similar places"

    ─────────────────────────────────────────────────────────────

Figure 19: The structure matrix effectively encodes the velocity profile
of the trajectory. GW compares these velocity profiles, making it blind
to where the trajectory actually goes in intention space.
```

### Why the Intention Space Matters

The intention space in Mimic-MJX is **semantically meaningful** - nearby points represent similar behaviors:

```
              Intention Space Has Semantic Structure

    The intention encoder is trained so that:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   Position in intention space = Behavioral semantics        │
    │                                                             │
    │   • Points in "rear region" → rearing behavior              │
    │   • Points in "walk region" → walking behavior              │
    │   • Points in "groom region" → grooming behavior            │
    │                                                             │
    │   Distance in intention space ≈ Behavioral dissimilarity    │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    W2 PRESERVES this semantic structure:

        "Cycles in similar regions" → "Similar behaviors" ✓

    GW DISCARDS this semantic structure:

        "Cycles with similar internal distances" →
        "Similar pace, possibly different behaviors" ✗

    ─────────────────────────────────────────────────────────────

Figure 20: The intention space is designed to encode behavioral semantics.
W2 leverages this by comparing positions, while GW ignores it by only
comparing internal relationships.
```

### When Temporal Dynamics Clustering IS Useful

GW's focus on temporal dynamics is not always wrong - it depends on the goal:

```
    ╔═══════════════════════════════════════════════════════════════╗
    ║              When to Use Each Distance Metric                 ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║  Use W2 when asking:                                          ║
    ║  ┌─────────────────────────────────────────────────────────┐  ║
    ║  │  "What behavior is the rodent doing?"                   │  ║
    ║  │  "Which clips show rearing vs walking vs grooming?"     │  ║
    ║  │  "Group clips by behavioral content"                    │  ║
    ║  └─────────────────────────────────────────────────────────┘  ║
    ║                                                               ║
    ║  Use GW when asking:                                          ║
    ║  ┌─────────────────────────────────────────────────────────┐  ║
    ║  │  "What is the movement style/tempo?"                    │  ║
    ║  │  "Which clips show jerky vs smooth movements?"          │  ║
    ║  │  "Group clips by temporal dynamics pattern"             │  ║
    ║  │  "Find clips with similar acceleration profiles"        │  ║
    ║  └─────────────────────────────────────────────────────────┘  ║
    ║                                                               ║
    ║  Use FGW when asking:                                         ║
    ║  ┌─────────────────────────────────────────────────────────┐  ║
    ║  │  "Group by behavior, but allow some tempo variation"    │  ║
    ║  │  "Find similar behaviors performed at different speeds" │  ║
    ║  └─────────────────────────────────────────────────────────┘  ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝

Figure 21: The choice of distance metric should match the clustering
objective. W2 for behavioral content, GW for temporal dynamics, and
FGW for a balance of both.
```

### Summary: GW Captures Pace, Not Content

| Aspect | W2 | GW |
|--------|----|----|
| **Captures** | WHERE in intention space | HOW FAST through intention space |
| **Clusters by** | Behavioral content | Temporal dynamics (pace/rhythm) |
| **Semantic meaning** | Preserved | Lost |
| **Invariant to** | Translation | Translation, rotation, scaling, direction |
| **Good for** | "What behavior?" | "What movement style?" |

**Key insight**: GW is not failing - it's successfully capturing something different than intended. If the goal is behavioral content clustering, W2 is the appropriate choice.

---

## The Solution: Fused Gromov-Wasserstein (FGW)

### Combining the Best of Both Worlds

FGW balances structural matching (GW) with feature matching (W2):

$$FGW(X, Y) = \alpha \cdot GW(X, Y) + (1 - \alpha) \cdot W_2(X, Y)$$

where $\alpha \in [0, 1]$ controls the trade-off:
- $\alpha = 0$: Pure W2 (position only)
- $\alpha = 0.5$: Balanced (recommended)
- $\alpha = 1$: Pure GW (structure only)

```
                  Fused Gromov-Wasserstein (FGW)

    FGW = α × GW (structure) + (1-α) × W2 (features)

    α = 0.0          α = 0.5            α = 1.0
    ────────         ────────           ────────
    Pure W2          Balanced           Pure GW
                     (Recommended)

    ┌────────┐      ┌────────┐         ┌────────┐
    │Position│      │Position│         │Structure│
    │  only  │      │   +    │         │  only   │
    │        │      │Structure│         │         │
    └────────┘      └────────┘         └────────┘

    Sensitive       Best of            Ignores
    to speed/       both               direction
    phase           worlds

Figure 12: FGW interpolates between W2 (position-sensitive) and GW
(structure-sensitive). The parameter α controls the balance, with
α=0.5 providing a good default for behavioral data.
```

### How FGW Handles Each Characteristics

| Scenario | W2 | GW | FGW ($\alpha=0.5$) |
|----------|----|----|-------------------|
| Same behavior, same speed | ✅ Low | ✅ Low | ✅ Low |
| Same behavior, different speed | ❌ High | ✅ Low | ✅ Low (structure dominates) |
| Same behavior, phase-shifted | ❌ High | ✅ Low | ✅ Low (structure matches) |
| Same behavior, different intensity | ❌ High | ✅ Low | ⚠️ Medium (depends on scale) |
| Walk forward vs backward | ✅ High | ❌ Low | ✅ High (W2 catches direction) |
| Completely different behaviors | ✅ High | ✅ High | ✅ High |

---

## Visual Comparison: W2 vs GW vs FGW

### Complete Example with Rear Behavior

```
              Visual Example: Clustering "Rear" Behaviors

    Input: Four DM cycles from different clips

    Clip 1: Slow full rear     Clip 2: Fast full rear
         ●                          ○
        / \                        / \
       /   \                      /   \
      /     \                    /     \
     ●───────●                  ○───────○
     80 frames                  30 frames

    Clip 3: Slow partial rear  Clip 4: Walk (different behavior)
         △                          □───□───□───□
        / \
       /   \
      △─────△                   linear trajectory
     60 frames                  50 frames

Figure 13: Four test cases representing variations in speed (Clips 1-2),
intensity (Clip 3), and a completely different behavior (Clip 4).
```

**W2 Distance Matrix**:

|         | Clip 1 | Clip 2 | Clip 3 | Clip 4 |
|---------|--------|--------|--------|--------|
| Clip 1  | 0      | HIGH   | MEDIUM | HIGH   |
| Clip 2  | HIGH   | 0      | HIGH   | HIGH   |
| Clip 3  | MEDIUM | HIGH   | 0      | HIGH   |
| Clip 4  | HIGH   | HIGH   | HIGH   | 0      |

W2 Clustering: {Clip 1, Clip 3}, {Clip 2}, {Clip 4} ❌ (Same behavior split by speed)

**GW Distance Matrix**:

|         | Clip 1 | Clip 2 | Clip 3 | Clip 4 |
|---------|--------|--------|--------|--------|
| Clip 1  | 0      | LOW    | LOW    | HIGH   |
| Clip 2  | LOW    | 0      | LOW    | HIGH   |
| Clip 3  | LOW    | LOW    | 0      | HIGH   |
| Clip 4  | HIGH   | HIGH   | HIGH   | 0      |

GW Clustering: {Clip 1, Clip 2, Clip 3}, {Clip 4} ✅ (All rears together)

**FGW Distance Matrix** ($\alpha = 0.5$):

|         | Clip 1 | Clip 2 | Clip 3 | Clip 4 |
|---------|--------|--------|--------|--------|
| Clip 1  | 0      | LOW    | MEDIUM | HIGH   |
| Clip 2  | LOW    | 0      | LOW    | HIGH   |
| Clip 3  | MEDIUM | LOW    | 0      | HIGH   |
| Clip 4  | HIGH   | HIGH   | HIGH   | 0      |

FGW Clustering: {Clip 1, Clip 2}, {Clip 3}, {Clip 4} ✅ (Full rears together, partial rear separate)

---

## The Two Dimensions of Behavioral Data

```
                        Understanding the Data

    Behavioral trajectories have TWO independent aspects:

    1. CONTENT (WHAT behavior)     →  Captured by POSITION in intention space
                                   →  W2 measures this

    2. DYNAMICS (HOW it unfolds)   →  Captured by VELOCITY through intention space
                                   →  GW measures this

    ═══════════════════════════════════════════════════════════════════

    For the question "What behavior is the rodent doing?":
    → W2 is the correct choice (clusters by content)

    For the question "What is the movement tempo/style?":
    → GW is the correct choice (clusters by dynamics)

    These are DIFFERENT questions with DIFFERENT correct answers.

    ═══════════════════════════════════════════════════════════════════
```

---

*Document updated with empirical findings from TopoVNL behavioral clustering experiments.*
