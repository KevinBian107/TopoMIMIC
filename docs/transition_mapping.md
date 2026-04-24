# Transition Mapping Documentation

How cluster labels are assigned from ground truth and how those labels map
back to GT transitions for the analysis pipeline.

The labeler uses the directional `"A -> B Direction"` format. A legacy
canonical `"A <-> B"` labeler is still callable for the scale-sweep
discovery logic (see `topo_mimic/experiment/auto_label.py`). The
discovered-cluster side and the GT side use the same per-window frame-count
rule for direction assignment (`direction_tag_from_window` on the cluster
side, `compute_window_turn` on the GT side); the discovered side adds a
per-sample majority vote on top. See [Unified direction rule](#unified-direction-rule)
below.

## Label kinds

The rodent ground-truth CSV (`behavior_labels/rodent_labels.csv`) stores TWO
kinds of labels in the same file, distinguished by the `Label` column value:

### Behavior labels (mutually exclusive)
At most one behavior covers any given frame.

- `Immobile`
- `Turn`
- `Rear`
- `Walk`

### Direction labels (overlay)
Directional flag that may overlap behavior segments in time. At most one
direction covers any given frame.

- `Left`
- `Right`
- `Straight`

`auto_label.BEHAVIOR_LABELS` and `auto_label.DIRECTION_LABELS` are the frozen
sets used throughout the codebase; `split_segments_by_kind(segments)` splits
a mixed dict loaded from CSV into `(behavior_segments, direction_segments)`.

## Auto-label algorithm (directional, primary)

Implemented in `auto_label.auto_label_clusters_directional`. Runs in two
stages.

### Stage 1 — per-member pair

Each cluster member is a `(clip_id, frame_indices)` pair. For each member:

1. Restrict attention to the frames in `frame_indices`.
2. Count how many of those frames fall inside each BEHAVIOR segment of that
   clip (direction segments ignored).
3. Take the top-2 behaviors by frame count.
4. Order them by first appearance — the behavior whose earliest covered
   frame occurs first is `from`, the other is `to`.
5. If only one behavior covered any frames, `(from, to) = (single_behavior,
   None)`.
6. If no behavior covered any frames, the member is dropped from the
   cluster-level vote.

### Stage 2 — cluster label

1. Majority vote on the per-member `(from, to)` pairs. Ties are deterministic
   (sorted by `(from, to)` ascending); the losing tied pairs are exposed
   under `alternatives`.
2. Per-sample direction vote (see [unified rule](#unified-direction-rule-2026-04-23)):
   - Each member sample (one `(clip_id, frame_indices)` pair) gets ONE
     direction tag by `direction_tag_from_window` applied to its own
     window's direction-segment overlap.
   - Samples with zero direction overlap abstain from the vote.
3. Cluster direction = plain majority vote across the per-sample tags.
   Ties resolve `Left > Right > Straight`. When no sample voted, fall back
   to `Straight`.
4. Render the cluster label string via `format_directional_label`:
   - Transition: `"{from} -> {to} {direction}"` (e.g. `"Immobile -> Turn Left"`)
   - Single:     `"{behavior} {direction}"`     (e.g. `"Immobile Straight"`)

Direction is assigned per-sample then voted at the cluster level, not by
pooling frame counts across the whole cluster. This prevents one sample
whose window happens to bleed into an adjacent turning behavior from
forcing every `[Straight]` sample under it to inherit a spurious Left/Right
tag.

## Label format reference

| Case | Label example | `ParsedLabel.comparison_type` |
|---|---|---|
| Transition with direction | `Immobile -> Turn Left` | `transition` |
| Transition (no directional frames) | `Immobile -> Turn Straight` | `transition` |
| Single behavior with direction | `Immobile Left` | `intersection` |
| Single behavior, straight | `Immobile Straight` | `intersection` |
| Empty (no GT coverage) | `""` | skipped |

Downstream consumers parse the label via
`topo_mimic/analysis/feature_analysis/behavior_parsing.parse_cluster_label`, which
whitespace-splits and buckets tokens into `primitives`
({`Immobile`, `Turn`, `Rear`, `Walk`}) and `compositional` ({`Left`,
`Right`, `Straight`}). The `->` arrow token is not a behavior and is
silently dropped, so the parser sees the same token sequence for `"A -> B
Left"` and `"A B Left"`.

## Legacy canonical labeler

`auto_label.auto_label_clusters` is the older format and remains active for
`topo_mimic/analysis/scale_sweep/discovery.py`. It:

- Takes only behavior segments (direction segments must be filtered out
  upstream, which `scale_sweep/data.py` now does via
  `split_segments_by_kind`).
- Applies a dominance threshold (default 20%) before declaring a
  transition; otherwise emits a singleton.
- Has a grooming-singleton override (dead code for the current rodent CSV
  but harmless).
- Emits `"A <-> B"` with `A <= B` alphabetically, or `"A"` (singleton).

The scale-sweep discovery table continues to group canonical labels by
alphabetical base transition so both directions of a pair map to one row.

## Running the auto-labeler

**Full benchmark run** (fits methods + labels results):
```bash
python3 -m topo_mimic.experiment.transitions configs/experiment/transitions.yaml
```

**Relabel only** (reuses cached `results.pkl` per method, does NOT refit):
```bash
python3 -m topo_mimic.experiment.transitions configs/experiment/transitions.yaml --only-relabel
```
Use this when the labeling logic changes but the clusters themselves are
still valid. Requires a previous non-`--only-relabel` run to have written
`{output_dir}/{method}/results.pkl` with `cluster_dict` in it.

**Labeling GUI**:
```yaml
# In configs/labeling/auto_label.yaml
launch_gui: true
```
```bash
python3 -m topo_mimic.labeling.main configs/labeling/auto_label.yaml
```
The GUI reads `labels_seed_{seed}.yaml` from the method's output directory
(written by either the full benchmark run or the `--only-relabel` pass) and
lets the user override cluster labels manually.

## Downstream analysis grouping

`ParsedLabel.gt_key` groups clusters by direction-agnostic base transition:

```
"Immobile -> Turn Left"  -> ("transition", ("Immobile", "Turn"))
"Turn -> Immobile Right" -> ("transition", ("Immobile", "Turn"))   # same key
"Immobile Left"          -> ("intersection", ("Immobile", "Left"))
```

This matches the grouping used by the GT transition counter in
`transition_ranking.py`, which also sorts the (from, to) pair
alphabetically when forming the base transition. Direction is retained in
the label and in `ParsedLabel.from_turn` / `to_turn` for display.

## Configuration

`configs/labeling/auto_label.yaml` knobs that affect labeling:

```yaml
behavior_labels_csv: "behavior_labels/rodent_labels.csv"
method: "w2_dmp_qpos_kmeans"   # which method's clusters to (re-)label
seed: 0
auto_label:
  dominance_threshold: 0.2     # only used by the canonical labeler
  grooming_label: "Grooming"   # only used by the canonical labeler
launch_gui: true
gui:
  port: 7860
  share: false
```

## Unified direction rule

The discovered side and the GT side use the same direction-assignment rule
so that the DTW-results and consistency tables share most of their rows.

### The per-window rule (shared by both sides)

Given a frame window `W` and the clip's Left/Right/Straight direction
segments, count how many of `W`'s frames fall inside each direction segment.
Apply this priority:

```
if Left > 0 or Right > 0:
    tag = "Left" if Left >= Right else "Right"
elif Straight > 0:
    tag = "Straight"
else:
    tag = None   # window has no direction overlap, caller abstains
```

Ties (`Left == Right > 0`) resolve to `Left`.

This rule lives in **two** places but is semantically identical:

| Side       | Function                                            | Window                                                |
|------------|-----------------------------------------------------|-------------------------------------------------------|
| Discovered | `auto_label.direction_tag_from_window`              | The member sample's recorded `frame_indices`          |
| GT         | `analysis.ground_truth.compute_window_turn`         | `[transition_frame ± window_frames/2)`, default K=100 |

### Discovered-side adds a per-sample vote on top

Each cluster member sample gets ONE tag (or abstains) via the per-window
rule. The cluster's final direction is the plain majority across sample
tags:

```python
counts = Counter(per_sample_tags)           # over members that voted
if no sample voted:
    cluster_direction = "Straight"
else:
    cluster_direction = max(
        (counts["Left"],  "Left"),
        (counts["Right"], "Right"),
        (counts["Straight"], "Straight"),
        key=lambda x: x[0],
    )[1]     # ties: Left > Right > Straight
```

`aggregate_direction_counts` produces the `Counter`,
`choose_cluster_direction` picks the winner.

### GT-side stores ONE tag per transition instance

`find_all_transitions(csv_path, window_frames=100)` populates a single
`turn` attribute on each `TransitionSegment` via `compute_window_turn`.
The `from_turn` / `to_turn` fields are also populated (by the
"nearest turn to transition_frame" rule in `get_turn_direction_for_segment`)
and preserved for display, but they are not used for DTW matching.

### Matching at DTW time

`metrics.extract_sequences_for_gt_label` filters with plain equality
against the unified tag:

```python
if filter_by_turn and t.turn != turn_direction:
    continue
```

All three directions (`Left`, `Right`, `Straight`) filter symmetrically.

### Window size

Controlled by `find_all_transitions(..., window_frames=100)`. The default
matches `AnalysisConfig.c2st_window_frames = 100` and is the natural choice
because discovered cluster member windows are also ~100 frames wide
(centered on some latent-space event). Setting both windows to the same
size aligns the frame statistics the two sides see.

### Why some DTW rows can still go missing

Even under the unified rule, a discovered cluster can end up with a label
no GT instance emits — e.g. `Immobile -> Turn [straight]`. GT's ±50-frame
window around every `Immobile -> Turn` boundary almost always catches
`Turn`'s Left/Right flag, so the GT-side `turn` is rarely `Straight` for
that transition. The cluster's wider pool of samples may vote differently.
When this happens, DTW drops the row (no GT to score against) but
consistency keeps it. This is a data-fact mismatch, not a rule mismatch,
and is rare in practice (2/22 rows in the current rodent benchmark).

## MMD / KID analysis

A kernel two-sample test that directly compares the distribution of
discovered cluster trajectories against real GT trajectories of the same
directional label. Complements DTW (which is time-warp-invariant but
distance-based per-pair) and C2ST (which trains a classifier and is
vulnerable to temporal-compression asymmetry).

### Definition

For each `(directed label, turn)` group of discovered clusters, gather the
cluster window set `X ⊂ ℝ^{n × (K·D)}` and the real-GT window set
`Y ⊂ ℝ^{m × (K·D)}`. Both sides use the same preprocessing as DTW (strip
first 7 root dims → 67 joint angles) and the same resample target
(K = 100 frames). Windows are flattened to `(K·D,)`-dim vectors, so the
kernel operates directly on the full trajectory.

Two kernels are reported in parallel:

- **KID (polynomial)** — `k(x, y) = ((x·y)/d + 1)^3`, `d = K·D`. This is
  the Bińkowski 2018 KID kernel, no bandwidth to tune.
- **MMD² (RBF)** — `k(x, y) = exp(-||x-y||² / (2σ²))`, with `σ` from the
  median heuristic on the pooled sample `X ∪ Y`.

Both estimators are the **unbiased MMD²** form (diagonals excluded from
`K_xx` and `K_yy`). Zero means distributions match in the kernel's RKHS;
the unbiased estimator can go slightly negative in finite samples when
distributions truly coincide (estimator variance, not a bug).

### Files

- `topo_mimic/analysis/feature_analysis/mmd_kid.py` — `compute_mmd_kid(X, Y)`,
  `polynomial_kernel`, `rbf_kernel`, `median_heuristic_sigma`,
  `mmd2_unbiased`.
- `topo_mimic/analysis/feature_analysis/metrics.py::run_mmd_analysis` —
  per-`(method, seed)` entrypoint mirroring `run_dtw_analysis`.
- Enabled via `AnalysisConfig.run_mmd_analysis: bool = True`.
- Per-run output: `outputs/benchmark_analysis/{method}/seed_{N}/mmd/{method}_seed{N}_mmd_results.csv`
  with columns `Label, Turn_Direction, Cluster_IDs, KID, MMD_RBF, Sigma,
  N_Cluster_Seqs, N_GT_Seqs`.
- Aggregation: `generate_results_table` loads those CSVs, pivots to
  `(transition-with-direction) × method`, computes mean ± std across seeds
  with 1-seed annotations, and writes `kid_table.{csv,png,pdf}` and
  `mmd_rbf_table.{csv,png,pdf}` (plus matching `*_std.csv`). Rendered with
  `higher_is_better=False` so the best-per-row highlight is the lowest
  distributional distance.

### How to read the numbers

Low KID / MMD² = cluster windows and real windows trace out similar
regions of flattened-trajectory space. The KID estimator on 6700-dim
vectors with n, m ≈ 20–50 has noticeable variance (so the std bars
matter). The companion RBF MMD² is rescaled by `σ`-median so absolute
magnitudes between the two kernels are not directly comparable — use each
within its own table.

Because the windows are flattened and not time-warped, both KID and MMD²
remain sensitive to temporal-compression asymmetry (cluster native span vs.
GT native span, both resampled to K = 100). DTW is the complement: it
absorbs temporal rescaling by design. Agreement between low DTW and low
KID/MMD² is strong evidence that cluster and real truly coincide.

## Cluster purity analysis

An independent metric that measures how faithfully a cluster's members
actually match its auto-assigned label. Lives in
`topo_mimic/analysis/feature_analysis/purity.py`.

### Definition

For a cluster with label `"X -> Y Z"` (or singleton `"X Z"`):

1. Parse label → `behavior_set = {X, Y}` (or `{X}`), `direction = Z`.
2. For each member sample:
   - For each frame in the sample's window, look up the concurrent GT
     behavior and GT direction.
   - Frame "belongs" iff (behavior ∈ `behavior_set`) AND (direction ==
     `direction`). Frames with no direction overlap count as `Straight`
     (matches the auto-labeler's abstain-on-zero fallback).
   - `sample_purity = n_matching / n_total_frames`, in `[0, 1]`.
3. `cluster_purity = mean(sample_purity across members)`.

Also reported: per-cluster std, min, max.

### Why it's useful

Directly quantifies what the unified rule is trying to fix. A cluster
labeled `"Rear -> Immobile Left"` with purity 0.05 tells you the label is
essentially noise — only 5% of the cluster's frames actually belong to the
claimed transition direction. A purity of 0.85 says the cluster lands
almost entirely inside the right GT region.

### Files

- `topo_mimic/analysis/feature_analysis/purity.py` — `run_purity_analysis(...)`,
  `compute_cluster_purity(...)`, `_parse_cluster_label(...)`.
- Enabled via `AnalysisConfig.run_purity_analysis: bool = True`.
- Per-run output: `outputs/benchmark_analysis/{method}/seed_{N}/purity/{method}_seed{N}_purity.csv`.
- Aggregation: `generate_results_table._load_individual_purity_results` →
  `create_purity_pivot_table` → `purity_table.{csv,png,pdf}` with mean ± std
  across seeds and 1-seed annotations, rendered with `higher_is_better=True`.

## Files

```
topo_mimic/experiment/
├── auto_label.py                  # BEHAVIOR_LABELS, DIRECTION_LABELS,
│                                  # direction_tag_from_window (shared rule),
│                                  # auto_label_clusters_directional (new),
│                                  # auto_label_clusters (legacy canonical)
├── labeling/
│   ├── core.py                    # ClusterLabel -> LabelRecommendation adapter
│   ├── main.py                    # CLI + GUI entry for relabeling a single method
│   └── gui.py                     # Gradio overrides
└── analysis/
    ├── behavior_parsing.py        # parse_cluster_label -> ParsedLabel, gt_key
    ├── ground_truth.py            # compute_window_turn, find_all_transitions,
    │                              # TransitionSegment.turn (unified tag)
    ├── metrics.py                 # DTW + MMD/KID entrypoints, uses t.turn
    ├── mmd_kid.py                 # KID / MMD² kernel math (new)
    ├── purity.py                  # cluster-purity analysis
    ├── transition_ranking.py      # GT-side transition counts (direction-agnostic)
    └── visualization.py           # Transition-comparison plots
```
