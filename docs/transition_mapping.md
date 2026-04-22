# Transition Mapping Documentation

How cluster labels are assigned from ground truth and how those labels map
back to GT transitions for the analysis pipeline.

**Rewrite date: 2026-04-21.** Replaces the legacy `"A <-> B"` canonical format with the new
directional `"A -> B Direction"` format. The legacy canonical labeler is still
callable for the scale-sweep discovery logic (see
`topo_mimic/benchmark/auto_label.py`).

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
2. Aggregate direction frame counts across ALL members of the cluster
   (`Left`, `Right`, `Straight` counts summed).
3. Pick a single direction per cluster:
   - If either `Left` or `Right` has ≥1 frame, pick the one with more
     frames; ties go to `Left`.
   - Otherwise fall back to `Straight`.
4. Render the cluster label string via `format_directional_label`:
   - Transition: `"{from} -> {to} {direction}"` (e.g. `"Immobile -> Turn Left"`)
   - Single:     `"{behavior} {direction}"`     (e.g. `"Immobile Straight"`)

Direction is chosen at the cluster level, not per-member. The user-facing
rule is: *if the cluster spent any time turning, the label reflects that
turn direction; Straight is a fallback for clusters that never overlapped
Left or Right frames.*

## Label format reference

| Case | Label example | `ParsedLabel.comparison_type` |
|---|---|---|
| Transition with direction | `Immobile -> Turn Left` | `transition` |
| Transition (no directional frames) | `Immobile -> Turn Straight` | `transition` |
| Single behavior with direction | `Immobile Left` | `intersection` |
| Single behavior, straight | `Immobile Straight` | `intersection` |
| Empty (no GT coverage) | `""` | skipped |

Downstream consumers parse the label via
`topo_mimic/benchmark/analysis/behavior_parsing.parse_cluster_label`, which
whitespace-splits and buckets tokens into `primitives`
({`Immobile`, `Turn`, `Rear`, `Walk`}) and `compositional` ({`Left`,
`Right`, `Straight`}). The `->` arrow token is not a behavior and is
silently dropped, so the parser sees the same token sequence for `"A -> B
Left"` and `"A B Left"`.

## Legacy canonical labeler

`auto_label.auto_label_clusters` is the older format and remains active for
`topo_mimic/benchmark/scale_sweep/discovery.py`. It:

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
python3 -m topo_mimic.benchmark.experiment_transitions configs/benchmark/experiment_transitions.yaml
```

**Relabel only** (reuses cached `results.pkl` per method, does NOT refit):
```bash
python3 -m topo_mimic.benchmark.experiment_transitions configs/benchmark/experiment_transitions.yaml --only-relabel
```
Use this when the labeling logic changes but the clusters themselves are
still valid. Requires a previous non-`--only-relabel` run to have written
`{output_dir}/{method}/results.pkl` with `cluster_dict` in it.

**Labeling GUI**:
```yaml
# In configs/benchmark/labeling.yaml
launch_gui: true
```
```bash
python3 -m topo_mimic.benchmark.labeling.main configs/benchmark/labeling.yaml
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

`configs/benchmark/labeling.yaml` knobs that affect labeling:

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

## Files

```
topo_mimic/benchmark/
├── auto_label.py                  # BEHAVIOR_LABELS, DIRECTION_LABELS,
│                                  # auto_label_clusters_directional (new),
│                                  # auto_label_clusters (legacy canonical)
├── labeling/
│   ├── core.py                    # ClusterLabel -> LabelRecommendation adapter
│   ├── main.py                    # CLI + GUI entry for relabeling a single method
│   └── gui.py                     # Gradio overrides
└── analysis/
    ├── behavior_parsing.py        # parse_cluster_label -> ParsedLabel, gt_key
    ├── transition_ranking.py      # GT-side transition counts (direction-agnostic)
    └── visualization.py           # Transition-comparison plots
```
