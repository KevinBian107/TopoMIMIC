# Automatic Label Assignment

Automatically assign behavior-transition labels to discovered clusters using
ground-truth behavior annotations from `behavior_labels/rodent.csv`. Labels
come from a simple **majority vote** over per-clip GT overlap counts; no
Wasserstein distance is computed.

## Overview

**Input per cluster**: a list of `(clip_id, frame_indices)` members.
**Input for GT**: a CSV with columns `Video, Behavior, Start_Frame, End_Frame`.

**Algorithm**:
1. **Per clip (cluster member)**:
   a. Count frames per behavior within the member's `frame_indices` using
      the clip's GT segments.
   b. If the **top behavior is Grooming**, label the member `Grooming`
      (singleton, no transition). This is the grooming exception.
   c. Else, if the **2nd-top behavior's fraction is below 20%** of the
      member's frames, label the member as the singleton top behavior.
   d. Else, label the member as the canonical transition
      `behav_a <-> behav_b`, where `(behav_a, behav_b)` is the pair of top
      two behaviors sorted alphabetically (so `"Immobility <-> Walking"`,
      never `"Walking <-> Immobility"`).
2. **Per cluster**: majority vote over all member labels. On a tie, all
   tied labels are returned as `alternatives`; the primary is the first
   tied label alphabetically (deterministic for I/O). Caller (GUI, or
   downstream analysis) should present the alternatives for manual review.

**Dominance threshold** (20%) and **grooming-override label** ("Grooming")
are configurable via `configs/benchmark/labeling.yaml` under the
`auto_label:` section.

## Configuration

Example `configs/benchmark/labeling.yaml`:

```yaml
experiment_dir: "outputs/benchmark_experiment_transitions"
behavior_labels_csv: "behavior_labels/rodent.csv"
method: "w2_dmp_int_kmeans"
seed: 0

auto_label:
  dominance_threshold: 0.2   # 2nd-top behavior must be >= this fraction
  grooming_label: "Grooming" # top == this => singleton label

save_csv: true
launch_gui: false
gui:
  port: 7860
  share: false
```

Legacy config sections (`distance:`, `gt_extraction:`) are accepted but
ignored. They were used by the old Wasserstein-distance pipeline which has
been removed.

## CLI

```bash
python3 -m topo_mimic.benchmark.labeling.main configs/benchmark/labeling.yaml
```

Set `launch_gui: true` in the config to open the Gradio GUI for manual
overrides after auto-labeling finishes.

## Outputs

All outputs land under `{experiment_dir}/{method}/`:

- `labels_seed_{seed}.yaml` — majority-vote primary labels per cluster, plus
  `alternatives` and `vote_counts` blocks for clusters with a tie.
- `sample_labels_seed_{seed}.yaml` — per-member detail: `clip_id`,
  `frame_start`, `frame_end`, `n_frames`, `label`, plus each cluster's
  `majority_label` and `alternatives`.
- `labeling/recommendations_seed_{seed}.csv` — adapter CSV for the GUI:
  `cluster_id, rank, label`, where `rank=0` is the primary and higher
  ranks are alternatives in alphabetical order.
- `labeling/cluster_votes_seed_{seed}.csv` — per-cluster voting summary:
  `cluster_id, primary, alternatives, n_members, vote_counts`.

## Manual overrides

The Gradio GUI reads `labels_seed_{seed}.yaml` on open and writes back to
the same file on save. Any manual edits override the auto-labels. To
re-run auto-labeling and overwrite manual edits, delete the yaml first
and rerun the CLI.

## Migration notes (2026-04-18)

The previous pipeline computed Wasserstein-2 distances between each
cluster's DM cycles and GT transition windows. That path has been removed
because all clips now have GT behavior annotations, making majority
voting both simpler and more direct. The `recommendations_seed_{seed}.csv`
schema kept the `cluster_id, rank, label` columns; the old `distance`
column is dropped.
