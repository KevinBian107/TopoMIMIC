# `topo_mimic.analysis`

Two analysis tracks run on top of the benchmark experiment output:

- **`feature_analysis/`** — evaluates the benchmark methods across the three feature spaces (intention / qpos / ekp) using DTW, MMD/KID, C2ST, purity, transition-graph, and MDS projection metrics.
- **`method_analysis/`** — characterizes properties of the DM primitives themselves (cycle asymmetry, θ-graph transitions, stationary-clip separation, topological incompressibility, pooled global manifold, multi-persistence granularity).

Plus two orthogonal sub-trees:

- **`scale_sweep/`** — baseline-fairness hyperparameter sweep for KPMS and Mapper (Appendix).
- **`tables/`** — unified paper-table generator.

## `feature_analysis/`

Eleven atomic blocks; each is runnable as `python -m topo_mimic.analysis.feature_analysis.blocks.<block> <config>` and writes to its own subdirectory under `outputs/analysis/feature_analysis/`.

| Block                  | What it does                                                                | Output subfolder           |
|------------------------|-----------------------------------------------------------------------------|----------------------------|
| `gt_sanity`            | DTW distinctiveness of top-K GT transitions — baseline for analysis         | `gt_sanity_check/`         |
| `transition_ranking`   | Count and rank canonical GT transition variants                             | `transition_ranking/`      |
| `mds`                  | MDS subspace projection of each cluster                                     | `<method>/<seed>/mds/`     |
| `transition_comparison`| Per-cluster 2D/3D scatter overlays                                          | `<method>/<seed>/transitions/`    |
| `transition_graph`     | kNN graph of transition clusters                                             | `<method>/<seed>/transition_graph/` |
| `dtw_metrics`          | Sequence-level DTW comparison + internal consistency                         | `<method>/<seed>/{dtw_metrics,consistency}/` |
| `dtw_visualization`    | Per-cluster DTW cost matrices + aligned filmstrip videos                    | `<method>/<seed>/dtw_visualization/` |
| `mmd`                  | MMD / KID kernel distances between cluster and matching GT                  | `<method>/<seed>/mmd/`     |
| `c2st`                 | Classifier-two-sample-test cluster separability                             | `<method>/<seed>/c2st/`    |
| `purity`               | Per-sample GT-overlap purity for each cluster                               | `<method>/<seed>/purity/`  |
| `classification`       | Per-cluster precision/recall/F1 (activates iff `sample_labels_seed_*.yaml` exists) | `classification/`    |

Orchestrator (runs all of the above sequentially, same as before the split):

```
python -m topo_mimic.analysis.feature_analysis.main_transitions \
       configs/analysis/feature_analysis/analysis_transitions.yaml
```

All eleven blocks currently share one YAML config; each block turns on only its own `run_*` flag inside the shared config. Per-block configs can be introduced later if desired (see `_block_runner.py`).

### Coverage--quality boxplot panels

`coverage_quality_panels.py` (formerly `method_scatter.py`) compares
method families on a two-axis (coverage, quality) plane. Reads the
aggregate CSVs that `dtw_metrics` writes
(`dtw_comparison_table.csv`, `consistency_comparison_table.csv`) and
the per-(method, seed) MMD CSVs under `<method>/<seed>/mmd/`, and
renders an `n_families x 2` small-multiples grid: rows = DM-WKC /
Mapper / KP-MoSeq, columns = negated internal consistency / negated
MMD, y-scale shared per column. One box per coverage tick within each
panel pools per-(transition, seed) cells. Both axes negated so
up-and-right is the ideal regime: tighter clusters, broader coverage.

```
python -m topo_mimic.analysis.feature_analysis.coverage_quality_panels
```

Output: `outputs/analysis/feature_analysis/coverage_quality_panels/coverage_quality_panels.{png,pdf}` plus a long-form `coverage_quality_panels_long.csv` of every plotted dot.

## `method_analysis/`

Method-property demos that study **why** DM primitives recover behavior transitions. These were originally `demos/property/` and a handful of loose scripts; they now have their own configs and output folders.

| Subpackage            | Focus                                                                                                              |
|-----------------------|--------------------------------------------------------------------------------------------------------------------|
| `property/cycle_asymmetry`       | Cycle origin from behavioral asymmetry (Fig 2 intuition)                                               |
| `property/transition_topology`   | θ-graph vs single-path transition styles, auto-discovery of rear-walk-rear                               |
| `property/theta_filmstrip`       | Paper-ready θ-graph filmstrip with endpoint cosine-similarity heatmap                                    |
| `property/stationary`            | DM-G vs stationary clips (Appendix: cycle-bearing vs simple-topology clips)                              |
| `property/rear_cycles`           | Cross-clip cycle validation grid (4 MP4 + 2 PNG)                                                         |
| `property/cycle_behavior_breakdown` | Categorize no-cycle vs many-cycle clips by behavior signature                                         |
| `incompressibility/`             | Figure 3: DM retains only the minimal frames needed to preserve topology. `run.py`, `batch.py`, `varying_k.py` |
| `dm_dim_reduction/`              | Empirical justification for the PCA/Takens preprocessing                                                 |
| `cycle_persistence/`             | Persistence-threshold sensitivity of cycle stats                                                         |
| `transition_stats/`              | GT transition frequency analysis                                                                         |
| `cycle_rendering/`               | Side-by-side DM-cycle rendering videos                                                                    |
| `global_manifold/`               | Pool clips by GT transition type, run DM on the pooled cloud                                            |
| `granularity/`                   | Multi-persistence DM extraction + stationary joint variance + redundancy DTW (3-stage pipeline)         |

Each sub-analysis has a corresponding config under `configs/analysis/method_analysis/`. Outputs land under `outputs/analysis/method_analysis/`.

## `scale_sweep/`

Baseline-fairness sweep (Appendix). Fits KPMS across five κ values and Mapper across 45 variants; reports per-variant discovery of the canonical transition repertoire. Single config: `configs/analysis/scale_sweep/scale_sweep.yaml`. Output: `outputs/analysis/scale_sweep/v1/cfg_<hash>/`.

## `tables/`

Three table families under one package with a unified CLI:

```
python -m topo_mimic.analysis.tables {all | paper | warping | results}
```

- `results_table` — main Nature-style summary tables.
- `paper_tables` — Figure 4 summary + Appendix DTW + Appendix consistency.
- `warping_tables` — DTW warping-heaviness LaTeX tables.

Outputs land under `outputs/analysis/tables/`.
