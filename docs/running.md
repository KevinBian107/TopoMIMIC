# Running the pipeline

The repository is organized into six stages. Run them in order with `bash bash_scripts/run_all.sh`, or call any stage individually.

## Quick start

```bash
bash bash_scripts/run_all.sh
```

Individual stages:

```bash
bash bash_scripts/run_extraction.sh        # DM cycles + paths, all 3 feature spaces
bash bash_scripts/run_experiment.sh        # benchmark clustering (all methods)
bash bash_scripts/run_feature_analysis.sh  # 11 atomic feature-analysis blocks
bash bash_scripts/run_method_analysis.sh   # property + incompressibility + global_manifold + granularity
bash bash_scripts/run_scale_sweep.sh       # KPMS + Mapper baseline fairness sweep
bash bash_scripts/run_tables.sh            # paper tables (results + Figure 4 + Appendix)
```

Override the Python binary globally via `PYTHON=/path/to/python3 bash bash_scripts/run_all.sh`.

## Stage 1: DM extraction

Runs discrete Morse on every clip across the three feature spaces (intention 16D, qpos 62D, ekp 69D). Each feature space has tuned Takens and persistence parameters.

```bash
python -m topo_mimic.extraction.dm_pipeline configs/extraction/dm_pipeline_intention.yaml
python -m topo_mimic.extraction.dm_pipeline configs/extraction/dm_pipeline_qpos.yaml
python -m topo_mimic.extraction.dm_pipeline configs/extraction/dm_pipeline_ekp.yaml
```

With `use_dm_path: true` (default) this extracts both DM cycles (H1) and DM paths (H0 bridge-corridors: maximal non-branching chains between branch points). Outputs land under `outputs/extraction/dm_pipeline_<space>/`.

## Stage 2: Benchmark experiment

```bash
python -m topo_mimic.experiment.transitions configs/experiment/transitions.yaml
```

Runs every enabled method (w2_dmc_*, w2_dmp_*, gw_dmc_int, w2_full, kpms_*, custom_vae, mapper_*). Writes per-method per-seed results under `outputs/experiment/transitions/`.

## Stage 3: Feature-space analysis (atomic)

Eleven independently-runnable blocks sharing one config. Each block writes to its own subdirectory under `outputs/analysis/feature_analysis/`.

```bash
# everything (same as before):
python -m topo_mimic.analysis.feature_analysis.main_transitions configs/analysis/feature_analysis/analysis_transitions.yaml

# or one block at a time:
python -m topo_mimic.analysis.feature_analysis.blocks.gt_sanity <cfg>
python -m topo_mimic.analysis.feature_analysis.blocks.transition_ranking <cfg>
python -m topo_mimic.analysis.feature_analysis.blocks.mds <cfg>
python -m topo_mimic.analysis.feature_analysis.blocks.transition_comparison <cfg>
python -m topo_mimic.analysis.feature_analysis.blocks.transition_graph <cfg>
python -m topo_mimic.analysis.feature_analysis.blocks.dtw_metrics <cfg>
python -m topo_mimic.analysis.feature_analysis.blocks.dtw_visualization <cfg>
python -m topo_mimic.analysis.feature_analysis.blocks.mmd <cfg>
python -m topo_mimic.analysis.feature_analysis.blocks.c2st <cfg>
python -m topo_mimic.analysis.feature_analysis.blocks.purity <cfg>
python -m topo_mimic.analysis.feature_analysis.blocks.classification <cfg>
```

After `dtw_metrics` and `mmd` have produced their aggregates, render the
coverage-vs-quality panels:

```bash
python -m topo_mimic.analysis.feature_analysis.coverage_quality_panels
```

This produces `outputs/analysis/feature_analysis/coverage_quality_panels/coverage_quality_panels.{png,pdf}`: a small-multiples grid with one row per method family and two columns (negated internal consistency, negated MMD), each box pooling per-(transition, seed) cells at a fixed coverage. Up-and-right is the ideal regime.

See [topo_mimic/analysis/README.md](../topo_mimic/analysis/README.md) for a per-block description.

## Stage 4: Method analysis

Method-property studies that characterize why the DM primitives work (cycle asymmetry, θ-graph transitions, stationary-clip discrimination, etc.), plus granularity and incompressibility.

```bash
python -m topo_mimic.analysis.method_analysis.property.cycle_asymmetry ...
python -m topo_mimic.analysis.method_analysis.property.transition_topology ...
python -m topo_mimic.analysis.method_analysis.property.theta_filmstrip ...
python -m topo_mimic.analysis.method_analysis.property.stationary ...
python -m topo_mimic.analysis.method_analysis.property.rear_cycles ...
python -m topo_mimic.analysis.method_analysis.incompressibility.run configs/analysis/method_analysis/incompressibility/incompressibility.yaml
python -m topo_mimic.analysis.method_analysis.incompressibility.batch configs/analysis/method_analysis/incompressibility/batch.yaml
python -m topo_mimic.analysis.method_analysis.incompressibility.varying_k configs/analysis/method_analysis/incompressibility/varying_k.yaml
python -m topo_mimic.analysis.method_analysis.global_manifold.run configs/analysis/method_analysis/global_manifold/rodent.yaml
python -m topo_mimic.analysis.method_analysis.granularity.main configs/analysis/method_analysis/granularity/granularity_{intention,qpos,ekp}.yaml
python -m topo_mimic.analysis.method_analysis.information_loss.main configs/analysis/method_analysis/information_loss/information_loss.yaml
```

The `information_loss` module quantifies the coarse-graining information loss
of each method by computing I(T;Y) between the per-frame primitive assignment
T and the GT directed-transition labels Y. Outputs land in
`outputs/analysis/method_analysis/information_loss/` (CSV + NPZ + IB-plane
PNG/PDF + appendix bar chart + joint-distribution heatmaps + LaTeX-source
table). It is also chained as the final stage of
`bash bash_scripts/run_method_analysis.sh`.

## Stage 5: Auto-labeling

```bash
python -m topo_mimic.labeling.main configs/labeling/auto_label.yaml
```

Each cluster is auto-labeled by majority GT overlap (directional, per-sample). Set `launch_gui: true` in the config for an interactive Gradio override GUI. Details in [docs/autoassign.md](autoassign.md).

## Stage 6: Scale sweep (fairness)

Sweeps KPMS (κ × num_states) and Mapper (lens × n_intervals × feature_space) and reports per-variant discovery of the canonical transitions. Addresses the "baselines under-tuned" reviewer concern.

```bash
python -m topo_mimic.analysis.scale_sweep configs/analysis/scale_sweep/scale_sweep.yaml
```

CLI flags: `--phases 0,1,2,3,4,5`, `--max-fits N`, `--force`. Prepend `JAX_PLATFORMS=cpu` if KPMS OOMs on GPU (Mapper is CPU-only anyway).

## Stage 7: Tables

```bash
python -m topo_mimic.analysis.tables              # all three table families
python -m topo_mimic.analysis.tables paper        # Figure 4 summary + Appendix DTW + Appendix consistency
python -m topo_mimic.analysis.tables warping      # DTW warping-heaviness LaTeX tables
python -m topo_mimic.analysis.tables results      # main results table
```

Outputs land under `outputs/analysis/tables/`.
