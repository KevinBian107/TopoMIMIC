# TopoVNL Codebase Structure

## Project overview

TopoVNL analyzes the latent space of deep-RL imitation policies (Mimic-mjx rodent) via topological data analysis. The pipeline extracts the discrete Morse graph of each clip's high-dimensional trajectory, treats its **cycles** (H₁) and **paths** (H₀ between density minima) as manifold primitives, and compares them via Wasserstein-2 distance to build a behavioral taxonomy.

```
Mimic-mjx rollouts                                            paper figures + tables
       │                                                            ▲
       ▼                                                            │
[extraction] → [experiment] → [analysis/feature_analysis] → [analysis/tables]
                              [analysis/method_analysis]
                              [analysis/scale_sweep]
                              [labeling]
```

## Top-level layout

```
TopoVNL/
├── topo_mimic/                # core library (see below)
├── PCD-Graph-Recon-DM/        # external C++ DM-graph backend (cloned dep)
├── configs/                   # YAML configs, mirrors topo_mimic/ layout
├── outputs/                   # pipeline outputs, mirrors topo_mimic/ layout
├── bash_scripts/              # per-stage shell drivers + run_all.sh
├── demos/                     # pedagogical / concept demos (not method results)
├── behavior_labels/           # rodent GT CSV
├── data/                      # input rollouts (.h5)
├── figures/                   # static figures referenced by the paper
├── paper/                     # paper manuscript + tables
├── docs/                      # documentation (this file lives here)
└── scratch/                   # investigation logs (gitignored)
```

## `topo_mimic/`

```
topo_mimic/
├── extraction/              # DM extraction (was topo_mimic/dmwk/)
│   ├── dm_pipeline.py           # entry: python -m topo_mimic.extraction.dm_pipeline
│   ├── skeletonization/         # DM graph build (cycles, H0 paths, Takens, remap)
│   ├── utils/                   # load_utils, feature_transforms, dataset_config
│   └── xml/                     # MuJoCo rodent model
├── experiment/              # benchmark clustering runners (was benchmark/)
│   ├── transitions.py           # main experiment: python -m topo_mimic.experiment.transitions
│   ├── minima.py                # alternate 'minima' experiment
│   ├── methods.py               # METHOD_REGISTRY (W2, GW, KPMS, VAE, Mapper)
│   ├── auto_label.py            # label-assignment primitives used by experiment + labeling
│   └── c2st/                    # classifier two-sample test
├── analysis/
│   ├── feature_analysis/        # analysis of methods across feature spaces
│   │   ├── main_transitions.py      # orchestrator (runs all blocks below)
│   │   ├── coverage_quality_panels.py   # per-family coverage-vs-quality boxplot grid
│   │   └── blocks/                  # atomic per-block entry points:
│   │       ├── gt_sanity.py             # GT-transition DTW distinctiveness
│   │       ├── transition_ranking.py    # GT transition variant counts
│   │       ├── mds.py                   # MDS cluster subspace projections
│   │       ├── transition_comparison.py
│   │       ├── transition_graph.py
│   │       ├── dtw_metrics.py           # DTW + consistency + warping (sync ratio, non-diag)
│   │       ├── dtw_visualization.py
│   │       ├── mmd.py
│   │       ├── c2st.py
│   │       ├── purity.py
│   │       ├── classification.py
│   │       └── render_gt.py
│   ├── method_analysis/         # method-property studies
│   │   ├── property/                # cycle_asymmetry, transition_topology,
│   │   │                            # theta_filmstrip, stationary, rear_cycles,
│   │   │                            # cycle_behavior_breakdown
│   │   ├── incompressibility/       # run.py, batch.py, varying_k.py
│   │   ├── global_manifold/         # run.py, common.py, filmstrip.py
│   │   ├── granularity/             # multi_persistence + stationary_joint_variance + redundancy
│   │   ├── dm_dim_reduction/        # empirical PCA justification
│   │   ├── cycle_persistence/       # persistence-threshold sensitivity
│   │   ├── transition_stats/        # GT transition frequency
│   │   ├── cycle_rendering/         # DM cycle videos
│   │   └── information_loss/        # I(T;Y) MI under coarsening + IB-plane figure
│   ├── scale_sweep/             # baseline-fairness KPMS + Mapper sweep
│   ├── mapper/                  # Mapper method implementation
│   └── tables/                  # unified paper table generator
│       ├── results_table.py
│       ├── paper_tables.py
│       └── warping_tables.py
├── labeling/                # auto-label + render-for-label
│   ├── main.py                  # auto-label + optional GUI
│   └── render_for_label.py      # TRACK vs STAC side-by-side videos for annotation
└── hyperparam_sweep/         # KPMS + VAE W&B sweeps
```

## `configs/` — mirrors `topo_mimic/`

```
configs/
├── extraction/              dm_pipeline_{intention,qpos,ekp}.yaml
├── experiment/              transitions.yaml, minima.yaml
├── analysis/
│   ├── feature_analysis/    analysis_transitions.yaml
│   ├── method_analysis/
│   │   ├── property/*.yaml
│   │   ├── incompressibility/{incompressibility,batch,varying_k}.yaml
│   │   ├── granularity/granularity_{intention,qpos,ekp}.yaml
│   │   ├── global_manifold/rodent.yaml
│   │   └── cycle_rendering.yaml
│   └── scale_sweep/         scale_sweep.yaml
├── labeling/                auto_label.yaml, render_for_label.yaml
├── hyperparam_sweep/        kpms_sweep.yaml, vae_sweep.yaml + wandb sweep specs
└── demos/                   concept demos only (synthetic / GT-only)
```

## `outputs/` — mirrors `topo_mimic/`

```
outputs/
├── extraction/dm_pipeline_{intention,qpos,ekp}/
├── experiment/transitions/
├── analysis/
│   ├── feature_analysis/    (per-method × per-seed subfolders)
│   ├── method_analysis/     (per-property subfolders)
│   ├── scale_sweep/v1/cfg_<hash>/
│   └── tables/              (warping_analysis/, CSV aggregates, PDF/PNG tables)
├── labeling/rendered_for_label/
└── demos/
```

## `bash_scripts/`

Six stage-scripts plus a `run_all.sh` aggregator:

```
run_extraction.sh          DM extraction × 3 feature spaces
run_experiment.sh          benchmark experiment
run_feature_analysis.sh    11 atomic feature-analysis blocks
run_method_analysis.sh     property + incompressibility + global_manifold + granularity
run_scale_sweep.sh         KPMS/Mapper fairness sweep
run_tables.sh              paper tables
run_all.sh                 chains all of the above
```

Override the Python binary with `PYTHON=/path/to/python3 bash bash_scripts/run_all.sh`.

## `demos/`

Concept demos only — no real-method outputs. Use these to understand TDA or to illustrate figures:

```
demo_torus.py                     synthetic torus, 6-stage SWR pipeline
demo_saddle.py                    synthetic saddle, cycle vs path
demo_behavior_overlap.py          synthetic behavior cluster reference panels
demo_transition_filmstrip.py      GT transition filmstrips (rodent only)
demo_transitions_projections_single_clip.py               rodent end-to-end transition analysis (was demo_rodent_transitions)
demo_cluster_filmstrips.py        cluster-label filmstrips
demo_dtw_metric.py                DTW cost matrix + filmstrip (metric illustration)
demo_feature_space_projections.py real vs DM-selected frames in PCA/UMAP/t-SNE
demo_clip_projections.py          clip-level analogue of the above
skeleton_filmstrip.py             helper (called by demo_transitions_projections_single_clip and global_manifold/filmstrip)
```

## Data flow

```
.h5 rollouts
      │ load_rollout_from_source
      ▼
Takens delay embedding (m, τ per feature space)
      │ extract_skeleton_pcd_dm
      ▼
DM graph = {cycles (H₁), paths (H₀ between minima)}
      │
      ▼  cache: outputs/extraction/dm_pipeline_<space>/all_{cycle,path}_dm.pkl
      │
      ▼  compute_w2_distance_matrix (cached)
      ▼
W2 distance matrix → MDS → KMeans
      │  cache: outputs/experiment/transitions/w2_distance_matrix_*.pkl
      ▼
Clustered transitions → feature_analysis + method_analysis + tables
```

## External dependencies

| Category                | Packages                                               |
|-------------------------|--------------------------------------------------------|
| Core scientific         | numpy, scipy, scikit-learn, pandas, matplotlib, seaborn |
| TDA                     | ripser, ot (POT), networkx                             |
| Dim reduction           | umap-learn, sklearn.manifold                           |
| Sim / DL                | jax, mujoco, h5py                                      |
| Optional (baselines)    | keypoint_moseq, torch                                  |
| C++                     | PHAT library, C++11 compiler                           |
