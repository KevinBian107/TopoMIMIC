# Reproducing the paper

Each table and figure, the command that produces it, and its output (run the pipeline stages in the [README](../README.md#quick-start) first). Outputs under `paper/` go to `paper/bian_NeurReps_2026/`.

| Paper | Command | Output |
|---|---|---|
| Table 1, Table 2, Tables 9–13 | `bash bash_scripts/run_tables.sh` | `tables/main_combined.tex`, `appendix_coverage.tex`, `appendix_{dtw_detailed,consistency,dtw_sync_ratio,non_diag_frac,kid}.tex` |
| Table 3 | `bash bash_scripts/run_scale_sweep.sh`, then `run_tables.sh` | `tables/appendix_scale_sweep.tex` |
| Table 4 | `python -m topo_mimic.analysis.method_analysis.skeleton_reduction` | `tables/appendix_skeleton_reduction.tex` |
| Table 5 | `python -m topo_mimic.analysis.method_analysis.k_sweep`, then `run_tables.sh` | `tables/appendix_k_sweep.tex` |
| Tables 6–7 | `python -m topo_mimic.analysis.method_analysis.path_persistence_sweep --stages ami downstream aggregate`, then `run_tables.sh` | `tables/appendix_path_persistence_{ami,down}.tex` |
| Table 8, Figure 2 | `python -m topo_mimic.analysis.method_analysis.information_loss.main configs/analysis/method_analysis/information_loss/information_loss_reduced.yaml` | `outputs/analysis/method_analysis/information_loss_reduced/figures/{information_loss_table.tex, main_panel.png}` |
| Table 14 | DM-G hyperparameters | `configs/extraction/dm_pipeline_{intention,qpos,ekp}.yaml` |
| Table 15 | MDS sweep in `bash bash_scripts/run_sensitivity.sh`, then `run_tables.sh` | `tables/appendix_mds_sweep.tex` |
| Figure 5 | `python -m topo_mimic.figure_generation.build_lexicon OUT.png` | behavior lexicon |
| Figures 6–7 | `python -m demos.demo_transition_projections --config configs/demos/demo_transition_projections.yaml` | `outputs/demos/illustrative/transition_projections/` |
| Figure 8 | `python -m topo_mimic.figure_generation.build_appendix_stationary` | `appendix_figures/appendix_stationary.png` |
| Figure 9 | `python -m topo_mimic.figure_generation.build_appendix_asymmetry` | `appendix_figures/appendix_asymmetry.png` |
| Figures 10–13 | `python -m topo_mimic.figure_generation.build_cluster_examples METHOD OUT.png` | cluster examples for DMP-Qpos, DMC-Int, KPMS-KP, Mapper-Qpos |

Figures 1, 3 and 4 are schematic. Figures 1 and 3 are composed by `build_figure_flow.py` and `build_figure1.py` from a trajectory render placed in `topo_mimic/figure_generation/external_data/`. Figure 4 is the MIMIC-MJX pipeline figure from Zhang et al. (2025).

DM-G skeletonization and primitive extraction are deterministic. The MDS, K-means and stochastic baselines run over seeds 0, 1 and 2, and the tables report their mean and standard deviation.
