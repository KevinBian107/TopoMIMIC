# TopoVNL Demos

Illustrative demonstrations of the topological analysis methods.

---

## Running Demos

All demos are run from the project root:

```bash
python -m demos.<subfolder>.demo_<name>
```

---

## Illustrative Demos

Visual examples of TDA concepts.

### Torus DM Demo
```bash
python -m demos.illustrative.demo_torus
```
Demonstrates discrete Morse graph extraction on a synthetic torus point cloud. Shows how the DM skeleton captures the topology of a known shape.

### Behavior Overlap Demo
```bash
python -m demos.illustrative.demo_behavior_overlap
```
Visualizes overlap between different behavioral clusters in feature space, demonstrating how topological methods can distinguish behaviors that overlap in simple projections.

### Saddle Point Demo
```bash
python -m demos.illustrative.demo_saddle
```
Demonstrates saddle point detection in the density landscape — the transition regions between behavioral basins.

### Transition Filmstrip
```bash
python -m demos.illustrative.demo_transition_filmstrip
python -m demos.illustrative.demo_transition_filmstrip --config configs/demos/illustrative/demo_transition_filmstrip.yaml
```
Renders filmstrip visualizations of behavioral transitions captured by DM paths and cycles.

### Feature-Space Projection Comparison (real vs. DM)
```bash
python -m demos.illustrative.demo_feature_space_projections
python -m demos.illustrative.demo_feature_space_projections --config configs/demos/illustrative/demo_feature_space_projections.yaml
```
Six Nature-style figures comparing real vs. DM-selected frames under the
three feature spaces (intention, qpos, ekp) and three projections (PCA,
UMAP, t-SNE). Each figure is a 2-row × 3-column grid — top row is real
frames (stride-subsampled from all 842 clips), bottom row is frames
selected by the named DM method (DMC or DMP) loaded from
`outputs/benchmark_experiment_transitions/w2_{dm}_{fs}_kmeans/results.pkl`.
Points are colored by the GT transition each frame belongs to (via
`find_all_transitions(window_frames=100)`); frames outside any transition
span are grey. A single horizontal legend sits below the grid. File
names: `proj_{dmc|dmp}_{pca|umap|tsne}.{png,pdf}` under
`outputs/demos/feature_space_projections/`.

### DTW Metric Visualization
```bash
python -m demos.illustrative.demo_dtw_metric
python -m demos.illustrative.demo_dtw_metric --config configs/demos/illustrative/demo_dtw_metric.yaml
python -m demos.illustrative.demo_dtw_metric --method w2_dmc_int_kmeans --seed 0
```
Nature-style per-cluster DTW comparison figures. For each unique transition cluster in a benchmark method, renders a panel with the DTW cost matrix (with warping path) on the left and two chronophotography filmstrips on the right: discovered sequence (blue tint) on top, ground truth (red tint) on bottom. Uses the same DTW computation as the main analysis pipeline. Default method: `w2_dmp_qpos_kmeans`.

---

## Property Demos

Demonstrations of model/method properties.

### Cycle Asymmetry Demo
```bash
python -m demos.property.demo_cycle_asymmetry
python -m demos.property.demo_cycle_asymmetry --config configs/demos/property/demo_cycle_asymmetry.yaml
```
Shows how DM cycles can reveal behavioral asymmetry: the motor program for entering a behavior is distinct from the motor program for exiting it. Demonstrates the endpoint matching condition for validating cycles.

### Transition Topology Demo
```bash
python -m demos.property.demo_transition_topology
python -m demos.property.demo_transition_topology --config configs/demos/property/demo_transition_topology.yaml
```
The primary demo for DM-path extraction. Compares DM paths (H0: edge-disjoint paths between density minima) with DM cycles (H1: fundamental loops) for behavioral transition recovery. `demo_transition_topology.py` is a comprehensive extension of `demo_cycle_asymmetry.py` — it adds edge-connectivity analysis, automatic clip discovery, and endpoint similarity.

### Theta-Graph Filmstrip
```bash
python -m demos.property.demo_theta_filmstrip
python -m demos.property.demo_theta_filmstrip --config configs/demos/property/demo_theta_filmstrip.yaml
```
Renders filmstrip visualizations of theta-graph structures (two edge-disjoint paths between density minima), showing the distinct forward and return transition trajectories.

### Stationary Behavior Demo (Appendix: DM-G Identifies Stationary Clips)
```bash
python -m demos.property.demo_stationary
python -m demos.property.demo_stationary --config configs/demos/property/demo_stationary.yaml
```
Compares clips with DM-G cycles (behaviorally rich) vs clips without cycles (stationary/simple topology). Produces: (1) `stationary_comparison.pdf` — Nature-style bar plots of root motion and body part variance for cycle vs no-cycle clips, (2) `stationary_filmstrip_grid.png` — grey-box chronophotography grid of representative no-cycle (stationary) clips. Used for the appendix section on stationary clip identification.

### Rear-Cycle Grid Demo
```bash
python -m demos.property.demo_rear_cycles
python -m demos.property.demo_rear_cycles --config configs/demos/property/demo_rear_cycles.yaml
python -m demos.property.demo_rear_cycles --clip-ids 42 99 103
```
Discovers N clips with X→rear→X pattern (z low→high→low), runs DM extraction on each, and produces four grid outputs: (1) `cycle_rotating_grid.gif` — rotating 3D PCA with synced video for DM cycles, (2) `cycle_filmstrip_grid.png` — chronophotography filmstrips for cycle segments, (3) `path_rotating_grid.gif` — rotating 3D PCA with synced video for DM paths, (4) `path_filmstrip_grid.png` — filmstrips for path segments.

### Topological Simplification (Figure 3)
```bash
python -m demos.property.incompressibility.incompressibility configs/scripts/incompressibility.yaml
python -m demos.property.incompressibility.batch_incompressibility
```
Validates topologically invariant simplification — that the DM skeleton retains only the minimal frames needed to preserve topological structure. `incompressibility.py` produces per-trial outputs including: (1) `static_intention_graph.png` — compact Nature-style plot of mean intention traces with DM-retained frames marked, (2) `dm_graph_visualization.png` — PCA/t-SNE/UMAP side-by-side with DM-cycle overlay and different markers for stationary vs moving clips, (3) `clip_filmstrips.png` — grey-box chronophotography filmstrips for each clip. `batch_incompressibility.py` produces the summary bar and line plots (Figure 3 top row). Use `--use-dm-path` to include DM path extraction alongside cycles.

---

## Transition Demos

End-to-end transition analysis pipelines.

### Rodent Transition Demo (Figure 2C)
```bash
python -m demos.transition_demos.demo_rodent_transitions
python -m demos.transition_demos.demo_rodent_transitions --config configs/demos/transition_demos/demo_rodent_transitions.yaml
```
Visualizes rodent behavioral transitions using PCA/UMAP/t-SNE with up to 3 rows per embedding plot: GT transitions, DM cycles, and DM paths. With `skeleton_analysis.enabled: true`, also produces skeleton filmstrip figures (Figure 2C) in `{output_dir}/skeleton_analysis_filmstrip/`.

---

## Global-Manifold Demos

Pool clips that share configured ground-truth transition types into a single point cloud, run DM on the *pooled* cloud per feature space, and visualize the global manifold structure. Entry points live in `demos/global_manifold/` with configs in `configs/demos/global_manifold/`. Two PDFs are produced per run:

1. `projection_grid.pdf` — per feature space × (DM-cycles, DM-paths) × (PCA, t-SNE, UMAP). Background points coloured by GT primitive behaviour (shape = turn direction for rodent). DM minima marked with gold stars, sampled paths/cycles overlaid with colour gradients.
2. `filmstrip_grid.pdf` — one box per configured transition pair, `k_per_pair` rows of cross-clip stitched filmstrips. Each pose is annotated with its source clip id so the cross-clip nature of a global path is visible.

### Rodent (3 feature spaces)
```bash
python -m demos.global_manifold.demo_rodent_global
python -m demos.global_manifold.demo_rodent_global --config configs/demos/global_manifold/rodent.yaml
python -m demos.global_manifold.demo_rodent_global --only-feature-space intention --skip-filmstrip
```
Defaults: dataset `rodent`, three transitions (`Rear↔Walking`, `Orienting↔Walking`, `Immobility↔Walking`), four primitives for label-minima selection (`Rear, Walking, Orienting, Immobility`), and feature spaces `intention / qpos / ekp`.

### Config knobs
- `transition_types`: list of `[A, B]` primitive pairs to select clips by and to sample paths for.
- `feature_spaces[].subsample_n`: legacy shortcut — stride-N subsample on the Takens pool before DM (default 3). Smaller = denser but slower DM. Equivalent to `subsample: {method: stride, n: N}`.
- `feature_spaces[].subsample`: explicit subsample spec.
  - `method: stride` (default) — take every `n`-th Takens point per clip.
  - `method: dbscan` — run global DBSCAN, keep all noise (sparse) points, thin each cluster by `dense_downsample_frac` (0.9 → drop 90% of dense-region points). `dbscan_eps` defaults to the median k-th-neighbor distance; `dbscan_min_samples` defaults to 10.
- `feature_spaces[].pca_dims`: set for qpos/ekp to strip residual drift; leave `null` for intention.
- `minima_selection.primitives`: behaviours to assign density minima to (one minimum per behaviour via kNN plurality vote over the pooled Takens cloud).
- `path_sampling.k_per_pair`: paths sampled per `(A, B)` pair; edge-disjoint paths first, then shortest-simple-path fill.
- Caches: `{cache_dir}/{feature_space}/{pool.npz, dm_output/, dm_bundle.pkl, projections.pkl}` — re-runs skip the expensive DM step.

