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
python -m demos.illustrative.demo_transition_filmstrip --config configs/demos/illustrative/demo_transition_filmstrip_walker.yaml
```
Renders filmstrip visualizations of behavioral transitions captured by DM paths and cycles.

### Bodies (Figure 1E)
```bash
python -m demos.illustrative.demo_bodies
python -m demos.illustrative.demo_bodies --config configs/demos/illustrative/demo_bodies.yaml
```
Produces a side-by-side chronophotography figure showing increasing morphological complexity: a generated, labeled multi-transition walker (left) with an arrow pointing to an imitated, unlabeled multi-transition rodent (right). Each panel is enclosed in a rounded grey box with a text label.

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

### Stationary Behavior Demo
```bash
python -m demos.property.demo_stationary
```
Analyzes stationary (low root displacement) clips, showing how DM cycles capture subtle movements like grooming and orienting even when the rodent is not locomoting.

### Rear-Cycle Grid Demo
```bash
python -m demos.property.demo_rear_cycles
python -m demos.property.demo_rear_cycles --config configs/demos/property/demo_rear_cycles.yaml
python -m demos.property.demo_rear_cycles --clip-ids 42 99 103
```
Discovers N clips with X→rear→X pattern (z low→high→low), runs DM extraction on each, and produces four grid outputs: (1) `cycle_rotating_grid.gif` — rotating 3D PCA with synced video for DM cycles, (2) `cycle_filmstrip_grid.png` — chronophotography filmstrips for cycle segments, (3) `path_rotating_grid.gif` — rotating 3D PCA with synced video for DM paths, (4) `path_filmstrip_grid.png` — filmstrips for path segments.

### Incompressibility Verification
```bash
python -m demos.property.incompressibility.incompressibility configs/scripts/incompressibility.yaml
python -m demos.property.incompressibility.batch_incompressibility
```
Verifies the incompressibility property of the DM skeleton. Use `--use-dm-path` to include DM path extraction alongside cycles.

---

## Transition Demos

End-to-end transition analysis pipelines.

### Walker Transitions — Path Analysis (Figure 2B)
```bash
python -m demos.transition_demos.demo_walker_transitions_path
python -m demos.transition_demos.demo_walker_transitions_path --config configs/demos/transition_demos/demo_walker_transitions_path.yaml
```
Analysis pipeline for long walker clips (~5000 frames) with multiple behavior transitions per clip. DM all-pairs-path extraction discovers behavior segments without relying on ground-truth labels. With `skeleton_analysis.enabled: true`, also produces skeleton filmstrip figures (Figure 2B) in `{output_dir}/skeleton_analysis_filmstrip/`.

### Rodent Transition Demo (Figure 2C)
```bash
python -m demos.transition_demos.demo_rodent_transitions
python -m demos.transition_demos.demo_rodent_transitions --config configs/demos/transition_demos/demo_rodent_transitions.yaml
```
Visualizes rodent behavioral transitions using PCA/UMAP/t-SNE with up to 3 rows per embedding plot: GT transitions, DM cycles, and DM paths. With `skeleton_analysis.enabled: true`, also produces skeleton filmstrip figures (Figure 2C) in `{output_dir}/skeleton_analysis_filmstrip/`.

---

## Walker Data Generation

```bash
python3 -m walker_generation configs/walker_generation/path_sequence.yaml
```

See [walker_generation.md](walker_generation.md) for full documentation.
