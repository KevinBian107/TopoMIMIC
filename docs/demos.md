# TopoVNL Demos

Illustrative demonstrations of the topological analysis methods.

---

## Running Demos

All demos are run from the project root:

```bash
python demos/<subfolder>/demo_<name>.py
```

---

## Illustrative Demos

Visual examples of TDA concepts.

### Torus DM Demo
```bash
python demos/illustrative/demo_torus.py
```
Demonstrates discrete Morse graph extraction on a synthetic torus point cloud. Shows how the DM skeleton captures the topology of a known shape.

### Behavior Overlap Demo
```bash
python demos/illustrative/demo_behavior_overlap.py
```
Visualizes overlap between different behavioral clusters in feature space, demonstrating how topological methods can distinguish behaviors that overlap in simple projections.

### Saddle Point Demo
```bash
python demos/illustrative/demo_saddle.py
```
Demonstrates saddle point detection in the density landscape — the transition regions between behavioral basins.

### Transition Filmstrip
```bash
python demos/illustrative/demo_transition_filmstrip.py
python demos/illustrative/demo_transition_filmstrip.py --config configs/demos/illustrative/demo_transition_filmstrip.yaml
python demos/illustrative/demo_transition_filmstrip.py --config configs/demos/illustrative/demo_transition_filmstrip_walker.yaml
```
Renders filmstrip visualizations of behavioral transitions captured by DM paths and cycles.

---

## Property Demos

Demonstrations of model/method properties.

### Cycle Asymmetry Demo
```bash
python demos/property/demo_cycle_asymmetry.py
python demos/property/demo_cycle_asymmetry.py --config configs/demos/property/demo_cycle_asymmetry.yaml
```
Shows how DM cycles can reveal behavioral asymmetry: the motor program for entering a behavior is distinct from the motor program for exiting it. Demonstrates the endpoint matching condition for validating cycles.

### Transition Topology Demo
```bash
python demos/property/demo_transition_topology.py
python demos/property/demo_transition_topology.py --config configs/demos/property/demo_transition_topology.yaml
```
The primary demo for DM-path extraction. Compares DM paths (H0: edge-disjoint paths between density minima) with DM cycles (H1: fundamental loops) for behavioral transition recovery. `demo_transition_topology.py` is a comprehensive extension of `demo_cycle_asymmetry.py` — it adds edge-connectivity analysis, automatic clip discovery, and endpoint similarity.

### Theta-Graph Filmstrip
```bash
python demos/property/demo_theta_filmstrip.py
python demos/property/demo_theta_filmstrip.py --config configs/demos/property/demo_theta_filmstrip.yaml
```
Renders filmstrip visualizations of theta-graph structures (two edge-disjoint paths between density minima), showing the distinct forward and return transition trajectories.

### Stationary Behavior Demo
```bash
python demos/property/demo_stationary.py
```
Analyzes stationary (low root displacement) clips, showing how DM cycles capture subtle movements like grooming and orienting even when the rodent is not locomoting.

### Rear-Cycle Grid Demo
```bash
python demos/property/demo_rear_cycles.py
python demos/property/demo_rear_cycles.py --config configs/demos/property/demo_rear_cycles.yaml
python demos/property/demo_rear_cycles.py --clip-ids 42 99 103
```
Discovers N clips with X→rear→X pattern (z low→high→low), runs DM extraction on each, and produces four grid outputs: (1) `cycle_rotating_grid.gif` — rotating 3D PCA with synced video for DM cycles, (2) `cycle_filmstrip_grid.png` — chronophotography filmstrips for cycle segments, (3) `path_rotating_grid.gif` — rotating 3D PCA with synced video for DM paths, (4) `path_filmstrip_grid.png` — filmstrips for path segments.

### Incompressibility Verification
```bash
python demos/property/incompressibility/incompressibility.py configs/scripts/incompressibility.yaml
python demos/property/incompressibility/batch_incompressibility.py
```
Verifies the incompressibility property of the DM skeleton. Use `--use-dm-path` to include DM path extraction alongside cycles.

---

## Transition Demos

End-to-end transition analysis pipelines.

### Walker Transitions — Path Analysis
```bash
python demos/transition_demos/demo_walker_transitions_path.py
python demos/transition_demos/demo_walker_transitions_path.py --config configs/demos/transition_demos/demo_walker_transitions_path.yaml
```
Analysis pipeline for long walker clips (~5000 frames) with multiple behavior transitions per clip. DM all-pairs-path extraction discovers behavior segments without relying on ground-truth labels.

### Rodent Transition Demo
```bash
python demos/transition_demos/demo_rodent_transitions.py
python demos/transition_demos/demo_rodent_transitions.py --config configs/demos/transition_demos/demo_rodent_transitions.yaml
```
Visualizes rodent behavioral transitions using PCA/UMAP/t-SNE with up to 3 rows per embedding plot: GT transitions, DM cycles, and DM paths.

---

## Walker Data Generation

```bash
python3 -m walker_generation configs/walker_generation/path_sequence.yaml
```

See [walker_generation.md](walker_generation.md) for full documentation.