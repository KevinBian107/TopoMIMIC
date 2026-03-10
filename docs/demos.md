# TopoVNL Demos

Illustrative demonstrations of the topological analysis methods.

---

## Running Demos

All demos are run as Python modules from the project root:

```bash
python -m demos.<demo_name>
```

---

## Demo Index

### Torus DM Demo
```bash
python -m demos.demo_torus
```
Demonstrates discrete Morse graph extraction on a synthetic torus point cloud. Shows how the DM skeleton captures the topology of a known shape.

### Stationary Behavior Demo
```bash
python -m demos.demo_stationary
```
Analyzes stationary (low root displacement) clips, showing how DM cycles capture subtle movements like grooming and orienting even when the rodent is not locomoting.

### Behavior Overlap Demo
```bash
python -m demos.demo_behavior_overlap
```
Visualizes overlap between different behavioral clusters in feature space, demonstrating how topological methods can distinguish behaviors that overlap in simple projections.

### Saddle Point Demo
```bash
python -m demos.demo_saddle
```
Demonstrates saddle point detection in the density landscape — the transition regions between behavioral basins.

### Cycle Asymmetry Demo
```bash
python -m demos.demo_cycle_asymmetry
```
Shows how DM cycles can reveal behavioral asymmetry: the motor program for entering a behavior is distinct from the motor program for exiting it. Demonstrates the endpoint matching condition for validating cycles.

### Transition Topology Demo
```bash
python -m demos.demo_transition_topology
```
The primary demo for DM-path extraction. Compares DM paths (H0: edge-disjoint paths between density minima) with DM cycles (H1: fundamental loops) for behavioral transition recovery. Demonstrates that paths are a reliable primary tool for detecting transitions, while cycles provide additional asymmetry information when the endpoint matching condition holds.

### Theta-Graph Filmstrip
```bash
python -m demos.demo_theta_filmstrip
```
Renders filmstrip visualizations of theta-graph structures (two edge-disjoint paths between density minima), showing the distinct forward and return transition trajectories.

### Transition Filmstrip
```bash
python -m demos.demo_transition_filmstrip
```
Renders filmstrip visualizations of behavioral transitions captured by DM paths and cycles.

### Rodent Transition Demo
```bash
python demos/demo_rodent_transitions.py
python demos/demo_rodent_transitions.py --config configs/demos/demo_rodent_transitions.yaml
```
Visualizes rodent behavioral transitions using PCA/UMAP/t-SNE with up to 3 rows per embedding plot: GT transitions, DM cycles, and DM paths — all colored by behavior transition labels looked up from the GT CSV. Finds all labeled transitions in the test clips via `find_all_transitions()`, groups by directed type, picks the top K by count, and runs DM extraction on the test-set data to extract cycle and path segments. Each DM segment is assigned a transition label by majority-vote of its per-frame GT behaviors. Outputs: `pca_comparison.png`, `umap_comparison.png`, `tsne_comparison.png` (multi-row), `transition_summary.txt`, DM extraction artifacts in `dm/`, and optionally `gt_transitions.mp4`.

---

## Walker Data Generation

Config-driven pipeline for generating walker behavior data. Two modes are available:

### Short-Sequence Generation (Single Transitions)
```bash
python3 -m walker_generation configs/walker_generation/short_sequence.yaml
```
Generates short clips (250 frames) each containing exactly one targeted behavior transition. Each `TransitionSpec(src, dst, n_clips)` restricts the schedule to a 2-mode pair, crops around the transition, and extracts Step 2 latent activations. Output: `data/walker.h5` + `walker_behavior_labels.csv` (2 rows per clip: source and destination segments).

### Long-Sequence Generation (Multi-Transition Clips)
```bash
python3 -m walker_generation configs/walker_generation/long_sequence.yaml
```
Generates long clips (~5000 frames) each containing multiple behavior transitions. Key config options:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_clips` | Number of clips to generate | 20 |
| `clip_length` | Frames per clip | 5000 |
| `n_transitions` | Transitions per clip | 8 |
| `source_pool` / `target_pool` | Behavior pools for random selection | 4 modes each |
| `variable_transition_length` | Whether transition durations vary | `true` |
| `transition_length_min/max` | Range for variable transition lengths | 50–150 |

Output: `data/walker_long.h5` + `walker_long_behavior_labels.csv` (multiple rows per clip, one per behavior segment). Optionally renders a demo clip video via `output.demo_clip_video`.

---

## Walker Transition Demos

### Walker Transitions — Short Clips
```bash
python demos/demo_walker_transitions_short.py
python demos/demo_walker_transitions_short.py --config configs/demos/demo_walker_transitions_short.yaml
```
End-to-end demo for short transition clips (one transition per clip). DM extraction → W2 clustering → GT comparison → PCA/UMAP/t-SNE plots. Each clip is treated as a single unit with stacked feature windows. Generates rendered videos of discovered clusters vs ground truth transition groups, plus embedding visualizations colored by GT and DM labels. Set `only_analysis: false` for the first run (computes DM + clustering from scratch), then `only_analysis: true` to re-run visualization on cached results.

### Walker Transitions — Long Sequences
```bash
python demos/demo_walker_transitions_long.py
python demos/demo_walker_transitions_long.py --config configs/demos/demo_walker_transitions_long.yaml
```
Analysis pipeline for long walker clips (~5000 frames) with multiple behavior transitions per clip. DM-path extraction is the natural chunking method for these long sequences — it discovers behavior segments without relying on ground-truth labels.

Two visualization modes:
- **Per-frame**: each frame is an individual 16D point colored by GT behavior label
- **Windowed**: sliding windows of N stacked frames colored by majority behavior

Also runs DM-cycle and DM-path extraction, W2 clustering, comparison plots (GT vs discovered), and grid rendering (rows = clips, columns = longest K DM segments found). Supports `only_analysis: true` for re-running visualization on cached results with annotated cluster labels.

---

## Incompressibility Verification

```bash
python -m demos.incompressibility.incompressibility configs/scripts/incompressibility.yaml
python -m demos.incompressibility.batch_incompressibility
```

Verifies the incompressibility property of the DM skeleton: that the topological structure cannot be simplified further without losing behavioral information. Use `--use-dm-path` to include DM path extraction alongside cycles.

---

*Document generated for TopoVNL demo documentation.*
