<div align="center">

# Shaped by What's Missing

**Topological Invariance Simplification Discovers Behavioral Transitions**

Kaiwen Bian<sup>1,2</sup> &nbsp;·&nbsp;
Eric J. Leonardis<sup>2</sup> &nbsp;·&nbsp;
Yuanjia Yang<sup>1,2</sup> &nbsp;·&nbsp;
Charles Zhang<sup>3</sup> &nbsp;·&nbsp;
Yusu Wang<sup>1</sup> &nbsp;·&nbsp;
Talmo D. Pereira<sup>2</sup>

<sup>1</sup>UC San Diego &nbsp;
<sup>2</sup>Salk Institute for Biological Studies &nbsp;
<sup>3</sup>Harvard University

Official code for the paper<br>
NeurIPS 2026 Workshop on Symmetry and Geometry in Neural Representations (NeurReps), Proceedings Track

[![Project Page](https://img.shields.io/badge/Project-Page-2b7bb9)](https://kbian.org/TopoMimic-website/)
![Venue](https://img.shields.io/badge/NeurReps%202026-Proceedings-6f42c1)
[![MIMIC-MJX](https://img.shields.io/badge/built%20on-MIMIC--MJX-orange)](https://mimic-mjx.talmolab.org/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-green.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
<!-- Add a Paper badge (PMLR / arXiv) and link the Venue badge to the workshop site once public. -->

</div>

<p align="center">
  <img src="docs/figures/method_overview.png" width="90%" alt="Method overview">
</p>

<p align="center"><em>
<b>Method overview.</b> A behavior trajectory is Takens-embedded and
skeletonized into a discrete-Morse graph (Step 1), decomposed into DM-cycles and
DM-paths (Step 2), and the pooled primitives are clustered under the
2-Wasserstein distance into behavioral transition clusters (Step 3).
</em></p>

## Abstract

Understanding how animals move between behaviors is a central question in
neuroethology, yet most segmentation methods first partition time into discrete
states, leaving transitions as the unmodeled residue between them. We take the
opposite view: in a suitable feature space, sustained behaviors are dense regions
of the trajectory's density landscape, and transitions are the sparse corridors
and loops that connect them. Because this structure is a property of the
landscape's shape rather than of any particular coordinates, it can be read off
directly with topology and should persist across feature spaces. We build this
landscape from frame-by-frame trajectories, extract its discrete-Morse graph, and
keep two kinds of structure: paths, the corridors a trajectory follows between
dense behavioral regions, and cycles, closed routes whose entry and exit take
different ways. Clustering these structures yields transitions directly, with no
upstream state partition. On a rat behavior dataset across three feature spaces,
including the latent space of a neural-network controller that imitates the
animal in a physics simulator, the approach recovers transitions at coverage
comparable to state-space and topological-clustering baselines, while carrying
substantially more information about the transition identity, forming clusters at
least as tight as the baselines', and matching annotated transitions with less
reliance on time-warping. More broadly, the results show that behavioral
transitions are a topological feature of a representation: recoverable from the
shape of its trajectories, and consistent across distinct feature spaces.

## Method

The method runs in three steps on each trajectory of a feature space.

1. **Skeletonize.** The trajectory is Takens-embedded so each point carries a
   short window of motion, and the discrete-Morse graph reconstruction of
   [Magee & Wang (2022)](https://doi.org/10.20382/jocg.v13i1a16) recovers the
   ridges of its *k*-NN density field as a one-dimensional skeleton *G*.
2. **Decompose.** A filtration-weighted minimum cycle basis of *G* gives the
   **DM-cycles** (H₁ loops). The remaining split tree, ordered by density, gives
   the **DM-paths**: peak-to-saddle corridors plus the essential
   global-max-to-global-min path.
3. **Cluster.** Primitives are compared as empirical distributions under the
   exact 2-Wasserstein distance, embedded with metric MDS, and clustered with
   *K*-means (DM-WKC). Ground-truth labels are used only for evaluation.

<p align="center">
  <img src="docs/figures/dm_primitives.png" width="90%" alt="Discrete-Morse skeleton and the two primitives">
</p>
<p align="center"><em>(A) The discrete-Morse graph recovers the ridges of a density
terrain. (B) The two primitives on one skeleton: the DM-cycle is the H₁ loop, and
the DM-paths are the split-tree persistence pairs plus the essential path.</em></p>

We run the same pipeline in three feature spaces of the
[MIMIC-MJX](https://mimic-mjx.talmolab.org/) rat:

| Space | Description | Dimension |
|---|---|---|
| **Intention** | Latent intention of the deep-RL imitation policy | 16 |
| **Qpos** | Rolled-out joint angles, root pose removed | 67 (PCA to 10) |
| **EKP** | Egocentric keypoints, heading removed | 60 (PCA to 10) |

## Results

**Transition information.** Per-frame mutual information $I(T;Y)$ between each
method's coarse-graining and the directed-transition class, and its
cardinality-corrected AMI.

| Method | $I(T;Y)$ (bits) | AMI |
|---|---|---|
| **DMP-Qpos** (ours) | **0.895** | **0.230** |
| **DMC-Int** (ours) | 0.873 | 0.213 |
| Keypoint-MoSeq (keypoints) | 0.525 | 0.159 |
| Mapper (qpos) | 0.322 | 0.121 |

- **DM primitives align with less time-warping.** They hold the lower
  non-diagonal DTW fraction on 9 of 11 directed transitions, and Keypoint-MoSeq
  leads on none.
- **Alignment and coherence match the strongest baselines.** Both DM variants
  have the lowest mean DTW to annotated boundaries (0.792 and 0.815, against
  0.822 for Mapper and 0.851 for Keypoint-MoSeq), and DMP-Qpos forms clusters
  as tight as Mapper's.
- **Coverage is comparable.** DMC-Int discovers 10 of the 16 directed
  transitions any method finds, against 11 for Mapper and 9 for Keypoint-MoSeq,
  and four transitions are recovered only by DM primitives.
- **The structure persists across representations.** The same corridors and
  loops appear in all three feature spaces.

See the paper for the full analyses (warping decomposition, information
bottleneck plane, baseline sweeps, and sensitivity to *k*, persistence and MDS
dimension).

## Installation

The code requires Linux, Python 3.12 and `g++`, and a CUDA GPU for the
Keypoint-MoSeq baselines and MuJoCo rendering. It builds on the
[`track-mjx`](https://github.com/talmolab/track-mjx) environment and the
[PCD-Graph-Recon-DM](https://github.com/lucasjmagee/PCD-Graph-Recon-DM)
discrete-Morse backend, cloned into the repository root:

```bash
git clone https://github.com/KevinBian107/TopoMIMIC.git && cd TopoMIMIC
# 1. JAX, MuJoCo and MuJoCo-MJX: install track-mjx (github.com/talmolab/track-mjx), then activate it
pip install -r requirements.txt
# 2. Discrete-Morse backend (tested at commit 711c58e)
git clone https://github.com/lucasjmagee/PCD-Graph-Recon-DM.git
(cd PCD-Graph-Recon-DM/dmpcd/code/persistence_swr && g++ DiMoSC.cpp -I./phat/include -std=c++11 -o spt_cpp)
(cd PCD-Graph-Recon-DM/dmpcd/code/phat_morse_src && g++ ComputeGraphReconstruction.cpp)
```

To keep the backend elsewhere, set `DMPCD_ROOT=/path/to/PCD-Graph-Recon-DM`. The
full setup, a verification snippet, and the rendering backends are in
[`docs/installation.md`](docs/installation.md).

**Data.** The policy rollouts (`rodent_ar.h5`: latent intention and joint angles
for 842 clips) are on
[Google Drive](https://drive.google.com/file/d/1exMwbJbJAqhJdKvT6ez9POY-bFGYbemN/view?usp=sharing)
and go in `data/`. The rat reference clips used for the egocentric keypoints
(`rodent_reference_clips.h5`) come from the MIMIC-MJX release on Hugging Face
([`talmolab/MIMIC-MJX`](https://huggingface.co/datasets/talmolab/MIMIC-MJX)) and
also go in `data/`. The proofread behavior labels are included in
`behavior_labels/rodent_labels.csv`; they were produced with
[TopoMimic-annotation](https://github.com/KevinBian107/TopoMimic-annotation) and
are used only for evaluation.

## Usage

Every stage reads a YAML config under `configs/` and is driven by a script under
`bash_scripts/`. Commands are run from the repository root; set
`PYTHON=/path/to/python` to choose the interpreter.

```bash
bash bash_scripts/run_all.sh                 # every stage below, in order
```

| Stage | Script | What it does |
|---|---|---|
| Extraction | `run_extraction.sh` | Takens embedding, DM skeleton, DM-cycles and DM-paths in each feature space |
| Benchmark | `run_experiment.sh` | W2 → MDS → *K*-means for the DM variants, plus Keypoint-MoSeq, Mapper and the full-clip control (3 seeds) |
| Evaluation | `run_feature_analysis.sh` | Cluster labeling, DTW alignment, consistency, warping decomposition, KID |
| Method analyses | `run_method_analysis.sh` | Information-theoretic analysis, behavioral asymmetry, stationary clips |
| Baseline sweep | `run_scale_sweep.sh` | 110-variant Keypoint-MoSeq and Mapper sweep |
| Sensitivity | `run_sensitivity.sh` | MDS dimension, *k*-neighbor, path persistence, skeleton reduction |
| Tables | `run_tables.sh` | Writes every paper table to `paper/bian_NeurReps_2026/tables/` |
| Figures | `run_figures.sh` | Renders the appendix figures (needs a MuJoCo GL backend, e.g. `MUJOCO_GL=egl`) |

The command behind every table and figure in the paper is listed in
[`docs/reproducing.md`](docs/reproducing.md).

## Repository layout

```
topo_mimic/
├── extraction/            Takens embedding, discrete-Morse skeleton, DM-cycles and DM-paths
├── experiment/            W2 → MDS → K-means benchmark, Keypoint-MoSeq and Mapper baselines
├── analysis/
│   ├── feature_analysis/      Cluster labeling, DTW, consistency, warping, KID
│   ├── method_analysis/       Information loss, asymmetry, sensitivity sweeps, stationary clips
│   ├── mapper/                Mapper baseline
│   └── scale_sweep/           Baseline hyperparameter sweep and its rescoring
├── figure_generation/     Paper tables and figures
└── labeling/              Auto-labeling and annotation rendering
configs/                   YAML configs, mirroring topo_mimic/
bash_scripts/              Stage drivers
behavior_labels/           Proofread ground-truth labels
demos/                     Illustrative demos
```

The full tree and where every output is written are in
[`docs/codebase_structure.md`](docs/codebase_structure.md).

## Citation

```bibtex
@inproceedings{bian2026shaped,
  title     = {Shaped by What's Missing: Topological Invariance Simplification Discovers Behavioral Transitions},
  author    = {Bian, Kaiwen and Leonardis, Eric J. and Yang, Yuanjia and Zhang, Charles and
               Wang, Yusu and Pereira, Talmo D.},
  booktitle = {Proceedings of the NeurIPS Workshop on Symmetry and Geometry in Neural Representations (NeurReps)},
  series    = {Proceedings of Machine Learning Research},
  year      = {2026}
}
```

## Acknowledgements

This work builds on [MIMIC-MJX](https://mimic-mjx.talmolab.org/)
([`track-mjx`](https://github.com/talmolab/track-mjx),
[`stac-mjx`](https://github.com/talmolab/stac-mjx)) for the biomechanical rat and
its imitation policy, [PCD-Graph-Recon-DM](https://github.com/lucasjmagee/PCD-Graph-Recon-DM)
for discrete-Morse graph reconstruction, and
[Keypoint-MoSeq](https://keypoint-moseq.readthedocs.io/) and
[tda-mapper](https://github.com/lucasimi/tda-mapper-python) for the baselines.

## License

Released under the BSD 3-Clause License; see [`LICENSE`](LICENSE). The
Keypoint-MoSeq dependencies (`keypoint-moseq`, `jax-moseq`) carry Harvard's
non-commercial academic license, and PCD-Graph-Recon-DM is used under its
authors' terms.
