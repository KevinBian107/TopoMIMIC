# Installation

This project depends on the [Mimic-mjx](https://github.com/talmolab/track-mjx) and [Stac-mjx](https://github.com/talmolab/stac-mjx) ecosystems plus the analysis packages in `requirements.txt`, and requires building two C++ executables for the discrete Morse backend.

## Python environment

Follow the `jax` stack pinned by Mimic-mjx. Install `Stac-mjx` before the TopoMIMIC extras (KPMS prefers a newer jax than Stac-mjx; both can coexist). Then install the TDA packages:

```bash
pip install -r requirements.txt
```

## Discrete Morse backend (C++)

TopoMIMIC uses the DM-graph reference implementation [PCD-Graph-Recon-DM](https://github.com/lucasjmagee/PCD-Graph-Recon-DM) (tested at commit `711c58e`). Clone it into the repository root and build the two executables the pipeline calls:

```bash
git clone https://github.com/lucasjmagee/PCD-Graph-Recon-DM.git
cd PCD-Graph-Recon-DM/dmpcd/code/persistence_swr && g++ DiMoSC.cpp -I./phat/include -std=c++11 -o spt_cpp
cd ../phat_morse_src && g++ ComputeGraphReconstruction.cpp
cd ../../../..
```

Verify:

```bash
ls -lh PCD-Graph-Recon-DM/dmpcd/code/persistence_swr/spt_cpp
ls -lh PCD-Graph-Recon-DM/dmpcd/code/phat_morse_src/a.out
```

To keep the backend elsewhere, set `DMPCD_ROOT=/path/to/PCD-Graph-Recon-DM`.

End-to-end check: reconstruct the skeleton of a noisy circle, which should report one loop.

```bash
python -c "
import numpy as np
from topo_mimic.extraction.skeletonization.skeletonization import extract_skeleton_pcd_dm, extract_all_cycles
t = np.linspace(0, 2 * np.pi, 300)
pts = np.c_[np.cos(t), np.sin(t)] + 0.03 * np.random.default_rng(0).normal(size=(300, 2))
_, G = extract_skeleton_pcd_dm(pts, k=5, pers_delta=0.1)
print('loops:', len(extract_all_cycles(G)))
"
```

## Rendering

Figures and videos render MuJoCo scenes. On GPU nodes use `MUJOCO_GL=egl`; this needs the GLVND loader `libEGL.so.1` (Ubuntu package `libegl1`) in addition to the NVIDIA driver. On CPU-only machines use `MUJOCO_GL=osmesa` with Mesa's `libOSMesa`.

## Data

The rollouts can be regenerated from Mimic-mjx + Stac-mjx. Pre-generated sets are available on Google Drive; the paper uses the clustering dataset, saved as `data/rodent_ar.h5`:

- [Labeling dataset](https://drive.google.com/file/d/1sYFkKQGJ0qTJGm0fEigXRn4dKGzGv7qQ/view?usp=drive_link) (tighter KL bound, better visualization)
- [Clustering dataset](https://drive.google.com/file/d/1exMwbJbJAqhJdKvT6ez9POY-bFGYbemN/view?usp=sharing) (looser KL bound, easier clustering)

Render videos for annotation:

```bash
python -m topo_mimic.labeling.render_for_label configs/labeling/render_for_label.yaml
```

Pre-rendered videos are also [available here](https://drive.google.com/drive/folders/1tCw60UmBeP8hPDwPVnugsva0yIyO_mFU?usp=sharing). The auto-labeling annotation framework lives at [TopoMimic-annotation](https://github.com/KevinBian107/TopoMimic-annotation); it uses the [behavior lexicon](https://docs.google.com/document/d/19UC5Mu5mj_zmh2jDSHkML7WvMbdC4DbXrow3f1WBxC4/edit?usp=sharing).
