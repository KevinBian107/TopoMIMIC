# Installation

This project depends on the [Mimic-mjx](https://github.com/talmolab/track-mjx) and [Stac-mjx](https://github.com/talmolab/stac-mjx) ecosystems plus a handful of TDA packages, and requires building two C++ executables for the discrete Morse backend.

## Python environment

Follow the `jax` stack pinned by Mimic-mjx. Install `Stac-mjx` before the TopoVNL extras (KPMS prefers a newer jax than Stac-mjx; both can coexist). Then install the TDA packages:

```bash
pip install -r requirements.txt
```

## Discrete Morse backend (C++)

TopoVNL uses the DM-graph implementation from [PCD-Graph-Recon-DM](https://github.com/lucasjmagee/PCD-Graph-Recon-DM/tree/main). After cloning, build both executables:

```bash
git clone https://github.com/lucasjmagee/PCD-Graph-Recon-DM.git

cd dmpcd/code/persistence_swr
g++ DiMoSC.cpp -I./phat/include -std=c++11 -o DiMoSC

cd dmpcd/code/phat_morse_src
g++ ComputeGraphReconstruction.cpp -std=c++11 -o ComputeGraphReconstruction
```

Verify:

```bash
ls -lh dmpcd/code/persistence_swr/spt_cpp
ls -lh dmpcd/code/phat_morse_src/a.out
```

## Data

The rollouts can be regenerated from Mimic-mjx + Stac-mjx. Pre-generated sets used in the paper are available via gdown:

- [Labeling dataset](https://drive.google.com/file/d/1sYFkKQGJ0qTJGm0fEigXRn4dKGzGv7qQ/view?usp=drive_link) (tighter KL bound, better visualization)
- [Clustering dataset](https://drive.google.com/file/d/1exMwbJbJAqhJdKvT6ez9POY-bFGYbemN/view?usp=sharing) (looser KL bound, easier clustering)

Render videos for annotation:

```bash
python -m topo_mimic.labeling.render_for_label configs/labeling/render_for_label.yaml
```

Pre-rendered videos are also [available here](https://drive.google.com/drive/folders/1tCw60UmBeP8hPDwPVnugsva0yIyO_mFU?usp=sharing). The auto-labeling annotation framework lives at [TopoMimic-annotation](https://github.com/KevinBian107/TopoMimic-annotation); it uses the [behavior lexicon](https://docs.google.com/document/d/19UC5Mu5mj_zmh2jDSHkML7WvMbdC4DbXrow3f1WBxC4/edit?usp=sharing).
