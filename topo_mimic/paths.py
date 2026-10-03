"""Repository-relative locations shared by the pipeline."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# C++/Python discrete-Morse backend (github.com/lucasjmagee/PCD-Graph-Recon-DM); its
# compiled binaries are invoked by relative path, so callers chdir into this directory.
DMPCD_ROOT = Path(os.environ.get("DMPCD_ROOT", REPO_ROOT / "PCD-Graph-Recon-DM")).resolve()
