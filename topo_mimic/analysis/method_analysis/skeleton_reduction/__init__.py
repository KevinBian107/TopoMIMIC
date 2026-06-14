"""Trajectory-to-skeleton reduction summary.

Reads the cached DM-skeletons (``all_G_sm.pkl``) and extracted primitives
(``all_cycle_dm.pkl``, ``all_path_dm.pkl``) for each feature space and reports
the size of a trajectory at each pipeline stage: original frames -> skeleton
vertices -> DM-cycle / DM-path length (frames). Emits a CSV and the paper
appendix table (``paper/tables/appendix_skeleton_reduction.tex``).
"""
from .config import ReductionConfig, load_config

__all__ = ["ReductionConfig", "load_config"]
