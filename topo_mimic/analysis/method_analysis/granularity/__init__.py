"""Granularity analysis module for examining persistence threshold effects."""

from .config import GranularityConfig
from .dm_multi_persistence import run_dm_multi_persistence
from .stationary_analysis import run_stationary_analysis
from .redundancy import run_redundancy_analysis
from .rendering import render_all_similar_pairs
from .plotting import generate_stationary_plots, generate_redundancy_plots

__all__ = [
    "GranularityConfig",
    "run_dm_multi_persistence",
    "run_stationary_analysis",
    "run_redundancy_analysis",
    "render_all_similar_pairs",
    "generate_stationary_plots",
    "generate_redundancy_plots",
]
