"""Rescore the baseline scale sweep for Table 3 (no refitting; reads cached cluster dicts).

Usage:
    python -m topo_mimic.analysis.scale_sweep.rescore
"""
from . import aligned, directed_turn, rescore_all

if __name__ == "__main__":
    rescore_all.main()
    directed_turn.main()
    aligned.main()
