"""Scale-sweep module.

KPMS (kappa x num_states) on keypoints plus Mapper (lens x n_intervals x feature_space)
producing a transition-discovery table. Orthogonal to the main benchmark.
"""

from topo_mimic.benchmark.scale_sweep.config import ScaleSweepConfig, load_config

__all__ = ["ScaleSweepConfig", "load_config"]
