"""Sweep of the path-SELECTION persistence threshold (`dm.path_persistence_delta`
in `extract_persistence_paths`), distinct from the DM-construction `pers_delta`.

Because this threshold only filters branches of the already-built skeleton (a
nested filtration), the sweep reuses the cached DM graph + filtration and
regenerates paths with the exact production code at each delta, with NO C++
re-extraction. Headline metric: transition information I(T;Y) / AMI of the DMP
coarse-graining vs delta (reuses the information_loss machinery).
"""
