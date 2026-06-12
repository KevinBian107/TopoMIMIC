"""kNN-`k` sensitivity sweep for the DM pipeline.

A SEPARATE experiment that reuses the EXACT main pipeline (extraction ->
experiment -> feature-analysis) by overriding only `dm.k_neighbors` and the
output/path keys in copies of the production YAMLs, then subprocessing the
unchanged production modules. Nothing in the main pipeline is modified.
"""
