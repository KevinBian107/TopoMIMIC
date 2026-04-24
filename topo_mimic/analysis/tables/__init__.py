"""
Unified table-generation package.

Three logically distinct table families consolidated under one package:
  - ``results_table`` - main Nature-style summary tables from analysis CSVs
  - ``paper_tables``  - Figure 4 summary, Appendix DTW, Appendix consistency
  - ``warping_tables`` - DTW warping-heaviness LaTeX tables (Appendix)

Run via:
    python -m topo_mimic.analysis.tables                  # generate all
    python -m topo_mimic.analysis.tables paper            # only paper tables
    python -m topo_mimic.analysis.tables warping          # only warping tables
    python -m topo_mimic.analysis.tables results          # only results table
"""
