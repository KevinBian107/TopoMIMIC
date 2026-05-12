"""Information-loss-under-coarse-graining experiment.

Per-frame mutual information between each method's pre-W2 primitive
assignment (T) and the GT directed-transition label (Y), reported on the
Information Bottleneck plane.

Module layout:
    labels.py     -- per-frame Y construction from rodent_labels.csv
    primitives.py -- per-method T extraction (DM, KPMS, VAE, Mapper)
    metrics.py    -- I(T;Y), NMI, AMI, H(T), per-clip bootstrap, shuffle test
    plotting.py   -- IB plane (main), bar chart + joint heatmaps (appendix)
    tables.py     -- CSV + LaTeX-rendered table
    main.py       -- orchestrator entry point
"""
