# Information Loss / IB-Plane Analysis

Quantifies how much GT-transition information each method's coarse-graining
preserves, as a single mutual-information number $I(T;\,Y)$ in bits.

- $T_t$ = the method's per-frame coarsening output (syllable / node /
  cluster id for KPMS / Mapper / VAE; per-instance cycle / path id for DM).
  `-1` denotes background (frame not in any primitive of the method).
- $Y_t$ = the GT directed transition class at frame $t$, derived from
  `behavior_labels/rodent_labels.csv` via
  `topo_mimic.analysis.feature_analysis.ground_truth.find_all_transitions`.

The headline visualisation is the **Information Bottleneck plane**:
$H(T)$ vs $I(T;\,Y)$, both in bits. A coarsening that lands upper-left of
the diagonal extracts more transition information per bit of representation
capacity than one in the lower-right.

## Run

```bash
python -m topo_mimic.analysis.method_analysis.information_loss.main \
    configs/analysis/method_analysis/information_loss/information_loss.yaml
```

This stage is also chained as the final step of `bash bash_scripts/run_method_analysis.sh`.

The orchestrator validates its MI estimator with a synthetic
independent-/-perfect-info / capacity-clamp sanity check at startup.

## Outputs

```
outputs/analysis/method_analysis/information_loss/
├── data/
│   ├── per_method_mi.csv               # one row per (method, seed)
│   ├── per_method_mi.npz                # bootstrap arrays per metric
│   └── joint_distributions/             # (T, Y) cross-tabs for heatmaps
└── figures/
    ├── ib_plane.{png,pdf}               # PRIMARY -- H(T) vs I(T;Y) scatter
    ├── mi_bar.{png,pdf}                  # appendix -- bar chart of I,NMI,AMI
    ├── joint_heatmaps.{png,pdf}          # appendix -- p(Y|T) heatmaps
    ├── information_loss_table.tex        # paper-ready LaTeX
    └── information_loss_table.png        # PNG render of the same
```

The LaTeX file is written to `outputs/` only; the paper is left untouched.

## Module map

| File           | Role                                                  |
|----------------|-------------------------------------------------------|
| `labels.py`    | Per-frame Y from `find_all_transitions`              |
| `primitives.py`| Per-frame T extraction (DM, KPMS, VAE, Mapper)        |
| `metrics.py`   | I(T;Y), NMI, AMI, H(T), bootstrap, shuffle floor     |
| `plotting.py`  | IB plane, bar chart, joint heatmaps                  |
| `tables.py`    | CSV / LaTeX / PNG main results table                 |
| `main.py`      | Orchestrator entry point                             |

## Notes for future tweaks

- `y_granularity` in the config controls Y cardinality:
  `"unordered"` / `"directed"` / `"directed_turn"`. Default is `"directed"`
  (matches the paper's 8-class transition classifier label space).
- `primitive_level` in each method block toggles per-frame-syllable
  (`"type"`, default) vs per-sustained-window instance ID
  (`"instance"`, more granular, captures the W2-feeding primitive). Use
  `"instance"` to compare like-with-like against DM cycles.
- The CSV column `I_TY_shuffled_bits` is the within-clip-shuffle
  permutation floor; subtracting it gives the *lift* (`lift_over_floor_bits`)
  used in the table.
