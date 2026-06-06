"""Information-loss orchestrator.

Reads the global config, walks every (method, feature_space, seed) entry,
extracts T via ``primitives.py``, builds Y once via ``labels.py``, runs the
MI suite in ``metrics.py``, writes the CSV+NPZ data products, then defers
figure rendering to ``plotting.py`` and ``tables.py``.

Usage:
    python -m topo_mimic.analysis.method_analysis.information_loss.main \\
        configs/analysis/method_analysis/information_loss/information_loss.yaml
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from topo_mimic.analysis.method_analysis.information_loss.labels import (
    Granularity,
    TransitionLabels,
    load_transition_labels,
    summarize_labels,
)
from topo_mimic.analysis.method_analysis.information_loss.metrics import (
    MIResult,
    compute_mi,
    joint_distribution,
    run_sanity_checks,
)
from topo_mimic.analysis.method_analysis.information_loss.primitives import (
    PrimitiveAssignment,
    load_dm_primitives,
    load_full_primitives,
    load_kpms_primitives,
    load_mapper_primitives,
    load_per_frame_labels_primitives,
    load_vae_primitives,
    summarize_assignment,
)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MethodSpec:
    name: str
    kind: str  # one of: "dm_cycle", "dm_path", "kpms", "vae", "mapper"
    feature_space: str | None = None
    seeds: list[int] = field(default_factory=list)
    # DM-only
    extraction_dir: str | None = None
    # KPMS/VAE/Mapper
    results_pkl: str | None = None
    kpms_variant: str | None = None  # "intention" / "keypoint"
    primitive_level: str = "type"  # "type" or "instance"


@dataclass
class InformationLossConfig:
    output_dir: str
    dataset: str = "rodent"
    n_clips: int = 842
    frames_per_clip: int = 500
    labels_csv: str = "behavior_labels/rodent_labels.csv"
    y_granularity: Granularity = "directed"
    min_class_count: int = 5
    n_bootstrap: int = 1000
    rng_seed: int = 0
    methods: list[MethodSpec] = field(default_factory=list)


def _parse_method(d: dict) -> MethodSpec:
    return MethodSpec(
        name=d["name"],
        kind=d["kind"],
        feature_space=d.get("feature_space"),
        seeds=list(d.get("seeds", [0])) or [0],
        extraction_dir=d.get("extraction_dir"),
        results_pkl=d.get("results_pkl"),
        kpms_variant=d.get("kpms_variant"),
        primitive_level=d.get("primitive_level", "type"),
    )


def load_config(path: str) -> InformationLossConfig:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    methods = [_parse_method(m) for m in cfg.get("methods", [])]
    return InformationLossConfig(
        output_dir=cfg["output_dir"],
        dataset=cfg.get("dataset", "rodent"),
        n_clips=int(cfg.get("n_clips", 842)),
        frames_per_clip=int(cfg.get("frames_per_clip", 500)),
        labels_csv=cfg.get("labels_csv", "behavior_labels/rodent_labels.csv"),
        y_granularity=cfg.get("y_granularity", "directed"),
        min_class_count=int(cfg.get("min_class_count", 5)),
        n_bootstrap=int(cfg.get("n_bootstrap", 1000)),
        rng_seed=int(cfg.get("rng_seed", 0)),
        methods=methods,
    )


# ---------------------------------------------------------------------------
# Loader dispatch
# ---------------------------------------------------------------------------


def _load_for_spec(
    spec: MethodSpec,
    seed: int,
    n_clips: int,
    frames_per_clip: int,
) -> PrimitiveAssignment:
    k = spec.kind
    if k in ("dm_cycle", "dm_path"):
        if spec.extraction_dir is None or spec.feature_space is None:
            raise ValueError(f"{spec.name}: dm methods need extraction_dir + feature_space")
        return load_dm_primitives(
            spec.extraction_dir,
            "cycle" if k == "dm_cycle" else "path",
            spec.feature_space,
            n_clips,
            frames_per_clip,
        )
    if spec.results_pkl is None:
        raise ValueError(f"{spec.name}: baseline methods need results_pkl")
    if k == "kpms":
        if spec.kpms_variant is None:
            raise ValueError(f"{spec.name}: kpms requires kpms_variant")
        return load_kpms_primitives(
            spec.results_pkl, seed, spec.kpms_variant,
            n_clips, frames_per_clip,
            primitive_level=spec.primitive_level,
        )
    if k == "vae":
        return load_vae_primitives(
            spec.results_pkl, seed,
            n_clips, frames_per_clip,
            primitive_level=spec.primitive_level,
        )
    if k == "mapper":
        if spec.feature_space is None:
            raise ValueError(f"{spec.name}: mapper requires feature_space")
        return load_mapper_primitives(
            spec.results_pkl, seed, spec.feature_space,
            n_clips, frames_per_clip,
            primitive_level=spec.primitive_level,
        )
    if k == "full":
        return load_full_primitives(
            spec.results_pkl, seed,
            n_clips, frames_per_clip,
        )
    raise ValueError(f"Unknown method kind {k!r} for {spec.name!r}")


# ---------------------------------------------------------------------------
# Results -> CSV/NPZ marshalling
# ---------------------------------------------------------------------------


def _result_to_row(r: MIResult, granularity: Granularity) -> dict[str, Any]:
    return {
        "method": r.method_name,
        "family": r.family,
        "feature_space": r.feature_space or "",
        "seed": -1 if r.seed is None else int(r.seed),
        "n_clips_labelled": r.n_clips,
        "n_frames_labelled": r.n_frames,
        "n_primitives": r.n_primitives,
        "H_T_bits": r.H_T,
        "H_Y_bits": r.H_Y,
        "I_TY_bits": r.I_TY,
        "H_Y_given_T_bits": r.H_Y_given_T,
        "NMI": r.NMI,
        "AMI": r.AMI,
        "efficiency": r.efficiency,
        "I_TY_shuffled_bits": r.I_TY_shuffled,
        "lift_over_floor_bits": r.I_TY - r.I_TY_shuffled,
        "I_TY_ci_lo": r.ci_lo.get("I_TY", 0.0),
        "I_TY_ci_hi": r.ci_hi.get("I_TY", 0.0),
        "NMI_ci_lo": r.ci_lo.get("NMI", 0.0),
        "NMI_ci_hi": r.ci_hi.get("NMI", 0.0),
        "AMI_ci_lo": r.ci_lo.get("AMI", 0.0),
        "AMI_ci_hi": r.ci_hi.get("AMI", 0.0),
        "Y_granularity": granularity,
    }


def _save_bootstrap_npz(results: list[MIResult], path: Path) -> None:
    if not results:
        return
    meta = np.array(
        [
            (
                r.method_name,
                r.family,
                r.feature_space or "",
                -1 if r.seed is None else r.seed,
            )
            for r in results
        ],
        dtype=[
            ("method", "U32"),
            ("family", "U8"),
            ("feature_space", "U16"),
            ("seed", "i4"),
        ],
    )
    payload: dict[str, np.ndarray] = {"metadata": meta}
    for key in ("H_T", "I_TY", "NMI", "AMI", "H_Y_given_T", "efficiency"):
        stacked = np.stack([r.bootstrap.get(key, np.array([])) for r in results], axis=0)
        payload[f"bootstrap_{key}"] = stacked
    np.savez_compressed(path, **payload)


# ---------------------------------------------------------------------------
# Orchestrator entry point
# ---------------------------------------------------------------------------


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    out_dir = Path(cfg.output_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / "joint_distributions").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Information Loss / IB-plane analysis")
    print("=" * 70)

    print("\n[0/3] sanity-checking the MI estimator on synthetic input")
    run_sanity_checks(verbose=True)

    print(f"\n[1/3] building Y at granularity={cfg.y_granularity!r}")
    labels = load_transition_labels(
        csv_path=cfg.labels_csv,
        n_clips=cfg.n_clips,
        frames_per_clip=cfg.frames_per_clip,
        granularity=cfg.y_granularity,
        min_class_count=cfg.min_class_count,
    )
    print(summarize_labels(labels))

    print(f"\n[2/3] computing MI for {len(cfg.methods)} method specs")
    rows: list[dict] = []
    all_results: list[MIResult] = []

    t0 = time.time()
    for i, spec in enumerate(cfg.methods, start=1):
        seeds = spec.seeds if spec.seeds else [None]
        for seed in seeds:
            seed_int = 0 if seed is None else int(seed)
            tag = f"[{i:>2}/{len(cfg.methods)}] {spec.name} seed={seed}"
            print(f"\n{tag}")
            try:
                assignment = _load_for_spec(spec, seed_int, cfg.n_clips, cfg.frames_per_clip)
            except (FileNotFoundError, KeyError, ValueError) as e:
                print(f"  SKIP: {e}")
                continue

            print(summarize_assignment(assignment))

            method_rng_seed = cfg.rng_seed + i * 1000 + seed_int
            result = compute_mi(
                T=assignment.T,
                Y=labels.Y,
                method_name=assignment.method_name,
                family=assignment.family,
                feature_space=assignment.feature_space,
                seed=assignment.seed,
                n_bootstrap=cfg.n_bootstrap,
                rng_seed=method_rng_seed,
            )
            print(
                f"  H(T)={result.H_T:.3f}  I(T;Y)={result.I_TY:.3f}  "
                f"NMI={result.NMI:.3f}  AMI={result.AMI:.3f}  "
                f"eff={result.efficiency:.3f}  floor={result.I_TY_shuffled:.3f}"
            )

            rows.append(_result_to_row(result, cfg.y_granularity))
            all_results.append(result)

            joint, T_vals, Y_vals = joint_distribution(assignment.T, labels.Y)
            joint_path = out_dir / "data" / "joint_distributions" / (
                f"{assignment.method_name.replace('-', '_').lower()}"
                f"_seed{seed_int}.npz"
            )
            np.savez_compressed(
                joint_path,
                joint=joint,
                T_values=T_vals,
                Y_values=Y_vals,
                class_names=np.array(labels.class_names, dtype=object),
            )

    csv_path = out_dir / "data" / "per_method_mi.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    npz_path = out_dir / "data" / "per_method_mi.npz"
    _save_bootstrap_npz(all_results, npz_path)
    print(f"Wrote {npz_path}")

    elapsed = time.time() - t0
    print(f"\n[3/3] CSV+NPZ done in {elapsed:.1f}s; render figures via plotting.py")

    try:
        from topo_mimic.analysis.method_analysis.information_loss import plotting, tables
    except ImportError:
        print("(plotting/tables not yet importable; rerun after their modules land)")
        return

    print("\nRendering figures...")
    plotting.render_all(out_dir=out_dir, results=all_results, labels=labels)
    tables.render_all(out_dir=out_dir, results=all_results, granularity=cfg.y_granularity)
    print(f"\nFigures written under {out_dir/'figures'}")


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        raise SystemExit("usage: python -m topo_mimic.analysis.method_analysis.information_loss.main <config.yaml>")
    run(args[0])


if __name__ == "__main__":
    main()
