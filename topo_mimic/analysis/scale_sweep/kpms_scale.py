"""KPMS scale sweep: kappa x num_states over a chosen modality (keypoint or
intention), EML + usage-ratio selector.

Per kappa, multiple num_states fits share a project_dir so that
`kpms.expected_marginal_likelihoods(project_dir, model_names)` can compare them.
Model selection per kappa picks the num_states with the highest EML among fits that
meet the `min_usage_ratio` floor. If no fit meets the floor, falls back to the
highest-usage-ratio fit and logs a warning.

Absolute clip_ids are preserved when converting KPMS-internal `recording_XXXX`
keys back to cluster_dict entries — the i-th entry of `sorted(rec_names)` maps
to `clip_ids[i]` (per Data Flow Tracer audit 2026-04-18).
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from topo_mimic.hyperparam_sweep.kpms.config import KPMSHyperparams
from topo_mimic.hyperparam_sweep.kpms.dual_fit import (
    KPMSFitResult,
    fit_kpms_intention,
    fit_kpms_keypoints,
)
from topo_mimic.hyperparam_sweep.metrics import (
    compute_active_syllables,
    compute_duration_stats,
    compute_eml,
    compute_syllable_usage_ratio,
    compute_transition_entropy,
)
from topo_mimic.analysis.scale_sweep.config import ScaleSweepConfig

log = logging.getLogger(__name__)


@dataclass
class KPMSVariantResult:
    variant_id: str
    kappa: float
    num_states: int
    seed: int
    eml: float
    usage_ratio: float
    active_syllables: int
    duration_mean: float
    duration_median: float
    transition_entropy: float
    model_name: str
    project_dir: str
    metrics_path: str
    cluster_dict_path: str
    labels_list_path: str

    @property
    def column_name(self) -> str:
        return self.variant_id


@dataclass
class KPMSSelection:
    kappa: float
    selected_num_states: int
    selected_variant_id: str
    selector_used: str   # 'eml' or 'usage_fallback'
    eml_values: dict[int, float]
    usage_ratios: dict[int, float]


def _kappa_tag(kappa: float) -> str:
    """Short scientific-notation tag for dir naming: 100000.0 -> '1e5'."""
    exponent = int(np.round(np.log10(kappa)))
    return f"1e{exponent}"


def _variant_id(kappa: float, num_states: int, modality: str = "keypoint") -> str:
    """Variant id, tagged by modality so intention and keypoint columns stay distinct
    in the discovery tables."""
    tag = {"keypoint": "kp", "intention": "int"}[modality]
    return f"kpms_{tag}_k{_kappa_tag(kappa)}_n{num_states}"


def _kpms_fit_result_to_cluster_dict(
    fit_result: KPMSFitResult,
    clip_ids: list[int],
) -> tuple[dict[int, list[tuple[int, np.ndarray]]], list[np.ndarray]]:
    """Convert KPMS per-recording syllable labels to a cluster_dict keyed by absolute clip_id.

    `fit_result.labels_list[i]` is guaranteed to correspond to `clip_ids[i]` because
    `dual_fit.py` sorts `rec_names` lexicographically, which equals numerical order for
    zero-padded `recording_{i:04d}` names — and prepare_keypoints_for_kpms enumerates
    the input in order.
    """
    labels_list = fit_result.labels_list
    if len(labels_list) != len(clip_ids):
        raise RuntimeError(
            f"KPMS returned {len(labels_list)} recordings but we passed {len(clip_ids)} clip_ids"
        )

    syllables_seen: set[int] = set()
    for per_clip in labels_list:
        syllables_seen.update(np.unique(per_clip).tolist())

    cluster_dict: dict[int, list[tuple[int, np.ndarray]]] = {}
    for sid in sorted(syllables_seen):
        members: list[tuple[int, np.ndarray]] = []
        for i, per_clip in enumerate(labels_list):
            frames = np.where(np.asarray(per_clip) == sid)[0].astype(np.int64)
            if frames.size == 0:
                continue
            members.append((clip_ids[i], frames))
        if members:
            cluster_dict[int(sid)] = members
    return cluster_dict, labels_list


def _save_variant(
    variant_out_dir: Path,
    fit_result: KPMSFitResult,
    clip_ids: list[int],
    variant_id: str,
    kappa: float,
    num_states: int,
    seed: int,
) -> tuple[dict[str, Any], dict[int, list[tuple[int, np.ndarray]]]]:
    variant_out_dir.mkdir(parents=True, exist_ok=True)

    cluster_dict, labels_list = _kpms_fit_result_to_cluster_dict(fit_result, clip_ids)
    with (variant_out_dir / "cluster_dict.pkl").open("wb") as f:
        pickle.dump(cluster_dict, f)
    with (variant_out_dir / "labels_list.pkl").open("wb") as f:
        pickle.dump(labels_list, f)

    duration = compute_duration_stats(fit_result.results)
    usage_ratio = compute_syllable_usage_ratio(fit_result.results, num_states)
    active_syllables = compute_active_syllables(fit_result.results)
    transition_entropy = compute_transition_entropy(fit_result.model)

    metrics = {
        "variant_id": variant_id,
        "kappa": float(kappa),
        "num_states": int(num_states),
        "seed": int(seed),
        "model_name": fit_result.model_name,
        "project_dir": fit_result.project_dir,
        "usage_ratio": float(usage_ratio),
        "active_syllables": int(active_syllables),
        "duration_mean": float(duration.mean),
        "duration_median": float(duration.median),
        "duration_std": float(duration.std),
        "transition_entropy": float(transition_entropy),
        "n_clusters_extracted": int(len(cluster_dict)),
    }
    (variant_out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics, cluster_dict


def run_kpms_sweep(
    keypoints: np.ndarray,
    clip_ids: list[int],
    cfg: ScaleSweepConfig,
    output_root: Path,
    force: bool = False,
    max_fits: int | None = None,
    modality: str = "keypoint",
) -> tuple[list[KPMSVariantResult], dict[float, KPMSSelection]]:
    """Sweep kappa x num_states; per kappa compute EML across num_states models and select best.

    Args:
        keypoints: FK-derived keypoints, shape (N, T, K, 3).
        clip_ids: absolute clip IDs, aligned with keypoints[0..N-1].
        cfg: full scale-sweep config; uses cfg.kpms and cfg.io.
        output_root: output/scale_sweep/v1_{hash}/kpms/
        force: rerun even if metrics.json exists.
        max_fits: cap total fits (smoke testing).

    Returns:
        results: all variant results.
        selections: kappa -> KPMSSelection with the best num_states per kappa.
    """
    results: list[KPMSVariantResult] = []
    selections: dict[float, KPMSSelection] = {}
    fits_remaining = max_fits if max_fits is not None else 10_000

    seed = cfg.kpms.seeds[0]

    for kappa in cfg.kpms.kappa_grid:
        kappa_dir = output_root / f"kappa_{_kappa_tag(kappa)}"
        kappa_dir.mkdir(parents=True, exist_ok=True)

        kpms_project_dir = kappa_dir / "kpms_project"
        model_names: list[str] = []
        variant_results_for_kappa: dict[int, KPMSVariantResult] = {}

        for idx, num_states in enumerate(cfg.kpms.num_states_grid):
            if fits_remaining <= 0:
                log.info(f"[KPMS] max_fits reached; stopping at κ={kappa} n_states={num_states}")
                break

            variant_id = _variant_id(kappa, num_states, modality)
            variant_out_dir = kappa_dir / f"states_{num_states}" / f"seed_{seed}"
            metrics_path = variant_out_dir / "metrics.json"

            if not force and metrics_path.exists():
                log.info(f"[KPMS] skip {variant_id} (metrics.json exists)")
                with metrics_path.open() as f:
                    m = json.load(f)
                vr = KPMSVariantResult(
                    variant_id=variant_id,
                    kappa=kappa,
                    num_states=num_states,
                    seed=seed,
                    eml=float("nan"),   # filled in after the sweep when we read eml_by_kappa
                    usage_ratio=float(m["usage_ratio"]),
                    active_syllables=int(m["active_syllables"]),
                    duration_mean=float(m["duration_mean"]),
                    duration_median=float(m["duration_median"]),
                    transition_entropy=float(m["transition_entropy"]),
                    model_name=m["model_name"],
                    project_dir=m["project_dir"],
                    metrics_path=str(metrics_path),
                    cluster_dict_path=str(variant_out_dir / "cluster_dict.pkl"),
                    labels_list_path=str(variant_out_dir / "labels_list.pkl"),
                )
                variant_results_for_kappa[num_states] = vr
                results.append(vr)
                model_names.append(m["model_name"])
                continue

            log.info(f"[KPMS] fitting {variant_id}  (fit {max_fits - fits_remaining + 1 if max_fits else '?'})")
            hp = KPMSHyperparams(
                kappa=float(kappa),
                latent_dim=cfg.kpms.latent_dim,
                num_states=num_states,
                ar_iters=cfg.kpms.ar_iters,
                full_iters=cfg.kpms.full_iters,
                model_type=cfg.kpms.model_type,
            )
            # Workaround for dual_fit.py::fit_kpms_model bug (Issue 3 from JAX auditor):
            # when clear_project=False, fit_kpms_model skips kpms.update_config, leaving
            # num_states/kappa stale in the project's config.yml. Refresh it manually.
            if idx > 0:
                import keypoint_moseq as kpms

                kpms.update_config(
                    str(kpms_project_dir),
                    num_states=num_states,
                    kappa=float(kappa),
                )
            fit_fn = fit_kpms_keypoints if modality == "keypoint" else fit_kpms_intention
            data_kw = "keypoint_data" if modality == "keypoint" else "intention_data"
            fit_result = fit_fn(**{data_kw: keypoints},
                n_states=num_states,
                project_dir=str(kpms_project_dir),
                hyperparams=hp,
                seed=seed,
                clear_project=(idx == 0),
            )
            metrics, _cluster_dict = _save_variant(
                variant_out_dir=variant_out_dir,
                fit_result=fit_result,
                clip_ids=clip_ids,
                variant_id=variant_id,
                kappa=kappa,
                num_states=num_states,
                seed=seed,
            )
            model_names.append(fit_result.model_name)

            vr = KPMSVariantResult(
                variant_id=variant_id,
                kappa=kappa,
                num_states=num_states,
                seed=seed,
                eml=float("nan"),
                usage_ratio=float(metrics["usage_ratio"]),
                active_syllables=int(metrics["active_syllables"]),
                duration_mean=float(metrics["duration_mean"]),
                duration_median=float(metrics["duration_median"]),
                transition_entropy=float(metrics["transition_entropy"]),
                model_name=fit_result.model_name,
                project_dir=fit_result.project_dir,
                metrics_path=str(metrics_path),
                cluster_dict_path=str(variant_out_dir / "cluster_dict.pkl"),
                labels_list_path=str(variant_out_dir / "labels_list.pkl"),
            )
            variant_results_for_kappa[num_states] = vr
            results.append(vr)
            fits_remaining -= 1

        # Release JAX compilation caches between kappa blocks to avoid cache growth
        # across 25 fits (Issue 1 from JAX auditor).
        try:
            import jax

            jax.clear_caches()
        except Exception:
            pass

        if len(model_names) >= 2:
            try:
                eml_result = compute_eml(str(kpms_project_dir), model_names)
                # map model_name -> eml score
                name_to_eml = dict(zip(model_names, eml_result.scores.tolist()))
                for ns, vr in variant_results_for_kappa.items():
                    vr.eml = float(name_to_eml.get(vr.model_name, float("nan")))
                    # patch the metrics.json with the EML
                    with Path(vr.metrics_path).open() as f:
                        m = json.load(f)
                    m["eml"] = vr.eml
                    Path(vr.metrics_path).write_text(json.dumps(m, indent=2))
            except Exception as e:
                log.warning(f"[KPMS] EML computation failed for κ={kappa}: {e}")
        else:
            log.warning(f"[KPMS] κ={kappa} has <2 fits; skipping EML computation")

        selection = _select_best_num_states(
            kappa=kappa,
            variants=variant_results_for_kappa,
            min_usage_ratio=cfg.kpms.selection.min_usage_ratio,
        )
        selections[kappa] = selection
        log.info(
            f"[KPMS] κ={kappa} selection: n_states={selection.selected_num_states} "
            f"via {selection.selector_used}"
        )

    selection_path = output_root / "selection.yaml"
    selection_path.write_text(yaml.safe_dump({
        str(k): {
            "kappa": float(s.kappa),
            "selected_num_states": int(s.selected_num_states),
            "selected_variant_id": s.selected_variant_id,
            "selector_used": s.selector_used,
            "eml_values": {int(n): float(e) for n, e in s.eml_values.items()},
            "usage_ratios": {int(n): float(u) for n, u in s.usage_ratios.items()},
        }
        for k, s in selections.items()
    }, sort_keys=False))
    log.info(f"[KPMS] selection written to {selection_path}")
    return results, selections


def _select_best_num_states(
    kappa: float,
    variants: dict[int, KPMSVariantResult],
    min_usage_ratio: float,
) -> KPMSSelection:
    """EML peak with usage_ratio floor; fallback to highest usage_ratio if no eligible."""
    eml_values = {ns: v.eml for ns, v in variants.items()}
    usage_ratios = {ns: v.usage_ratio for ns, v in variants.items()}

    eligible = {
        ns: v for ns, v in variants.items()
        if v.usage_ratio >= min_usage_ratio and not np.isnan(v.eml)
    }

    if eligible:
        best_ns = max(eligible, key=lambda ns: eligible[ns].eml)
        selector = "eml"
    else:
        log.warning(
            f"[KPMS] κ={kappa}: no variant with usage_ratio >= {min_usage_ratio} and valid EML; "
            f"falling back to highest-usage variant"
        )
        best_ns = max(variants, key=lambda ns: variants[ns].usage_ratio)
        selector = "usage_fallback"

    return KPMSSelection(
        kappa=float(kappa),
        selected_num_states=int(best_ns),
        selected_variant_id=variants[best_ns].variant_id,
        selector_used=selector,
        eml_values=eml_values,
        usage_ratios=usage_ratios,
    )
