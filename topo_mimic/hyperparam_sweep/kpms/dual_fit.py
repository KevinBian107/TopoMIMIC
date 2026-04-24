"""Dual KPMS fitting for intention and keypoint modalities."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import jax
    import keypoint_moseq as kpms
    jax.config.update("jax_enable_x64", True)
    HAS_KPMS = True
except ImportError:
    HAS_KPMS = False

from topo_mimic.hyperparam_sweep.kpms.config import KPMSHyperparams
from topo_mimic.hyperparam_sweep.keypoint_loader import (
    prepare_intention_for_kpms,
    prepare_keypoints_for_kpms,
)
from topo_mimic.hyperparam_sweep.metrics import (
    compute_eml,
    compute_syllable_consistency,
    compute_duration_kl,
    compute_duration_stats,
    compute_active_syllables,
    compute_transition_entropy,
    EMLResult,
)


@dataclass
class KPMSFitResult:
    model: dict
    results: dict
    data: dict
    metadata: tuple
    model_name: str
    project_dir: str
    labels_list: list[np.ndarray]


@dataclass
class DualFitResult:
    intention_result: KPMSFitResult
    keypoint_result: KPMSFitResult
    syllable_ari: float
    syllable_nmi: float
    duration_kl: float
    eml_intention: EMLResult | None
    eml_keypoint: EMLResult | None


def fit_kpms_model(
    coordinates: dict[str, np.ndarray],
    confidences: dict[str, np.ndarray],
    n_states: int,
    project_dir: str,
    hyperparams: KPMSHyperparams,
    seed: int,
    modality_name: str,
    bodyparts: list[str] | None = None,
    fix_heading: bool = False,
    clear_project: bool = True,
) -> KPMSFitResult:
    if not HAS_KPMS:
        raise ImportError("keypoint_moseq not installed")

    project_path = Path(project_dir)
    if clear_project and project_path.exists():
        shutil.rmtree(project_path)
    project_path.mkdir(parents=True, exist_ok=True)

    first_key = next(iter(coordinates))
    first_coords = coordinates[first_key]

    if first_coords.ndim == 3:
        n_keypoints = first_coords.shape[1]
        if bodyparts is None:
            bodyparts = [f"kp_{i}" for i in range(n_keypoints)]
    else:
        n_dim = first_coords.shape[1]
        if bodyparts is None:
            bodyparts = [f"dim_{i}" for i in range(n_dim)]

    config_file = project_path / "config.yml"
    if clear_project or not config_file.exists():
        kpms.setup_project(
            project_dir,
            fix_heading=fix_heading,
            bodyparts=bodyparts,
            skeleton=[],
            fps=30,
            overwrite=True,
        )

        kpms.update_config(
            project_dir,
            fix_heading=fix_heading,
            bodyparts=bodyparts,
            use_bodyparts=bodyparts,
            anterior_bodyparts=[bodyparts[0]],
            posterior_bodyparts=[bodyparts[-1]],
            verbose=False,
            latent_dim=hyperparams.latent_dim,
            num_states=n_states,
            kappa=hyperparams.kappa,
        )

    config = kpms.load_config(project_dir)
    data, metadata = kpms.format_data(coordinates, confidences, **config)

    model_name = f"{modality_name}_{hyperparams.model_type}_states{n_states}_seed{seed}"

    model = kpms.init_model(
        data=data,
        metadata=metadata,
        states=None,
        **config,
        seed=jax.random.PRNGKey(seed + n_states * 1000),
    )
    model = kpms.update_hypparams(model, kappa=hyperparams.kappa)

    model = kpms.fit_model(
        model=model,
        data=data,
        metadata=metadata,
        project_dir=project_dir,
        model_name=model_name,
        ar_only=True,
        num_iters=hyperparams.ar_iters,
        num_states=n_states,
        generate_progress_plots=False,
    )[0]

    if hyperparams.model_type == "slds":
        model = kpms.update_hypparams(model, kappa=hyperparams.kappa)
        kpms.fit_model(
            model=model,
            data=data,
            metadata=metadata,
            project_dir=project_dir,
            model_name=model_name,
            ar_only=False,
            start_iter=hyperparams.ar_iters,
            num_iters=hyperparams.full_iters,
            generate_progress_plots=False,
        )

    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
    model_ckpt, _, _, _ = kpms.load_checkpoint(project_dir, model_name)
    model_ckpt["project_dir"] = project_dir
    model_ckpt["model_name"] = model_name

    results = kpms.extract_results(model_ckpt, metadata, project_dir, model_name)

    rec_names = sorted(results.keys())
    labels_list = [results[rn]["syllable"] for rn in rec_names]

    return KPMSFitResult(
        model=model_ckpt,
        results=results,
        data=data,
        metadata=metadata,
        model_name=model_name,
        project_dir=project_dir,
        labels_list=labels_list,
    )


def fit_kpms_intention(
    intention_data: np.ndarray,
    n_states: int,
    project_dir: str,
    hyperparams: KPMSHyperparams,
    seed: int,
    clear_project: bool = True,
) -> KPMSFitResult:
    coordinates, confidences = prepare_intention_for_kpms(intention_data)
    n_dim = intention_data.shape[2]
    bodyparts = [f"dim_{i}" for i in range(n_dim)]

    return fit_kpms_model(
        coordinates=coordinates,
        confidences=confidences,
        n_states=n_states,
        project_dir=project_dir,
        hyperparams=hyperparams,
        seed=seed,
        modality_name="intention",
        bodyparts=bodyparts,
        fix_heading=False,
        clear_project=clear_project,
    )


def fit_kpms_keypoints(
    keypoint_data: np.ndarray,
    n_states: int,
    project_dir: str,
    hyperparams: KPMSHyperparams,
    seed: int,
    kp_names: list[str] | None = None,
    clear_project: bool = True,
) -> KPMSFitResult:
    coordinates, confidences = prepare_keypoints_for_kpms(keypoint_data)

    if kp_names is None:
        n_keypoints = keypoint_data.shape[2]
        kp_names = [f"kp_{i}" for i in range(n_keypoints)]

    return fit_kpms_model(
        coordinates=coordinates,
        confidences=confidences,
        n_states=n_states,
        project_dir=project_dir,
        hyperparams=hyperparams,
        seed=seed,
        modality_name="keypoint",
        bodyparts=kp_names,
        fix_heading=True,
        clear_project=clear_project,
    )


def run_dual_fit(
    intention_data: np.ndarray,
    keypoint_data: np.ndarray,
    hyperparams: KPMSHyperparams,
    output_dir: str,
    seed: int,
    kp_names: list[str] | None = None,
    compute_eml_flag: bool = False,
) -> DualFitResult:
    int_project_dir = f"{output_dir}/intention_seed{seed}"
    kp_project_dir = f"{output_dir}/keypoint_seed{seed}"

    print(f"Fitting intention KPMS (seed={seed})...")
    int_result = fit_kpms_intention(
        intention_data=intention_data,
        n_states=hyperparams.num_states,
        project_dir=int_project_dir,
        hyperparams=hyperparams,
        seed=seed,
    )

    print(f"Fitting keypoint KPMS (seed={seed})...")
    kp_result = fit_kpms_keypoints(
        keypoint_data=keypoint_data,
        n_states=hyperparams.num_states,
        project_dir=kp_project_dir,
        hyperparams=hyperparams,
        seed=seed,
        kp_names=kp_names,
    )

    consistency = compute_syllable_consistency(int_result.results, kp_result.results)
    duration_kl = compute_duration_kl(int_result.results, kp_result.results)

    eml_intention = None
    eml_keypoint = None

    if compute_eml_flag:
        try:
            eml_intention = compute_eml(int_project_dir, [int_result.model_name])
            eml_keypoint = compute_eml(kp_project_dir, [kp_result.model_name])
        except Exception as e:
            print(f"Warning: EML computation failed: {e}")

    return DualFitResult(
        intention_result=int_result,
        keypoint_result=kp_result,
        syllable_ari=consistency.ari,
        syllable_nmi=consistency.nmi,
        duration_kl=duration_kl,
        eml_intention=eml_intention,
        eml_keypoint=eml_keypoint,
    )


def run_ensemble_dual_fit(
    intention_data: np.ndarray,
    keypoint_data: np.ndarray,
    hyperparams: KPMSHyperparams,
    output_dir: str,
    seeds: list[int],
    kp_names: list[str] | None = None,
) -> tuple[list[DualFitResult], dict[str, Any]]:
    int_project_dir = f"{output_dir}/intention_ensemble"
    kp_project_dir = f"{output_dir}/keypoint_ensemble"

    int_results = []
    kp_results = []

    for i, seed in enumerate(seeds):
        print(f"\n{'='*40}")
        print(f"Running dual fit with seed {seed} ({i+1}/{len(seeds)})")
        print(f"{'='*40}")

        clear_project = (i == 0)

        print(f"Fitting intention KPMS (seed={seed})...")
        int_result = fit_kpms_intention(
            intention_data=intention_data,
            n_states=hyperparams.num_states,
            project_dir=int_project_dir,
            hyperparams=hyperparams,
            seed=seed,
            clear_project=clear_project,
        )
        int_results.append(int_result)

        print(f"Fitting keypoint KPMS (seed={seed})...")
        kp_result = fit_kpms_keypoints(
            keypoint_data=keypoint_data,
            n_states=hyperparams.num_states,
            project_dir=kp_project_dir,
            hyperparams=hyperparams,
            seed=seed,
            kp_names=kp_names,
            clear_project=clear_project,
        )
        kp_results.append(kp_result)

    int_model_names = [r.model_name for r in int_results]
    kp_model_names = [r.model_name for r in kp_results]

    print(f"\nComputing EML for intention models: {int_model_names}")
    print(f"Computing EML for keypoint models: {kp_model_names}")

    try:
        int_eml = compute_eml(int_project_dir, int_model_names)
    except Exception as e:
        print(f"Warning: Intention EML computation failed: {e}")
        import traceback
        traceback.print_exc()
        int_eml = None

    try:
        kp_eml = compute_eml(kp_project_dir, kp_model_names)
    except Exception as e:
        print(f"Warning: Keypoint EML computation failed: {e}")
        import traceback
        traceback.print_exc()
        kp_eml = None

    results = []
    for int_res, kp_res in zip(int_results, kp_results):
        consistency = compute_syllable_consistency(int_res.results, kp_res.results)
        duration_kl = compute_duration_kl(int_res.results, kp_res.results)
        results.append(DualFitResult(
            intention_result=int_res,
            keypoint_result=kp_res,
            syllable_ari=consistency.ari,
            syllable_nmi=consistency.nmi,
            duration_kl=duration_kl,
            eml_intention=None,
            eml_keypoint=None,
        ))

    aris = [r.syllable_ari for r in results]
    nmis = [r.syllable_nmi for r in results]
    duration_kls = [r.duration_kl for r in results]

    ensemble_metrics = {
        "eml_intention": int_eml,
        "eml_keypoint": kp_eml,
        "syllable_ari_mean": float(np.mean(aris)),
        "syllable_ari_std": float(np.std(aris)),
        "syllable_nmi_mean": float(np.mean(nmis)),
        "syllable_nmi_std": float(np.std(nmis)),
        "duration_kl_mean": float(np.mean(duration_kls)),
        "duration_kl_std": float(np.std(duration_kls)),
    }

    return results, ensemble_metrics
