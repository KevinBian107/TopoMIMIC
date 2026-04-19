"""Scale-sweep orchestrator. Phases 0-5 with skip-if-exists gating.

Phase 0: load data (rodent.csv coverage assertion).
Phase 1: enumerate real transitions, freeze list.
Phase 2: Mapper sweep (45 variants).
Phase 3: KPMS sweep (25 fits).
Phase 4: discovery table.
Phase 5: trace PNGs.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import yaml

from topo_mimic.benchmark.scale_sweep.config import ScaleSweepConfig
from topo_mimic.benchmark.scale_sweep.data import load_scale_sweep_data
from topo_mimic.benchmark.scale_sweep.discovery import (
    assemble_variant_metas,
    build_discovery_tables,
)
from topo_mimic.benchmark.scale_sweep.discovery_connections import (
    build_connection_discovery_tables,
)
from topo_mimic.benchmark.scale_sweep.kpms_scale import run_kpms_sweep
from topo_mimic.benchmark.scale_sweep.labels import (
    enumerate_real_transitions,
    load_real_transitions_yaml,
    write_real_transitions_yaml,
)
from topo_mimic.benchmark.scale_sweep.mapper_scale import run_mapper_sweep
from topo_mimic.benchmark.scale_sweep.kpms_reconstruction import (
    generate_kpms_reconstruction_pngs_for_variant,
)
from topo_mimic.benchmark.scale_sweep.mapper_diagnostics import (
    _compute_pca_2d_for_feature_space,
    generate_mapper_diagnostics,
    reconstruct_labels_list_from_cluster_dict,
    save_per_clip_pca_overlay_png,
)
from topo_mimic.benchmark.scale_sweep.mapper_scale import _preprocess
from topo_mimic.benchmark.scale_sweep.plots import (
    _behavior_color_map,
    _pick_exemplar_clips,
    generate_trace_pngs_for_variant,
    save_discovery_eyeball_png,
)

log = logging.getLogger(__name__)


def _output_root(cfg: ScaleSweepConfig) -> Path:
    return Path(cfg.io.output_dir) / f"cfg_{cfg.config_hash()}"


def _freeze_config(cfg: ScaleSweepConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config_used.yaml"
    path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))


def run_scale_sweep(
    cfg: ScaleSweepConfig,
    phases: list[int] | None = None,
    force: bool = False,
    max_kpms_fits: int | None = None,
) -> int:
    if phases is None:
        phases = [0, 1, 2, 3, 4, 5]
    out_root = _output_root(cfg)
    out_root.mkdir(parents=True, exist_ok=True)
    _freeze_config(cfg, out_root)
    log.info(f"scale-sweep output root: {out_root}")

    data = None
    transitions: list[str] = []
    transition_counts: dict[str, int] = {}
    mapper_results = []
    kpms_results = []
    kpms_selections: dict[float, object] = {}

    if 0 in phases:
        log.info("=== Phase 0: load data ===")
        data = load_scale_sweep_data(cfg)

    if 1 in phases:
        log.info("=== Phase 1: enumerate real transitions ===")
        rt_path = out_root / "real_transitions.yaml"
        if not force and rt_path.exists():
            transitions, transition_counts = load_real_transitions_yaml(rt_path)
            log.info(f"[P1] skip (loaded {len(transitions)} transitions from cache)")
        else:
            if data is None:
                data = load_scale_sweep_data(cfg)
            transitions, transition_counts = enumerate_real_transitions(
                data.segments_by_clip, data.clip_ids,
                groom_singleton=cfg.discovery.groom_singleton,
                include_other_singletons=False,
            )
            write_real_transitions_yaml(
                rt_path, transitions, transition_counts,
                clip_range=(cfg.data.start_clip, cfg.data.end_clip),
                n_clips_used=len(data.clip_ids),
            )

    if 2 in phases and cfg.mapper.enabled:
        log.info("=== Phase 2: Mapper sweep ===")
        if data is None:
            data = load_scale_sweep_data(cfg)
        mapper_results = run_mapper_sweep(
            data=data, cfg=cfg, output_root=out_root / "mapper", force=force,
        )
        log.info(f"[P2] Mapper: {len(mapper_results)} variants")

    if 3 in phases and cfg.kpms.enabled:
        log.info("=== Phase 3: KPMS sweep ===")
        if data is None:
            data = load_scale_sweep_data(cfg)
        kpms_results, kpms_selections = run_kpms_sweep(
            keypoints=data.keypoints,
            clip_ids=data.clip_ids,
            cfg=cfg,
            output_root=out_root / "kpms",
            force=force,
            max_fits=max_kpms_fits,
        )
        log.info(
            f"[P3] KPMS: {len(kpms_results)} fits, {len(kpms_selections)} kappa selections"
        )

    if 4 in phases:
        log.info("=== Phase 4: discovery table ===")
        if not transitions:
            transitions, transition_counts = load_real_transitions_yaml(
                out_root / "real_transitions.yaml"
            )
        if data is None:
            data = load_scale_sweep_data(cfg)
        if not mapper_results and cfg.mapper.enabled:
            mapper_results = run_mapper_sweep(
                data=data, cfg=cfg, output_root=out_root / "mapper", force=False,
            )
        if not kpms_results and cfg.kpms.enabled:
            kpms_results, kpms_selections = run_kpms_sweep(
                keypoints=data.keypoints, clip_ids=data.clip_ids,
                cfg=cfg, output_root=out_root / "kpms", force=False,
                max_fits=max_kpms_fits,
            )
        long_df = build_discovery_tables(
            transitions=transitions,
            transition_counts=transition_counts,
            kpms_results=kpms_results,
            kpms_selections=kpms_selections,
            mapper_results=mapper_results,
            segments=data.segments_by_clip,
            cfg=cfg.discovery,
            output_dir=out_root / "discovery",
        )
        log.info(f"[P4] discovery_long.csv: {len(long_df)} rows")

        variant_metas = assemble_variant_metas(
            kpms_results=kpms_results,
            kpms_selections=kpms_selections,
            mapper_results=mapper_results,
        )
        connection_long = build_connection_discovery_tables(
            transitions=transitions,
            transition_counts=transition_counts,
            variant_metas=variant_metas,
            segments=data.segments_by_clip,
            clip_ids=data.clip_ids,
            cfg=cfg.discovery,
            output_dir=out_root / "discovery",
        )
        log.info(
            f"[P4] discovery_connections_long.csv: {len(connection_long)} rows"
        )

        if cfg.visualization.discovery_eyeball_png:
            import pandas as pd

            conn_binary_path = out_root / "discovery" / "discovery_connections_binary_wide.csv"
            if conn_binary_path.exists():
                conn_binary_wide = pd.read_csv(conn_binary_path, index_col=0)
                save_discovery_eyeball_png(
                    conn_binary_wide,
                    out_root / "discovery" / "discovery_connections_eyeball.png",
                )

        if cfg.visualization.discovery_eyeball_png:
            import pandas as pd

            binary_wide = pd.read_csv(
                out_root / "discovery" / "discovery_binary_wide.csv", index_col=0
            )
            save_discovery_eyeball_png(
                binary_wide,
                out_root / "discovery" / "discovery_eyeball.png",
            )

    if 5 in phases and cfg.visualization.trace_png.enabled:
        log.info("=== Phase 5: trace PNGs ===")
        if data is None:
            data = load_scale_sweep_data(cfg)

        behaviors = sorted({
            s.behavior
            for segs in data.segments_by_clip.values()
            for s in segs
        })
        colors = _behavior_color_map(behaviors)
        exemplars = _pick_exemplar_clips(
            data.clip_ids, data.segments_by_clip,
            n=cfg.visualization.trace_png.n_exemplar_clips,
        )
        log.info(f"[P5] exemplar clips: {exemplars}")

        trace_dir = out_root / "traces"
        count = 0
        if not kpms_results and cfg.kpms.enabled:
            kpms_results, kpms_selections = run_kpms_sweep(
                keypoints=data.keypoints, clip_ids=data.clip_ids,
                cfg=cfg, output_root=out_root / "kpms", force=False,
                max_fits=max_kpms_fits,
            )
        if not mapper_results and cfg.mapper.enabled:
            mapper_results = run_mapper_sweep(
                data=data, cfg=cfg, output_root=out_root / "mapper", force=False,
            )
        for vr in kpms_results:
            if kpms_selections and vr.kappa in kpms_selections:
                sel = kpms_selections[vr.kappa]
                if sel.selected_variant_id != vr.variant_id:
                    continue
            paths = generate_trace_pngs_for_variant(
                variant_id=vr.variant_id,
                labels_list_path=vr.labels_list_path,
                clip_ids=data.clip_ids,
                exemplar_clips=exemplars,
                segments_by_clip=data.segments_by_clip,
                output_dir=trace_dir,
                behavior_colors=colors,
                n_frames_per_clip=data.n_frames,
            )
            count += len(paths)

            try:
                paths = generate_kpms_reconstruction_pngs_for_variant(
                    variant_id=vr.variant_id,
                    project_dir=vr.project_dir,
                    model_name=vr.model_name,
                    keypoints_array=data.keypoints,
                    clip_ids=data.clip_ids,
                    exemplar_clips=exemplars,
                    kp_names=data.kp_names,
                    out_dir=trace_dir,
                )
                count += len(paths)
            except Exception as e:
                log.warning(f"[P5] kpms reconstruction failed for {vr.variant_id}: {e}")

        mapper_diag_dir = out_root / "mapper_diagnostics"
        pca_cache: dict[str, np.ndarray] = {}
        preprocessed_cache: dict[str, np.ndarray] = {}
        import numpy as np
        import pickle

        def _get_preprocessed_for(feature_space: str) -> np.ndarray:
            if feature_space not in preprocessed_cache:
                preprocessed_cache[feature_space] = _preprocess(data, feature_space, cfg)
            return preprocessed_cache[feature_space]

        def _get_pca_for(feature_space: str) -> np.ndarray:
            if feature_space in pca_cache:
                return pca_cache[feature_space]
            cache_path = mapper_diag_dir / f"pca_2d_{feature_space}.npy"
            if cache_path.exists():
                arr = np.load(cache_path)
                pca_cache[feature_space] = arr
                return arr
            log.info(f"[P5] computing PCA-2D for feature_space={feature_space}")
            preprocessed = _get_preprocessed_for(feature_space)
            takens = cfg.mapper.takens_per_feature_space[feature_space]
            arr = _compute_pca_2d_for_feature_space(preprocessed, takens.m, takens.tau)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, arr)
            pca_cache[feature_space] = arr
            return arr

        for vr in mapper_results:
            labels_path = Path(vr.cluster_dict_path).parent / "labels_list.pkl"
            if not labels_path.exists():
                log.info(f"[P5] reconstructing labels_list.pkl for {vr.variant_id}")
                reconstruct_labels_list_from_cluster_dict(
                    cluster_dict_path=vr.cluster_dict_path,
                    clip_ids=data.clip_ids,
                    n_frames_per_clip=data.n_frames,
                    out_path=labels_path,
                )
            paths = generate_trace_pngs_for_variant(
                variant_id=vr.variant_id,
                labels_list_path=str(labels_path),
                clip_ids=data.clip_ids,
                exemplar_clips=exemplars,
                segments_by_clip=data.segments_by_clip,
                output_dir=trace_dir,
                behavior_colors=colors,
                n_frames_per_clip=data.n_frames,
            )
            count += len(paths)

            try:
                pca_2d = _get_pca_for(vr.feature_space)
                paths = generate_mapper_diagnostics(
                    variant_id=vr.variant_id,
                    feature_space=vr.feature_space,
                    lens_name=vr.lens,
                    n_intervals=vr.n_intervals,
                    variant_dir=Path(vr.cluster_dict_path).parent,
                    segments=data.segments_by_clip,
                    behavior_colors=colors,
                    out_root=mapper_diag_dir,
                    pca_2d=pca_2d,
                )
                count += len(paths)
            except Exception as e:
                log.warning(f"[P5] mapper_diagnostics failed for {vr.variant_id}: {e}")

            try:
                preprocessed = _get_preprocessed_for(vr.feature_space)
                takens = cfg.mapper.takens_per_feature_space[vr.feature_space]
                with Path(labels_path).open("rb") as f:
                    labels_list = pickle.load(f)
                cid_to_idx = {cid: i for i, cid in enumerate(data.clip_ids)}
                for cid in exemplars:
                    if cid not in cid_to_idx:
                        continue
                    pca_png = trace_dir / vr.variant_id / f"clip_{cid}_pca.png"
                    if pca_png.exists():
                        continue
                    save_per_clip_pca_overlay_png(
                        clip_id=cid,
                        clip_data=preprocessed[cid_to_idx[cid]],
                        takens_m=takens.m,
                        takens_tau=takens.tau,
                        predicted_labels_per_frame=np.asarray(labels_list[cid_to_idx[cid]]),
                        gt_segments=data.segments_by_clip.get(cid, []),
                        out_path=pca_png,
                        title=f"{vr.variant_id} | clip_{cid} | per-clip PCA-2D",
                        behavior_colors=colors,
                    )
                    count += 1
            except Exception as e:
                log.warning(f"[P5] per-clip PCA overlay failed for {vr.variant_id}: {e}")
        log.info(f"[P5] PNGs (traces + mapper_diagnostics): {count} files")

    log.info(f"done. output at {out_root}")
    return 0
