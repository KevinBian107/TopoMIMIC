"""Minima experiment: extract behavioral minima from DM graphs and state-space methods."""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import pickle as pkl
import sys
from dataclasses import dataclass, field

import h5py
import numpy as np
import yaml
import mujoco
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

from topo_mimic.extraction.skeletonization.rendering import render_grid_clusters_side_by_side


@dataclass
class DMSourceEntry:
    results_dir: str


@dataclass
class SSSourceEntry:
    experiment_dir: str


@dataclass
class MinimaExperimentConfig:
    dataset: str = "rodent"
    output_dir: str = "outputs/benchmark_experiment_minima"
    data_path: str = ""
    rodent_rollout_dir: str = ""
    behavior_labels_csv: str = ""

    dm_sources: dict[str, DMSourceEntry] = field(default_factory=dict)
    state_space_sources: dict[str, SSSourceEntry] = field(default_factory=dict)

    per_clip_n: int = 3
    top_k_minima: int = 4
    minima_frame_radius: int = 10
    minima_n_frames: int = 5
    render_target_size: tuple[int, int] = (320, 240)
    xml_path: str = ""
    camera: str = "close_profile"

    n_clusters: int = 10
    n_samples_per_cluster: int = 5
    min_neighborhood_frames: int = 10
    skip_clips: tuple[int, ...] = ()

    def __post_init__(self):
        from topo_mimic.extraction.utils.dataset_config import get_dataset_preset

        preset = get_dataset_preset(self.dataset)

        if not self.data_path and not self.rodent_rollout_dir:
            self.data_path = preset.default_data_path
        elif not self.data_path:
            self.data_path = self.rodent_rollout_dir
        self.rodent_rollout_dir = self.data_path

        if not self.behavior_labels_csv:
            self.behavior_labels_csv = preset.labels_csv
        if not self.xml_path:
            self.xml_path = preset.xml_path
        if not self.skip_clips and preset.skip_clips:
            self.skip_clips = preset.skip_clips


def load_config(config_path: str) -> MinimaExperimentConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dm_sources = {}
    for name, src in cfg.get("dm_sources", {}).items():
        dm_sources[name] = DMSourceEntry(results_dir=src["results_dir"])

    ss_sources = {}
    for name, src in cfg.get("state_space_sources", {}).items():
        ss_sources[name] = SSSourceEntry(experiment_dir=src["experiment_dir"])

    data_path = cfg.get("data_path", cfg.get("rodent_rollout_dir", ""))

    return MinimaExperimentConfig(
        dataset=cfg.get("dataset", "rodent"),
        output_dir=cfg.get("output_dir", "outputs/benchmark_experiment_minima"),
        data_path=data_path,
        behavior_labels_csv=cfg.get("behavior_labels_csv", ""),
        dm_sources=dm_sources,
        state_space_sources=ss_sources,
        per_clip_n=cfg.get("per_clip_n", 3),
        top_k_minima=cfg.get("top_k_minima", 4),
        minima_frame_radius=cfg.get("minima_frame_radius", 10),
        minima_n_frames=cfg.get("minima_n_frames", 5),
        render_target_size=tuple(cfg.get("render_target_size", [320, 240])),
        xml_path=cfg.get("xml_path", ""),
        camera=cfg.get("camera", "close_profile"),
        n_clusters=cfg.get("n_clusters", 10),
        n_samples_per_cluster=cfg.get("n_samples_per_cluster", 5),
        min_neighborhood_frames=cfg.get("min_neighborhood_frames", 10),
        skip_clips=tuple(cfg.get("skip_clips", ())),
    )


@dataclass
class MinimaEntry:
    clip_id: int
    frame_center: int
    frame_start: int
    frame_end: int
    label: str


def _load_behavior_labels(
    behavior_labels_csv: str,
    n_clips: int,
    n_frames: int,
) -> dict[int, list[str]]:
    import pandas as pd
    if not os.path.exists(behavior_labels_csv):
        return {}
    df = pd.read_csv(behavior_labels_csv)
    labels = {}
    for _, row in df.iterrows():
        video = str(row["Video"])
        clip_id = int(video.replace("clip_", "").replace(".mp4", ""))
        if clip_id not in labels:
            labels[clip_id] = [""] * n_frames
        start = int(row["Start_Frame"])
        end = int(row["End_Frame"])
        beh = str(row["Behavior"])
        for f in range(start, min(end, n_frames)):
            labels[clip_id][f] = beh
    return labels


def _select_spatially_diverse(
    candidates: list[MinimaEntry],
    top_k: int,
) -> list[MinimaEntry]:
    if len(candidates) <= top_k:
        return candidates

    selected = [candidates[0]]
    remaining = list(candidates[1:])

    while len(selected) < top_k and remaining:
        best_idx = -1
        best_min_dist = -1
        for i, cand in enumerate(remaining):
            min_dist = min(abs(cand.frame_center - s.frame_center) for s in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return selected


def extract_dm_minima_for_clip(
    clip_id: int,
    results_dir: str,
    G_sm,
    n_frames: int,
    top_k: int = 4,
    frame_radius: int = 10,
    frame_labels: list[str] | None = None,
) -> list[MinimaEntry]:
    from topo_mimic.extraction.skeletonization.remap import (
        parse_swr_remap,
        remap_cycle_to_original,
    )
    from topo_mimic.extraction.skeletonization.dm_path import (
        _parse_persistence_pairs,
        _parse_sorted_weights,
    )

    clip_dir = os.path.join(results_dir, "clips", f"clip_{clip_id}")
    if not os.path.isdir(clip_dir):
        return []

    pairs = _parse_persistence_pairs(clip_dir)
    weights = _parse_sorted_weights(clip_dir)

    type0_pairs = [(v0, v1, pers) for v0, v1, t, pers in pairs if t == 0]
    type0_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    remap = parse_swr_remap(clip_dir)
    n_takens = max(remap.values()) + 1 if remap else n_frames

    candidate_pool_size = max(top_k * 5, 30)
    seen = set()
    candidates = []
    for v0, v1, pers in type0_pairs:
        w0 = weights.get(v0, float("inf"))
        w1 = weights.get(v1, float("inf"))
        min_v = v0 if w0 < w1 else v1
        if min_v in seen or min_v not in G_sm:
            continue
        seen.add(min_v)

        orig_frames = remap_cycle_to_original([min_v], remap, n_takens)
        if len(orig_frames) == 0:
            continue
        orig_idx = int(orig_frames[0])
        orig_idx = min(orig_idx, n_frames - 1)

        start = max(0, orig_idx - frame_radius)
        end = min(n_frames, orig_idx + frame_radius + 1)

        candidates.append(MinimaEntry(
            clip_id=clip_id,
            frame_center=orig_idx,
            frame_start=start,
            frame_end=end,
            label="",
        ))
        if len(candidates) >= candidate_pool_size:
            break

    selected = _select_spatially_diverse(candidates, top_k)

    for i, entry in enumerate(selected):
        beh = ""
        if frame_labels and entry.frame_center < len(frame_labels):
            beh = frame_labels[entry.frame_center]
        entry.label = f"M{i}" + (f" ({beh})" if beh else "")

    return selected


def extract_kpms_minima_for_clip(
    clip_id: int,
    syllables: np.ndarray,
    top_k: int = 4,
    min_duration: int = 10,
    frame_labels: list[str] | None = None,
) -> list[MinimaEntry]:
    from topo_mimic.experiment.methods import find_sustained_segments

    segments = find_sustained_segments(syllables, min_duration=min_duration)
    segments.sort(key=lambda s: s.duration, reverse=True)

    candidate_pool_size = max(top_k * 5, 30)
    candidates = []
    for seg in segments[:candidate_pool_size]:
        center = (seg.start_frame + seg.end_frame) // 2
        candidates.append(MinimaEntry(
            clip_id=clip_id,
            frame_center=center,
            frame_start=seg.start_frame,
            frame_end=seg.end_frame,
            label=f"S{seg.syllable_id}",
        ))

    selected = _select_spatially_diverse(candidates, top_k)

    for i, entry in enumerate(selected):
        beh = ""
        if frame_labels and entry.frame_center < len(frame_labels):
            beh = frame_labels[entry.frame_center]
        syll_id = entry.label.replace("S", "")
        entry.label = f"S{syll_id}" + (f" ({beh})" if beh else "")

    return selected


def render_minima_grid_static(
    entries: list[MinimaEntry],
    qpos_data: np.ndarray,
    xml_path: str,
    camera: str,
    target_size: tuple[int, int],
    n_frames: int,
) -> np.ndarray | None:
    if not entries:
        return None

    w, h = target_size
    n_rows = len(entries)
    n_cols = n_frames

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=h, width=w)

    grid = np.zeros((n_rows * h, n_cols * w, 3), dtype=np.uint8)

    for row, entry in enumerate(entries):
        neighborhood = list(range(entry.frame_start, entry.frame_end))
        if len(neighborhood) == 0:
            continue
        if len(neighborhood) <= n_cols:
            sample_frames = neighborhood
        else:
            step = max(1, len(neighborhood) // n_cols)
            sample_frames = neighborhood[::step][:n_cols]

        clip_qpos = qpos_data[entry.clip_id]
        for col, fi in enumerate(sample_frames):
            if fi >= len(clip_qpos):
                continue
            mj_data.qpos[:] = clip_qpos[fi]
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=camera)
            frame = renderer.render().copy()
            grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = frame

    renderer.close()
    return grid


def _add_row_labels(grid: np.ndarray, labels: list[str], tile_h: int) -> Image.Image:
    img = Image.fromarray(grid)
    label_width = 120
    new_img = Image.new("RGB", (img.width + label_width, img.height), (255, 255, 255))
    new_img.paste(img, (label_width, 0))
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    for i, label in enumerate(labels):
        y = i * tile_h + tile_h // 2 - 6
        draw.text((4, y), label, fill=(0, 0, 0), font=font)
    return new_img



def _process_source(
    source_name: str,
    source_type: str,
    config: MinimaExperimentConfig,
    qpos_data: np.ndarray,
    n_clips: int,
    n_frames_per_clip: int,
    behavior_labels: dict[int, list[str]],
    all_G_sm: list | None = None,
    results_dir: str = "",
    ss_labels: np.ndarray | None = None,
) -> None:
    source_output_dir = os.path.join(config.output_dir, source_name)
    os.makedirs(source_output_dir, exist_ok=True)

    skip_set = set(config.skip_clips)
    clip_ids = [i for i in range(n_clips) if i not in skip_set]
    if source_type == "dm" and all_G_sm:
        clip_ids = [i for i in clip_ids if i < len(all_G_sm) and all_G_sm[i] is not None]

    example_clips = clip_ids[:config.per_clip_n]

    print(f"  Rendering per-clip minima grids for {len(example_clips)} clips...")
    for clip_id in example_clips:
        frame_labels = behavior_labels.get(clip_id)

        if source_type == "dm":
            entries = extract_dm_minima_for_clip(
                clip_id=clip_id,
                results_dir=results_dir,
                G_sm=all_G_sm[clip_id],
                n_frames=n_frames_per_clip,
                top_k=config.top_k_minima,
                frame_radius=config.minima_frame_radius,
                frame_labels=frame_labels,
            )
        else:
            if ss_labels is None:
                continue
            syllables = ss_labels[clip_id] if clip_id < len(ss_labels) else None
            if syllables is None:
                continue
            entries = extract_kpms_minima_for_clip(
                clip_id=clip_id,
                syllables=np.asarray(syllables),
                top_k=config.top_k_minima,
                min_duration=config.min_neighborhood_frames,
                frame_labels=frame_labels,
            )

        if not entries:
            print(f"    Clip {clip_id}: no minima found")
            continue

        grid = render_minima_grid_static(
            entries=entries,
            qpos_data=qpos_data,
            xml_path=config.xml_path,
            camera=config.camera,
            target_size=config.render_target_size,
            n_frames=config.minima_n_frames,
        )
        if grid is not None:
            labels_list = [e.label for e in entries]
            labeled = _add_row_labels(grid, labels_list, config.render_target_size[1])
            save_path = os.path.join(source_output_dir, f"clip_{clip_id}_minima_grid.png")
            labeled.save(save_path)
            print(f"    Saved {save_path}")

    print(f"  Collecting all minima neighborhoods for W2 clustering...")
    all_minima_windows: list[tuple[int, np.ndarray]] = []
    all_minima_entries: list[MinimaEntry] = []

    for clip_id in clip_ids:
        frame_labels = behavior_labels.get(clip_id)

        if source_type == "dm":
            entries = extract_dm_minima_for_clip(
                clip_id=clip_id,
                results_dir=results_dir,
                G_sm=all_G_sm[clip_id],
                n_frames=n_frames_per_clip,
                top_k=config.top_k_minima,
                frame_radius=config.minima_frame_radius,
                frame_labels=frame_labels,
            )
        else:
            if ss_labels is None:
                continue
            syllables = ss_labels[clip_id] if clip_id < len(ss_labels) else None
            if syllables is None:
                continue
            entries = extract_kpms_minima_for_clip(
                clip_id=clip_id,
                syllables=np.asarray(syllables),
                top_k=config.top_k_minima,
                min_duration=config.min_neighborhood_frames,
                frame_labels=frame_labels,
            )

        for entry in entries:
            frames = np.arange(entry.frame_start, entry.frame_end)
            if len(frames) < config.min_neighborhood_frames:
                continue
            all_minima_windows.append((entry.clip_id, frames))
            all_minima_entries.append(entry)

    print(f"  Collected {len(all_minima_windows)} valid minima neighborhoods")

    if len(all_minima_windows) < config.n_clusters:
        print(f"  Too few neighborhoods for clustering, skipping W2 stage")
        return

    print(f"  Loading intention data for W2 distances...")
    with h5py.File(config.rodent_rollout_dir, "r") as f:
        intention_data = np.array(f["intention"])

    n_intention_frames = intention_data.shape[1]
    all_minima_windows = [
        (ci, fi[fi < n_intention_frames]) for ci, fi in all_minima_windows
    ]
    all_minima_windows = [
        (ci, fi) for ci, fi in all_minima_windows if len(fi) > 0
    ]

    print(f"  Computing W2 distance matrix...")
    try:
        import ot
    except ImportError:
        print("  POT not installed, skipping W2 clustering")
        return

    M = len(all_minima_windows)
    D = np.zeros((M, M), dtype=float)
    for i in range(M):
        ci, fi = all_minima_windows[i]
        Xi = intention_data[ci, fi]
        ni = len(fi)
        wi = np.full(ni, 1.0 / ni)
        for j in range(i + 1, M):
            cj, fj = all_minima_windows[j]
            Xj = intention_data[cj, fj]
            nj = len(fj)
            wj = np.full(nj, 1.0 / nj)
            C = ot.dist(Xi, Xj)
            val = ot.emd2(wi, wj, C)
            D[i, j] = D[j, i] = val

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{M} rows computed")

    print(f"  Running MDS + KMeans (k={config.n_clusters})...")
    seed = 0
    mds = MDS(n_components=min(3, M - 1), dissimilarity="precomputed", random_state=seed)
    coords = mds.fit_transform(D)
    n_k = min(config.n_clusters, M)
    km = KMeans(n_clusters=n_k, random_state=seed, n_init=10)
    cluster_ids_arr = km.fit_predict(coords)

    cluster_entries: dict[int, list[MinimaEntry]] = {}
    cluster_gt_labels: dict[int, list[str]] = {}
    for idx, cid in enumerate(cluster_ids_arr):
        cid = int(cid)
        cluster_entries.setdefault(cid, []).append(all_minima_entries[idx])
        beh = ""
        entry = all_minima_entries[idx]
        fl = behavior_labels.get(entry.clip_id)
        if fl and entry.frame_center < len(fl):
            beh = fl[entry.frame_center]
        if beh:
            cluster_gt_labels.setdefault(cid, []).append(beh)

    cluster_label_strs = {}
    for cid in sorted(cluster_entries.keys()):
        gt = cluster_gt_labels.get(cid, [])
        if gt:
            from collections import Counter
            majority = Counter(gt).most_common(1)[0][0]
            cluster_label_strs[cid] = f"C{cid} ({majority})"
        else:
            cluster_label_strs[cid] = f"C{cid}"

    print(f"  Rendering cluster grid video...")
    w, h = config.render_target_size
    mj_model = mujoco.MjModel.from_xml_path(config.xml_path)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=h, width=w)

    cluster_chunks: dict[int, list[list[np.ndarray]]] = {}
    cluster_clip_ids: dict[int, list[int]] = {}
    for cid in sorted(cluster_entries.keys()):
        samples = cluster_entries[cid][:config.n_samples_per_cluster]
        chunk_list = []
        clip_id_list = []
        for entry in samples:
            clip_qpos = qpos_data[entry.clip_id]
            frames_rendered = []
            for fi in range(entry.frame_start, entry.frame_end):
                if fi >= len(clip_qpos):
                    continue
                mj_data.qpos[:] = clip_qpos[fi]
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(mj_data, camera=config.camera)
                frames_rendered.append(renderer.render().copy())
            if frames_rendered:
                chunk_list.append(frames_rendered)
                clip_id_list.append(entry.clip_id)
        if chunk_list:
            cluster_chunks[cid] = chunk_list
            cluster_clip_ids[cid] = clip_id_list

    renderer.close()

    if cluster_chunks:
        video_path = os.path.join(source_output_dir, "minima_clusters_grid.mp4")
        render_grid_clusters_side_by_side(
            cluster_chunks=cluster_chunks,
            out_path=video_path,
            framerate=30,
            max_samples_per_cluster=config.n_samples_per_cluster,
            target_size=config.render_target_size,
            cluster_labels=cluster_label_strs,
            cluster_clip_ids=cluster_clip_ids,
        )
        print(f"  Saved {video_path}")

    assignments_path = os.path.join(source_output_dir, "minima_clusters.pkl")
    with open(assignments_path, "wb") as f:
        pkl.dump({
            "cluster_ids": cluster_ids_arr,
            "entries": all_minima_entries,
            "distance_matrix": D,
            "mds_coords": coords,
            "cluster_labels": cluster_label_strs,
        }, f)
    print(f"  Saved cluster data to {assignments_path}")


def run_experiment(config: MinimaExperimentConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 60)
    print("Minima Experiment Pipeline")
    print("=" * 60)

    print("Loading qpos data for rendering...")
    with h5py.File(config.rodent_rollout_dir, "r") as f:
        qpos_data = np.array(f["qpos_rollout"])
    n_clips, n_frames_per_clip = qpos_data.shape[0], qpos_data.shape[1]
    print(f"  qpos: {qpos_data.shape}")

    behavior_labels = _load_behavior_labels(
        config.behavior_labels_csv, n_clips, n_frames_per_clip,
    )

    for source_name, dm_src in config.dm_sources.items():
        print(f"\n=== DM source: {source_name} ===")
        graph_path = os.path.join(dm_src.results_dir, "all_G_sm.pkl")
        if not os.path.exists(graph_path):
            print(f"  all_G_sm.pkl not found at {graph_path}, skipping")
            continue
        with open(graph_path, "rb") as f:
            all_G_sm = pkl.load(f)
        print(f"  Loaded {len(all_G_sm)} DM graphs from {dm_src.results_dir}")

        _process_source(
            source_name=source_name,
            source_type="dm",
            config=config,
            qpos_data=qpos_data,
            n_clips=n_clips,
            n_frames_per_clip=n_frames_per_clip,
            behavior_labels=behavior_labels,
            all_G_sm=all_G_sm,
            results_dir=dm_src.results_dir,
        )

    for source_name, ss_src in config.state_space_sources.items():
        print(f"\n=== State-space source: {source_name} ===")
        results_path = os.path.join(ss_src.experiment_dir, "all_results.pkl")
        if not os.path.exists(results_path):
            print(f"  all_results.pkl not found at {results_path}, skipping")
            continue
        with open(results_path, "rb") as f:
            all_results = pkl.load(f)

        method_results = all_results.get(source_name)
        if method_results is None:
            print(f"  Method '{source_name}' not found in results, skipping")
            continue

        seeds = sorted([k for k in method_results.keys() if isinstance(k, int)])
        if not seeds:
            print(f"  No seeds found, skipping")
            continue

        result = method_results.get(seeds[0], {})
        labels_arr = result.get("labels")
        if labels_arr is None:
            print(f"  No labels found for seed {seeds[0]}, skipping")
            continue

        _process_source(
            source_name=source_name,
            source_type="state_space",
            config=config,
            qpos_data=qpos_data,
            n_clips=n_clips,
            n_frames_per_clip=n_frames_per_clip,
            behavior_labels=behavior_labels,
            ss_labels=np.asarray(labels_arr),
        )

    print("\n" + "=" * 60)
    print("Minima Experiment Complete!")
    print("=" * 60)
    print(f"Results saved to: {config.output_dir}")


def main(config_path: str = "configs/experiment/minima.yaml"):
    if os.path.exists(config_path):
        cfg = load_config(config_path)
    else:
        cfg = MinimaExperimentConfig()

    run_experiment(cfg)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/experiment/minima.yaml"
    main(config_path)
