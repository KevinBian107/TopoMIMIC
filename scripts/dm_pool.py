import os
import sys
import csv
import numpy as np
import functools
import multiprocessing as mp
import pickle as pkl
from tqdm import tqdm
from vnl_tda.utils.load_utils import get_aggregate_data
from vnl_tda.pipelines.tda.skeletonization.skeletonization import extract_skeleton_pcd_dm

RODENT_ROLLOUT_DIR = "/root/vast/kaiwen/TopoVNL/data/rodent_ar"
LABELS_CSV = "/root/vast/kaiwen/TopoVNL/behavior_book.csv"
BEHAVIORS = ["rear","walk","walk_wait","head_turn_z","head_turn_xy","groom_down","groom_up"]
MATCH_MODE = "any"
MAX_CLIPS_LOOK = 100
MAX_CLIPS = 10
PERS_DELTA = 0.75
PERS_DELTA_POOL = 0.75
K_NEIGHBORS = 5
EPSILON = 0.99
OUTPUT_ROOT = "dm_results_pool"
M = 3
TAU = 6
LOOP_SELECT = "longest"
BEHAVIOR_COLUMNS = ["rear","walk","walk_wait","head_turn_z","head_turn_xy","groom_down","groom_up"]


def takens_points(
    X: np.ndarray,
    m: int,
    tau: int,
) -> np.ndarray:
    """
    Construct Takens' delay embedding from time series.

    Args:
        X: Time series array of shape (n_frames, D)
        m: Embedding dimension
        tau: Time delay

    Returns:
        Delay embedded array of shape (n_valid_points, m * D)
    """
    n_frames, D = X.shape
    start = (m - 1) * tau
    if n_frames <= start:
        return np.empty((0, m * D), dtype=X.dtype)
    idxs = np.arange(start, n_frames)
    return np.stack([X[[j - a * tau for a in range(m)]].reshape(-1) for j in idxs], axis=0)


def load_behavior_csv(
    csv_path: str,
    behaviors: list[str],
    mode: str = "all",
    max_look: int | None = None,
) -> tuple[list[int], int]:
    """
    Load clip indices matching specified behaviors from CSV.

    Args:
        csv_path: Path to behavior labels CSV
        behaviors: List of behavior names to match
        mode: Match mode ("all" requires all behaviors)
        max_look: Maximum clip index to consider

    Returns:
        Tuple of (sorted list of matching clip indices, total rows in CSV)
    """
    behaviors = [b.strip().lower() for b in behaviors]
    valid = [c.lower() for c in BEHAVIOR_COLUMNS]
    for b in behaviors:
        if b not in valid:
            raise ValueError(f"Unknown behavior '{b}'. Valid: {BEHAVIOR_COLUMNS}")
    selected = []
    total_rows = 0
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=",", skipinitialspace=True)
        for row in reader:
            total_rows += 1
            r = { (k or "").strip().lstrip("\ufeff").lower(): (v or "").strip() for k, v in row.items() }
            if "clip_idx" not in r:
                continue
            try:
                clip_idx = int(r["clip_idx"])
            except:
                continue
            if max_look is not None and clip_idx >= max_look:
                continue
            flags = []
            for b in behaviors:
                try:
                    flags.append(int(r.get(b, "0")))
                except:
                    flags.append(0)
            ok = all(flags)
            if ok:
                selected.append(clip_idx)
    return sorted(selected), total_rows


def ensure_TD(
    x: np.ndarray,
    expected_D: int | None = None,
) -> np.ndarray:
    """
    Ensure array is 2D with shape (T, D).

    Args:
        x: Input array
        expected_D: Expected feature dimension

    Returns:
        Reshaped array of shape (T, D)
    """
    x = np.asarray(x)
    if x.ndim == 2:
        return x
    if x.ndim == 1:
        if expected_D is not None and x.size % expected_D == 0:
            return x.reshape(-1, expected_D)
        for D in [16, 32, 64]:
            if x.size % D == 0:
                return x.reshape(-1, D)
        raise ValueError(f"Per-clip intention returned 1D of size {x.size} and could not infer feature dim.")
    raise ValueError(f"Unexpected intention array shape {x.shape}")


def select_loop(
    cycles: list[np.ndarray],
    mode: str = "longest",
) -> np.ndarray | None:
    """
    Select a single loop from list of cycles.

    Args:
        cycles: List of cycle index arrays
        mode: Selection mode ("longest" or "shortest")

    Returns:
        Selected cycle indices or None if no cycles
    """
    if not cycles:
        return None
    if mode == "longest":
        return max(cycles, key=len)
    elif mode == "shortest":
        return min(cycles, key=len)
    else:
        raise ValueError(f"Unknown loop selection mode: {mode}")


def extract_loop_intentions(
    int_clip: np.ndarray,
    loop_indices: np.ndarray,
    m: int,
    tau: int,
) -> np.ndarray:
    """
    Extract raw intention vectors corresponding to loop indices.

    Args:
        int_clip: Intention array for clip of shape (n_frames, D)
        loop_indices: Indices into Takens-embedded space
        m: Embedding dimension used
        tau: Time delay used

    Returns:
        Intention vectors of shape (len(loop_indices), D)
    """
    start_offset = (m - 1) * tau
    original_frame_indices = loop_indices + start_offset
    original_frame_indices = np.clip(original_frame_indices, 0, len(int_clip) - 1)
    return int_clip[original_frame_indices]


def main():
    rodent_rollout_dir = RODENT_ROLLOUT_DIR
    matched, total_labeled = load_behavior_csv(LABELS_CSV, BEHAVIORS, mode=MATCH_MODE, max_look=MAX_CLIPS_LOOK)

    print(f"Behavior labels CSV contains {total_labeled} labeled clips")
    if MAX_CLIPS_LOOK is not None and MAX_CLIPS_LOOK > total_labeled:
        print(f"WARNING: MAX_CLIPS_LOOK ({MAX_CLIPS_LOOK}) exceeds labeled clips ({total_labeled}). "
              f"Only {total_labeled} clips have behavior labels.")
    if MAX_CLIPS is not None and MAX_CLIPS > total_labeled:
        print(f"WARNING: MAX_CLIPS ({MAX_CLIPS}) exceeds labeled clips ({total_labeled}). "
              f"Cannot process more than {total_labeled} clips with behavior filtering.")
    if MAX_CLIPS is not None and MAX_CLIPS > len(matched):
        print(f"WARNING: MAX_CLIPS ({MAX_CLIPS}) exceeds matched clips ({len(matched)}). "
              f"Only {len(matched)} clips match the specified behaviors.")

    if not matched:
        raise RuntimeError("No clips matched the requested behaviors.")
    if MAX_CLIPS is not None:
        matched = matched[:MAX_CLIPS]
    print(f"Selected {len(matched)} clips: {matched}")

    print("\n=== Stage 1: Loading intention data ===")
    work_rodent = functools.partial(get_aggregate_data, "/intention", [], path=rodent_rollout_dir)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        int_list = list(tqdm(pool.imap(work_rodent, matched), total=len(matched), desc="Loading clips"))
    int_per_clip = [ensure_TD(x) for x in int_list]

    beh_tag = "_".join(BEHAVIORS)
    base_output_dir = f"{OUTPUT_ROOT}_m{M}_tau{TAU}_p{PERS_DELTA}_{MATCH_MODE}_{beh_tag}"
    os.makedirs(base_output_dir, exist_ok=True)
    per_clip_dir = os.path.join(base_output_dir, "per_clip")
    os.makedirs(per_clip_dir, exist_ok=True)

    print("\n=== Stage 2: Computing DM graph per clip ===")
    all_cycles_per_clip = []
    all_G_per_clip = []
    selected_loops = []

    for i, (clip_idx, int_clip) in enumerate(tqdm(zip(matched, int_per_clip), total=len(matched), desc="Per-clip DM")):
        pts = takens_points(int_clip, m=M, tau=TAU)
        if pts.size == 0:
            print(f"  Clip {clip_idx}: skipped (no points after Takens)")
            all_cycles_per_clip.append([])
            all_G_per_clip.append(None)
            selected_loops.append(None)
            continue

        clip_output_dir = os.path.join(per_clip_dir, f"clip_{clip_idx}")
        cycle_inds, G_dm = extract_skeleton_pcd_dm(
            pts,
            k=K_NEIGHBORS,
            pers_delta=PERS_DELTA,
            epsilon=EPSILON,
            output_dir=clip_output_dir,
            cleanup=True,
        )
        cycle_inds = [np.array(inds, dtype=int) for inds in cycle_inds]
        all_cycles_per_clip.append(cycle_inds)
        all_G_per_clip.append(G_dm)

        selected = select_loop(cycle_inds, mode=LOOP_SELECT)
        selected_loops.append(selected)

        n_cycles = len(cycle_inds)
        loop_len = len(selected) if selected is not None else 0
        print(f"  Clip {clip_idx}: {n_cycles} cycles, selected {LOOP_SELECT} loop has {loop_len} points")

    print("\n=== Stage 3: Extracting loop intentions and pooling ===")
    pooled_intentions = []
    clip_loop_metadata = []

    for clip_idx, int_clip, loop_inds in zip(matched, int_per_clip, selected_loops):
        if loop_inds is None or len(loop_inds) == 0:
            continue
        loop_int = extract_loop_intentions(int_clip, loop_inds, m=M, tau=TAU)
        pooled_intentions.append(loop_int)
        clip_loop_metadata.append({
            "clip_idx": clip_idx,
            "loop_length": len(loop_inds),
            "start_idx_in_pool": sum(len(p) for p in pooled_intentions[:-1]),
        })

    if not pooled_intentions:
        raise RuntimeError("No loops extracted from any clip. Cannot proceed with pooled DM.")

    pooled_int = np.vstack(pooled_intentions)
    print(f"Pooled {len(pooled_intentions)} loops from {len(matched)} clips")
    print(f"Total pooled intention frames: {len(pooled_int)}")

    print("\n=== Stage 4: Takens embedding on pooled loop intentions ===")
    pooled_pts = takens_points(pooled_int, m=M, tau=TAU)
    print(f"Total data points for pooled DM graph computation: {len(pooled_pts)}")

    if pooled_pts.size == 0:
        raise RuntimeError("Takens embedding on pooled data produced no points.")

    print("\n=== Stage 5: Computing pooled DM graph ===")
    pooled_output_dir = os.path.join(base_output_dir, "pooled")
    os.makedirs(pooled_output_dir, exist_ok=True)

    pooled_cycles, pooled_G_dm = extract_skeleton_pcd_dm(
        pooled_pts,
        k=K_NEIGHBORS,
        pers_delta=PERS_DELTA_POOL,
        epsilon=EPSILON,
        output_dir=pooled_output_dir,
        cleanup=False,
    )
    pooled_cycles = [np.array(inds, dtype=int) for inds in pooled_cycles]

    print(f"\nPooled DM graph: {len(pooled_cycles)} cycles, {pooled_G_dm.number_of_edges()} edges")

    print("\n=== Saving results ===")
    with open(os.path.join(base_output_dir, "matched_clip_indices.pkl"), "wb") as f:
        pkl.dump(matched, f)
    with open(os.path.join(base_output_dir, "per_clip_cycles.pkl"), "wb") as f:
        pkl.dump(all_cycles_per_clip, f)
    with open(os.path.join(base_output_dir, "per_clip_graphs.pkl"), "wb") as f:
        pkl.dump(all_G_per_clip, f)
    with open(os.path.join(base_output_dir, "selected_loops.pkl"), "wb") as f:
        pkl.dump(selected_loops, f)
    with open(os.path.join(base_output_dir, "pooled_intentions.pkl"), "wb") as f:
        pkl.dump(pooled_int, f)
    with open(os.path.join(base_output_dir, "pooled_cycles.pkl"), "wb") as f:
        pkl.dump(pooled_cycles, f)
    with open(os.path.join(base_output_dir, "pooled_G_dm.pkl"), "wb") as f:
        pkl.dump(pooled_G_dm, f)
    with open(os.path.join(base_output_dir, "clip_loop_metadata.pkl"), "wb") as f:
        pkl.dump(clip_loop_metadata, f)

    print(f"\n=== Summary ===")
    print(f"Clips processed: {len(matched)}")
    print(f"Clips with valid loops: {len(pooled_intentions)}")
    print(f"Total pooled intention frames: {len(pooled_int)}")
    print(f"Total Takens points for pooled DM: {len(pooled_pts)}")
    print(f"Pooled DM cycles found: {len(pooled_cycles)}")
    print(f"Results saved to: {base_output_dir}")


if __name__ == "__main__":
    main()
