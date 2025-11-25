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
BEHAVIORS = ["walk", "rear"]
MATCH_MODE = "any"
MAX_CLIPS_LOOK = 500
MAX_CLIPS = 10
PERS_DELTA = 0.75
K_NEIGHBORS = 5
EPSILON = 0.99
OUTPUT_ROOT = "dm_results_pool"
M = 3
TAU = 6
BEHAVIOR_COLUMNS = ["rear","walk","walk_wait","head_turn_z","head_turn_xy","groom_down","groom_up"]

def takens_points(X, m=M, tau=TAU):
    n_frames, D = X.shape
    start = (m - 1) * tau
    if n_frames <= start:
        return np.empty((0, m * D), dtype=X.dtype)
    idxs = np.arange(start, n_frames)
    return np.stack([X[[j - a * tau for a in range(m)]].reshape(-1) for j in idxs], axis=0)

def load_behavior_csv(csv_path, behaviors, mode="all", max_look=None):
    behaviors = [b.strip().lower() for b in behaviors]
    valid = [c.lower() for c in BEHAVIOR_COLUMNS]
    for b in behaviors:
        if b not in valid:
            raise ValueError(f"Unknown behavior '{b}'. Valid: {BEHAVIOR_COLUMNS}")
    selected = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=",", skipinitialspace=True)
        for row in reader:
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
            ok = all(flags)  # must have all specified behaviors
            if ok:
                selected.append(clip_idx)
    return sorted(selected)

def ensure_TD(x, expected_D=None):
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

def main():
    rodent_rollout_dir = RODENT_ROLLOUT_DIR
    matched = load_behavior_csv(LABELS_CSV, BEHAVIORS, mode=MATCH_MODE, max_look=MAX_CLIPS_LOOK)
    if not matched:
        raise RuntimeError("No clips matched the requested behaviors.")
    if MAX_CLIPS is not None:
        matched = matched[:MAX_CLIPS]
        print(f'Selected {matched} clips')

    work_rodent = functools.partial(get_aggregate_data, "/intention", [], path=rodent_rollout_dir)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        int_list = list(tqdm(pool.imap(work_rodent, matched), total=len(matched)))

    int_per_clip = [ensure_TD(x) for x in int_list]
    pts_list = [takens_points(clip, m=M, tau=TAU) for clip in int_per_clip]
    pts_list = [p for p in pts_list if p.size > 0]
    if not pts_list:
        raise RuntimeError("Takens embedding produced no points. Check M/TAU vs clip lengths.")
    pts = np.vstack(pts_list)
    sys.path.append("/root/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")

    beh_tag = "_".join(BEHAVIORS)
    base_output_dir = f"{OUTPUT_ROOT}_m{M}_tau{TAU}_p{PERS_DELTA}_{MATCH_MODE}_{beh_tag}"
    os.makedirs(base_output_dir, exist_ok=True)

    cycle_inds, G_dm = extract_skeleton_pcd_dm(
        pts,
        k=K_NEIGHBORS,
        pers_delta=PERS_DELTA,
        epsilon=EPSILON,
        output_dir=base_output_dir,
        cleanup=False,
    )
    cycle_inds = [np.array(inds, dtype=int) for inds in cycle_inds]

    with open(os.path.join(base_output_dir, "matched_clip_indices.pkl"), "wb") as f:
        pkl.dump(matched, f)
    with open(os.path.join(base_output_dir, "all_G_dm.pkl"), "wb") as f:
        pkl.dump(G_dm, f)
    with open(os.path.join(base_output_dir, "all_cycle_dm.pkl"), "wb") as f:
        pkl.dump(cycle_inds, f)

    print(f"Matched clips: {matched}")
    print(f"Saved results to: {base_output_dir}")

if __name__ == "__main__":
    main()