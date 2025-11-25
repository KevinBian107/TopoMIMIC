import os
import sys
import numpy as np
import functools
import multiprocessing as mp
import pickle as pkl
from tqdm import tqdm
from vnl_tda.utils.load_utils import get_aggregate_data
from vnl_tda.pipelines.tda.skeletonization.skeletonization import extract_skeleton_pcd_dm

M = 3
TAU = 6

def takens_points(X, m=M, tau=TAU):
    n_frames, D = X.shape
    start = (m - 1) * tau
    idxs = np.arange(start, n_frames)
    pts = np.stack([X[[j - a * tau for a in range(m)]].reshape(-1) for j in idxs], axis=0)
    return pts

def main():
    rodent_rollout_dir = "/root/vast/kaiwen/TopoVNL/data/rodent_ar"
    num_clips = 800
    work_rodent = functools.partial(get_aggregate_data, "/intention", [], path=rodent_rollout_dir)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        int_rodent = list(tqdm(pool.imap(work_rodent, range(num_clips)), total=num_clips))
    int_rodent = np.vstack(int_rodent)
    int_rodent = int_rodent.reshape([num_clips, 499, 16])
    sys.path.append("/root/vast/kaiwen/TopoVNL/PCD-Graph-Recon-DM")

    pers_delta = 0.75
    num_clips_to_compare = 800
    k_neighbors = 5
    epsilon = 0.99

    all_cycle_dm = []
    all_pts = []
    all_G_sm = []

    base_output_dir = f"dm_results_m{M}_tau{TAU}_p{pers_delta}"
    os.makedirs(base_output_dir, exist_ok=True)

    selected_int_rodent = int_rodent[:num_clips]
    for clip_id in range(num_clips_to_compare):
        clip = selected_int_rodent[clip_id]
        pts = takens_points(clip, m=M, tau=TAU)
        all_pts.append(pts)
        clip_output_dir = os.path.join(base_output_dir, f"clip_{clip_id}")
        cycle_inds, G_sm = extract_skeleton_pcd_dm(
            pts, k=k_neighbors, pers_delta=pers_delta, epsilon=epsilon, output_dir=clip_output_dir, cleanup=False
        )
        cycle_inds = [np.array(inds, dtype=int) for inds in cycle_inds]
        all_cycle_dm.append(cycle_inds)
        all_G_sm.append(G_sm)
        print(f"Clip {clip_id}: {len(cycle_inds)} cycles, {G_sm.number_of_edges()} edges")

    with open(os.path.join(base_output_dir, "all_G_sm.pkl"), "wb") as f:
        pkl.dump(all_G_sm, f)
    with open(os.path.join(base_output_dir, "all_cycle_dm.pkl"), "wb") as f:
        pkl.dump(all_cycle_dm, f)

if __name__ == "__main__":
    main()