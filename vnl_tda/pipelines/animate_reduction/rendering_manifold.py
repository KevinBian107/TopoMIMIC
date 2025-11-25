"""Mujoco rendering help, particularly for rendering all kinds of dimensionality reduction projectionwith mujoco clip rendering"""

import os

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.animation as animation

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap

from PIL import Image

import multiprocessing as mp
import functools
import h5py

from vnl_tda.utils.rendering import render_from_saved_rollout
from vnl_tda.utils.load_utils import load_from_h5py, get_aggregate_data
from vnl_tda.utils.rendering import display_video
from PIL import Image


def render_pca_with_clips(int_array, rollout_dir, clip_id=1, clip_length=599, walker_name="fly",):
    """
    Main entry for rendering clips with PCA rendering on the side.
    """
    start = clip_id * clip_length 
    end = start + clip_length
    global_subset_indices = np.arange(start, end)

    with h5py.File(rollout_dir + f"/clip_{clip_id}.h5", "r") as h5file:
        rollout = load_from_h5py(h5file)
        act = get_aggregate_data("/activations", ['intention'], clip_id, path=rollout_dir)
    
    if len(act) > clip_length:    
        act = act[:clip_length] # ref only have clip_length
        
    scaler = StandardScaler()
    act = scaler.fit_transform(act)
    int_scaled = scaler.fit_transform(int_array)

    pca_global = PCA(n_components=2, random_state=42)
    pca_local = PCA(n_components=2, random_state=42)
    global_embedding = pca_global.fit(int_scaled).transform(int_scaled)
    local_embedding = pca_local.fit(act).transform(act)
    global_labels = KMeans(n_clusters=4, random_state=42).fit(global_embedding).labels_
    local_labels = KMeans(n_clusters=4, random_state=42).fit(local_embedding).labels_

    frames = render_with_global_and_local_pca_progression(
        rollout=rollout,
        walker_name=walker_name,
        pca_global=global_embedding, 
        cluster_global=global_labels,
        global_subset_indices=global_subset_indices,
        pca_local=local_embedding,
        cluster_local=local_labels,
        var_lcoal=pca_local.explained_variance_ratio_,
        var_global=pca_global.explained_variance_ratio_,
    )
    
    return frames
    

def render_with_global_and_local_pca_progression(
    rollout: dict,
    walker_name: str,
    pca_global: np.ndarray,
    cluster_global: np.ndarray,
    global_subset_indices: np.ndarray,
    pca_local: np.ndarray,
    cluster_local: np.ndarray,
    var_lcoal: np.ndarray,
    var_global: np.ndarray,
):
    """
    Render the MuJoCo frames side-by-side with a figure showing
    BOTH global PCA + local PCA for each frame.
    """
    
    frames_mujoco = render_from_saved_rollout(rollout, walker_name)[1:]
    T = len(frames_mujoco)
    
    if T != len(pca_local):
        raise ValueError(f"Mismatch: {T} MuJoCo frames vs {len(pca_local)} local PCA steps.")
    if T != len(global_subset_indices):
        raise ValueError(f"Mismatch: {T} frames vs {len(global_subset_indices)} global_subset_indices.")
    
    info = rollout["info"]
    # clip_idx = int(info[0]["clip_idx"])
    clip_idx = int(info["clip_idx"][0])

    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    worker_args = []
    for frame_idx in range(T):
        worker_args.append((
            frame_idx,
            pca_global,
            cluster_global,
            global_subset_indices,
            pca_local,
            cluster_local,
            T,
            clip_idx,
            var_lcoal,
            var_global,
        ))
    
    print("Rendering PCA (global+local) progression...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        frames_pca = pool.starmap(global_local_pca_worker, worker_args)
        
    concat_frames = []
    for idx in range(T):
        concat_img = np.hstack([frames_mujoco[idx], frames_pca[idx]])
        concat_frames.append(concat_img)

    matplotlib.use(orig_backend)
    return concat_frames


def plot_pca_intention(
    idx,
    episode_start,
    pca: PCA,
    pca_projections: np.ndarray,
    clip_idx: int,
    feature_name: str,
    n_components: int = 4,
    terminated: bool = False,
    window_size: int = 530,
) -> np.ndarray:
    """
    plot pca intention progression of the episode
    """
    max_y = np.max(list(pca_projections[:, :n_components]))
    min_y = np.min(list(pca_projections[:, :n_components]))
    y_lim = (min_y - 0.2, max_y + 0.2)
    idx_in_this_episode = idx - episode_start  # the current timestep in this episode
    plt.figure(figsize=(9.6, 4.8))
    for pc_ind in range(n_components):
        # Plot the PCA projection of the episode
        plt.plot(
            pca_projections[episode_start:idx, pc_ind],
            label=f"PC {pc_ind} ({pca.explained_variance_ratio_[pc_ind]*100:.1f}%)",
        )
        plt.scatter(idx - episode_start, pca_projections[idx - 1, pc_ind])
    if terminated:
        # Mark the episode termination
        plt.axvline(x=idx - episode_start, color="r", linestyle="-")
        plt.text(
            idx - episode_start - 8,  # Adjust the x-offset as needed
            sum(y_lim) / 2,  # Adjust the y-position as needed
            "Episode Terminated",
            color="r",
            rotation=90,
        )  # Rotate the text vertically
    if idx_in_this_episode <= window_size:
        plt.xlim(0, window_size)
    else:
        plt.xlim(idx_in_this_episode - window_size, idx_in_this_episode)  # dynamically move xlim as time progress
    plt.ylim(*y_lim)
    plt.legend(loc="upper right")
    plt.xlabel("Timestep")
    plt.title(f"PCA {feature_name} Progression for Clip {clip_idx}")  # TODO make it configurable
    # Get the current figure
    fig = plt.gcf()
    # Create a canvas for rendering
    canvas = FigureCanvasAgg(fig)
    # Render the canvas to a buffer
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    # Convert the buffer to a PIL Image
    image = Image.frombytes("RGBA", (width, height), s)
    rgb_array = np.array(image.convert("RGB"))
    return rgb_array


def render_with_pca_progression(
    rollout: dict,
    pca_projections: np.ndarray,
    n_components: int = 4,
    feature_name: str = "ctrl",
) -> list:
    """
    render with the rewards progression graph concat alongside with the rendering.
    """
    frames_mujoco = render_from_saved_rollout(rollout)[1:]
    # skip the first frame, since we don't have intention for the first frame
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
    
    info = rollout["info"]
    # clip_idx = int(info[0]["clip_idx"])
    clip_idx = int(info["clip_idx"][0])
        
    worker = functools.partial(
        plot_pca_intention,
        episode_start=0,
        clip_idx=clip_idx,
        pca_projections=pca_projections,
        n_components=n_components,
        feature_name=feature_name,
    )
    print("Rendering with PCA progression...")
    # Use multiprocessing to parallelize the rendering of the reward graph
    with mp.Pool(processes=mp.cpu_count()) as pool:
        frames_pca = pool.map(worker, range(len(rollout["qposes_rollout"])))
    concat_frames = []
    episode_start = 0
    # implement reset logics of the reward graph too.
    print("Concatenating frames...")
    for idx, frame in tqdm(enumerate(frames_mujoco)):
        concat_frames.append(np.hstack([frame, frames_pca[idx]]))
    reward_plot = plot_pca_intention(
        len(frames_mujoco) - 1, episode_start, pca_projections, clip_idx, feature_name, n_components, terminated=True
    )
    plt.close("all")  # Figure auto-closing upon backend switching is deprecated.
    matplotlib.use(orig_backend)
    for _ in range(50):
        concat_frames.append(np.hstack([frames_mujoco[-1], reward_plot]))  # create stoppage when episode terminates
    return concat_frames


def global_local_pca_worker(
    frame_idx: int,
    pca_global: np.ndarray,
    cluster_global: np.ndarray,
    global_subset_indices: np.ndarray,
    pca_local: np.ndarray,
    cluster_local: np.ndarray,
    T: int,
    clip_idx: int,
    var_lcoal: np.ndarray,
    var_global: np.ndarray,
) -> np.ndarray:
    """
    The picklable worker function called by Pool.
    """
    is_terminated = (frame_idx == T - 1)
    return plot_global_and_local_pca_intention_with_clusters(
        pca_global=pca_global,
        cluster_global=cluster_global,
        global_subset_indices=global_subset_indices,
        pca_local=pca_local,
        cluster_local=cluster_local,
        frame_idx_local=frame_idx,
        clip_idx=clip_idx,
        var_lcoal=var_lcoal,
        var_global=var_global,
        terminated=is_terminated,
    )
    

def plot_global_and_local_pca_intention_with_clusters(
    pca_global: np.ndarray,
    cluster_global: np.ndarray, 
    global_subset_indices: np.ndarray, 
    pca_local: np.ndarray, 
    cluster_local: np.ndarray,
    frame_idx_local: int,
    clip_idx: int,
    var_lcoal: np.ndarray,
    var_global: np.ndarray,
    terminated: bool = False
):
    """
    Plot side-by-side:
    1) Global PCA (left subplot): entire dataset in gray, this clip in color, highlight current frame.
    2) Local PCA (right subplot): just this clip, highlight current frame.
    """

    # assume the local embedding is T points and we have T frames. frame_idx_local in [0..T).
    T = len(pca_local)
    if frame_idx_local >= T:
        raise ValueError(f"frame_idx_local={frame_idx_local} exceeds local clip length={T}.")

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12.8, 4.8), dpi=100)
    
    all_x_g = pca_global[:, 0]
    all_y_g = pca_global[:, 1]
    
    # light grey plot
    ax1.scatter(all_x_g, all_y_g, color="lightgray", alpha=0.5, s=10, label="All data (global)")

    # highlight just the points belonging to this clip
    # pca_global_clip shape is (T, 2) if global_subset_indices has length T
    pca_global_clip = pca_global[global_subset_indices]
    cluster_global_clip = cluster_global[global_subset_indices]

    x_clip_g = pca_global_clip[:, 0]
    y_clip_g = pca_global_clip[:, 1]

    sc = ax1.scatter(
        x_clip_g,
        y_clip_g,
        c=cluster_global_clip,
        cmap="tab10",
        alpha=0.7,
        s=30,
        label=f"Clip {clip_idx}"
    )

    # frame_idx_local corresponds to subset_indices[frame_idx_local] in the global space
    current_global_idx = global_subset_indices[frame_idx_local]
    cur_x_g = pca_global[current_global_idx, 0]
    cur_y_g = pca_global[current_global_idx, 1]
    cur_clust_g = cluster_global[current_global_idx]

    ax1.scatter(
        cur_x_g, 
        cur_y_g, 
        c=[cur_clust_g], 
        cmap="tab10",
        edgecolor="black",
        s=100,
        alpha=1.0
    )
    ax1.set_title(f"Global PCA (Clip {clip_idx})")
    ax1.set_xlabel(f"PC1 (global) ({var_global[0]*100:.2f}%)")
    ax1.set_ylabel(f"PC2 (global) ({var_global[1]*100:.2f}%)")
    

    x_l = pca_local[:, 0]
    y_l = pca_local[:, 1]
    sc2 = ax2.scatter(
        x_l, 
        y_l, 
        c=cluster_local, 
        cmap="tab10",
        alpha=0.7,
        s=30,
        label=f"Clip {clip_idx} local"
    )

    ax2.scatter(
        x_l[frame_idx_local],
        y_l[frame_idx_local],
        c=[cluster_local[frame_idx_local]],
        cmap="tab10",
        edgecolor="black",
        s=100,
        alpha=1.0
    )
    local_title_str = f"Local PCA: Clip {clip_idx}, Frame {frame_idx_local}"
    if terminated:
        local_title_str += " (terminated)"
    ax2.set_title(local_title_str)
    ax2.set_xlabel(f"PC1 (local) ({var_lcoal[0]*100:.2f}%)")
    ax2.set_ylabel(f"PC2 (local) ({var_lcoal[1]*100:.2f}%)")

    plt.tight_layout()

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    plt.close(fig)
    return img


def render_umap_with_clips(int_array, rollout_dir, clip_id=1, clip_length=599, walker_name="fly", window_size=50, stride=1):
    """
    Main entry function of rendering UMAP with clips.
    """

    def sliding_window_umap(int_array, window_size=30, stride=10, n_components=3):
        windows = []
        for i in range(0, int_array.shape[0] - window_size + 1, stride):
            window = int_array[i:i + window_size].reshape(-1)
            windows.append(window)
        windows = np.array(windows)
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(windows)
    
    start = clip_id * clip_length
    end = start + clip_length
    global_subset_indices = np.arange(start, end)

    with h5py.File(f"{rollout_dir}/clip_{clip_id}.h5", "r") as h5file:
        rollout = load_from_h5py(h5file)
        act = get_aggregate_data("/activations", ['intention'], clip_id, path=rollout_dir)

    if len(act) > clip_length:
        act = act[:clip_length]

    scaler = StandardScaler()
    act_scaled = scaler.fit_transform(act)
    int_scaled = scaler.fit_transform(int_array.reshape(-1, int_array.shape[-1]))

    # UMAP with sliding windows
    umap_global = sliding_window_umap(int_scaled, window_size=window_size, stride=stride, n_components=3)
    umap_local = sliding_window_umap(act_scaled, window_size=window_size, stride=1, n_components=3)

    global_labels = KMeans(n_clusters=4, random_state=42).fit(umap_global).labels_
    local_labels = KMeans(n_clusters=4, random_state=42).fit(umap_local).labels_
    
    aligned_global_indices = global_subset_indices[window_size//2:window_size//2 + len(umap_local)]
    frames = render_with_global_and_local_umap_progression(
        rollout=rollout,
        walker_name=walker_name,
        umap_global=umap_global,
        cluster_global=global_labels,
        cluster_local=local_labels,
        global_subset_indices=aligned_global_indices,
        umap_local=umap_local,
        clip_idx=global_subset_indices,
    )
    return frames
    

def global_local_umap_worker(
    frame_idx: int,
    umap_global: np.ndarray,
    cluster_global: np.ndarray,
    global_subset_indices: np.ndarray,
    umap_local: np.ndarray,
    cluster_local: np.ndarray,
    T: int,
    clip_idx: int,
    terminated: bool = False,
) -> np.ndarray:
    """
    Worker function for UMAP rendering
    """
    return plot_global_and_local_umap_intention_with_clusters(
        umap_global=umap_global,
        cluster_global=cluster_global,
        global_subset_indices=global_subset_indices,
        umap_local=umap_local,
        cluster_local=cluster_local,
        frame_idx_local=frame_idx,
        clip_idx=clip_idx,
        terminated=terminated,
    )


def plot_global_and_local_umap_intention_with_clusters(
    umap_global: np.ndarray,
    cluster_global: np.ndarray,
    global_subset_indices: np.ndarray,
    umap_local: np.ndarray,
    cluster_local: np.ndarray,
    frame_idx_local: int,
    clip_idx: int,
    terminated: bool = False
) -> np.ndarray:
    """
    Plot side-by-side:
    1) Global UMAP (3D): entire dataset in gray, this clip in color, highlight current frame.
    2) Local UMAP (3D): just this clip, highlight current frame.
    """

    T = len(umap_local)
    if frame_idx_local >= T:
        raise ValueError(f"frame_idx_local={frame_idx_local} exceeds local clip length={T}.")

    fig = plt.figure(figsize=(13, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # === Global view ===
    # ax1.scatter(
    #     umap_global[:, 0], umap_global[:, 1], umap_global[:, 2],
    #     color='lightgray', alpha=0.1, s=10, label='All data'
    # )

    umap_global_clip = umap_global[global_subset_indices]
    cluster_global_clip = cluster_global[global_subset_indices]
    
    # ax1.scatter(
    #     umap_global_clip[:, 0],
    #     umap_global_clip[:, 1],
    #     umap_global_clip[:, 2],
    #     c=cluster_global_clip,
    #     cmap="tab10",
    #     s=25,
    #     alpha=0.9,
    #     label=f"Clip {clip_idx}"
    # )

    current_global_idx = global_subset_indices[frame_idx_local]
    
    # ax1.scatter(
    #     umap_global[current_global_idx, 0],
    #     umap_global[current_global_idx, 1],
    #     umap_global[current_global_idx, 2],
    #     c=[cluster_global[current_global_idx]],
    #     cmap="tab10",
    #     edgecolors='black',
    #     s=80,
    #     alpha=0.02,
    # )

    is_clip = np.zeros(len(umap_global), dtype=bool)
    is_clip[global_subset_indices] = True

    # scatter only the non-clip points, very faint
    non_clip_idx = np.where(~is_clip)[0]
    ax1.scatter(
        umap_global[non_clip_idx, 0],
        umap_global[non_clip_idx, 1],
        umap_global[non_clip_idx, 2],
        color='lightgray',
        alpha=0.01,    # super low
        s=3,           # small
        label='Other data'
    )

    # then plot clip points on top
    ax1.scatter(
        umap_global_clip[:, 0],
        umap_global_clip[:, 1],
        umap_global_clip[:, 2],
        c=cluster_global_clip,
        cmap="tab10",
        s=25,
        alpha=0.9,
        label=f"Clip {clip_idx}"
    )

    # highlight the current frame
    ax1.scatter(
        *umap_global[current_global_idx], 
        c=[cluster_global[current_global_idx]],
        cmap="tab10",
        edgecolors='black',
        s=80,
        alpha=1.0
        )

    # ax1.set_title(f"Global UMAP (Clip {clip_idx})")
    ax1.set_title(f"Global UMAP")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.set_zlabel("UMAP 3")
    ax1.view_init(elev=20, azim=-60)

    # === Local view ===
    ax2.scatter(
        umap_local[:, 0],
        umap_local[:, 1],
        umap_local[:, 2],
        c=cluster_local,
        cmap="tab10",
        alpha=0.7,
        s=25,
        label=f"Clip {clip_idx} local"
    )
    ax2.scatter(
        umap_local[frame_idx_local, 0],
        umap_local[frame_idx_local, 1],
        umap_local[frame_idx_local, 2],
        c=[cluster_local[frame_idx_local]],
        cmap="tab10",
        edgecolors='black',
        s=80
    )

    # title = f"Local UMAP: Clip {clip_idx}, Frame {frame_idx_local}"
    title = f"Local UMAP"
    if terminated:
        title += " (terminated)"
    ax2.set_title(title)
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    ax2.set_zlabel("UMAP 3")
    ax2.view_init(elev=20, azim=-60)

    plt.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    plt.close(fig)
    return image


def resize_img_to_match_height(base_img: np.ndarray, img_to_resize: np.ndarray) -> np.ndarray:
    """
    Resize img_to_resize to have the same height as base_img, preserving aspect ratio using PIL.
    """
    target_height = base_img.shape[0]
    orig_height, orig_width = img_to_resize.shape[:2]
    new_width = int(orig_width * (target_height / orig_height))

    pil_img = Image.fromarray(img_to_resize)
    resized_pil_img = pil_img.resize((new_width, target_height), Image.BILINEAR)
    return np.array(resized_pil_img)


def render_with_global_and_local_umap_progression(rollout, walker_name, umap_global, 
                                                  cluster_global, cluster_local, global_subset_indices, umap_local, clip_idx):
    """
    Render both locla and global trajectories for UMAP + Mujoco clips
    """
    
    frames_mujoco = render_from_saved_rollout(rollout, walker_name)[1:]
    frames_mujoco = frames_mujoco[:len(umap_local)]
    T = len(frames_mujoco)
    if T != len(umap_local):
        raise ValueError(f"Mismatch: {T} frames vs {len(umap_local)} local UMAP steps.")
    if T != len(global_subset_indices):
        raise ValueError(f"Mismatch: {T} vs {len(global_subset_indices)} global indices.")
    
    worker_args = []
    for frame_idx in range(T):
        worker_args.append((
            frame_idx,
            umap_global,
            cluster_global,
            global_subset_indices,
            umap_local,
            cluster_local,
            T,
            clip_idx,
        ))

    print("Rendering 3D UMAP progression...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        frames_umap = pool.starmap(global_local_umap_worker, worker_args)

    concat_frames = [
        np.hstack([mujoco, resize_img_to_match_height(mujoco, umap_img)])
        for mujoco, umap_img in zip(frames_mujoco, frames_umap)
    ]

    return concat_frames