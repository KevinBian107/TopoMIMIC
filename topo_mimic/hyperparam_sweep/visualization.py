"""Visualization utilities for KPMS hyperparameter sweep."""

import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from topo_mimic.hyperparam_sweep.metrics import (
    compute_duration_stats,
    compute_active_syllables,
    EMLResult,
)


def plot_duration_distribution(
    results: dict,
    title: str = "Syllable Duration Distribution",
    bins: int = 50,
    max_duration: int = 200,
) -> "Figure":
    if not HAS_MPL:
        raise ImportError("matplotlib not installed")

    durations = []
    for rec_name, rec_results in results.items():
        syllables = np.array(rec_results["syllable"])
        if len(syllables) == 0:
            continue
        current_syll = syllables[0]
        current_len = 1
        for i in range(1, len(syllables)):
            if syllables[i] == current_syll:
                current_len += 1
            else:
                durations.append(current_len)
                current_syll = syllables[i]
                current_len = 1
        durations.append(current_len)

    fig, ax = plt.subplots(figsize=(8, 5))

    if durations:
        ax.hist(durations, bins=bins, range=(0, max_duration), edgecolor="black", alpha=0.7)
        stats = compute_duration_stats(results)
        ax.axvline(stats.mean, color="red", linestyle="--", label=f"Mean: {stats.mean:.1f}")
        ax.axvline(stats.median, color="orange", linestyle="--", label=f"Median: {stats.median:.1f}")
        ax.legend()

    ax.set_xlabel("Duration (frames)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()

    return fig


def plot_syllable_frequencies(
    results: dict,
    title: str = "Syllable Frequencies",
    top_k: int = 30,
) -> "Figure":
    if not HAS_MPL:
        raise ImportError("matplotlib not installed")

    syllable_counts = {}
    for rec_name, rec_results in results.items():
        syllables = np.array(rec_results["syllable"])
        unique, counts = np.unique(syllables, return_counts=True)
        for s, c in zip(unique, counts):
            syllable_counts[int(s)] = syllable_counts.get(int(s), 0) + c

    sorted_items = sorted(syllable_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_items = sorted_items[:top_k]

    fig, ax = plt.subplots(figsize=(10, 5))

    if sorted_items:
        syllables = [str(s) for s, _ in sorted_items]
        counts = [c for _, c in sorted_items]
        ax.bar(syllables, counts, edgecolor="black", alpha=0.7)
        ax.set_xticks(range(len(syllables)))
        ax.set_xticklabels(syllables, rotation=45, ha="right")

    ax.set_xlabel("Syllable ID")
    ax.set_ylabel("Total Frames")
    ax.set_title(title)
    plt.tight_layout()

    return fig


def compute_empirical_transition_matrix(results: dict, n_states: int) -> np.ndarray:
    """Compute empirical transition matrix from syllable sequences."""
    trans_counts = np.zeros((n_states, n_states))

    for rec_name, rec_results in results.items():
        syllables = np.array(rec_results["syllable"])
        for i in range(len(syllables) - 1):
            s_from = int(syllables[i])
            s_to = int(syllables[i + 1])
            if 0 <= s_from < n_states and 0 <= s_to < n_states:
                trans_counts[s_from, s_to] += 1

    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1)
    trans_probs = trans_counts / row_sums

    return trans_probs


def plot_transition_matrix(
    model: dict,
    title: str = "Transition Probability Matrix",
    top_k: int = 20,
    results: dict | None = None,
) -> "Figure":
    if not HAS_MPL:
        raise ImportError("matplotlib not installed")

    fig, ax = plt.subplots(figsize=(10, 8))

    trans_matrix = None

    if "params" in model:
        params = model["params"]
        if "betas" in params:
            betas = np.array(params["betas"])
            if betas.ndim == 1:
                from scipy.special import softmax
                n_states = len(betas)
                trans_matrix = np.tile(softmax(betas), (n_states, 1))
            else:
                trans_matrix = betas
        elif "trans_probs" in params:
            trans_matrix = np.array(params["trans_probs"])
        elif "Pi" in params:
            trans_matrix = np.array(params["Pi"])

    if trans_matrix is None and "hypparams" in model:
        hypparams = model["hypparams"]
        if "betas" in hypparams:
            betas = np.array(hypparams["betas"])
            if betas.ndim == 1:
                from scipy.special import softmax
                n_states = len(betas)
                trans_matrix = np.tile(softmax(betas), (n_states, 1))
            else:
                trans_matrix = betas

    if trans_matrix is None and results is not None:
        n_states = top_k
        if "hypparams" in model and "num_states" in model["hypparams"]:
            n_states = model["hypparams"]["num_states"]
        trans_matrix = compute_empirical_transition_matrix(results, n_states)

    if trans_matrix is None:
        ax.text(0.5, 0.5, f"No transition matrix found\nModel keys: {list(model.keys())}\nParams keys: {list(model.get('params', {}).keys())}",
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(title)
        plt.tight_layout()
        return fig

    if trans_matrix.ndim == 2:
        n_states = min(trans_matrix.shape[0], top_k)
        trans_matrix = trans_matrix[:n_states, :n_states]

        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1)
        trans_matrix_norm = trans_matrix / row_sums

        im = ax.imshow(trans_matrix_norm, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax, label="Probability")

        ax.set_xlabel("To State")
        ax.set_ylabel("From State")
    else:
        ax.text(0.5, 0.5, f"Trans matrix has unexpected shape: {trans_matrix.shape}",
                ha='center', va='center', transform=ax.transAxes, fontsize=10)

    ax.set_title(title)
    plt.tight_layout()

    return fig


def plot_eml_ensemble(
    eml_result: EMLResult,
    model_names: list[str],
    title: str = "Expected Marginal Likelihood Ensemble",
) -> "Figure":
    if not HAS_MPL:
        raise ImportError("matplotlib not installed")

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(model_names))
    ax.bar(x, eml_result.scores, yerr=eml_result.std_errs, capsize=3, alpha=0.7, edgecolor="black")
    ax.axhline(eml_result.mean_score, color="red", linestyle="--", label=f"Mean: {eml_result.mean_score:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_xlabel("Model")
    ax.set_ylabel("EML Score")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    return fig


def plot_syllable_consistency_confusion(
    int_results: dict,
    kp_results: dict,
    title: str = "Intention vs Keypoint Syllable Co-occurrence",
    top_k: int = 15,
) -> "Figure":
    if not HAS_MPL:
        raise ImportError("matplotlib not installed")

    int_syllables = []
    kp_syllables = []

    common_keys = set(int_results.keys()) & set(kp_results.keys())
    for key in sorted(common_keys):
        int_s = np.array(int_results[key]["syllable"])
        kp_s = np.array(kp_results[key]["syllable"])
        min_len = min(len(int_s), len(kp_s))
        int_syllables.extend(int_s[:min_len].tolist())
        kp_syllables.extend(kp_s[:min_len].tolist())

    fig, ax = plt.subplots(figsize=(10, 8))

    if int_syllables and kp_syllables:
        int_unique = sorted(set(int_syllables))
        kp_unique = sorted(set(kp_syllables))

        int_counts = {s: 0 for s in int_unique}
        for s in int_syllables:
            int_counts[s] += 1
        top_int = sorted(int_counts.keys(), key=lambda x: int_counts[x], reverse=True)[:top_k]

        kp_counts = {s: 0 for s in kp_unique}
        for s in kp_syllables:
            kp_counts[s] += 1
        top_kp = sorted(kp_counts.keys(), key=lambda x: kp_counts[x], reverse=True)[:top_k]

        cooccur = np.zeros((len(top_int), len(top_kp)))
        int_to_idx = {s: i for i, s in enumerate(top_int)}
        kp_to_idx = {s: i for i, s in enumerate(top_kp)}

        for i_s, k_s in zip(int_syllables, kp_syllables):
            if i_s in int_to_idx and k_s in kp_to_idx:
                cooccur[int_to_idx[i_s], kp_to_idx[k_s]] += 1

        row_sums = cooccur.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1)
        cooccur_norm = cooccur / row_sums

        im = ax.imshow(cooccur_norm, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax, label="P(Keypoint | Intention)")

        ax.set_xticks(range(len(top_kp)))
        ax.set_xticklabels([str(s) for s in top_kp], rotation=45, ha="right")
        ax.set_yticks(range(len(top_int)))
        ax.set_yticklabels([str(s) for s in top_int])

        ax.set_xlabel("Keypoint Syllable")
        ax.set_ylabel("Intention Syllable")

    ax.set_title(title)
    plt.tight_layout()

    return fig


def plot_dual_duration_comparison(
    int_results: dict,
    kp_results: dict,
    title: str = "Duration Distribution Comparison",
    bins: int = 50,
    max_duration: int = 200,
) -> "Figure":
    if not HAS_MPL:
        raise ImportError("matplotlib not installed")

    def get_durations(results):
        durations = []
        for rec_name, rec_results in results.items():
            syllables = np.array(rec_results["syllable"])
            if len(syllables) == 0:
                continue
            current_syll = syllables[0]
            current_len = 1
            for i in range(1, len(syllables)):
                if syllables[i] == current_syll:
                    current_len += 1
                else:
                    durations.append(current_len)
                    current_syll = syllables[i]
                    current_len = 1
            durations.append(current_len)
        return durations

    int_durations = get_durations(int_results)
    kp_durations = get_durations(kp_results)

    fig, ax = plt.subplots(figsize=(10, 5))

    bin_edges = np.linspace(0, max_duration, bins + 1)

    ax.hist(int_durations, bins=bin_edges, alpha=0.5, label="Intention", edgecolor="blue")
    ax.hist(kp_durations, bins=bin_edges, alpha=0.5, label="Keypoint", edgecolor="orange")

    ax.set_xlabel("Duration (frames)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    return fig


def _ensure_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def _get_syllable_color(syllable_id: int, n_colors: int = 30) -> tuple[int, int, int]:
    if not HAS_MPL:
        hue = (syllable_id * 137) % 360
        r = int(128 + 127 * np.sin(np.radians(hue)))
        g = int(128 + 127 * np.sin(np.radians(hue + 120)))
        b = int(128 + 127 * np.sin(np.radians(hue + 240)))
        return (r, g, b)

    cmap = plt.cm.get_cmap("tab20", n_colors)
    color = cmap(syllable_id % n_colors)
    return (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))


RODENT_SKELETON = [
    ('Snout', 'SpineF'), ('SpineF', 'SpineM'), ('SpineM', 'SpineL'), ('SpineL', 'TailBase'),
    ('SpineF', 'ShoulderL'), ('ShoulderL', 'ElbowL'), ('ElbowL', 'WristL'), ('WristL', 'HandL'),
    ('SpineF', 'ShoulderR'), ('ShoulderR', 'ElbowR'), ('ElbowR', 'WristR'), ('WristR', 'HandR'),
    ('SpineL', 'HipL'), ('HipL', 'KneeL'), ('KneeL', 'AnkleL'), ('AnkleL', 'FootL'),
    ('SpineL', 'HipR'), ('HipR', 'KneeR'), ('KneeR', 'AnkleR'), ('AnkleR', 'FootR'),
    ('Snout', 'EarL'), ('Snout', 'EarR'),
]


def render_keypoint_3d_video(
    keypoint_data: np.ndarray,
    output_path: str,
    kp_names: list[str] | None = None,
    clip_idx: int = 0,
    fps: int = 30,
    figsize: tuple[int, int] = (8, 8),
    elev: float = 20,
    azim: float = -60,
    title: str = "Keypoint 3D Trajectory",
) -> str:
    if not HAS_MPL:
        raise ImportError("matplotlib not installed")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    P = np.array(keypoint_data[clip_idx])
    T, K, D = P.shape

    if kp_names is None:
        kp_names = [f"kp_{i}" for i in range(K)]

    name_to_idx = {n: i for i, n in enumerate(kp_names)}
    edges = []
    for a, b in RODENT_SKELETON:
        if a in name_to_idx and b in name_to_idx:
            edges.append((name_to_idx[a], name_to_idx[b]))

    mins = P.reshape(-1, 3).min(0)
    maxs = P.reshape(-1, 3).max(0)
    rng = float(np.max(maxs - mins))
    ctr = (mins + maxs) / 2
    pad = 0.05 * rng if rng > 0 else 1e-3

    try:
        from matplotlib.animation import FuncAnimation, FFMpegWriter

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')

        pts = ax.plot(P[0, :, 0], P[0, :, 1], P[0, :, 2], 'o', ms=4, color='blue')[0]
        lines = [
            ax.plot(
                [P[0, i, 0], P[0, j, 0]],
                [P[0, i, 1], P[0, j, 1]],
                [P[0, i, 2], P[0, j, 2]],
                '-', color='gray', linewidth=2
            )[0]
            for i, j in edges
        ]

        ax.set_xlim(ctr[0] - rng / 2 - pad, ctr[0] + rng / 2 + pad)
        ax.set_ylim(ctr[1] - rng / 2 - pad, ctr[1] + rng / 2 + pad)
        ax.set_zlim(ctr[2] - rng / 2 - pad, ctr[2] + rng / 2 + pad)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{title} (Clip {clip_idx})')
        ax.view_init(elev=elev, azim=azim)

        frame_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=10)

        def update(t):
            pts.set_data(P[t, :, 0], P[t, :, 1])
            pts.set_3d_properties(P[t, :, 2])
            for ln, (i, j) in zip(lines, edges):
                ln.set_data([P[t, i, 0], P[t, j, 0]], [P[t, i, 1], P[t, j, 1]])
                ln.set_3d_properties([P[t, i, 2], P[t, j, 2]])
            frame_text.set_text(f'Frame {t}/{T}')
            return [pts, *lines, frame_text]

        ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)

        writer = FFMpegWriter(fps=fps, bitrate=2400)
        ani.save(output_path, writer=writer)
        plt.close(fig)

        print(f"Saved 3D keypoint video: {output_path}")
        return output_path

    except Exception as e:
        print(f"Warning: Could not create 3D animation video: {e}")
        print("Falling back to frame-by-frame rendering...")

        if not HAS_IMAGEIO:
            raise ImportError("Neither FFMpegWriter nor imageio-ffmpeg available")

        dpi = 100
        fig_w, fig_h = figsize
        canvas_w, canvas_h = int(fig_w * dpi), int(fig_h * dpi)

        writer = imageio_ffmpeg.write_frames(
            output_path, (canvas_w, canvas_h), fps=fps, codec="h264", macro_block_size=1
        )
        writer.send(None)

        for t in range(T):
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(projection='3d')

            ax.scatter(P[t, :, 0], P[t, :, 1], P[t, :, 2], s=20, c='blue')
            for i, j in edges:
                ax.plot(
                    [P[t, i, 0], P[t, j, 0]],
                    [P[t, i, 1], P[t, j, 1]],
                    [P[t, i, 2], P[t, j, 2]],
                    '-', color='gray', linewidth=2
                )

            ax.set_xlim(ctr[0] - rng / 2 - pad, ctr[0] + rng / 2 + pad)
            ax.set_ylim(ctr[1] - rng / 2 - pad, ctr[1] + rng / 2 + pad)
            ax.set_zlim(ctr[2] - rng / 2 - pad, ctr[2] + rng / 2 + pad)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{title} (Clip {clip_idx}, Frame {t}/{T})')
            ax.view_init(elev=elev, azim=azim)

            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.send(frame)
            plt.close(fig)

        writer.close()
        print(f"Saved 3D keypoint video (frame-by-frame): {output_path}")
        return output_path


def render_dual_syllable_video(
    frames: list[np.ndarray],
    intention_syllables: np.ndarray,
    keypoint_syllables: np.ndarray,
    output_path: str,
    fps: int = 30,
    bar_height: int = 30,
) -> str:
    if not HAS_IMAGEIO:
        raise ImportError("imageio-ffmpeg or PIL not installed")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if len(frames) == 0:
        raise ValueError("No frames provided")

    first_frame = _ensure_rgb_uint8(frames[0])
    H, W = first_frame.shape[:2]

    canvas_h = H + 2 * bar_height + 20
    canvas_w = W

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()

    writer = imageio_ffmpeg.write_frames(
        output_path, (canvas_w, canvas_h), fps=fps, codec="h264", macro_block_size=1
    )
    writer.send(None)

    n_frames = min(len(frames), len(intention_syllables), len(keypoint_syllables))

    for t in range(n_frames):
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        int_syll = int(intention_syllables[t])
        int_color = _get_syllable_color(int_syll)
        draw.rectangle([0, 0, canvas_w, bar_height], fill=int_color)
        draw.text((5, 5), f"Int: S{int_syll}", font=font, fill=(255, 255, 255))

        frame = _ensure_rgb_uint8(frames[t])
        frame_img = Image.fromarray(frame)
        canvas.paste(frame_img, (0, bar_height))

        kp_syll = int(keypoint_syllables[t])
        kp_color = _get_syllable_color(kp_syll)
        draw.rectangle([0, bar_height + H, canvas_w, bar_height + H + bar_height], fill=kp_color)
        draw.text((5, bar_height + H + 5), f"KP: S{kp_syll}", font=font, fill=(255, 255, 255))

        draw.text(
            (canvas_w - 80, bar_height + H + bar_height + 2),
            f"Frame {t}/{n_frames}",
            font=font,
            fill=(0, 0, 0),
        )

        writer.send(np.asarray(canvas, dtype=np.uint8))

    writer.close()
    return output_path


def plot_reconstruction_timeseries(
    original_data: np.ndarray,
    reconstructed_data: np.ndarray,
    clip_idx: int = 0,
    n_dims: int = 5,
    title_prefix: str = "Reconstruction",
    max_frames: int = 500,
) -> "Figure":
    if not HAS_MPL:
        raise ImportError("matplotlib not installed")

    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    n_frames = min(original_data.shape[1], reconstructed_data.shape[1], max_frames)
    t = np.arange(n_frames)

    for dim_idx in range(n_dims):
        ax = axes[dim_idx]

        orig = original_data[clip_idx, :n_frames, dim_idx]
        recon = reconstructed_data[clip_idx, :n_frames, dim_idx]

        ax.plot(t, orig, label="Original", color="blue", alpha=0.7, linewidth=1)
        ax.plot(t, recon, label="Reconstructed", color="red", alpha=0.7, linewidth=1)

        mse = np.mean((orig - recon) ** 2)
        ax.set_ylabel(f"Dim {dim_idx}")
        ax.set_title(f"Dimension {dim_idx} (MSE: {mse:.4f})", fontsize=10)
        if dim_idx == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Frame")
    fig.suptitle(f"{title_prefix} - Clip {clip_idx}", fontsize=12)
    plt.tight_layout()

    return fig


def plot_reconstruction_comparison(
    int_original: np.ndarray,
    int_reconstructed: np.ndarray,
    kp_original: np.ndarray,
    kp_reconstructed: np.ndarray,
    clip_idx: int = 0,
    n_dims: int = 5,
    max_frames: int = 500,
) -> "Figure":
    if not HAS_MPL:
        raise ImportError("matplotlib not installed")

    fig, axes = plt.subplots(n_dims, 2, figsize=(14, 2 * n_dims), sharex=True)

    n_frames_int = min(int_original.shape[1], int_reconstructed.shape[1], max_frames)
    n_frames_kp = min(kp_original.shape[1], kp_reconstructed.shape[1], max_frames)
    t_int = np.arange(n_frames_int)
    t_kp = np.arange(n_frames_kp)

    for dim_idx in range(n_dims):
        ax_int = axes[dim_idx, 0]
        orig_int = int_original[clip_idx, :n_frames_int, dim_idx]
        recon_int = int_reconstructed[clip_idx, :n_frames_int, dim_idx]

        ax_int.plot(t_int, orig_int, label="Original", color="blue", alpha=0.7, linewidth=1)
        ax_int.plot(t_int, recon_int, label="Reconstructed", color="red", alpha=0.7, linewidth=1)

        mse_int = np.mean((orig_int - recon_int) ** 2)
        ax_int.set_ylabel(f"Dim {dim_idx}")
        if dim_idx == 0:
            ax_int.set_title(f"Intention (MSE: {mse_int:.4f})", fontsize=10)
            ax_int.legend(loc="upper right", fontsize=8)
        else:
            ax_int.set_title(f"MSE: {mse_int:.4f}", fontsize=9)

        ax_kp = axes[dim_idx, 1]
        kp_dim = min(dim_idx, kp_original.shape[2] - 1)
        orig_kp = kp_original[clip_idx, :n_frames_kp, kp_dim]
        recon_kp = kp_reconstructed[clip_idx, :n_frames_kp, kp_dim]

        if orig_kp.ndim > 1:
            orig_kp = orig_kp.flatten()[:n_frames_kp]
            recon_kp = recon_kp.flatten()[:n_frames_kp]

        ax_kp.plot(t_kp, orig_kp, label="Original", color="blue", alpha=0.7, linewidth=1)
        ax_kp.plot(t_kp, recon_kp, label="Reconstructed", color="red", alpha=0.7, linewidth=1)

        mse_kp = np.mean((orig_kp - recon_kp) ** 2)
        if dim_idx == 0:
            ax_kp.set_title(f"Keypoint (MSE: {mse_kp:.4f})", fontsize=10)
            ax_kp.legend(loc="upper right", fontsize=8)
        else:
            ax_kp.set_title(f"MSE: {mse_kp:.4f}", fontsize=9)

    axes[-1, 0].set_xlabel("Frame")
    axes[-1, 1].set_xlabel("Frame")
    fig.suptitle(f"Reconstruction Comparison - Clip {clip_idx}", fontsize=12)
    plt.tight_layout()

    return fig


def create_all_plots(
    int_model: dict,
    kp_model: dict,
    int_results: dict,
    kp_results: dict,
    int_original: np.ndarray | None = None,
    int_reconstructed: np.ndarray | None = None,
    kp_original: np.ndarray | None = None,
    kp_reconstructed: np.ndarray | None = None,
    eml_intention: EMLResult | None = None,
    eml_keypoint: EMLResult | None = None,
    int_model_names: list[str] | None = None,
    kp_model_names: list[str] | None = None,
    n_recon_dims: int = 5,
) -> dict[str, "Figure"]:
    plots = {}

    plots["intention/duration_dist"] = plot_duration_distribution(
        int_results, title="Intention Syllable Duration Distribution"
    )
    plots["keypoint/duration_dist"] = plot_duration_distribution(
        kp_results, title="Keypoint Syllable Duration Distribution"
    )

    plots["intention/syllable_freq"] = plot_syllable_frequencies(
        int_results, title="Intention Syllable Frequencies"
    )
    plots["keypoint/syllable_freq"] = plot_syllable_frequencies(
        kp_results, title="Keypoint Syllable Frequencies"
    )

    plots["intention/transition_matrix"] = plot_transition_matrix(
        int_model, title="Intention Transition Matrix", results=int_results
    )
    plots["keypoint/transition_matrix"] = plot_transition_matrix(
        kp_model, title="Keypoint Transition Matrix", results=kp_results
    )

    plots["dual/duration_comparison"] = plot_dual_duration_comparison(
        int_results, kp_results, title="Duration Distribution: Intention vs Keypoint"
    )

    plots["dual/syllable_confusion"] = plot_syllable_consistency_confusion(
        int_results, kp_results, title="Intention vs Keypoint Syllable Co-occurrence"
    )

    if int_original is not None and int_reconstructed is not None:
        plots["intention/reconstruction"] = plot_reconstruction_timeseries(
            int_original, int_reconstructed,
            clip_idx=0, n_dims=n_recon_dims, title_prefix="Intention Reconstruction"
        )

    if kp_original is not None and kp_reconstructed is not None:
        plots["keypoint/reconstruction"] = plot_reconstruction_timeseries(
            kp_original, kp_reconstructed,
            clip_idx=0, n_dims=n_recon_dims, title_prefix="Keypoint Reconstruction"
        )

    if (int_original is not None and int_reconstructed is not None and
        kp_original is not None and kp_reconstructed is not None):
        plots["dual/reconstruction_comparison"] = plot_reconstruction_comparison(
            int_original, int_reconstructed,
            kp_original, kp_reconstructed,
            clip_idx=0, n_dims=n_recon_dims,
        )

    if eml_intention is not None and int_model_names is not None:
        plots["intention/eml_ensemble"] = plot_eml_ensemble(
            eml_intention, int_model_names, title="Intention EML Ensemble"
        )

    if eml_keypoint is not None and kp_model_names is not None:
        plots["keypoint/eml_ensemble"] = plot_eml_ensemble(
            eml_keypoint, kp_model_names, title="Keypoint EML Ensemble"
        )

    return plots
