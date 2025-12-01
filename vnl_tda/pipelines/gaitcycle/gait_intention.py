'''Gait cycle + intention analysis entry point, all the functions used for gait + intention analysis should be called here,
some of the functions here are for gait x intention analysis, which is not an essential part of this study'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde
from scipy.signal import butter, filtfilt, hilbert
from sklearn.preprocessing import StandardScaler

from vnl_tda.pipelines.gaitcycle.utils import (
    create_time_windows,
    create_color_by_phase,
    sample_diverse_windows,
    apply_umap_to_intention_clusters,
    cluster_gait_distribution,
    visualize_gait_manifold
    )


def sampled_umap_with_gait(
    int_data: np.ndarray,
    jf_array: np.ndarray,
    window_size: int = 31,
    stride: int = 1,
    n_samples: int = 10000,
) -> dict:
    """
    UMAP on sampled windows with gait analysis and intention correlations.

    Args:
        int_data: Intention data array
        jf_array: Joint force array of shape (n_clips, n_frames, n_joints)
        window_size: Size of sliding window
        stride: Step size between windows
        n_samples: Number of samples for UMAP

    Returns:
        Dict with 'embedded', 'clusters', 'gait_metrics', 'cluster_labels'
    """

    print(f"Processing data with paper methods, window_size={window_size}")
    windows, center_frames = create_time_windows(int_data, window_size, stride)

    # sample diverse windows
    sampled_windows, sample_indices = sample_diverse_windows(
        windows, n_samples=min(n_samples, len(windows))
    )
    sampled_center_frames = center_frames[sample_indices]

    # extract gait metrics
    gait_metrics = estimate_gait_metrics(jf_array, sampled_center_frames, window_size)

    if len(gait_metrics["frequencies"]) == 0:
        raise ValueError(
            "No valid gait metrics were extracted! Check force array shapes and thresholding."
        )

    valid_indices = np.where(
        np.isin(sampled_center_frames, gait_metrics["valid_center_frames"])
    )[0]

    if len(valid_indices) > 10:
        print(f"Using {len(valid_indices)} clusters with valid gait metrics")
        clusters_for_umap = sampled_windows[valid_indices]
    else:
        print("Using all sampled clusters for UMAP")
        clusters_for_umap = sampled_windows

    embedded = apply_umap_to_intention_clusters(clusters_for_umap, n_components=3)

    if len(gait_metrics["phases"]) > len(embedded):
        for key in gait_metrics:
            if isinstance(gait_metrics[key], np.ndarray):
                gait_metrics[key] = gait_metrics[key][: len(embedded)]
    elif len(gait_metrics["phases"]) < len(embedded):
        embedded = embedded[: len(gait_metrics["phases"])]

    n_clusters = min(4, len(embedded) // 50)
    n_clusters = max(2, n_clusters)
    cluster_labels = cluster_gait_distribution(
        embedded, gait_metrics, n_clusters
    )

    visualize_gait_manifold(embedded, gait_metrics)
    analyze_cyclic_structure(embedded, gait_metrics)

    return {
        "embedded": embedded,
        "clusters": clusters_for_umap,
        "gait_metrics": gait_metrics,
        "cluster_labels": cluster_labels,
    }


def estimate_gait_metrics(
    cfrc: np.ndarray,
    center_frames: np.ndarray,
    window_size: int = 31,
    fps: int = 30,
) -> dict:
    """
    Extract gait metrics from force data.

    Args:
        cfrc: Contact force array of shape (n_clips, n_frames, n_feet)
        center_frames: Array of center frame indices
        window_size: Size of analysis window
        fps: Frames per second for frequency computation

    Returns:
        Dict with 'phases', 'frequencies', 'feet_in_stance', etc.
    """
    frame_mapping = {}
    global_idx = 0

    for clip_idx in range(cfrc.shape[0]):
        n_frames = cfrc[clip_idx].shape[0]
        for frame_idx in range(n_frames):
            frame_mapping[global_idx] = (clip_idx, frame_idx)
            global_idx += 1

    phases = []
    frequencies = []
    feet_in_stance = []
    stance_durations = []
    swing_durations = []
    duty_factors = []
    asymmetry_values = []
    clip_indices = []
    valid_center_frames = []

    for frame_idx, center_frame in enumerate(center_frames):
        if center_frame in frame_mapping:
            clip_idx, clip_frame = frame_mapping[center_frame]

            clip_data = cfrc[clip_idx]
            feet_forces = clip_data

            half_window = window_size // 2
            if (
                clip_frame >= half_window
                and clip_frame < clip_data.shape[0] - half_window
            ):
                try:
                    # threshold force indicating stance
                    contact_threshold = 0.1
                    contacts = feet_forces > contact_threshold

                    # get number of feet in stance at center frame
                    feet_in_stance.append(np.sum(contacts[clip_frame]))

                    # calculate duty factor (as in the paper)
                    # extract a longer cluster for better stance/swing detection
                    extended_start = max(0, clip_frame - window_size)
                    extended_end = min(clip_data.shape[0], clip_frame + window_size)
                    extended_contacts = contacts[extended_start:extended_end]

                    # calculate stance and swing durations
                    stance_duration = []
                    swing_duration = []

                    for foot in range(extended_contacts.shape[1]):
                        foot_contacts = extended_contacts[:, foot]

                        # find transitions (0->1 is swing->stance, 1->0 is stance->swing)
                        transitions = np.diff(foot_contacts.astype(int))
                        swing_to_stance = np.where(transitions == 1)[0]
                        stance_to_swing = np.where(transitions == -1)[0]

                        if len(swing_to_stance) > 0 and len(stance_to_swing) > 0:
                            # calculate durations of complete stance and swing phases
                            complete_stances = []
                            complete_swings = []

                            for i in range(len(swing_to_stance)):
                                for j in range(len(stance_to_swing)):
                                    if stance_to_swing[j] > swing_to_stance[i]:
                                        # found a complete stance
                                        complete_stances.append(
                                            stance_to_swing[j] - swing_to_stance[i]
                                        )
                                        break

                            for i in range(len(stance_to_swing)):
                                for j in range(len(swing_to_stance)):
                                    if swing_to_stance[j] > stance_to_swing[i]:
                                        # found a complete swing
                                        complete_swings.append(
                                            swing_to_stance[j] - stance_to_swing[i]
                                        )
                                        break

                            if complete_stances:
                                stance_duration.append(np.mean(complete_stances) / fps)
                            if complete_swings:
                                swing_duration.append(np.mean(complete_swings) / fps)

                    if stance_duration and swing_duration:
                        avg_stance = np.mean(stance_duration)
                        avg_swing = np.mean(swing_duration)
                        avg_duty_factor = avg_stance / (avg_stance + avg_swing)

                        stance_durations.append(avg_stance)
                        swing_durations.append(avg_swing)
                        duty_factors.append(avg_duty_factor)
                    else:
                        # default values if we couldn't calculate
                        stance_durations.append(0.1)
                        swing_durations.append(0.05)
                        duty_factors.append(0.67)

                    # asymmetry findings
                    n_feet = len(feet_forces)
                    if n_feet >= 2:
                        mid_idx = n_feet // 2
                        left_contacts = contacts[clip_frame, :mid_idx]
                        right_contacts = contacts[clip_frame, mid_idx : 2 * mid_idx]
                        asymmetry = np.mean(left_contacts) - np.mean(right_contacts)
                        asymmetry_values.append(asymmetry)
                    else:
                        asymmetry_values.append(0)

                    # phase and frequency using Hilbert transform as in the paper & use the mean vertical force across all feet
                    mean_force = np.mean(feet_forces, axis=1)

                    # cluster around center frame
                    cluster_start = max(0, clip_frame - half_window)
                    cluster_end = min(clip_data.shape[0], clip_frame + half_window)
                    cluster_force = mean_force[cluster_start:cluster_end]

                    if len(cluster_force) >= 5:
                        # filter the signal
                        b, a = butter(4, [0.5, 10], fs=fps, btype="band")
                        try:
                            filtered_force = filtfilt(b, a, cluster_force)

                            # hilbert transform
                            analytic_signal = hilbert(filtered_force)

                            # phase at center
                            center_idx = clip_frame - cluster_start
                            if center_idx < len(analytic_signal):
                                phase = np.angle(analytic_signal[center_idx])

                                # frequency
                                unwrapped_phase = np.unwrap(np.angle(analytic_signal))
                                if len(unwrapped_phase) > 1:
                                    freq = (
                                        np.mean(np.diff(unwrapped_phase))
                                        / (2.0 * np.pi)
                                        * fps
                                    )
                                else:
                                    freq = 1.0

                                phases.append(phase)
                                frequencies.append(freq)

                                # store valid center frame and clip index
                                valid_center_frames.append(center_frame)
                                clip_indices.append(clip_idx)
                        except:
                            # skip if filtering fails
                            continue
                    else:
                        # skip if cluster is too small
                        continue
                except Exception as e:
                    # skip if processing fails
                    continue

    metrics = {
        "phases": np.array(phases) if phases else np.array([]),
        "frequencies": np.array(frequencies) if frequencies else np.array([]),
        "feet_in_stance": np.array(feet_in_stance) if feet_in_stance else np.array([]),
        "stance_durations": (
            np.array(stance_durations) if stance_durations else np.array([])
        ),
        "swing_durations": (
            np.array(swing_durations) if swing_durations else np.array([])
        ),
        "duty_factor": np.array(duty_factors) if duty_factors else np.array([]),
        "asymmetry": np.array(asymmetry_values) if asymmetry_values else np.array([]),
        "clip_indices": np.array(clip_indices) if clip_indices else np.array([]),
        "valid_center_frames": (
            np.array(valid_center_frames) if valid_center_frames else np.array([])
        ),
    }

    print(
        f"Extracted gait metrics for {len(phases)} frames out of {len(center_frames)} requested frames"
    )

    return metrics


def compute_phase_using_hilbert(
    intentions: np.ndarray,
    fps: int = 30,
) -> dict:
    """
    Compute instantaneous phases using Hilbert transform.

    Args:
        intentions: Time series array of shape (n_timesteps, n_features)
        fps: Sampling rate for frequency computation

    Returns:
        Dict with 'phases' and 'frequencies' arrays
    """
    n_timesteps, n_features = intentions.shape

    all_phases = np.zeros((n_timesteps, n_features))
    all_frequencies = np.zeros((n_timesteps, n_features))

    # process each component separately
    for i in range(n_features):
        # extract this component's time series
        signal = intentions[:, i]

        # filter the signal as in the paper (bandpass filter)
        b, a = butter(4, [0.5, 10], fs=fps, btype="band")
        try:
            filtered_signal = filtfilt(b, a, signal)
        except ValueError:
            # for short signals, use a simpler filter
            filtered_signal = signal - np.mean(signal)

        # Hilbert transform to compute the analytic signal
        try:
            analytic_signal = hilbert(filtered_signal)

            # extract instantaneous phase and unwrapped phase
            instantaneous_phase = np.angle(analytic_signal)
            unwrapped_phase = np.unwrap(instantaneous_phase)

            # instantaneous frequency from unwrapped phase
            instantaneous_frequency = np.diff(unwrapped_phase) / (2.0 * np.pi) * fps
            instantaneous_frequency = np.append(
                instantaneous_frequency, instantaneous_frequency[-1]
            )

            # smooth frequency with Savitzky-Golay filter as in the paper
            cluster_length = min(15, len(instantaneous_frequency) // 2)
            if cluster_length > 3 and cluster_length % 2 == 1:
                from scipy.signal import savgol_filter

                instantaneous_frequency = savgol_filter(
                    instantaneous_frequency, cluster_length, 3
                )

            all_phases[:, i] = instantaneous_phase
            all_frequencies[:, i] = instantaneous_frequency

        except:
            all_phases[:, i] = np.linspace(-np.pi, np.pi, n_timesteps)
            all_frequencies[:, i] = np.ones(n_timesteps)

    return {"phases": all_phases, "frequencies": all_frequencies}


def analyze_cyclic_structure(
    embedded: np.ndarray,
    gait_metrics: dict,
) -> None:
    """
    Analyze and visualize cyclic structure of the gait manifold.

    Args:
        embedded: UMAP embedding of shape (n_samples, 3)
        gait_metrics: Dict with 'phases', 'frequencies', 'feet_in_stance', 'duty_factor'
    """
    min_size = min(len(embedded), len(gait_metrics["phases"]))
    phases = gait_metrics["phases"][:min_size]
    frequencies = gait_metrics["frequencies"][:min_size]
    feet_in_stance = gait_metrics["feet_in_stance"][:min_size]
    duty_factor = gait_metrics["duty_factor"][:min_size]

    plt.figure(figsize=(16, 6))

    # global phase vs. feet in stance
    plt.subplot(131)
    phase_bins = np.linspace(-np.pi, np.pi, 20)
    binned_phases = np.digitize(phases, phase_bins) - 1

    # calculate statistics for each bin
    mean_feet = np.zeros(len(phase_bins) - 1)
    std_feet = np.zeros(len(phase_bins) - 1)

    for i in range(len(phase_bins) - 1):
        bin_mask = binned_phases == i
        if np.sum(bin_mask) > 0:
            mean_feet[i] = np.mean(feet_in_stance[bin_mask])
            std_feet[i] = np.std(feet_in_stance[bin_mask])

    bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    plt.plot(bin_centers, mean_feet, "b-", linewidth=2)
    plt.fill_between(
        bin_centers, mean_feet - std_feet, mean_feet + std_feet, color="b", alpha=0.2
    )

    plt.xlabel("Global Phase (radians)")
    plt.ylabel("Feet in Stance")
    plt.title("Feet in Stance vs. Phase")
    plt.xlim(-np.pi, np.pi)
    plt.xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
    )

    # frequency distribution
    plt.subplot(132)
    plt.hist(frequencies, bins=30, density=True, alpha=0.7)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Probability Density")
    plt.title("Distribution of Walking Frequencies")

    z = np.polyfit(frequencies, duty_factor, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(np.min(frequencies), np.max(frequencies), 100)
    plt.plot(x_trend, p(x_trend), "r--", linewidth=2)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Duty Factor")
    plt.title("Duty Factor vs. Frequency")

    plt.tight_layout()
    plt.show()