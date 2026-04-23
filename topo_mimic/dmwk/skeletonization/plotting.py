"""Functions for plotting the skeletons and post-processed clusters"""

from dataclasses import dataclass
import numpy as np
from collections import defaultdict
from scipy.linalg import subspace_angles, eigh
import matplotlib.pyplot as plt
import pandas as pd

from topo_mimic.dmwk.utils.load_utils import load_rollout_from_source


@dataclass
class BehaviorSegment:
    clip_id: int
    behavior: str
    start_frame: int
    end_frame: int

    @property
    def frames(self) -> np.ndarray:
        return np.arange(self.start_frame, self.end_frame + 1)

    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame + 1


PRIMITIVE_BEHAVIORS = {
    "Walking", "Immobility", "Grooming", "Orienting",
    "Supported Rear", "Unsupported Rear"
}
COMPOSITIONAL_BEHAVIORS = {"Left Turn", "Right Turn"}


def load_behavior_segments(csv_path: str) -> dict[str, list[BehaviorSegment]]:
    """
    Load frame-level behavior segments from behavior_labels.csv.

    Args:
        csv_path: Path to behavior_labels.csv with columns:
            Video, Behavior, Start_Time, End_Time, Start_Frame, End_Frame

    Returns:
        Dict mapping behavior name -> list of BehaviorSegment
    """
    df = pd.read_csv(csv_path)
    segments = defaultdict(list)

    for _, row in df.iterrows():
        start_frame = int(row["Start_Frame"])
        end_frame = int(row["End_Frame"])
        if start_frame == end_frame == 0:
            continue

        video = str(row["Video"])
        clip_id = int(video.replace("clip_", "").replace(".mp4", ""))
        behavior = str(row["Label"] if "Label" in df.columns else row["Behavior"])

        segments[behavior].append(BehaviorSegment(
            clip_id=clip_id,
            behavior=behavior,
            start_frame=start_frame,
            end_frame=end_frame,
        ))

    return dict(segments)


def plot_cluster_subspace_mds_separate(
    cluster_cycles_by_id: dict,
    cluster_labels: dict,
    turn_directions: dict = None,
    target_clusters: list = None,
    rodent_rollout_dir: str = "",
    behavior_csv_path: str = "",
    behaviors: list[str] | None = None,
    k_cycles: int = 5,
    s_top: int = 10,
    pad: int = 0,
    state_key: str = "qpos",
    sample_strategy: str = "random",
    seed: int = 0,
    plot: bool = True,
    figsize: tuple[float, float] = (7, 6),
    save_path: str | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    MDS projection showing cluster subspaces and individual behavior subspaces.

    Unlike plot_cluster_subspace_mds which takes unions of behaviors,
    this function shows each behavior (Rear, Walk, Left Turn, Right Turn, etc.)
    as a separate point in MDS space.

    Args:
        cluster_cycles_by_id: Dict cluster_id -> list of (clip_id, frames)
        cluster_labels: Dict cluster_id -> label string
        target_clusters: List of cluster IDs or names to include
        rodent_rollout_dir: Path to rollout data
        behavior_csv_path: Path to behavior_labels.csv (frame-level format)
        behaviors: List of behavior names to include, or None for all
        k_cycles: Number of cycles to sample per cluster
        s_top: Top eigenvectors to use for subspace
        pad: Frame padding around cycle indices
        state_key: Key for state data ("qpos" or "states")
        sample_strategy: "random", "longest", or "shortest"
        seed: Random seed
        plot: Whether to generate plot
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Tuple of (MDS coordinates, labels, types)
    """
    rng = np.random.default_rng(seed)

    if turn_directions is None:
        turn_directions = {}
    if target_clusters is None:
        target_clusters = list(cluster_cycles_by_id.keys())

    def get_cluster_display_name(cid: int) -> str:
        label = cluster_labels.get(cid, f"C{cid}")
        turn = turn_directions.get(cid, "")
        if turn:
            turn_short = {"left": "L", "right": "R", "straight": "S", "non-deterministic": "ND"}.get(turn.lower(), "")
            if turn_short:
                label = f"[{turn_short}] {label}"
        return label

    def load_qpos(clip_id):
        roll = load_rollout_from_source(int(clip_id), rodent_rollout_dir)
        try:
            Q = np.asarray(roll[state_key])
        except KeyError:
            Q = np.asarray(roll["states"])
        return Q[:, 7:]

    def to_cid(x):
        if isinstance(x, int):
            return x
        matches = [k for k, v in cluster_labels.items() if v == str(x)]
        if not matches:
            raise ValueError(f"Cluster '{x}' not found")
        return matches[0]

    behavior_segments = load_behavior_segments(behavior_csv_path)

    if behaviors is None:
        behaviors = sorted(behavior_segments.keys())

    bases = {}
    labels_pretty = {}
    types = {}
    s_global = None

    cids = [to_cid(c) for c in target_clusters]
    for cid in cids:
        entries = cluster_cycles_by_id.get(cid, [])
        if not entries:
            continue

        qcache = {}
        cycle_pool = []
        for cc, arr in entries:
            cc = int(cc)
            if cc not in qcache:
                qcache[cc] = load_qpos(cc)
            a = np.asarray(arr, int)
            raw_len = len(np.unique(a))
            cycle_pool.append({"clip": cc, "frames": a, "raw_len": raw_len})

        if not cycle_pool:
            continue

        def build_Q_cyc(sel_items):
            by_clip = {}
            for it in sel_items:
                by_clip.setdefault(it["clip"], []).append(it["frames"])
            chunks = []
            for cc, fr_lists in by_clip.items():
                Qc = qcache.get(cc, load_qpos(cc))
                T = Qc.shape[0]
                union = set()
                for a in fr_lists:
                    lb = np.clip(a - pad, 0, T - 1)
                    ub = np.clip(a + pad, 0, T - 1)
                    for s_, e_ in zip(lb, ub):
                        union.update(range(int(s_), int(e_) + 1))
                if union:
                    idx = np.array(sorted(union), int)
                    chunks.append(Qc[idx])
            return np.vstack(chunks) if chunks else None

        n_eff = min(k_cycles, len(cycle_pool))
        if n_eff <= 0:
            continue

        perm = rng.permutation(len(cycle_pool))
        shuffled = [cycle_pool[i] for i in perm]

        if sample_strategy == "longest":
            sel = sorted(shuffled, key=lambda t: t["raw_len"], reverse=True)[:n_eff]
        elif sample_strategy == "shortest":
            sel = sorted(shuffled, key=lambda t: t["raw_len"])[:n_eff]
        else:
            idx = rng.choice(len(cycle_pool), size=n_eff, replace=False)
            sel = [cycle_pool[i] for i in idx]

        Q_cyc = build_Q_cyc(sel)
        if Q_cyc is None or Q_cyc.shape[0] < 2:
            continue

        Qc = Q_cyc - Q_cyc.mean(0, keepdims=True)
        Cc = np.cov(Qc.T)
        e, U = np.linalg.eigh(Cc)
        ord_c = e.argsort()[::-1]
        U = U[:, ord_c]
        s_use = min(s_top, U.shape[0])

        key = f"cycle_{cid}"
        bases[key] = U[:, :s_use]
        labels_pretty[key] = get_cluster_display_name(cid)
        types[key] = "cycle"
        s_global = s_use if s_global is None else min(s_global, s_use)

    for behavior_name in behaviors:
        segments = behavior_segments.get(behavior_name, [])
        if not segments:
            continue

        all_frames_data = []
        for seg in segments:
            try:
                Q = load_qpos(seg.clip_id)
                start = max(0, seg.start_frame)
                end = min(Q.shape[0], seg.end_frame + 1)
                if end > start:
                    all_frames_data.append(Q[start:end])
            except Exception:
                continue

        if not all_frames_data:
            continue

        Q_full = np.vstack(all_frames_data)
        if Q_full.shape[0] < 2:
            continue

        Qf = Q_full - Q_full.mean(0, keepdims=True)
        Cf = np.cov(Qf.T)
        e, U = np.linalg.eigh(Cf)
        ord_f = e.argsort()[::-1]
        U = U[:, ord_f]
        s_use = min(s_top, U.shape[0])

        behavior_type = "primitive" if behavior_name in PRIMITIVE_BEHAVIORS else "compositional"
        key = f"behavior_{behavior_name.replace(' ', '_')}"
        bases[key] = U[:, :s_use]
        labels_pretty[key] = behavior_name
        types[key] = behavior_type
        s_global = s_use if s_global is None else min(s_global, s_use)

    keys = sorted(bases.keys())
    n = len(keys)
    if n == 0 or s_global is None:
        raise RuntimeError("No valid bases for MDS.")

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            Ui = bases[keys[i]][:, :s_global]
            Uj = bases[keys[j]][:, :s_global]
            ang = subspace_angles(Ui, Uj) * 180.0 / np.pi
            d = ang.mean()
            D[i, j] = D[j, i] = d

    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    pos = w > 0
    if pos.sum() < 2:
        raise RuntimeError("Not enough positive eigenvalues for 2D MDS.")
    k = 2
    lam = np.sqrt(w[:k])
    X = V[:, :k] * lam

    if plot:
        fig, ax = plt.subplots(figsize=figsize)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        cycle_marker = "o"
        primitive_marker = "s"
        compositional_marker = "^"
        cycle_size = 80
        behavior_size = 120

        cycle_first = True
        primitive_first = True
        compositional_first = True

        label_xy = X.copy()
        label_xy[:, 0] += 0.04
        label_xy[:, 1] += 0.04

        min_dist = 0.08
        for _ in range(200):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = label_xy[i, 0] - label_xy[j, 0]
                    dy = label_xy[i, 1] - label_xy[j, 1]
                    dist2 = dx * dx + dy * dy
                    if dist2 < min_dist * min_dist:
                        dist = np.sqrt(dist2) + 1e-8
                        push = 0.5 * (min_dist - dist)
                        ux = dx / dist
                        uy = dy / dist
                        label_xy[i, 0] += ux * push
                        label_xy[i, 1] += uy * push
                        label_xy[j, 0] -= ux * push
                        label_xy[j, 1] -= uy * push
                        moved = True
            if not moved:
                break

        for i, key in enumerate(keys):
            x, y = X[i, 0], X[i, 1]
            lx, ly = label_xy[i, 0], label_xy[i, 1]
            lbl = labels_pretty[key]
            t = types[key]

            color_idx = i % len(palette)
            color = palette[color_idx]

            if t == "cycle":
                ax.scatter(
                    x, y, s=cycle_size, marker=cycle_marker,
                    edgecolor="k", linewidth=0.6, zorder=3, color=color,
                    label="Cluster" if cycle_first else None,
                )
                cycle_first = False
            elif t == "primitive":
                ax.scatter(
                    x, y, s=behavior_size, marker=primitive_marker,
                    edgecolor="k", linewidth=0.8, zorder=4, color=color,
                    label="Primitive behavior" if primitive_first else None,
                )
                primitive_first = False
            else:
                ax.scatter(
                    x, y, s=behavior_size, marker=compositional_marker,
                    edgecolor="k", linewidth=0.8, zorder=4, color=color,
                    label="Compositional behavior" if compositional_first else None,
                )
                compositional_first = False

            ax.text(lx, ly, lbl, fontsize=11, ha="left", va="bottom")

        ax.set_xlabel("MDS dimension 1", fontsize=13)
        ax.set_ylabel("MDS dimension 2", fontsize=13)
        ax.axhline(0, linewidth=0.7, color="k", alpha=0.2)
        ax.axvline(0, linewidth=0.7, color="k", alpha=0.2)

        handles, leg_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, fontsize=10, loc="best")

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=True)
            fig.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)

        plt.show()

    return X, [labels_pretty[k] for k in keys], [types[k] for k in keys]
