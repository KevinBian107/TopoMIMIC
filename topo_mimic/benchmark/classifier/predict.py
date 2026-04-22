"""Load trained classifier checkpoints and emit per-window predictions.

Used both by cluster_eval (analysis stage) and by smoke tests. Predictions
are produced as an ensemble over all 5 fold checkpoints: softmax-mean across
folds. This gives an approximately unbiased prediction for clips that
appear in the fold-0 training set but in some other fold's val set.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from topo_mimic.benchmark.classifier import TOP_8_TRANSITIONS
from topo_mimic.benchmark.classifier.model import BehaviorClassifier
from topo_mimic.benchmark.classifier.preprocessing import normalize_joints


class LoadedClassifier:
    """Holds all 5 fold params + per-fold joint normalization stats + per-fold
    val_clips so that callers can run leave-clip-out inference.

    Two inference modes:
      - ``ensemble_softmax(X_unnorm)``: simple average of softmax across all
        5 folds. Biased upward for inputs whose source clip appeared in ≥1
        fold's training split (4/5 of folds by construction).
      - ``held_out_softmax(X_unnorm, clip_ids)``: for each input, uses ONLY
        the fold(s) whose val_clips contain that input's clip. If no fold
        held the clip out (the clip was never labeled), falls back to the
        full ensemble and marks a warning via the returned ``n_folds_used``
        array.
    """

    def __init__(self, ver_dir: Path):
        self.ver_dir = ver_dir
        meta_path = ver_dir / "metadata.json"
        with open(meta_path) as f:
            self.metadata: dict = json.load(f)
        self.K: int = int(self.metadata["K"])
        self.joint_start: int = int(self.metadata["joint_start"])
        self.labels: list[str] = list(self.metadata["label_vocab"])
        if self.labels != TOP_8_TRANSITIONS:
            raise ValueError(
                f"Checkpoint label vocab {self.labels} != TOP_8_TRANSITIONS"
            )
        hidden = tuple(self.metadata["hidden_sizes"])
        dropout = float(self.metadata["dropout"])
        self.model = BehaviorClassifier(
            n_classes=len(self.labels), hidden_sizes=hidden, dropout=dropout,
        )

        init_rng = jax.random.PRNGKey(0)
        D = int(self.metadata.get("D_preprocessed", 74))
        example = jnp.zeros((1, self.K, D), dtype=jnp.float32)
        template = self.model.init({"params": init_rng, "dropout": init_rng},
                                   example, training=False)

        self.fold_params: list[dict] = []
        self.fold_norm: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(int(self.metadata["n_folds"])):
            pth = ver_dir / f"fold_{i}.msgpack"
            with open(pth, "rb") as f:
                raw = f.read()
            self.fold_params.append(serialization.from_bytes(template, raw))
            arr = np.load(ver_dir / f"norm_stats_fold_{i}.npz")
            self.fold_norm.append((arr["mean"], arr["std"]))

        # per_fold_val_clips is the honest leave-clip-out ticket; older
        # checkpoints may not have it, in which case only ensemble inference
        # is safe (and flagged).
        raw_val = self.metadata.get("per_fold_val_clips")
        if raw_val is None:
            self.fold_val_clips: list[set[int]] = []
        else:
            self.fold_val_clips = [set(int(c) for c in fv) for fv in raw_val]

        @jax.jit
        def _forward(params, x):
            return self.model.apply(params, x, training=False)

        self._forward = _forward

    def _per_fold_softmax(
        self, fold_idx: int, X_unnorm: np.ndarray, batch_size: int = 512,
    ) -> np.ndarray:
        params = self.fold_params[fold_idx]
        mean, std = self.fold_norm[fold_idx]
        X_n = normalize_joints(X_unnorm, self.joint_start, mean, std)
        parts = []
        for i in range(0, len(X_n), batch_size):
            logits = self._forward(params, jnp.asarray(X_n[i:i + batch_size]))
            parts.append(np.asarray(jax.nn.softmax(logits, axis=-1)))
        return np.concatenate(parts, axis=0)

    def ensemble_softmax(self, X_unnorm: np.ndarray, batch_size: int = 512) -> np.ndarray:
        soft_sum = np.zeros((len(X_unnorm), len(self.labels)), dtype=np.float32)
        for fi in range(len(self.fold_params)):
            soft_sum += self._per_fold_softmax(fi, X_unnorm, batch_size)
        return soft_sum / len(self.fold_params)

    def held_out_softmax(
        self,
        X_unnorm: np.ndarray,
        clip_ids: np.ndarray,
        batch_size: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Leave-clip-out ensemble. Returns (softmax, n_folds_used).

        For each input index i, the returned ``softmax[i]`` is the mean
        softmax over the set of folds whose val_clips contain ``clip_ids[i]``.
        ``n_folds_used[i]`` counts how many folds contributed (usually 1 with
        5-fold clip-level CV, occasionally 0 for clips that weren't in any
        fold's val split — those fall back to the full ensemble).
        """
        N = len(X_unnorm)
        if not self.fold_val_clips:
            # old checkpoint without val_clips metadata — fall back to full
            # ensemble; caller can detect the all-5 count.
            soft = self.ensemble_softmax(X_unnorm, batch_size=batch_size)
            return soft, np.full(N, len(self.fold_params), dtype=np.int32)

        clip_arr = np.asarray(clip_ids, dtype=np.int64)
        soft_accum = np.zeros((N, len(self.labels)), dtype=np.float32)
        count_accum = np.zeros(N, dtype=np.int32)
        for fi, vc in enumerate(self.fold_val_clips):
            mask = np.array([int(c) in vc for c in clip_arr], dtype=bool)
            if not mask.any():
                continue
            fold_soft = self._per_fold_softmax(fi, X_unnorm[mask], batch_size=batch_size)
            soft_accum[mask] += fold_soft
            count_accum[mask] += 1

        # Accumulate-then-normalize (mean across contributing folds) for the
        # covered rows, then overwrite uncovered rows with the already-
        # normalized full ensemble. `n_folds_used` reports how many folds
        # held each clip out (0 == fell back to full ensemble, count
        # stored as n_folds for bookkeeping).
        covered = count_accum > 0
        if covered.any():
            denom = count_accum[covered].astype(np.float32)[:, None]
            soft_accum[covered] = soft_accum[covered] / denom

        uncovered = ~covered
        if uncovered.any():
            soft_accum[uncovered] = self.ensemble_softmax(
                X_unnorm[uncovered], batch_size=batch_size,
            )
            count_accum[uncovered] = len(self.fold_params)  # for reporting only

        return soft_accum, count_accum
