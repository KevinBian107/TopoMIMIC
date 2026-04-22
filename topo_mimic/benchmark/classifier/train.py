"""Train the 8-class transition classifier with 5-fold clip-level CV.

Writes to `outputs/classifier_ckpts/qpos/v{hash}/`:
  - fold_{0..4}.msgpack   (Flax params per fold)
  - norm_stats_fold_{i}.npz  (joint mean/std per fold)
  - per_class_f1.csv
  - confusion_matrix.png + .csv (aggregated across folds)
  - training_curves.png
  - metadata.json  (K, label vocab, hash, gate pass/fail)

Usage:
    python3 -m topo_mimic.benchmark.classifier.train \
        --labels_csv behavior_labels/rodent_labels.csv \
        --raw_h5 data/rodent_ar.h5 \
        --output_dir outputs/classifier_ckpts/qpos \
        --K 100 \
        --epochs 50 \
        --batch_size 256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from topo_mimic.benchmark.classifier import TOP_8_TRANSITIONS, IDX_TO_LABEL
from topo_mimic.benchmark.classifier.data import (
    ClassifierDataset,
    build_full_dataset,
    normalize_with_train_stats,
    upsample_to_balance,
)
from topo_mimic.benchmark.classifier.model import BehaviorClassifier
from topo_mimic.benchmark.classifier.preprocessing import get_joint_start_index


@dataclass
class TrainConfig:
    labels_csv: str = "behavior_labels/rodent_labels.csv"
    raw_h5: str = "data/rodent_ar.h5"
    output_dir: str = "outputs/classifier_ckpts/qpos"
    K: int = 100
    exclude_xy: bool = True
    do_handle_rotation: bool = True
    pad_frames: int = 0
    n_folds: int = 5
    n_folds_seed: int = 42
    epochs: int = 50
    batch_size: int = 256
    lr: float = 5.0e-4
    weight_decay: float = 1.0e-4
    dropout: float = 0.1
    hidden_sizes: tuple[int, ...] = (512, 256, 128)
    upsample_train: bool = True
    upsample_seed: int = 7
    early_stop_patience: int = 5


def _csv_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def _per_class_f1_from_cm(cm: np.ndarray) -> np.ndarray:
    n = cm.shape[0]
    f1s = np.zeros(n, dtype=np.float64)
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s[i] = 2 * prec * rec / max(prec + rec, 1e-9)
    return f1s


def _save_confusion(cm: np.ndarray, labels: list[str], out_png: str, out_csv: str) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Transition classifier — 5-fold aggregated confusion")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]:d})",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=7,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.to_csv(out_csv)


def _save_curves(per_fold_history: list[dict], out_png: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    for i, hist in enumerate(per_fold_history):
        ax1.plot(hist["train_loss"], label=f"fold {i}")
        ax2.plot(hist["val_macro_f1"], label=f"fold {i}")
    ax1.set_title("train loss per epoch")
    ax2.set_title("val macro-F1 per epoch")
    for a in (ax1, ax2):
        a.set_xlabel("epoch")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _make_train_step(model: BehaviorClassifier, optimizer, class_weights):
    import jax
    import jax.numpy as jnp

    def loss_fn(params, x, y, key):
        logits = model.apply(params, x, training=True, rngs={"dropout": key})
        logp = jax.nn.log_softmax(logits)
        w = jnp.asarray(class_weights)
        return -(w[y] * logp[jnp.arange(x.shape[0]), y]).mean()

    @jax.jit
    def step(params, opt_state, x, y, key):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optimize_apply(params, updates)
        return params, opt_state, loss

    def optimize_apply(params, updates):
        import optax
        return optax.apply_updates(params, updates)

    return step


def train_one_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    cfg: TrainConfig,
    fold_idx: int,
) -> tuple[dict, dict, np.ndarray]:
    """Train one fold, return (best_params, history, val_cm)."""
    import jax
    import jax.numpy as jnp
    import optax

    key = jax.random.PRNGKey(cfg.n_folds_seed + fold_idx)

    model = BehaviorClassifier(
        n_classes=len(TOP_8_TRANSITIONS),
        hidden_sizes=tuple(cfg.hidden_sizes),
        dropout=cfg.dropout,
    )
    params = model.init({"params": key, "dropout": key},
                        jnp.asarray(X_tr[:1]), training=False)

    # per-fold class weights from train split only
    counts = np.array([(y_tr == c).sum() for c in range(len(TOP_8_TRANSITIONS))],
                      dtype=np.float64)
    safe = np.maximum(counts, 1)
    class_weights = len(y_tr) / (len(TOP_8_TRANSITIONS) * safe)
    class_weights = np.clip(class_weights, 0.3, 20.0).astype(np.float32)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(cfg.lr, weight_decay=cfg.weight_decay),
    )
    opt_state = tx.init(params)

    def loss_fn(params, x, y, key):
        logits = model.apply(params, x, training=True, rngs={"dropout": key})
        logp = jax.nn.log_softmax(logits)
        w = jnp.asarray(class_weights)
        return -(w[y] * logp[jnp.arange(x.shape[0]), y]).mean()

    @jax.jit
    def step(params, opt_state, x, y, key):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y, key)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_logits(params, x):
        return model.apply(params, x, training=False)

    history = {"train_loss": [], "val_macro_f1": []}
    best_f1 = -1.0
    best_params = params
    patience = cfg.early_stop_patience

    for epoch in range(cfg.epochs):
        perm = np.random.default_rng(epoch).permutation(len(X_tr))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            key, sub = jax.random.split(key)
            params, opt_state, loss = step(
                params, opt_state,
                jnp.asarray(X_tr[idx]), jnp.asarray(y_tr[idx]), sub,
            )
            epoch_loss += float(loss)
            n_batches += 1
        history["train_loss"].append(epoch_loss / max(n_batches, 1))

        # val in chunks
        val_logits = np.concatenate([
            np.asarray(eval_logits(params, jnp.asarray(X_val[i:i + 512])))
            for i in range(0, len(X_val), 512)
        ], axis=0)
        pred = val_logits.argmax(axis=1)
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        history["val_macro_f1"].append(float(f1))

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            patience = cfg.early_stop_patience
        else:
            patience -= 1
            if patience <= 0:
                break

    # final val confusion using best_params
    val_logits = np.concatenate([
        np.asarray(eval_logits(best_params, jnp.asarray(X_val[i:i + 512])))
        for i in range(0, len(X_val), 512)
    ], axis=0)
    pred = val_logits.argmax(axis=1)
    cm = confusion_matrix(y_val, pred, labels=np.arange(len(TOP_8_TRANSITIONS)))
    return best_params, history, cm


def _save_params(params, path: Path) -> None:
    from flax import serialization
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))


def run_training(cfg: TrainConfig) -> dict:
    os.makedirs(cfg.output_dir, exist_ok=True)
    csv_hash = _csv_hash(cfg.labels_csv)
    ver_dir = Path(cfg.output_dir) / f"v{csv_hash}"
    ver_dir.mkdir(parents=True, exist_ok=True)

    X, y, clip_ids, instances, stats = build_full_dataset(
        behavior_labels_csv=cfg.labels_csv,
        raw_h5_path=cfg.raw_h5,
        K=cfg.K,
        exclude_xy=cfg.exclude_xy,
        do_handle_rotation=cfg.do_handle_rotation,
        pad_frames=cfg.pad_frames,
    )

    # drop any instances that fell outside the top-8 (shouldn't happen given
    # find_directional_transitions filtering, but be defensive)
    valid = np.isin(y, np.arange(len(TOP_8_TRANSITIONS)))
    X, y, clip_ids = X[valid], y[valid], clip_ids[valid]

    print(f"[classifier] total instances: {len(y)} across {stats['n_labeled_clips']} clips")
    for lbl, c in stats["per_label_counts"].items():
        print(f"  {lbl}: {c}")

    unique_clips = np.array(sorted(set(clip_ids)), dtype=np.int64)
    # stratify clips by their modal transition
    clip_modal = []
    for cid in unique_clips:
        mask = clip_ids == cid
        clip_modal.append(int(np.bincount(y[mask], minlength=len(TOP_8_TRANSITIONS)).argmax()))
    clip_modal = np.array(clip_modal, dtype=np.int64)
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.n_folds_seed)

    joint_start = get_joint_start_index(cfg.exclude_xy, cfg.do_handle_rotation)

    agg_cm = np.zeros((len(TOP_8_TRANSITIONS), len(TOP_8_TRANSITIONS)), dtype=np.int64)
    per_fold_history: list[dict] = []
    per_fold_f1: list[float] = []
    fold_val_clips: list[list[int]] = []

    for fold_idx, (tr_clip_idx, val_clip_idx) in enumerate(skf.split(unique_clips, clip_modal)):
        tr_clips = unique_clips[tr_clip_idx]
        val_clips = unique_clips[val_clip_idx]
        fold_val_clips.append([int(c) for c in val_clips])
        mask_tr = np.isin(clip_ids, tr_clips)
        mask_val = np.isin(clip_ids, val_clips)

        X_tr, y_tr, c_tr = X[mask_tr], y[mask_tr], clip_ids[mask_tr]
        X_val, y_val = X[mask_val], y[mask_val]

        # normalize using train-split joint stats (ORIGINAL distribution,
        # pre-upsampling, to avoid bias from duplicated minority samples)
        X_tr_n, X_val_n, mean, std = normalize_with_train_stats(X_tr, X_val, joint_start)
        np.savez(ver_dir / f"norm_stats_fold_{fold_idx}.npz", mean=mean, std=std,
                 joint_start=joint_start)

        # balance train classes via upsampling (val kept natural). When this
        # is on, inverse-frequency class_weighting in train_one_fold becomes a
        # near-no-op — that's intentional: one rebalancing mechanism is enough.
        if cfg.upsample_train:
            X_tr_n, y_tr_bal, _ = upsample_to_balance(
                X_tr_n, y_tr, c_tr, target_count=None, rng_seed=cfg.upsample_seed + fold_idx,
            )
            y_tr = y_tr_bal

        print(f"\n[fold {fold_idx}] tr clips={len(tr_clips)} val clips={len(val_clips)} "
              f"tr windows={len(y_tr)} val windows={len(y_val)}")

        best_params, history, cm = train_one_fold(
            X_tr_n, y_tr, X_val_n, y_val, cfg, fold_idx,
        )
        agg_cm += cm
        per_fold_history.append(history)
        fold_f1 = f1_score(
            y_val,
            cm.argmax(axis=0) if False else np.argmax(
                _predict_logits(best_params, X_val_n, cfg), axis=1,
            ),
            average="macro", zero_division=0,
        )
        per_fold_f1.append(float(fold_f1))
        print(f"[fold {fold_idx}] val macro-F1 = {fold_f1:.3f}")

        _save_params(best_params, ver_dir / f"fold_{fold_idx}.msgpack")

    per_class_f1 = _per_class_f1_from_cm(agg_cm)
    macro_f1 = float(per_class_f1.mean())

    # reporting
    pc_df = pd.DataFrame({
        "label": TOP_8_TRANSITIONS,
        "f1": per_class_f1,
        "support_val": agg_cm.sum(axis=1),
    })
    pc_df.to_csv(ver_dir / "per_class_f1.csv", index=False)
    _save_confusion(agg_cm, TOP_8_TRANSITIONS,
                    str(ver_dir / "confusion_matrix.png"),
                    str(ver_dir / "confusion_matrix.csv"))
    _save_curves(per_fold_history, str(ver_dir / "training_curves.png"))

    metadata = {
        "csv_hash": csv_hash,
        "labels_csv": cfg.labels_csv,
        "raw_h5": cfg.raw_h5,
        "K": cfg.K,
        "n_folds": cfg.n_folds,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "dropout": cfg.dropout,
        "hidden_sizes": list(cfg.hidden_sizes),
        "exclude_xy": cfg.exclude_xy,
        "do_handle_rotation": cfg.do_handle_rotation,
        "joint_start": joint_start,
        "D_preprocessed": int(stats["D_preprocessed"]),
        "label_vocab": TOP_8_TRANSITIONS,
        "per_label_counts": stats["per_label_counts"],
        "per_fold_macro_f1": per_fold_f1,
        "per_fold_val_clips": fold_val_clips,
        "aggregate_macro_f1": macro_f1,
        "per_class_f1": {lbl: float(f) for lbl, f in zip(TOP_8_TRANSITIONS, per_class_f1)},
    }
    with open(ver_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[classifier] aggregate macro-F1 = {macro_f1:.3f}")
    print(f"[classifier] outputs at {ver_dir}")
    return metadata


def _predict_logits(params, X: np.ndarray, cfg: TrainConfig) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    model = BehaviorClassifier(
        n_classes=len(TOP_8_TRANSITIONS),
        hidden_sizes=tuple(cfg.hidden_sizes),
        dropout=cfg.dropout,
    )

    @jax.jit
    def fwd(p, x):
        return model.apply(p, x, training=False)

    parts = []
    for i in range(0, len(X), 512):
        parts.append(np.asarray(fwd(params, jnp.asarray(X[i:i + 512]))))
    return np.concatenate(parts, axis=0)


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--labels_csv", default=TrainConfig.labels_csv)
    p.add_argument("--raw_h5", default=TrainConfig.raw_h5)
    p.add_argument("--output_dir", default=TrainConfig.output_dir)
    p.add_argument("--K", type=int, default=TrainConfig.K)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--no_upsample", action="store_true")
    a = p.parse_args()
    return TrainConfig(
        labels_csv=a.labels_csv, raw_h5=a.raw_h5, output_dir=a.output_dir,
        K=a.K, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        upsample_train=not a.no_upsample,
    )


if __name__ == "__main__":
    run_training(_parse_args())
