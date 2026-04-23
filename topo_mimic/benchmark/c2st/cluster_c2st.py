"""Per-cluster C2ST driver.

Given a cluster and its real-GT counterpart, trains three binary
classifiers (LogReg/RF/MLP) via 5-fold stratified CV and returns a dict of
mean held-out accuracies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from topo_mimic.benchmark.c2st.models import make_classifier


def run_c2st_single(
    X_cluster: np.ndarray,     # (n, K, D) float32
    X_real: np.ndarray,        # (n, K, D) balanced to |X_cluster|
    model_names: tuple[str, ...] = ("logreg", "rf", "mlp"),
    n_folds: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Returns {model_name: mean_cv_accuracy} for ONE cluster.

    X_cluster and X_real must have the same length n >= n_folds.
    """
    n = X_cluster.shape[0]
    if n < n_folds:
        return {m: float("nan") for m in model_names}

    X = np.concatenate([X_cluster, X_real], axis=0).reshape(2 * n, -1).astype(np.float32)
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    out: dict[str, float] = {}
    for name in model_names:
        fold_accs: list[float] = []
        for fold_idx, (tr, va) in enumerate(skf.split(X, y)):
            scaler = StandardScaler().fit(X[tr])
            Xtr = scaler.transform(X[tr])
            Xva = scaler.transform(X[va])
            clf = make_classifier(name, seed=seed + fold_idx)
            clf.fit(Xtr, y[tr])
            fold_accs.append(float(clf.score(Xva, y[va])))
        out[name] = float(np.mean(fold_accs)) if fold_accs else float("nan")
    return out
