"""Three binary-classifier families for C2ST.

Unified factory so the driver iterates by model name. All sklearn so the
interface is identical: fit(X, y), predict(X), score(X, y).
"""

from __future__ import annotations

from typing import Any


def make_classifier(name: str, seed: int = 0) -> Any:
    name = name.lower()
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200, max_depth=10, n_jobs=1, random_state=seed,
        )
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        # Small net for tiny-n C2ST. 7400-dim input + (64, 16) keeps params
        # under ~480k. No early_stopping (validation fraction of a tiny
        # training fold is too noisy to drive the stop rule).
        return MLPClassifier(
            hidden_layer_sizes=(64, 16),
            activation="relu",
            alpha=1e-2,
            max_iter=500,
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(f"unknown C2ST model {name!r}; expected 'logreg', 'rf', or 'mlp'")


MODEL_DISPLAY_NAMES = {
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "mlp": "MLP (256, 64)",
}
