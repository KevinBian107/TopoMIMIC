"""Classification metrics for walker clusters with GT labels."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _load_sample_labels(experiment_dir: str, method: str, seed: int) -> dict | None:
    path = Path(experiment_dir) / method / f"sample_labels_seed_{seed}.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def _normalize_label(label: str) -> str:
    if not label:
        return ""
    parts = label.strip().split()
    if len(parts) == 2:
        return "↔".join(sorted(parts))
    return parts[0] if parts else ""


def compute_cluster_classification_metrics(
    experiment_dir: str,
    method: str,
    seed: int,
) -> pd.DataFrame | None:
    data = _load_sample_labels(experiment_dir, method, seed)
    if data is None:
        return None

    clusters = data.get("clusters", {})
    rows = []

    for cluster_id, cinfo in sorted(clusters.items()):
        majority = _normalize_label(cinfo.get("majority_label", ""))
        samples = cinfo.get("samples", [])
        if not samples or not majority:
            continue
        if "↔" not in majority:
            continue

        gt_labels = [_normalize_label(s.get("label", "")) for s in samples]
        gt_labels = [l for l in gt_labels if l]
        if not gt_labels:
            continue

        n_total = len(gt_labels)
        n_correct = sum(1 for l in gt_labels if l == majority)

        accuracy = n_correct / n_total if n_total > 0 else 0.0

        label_counts = Counter(gt_labels)
        tp = n_correct
        fp = n_total - n_correct
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        rows.append({
            "cluster": cluster_id,
            "label": majority,
            "n_samples": n_total,
            "n_correct": n_correct,
            "accuracy": accuracy,
            "precision": precision,
            "dominant_frac": label_counts.most_common(1)[0][1] / n_total if n_total > 0 else 0.0,
        })

    if not rows:
        return None
    return pd.DataFrame(rows)


def compute_global_classification_metrics(
    experiment_dir: str,
    method: str,
    seed: int,
) -> dict | None:
    data = _load_sample_labels(experiment_dir, method, seed)
    if data is None:
        return None

    clusters = data.get("clusters", {})

    y_true = []
    y_pred = []

    for cluster_id, cinfo in clusters.items():
        majority = _normalize_label(cinfo.get("majority_label", ""))
        if not majority or "↔" not in majority:
            continue
        for s in cinfo.get("samples", []):
            gt = _normalize_label(s.get("label", ""))
            if gt:
                y_true.append(gt)
                y_pred.append(majority)

    if not y_true:
        return None

    all_labels = sorted(set(y_true) | set(y_pred))
    n = len(y_true)

    overall_accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n

    per_class = {}
    for label in all_labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in y_true if t == label),
        }

    macro_precision = np.mean([v["precision"] for v in per_class.values()])
    macro_recall = np.mean([v["recall"] for v in per_class.values()])
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])

    weighted_precision = sum(v["precision"] * v["support"] for v in per_class.values()) / n
    weighted_recall = sum(v["recall"] * v["support"] for v in per_class.values()) / n
    weighted_f1 = sum(v["f1"] * v["support"] for v in per_class.values()) / n

    return {
        "accuracy": overall_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "n_samples": n,
        "n_classes": len(all_labels),
        "per_class": per_class,
    }


def run_cluster_classification_analysis(
    experiment_dir: str,
    methods: list[str],
    seeds: list[int],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    metric_names = ["precision", "recall", "f1"]
    per_class_tables: dict[str, pd.DataFrame] = {}
    aggregate_rows = []

    for metric in metric_names:
        all_rows = []
        for method in methods:
            for seed in seeds:
                global_metrics = compute_global_classification_metrics(
                    experiment_dir, method, seed,
                )
                if global_metrics is None:
                    continue

                method_key = f"{method}_seed{seed}"
                for label, class_metrics in sorted(global_metrics["per_class"].items()):
                    all_rows.append({
                        "label": label,
                        "method": method_key,
                        "value": class_metrics[metric],
                        "support": class_metrics["support"],
                    })

        if not all_rows:
            continue

        df = pd.DataFrame(all_rows)
        pivot = df.pivot_table(index="label", columns="method", values="value", aggfunc="first")

        support_df = df.pivot_table(index="label", columns="method", values="support", aggfunc="first")
        support_row = support_df.iloc[0] if len(support_df) > 0 else None

        mean_row = pivot.mean(axis=0)
        mean_df = pd.DataFrame([mean_row], index=["mean"])
        pivot = pd.concat([pivot, mean_df])

        per_class_tables[metric] = pivot
        csv_path = os.path.join(output_dir, f"cluster_{metric}_table.csv")
        pivot.to_csv(csv_path)
        print(f"  Saved {metric} table: {csv_path}")

    for method in methods:
        for seed in seeds:
            global_metrics = compute_global_classification_metrics(
                experiment_dir, method, seed,
            )
            if global_metrics is None:
                continue

            method_key = f"{method}_seed{seed}"
            aggregate_rows.append({
                "method": method_key,
                "accuracy": global_metrics["accuracy"],
                "macro_precision": global_metrics["macro_precision"],
                "macro_recall": global_metrics["macro_recall"],
                "macro_f1": global_metrics["macro_f1"],
                "weighted_precision": global_metrics["weighted_precision"],
                "weighted_recall": global_metrics["weighted_recall"],
                "weighted_f1": global_metrics["weighted_f1"],
                "n_samples": global_metrics["n_samples"],
                "n_classes": global_metrics["n_classes"],
            })

    if aggregate_rows:
        agg_df = pd.DataFrame(aggregate_rows).set_index("method").T
        csv_path = os.path.join(output_dir, "aggregate_metrics_table.csv")
        agg_df.to_csv(csv_path)
        print(f"  Saved aggregate table: {csv_path}")

        print("\n  Aggregate Classification Metrics:")
        print(agg_df.to_string())
