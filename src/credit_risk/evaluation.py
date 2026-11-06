"""Probability, threshold and subgroup evaluation with explicit policy assumptions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def classification_metrics(y, probability, threshold=0.5):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
        "brier": float(brier_score_loss(y, probability)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "review_rate": float(prediction.mean()),
        "fpr": float(fp / (fp + tn)) if fp + tn else None,
    }


def threshold_table(y, probability, fn_cost=5.0, fp_cost=1.0):
    """Illustrative error-cost units, never measured currency or saved defaults."""
    if fn_cost <= 0 or fp_cost <= 0:
        raise ValueError("Both error-cost assumptions must be positive.")
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        row = classification_metrics(y, probability, float(threshold))
        row["cost_units"] = float(fn_cost * row["fn"] + fp_cost * row["fp"])
        row["cost_units_per_1000"] = row["cost_units"] / len(y) * 1000
        rows.append(row)
    return pd.DataFrame(rows)


def choose_threshold(y, probability, fn_cost=5.0, fp_cost=1.0):
    table = threshold_table(y, probability, fn_cost, fp_cost)
    # Equal costs: fewer reviews first, then the larger threshold for determinism.
    selected = table.sort_values(["cost_units", "review_rate", "threshold"], ascending=[True, True, False]).iloc[0]
    return float(selected.threshold), table


def bootstrap_intervals(y, probability, threshold, n_bootstrap=500, seed=42):
    """Customer-level nonparametric intervals conditional on the fitted model.

    These do not account for model-selection uncertainty or future distribution
    shifts; the source has one record per customer.
    """
    y, probability = np.asarray(y), np.asarray(probability)
    generator = np.random.default_rng(seed)
    metrics = {key: [] for key in ["average_precision", "roc_auc", "brier", "recall", "precision", "f1"]}
    for _ in range(n_bootstrap):
        indices = generator.integers(0, len(y), len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        result = classification_metrics(y[indices], probability[indices], threshold)
        for key in metrics:
            metrics[key].append(result[key])
    return {key: {"low": float(np.quantile(values, 0.025)), "high": float(np.quantile(values, 0.975))} for key, values in metrics.items()}


def subgroup_metrics(raw, y, probability, threshold, split="test"):
    rows = []
    attributes = {
        "sex": raw.sex.map({1: "source_code_1", 2: "source_code_2"}).fillna("unknown"),
        "age": pd.cut(raw.age, bins=[0, 30, 45, 65, 200], labels=["under_30", "30_to_44", "45_to_64", "65_plus"], right=False).astype(str),
    }
    y, probability = np.asarray(y), np.asarray(probability)
    for attribute, labels in attributes.items():
        for group in sorted(labels.unique()):
            mask = np.asarray(labels == group)
            result = classification_metrics(y[mask], probability[mask], threshold)
            rows.append({"split": split, "attribute": attribute, "group": group,
                         "n": int(mask.sum()), "prevalence": float(y[mask].mean()), **result})
    return pd.DataFrame(rows)


def review_budget_metrics(y, probability, fraction=0.10):
    """Ranking benchmark with stable ties; a hypothetical review budget only."""
    y, probability = np.asarray(y), np.asarray(probability)
    count = max(1, int(np.ceil(len(y) * fraction)))
    indices = np.argsort(-probability, kind="stable")[:count]
    return {"review_fraction": fraction, "reviewed": count,
            "precision_at_budget": float(y[indices].mean()),
            "recall_at_budget": float(y[indices].sum() / y.sum())}
