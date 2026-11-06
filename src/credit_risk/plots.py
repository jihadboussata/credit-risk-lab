"""Static review artifacts generated from saved, real experiment results."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

COLORS = ["#087f8c", "#2b59c3", "#e09f3e", "#7158a6", "#c44536", "#2f855a", "#8795a1"]


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(root: Path, training, metrics, predictions, winner, threshold, thresholds, importance, subgroups):
    destination = root / "reports/figures"
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#263445",
                         "text.color": "#14283f", "axes.titleweight": "bold", "font.size": 10})
    test = metrics[(metrics.split == "test") & ~metrics.model.str.endswith("__policy")].sort_values("average_precision")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(test.model, test.average_precision, color=[COLORS[0] if name == winner else "#9dbbc8" for name in test.model])
    ax.axvline(training.target.mean(), color="#e09f3e", ls="--", label="Training prevalence reference")
    ax.set(xlabel="Average precision (higher is better)", title="Frozen test comparison · winner selected on validation")
    ax.legend(loc="lower right")
    save(fig, destination / "model_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for color, (name, block) in zip(COLORS, predictions.groupby("model", sort=True)):
        fpr, tpr, _ = roc_curve(block.y_true, block.probability)
        precision, recall, _ = precision_recall_curve(block.y_true, block.probability)
        axes[0].plot(fpr, tpr, label=name, color=color)
        # Step rendering preserves the precision-recall definition, especially
        # for constant-score baselines whose artificial endpoint is (0, 1).
        axes[1].step(recall, precision, where="post", label=name, color=color)
    axes[0].plot([0, 1], [0, 1], "--", color="#a0aab7")
    axes[0].set(xlabel="False positive rate", ylabel="Recall", title="Test ROC curves")
    axes[1].axhline(predictions.y_true.mean(), color="#a0aab7", ls="--")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Test precision–recall curves")
    axes[1].legend(fontsize=8)
    save(fig, destination / "roc_pr_curves.png")

    champion = predictions[predictions.model == winner]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    observed, predicted = calibration_curve(champion.y_true, champion.probability, n_bins=10, strategy="quantile")
    axes[0].plot(predicted, observed, "o-", color=COLORS[0])
    axes[0].plot([0, 1], [0, 1], "--", color="#a0aab7")
    axes[0].set(xlabel="Mean predicted probability", ylabel="Observed default rate", title="Champion · test reliability")
    axes[1].hist(champion.probability, bins=25, color=COLORS[0], alpha=.8)
    axes[1].set(xlabel="Default probability", ylabel="Customers", title="Test score distribution")
    save(fig, destination / "calibration.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(champion.y_true, champion.probability >= threshold, display_labels=["No default", "Default"], ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Test confusion matrix · frozen threshold {threshold:.2f}")
    save(fig, destination / "confusion_matrix.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for field, color in zip(["precision", "recall", "review_rate"], COLORS):
        axes[0].plot(thresholds.threshold, thresholds[field], label=field, color=color)
    axes[0].axvline(threshold, color="#14283f", ls="--")
    axes[0].set(xlabel="Threshold", ylabel="Fraction", title="Validation threshold trade-offs")
    axes[0].legend()
    axes[1].plot(thresholds.threshold, thresholds.cost_units_per_1000, color=COLORS[0])
    axes[1].axvline(threshold, color="#14283f", ls="--")
    axes[1].set(xlabel="Threshold", ylabel="Illustrative cost units / 1,000 customers", title="Validation cost · configured error-cost assumptions")
    save(fig, destination / "threshold_tradeoffs.png")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    block = importance.nlargest(12, "importance_mean").sort_values("importance_mean")
    ax.barh(block.feature, block.importance_mean, xerr=block.importance_std, color=COLORS[0], alpha=.9)
    ax.set(xlabel="Drop in validation average precision", title="Raw-feature permutation importance · association only")
    save(fig, destination / "feature_importance.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    counts = training.target.value_counts().sort_index()
    axes[0].bar(["No default", "Default"], counts.values, color=["#b9cbd9", COLORS[0]])
    axes[0].set(title="Training class balance", ylabel="Customers")
    training.groupby("pay_0").target.mean().plot.bar(ax=axes[1], color=COLORS[0])
    axes[1].set(title="Training outcome by September status", xlabel="Repayment status (source code)", ylabel="Default rate")
    groups = pd.cut(training.limit_bal, [0, 50000, 150000, 300000, np.inf],
                    labels=["<50k", "50–150k", "150–300k", "≥300k"], right=False)
    training.groupby(groups, observed=True).target.mean().plot.bar(ax=axes[2], color=COLORS[0])
    axes[2].set(title="Training outcome by credit limit", xlabel="NTD credit-limit band", ylabel="Default rate")
    plt.setp(axes[2].get_xticklabels(), rotation=0)
    save(fig, destination / "training_eda.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, attribute in zip(axes, ["sex", "age"]):
        block = subgroups[subgroups.attribute == attribute]
        x = np.arange(len(block))
        ax.bar(x - .18, block.recall, width=.36, label="Recall", color=COLORS[0])
        ax.bar(x + .18, block.fpr, width=.36, label="False positive rate", color=COLORS[1])
        ax.set_xticks(x, [f"{g}\nn={n}" for g, n in zip(block.group, block.n)])
        ax.set(title=f"Test audit by {attribute}", ylim=(0, 1), ylabel="Rate")
        ax.legend(fontsize=8)
    fig.suptitle("Descriptive subgroup audit · small groups have high uncertainty", y=1.05)
    save(fig, destination / "subgroup_audit.png")
