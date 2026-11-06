"""Reconcile committed study evidence without source downloads or retraining.

These checks catch stale/mismatched reports, policy arithmetic mistakes, split
overlap, and selecting a champion from the wrong published metric. They do not
prove the absence of all leakage or validate future-population performance.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

REPORTS = Path(__file__).resolve().parents[1] / "reports"
MODELS = {"dummy", "logistic_l1", "logistic_l2", "decision_tree", "random_forest", "xgboost", "svm_rbf"}


@pytest.fixture(scope="module")
def study():
    path = REPORTS / "summary.json"
    if not path.exists():
        pytest.skip("No saved study yet; run the experiment to create report artifacts.")
    return json.loads(path.read_text(encoding="utf-8"))


def test_saved_split_manifest_is_complete_and_predictions_respect_it(study):
    manifest = pd.read_csv(REPORTS / "split_manifest.csv")
    assert manifest.id.is_unique
    assert len(manifest) == study["dataset"]["rows"] == 30000
    assert manifest.groupby("split").size().to_dict() == study["split_counts"]
    assert set(manifest.split) == {"train", "validation", "test"}
    for split in ["validation", "test"]:
        predictions = pd.read_csv(REPORTS / f"{split}_predictions.csv", float_precision="round_trip")
        assert set(predictions.model) == MODELS
        expected = set(manifest.loc[manifest.split == split, "id"])
        for _, block in predictions.groupby("model"):
            assert block.id.is_unique
            assert set(block.id) == expected
            assert set(block.y_true.unique()) == {0, 1}
            assert np.isfinite(block.probability).all()
            assert block.probability.between(0, 1).all()
            assert block.y_true.mean() == pytest.approx(study["split_prevalence"][split])
        labels = predictions.pivot(index="id", columns="model", values="y_true")
        assert (labels.nunique(axis=1) == 1).all()


def test_all_saved_metrics_reconstruct_from_prediction_files(study):
    metrics = pd.read_csv(REPORTS / "metrics.csv")
    assert not metrics.duplicated(["model", "split"]).any()
    for split in ["validation", "test"]:
        predictions = pd.read_csv(REPORTS / f"{split}_predictions.csv", float_precision="round_trip")
        for _, row in metrics[metrics.split == split].iterrows():
            model = row.model.removesuffix("__policy")
            block = predictions[predictions.model == model]
            assert len(block) == study["split_counts"][split]
            y, p = block.y_true.to_numpy(), block.probability.to_numpy()
            positive = p >= row.threshold
            expected_counts = {
                "tn": ((y == 0) & ~positive).sum(), "fp": ((y == 0) & positive).sum(),
                "fn": ((y == 1) & ~positive).sum(), "tp": ((y == 1) & positive).sum(),
            }
            for key, expected in expected_counts.items():
                assert row[key] == expected
            assert row.average_precision == pytest.approx(average_precision_score(y, p), abs=1e-12)
            assert row.roc_auc == pytest.approx(roc_auc_score(y, p), abs=1e-12)
            assert row.brier == pytest.approx(np.mean((p - y) ** 2), abs=1e-12)
            assert row.review_rate == pytest.approx(positive.mean(), abs=1e-12)


def test_champion_selection_and_summary_match_saved_evidence(study):
    selection = study["model_selection"]
    assert selection["winner_selected_before_test"] is True
    assert selection["test_comparison_is_descriptive"] is True
    scores = selection["validation_average_precision"]
    assert set(scores) == MODELS - {"dummy"}
    assert study["winner"] == max(scores, key=scores.get)
    metrics = pd.read_csv(REPORTS / "metrics.csv")
    policy = metrics[(metrics.model == study["winner"] + "__policy") & (metrics.split == "test")].iloc[0]
    for key in ["average_precision", "roc_auc", "brier", "precision", "recall", "f1", "threshold", "tn", "fp", "fn", "tp", "review_rate"]:
        assert study["test_metrics"][key] == pytest.approx(policy[key], abs=1e-12)
    assert study["threshold"] == pytest.approx(policy.threshold)
    for bounds in study["bootstrap95"].values():
        assert 0 <= bounds["low"] <= bounds["high"] <= 1
    assert study["bootstrap_replicates"] >= 500
    calibration = study["calibration"]
    if calibration["method"] == "sigmoid":
        assert calibration["sigmoid_validation_brier"] < calibration["raw_validation_brier"]
    elif calibration["method"] == "none":
        assert calibration["raw_validation_brier"] <= calibration["sigmoid_validation_brier"]
    else:
        assert calibration["method"] == "sigmoid_svm_required"
        assert study["winner"] == "svm_rbf"


def test_frozen_threshold_and_error_costs_reconcile(study):
    table = pd.read_csv(REPORTS / "threshold_analysis.csv")
    fn_cost = study["costs"]["false_negative"]
    fp_cost = study["costs"]["false_positive"]
    assert np.allclose(table.cost_units, fn_cost * table.fn + fp_cost * table.fp)
    assert np.allclose(table.cost_units_per_1000, table.cost_units / study["split_counts"]["validation"] * 1000)
    frozen = table.iloc[np.abs(table.threshold - study["threshold"]).argmin()]
    assert frozen.threshold == pytest.approx(study["threshold"])
    assert frozen.cost_units == table.cost_units.min()
    validation = pd.read_csv(REPORTS / "validation_predictions.csv", float_precision="round_trip")
    block = validation[validation.model == study["winner"]]
    positive = block.probability >= study["threshold"]
    assert frozen.fp == ((block.y_true == 0) & positive).sum()
    assert frozen.fn == ((block.y_true == 1) & ~positive).sum()
    result = study["test_metrics"]
    assert result["cost_units"] == fn_cost * result["fn"] + fp_cost * result["fp"]
    assert result["cost_units_per_1000"] == pytest.approx(result["cost_units"] / study["split_counts"]["test"] * 1000)


def test_full_search_contains_all_declared_model_families(study):
    candidates = pd.read_csv(REPORTS / "cv_results.csv")
    assert set(candidates.model) == MODELS - {"dummy"}
    assert np.isfinite(candidates.mean_cv_average_precision).all()
    if study["run_mode"] == "full":
        assert (candidates.groupby("model").size() >= 4).all()
    for model, parameters in study["model_selection"]["best_parameters"].items():
        matches = candidates["params"].apply(lambda text: json.loads(text) == parameters)
        selected = candidates[(candidates.model == model) & matches]
        assert len(selected) == 1
        assert selected.iloc[0]["rank"] == 1
