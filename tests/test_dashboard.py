"""Consequential dashboard checks: data boundaries, arithmetic, safe scenarios."""
import json
import numpy as np
import pandas as pd
import pytest

from app.dashboard import create_app, synthetic_profile, threshold_metrics, winner_name


def test_policy_cost_and_workload():
    result = threshold_metrics([0, 0, 1, 1], [0.1, 0.7, 0.4, 0.9], 0.5, 5, 1)
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 1, 1, 1)
    assert result["cost"] == 6
    assert result["review_rate"] == 0.5
    assert result["precision"] == result["recall"] == 0.5


def test_threshold_equality_and_empty_review():
    assert threshold_metrics([0, 1], [0.1, 0.5], 0.5)["tp"] == 1
    result = threshold_metrics([0, 1], [0.1, 0.5], 1)
    assert result["precision"] == 0 and result["recall"] == 0


@pytest.mark.parametrize("threshold,fn_cost,fp_cost", [(1.1, 5, 1), (0.5, -1, 1), (0.5, np.nan, 1)])
def test_invalid_threshold_and_costs(threshold, fn_cost, fp_cost):
    with pytest.raises(ValueError):
        threshold_metrics([0, 1], [0.1, 0.9], threshold, fn_cost, fp_cost)


@pytest.mark.parametrize("labels,probabilities", [([], []), ([0], [np.nan]), ([2], [0.1]), ([0], [1.1]), ([0, 1], [0.1])])
def test_invalid_policy_vectors(labels, probabilities):
    with pytest.raises(ValueError):
        threshold_metrics(labels, probabilities, 0.5)


def test_does_not_select_winner_on_test():
    metrics = pd.DataFrame({"model": ["A", "B"], "split": ["test", "test"], "average_precision": [0.1, 0.9]})
    assert winner_name({}, metrics) is None
    assert winner_name({"winner": "A"}, metrics) == "A"


def test_synthetic_profile_contract():
    profile = synthetic_profile(150000, 50000, 5000, 2)
    assert profile.shape == (1, 24)
    assert profile.loc[0, "id"] == -1
    assert profile.loc[0, "pay_6"] == 2
    assert profile.loc[0, "bill_amt6"] == 50000
    with pytest.raises(ValueError):
        synthetic_profile(-1, 50000, 5000, 2)


def test_app_renders_without_training_artifacts(tmp_path):
    app = create_app(tmp_path)
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    layout = client.get("/_dash-layout")
    assert layout.status_code == 200
    assert b"Not trained" in layout.data
    assert b"VALIDATION WHAT-IF" in layout.data


def test_threshold_explorer_uses_validation_not_test(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "summary.json").write_text(json.dumps({"winner": "logistic_l2", "threshold": 0.4}))
    for split, probabilities in [("validation", [0.1, 0.9]), ("test", [0.9, 0.1])]:
        pd.DataFrame({"model": ["logistic_l2"] * 2, "id": [1, 2], "y_true": [0, 1], "probability": probabilities}).to_csv(reports / f"{split}_predictions.csv", index=False)
    app = create_app(tmp_path)
    callback = next(value["callback"].__wrapped__ for key, value in app.callback_map.items() if "policy-outcome" in key)
    _, _, fig = callback(0.5, 5, 1)
    assert np.array_equal(np.asarray(fig.data[0].z), [[1, 0], [0, 1]])
