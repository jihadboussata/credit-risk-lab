"""Decision arithmetic, reproducibility and split boundary checks."""
import numpy as np
import pytest
from credit_risk.evaluation import classification_metrics, choose_threshold, bootstrap_intervals
from credit_risk.train import create_splits


def test_metrics_match_hand_calculated_confusion():
    result = classification_metrics([0, 0, 1, 1], [.1, .7, .4, .9], .5)
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 1, 1, 1)
    assert result["precision"] == result["recall"] == result["f1"] == .5
    assert result["brier"] == pytest.approx((.01 + .49 + .36 + .01) / 4)


def test_threshold_uses_cost_assumption_and_is_deterministic():
    y, p = [0, 0, 1, 1], [.1, .7, .4, .9]
    selected, table = choose_threshold(y, p, 5, 1)
    row = table[table.threshold == selected].iloc[0]
    assert row.cost_units == table.cost_units.min()
    assert row.cost_units == 1
    assert row.recall == 1
    assert choose_threshold(y, p, 5, 1)[0] == selected
    with pytest.raises(ValueError):
        choose_threshold(y, p, -1, 1)


def test_bootstrap_is_reproducible_and_intervals_ordered():
    y = np.tile([0, 0, 0, 1], 30)
    p = np.linspace(.01, .99, len(y))
    first = bootstrap_intervals(y, p, .3, n_bootstrap=30)
    assert first == bootstrap_intervals(y, p, .3, n_bootstrap=30)
    for bounds in first.values():
        assert 0 <= bounds["low"] <= bounds["high"] <= 1


def test_customer_splits_are_disjoint_complete_and_reproducible(raw_accounts):
    partitions = create_splits(raw_accounts)
    ids = {name: set(frame.id) for name, frame in partitions.items()}
    assert not ids["train"] & ids["validation"]
    assert not ids["train"] & ids["test"]
    assert not ids["validation"] & ids["test"]
    assert set.union(*ids.values()) == set(raw_accounts.id)
    repeated = create_splits(raw_accounts)
    for name in partitions:
        assert partitions[name].id.tolist() == repeated[name].id.tolist()
