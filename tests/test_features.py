"""Guard against leakage, undefined ratios and inference/training disagreement."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from credit_risk.features import CreditFeatures, make_pipeline, make_preprocessor, safe_ratio


def test_protected_target_and_identifier_changes_cannot_change_features(raw_accounts):
    transformer = CreditFeatures().fit(raw_accounts)
    expected = transformer.transform(raw_accounts)
    changed = raw_accounts.copy()
    changed["id"] = -1
    changed["target"] = 1 - changed.target
    changed["sex"] = 999
    changed["age"] = 999
    pd.testing.assert_frame_equal(expected, transformer.transform(changed))
    assert not {"target", "id", "sex", "age"}.intersection(expected.columns)


def test_undefined_ratios_are_missing_not_infinite():
    result = safe_ratio(pd.Series([10., 0., -5.]), pd.Series([0., 0., 2.]))
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == -2.5


def test_zero_balances_and_unseen_categories_can_be_scored(raw_accounts):
    pipeline = make_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(raw_accounts.iloc[:40], raw_accounts.target.iloc[:40])
    incoming = raw_accounts.iloc[40:].copy()
    incoming["limit_bal"] = 0
    incoming["pay_0"] = 8
    incoming["marriage"] = 99
    probability = pipeline.predict_proba(incoming)[:, 1]
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()


def test_predictions_are_independent_of_column_order_and_audit_fields(raw_accounts):
    pipeline = make_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(raw_accounts, raw_accounts.target)
    incoming = raw_accounts.drop(columns=["id", "sex", "age", "target"])
    np.testing.assert_allclose(pipeline.predict_proba(raw_accounts),
                               pipeline.predict_proba(incoming[incoming.columns[::-1]]))


def test_numeric_scaler_uses_only_training_values(raw_accounts):
    train = raw_accounts.iloc[:36].copy()
    preprocessor = make_preprocessor().fit(train)
    scaler = preprocessor.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"]
    assert scaler.mean_[0] == pytest.approx(train.limit_bal.mean())
    previous = scaler.mean_.copy()
    incoming = raw_accounts.iloc[36:].copy()
    incoming["limit_bal"] = 1000000000
    preprocessor.transform(incoming)
    np.testing.assert_array_equal(scaler.mean_, previous)


def test_missing_required_field_has_actionable_error(raw_accounts):
    with pytest.raises(ValueError, match="limit_bal"):
        CreditFeatures().fit(raw_accounts.drop(columns="limit_bal"))


def test_status_codes_stay_distinct_and_negative_bills_are_preserved(raw_accounts):
    raw_accounts.loc[:2, "pay_0"] = [-2, -1, 0]
    raw_accounts.loc[0, "bill_amt1"] = -300
    result = CreditFeatures().fit_transform(raw_accounts)
    assert result.loc[:2, "pay_0"].tolist() == ["-2", "-1", "0"]
    assert result.loc[0, "bill_amt1"] == -300
