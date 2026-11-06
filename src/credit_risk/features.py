"""Deterministic domain features and training-fitted preprocessing.

The target, identifiers, sex and age can never enter the primary feature matrix.
Education and marriage remain possible proxies; excluding direct attributes does
not establish fairness. Feature construction never uses future payment outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

STATUS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
BILLS = [f"bill_amt{i}" for i in range(1, 7)]
PAYMENTS = [f"pay_amt{i}" for i in range(1, 7)]
NUMERIC = ["limit_bal", *BILLS, *PAYMENTS]
CATEGORICAL = ["education", "marriage", *STATUS]
ENGINEERED = [
    "delayed_month_count", "max_delay", "recent_delay_change",
    "mean_bill", "bill_change", "mean_payment", "payment_change",
    "latest_balance_to_limit", "mean_balance_to_limit",
    "payment_to_abs_bill", "zero_payment_months",
]
REQUIRED_FEATURES = [*NUMERIC, *CATEGORICAL]


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Undefined ratios become missing, to be imputed using training data only."""
    return numerator.div(denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


class CreditFeatures(TransformerMixin, BaseEstimator):
    """Build an explicit allowlist of features from canonical raw columns.

    PAY_0 is September and PAY_6 is April. Negative repayment codes and negative
    balances are retained as reported, rather than silently changed into defaults
    or missing values. Ratio names describe recorded amounts, not causal behavior.
    """

    def __init__(self, engineered: bool = True):
        self.engineered = engineered

    def fit(self, X: pd.DataFrame, y=None):
        self._validate(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def _validate(self, X):
        missing = sorted(set(REQUIRED_FEATURES) - set(X.columns))
        if missing:
            raise ValueError(f"Missing required raw columns: {missing}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate(X)
        out = X.loc[:, REQUIRED_FEATURES].copy()
        for col in REQUIRED_FEATURES:
            out[col] = pd.to_numeric(out[col], errors="raise")
        # Keep valid groups; consolidate undocumented education values explicitly.
        out["education"] = out["education"].where(out["education"].isin([1, 2, 3, 4]), 0)
        if self.engineered:
            out["delayed_month_count"] = X[STATUS].gt(0).sum(axis=1)
            out["max_delay"] = X[STATUS].clip(lower=0).max(axis=1)
            out["recent_delay_change"] = X["pay_0"].clip(lower=0) - X["pay_2"].clip(lower=0)
            out["mean_bill"] = X[BILLS].mean(axis=1)
            out["bill_change"] = X["bill_amt1"] - X["bill_amt6"]
            out["mean_payment"] = X[PAYMENTS].mean(axis=1)
            out["payment_change"] = X["pay_amt1"] - X["pay_amt6"]
            out["latest_balance_to_limit"] = safe_ratio(X["bill_amt1"], X["limit_bal"])
            out["mean_balance_to_limit"] = safe_ratio(out["mean_bill"], X["limit_bal"])
            out["payment_to_abs_bill"] = safe_ratio(out["mean_payment"], X[BILLS].abs().mean(axis=1))
            out["zero_payment_months"] = X[PAYMENTS].eq(0).sum(axis=1)
        for col in CATEGORICAL:
            out[col] = out[col].astype(int).astype(str)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray([*REQUIRED_FEATURES, *(ENGINEERED if self.engineered else [])], dtype=object)


def make_preprocessor(engineered=True):
    """Return a fresh estimator; imputation and scaling are fitted inside each fold."""
    numeric = [*NUMERIC, *(ENGINEERED if engineered else [])]
    columns = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("features", CreditFeatures(engineered=engineered)), ("columns", columns)])


def make_pipeline(model, engineered=True):
    return Pipeline([("preprocess", make_preprocessor(engineered)), ("model", model)])
