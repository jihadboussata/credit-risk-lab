"""Synthetic fixtures; tests never download data or expose customer records."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_accounts():
    rng = np.random.default_rng(8)
    n = 48
    frame = pd.DataFrame({
        "id": np.arange(1, n + 1), "limit_bal": rng.integers(1, 20, n) * 10000,
        "sex": rng.choice([1, 2], n), "age": rng.integers(21, 75, n),
        "education": rng.choice([0, 1, 2, 3, 4, 5, 6], n),
        "marriage": rng.choice([0, 1, 2, 3], n),
    })
    for col in ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]:
        frame[col] = rng.choice([-2, -1, 0, 1, 2], n)
    for month in range(1, 7):
        frame[f"bill_amt{month}"] = rng.integers(-500, 150000, n)
        frame[f"pay_amt{month}"] = rng.integers(0, 10000, n)
    frame["target"] = np.tile([0, 0, 0, 1], n // 4)
    return frame
