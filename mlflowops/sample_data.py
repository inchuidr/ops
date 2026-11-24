"""Synthetic customer-churn data generator (used for the bundled sample + tests).

NOTE: this data is synthetic. Metrics computed on it say nothing about real-world performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    payment = rng.choice(["Credit card", "Bank transfer", "Electronic check", "Mailed check"], n)
    monthly = np.round(np.clip(rng.normal(65, 25, n), 20, 120), 2)
    calls = rng.poisson(1.5, n)
    total = np.round(monthly * tenure * rng.uniform(0.9, 1.05, n), 2)
    logit = (-0.5 - 0.04 * tenure + 0.02 * (monthly - 65) + 0.45 * calls
             + np.where(contract == "Month-to-month", 1.0, np.where(contract == "One year", -0.3, -1.2))
             + np.where(payment == "Electronic check", 0.5, 0.0))
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "tenure_months": tenure, "monthly_charges": monthly, "total_charges": total,
        "support_calls": calls, "contract": contract, "payment_method": payment, "churn": churn,
    })
    df.loc[rng.random(n) < 0.03, "total_charges"] = np.nan  # realistic missing values
    return df
