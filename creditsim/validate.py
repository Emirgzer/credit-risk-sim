"""Data quality checks that run before any model sees the data."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "borrower_id", "age", "employment_type", "employment_years", "annual_income",
    "home_ownership", "credit_history_years", "num_open_accounts", "credit_utilisation",
    "late_payments_12m", "loan_purpose", "loan_amount", "term_months", "interest_rate",
    "monthly_payment", "debt_to_income", "default",
}

# column -> (min, max); NaN is allowed only where listed in NULLABLE
RANGES = {
    "age": (18, 75),
    "employment_years": (0, 60),
    "annual_income": (0, 5_000_000),
    "credit_history_years": (0, 60),
    "credit_utilisation": (0, 1),
    "late_payments_12m": (0, 50),
    "loan_amount": (500, 50_000),
    "interest_rate": (0, 1),
    "debt_to_income": (0, 2),
}
NULLABLE = {"employment_years", "credit_history_years"}


def validate(df: pd.DataFrame, default_rate_bounds: tuple[float, float] = (0.01, 0.5)) -> list[str]:
    """Return a list of human-readable problems. An empty list means the data passed."""
    problems: list[str] = []

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]

    if df["borrower_id"].duplicated().any():
        problems.append(f"{int(df['borrower_id'].duplicated().sum())} duplicate borrower_id values")

    for col, (lo, hi) in RANGES.items():
        values = df[col]
        nulls = int(values.isna().sum())
        if nulls and col not in NULLABLE:
            problems.append(f"{col}: {nulls} unexpected missing values")
        out = int(((values < lo) | (values > hi)).sum())
        if out:
            problems.append(f"{col}: {out} values outside [{lo}, {hi}]")

    if not set(df["default"].unique()) <= {0, 1}:
        problems.append("default: labels must be 0 or 1")
    else:
        rate = float(df["default"].mean())
        lo, hi = default_rate_bounds
        if not lo <= rate <= hi:
            problems.append(f"default rate {rate:.3f} outside expected range [{lo}, {hi}]")

    return problems
