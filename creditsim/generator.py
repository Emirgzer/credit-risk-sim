"""Synthetic borrower and loan data with a known, controllable default process.

Every borrower gets realistic-looking features (income, debt, credit history,
loan terms). The probability of default comes from a logistic model over those
features, so we know the "true" risk drivers and can check whether a model
trained on the data recovers them.

A share of rows can be turned into edge cases (zero income, missing history,
extreme debt-to-income, ...) to stress-test downstream code and models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EMPLOYMENT_TYPES = ["employed", "self_employed", "unemployed", "student", "retired"]
EMPLOYMENT_PROBS = [0.68, 0.14, 0.05, 0.06, 0.07]
HOME_OWNERSHIP = ["rent", "mortgage", "own"]
HOME_PROBS = [0.45, 0.40, 0.15]
TERMS_MONTHS = [12, 24, 36, 48, 60]
PURPOSES = ["car", "home_improvement", "debt_consolidation", "education", "other"]
PURPOSE_PROBS = [0.25, 0.20, 0.30, 0.10, 0.15]

EDGE_CASE_TYPES = [
    "zero_income",
    "missing_history",
    "extreme_dti",
    "max_loan",
    "thin_file_young",
]


@dataclass
class GeneratorConfig:
    n: int = 50_000
    seed: int = 42
    target_default_rate: float = 0.08
    edge_case_share: float = 0.02


def _annuity_payment(principal: np.ndarray, annual_rate: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Monthly payment for a fully amortising loan."""
    r = annual_rate / 12
    return principal * r / (1 - (1 + r) ** (-months))


def _risk_logit(df: pd.DataFrame) -> np.ndarray:
    """Linear risk score before the intercept is calibrated.

    These coefficients are the ground truth a model should rediscover:
    higher debt-to-income, utilisation and late payments raise risk;
    income, employment length and credit history lower it.
    """
    income_k = np.log1p(df["annual_income"].fillna(0) / 1000)
    emp = df["employment_type"]
    return (
        2.2 * df["debt_to_income"].clip(upper=2.0)
        + 1.8 * df["credit_utilisation"]
        + 0.45 * df["late_payments_12m"]
        - 0.35 * income_k
        - 0.04 * df["employment_years"].fillna(0)
        - 0.05 * df["credit_history_years"].fillna(0)
        + 0.9 * (emp == "unemployed")
        + 0.3 * (emp == "student")
        + 0.25 * (df["home_ownership"] == "rent")
        + 0.006 * (df["term_months"] - 36)
    ).to_numpy(dtype=float)


def _calibrate_intercept(logit: np.ndarray, target_rate: float) -> float:
    """Find the intercept so the mean default probability hits the target (bisection)."""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = np.mean(1 / (1 + np.exp(-(logit + mid))))
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _base_population(n: int, rng: np.random.Generator) -> pd.DataFrame:
    age = np.clip(rng.normal(40, 12, n), 18, 75).round().astype(int)
    employment_type = rng.choice(EMPLOYMENT_TYPES, n, p=EMPLOYMENT_PROBS)

    income = rng.lognormal(mean=np.log(32_000), sigma=0.5, size=n)
    income = np.where(employment_type == "unemployed", income * 0.25, income)
    income = np.where(employment_type == "student", income * 0.35, income)
    income = np.where(employment_type == "retired", income * 0.7, income)

    working_years = np.maximum(age - 18, 0)
    employment_years = np.minimum(rng.gamma(2.0, 3.0, n), working_years)
    employment_years = np.where(np.isin(employment_type, ["unemployed", "student"]), 0, employment_years)
    credit_history_years = np.minimum(rng.gamma(2.5, 3.5, n), working_years)

    credit_utilisation = rng.beta(2, 5, n)
    late_payments = rng.poisson(0.2 + 1.5 * credit_utilisation**2)

    loan_amount = np.clip(rng.lognormal(np.log(8_000), 0.7, n), 500, 50_000).round(-1)
    term_months = rng.choice(TERMS_MONTHS, n)
    interest_rate = np.clip(0.05 + 0.10 * credit_utilisation + 0.02 * late_payments + rng.normal(0, 0.01, n), 0.03, 0.35)
    monthly_payment = _annuity_payment(loan_amount, interest_rate, term_months)

    existing_monthly_debt = income / 12 * rng.beta(2, 8, n)
    # capped at 2.0 (debt payments at twice income): beyond that the ratio carries no extra information
    debt_to_income = np.minimum((existing_monthly_debt + monthly_payment) * 12 / np.maximum(income, 1), 2.0)

    return pd.DataFrame(
        {
            "borrower_id": [f"B{i:07d}" for i in range(n)],
            "age": age,
            "employment_type": employment_type,
            "employment_years": employment_years.round(1),
            "annual_income": income.round(0),
            "home_ownership": rng.choice(HOME_OWNERSHIP, n, p=HOME_PROBS),
            "credit_history_years": credit_history_years.round(1),
            "num_open_accounts": rng.poisson(4, n),
            "credit_utilisation": credit_utilisation.round(3),
            "late_payments_12m": late_payments,
            "loan_purpose": rng.choice(PURPOSES, n, p=PURPOSE_PROBS),
            "loan_amount": loan_amount,
            "term_months": term_months,
            "interest_rate": interest_rate.round(4),
            "monthly_payment": monthly_payment.round(2),
            "debt_to_income": debt_to_income.round(3),
        }
    )


def _inject_edge_cases(df: pd.DataFrame, share: float, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["edge_case"] = ""
    k = int(round(len(df) * share))
    if k == 0:
        return df
    idx = rng.choice(df.index, size=k, replace=False)
    kinds = rng.choice(EDGE_CASE_TYPES, size=k)
    for i, kind in zip(idx, kinds):
        if kind == "zero_income":
            df.loc[i, ["annual_income", "employment_type", "employment_years"]] = [0.0, "unemployed", 0.0]
            df.loc[i, "debt_to_income"] = 2.0  # capped: any payment on zero income is unaffordable
        elif kind == "missing_history":
            df.loc[i, ["credit_history_years", "employment_years"]] = [np.nan, np.nan]
        elif kind == "extreme_dti":
            df.loc[i, "debt_to_income"] = round(float(rng.uniform(1.0, 2.0)), 3)
        elif kind == "max_loan":
            df.loc[i, ["loan_amount", "term_months"]] = [50_000.0, 60]
        elif kind == "thin_file_young":
            df.loc[i, ["age", "credit_history_years", "num_open_accounts"]] = [18, 0.0, 0]
        df.loc[i, "edge_case"] = kind
    return df


def generate(config: GeneratorConfig | None = None) -> pd.DataFrame:
    """Generate a borrower dataset with a `default` label (1 = defaulted)."""
    config = config or GeneratorConfig()
    rng = np.random.default_rng(config.seed)

    df = _base_population(config.n, rng)
    df = _inject_edge_cases(df, config.edge_case_share, rng)

    logit = _risk_logit(df)
    intercept = _calibrate_intercept(logit, config.target_default_rate)
    pd_true = 1 / (1 + np.exp(-(logit + intercept)))

    df["true_pd"] = pd_true.round(5)
    df["default"] = (rng.random(config.n) < pd_true).astype(int)
    return df
