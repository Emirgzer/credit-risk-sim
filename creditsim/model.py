"""Train and evaluate probability-of-default (PD) models on the synthetic data.

Two models are compared:
- logistic regression: the industry-standard, explainable baseline for scorecards
- histogram gradient boosting: a non-linear challenger

Metrics are the ones credit risk teams use: ROC AUC, Gini, KS, Brier score and
a calibration table by decile. Predicted PDs are also mapped to a points-based
credit score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    "age", "employment_years", "annual_income", "credit_history_years", "num_open_accounts",
    "credit_utilisation", "late_payments_12m", "loan_amount", "term_months", "debt_to_income",
]
CATEGORICAL = ["employment_type", "home_ownership", "loan_purpose"]
TARGET = "default"
# interest_rate and monthly_payment are left out on purpose: the rate is priced
# from risk, so using it would leak the answer into the model.


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def build_models(seed: int = 42) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("prep", _preprocessor()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "gradient_boosting": Pipeline([
            ("prep", _preprocessor()),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)),
        ]),
    }


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Max distance between the cumulative distributions of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def calibration_table(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed default rate per score decile."""
    df = pd.DataFrame({"y": y_true, "p": y_score})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False) + 1
    table = df.groupby("decile").agg(count=("y", "size"), predicted_pd=("p", "mean"), observed_rate=("y", "mean"))
    return table.round(4)


def pd_to_score(pd_: np.ndarray, base_score: float = 600, base_odds: float = 50, pdo: float = 20) -> np.ndarray:
    """Standard scorecard scaling: `base_score` at good:bad odds of `base_odds`, +`pdo` points per doubling of odds."""
    pd_ = np.clip(pd_, 1e-6, 1 - 1e-6)
    odds_good = (1 - pd_) / pd_
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return np.round(offset + factor * np.log(odds_good)).astype(int)


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y_true, y_score)
    return {
        "roc_auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(ks_statistic(y_true, y_score), 4),
        "brier": round(float(brier_score_loss(y_true, y_score)), 4),
        "log_loss": round(float(log_loss(y_true, y_score)), 4),
        "mean_predicted_pd": round(float(np.mean(y_score)), 4),
        "observed_default_rate": round(float(np.mean(y_true)), 4),
    }


@dataclass
class TrainResult:
    metrics: dict[str, dict[str, float]]
    calibration: dict[str, pd.DataFrame]
    models: dict[str, Pipeline]
    test_frame: pd.DataFrame


def train_and_evaluate(df: pd.DataFrame, seed: int = 42, test_size: float = 0.25) -> TrainResult:
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET].to_numpy()
    X_train, X_test, y_train, y_test, _, idx_test = train_test_split(
        X, y, df.index, test_size=test_size, stratify=y, random_state=seed
    )

    metrics, calibration, models = {}, {}, build_models(seed)
    test_frame = df.loc[idx_test, ["borrower_id"]].copy()
    if "true_pd" in df:
        # the ceiling: how well a model could do if it knew the true default probabilities
        metrics["true_pd_oracle"] = evaluate(y_test, df.loc[idx_test, "true_pd"].to_numpy())

    for name, model in models.items():
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        metrics[name] = evaluate(y_test, p)
        calibration[name] = calibration_table(y_test, p)
        test_frame[f"pd_{name}"] = p.round(5)
        test_frame[f"score_{name}"] = pd_to_score(p)

    return TrainResult(metrics=metrics, calibration=calibration, models=models, test_frame=test_frame)
