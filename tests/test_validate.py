import numpy as np

from creditsim import GeneratorConfig, generate, validate


def _sample():
    return generate(GeneratorConfig(n=1_000, edge_case_share=0))


def test_detects_missing_column():
    df = _sample().drop(columns=["loan_amount"])
    assert any("missing columns" in p for p in validate(df))


def test_detects_out_of_range_values():
    df = _sample()
    df.loc[0, "credit_utilisation"] = 1.5
    assert any(p.startswith("credit_utilisation") for p in validate(df))


def test_detects_unexpected_nulls():
    df = _sample()
    df.loc[0, "annual_income"] = np.nan
    assert any("unexpected missing" in p for p in validate(df))


def test_allows_nulls_where_expected():
    df = _sample()
    df.loc[0, "credit_history_years"] = np.nan
    assert validate(df) == []


def test_detects_duplicate_ids():
    df = _sample()
    df.loc[1, "borrower_id"] = df.loc[0, "borrower_id"]
    assert any("duplicate" in p for p in validate(df))
