import numpy as np

from creditsim import GeneratorConfig, generate, validate


def test_same_seed_gives_same_data():
    a = generate(GeneratorConfig(n=2_000, seed=7))
    b = generate(GeneratorConfig(n=2_000, seed=7))
    assert a.equals(b)


def test_default_rate_close_to_target():
    df = generate(GeneratorConfig(n=20_000, target_default_rate=0.10))
    assert abs(df["default"].mean() - 0.10) < 0.01
    assert abs(df["true_pd"].mean() - 0.10) < 0.002


def test_edge_case_share_and_types():
    df = generate(GeneratorConfig(n=10_000, edge_case_share=0.05))
    edge = df[df["edge_case"] != ""]
    assert len(edge) == 500
    zero = edge[edge["edge_case"] == "zero_income"]
    assert (zero["annual_income"] == 0).all()
    assert edge.loc[edge["edge_case"] == "missing_history", "credit_history_years"].isna().all()


def test_risk_drivers_move_default_probability():
    df = generate(GeneratorConfig(n=20_000, edge_case_share=0))
    high_util = df["credit_utilisation"] > df["credit_utilisation"].quantile(0.9)
    low_util = df["credit_utilisation"] < df["credit_utilisation"].quantile(0.1)
    assert df.loc[high_util, "true_pd"].mean() > df.loc[low_util, "true_pd"].mean()


def test_generated_data_passes_validation():
    df = generate(GeneratorConfig(n=5_000))
    assert validate(df) == []
