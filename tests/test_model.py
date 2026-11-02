import numpy as np

from creditsim import GeneratorConfig, generate
from creditsim.model import calibration_table, ks_statistic, pd_to_score, train_and_evaluate


def test_models_beat_random_and_stay_below_oracle():
    df = generate(GeneratorConfig(n=8_000, seed=1))
    result = train_and_evaluate(df, seed=1)
    oracle = result.metrics["true_pd_oracle"]["roc_auc"]
    for name in ("logistic_regression", "gradient_boosting"):
        auc = result.metrics[name]["roc_auc"]
        assert 0.65 < auc <= oracle + 0.02


def test_ks_is_one_for_perfect_separation():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_score_scaling():
    # base_odds 50:1 good:bad -> PD 1/51 -> score 600; doubling the odds adds 20 points
    assert pd_to_score(np.array([1 / 51]))[0] == 600
    assert pd_to_score(np.array([1 / 101]))[0] == 620
    assert pd_to_score(np.array([0.5]))[0] < pd_to_score(np.array([0.05]))[0]


def test_calibration_table_has_ten_deciles():
    rng = np.random.default_rng(0)
    p = rng.random(1_000)
    y = (rng.random(1_000) < p).astype(int)
    table = calibration_table(y, p)
    assert len(table) == 10
    assert table["observed_rate"].iloc[-1] > table["observed_rate"].iloc[0]
