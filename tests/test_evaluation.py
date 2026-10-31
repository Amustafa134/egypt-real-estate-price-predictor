"""Tests for src/evaluation/metrics.py. Synthetic data only, no file I/O."""

import numpy as np
import pytest

from src.evaluation.metrics import regression_metrics


def test_perfect_predictions():
    y = np.log([1_000_000, 5_000_000, 20_000_000])
    m = regression_metrics(y, y)
    assert m["rmse_log"] == pytest.approx(0.0)
    assert m["mae_egp"] == pytest.approx(0.0)
    assert m["r2_log"] == pytest.approx(1.0)
    assert m["median_abs_pct_error"] == pytest.approx(0.0)
    assert m["within_20pct"] == pytest.approx(1.0)


def test_egp_errors_are_computed_on_real_prices():
    # Listing 1: true 1.0M, predicted 1.1M  -> 10% over, 100k error
    # Listing 2: true 2.0M, predicted 1.5M  -> 25% under, 500k error
    y_true = np.log([1_000_000, 2_000_000])
    y_pred = np.log([1_100_000, 1_500_000])
    m = regression_metrics(y_true, y_pred)

    assert m["mae_egp"] == pytest.approx(300_000)
    assert m["median_abs_pct_error"] == pytest.approx(0.175)
    assert m["within_20pct"] == pytest.approx(0.5)


def test_returns_all_expected_keys():
    y = np.log([1e6, 2e6, 3e6])
    m = regression_metrics(y, y + 0.1)
    assert set(m) == {
        "rmse_log",
        "mae_log",
        "r2_log",
        "rmse_egp",
        "mae_egp",
        "r2_egp",
        "median_abs_pct_error",
        "within_20pct",
    }


def test_log_errors_match_constant_shift():
    y = np.log([1e6, 2e6, 3e6, 4e6])
    m = regression_metrics(y, y + 0.1)
    assert m["rmse_log"] == pytest.approx(0.1)
    assert m["mae_log"] == pytest.approx(0.1)
