from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freight.evaluate import (
    all_and_clean,
    day_bias,
    day_debiased_mae,
    day_debiased_predictions,
    regression_metrics,
    slice_table,
)


def test_metrics_values() -> None:
    y = np.array([100.0, 200.0, 400.0])
    p = np.array([110.0, 180.0, 400.0])
    m = regression_metrics(y, p)
    assert m["mae"] == 10.0
    assert np.isclose(m["rmse"], np.sqrt((100 + 400) / 3))
    assert np.isclose(m["mape"], 100 * (0.1 + 0.1 + 0) / 3)
    assert m["medae"] == 10.0
    assert np.isclose(m["median_bias"], 0.0)


def test_all_vs_clean() -> None:
    y = pd.Series([100.0, 100.0, 500.0])
    p = np.array([100.0, 100.0, 100.0])
    out = all_and_clean(y, p, pd.Series([True, True, False]))
    assert out["clean"]["mae"] == 0.0
    assert np.isclose(out["all"]["mae"], 400 / 3)


def test_slice_table_uses_clean_rows() -> None:
    f = pd.DataFrame(
        {"actual": [1.0, 2.0, 9.0], "pred": [1.0, 2.0, 1.0], "clean": [True, True, False], "g": ["a", "b", "a"]}
    )
    t = slice_table(f, "g")
    assert set(t["g"]) == {"a", "b"} and (t["mae"] == 0).all()


def test_day_debiased_mae_removes_constant_daily_bias() -> None:
    """A constant per-day multiplicative bias is day-level (L's) error: debiased MAE is exactly 0."""
    rng = np.random.default_rng(0)
    dates = pd.Series(pd.to_datetime("2025-05-01") + pd.to_timedelta(rng.integers(0, 10, 300), unit="D"))
    actual = rng.uniform(300, 3000, 300)
    bias = pd.Series(rng.normal(0, 0.05, 10), index=pd.date_range("2025-05-01", periods=10))
    pred = actual * np.exp(dates.map(bias).to_numpy())
    clean = np.ones(300, dtype=bool)
    clean[:15] = False
    actual_noisy = actual.copy()
    actual_noisy[:15] *= 4.0  # corrupted labels must not move b_d or enter the MAE
    assert regression_metrics(actual_noisy[clean], pred[clean])["mae"] > 1.0
    assert day_debiased_mae(actual_noisy, pred, dates, clean) == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(day_debiased_predictions(actual_noisy, pred, dates, clean)[clean], actual[clean])


def test_day_debiased_mae_keeps_cross_sectional_error() -> None:
    """Errors whose within-day median log ratio is 0 are untouched: debiased MAE == plain MAE."""
    dates = pd.Series(pd.to_datetime(["2025-05-01"] * 3 + ["2025-05-02"] * 3))
    actual = np.array([100.0, 200.0, 400.0, 100.0, 200.0, 400.0])
    pred = actual * np.array([1.1, 1.0, 0.9, 0.8, 1.0, 1.2])
    assert day_debiased_mae(actual, pred, dates) == pytest.approx(regression_metrics(actual, pred)["mae"])
    b = day_bias(actual, pred * 1.05, dates)
    np.testing.assert_allclose(b.to_numpy(), np.log(1.05))
