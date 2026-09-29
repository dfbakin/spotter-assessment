"""Market-level model L(date), Option-A chart rows and the cb_level feature contract (no CatBoost training)."""

from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import pytest

from freight import config
from freight.backtest import EXPERIMENTS, BacktestData, _feature_view
from freight.data import load_december, load_validation
from freight.features import build_features, city_coordinates, prepare_chart_rows
from freight.level import MarketLevel, level_design, qe_ramp
from freight.models import LaneLevelBaseline, LevelCatBoostRate
from freight.outliers import RpmReference
from freight.split import get_folds

_MONTH_LIKE = re.compile(r"(month|week|quarter|doy|day_?of_?year|year)")


def _synthetic_market(days: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(config.DATE_ORIGIN, periods=days, freq="D", name="date")
    mi = 0.95 + np.cumsum(rng.normal(0, 0.01, days)) + 0.05 * np.sin(np.arange(days) / 9)
    return pd.DataFrame({"mi_day_mean": mi, "qs_day_mean": 2.0 + 0.1 * mi}, index=idx)


def test_market_level_recovers_known_coefficients() -> None:
    market = _synthetic_market()
    rng = np.random.default_rng(1)
    true = {"const": -0.16, "mi_day_mean": 0.12, "t": 0.0002, "qe_ramp": 0.033}
    per_day = rng.integers(20, 200, len(market))
    dates = np.repeat(market.index.to_numpy(), per_day)
    rows = pd.DataFrame({"date": dates})
    X = level_design(rows["date"], market)
    level = X.to_numpy() @ np.array([true[c] for c in X.columns])
    residual = level + rng.normal(0, 0.02, len(rows))  # per-row noise, averaged per day

    fitted = MarketLevel(market).fit(rows, residual)
    for name, value in true.items():
        assert fitted.coef_[name] == pytest.approx(value, rel=0.08, abs=2e-3), name
    assert fitted.coef_["t"] == pytest.approx(true["t"], rel=0.05)
    assert fitted.r2_ > 0.9
    assert fitted.n_days_ == len(market) and fitted.n_rows_ == len(rows)
    # predictions follow the design
    np.testing.assert_allclose(fitted.predict_dates(market.index[:5]), level[np.cumsum(per_day)[:5] - 1], atol=5e-3)


def test_market_level_terms_are_configurable() -> None:
    market = _synthetic_market()
    rows = pd.DataFrame({"date": np.repeat(market.index.to_numpy(), 10)})
    fitted = MarketLevel(market, terms=("mi_day_mean", "qe_ramp")).fit(rows, np.zeros(len(rows)))
    assert list(fitted.coef_.index) == ["const", "mi_day_mean", "qe_ramp"]
    with pytest.raises(KeyError):
        MarketLevel(market, terms=("month",)).fit(rows, np.zeros(len(rows)))


@pytest.mark.parametrize(
    "date, expected",
    [
        ("2025-12-15", 0.0),
        ("2025-12-31", 1.0),
        ("2025-11-30", 0.0),
        ("2025-09-30", 1.0),
        ("2025-06-23", (23 - 15) / (30 - 15)),
        ("2025-03-01", 0.0),
        ("2025-12-16", 1 / 16),
    ],
)
def test_qe_ramp_exact_values(date: str, expected: float) -> None:
    assert qe_ramp([pd.Timestamp(date)])[0] == pytest.approx(expected, abs=1e-12)


def test_level_fits_take_training_rows_explicitly() -> None:
    assert list(inspect.signature(MarketLevel.fit).parameters) == ["self", "train_rows", "residual"]
    assert list(inspect.signature(LaneLevelBaseline.fit).parameters) == ["self", "train_rows", "reference"]
    assert list(inspect.signature(LevelCatBoostRate.fit).parameters)[:3] == ["self", "X", "train_rows"]


def test_level_uses_only_training_rows(real_train: pd.DataFrame, real_market: pd.DataFrame) -> None:
    """The fitted level depends on training rows alone; test-window labels cannot move it."""
    fold = get_folds(["F1"])[0]
    train = real_train[fold.train_mask(real_train)]
    reference = RpmReference().fit(train)
    yard = LaneLevelBaseline(real_market).fit(train, reference)
    assert yard.level_.daily_.index.max() <= fold.train_end
    assert yard.level_.n_rows_ == len(train)

    shocked = real_train.copy()
    shocked.loc[fold.test_mask(shocked), "posted_rate"] *= 5.0
    yard_b = LaneLevelBaseline(real_market).fit(shocked[fold.train_mask(shocked)], reference)
    pd.testing.assert_series_equal(yard.level_.coef_, yard_b.level_.coef_)


def test_december_rows_get_option_a_market(real_train: pd.DataFrame, real_market: pd.DataFrame) -> None:
    coords = city_coordinates(real_train, load_validation())
    rows = prepare_chart_rows(load_december(), real_market, coords)
    X, _ = build_features(rows, real_market)

    assert len(X) == 31
    np.testing.assert_allclose(X["mi_dev"], 0.0, atol=1e-12)
    np.testing.assert_allclose(X["qs_dev"], 0.0, atol=1e-12)
    assert (X["mi_missing"] == 0).all()
    market_cols = ["market_index", "quote_signal", "mi_day_mean", "mi_dev", "qs_dev", "mi_ma7", "mi_ma28"]
    assert not X[market_cols].isna().any().any()
    dec = real_market.loc["2025-12-01":"2025-12-31"]
    np.testing.assert_allclose(X["mi_day_mean"].to_numpy(), dec["mi_day_mean"].to_numpy())
    # 32,000 lb is an ordinary weight: every weight flag is off
    flags = ["weight_missing", "weight_was_negative", "weight_at_cap", "weight_at_floor"]
    assert (X[flags] == 0).all().all()
    assert not X.drop(columns=config.CAT_FEATURES).isna().any().any()


def test_build_features_fills_absent_market_columns(real_train: pd.DataFrame, real_market: pd.DataFrame) -> None:
    rows = real_train.head(50).drop(columns=["market_index", "quote_signal", "mi_missing"])
    X, _ = build_features(rows, real_market)
    np.testing.assert_allclose(X["mi_dev"], 0.0, atol=1e-12)
    np.testing.assert_allclose(X["qs_dev"], 0.0, atol=1e-12)


def test_cb_level_feature_list_excludes_level_carriers(real_train: pd.DataFrame, real_market: pd.DataFrame) -> None:
    X, cats = build_features(real_train.head(300), real_market, use_time=True)
    data = BacktestData(frame=real_train.head(300), X=X, cat_features=cats, market_table=real_market)
    level_cb = [c for c in EXPERIMENTS.values() if c.model == "level_catboost"]
    assert {c.name for c in level_cb} >= {"cb_level", "cb_level_mi", "cb_level_noqs"}
    for cfg in level_cb:
        cols, cat_cols = _feature_view(cfg, data)
        allowed_extra = {"mi_day_mean"} if cfg.name == "cb_level_mi" else set()
        leaked = (set(cols.columns) & set(config.LEVEL_CB_EXCLUDED)) - allowed_extra
        assert not leaked, (cfg.name, leaked)
        assert not [c for c in cols.columns if _MONTH_LIKE.search(c)], cfg.name
        assert cat_cols == config.CAT_FEATURES
    cb_level_cols = list(_feature_view(EXPERIMENTS["cb_level"], data)[0].columns)
    assert cb_level_cols == config.LEVEL_CB_FEATURES
    for col in ["mi_dev", "qs_dev", "mi_missing", "dow", "dom"]:
        assert col in cb_level_cols
