from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from freight import config
from freight.data import clean
from freight.features import build_features, select_features
from freight.market import build_market_table


_ABS_TIME = re.compile(r"^(month|week|week_?of_?year|iso_?week|doy|day_?of_?year|quarter|year)$")


def _forbidden_like(col: str) -> bool:
    c = col.lower()
    return c in config.FORBIDDEN_FEATURES or bool(_ABS_TIME.match(c))


def test_no_extrapolating_calendar_columns(real_train: pd.DataFrame, real_market: pd.DataFrame) -> None:
    X, _ = build_features(real_train, real_market, use_time=True)
    assert not [c for c in X.columns if _forbidden_like(c)]


def test_only_weight_may_be_nan(real_train: pd.DataFrame, real_market: pd.DataFrame) -> None:
    X, _ = build_features(real_train, real_market)
    nan_cols = X.columns[X.isna().any()].tolist()
    assert nan_cols == ["weight"]
    num = X.drop(columns=config.CAT_FEATURES + ["weight"])
    assert np.isfinite(num.to_numpy(dtype=float)).all()


def test_expected_columns_and_categoricals(real_train: pd.DataFrame, real_market: pd.DataFrame) -> None:
    X, cats = build_features(real_train, real_market)
    assert list(X.columns) == config.ALL_FEATURES + config.OPTIONAL_FEATURES
    assert cats == config.CAT_FEATURES
    X_t, _ = build_features(real_train, real_market, use_time=True)
    assert list(X_t.columns) == config.ALL_FEATURES + config.OPTIONAL_FEATURES + config.TIME_FEATURES
    assert X_t["t_days"].min() == 0


def test_features_do_not_use_the_target(synthetic: pd.DataFrame) -> None:
    df = clean(synthetic)
    market = build_market_table(df)
    X1, _ = build_features(df, market)
    df2 = df.copy()
    df2["posted_rate"] = df2["posted_rate"] * 7.0
    X2, _ = build_features(df2, market)
    pd.testing.assert_frame_equal(X1, X2)


def test_market_index_filled_with_day_mean(synthetic: pd.DataFrame) -> None:
    df = clean(synthetic)
    market = build_market_table(df)
    X, _ = build_features(df, market)
    miss = df["market_index"].isna()
    day_mean = market["mi_day_mean"].reindex(df.loc[miss, "date"].to_numpy()).to_numpy()
    np.testing.assert_allclose(X.loc[miss, "market_index"].to_numpy(), day_mean)
    assert (X.loc[miss, "mi_missing"] == 1).all()


def test_market_windows_are_trailing(synthetic: pd.DataFrame) -> None:
    """Changing the future must not change today's rolling features."""
    df = clean(synthetic)
    cutoff = df["date"].sort_values().iloc[len(df) // 2]
    base = build_market_table(df)
    shocked = df.copy()
    shocked.loc[shocked["date"] > cutoff, "market_index"] += 10.0
    after = build_market_table(shocked)
    past = base.index <= cutoff
    pd.testing.assert_frame_equal(base.loc[past], after.loc[past])


def test_requires_cleaned_frame(synthetic: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        build_features(synthetic, build_market_table(synthetic))


def test_select_features_drops_family(real_train: pd.DataFrame, real_market: pd.DataFrame) -> None:
    X, cats = build_features(real_train.head(200), real_market)
    Xs, cs = select_features(X, cats, config.MARKET_DERIVED_FEATURES + ["lane"])
    assert not set(config.MARKET_DERIVED_FEATURES + ["lane"]) & set(Xs.columns)
    assert "lane" not in cs
