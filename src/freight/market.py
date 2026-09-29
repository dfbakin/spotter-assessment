"""Daily market table built from INPUT columns only.

EDA showed ``market_index`` is a market-wide daily signal (within-day sd 0.025 vs
across-day sd 0.17, city explains <0.3% of within-day variance). Averaging it over all loads
of a day therefore denoises each load's reading. The table may pool train AND validation
rows because it never touches ``posted_rate``: at prediction time these inputs are known.
Rolling windows are trailing (they end on the current day) so no future market state leaks
into a row.
"""

from __future__ import annotations

import pandas as pd

MARKET_COLUMNS: list[str] = [
    "mi_day_mean",
    "mi_day_std",
    "n_loads",
    "qs_day_mean",
    "mi_ma7",
    "mi_ma28",
]


def build_market_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ``market_index`` / ``quote_signal`` per calendar date.

    Returns a frame indexed by every calendar date between the first and last date seen,
    so that 7/28-row rolling windows equal 7/28 calendar days.
    """
    cols = ["date", "market_index", "quote_signal"]
    rows = pd.concat([f[cols] for f in frames], ignore_index=True)
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()

    daily = rows.groupby("date").agg(
        mi_day_mean=("market_index", "mean"),
        mi_day_std=("market_index", "std"),
        n_loads=("market_index", "size"),
        qs_day_mean=("quote_signal", "mean"),
    )
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D", name="date")
    daily = daily.reindex(calendar)
    daily["n_loads"] = daily["n_loads"].fillna(0).astype(int)

    # Trailing windows over DAILY means (not over loads) so busy days do not dominate.
    daily["mi_ma7"] = daily["mi_day_mean"].rolling(7, min_periods=1).mean()
    daily["mi_ma28"] = daily["mi_day_mean"].rolling(28, min_periods=1).mean()
    return daily[MARKET_COLUMNS]


def join_market(df: pd.DataFrame, market_table: pd.DataFrame) -> pd.DataFrame:
    """Left-join the daily market columns onto each row by calendar date."""
    dates = pd.to_datetime(df["date"]).dt.normalize()
    joined = market_table.reindex(dates.to_numpy())
    joined.index = df.index
    return pd.concat([df, joined], axis=1)


def fill_market_index(df: pd.DataFrame, market_table: pd.DataFrame) -> pd.Series:
    """Return ``market_index`` with gaps filled by that date's market-wide mean.

    Because the index is a daily market signal, the day mean is the best available estimate
    of a missing reading. If a whole day were missing we fall back to the trailing 7-day mean.
    """
    dates = pd.to_datetime(df["date"]).dt.normalize().to_numpy()
    day_mean = pd.Series(market_table["mi_day_mean"].reindex(dates).to_numpy(), index=df.index)
    ma7 = pd.Series(market_table["mi_ma7"].reindex(dates).to_numpy(), index=df.index)
    return df["market_index"].fillna(day_mean).fillna(ma7)
