"""Market price level L(date): a small linear model of the day's overall rpm level.

Phase-1 finding: every tree model predicted the first test month well and under-predicted
the second by 5-7%. The error sits in the *daily price level*: trees cannot extrapolate an
upward trend, and month-proxy features (quote_signal's monthly offset, days_to_quarter_end,
trailing market means) let them reuse older, lower levels. A per-day index -- the mean over a
day's clean loads of log(rpm / lane x equipment median rpm) -- is explained almost entirely
(R^2 ~ 0.96) by three smooth, extrapolatable terms:

* ``mi_day_mean``: the day's market-wide market_index (built from INPUT columns, so it is
  known for Nov-Dec too);
* ``t``: days since ``config.DATE_ORIGIN`` (a slow upward drift, ~ +0.6% / month);
* ``qe_ramp``: a quarter-end calendar ramp -- 0 outside Mar/Jun/Sep/Dec and up to day 15,
  then linear to 1 on the month's last day.

The model is ``log(rpm) = L(date) + f(x)``: :class:`MarketLevel` supplies ``L`` and a
cross-sectional CatBoost model (see :mod:`freight.models`) learns ``f`` with ``L`` passed as
its baseline, so the trees only model what the level does not explain.

Leakage rule: :meth:`MarketLevel.fit` takes the fold's labelled TRAINING rows explicitly;
the market table it reads covariates from is built from input columns only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from freight import config

LEVEL_TERMS: tuple[str, ...] = ("mi_day_mean", "t", "qe_ramp")
QE_MONTHS: frozenset[int] = frozenset({3, 6, 9, 12})
QE_RAMP_START_DAY: int = 15


def _as_dates(dates: pd.Series | pd.DatetimeIndex | Iterable) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).to_numpy())).normalize()


def qe_ramp(dates: pd.Series | pd.DatetimeIndex | Iterable) -> np.ndarray:
    """Quarter-end ramp: ``clip((dom - 15) / (days_in_month - 15), 0, 1)`` in Mar/Jun/Sep/Dec.

    Zero in every other month. EDA: in quarter-end months days 1-10 sit ~+0.05% above trend,
    days 11-20 ~+0.8% and days 21-31 ~+2.1%; non-quarter-end months are flat.
    """
    d = _as_dates(dates)
    dom = d.day.to_numpy(dtype=float)
    dim = d.days_in_month.to_numpy(dtype=float)
    ramp = np.clip((dom - QE_RAMP_START_DAY) / (dim - QE_RAMP_START_DAY), 0.0, 1.0)
    in_qe = np.isin(d.month.to_numpy(), list(QE_MONTHS))
    return np.where(in_qe, ramp, 0.0)


def t_days(dates: pd.Series | pd.DatetimeIndex | Iterable) -> np.ndarray:
    """Days since ``config.DATE_ORIGIN``."""
    d = _as_dates(dates)
    return np.asarray((d - pd.Timestamp(config.DATE_ORIGIN)).days, dtype=float)


def level_design(
    dates: pd.Series | pd.DatetimeIndex | Iterable,
    market_table: pd.DataFrame,
    terms: Iterable[str] = LEVEL_TERMS,
) -> pd.DataFrame:
    """Design matrix ``[const, *terms]`` for the given dates (one row per date given)."""
    d = _as_dates(dates)
    cols: dict[str, np.ndarray] = {"const": np.ones(len(d))}
    for term in terms:
        if term == "mi_day_mean":
            mi = market_table["mi_day_mean"].reindex(d).to_numpy(dtype=float)
            if np.isnan(mi).any():
                missing = sorted({str(x.date()) for x in d[np.isnan(mi)]})[:5]
                raise ValueError(f"market table has no mi_day_mean for dates like {missing}")
            cols["mi_day_mean"] = mi
        elif term == "t":
            cols["t"] = t_days(d)
        elif term == "qe_ramp":
            cols["qe_ramp"] = qe_ramp(d)
        else:
            raise KeyError(f"unknown level term {term!r}; known: {LEVEL_TERMS}")
    return pd.DataFrame(cols, index=d)


def daily_mean(dates: pd.Series, values: pd.Series | np.ndarray) -> pd.DataFrame:
    """Per-date mean of ``values`` and the row count behind it (index = date)."""
    frame = pd.DataFrame({"date": _as_dates(dates), "value": np.asarray(values, dtype=float)})
    return frame.groupby("date")["value"].agg(value="mean", n="size")


def weighted_ols(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    """Weighted least squares; returns ``(beta, weighted R^2)``."""
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    resid = y - X @ beta
    ybar = np.average(y, weights=w)
    ss_res = float(np.sum(w * resid**2))
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    return beta, (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


@dataclass
class MarketLevel:
    """Linear daily level ``L(d) = b0 + b_mi * mi_day_mean(d) + b_t * t(d) + b_qe * qe_ramp(d)``.

    Fitted by OLS on per-day mean residuals, weighted by the number of training rows that
    day (a day with 200 loads has a far less noisy mean than one with 20). ``terms`` selects
    the covariates so ablations can drop the trend or the quarter-end ramp.
    """

    market_table: pd.DataFrame = field(repr=False)
    terms: tuple[str, ...] = LEVEL_TERMS
    coef_: pd.Series = field(default_factory=pd.Series)
    r2_: float = float("nan")
    n_days_: int = 0
    n_rows_: int = 0
    daily_: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def fit(self, train_rows: pd.DataFrame, residual: pd.Series | np.ndarray) -> "MarketLevel":
        """Fit on labelled TRAINING rows and their residuals (same length / order).

        ``residual`` is whatever the level must explain, e.g. ``log(rpm / reference rpm)``
        (initialisation) or ``log(rpm) - f(x)`` (backfitting).
        """
        residual = np.asarray(residual, dtype=float)
        if len(residual) != len(train_rows):
            raise ValueError("residual must align with train_rows")
        daily = daily_mean(train_rows["date"], residual)
        X = level_design(daily.index, self.market_table, self.terms)
        beta, r2 = weighted_ols(X.to_numpy(), daily["value"].to_numpy(), daily["n"].to_numpy(dtype=float))
        self.coef_ = pd.Series(beta, index=X.columns)
        self.r2_ = r2
        self.n_days_ = len(daily)
        self.n_rows_ = len(train_rows)
        self.daily_ = daily.assign(fitted=X.to_numpy() @ beta)
        return self

    def predict_dates(self, dates: pd.Series | pd.DatetimeIndex | Iterable) -> np.ndarray:
        """L for each given date."""
        if self.coef_.empty:
            raise RuntimeError("MarketLevel is not fitted")
        X = level_design(dates, self.market_table, self.terms)
        return X.to_numpy() @ self.coef_.to_numpy()

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """L for each row of a frame with a ``date`` column."""
        return self.predict_dates(rows["date"])

    def coefficients(self) -> dict[str, float]:
        """Coefficients plus fit statistics (JSON-friendly)."""
        out = {k: float(v) for k, v in self.coef_.items()}
        out.update({"r2_daily_weighted": float(self.r2_), "n_days": self.n_days_, "n_rows": self.n_rows_})
        return out
