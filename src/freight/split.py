"""Forward-time folds and the cold-start-city holdout.

Validation is Nov-Dec 2025 after a Jan-Oct training period, so every development split is
forward in time: a model is only ever scored on dates strictly after the dates it trained
on. Random K-fold would let the model interpolate within weeks it has already seen and
overstate accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from freight import config


@dataclass(frozen=True)
class Fold:
    """Expanding-window fold; all bounds are inclusive calendar dates."""

    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        if not self.train_end < self.test_start:
            raise ValueError(f"{self.name}: training must end before the test window starts")

    @staticmethod
    def _between(dates: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        d = pd.to_datetime(dates).dt.normalize()
        return (d >= start) & (d <= end)

    def train_mask(self, df: pd.DataFrame) -> pd.Series:
        return self._between(df["date"], self.train_start, self.train_end)

    def test_mask(self, df: pd.DataFrame) -> pd.Series:
        return self._between(df["date"], self.test_start, self.test_end)


def get_folds(names: Iterable[str] | None = None) -> list[Fold]:
    """Fold objects from ``config.FOLD_DATES`` (all of them by default, in calendar order)."""
    names = list(names) if names is not None else list(config.FOLD_DATES)
    folds = []
    for name in names:
        if name not in config.FOLD_DATES:
            raise KeyError(f"unknown fold {name!r}; known: {list(config.FOLD_DATES)}")
        bounds = [pd.Timestamp(d) for d in config.FOLD_DATES[name]]
        folds.append(Fold(name, *bounds))
    return folds


def iter_folds(df: pd.DataFrame, names: Iterable[str] | None = None) -> Iterator[tuple[Fold, pd.Series, pd.Series]]:
    """Yield ``(fold, train_mask, test_mask)`` over ``df``."""
    for fold in get_folds(names):
        yield fold, fold.train_mask(df), fold.test_mask(df)


def _touches(df: pd.DataFrame, cities: set[str]) -> pd.Series:
    return df["pickup"].isin(cities) | df["delivery"].isin(cities)


@dataclass(frozen=True)
class ColdStartSplit:
    cities: tuple[str, ...]
    train_keep: pd.Series  # training rows NOT touching a held-out city
    test_cold: pd.Series  # test rows touching a held-out city


def cold_start_holdout(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_cities: int = 8,
    seed: int = config.SEED,
) -> ColdStartSplit:
    """Simulate the 8 validation-only cities inside a backtest fold.

    Picks ``n_cities`` cities (uniformly, seeded) among those seen in both the training and
    the test window, removes EVERY training row whose pickup or delivery is one of them, and
    marks the test rows that touch them. Scoring those rows measures how well geography
    features (coordinates, distance, bearing) substitute for an unseen city ID.
    """
    train_cities = set(train_df["pickup"]) | set(train_df["delivery"])
    test_cities = set(test_df["pickup"]) | set(test_df["delivery"])
    candidates = sorted(train_cities & test_cities)
    if len(candidates) < n_cities:
        raise ValueError(f"only {len(candidates)} candidate cities for {n_cities} requested")
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(candidates, size=n_cities, replace=False).tolist())
    return ColdStartSplit(
        cities=tuple(sorted(chosen)),
        train_keep=~_touches(train_df, chosen),
        test_cold=_touches(test_df, chosen),
    )
