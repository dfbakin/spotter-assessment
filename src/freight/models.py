"""Models: lane-median yardsticks and CatBoost wrappers that work in dollars.

* :class:`LaneMedianBaseline` -- lane x equipment median rpm x distance (no model).
* :class:`LaneLevelBaseline` -- the same reference times ``exp(L(date))`` from
  :class:`freight.level.MarketLevel` (market level, no trees).
* :class:`CatBoostRate` -- CatBoost on a log target, optionally started from a baseline.
* :class:`LevelCatBoostRate` -- ``log(rpm) = L(date) + f(x)``: market level plus a
  cross-sectional CatBoost fitted with ``L`` as its baseline, with optional backfitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from freight import config
from freight.level import LEVEL_TERMS, MarketLevel
from freight.outliers import RpmReference

TargetMode = Literal["log_rpm", "log_rate"]

DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "loss_function": "RMSE",
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": config.SEED,
    "thread_count": config.THREAD_COUNT,
    "verbose": 0,
    "allow_writing_files": False,
}


# Run-level settings a tuned-parameter file must not override: the loss is fixed (RMSE on the
# log target) and seeds / threads belong to each run, not to the hyper-parameters.
RUN_LEVEL_PARAMS: frozenset[str] = frozenset(
    {"loss_function", "random_seed", "thread_count", "verbose", "allow_writing_files", "task_type"}
)


def load_tuned_params(path: Path | str = config.BEST_PARAMS_PATH) -> dict[str, Any]:
    """CatBoost overrides stored under ``"params"`` in a ``best_params.json`` (phase-2b tuning).

    The result is meant to be merged over :data:`DEFAULT_CATBOOST_PARAMS`, exactly like an
    experiment's ``catboost_params``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make tune` first")
    blob = json.loads(path.read_text())
    params = dict(blob["params"])
    forbidden = sorted(set(params) & RUN_LEVEL_PARAMS)
    if forbidden:
        raise ValueError(f"{path}: run-level settings {forbidden} do not belong in tuned params")
    return params


def log_rpm(rows: pd.DataFrame) -> np.ndarray:
    """log(posted_rate / distance) for labelled rows."""
    return np.log(rows[config.TARGET].to_numpy(dtype=float) / rows["distance"].to_numpy(dtype=float))


class LaneMedianBaseline:
    """Yardstick: median rpm of the lane x equipment (-> lane -> equipment x band) x distance.

    It is what a pricing desk would quote from history with no model. Anything the
    CatBoost model gains over it is the value of market / calendar / geography features.
    """

    def __init__(self) -> None:
        self.reference = RpmReference()

    def fit(self, train_rows: pd.DataFrame) -> "LaneMedianBaseline":
        self.reference.fit(train_rows)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return (self.reference.reference_rpm(rows) * rows["distance"]).to_numpy()


@dataclass
class CatBoostRate:
    """CatBoost regressor on a log target, returning predictions in dollars.

    Target modes
    * ``log_rpm``: learn log(rate / distance), multiply back by distance. Rate per mile is
      the unit freight is priced in, so distance's first-order effect is removed and trees
      only model the (smaller) taper and premiums.
    * ``log_rate``: learn log(rate) directly; the model must rediscover the distance scale.

    Logs make errors multiplicative (a 5% miss costs the same on a $400 and a $4,000 load)
    and RMSE on logs estimates the conditional geometric mean, i.e. the typical price,
    rather than being pulled up by the right-skewed corrupted labels.
    """

    target: TargetMode = "log_rpm"
    params: dict[str, Any] = field(default_factory=dict)
    model_: CatBoostRegressor | None = field(default=None, repr=False)
    feature_names_: list[str] = field(default_factory=list, repr=False)
    cat_features_: list[str] = field(default_factory=list, repr=False)

    def _to_target(self, rate: np.ndarray, distance: np.ndarray) -> np.ndarray:
        if self.target == "log_rpm":
            return np.log(rate / distance)
        if self.target == "log_rate":
            return np.log(rate)
        raise ValueError(f"unknown target mode {self.target!r}")

    def _from_target(self, z: np.ndarray, distance: np.ndarray) -> np.ndarray:
        return np.exp(z) * distance if self.target == "log_rpm" else np.exp(z)

    def fit(
        self,
        X: pd.DataFrame,
        rate: pd.Series | np.ndarray,
        cat_features: list[str],
        baseline: np.ndarray | None = None,
    ) -> "CatBoostRate":
        """Fit on features ``X`` (must include ``distance``) and dollar targets ``rate``.

        ``baseline`` (target scale, one value per row) makes boosting start from it instead
        of from the mean, so the trees only learn the remainder. CatBoost then disables
        ``boost_from_average`` itself.
        """
        y = self._to_target(np.asarray(rate, dtype=float), X["distance"].to_numpy(dtype=float))
        self.feature_names_ = list(X.columns)
        self.cat_features_ = list(cat_features)
        self.model_ = CatBoostRegressor(**{**DEFAULT_CATBOOST_PARAMS, **self.params})
        base = None if baseline is None else np.asarray(baseline, dtype=float)
        self.model_.fit(Pool(X, y, cat_features=self.cat_features_, baseline=base))
        return self

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Tree output on the target scale, WITHOUT any baseline."""
        if self.model_ is None:
            raise RuntimeError("model is not fitted")
        # A Pool without baseline returns the trees only (CatBoost would add a Pool baseline).
        return np.asarray(self.model_.predict(Pool(X[self.feature_names_], cat_features=self.cat_features_)))

    def predict(self, X: pd.DataFrame, baseline: np.ndarray | None = None) -> np.ndarray:
        """Predict dollars; pass the same kind of ``baseline`` the model was fitted with."""
        z = self.predict_raw(X)
        if baseline is not None:
            z = z + np.asarray(baseline, dtype=float)
        return self._from_target(z, X["distance"].to_numpy(dtype=float))

    def feature_importance(self) -> pd.DataFrame:
        """PredictionValuesChange importance, sorted descending."""
        if self.model_ is None:
            raise RuntimeError("model is not fitted")
        imp = self.model_.get_feature_importance(type="PredictionValuesChange")
        return (
            pd.DataFrame({"feature": self.feature_names_, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


@dataclass
class LaneLevelBaseline:
    """Level-only yardstick: ``rate = distance * ref_rpm(lane, equipment) * exp(L(date))``.

    Same reference and same level initialisation as :class:`LevelCatBoostRate`, no trees.
    Its gap to :class:`LaneMedianBaseline` is the value of modelling the daily price level;
    the gap from it to :class:`LevelCatBoostRate` is the value of the cross-sectional trees.
    """

    market_table: pd.DataFrame = field(repr=False)
    level_terms: tuple[str, ...] = LEVEL_TERMS
    reference_: RpmReference | None = field(default=None, repr=False)
    level_: MarketLevel | None = field(default=None, repr=False)

    def fit(self, train_rows: pd.DataFrame, reference: RpmReference) -> "LaneLevelBaseline":
        """``train_rows``: clean labelled training rows; ``reference`` fitted on training rows."""
        self.reference_ = reference
        residual = log_rpm(train_rows) - np.log(reference.reference_rpm(train_rows).to_numpy())
        self.level_ = MarketLevel(self.market_table, self.level_terms).fit(train_rows, residual)
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if self.reference_ is None or self.level_ is None:
            raise RuntimeError("model is not fitted")
        ref = self.reference_.reference_rpm(rows).to_numpy()
        return rows["distance"].to_numpy(dtype=float) * ref * np.exp(self.level_.predict(rows))


@dataclass
class LevelCatBoostRate:
    """``log(rpm) = L(date) + f(x)``: market level + cross-sectional CatBoost.

    Fitting (``n_rounds`` = K backfitting rounds):

    1. initialise ``L`` on daily mean residuals ``log(rpm / ref_rpm)`` against the fold's
       lane x equipment reference (training rows only);
    2. fit CatBoost on ``log(rpm)`` with baseline ``L(date_i)`` -- trees learn ``f``, i.e.
       lane structure and per-load deviations, but not the daily level;
    3. for K >= 2: refit ``L`` on daily mean residuals ``log(rpm) - f(x)`` (CatBoost's
       in-sample tree output), then refit CatBoost with the new baseline; repeat.

    ``f`` absorbs the lane level, so ``L``'s intercept is only identified up to a constant
    between rounds; the slope coefficients are what backfitting can move.
    Prediction: ``rate = distance * exp(L(date) + f(x))``.
    """

    market_table: pd.DataFrame = field(repr=False)
    level_terms: tuple[str, ...] = LEVEL_TERMS
    n_rounds: int = 1
    params: dict[str, Any] = field(default_factory=dict)
    level_: MarketLevel | None = field(default=None, repr=False)
    cb_: CatBoostRate | None = field(default=None, repr=False)
    history_: list[dict[str, float]] = field(default_factory=list, repr=False)

    def fit(
        self,
        X: pd.DataFrame,
        train_rows: pd.DataFrame,
        cat_features: list[str],
        reference: RpmReference,
    ) -> "LevelCatBoostRate":
        """``X``: features of ``train_rows`` (same index); ``reference`` fitted on training rows."""
        if self.n_rounds < 1:
            raise ValueError("n_rounds must be >= 1")
        y = log_rpm(train_rows)
        residual = y - np.log(reference.reference_rpm(train_rows).to_numpy())
        self.history_ = []
        for k in range(self.n_rounds):
            if k > 0:
                residual = y - self.cb_.predict_raw(X)  # type: ignore[union-attr]
            self.level_ = MarketLevel(self.market_table, self.level_terms).fit(train_rows, residual)
            L = self.level_.predict(train_rows)
            self.cb_ = CatBoostRate(target="log_rpm", params=dict(self.params))
            self.cb_.fit(X, train_rows[config.TARGET], cat_features, baseline=L)
            in_sample = L + self.cb_.predict_raw(X)
            self.history_.append(
                {
                    "round": k + 1,
                    **self.level_.coefficients(),
                    "in_sample_rmse_log": float(np.sqrt(np.mean((y - in_sample) ** 2))),
                }
            )
        return self

    def predict_components(self, X: pd.DataFrame, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """``(L, f)`` per row on the log-rpm scale."""
        if self.level_ is None or self.cb_ is None:
            raise RuntimeError("model is not fitted")
        return self.level_.predict(rows), self.cb_.predict_raw(X)

    def predict(self, X: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
        """Dollars; ``rows`` supplies ``date`` for the level, ``X`` the tree features."""
        L, f = self.predict_components(X, rows)
        return X["distance"].to_numpy(dtype=float) * np.exp(L + f)

    def feature_importance(self) -> pd.DataFrame:
        if self.cb_ is None:
            raise RuntimeError("model is not fitted")
        return self.cb_.feature_importance()
