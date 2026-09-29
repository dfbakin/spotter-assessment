"""Final model: fit on all Jan-Oct, predict validation (Nov-Dec) and the December chart lane.

The model is the one selected in the backtests (``artifacts/RESULTS.md``):

* ``rate = distance * exp(L(date) + f(x))``;
* ``L`` -- :class:`freight.level.MarketLevel`, weighted OLS of the daily mean
  ``log(rpm / reference rpm)`` on ``[1, mi_day_mean, t, qe_ramp]``, fitted on ALL clean
  Jan-Oct rows (K = 1, no backfitting);
* ``f`` -- the ``cb_level_noqs`` CatBoost trees with ``L`` as baseline; hyper-parameters and
  tree features exactly as in ``artifacts/tuning/best_params.json`` (the ``cb_level_tuned``
  experiment config reads the same file);
* seed ensemble: one CatBoost per seed on the same ``L``; log-predictions are averaged
  (geometric mean of the rates).

Training rows are all Jan-Oct rows minus the corrupted targets flagged by an
:class:`freight.outliers.RpmReference` fitted on all Jan-Oct (``|log ratio| > log 1.6``).
The market table pools train + validation INPUT columns (no labels), exactly as in the
backtests. Chart rows get "Option A" market inputs (market_index / quote_signal := the day's
market-wide mean, so ``mi_dev = qs_dev = 0``) and coordinates from the city table.

Persistence: the CatBoost models go to ``artifacts/models/`` (git-ignored, ``.cbm``); the
model card with ``L``'s coefficients and the provenance to ``artifacts/final/model_card.json``.
:func:`load_final` re-derives ``L`` from the data and refuses to load if it no longer matches
the card.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from freight import config
from freight.backtest import EXPERIMENTS, BacktestData, _feature_view, catboost_params, load_backtest_data
from freight.data import clean, load_december, load_validation
from freight.features import build_features, city_coordinates, prepare_chart_rows
from freight.level import MarketLevel, qe_ramp
from freight.models import DEFAULT_CATBOOST_PARAMS, CatBoostRate, log_rpm
from freight.outliers import RpmReference, flag_corrupted

# cb_level_tuned = cb_level_noqs trees + best_params.json (chosen params = CatBoost defaults).
FINAL_EXPERIMENT: str = "cb_level_tuned"
FINAL_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
MODELS_DIR: Path = config.ARTIFACTS_DIR / "models"
FINAL_DIR: Path = config.ARTIFACTS_DIR / "final"
MODEL_CARD_PATH: Path = FINAL_DIR / "model_card.json"
VALIDATION_OUT: Path = config.ROOT / "validation_predictions.csv"
DECEMBER_OUT: Path = config.ROOT / "december_chart_predictions.csv"
COMPONENTS_OUT: Path = FINAL_DIR / "december_components.csv"
DECEMBER_START, DECEMBER_END = "2025-12-01", "2025-12-31"


# --------------------------------------------------------------------------- model object
@dataclass
class FinalModel:
    """``L`` + one CatBoost ``f`` per seed, plus everything needed to build features."""

    data: BacktestData = field(repr=False)
    reference: RpmReference = field(repr=False)
    level: MarketLevel = field(repr=False)
    models: list[CatBoostRate] = field(repr=False)
    seeds: tuple[int, ...]
    tree_features: list[str]
    cat_features: list[str]
    coords: pd.DataFrame = field(repr=False)
    card: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def market_table(self) -> pd.DataFrame:
        return self.data.market_table

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Tree-feature matrix for CLEANED rows (see :func:`freight.data.clean`)."""
        X, _ = build_features(rows, self.market_table, use_time=True)
        return X[self.tree_features]

    def tree_parts(self, X: pd.DataFrame) -> np.ndarray:
        """``f`` of every seed, shape ``(n_seeds, n_rows)``, log-rpm scale without baseline."""
        return np.vstack([m.predict_raw(X) for m in self.models])

    def components(self, rows: pd.DataFrame, level: MarketLevel | None = None) -> tuple[np.ndarray, np.ndarray]:
        """``(L, mean over seeds of f)`` per row; ``level`` overrides ``L`` (neutral market)."""
        X = self.features(rows)
        return (level or self.level).predict(rows), ensemble_log_mean(self.tree_parts(X))

    def predict_rows(self, rows: pd.DataFrame, level: MarketLevel | None = None) -> np.ndarray:
        """Dollars for cleaned rows: ``distance * exp(L + mean_s f_s)``."""
        L, f = self.components(rows, level)
        return rows["distance"].to_numpy(dtype=float) * np.exp(L + f)


def ensemble_log_mean(parts: np.ndarray) -> np.ndarray:
    """Average seed outputs on the log scale (rates: geometric mean across seeds)."""
    parts = np.atleast_2d(np.asarray(parts, dtype=float))
    return parts.mean(axis=0)


def neutral_market_table(market: pd.DataFrame, start: str = DECEMBER_START, end: str = DECEMBER_END) -> pd.DataFrame:
    """Copy of the market table with every ``mi_day_mean`` in ``[start, end]`` set to its mean."""
    out = market.copy()
    window = (out.index >= start) & (out.index <= end)
    out.loc[window, "mi_day_mean"] = out.loc[window, "mi_day_mean"].mean()
    return out


# --------------------------------------------------------------------------- fit / load
def _final_config():  # noqa: ANN202 - ExperimentConfig
    cfg = EXPERIMENTS[FINAL_EXPERIMENT]
    if cfg.model != "level_catboost" or cfg.backfit_rounds != 1:
        raise RuntimeError(f"{FINAL_EXPERIMENT} is expected to be a K=1 level_catboost config")
    return cfg


def _training_rows(data: BacktestData) -> tuple[RpmReference, pd.DataFrame, pd.Series]:
    """Reference fitted on all Jan-Oct rows; clean fit rows; the corrupted flags."""
    frame = data.frame
    reference = RpmReference().fit(frame)
    flagged = flag_corrupted(reference, frame, config.OUTLIER_LOG_RATIO_THRESHOLD)
    return reference, frame[~flagged], flagged


def _fit_level(data: BacktestData, reference: RpmReference, fit_rows: pd.DataFrame) -> MarketLevel:
    """``L`` initialised from the reference residuals, exactly as in ``LevelCatBoostRate`` (K=1)."""
    residual = log_rpm(fit_rows) - np.log(reference.reference_rpm(fit_rows).to_numpy())
    return MarketLevel(data.market_table, _final_config().level_terms).fit(fit_rows, residual)


def _tree_view(data: BacktestData) -> tuple[pd.DataFrame, list[str], list[str]]:
    cfg = _final_config()
    X, cats = _feature_view(cfg, data)
    best = json.loads(config.BEST_PARAMS_PATH.read_text())
    if list(X.columns) != best["tree_features"]:
        raise RuntimeError("tree features of the final config differ from best_params.json")
    leaked = set(X.columns) & (set(config.LEVEL_CB_EXCLUDED) | {"qs_dev"} | set(config.FORBIDDEN_FEATURES))
    if leaked:
        raise RuntimeError(f"level / month carriers in the tree features: {sorted(leaked)}")
    return X, list(X.columns), cats


def _coords(data: BacktestData) -> pd.DataFrame:
    return city_coordinates(data.frame, load_validation())


def fit_final(seeds: tuple[int, ...] = FINAL_SEEDS, data: BacktestData | None = None, verbose: bool = True) -> FinalModel:
    """Fit ``L`` once and one CatBoost per seed on all clean Jan-Oct rows (sequentially)."""
    data = data or load_backtest_data()
    reference, fit_rows, flagged = _training_rows(data)
    level = _fit_level(data, reference, fit_rows)
    X, tree_features, cats = _tree_view(data)
    L = level.predict(fit_rows)

    models: list[CatBoostRate] = []
    fit_seconds: list[float] = []
    for seed in seeds:
        t0 = time.perf_counter()
        cb = CatBoostRate(target="log_rpm", params=catboost_params(_final_config(), seed))
        cb.fit(X.loc[fit_rows.index], fit_rows[config.TARGET], cats, baseline=L)
        fit_seconds.append(round(time.perf_counter() - t0, 1))
        models.append(cb)
        if verbose:
            print(f"  CatBoost seed {seed}: {fit_seconds[-1]} s", flush=True)

    model = FinalModel(data, reference, level, models, tuple(seeds), tree_features, cats, _coords(data))
    in_sample = L + ensemble_log_mean(model.tree_parts(X.loc[fit_rows.index]))
    rmse_log = float(np.sqrt(np.mean((log_rpm(fit_rows) - in_sample) ** 2)))
    model.card = _model_card(model, fit_rows, flagged, fit_seconds, rmse_log)
    return model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "catboost", "matplotlib", "optuna"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def _model_path(seed: int) -> Path:
    return MODELS_DIR / f"catboost_seed{seed}.cbm"


def _model_card(
    model: FinalModel, fit_rows: pd.DataFrame, flagged: pd.Series, fit_seconds: list[float], rmse_log: float
) -> dict[str, Any]:
    best = json.loads(config.BEST_PARAMS_PATH.read_text())
    tuned = catboost_params(_final_config(), None)
    market = model.market_table
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "model": "rate = distance * exp(L(date) + f(x)); f = mean over seeds of CatBoost (log-scale average)",
        "experiment_config": FINAL_EXPERIMENT,
        "level": {
            "terms": ["const", *model.level.terms],
            "coefficients": {k: float(v) for k, v in model.level.coef_.items()},
            "r2_daily_weighted": float(model.level.r2_),
            "n_days": int(model.level.n_days_),
            "n_rows": int(model.level.n_rows_),
            "initialisation": "daily mean of log(rpm / RpmReference rpm), weighted by rows per day; K = 1",
            "t_origin": config.DATE_ORIGIN,
        },
        "catboost": {
            "params": {**DEFAULT_CATBOOST_PARAMS, **tuned, "random_seed": "per seed"},
            "params_source": str(config.BEST_PARAMS_PATH.relative_to(config.ROOT)),
            "params_kind": best["source"]["kind"],
            "seeds": list(model.seeds),
            "fit_seconds": dict(zip(map(str, model.seeds), fit_seconds)),
            "tree_features": model.tree_features,
            "cat_features": model.cat_features,
            "model_files": [str(_model_path(s).relative_to(config.ROOT)) for s in model.seeds],
        },
        "training": {
            "n_rows_window": int(len(model.data.frame)),
            "n_excluded_corrupted": int(flagged.sum()),
            "n_rows_used": int(len(fit_rows)),
            "date_range": [str(fit_rows["date"].min().date()), str(fit_rows["date"].max().date())],
            "outlier_threshold_log_ratio": float(config.OUTLIER_LOG_RATIO_THRESHOLD),
            "outlier_threshold_ratio": float(np.exp(config.OUTLIER_LOG_RATIO_THRESHOLD)),
            "outlier_reference": "RpmReference fitted on all Jan-Oct rows (two-stage hierarchical medians)",
            "in_sample_rmse_log_rpm": rmse_log,
        },
        "market_table": {
            "source": "INPUT columns of train + validation (no labels)",
            "date_range": [str(market.index.min().date()), str(market.index.max().date())],
        },
        "data_sha256": {
            p.name: _sha256(p) for p in (config.TRAIN_PATH, config.VALIDATION_PATH, config.DECEMBER_PATH)
        },
        "versions": _versions(),
    }


def save_final(model: FinalModel) -> None:
    """CatBoost models -> ``artifacts/models/``; model card -> ``artifacts/final/model_card.json``."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for seed, cb in zip(model.seeds, model.models):
        cb.model_.save_model(str(_model_path(seed)))  # type: ignore[union-attr]
    MODEL_CARD_PATH.write_text(json.dumps(model.card, indent=2) + "\n")


def models_available(seeds: tuple[int, ...] = FINAL_SEEDS) -> bool:
    return MODEL_CARD_PATH.exists() and all(_model_path(s).exists() for s in seeds)


def load_final(data: BacktestData | None = None) -> FinalModel:
    """Rebuild data, reference and ``L`` from the inputs; load the saved CatBoost models.

    Raises if the re-derived ``L`` or the excluded-row count differ from the model card, i.e.
    if the data changed since training.
    """
    card = json.loads(MODEL_CARD_PATH.read_text())
    data = data or load_backtest_data()
    reference, fit_rows, flagged = _training_rows(data)
    level = _fit_level(data, reference, fit_rows)
    saved = pd.Series(card["level"]["coefficients"])
    if int(flagged.sum()) != card["training"]["n_excluded_corrupted"] or not np.allclose(
        level.coef_.reindex(saved.index).to_numpy(), saved.to_numpy(), rtol=1e-9, atol=1e-12
    ):
        raise RuntimeError("data no longer reproduce the model card; rerun `make train`")
    _, tree_features, cats = _tree_view(data)
    if tree_features != card["catboost"]["tree_features"]:
        raise RuntimeError("tree features differ from the model card; rerun `make train`")
    seeds = tuple(card["catboost"]["seeds"])
    models = []
    for seed in seeds:
        cb = CatBoostRate(target="log_rpm", params=catboost_params(_final_config(), seed))
        cb.model_ = CatBoostRegressor()
        cb.model_.load_model(str(_model_path(seed)))
        cb.feature_names_, cb.cat_features_ = list(tree_features), list(cats)
        models.append(cb)
    return FinalModel(data, reference, level, models, seeds, tree_features, cats, _coords(data), card)


def load_or_fit(verbose: bool = True) -> FinalModel:
    """Load the saved final model, or fit and save it when the files are missing."""
    if models_available():
        return load_final()
    if verbose:
        print("saved models not found -> fitting the final model", flush=True)
    model = fit_final(verbose=verbose)
    save_final(model)
    return model


# --------------------------------------------------------------------------- predictions
def validation_rows() -> pd.DataFrame:
    """Cleaned validation rows (weight flags, mi_missing), original order."""
    return clean(load_validation())


def predict_validation(model: FinalModel) -> pd.DataFrame:
    """``load_id, predicted_rate`` (unrounded) in the order of the submission template."""
    rows = validation_rows()
    preds = pd.DataFrame({"load_id": rows["load_id"], "predicted_rate": model.predict_rows(rows)})
    template = pd.read_csv(config.TEMPLATE_PATH)
    return format_validation(template, preds)


def format_validation(template: pd.DataFrame, preds: pd.DataFrame, decimals: int | None = None) -> pd.DataFrame:
    """Fill the template's ``predicted_rate`` by ``load_id`` (template order and columns kept)."""
    lookup = preds.set_index("load_id")["predicted_rate"]
    if lookup.index.duplicated().any():
        raise ValueError("duplicate load_id in predictions")
    out = template[["load_id"]].copy()
    out["predicted_rate"] = out["load_id"].map(lookup).to_numpy(dtype=float)
    if out["predicted_rate"].isna().any():
        raise ValueError(f"{int(out['predicted_rate'].isna().sum())} template load_ids have no prediction")
    if decimals is not None:
        out["predicted_rate"] = out["predicted_rate"].round(decimals)
    return out[list(template.columns)]


def raw_december() -> pd.DataFrame:
    """The chart input file as written (dates kept as strings, original column order)."""
    return pd.read_csv(config.DECEMBER_PATH)


def december_rows(model: FinalModel) -> pd.DataFrame:
    """Cleaned chart rows with coordinates and Option-A market inputs."""
    return prepare_chart_rows(load_december(), model.market_table, model.coords)


def neutral_level(model: FinalModel) -> MarketLevel:
    """``L`` evaluated on the neutral-market table (December mi_day_mean := its mean)."""
    return replace(model.level, market_table=neutral_market_table(model.market_table))


def format_december(raw: pd.DataFrame, rates: np.ndarray, decimals: int | None = None) -> pd.DataFrame:
    """The chart file with ``predicted_rate`` filled; every other cell is left as read."""
    if len(rates) != len(raw):
        raise ValueError("one prediction per chart row expected")
    out = raw.copy()
    out["predicted_rate"] = np.asarray(rates, dtype=float)
    if decimals is not None:
        out["predicted_rate"] = out["predicted_rate"].round(decimals)
    return out[list(raw.columns)]


def predict_december(model: FinalModel, neutral: bool = False) -> pd.DataFrame:
    """The 31 chart rows (original columns / order) with ``predicted_rate`` filled (Option A).

    ``neutral=True`` replaces every December mi_day_mean by the December average in ``L``.
    """
    rows = december_rows(model)
    rates = model.predict_rows(rows, neutral_level(model) if neutral else None)
    return format_december(raw_december(), rates)


def december_components(model: FinalModel) -> pd.DataFrame:
    """Per chart date: Option-A and neutral predictions and their parts (log-rpm scale)."""
    rows = december_rows(model)
    X = model.features(rows)
    parts = model.tree_parts(X)
    f = ensemble_log_mean(parts)
    L = model.level.predict(rows)
    L_neutral = neutral_level(model).predict(rows)
    distance = rows["distance"].to_numpy(dtype=float)
    dates = pd.to_datetime(rows["date"]).dt.normalize()
    return pd.DataFrame(
        {
            "date": dates.dt.date.astype(str).to_numpy(),
            "dow": dates.dt.day_name().str[:3].to_numpy(),
            "pred_optionA": distance * np.exp(L + f),
            "pred_neutral": distance * np.exp(L_neutral + f),
            "L": L,
            "L_neutral": L_neutral,
            "mi_day_mean": model.market_table["mi_day_mean"].reindex(dates.to_numpy()).to_numpy(),
            "qe_ramp": qe_ramp(dates),
            "tree_f": f,
            "tree_f_seed_sd": parts.std(axis=0, ddof=1) if len(parts) > 1 else np.zeros(len(f)),
        }
    )


def write_deliverables(model: FinalModel) -> dict[str, Path]:
    """Write the two root CSVs (2 decimals) and ``artifacts/final/december_components.csv``."""
    template = pd.read_csv(config.TEMPLATE_PATH)
    validation = format_validation(template, predict_validation(model), decimals=2)
    validation.to_csv(VALIDATION_OUT, index=False)

    december = predict_december(model)
    december["predicted_rate"] = december["predicted_rate"].round(2)
    december.to_csv(DECEMBER_OUT, index=False)

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    december_components(model).round(6).to_csv(COMPONENTS_OUT, index=False)
    return {"validation": VALIDATION_OUT, "december": DECEMBER_OUT, "components": COMPONENTS_OUT}
