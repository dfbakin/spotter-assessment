"""Forward-time backtest harness.

One :class:`ExperimentConfig` = one hypothesis. The harness runs it over the requested folds
with a strict information cutoff per fold:

* features come from input columns only (computed once for all rows);
* everything label-derived -- the outlier reference, the baseline medians, the model -- is
  fitted on that fold's training rows;
* test labels are used only for scoring and for the evaluation "clean" mask.

Outputs (under ``artifacts/backtest``): ``<name>.json`` with config, per-run metrics,
median bias by test month, level coefficients, per-fold mean/sd across seeds, threshold and
counts; one appended row per run in ``summary.csv``; per-run predictions in
``preds/<name>_<key>.csv``.

A run is one (fold, cold-start seed, CatBoost seed) combination. Its key is the fold name,
plus ``_s<k>`` for a cold-start city draw and ``_r<k>`` for a CatBoost ``random_seed`` when
the experiment lists ``seeds`` (otherwise CatBoost uses ``config.SEED``, as in phase 1).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from freight import config
from freight.data import clean, load_train, load_validation
from freight.evaluate import all_and_clean, day_debiased_mae, regression_metrics, standard_slices
from freight.features import build_features, select_features
from freight.level import LEVEL_TERMS
from freight.market import build_market_table
from freight.models import (
    CatBoostRate,
    LaneLevelBaseline,
    LaneMedianBaseline,
    LevelCatBoostRate,
    TargetMode,
    load_tuned_params,
)
from freight.outliers import RpmReference, flag_corrupted
from freight.split import Fold, cold_start_holdout, get_folds

PRED_KEY_COLUMNS: list[str] = [
    "load_id",
    "date",
    "pickup",
    "delivery",
    "equipment",
    "distance",
    "weight",
    "market_index",
]


# --------------------------------------------------------------------------- configs
ModelKind = Literal["baseline", "catboost", "level_baseline", "level_catboost"]


@dataclass(frozen=True)
class ExperimentConfig:
    exp_id: str
    name: str
    description: str
    model: ModelKind = "catboost"
    target: TargetMode = "log_rpm"
    drop_outliers: bool = True  # drop flagged corrupted rows from TRAINING
    use_time: bool = False  # add the t_days time coordinate
    exclude_features: tuple[str, ...] = ()
    extra_features: tuple[str, ...] = ()  # opt-in columns from config.OPTIONAL_FEATURES
    # Explicit feature whitelist; when non-empty it replaces the exclude-based selection
    # (``exclude_features`` is still subtracted from it, ``extra_features`` added).
    features: tuple[str, ...] = ()
    folds: tuple[str, ...] = tuple(config.FOLD_DATES)
    cold_start_seeds: tuple[int, ...] = ()  # non-empty -> run the cold-start-city protocol
    seeds: tuple[int, ...] = ()  # CatBoost random_seed values; empty -> config.SEED once
    importance_folds: tuple[str, ...] = ()
    catboost_params: dict[str, Any] = field(default_factory=dict)
    # market-level models (freight.level): which terms L uses, and backfitting rounds K
    level_terms: tuple[str, ...] = LEVEL_TERMS
    backfit_rounds: int = 1
    # JSON file (relative to config.ROOT) whose "params" are merged under catboost_params,
    # e.g. the phase-2b tuning result artifacts/tuning/best_params.json. Read at run time.
    params_file: str = ""


_LEVEL_CB = tuple(config.LEVEL_CB_FEATURES)
_NO_TREND = tuple(t for t in LEVEL_TERMS if t != "t")
_NO_RAMP = tuple(t for t in LEVEL_TERMS if t != "qe_ramp")


EXPERIMENTS: dict[str, ExperimentConfig] = {
    c.name: c
    for c in [
        ExperimentConfig("E0", "yardstick", "Lane-median rpm x distance, fit on clean training rows", model="baseline"),
        ExperimentConfig(
            "E1", "cb_base", "CatBoost, log_rpm, corrupted rows dropped, no time coordinate", importance_folds=("F3",)
        ),
        ExperimentConfig("E2", "cb_lograte", "E1 with a log_rate target", target="log_rate"),
        # Re-run with CatBoost seeds 0-2 in phase 2a as the multi-seed reference.
        ExperimentConfig("E3", "cb_time", "E1 + t_days time coordinate", use_time=True, seeds=config.LEVEL_SEEDS),
        ExperimentConfig("E4", "cb_keep_outliers", "E1 but corrupted rows kept in training", drop_outliers=False),
        ExperimentConfig(
            "E5",
            "cb_raw_market",
            "E1 without daily-market-table features (raw market_index + quote_signal only)",
            exclude_features=tuple(config.MARKET_DERIVED_FEATURES),
        ),
        ExperimentConfig(
            "E6",
            "cb_cold_start",
            "E1 with 8 random cities removed from training (seeds 0, 1): cold vs warm MAE",
            folds=("F2", "F3"),
            cold_start_seeds=(0, 1),
        ),
        # Diagnostics (added after E0-E6): in a short expanding window days_to_quarter_end
        # identifies the calendar month (e.g. 0-29 = March only when training on Jan-Apr),
        # so together with slow market features it can act as a hidden time proxy.
        ExperimentConfig(
            "E7", "cb_no_qtr", "E1 without days_to_quarter_end", exclude_features=("days_to_quarter_end",)
        ),
        ExperimentConfig(
            "E8",
            "cb_raw_market_no_qtr",
            "E5 without days_to_quarter_end",
            exclude_features=(*config.MARKET_DERIVED_FEATURES, "days_to_quarter_end"),
        ),
        ExperimentConfig(
            "E9",
            "cb_time_no_qtr",
            "E3 without days_to_quarter_end",
            use_time=True,
            exclude_features=("days_to_quarter_end",),
        ),
        # quote_signal's monthly offset is a second hidden month identifier (see config).
        ExperimentConfig(
            "E10",
            "cb_qs_dev",
            "E1 with quote_signal replaced by qs_dev (quote_signal - day mean)",
            exclude_features=("quote_signal",),
            extra_features=("qs_dev",),
        ),
        ExperimentConfig(
            "E11",
            "cb_qs_dev_no_ma",
            "E10 without the slow trailing means mi_ma7 / mi_ma28",
            exclude_features=("quote_signal", "mi_ma7", "mi_ma28"),
            extra_features=("qs_dev",),
        ),
        # ---- phase 2a: log(rpm) = L(date) + f(x), see freight.level / LevelCatBoostRate.
        ExperimentConfig(
            "E12",
            "yardstick_level",
            "Lane-median rpm x exp(L(date)); L = OLS of daily residuals on [1, mi_day_mean, t, qe_ramp]",
            model="level_baseline",
        ),
        ExperimentConfig(
            "E13",
            "cb_level",
            "L(date) as CatBoost baseline; trees on cross-sectional features only (K=1)",
            model="level_catboost",
            features=_LEVEL_CB,
            seeds=config.LEVEL_SEEDS,
            importance_folds=("F3",),
        ),
        ExperimentConfig(
            "E14",
            "cb_level_mi",
            "cb_level + mi_day_mean as a tree feature (level x lane interactions)",
            model="level_catboost",
            features=(*_LEVEL_CB, "mi_day_mean"),
            seeds=config.LEVEL_SEEDS,
        ),
        ExperimentConfig(
            "E15",
            "cb_level_noqs",
            "cb_level without qs_dev",
            model="level_catboost",
            features=_LEVEL_CB,
            exclude_features=("qs_dev",),
            # Selected in phase 2a (seeds 0, 1); seed 2 added in phase 2b as the default-params
            # reference for cb_level_tuned (seeds 0, 1 reproduce exactly).
            seeds=config.LEVEL_SEEDS,
        ),
        ExperimentConfig(
            "E16",
            "cb_level_notrend",
            "cb_level with L = [1, mi_day_mean, qe_ramp] (no trend: 'what if the drift stops')",
            model="level_catboost",
            features=_LEVEL_CB,
            level_terms=_NO_TREND,
            seeds=(0, 1),
        ),
        ExperimentConfig(
            "E17",
            "cb_level_noramp",
            "cb_level with L = [1, mi_day_mean, t] (no quarter-end ramp)",
            model="level_catboost",
            features=_LEVEL_CB,
            level_terms=_NO_RAMP,
            seeds=(0, 1),
        ),
        ExperimentConfig(
            "E18",
            "cb_level_cold",
            "cb_level with 8 random cities removed from training (city draws 0, 1; CatBoost seed 0)",
            model="level_catboost",
            features=_LEVEL_CB,
            folds=("F3",),
            cold_start_seeds=(0, 1),
            seeds=(0,),
        ),
        ExperimentConfig(
            "E19",
            "cb_level_k2",
            "cb_level with K=2 backfitting rounds (L refit on log(rpm) - f(x)), seed 0",
            model="level_catboost",
            features=_LEVEL_CB,
            seeds=(0,),
            backfit_rounds=2,
        ),
        # ---- phase 2b: CatBoost hyper-parameters of f tuned with Optuna on F1+F2
        # (freight.tuning, scripts/tune.py); F3 is scored once with the chosen params.
        ExperimentConfig(
            "E20",
            "cb_level_tuned",
            "cb_level_noqs with Optuna-tuned CatBoost params (artifacts/tuning/best_params.json)",
            model="level_catboost",
            features=_LEVEL_CB,
            exclude_features=("qs_dev",),
            seeds=config.LEVEL_SEEDS,
            importance_folds=("F3",),
            params_file=str(config.BEST_PARAMS_PATH.relative_to(config.ROOT)),
        ),
    ]
}


# Phase 2a group (``scripts/backtest.py level``): level-model experiments plus references.
LEVEL_EXPERIMENTS: tuple[str, ...] = (
    "yardstick",
    "yardstick_level",
    "cb_level",
    "cb_level_k2",
    "cb_time",
    "cb_level_mi",
    "cb_level_noqs",
    "cb_level_notrend",
    "cb_level_noramp",
    "cb_level_cold",
)
# Phase 2b group (``scripts/backtest.py tuned``): selected config with default vs tuned params.
TUNED_EXPERIMENTS: tuple[str, ...] = ("cb_level_noqs", "cb_level_tuned")
EXPERIMENT_GROUPS: dict[str, tuple[str, ...]] = {"level": LEVEL_EXPERIMENTS, "tuned": TUNED_EXPERIMENTS}


# --------------------------------------------------------------------------- data
@dataclass
class BacktestData:
    """Cleaned labelled rows plus the full feature matrix (inputs only, built once)."""

    frame: pd.DataFrame
    X: pd.DataFrame  # includes t_days; dropped unless an experiment asks for it
    cat_features: list[str]
    market_table: pd.DataFrame  # train + validation inputs; covers Jan-Dec


def load_backtest_data() -> BacktestData:
    train = clean(load_train())
    # Market table pools train + validation INPUT columns (no labels) -- allowed and
    # identical to what is available when predicting Nov-Dec.
    validation = load_validation()
    market_table = build_market_table(train, validation)
    del validation
    X, cat_features = build_features(train, market_table, use_time=True)
    return BacktestData(frame=train, X=X, cat_features=cat_features, market_table=market_table)


# --------------------------------------------------------------------------- core
def _feature_view(cfg: ExperimentConfig, data: BacktestData) -> tuple[pd.DataFrame, list[str]]:
    if cfg.features:
        wanted = [c for c in (*cfg.features, *cfg.extra_features) if c not in cfg.exclude_features]
        unknown = set(wanted) - set(data.X.columns)
        if unknown:
            raise KeyError(f"{cfg.name}: unknown features {sorted(unknown)}")
        wanted = list(dict.fromkeys(wanted))
        return data.X[wanted], [c for c in data.cat_features if c in wanted]
    exclude = list(cfg.exclude_features)
    exclude += [c for c in config.OPTIONAL_FEATURES if c not in cfg.extra_features]
    if not cfg.use_time:
        exclude += config.TIME_FEATURES
    return select_features(data.X, data.cat_features, exclude)


def catboost_params(cfg: ExperimentConfig, seed: int | None) -> dict[str, Any]:
    """Experiment CatBoost overrides (``params_file`` first, then ``catboost_params``) plus
    the run's ``random_seed`` (if seeded)."""
    params = load_tuned_params(config.ROOT / cfg.params_file) if cfg.params_file else {}
    params.update(cfg.catboost_params)
    if seed is not None:
        params["random_seed"] = seed
    return params


def _fit_predict(
    cfg: ExperimentConfig,
    data: BacktestData,
    X: pd.DataFrame,
    cat_features: list[str],
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    reference: RpmReference,
    seed: int | None,
) -> tuple[np.ndarray, pd.DataFrame | None, list[dict[str, float]] | None]:
    """Returns ``(test predictions, feature importance or None, level history or None)``.

    ``train_rows`` are the rows the model is fitted on (clean training rows by default);
    ``reference`` is the fold's rpm reference fitted on the training window.
    """
    if cfg.model == "baseline":
        return LaneMedianBaseline().fit(train_rows).predict(test_rows), None, None
    if cfg.model == "level_baseline":
        yard = LaneLevelBaseline(data.market_table, cfg.level_terms).fit(train_rows, reference)
        return yard.predict(test_rows), None, [yard.level_.coefficients()]  # type: ignore[union-attr]
    if cfg.model == "level_catboost":
        if cfg.target != "log_rpm":
            raise ValueError("level_catboost only supports the log_rpm target")
        lvl = LevelCatBoostRate(
            data.market_table, cfg.level_terms, cfg.backfit_rounds, catboost_params(cfg, seed)
        )
        lvl.fit(X.loc[train_rows.index], train_rows, cat_features, reference)
        return lvl.predict(X.loc[test_rows.index], test_rows), lvl.feature_importance(), lvl.history_
    model = CatBoostRate(target=cfg.target, params=catboost_params(cfg, seed))
    model.fit(X.loc[train_rows.index], train_rows[config.TARGET], cat_features)
    return model.predict(X.loc[test_rows.index]), model.feature_importance(), None


def _city_status(train_rows: pd.DataFrame, test_rows: pd.DataFrame, cold_mask: pd.Series | None) -> pd.Series:
    """'cold' if a test row touches a city absent from (or removed from) training."""
    if cold_mask is None:
        seen = set(train_rows["pickup"]) | set(train_rows["delivery"])
        cold_mask = ~(test_rows["pickup"].isin(seen) & test_rows["delivery"].isin(seen))
    return pd.Series(np.where(cold_mask, "cold", "warm"), index=test_rows.index)


def month_metrics(preds: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Clean-row median bias / MAE / n per calendar month of the test window."""
    clean_preds = preds[preds["clean"]]
    month = pd.to_datetime(clean_preds["date"]).dt.to_period("M").astype(str)
    out: dict[str, dict[str, float]] = {}
    for key, grp in clean_preds.groupby(month):
        out[str(key)] = {
            "median_bias": float(np.median(grp["pred"] / grp["actual"]) - 1),
            "mae": float(np.mean(np.abs(grp["pred"] - grp["actual"]))),
            "n": int(len(grp)),
        }
    return out


def run_fold(
    cfg: ExperimentConfig,
    data: BacktestData,
    fold: Fold,
    cold_seed: int | None = None,
    seed: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame | None]:
    """Fit on the fold's training window, score its test window.

    ``cold_seed`` selects a cold-start city draw, ``seed`` the CatBoost ``random_seed``
    (``None`` keeps ``config.SEED``). Returns ``(result dict, prediction frame, feature
    importance or None)``.
    """
    frame = data.frame
    train_rows = frame[fold.train_mask(frame)]
    test_rows = frame[fold.test_mask(frame)]

    # Evaluation clean mask: reference fitted on the full fold training window, so every
    # experiment on this fold (cold-start variants included) is scored on identical rows.
    eval_reference = RpmReference().fit(train_rows)
    test_log_ratio = eval_reference.log_ratio(test_rows)
    test_flagged = test_log_ratio.abs() > config.OUTLIER_LOG_RATIO_THRESHOLD

    cold = None
    if cold_seed is not None:
        cold = cold_start_holdout(train_rows, test_rows, n_cities=8, seed=cold_seed)
        train_rows = train_rows[cold.train_keep]

    # Training outlier flags come from a reference fitted on exactly the rows the model sees.
    train_reference = eval_reference if cold is None else RpmReference().fit(train_rows)
    train_flagged = flag_corrupted(train_reference, train_rows)
    fit_rows = train_rows[~train_flagged] if cfg.drop_outliers else train_rows

    X, cat_features = _feature_view(cfg, data)
    t0 = time.perf_counter()
    pred, importance, level_history = _fit_predict(
        cfg, data, X, cat_features, fit_rows, test_rows, train_reference, seed
    )
    fit_seconds = time.perf_counter() - t0

    clean_mask = ~test_flagged
    preds = test_rows[PRED_KEY_COLUMNS].copy()
    preds["actual"] = test_rows[config.TARGET]
    preds["pred"] = pred
    preds["clean"] = clean_mask
    preds["log_ratio"] = test_log_ratio
    preds["city_status"] = _city_status(train_rows, test_rows, None if cold is None else cold.test_cold)
    metrics = all_and_clean(preds["actual"], preds["pred"], preds["clean"])
    # Selection instrument (phase 2b): clean MAE after removing each test day's median log
    # error, i.e. the cross-sectional error of f without L's level error. Not a performance number.
    metrics["clean"]["mae_debiased"] = day_debiased_mae(preds["actual"], preds["pred"], preds["date"], preds["clean"])

    result: dict[str, Any] = {
        "fold": fold.name,
        "cb_seed": seed,
        "cold_seed": cold_seed,
        "train_period": [str(fold.train_start.date()), str(fold.train_end.date())],
        "test_period": [str(fold.test_start.date()), str(fold.test_end.date())],
        "n_train_window": int(fold.train_mask(frame).sum()),
        "n_train_available": len(train_rows),
        "n_train_flagged": int(train_flagged.sum()),
        "n_train_used": len(fit_rows),
        "n_test": len(test_rows),
        "n_test_flagged": int(test_flagged.sum()),
        "fit_seconds": round(fit_seconds, 1),
        "metrics": metrics,
        "month": month_metrics(preds),
        "slices": {k: v.to_dict(orient="records") for k, v in standard_slices(preds).items()},
    }
    if level_history is not None:
        result["level"] = level_history
    if cold is not None:
        clean_preds = preds[preds["clean"]]
        result["cold_start"] = {
            "seed": cold_seed,
            "cities": list(cold.cities),
            "n_train_removed": int(fold.train_mask(frame).sum() - len(train_rows)),
            "cold_clean": regression_metrics(*_ap(clean_preds[clean_preds["city_status"] == "cold"])),
            "warm_clean": regression_metrics(*_ap(clean_preds[clean_preds["city_status"] == "warm"])),
        }
    return result, preds, importance


def _ap(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return frame["actual"], frame["pred"]


def _summary_row(cfg: ExperimentConfig, key: str, res: dict[str, Any], run_at: str) -> dict[str, Any]:
    m_all, m_clean = res["metrics"]["all"], res["metrics"]["clean"]
    row: dict[str, Any] = {
        "run_at": run_at,
        "exp_id": cfg.exp_id,
        "experiment": cfg.name,
        "fold": key,
        "fold_name": res["fold"],
        "cb_seed": res.get("cb_seed"),
        "cold_seed": res.get("cold_seed"),
        "n_train_used": res["n_train_used"],
        "n_test": res["n_test"],
        "n_test_clean": int(m_clean["n"]),
        "mae_all": m_all["mae"],
        "mae_clean": m_clean["mae"],
        "rmse_all": m_all["rmse"],
        "rmse_clean": m_clean["rmse"],
        "mape_all": m_all["mape"],
        "mape_clean": m_clean["mape"],
        "medae_clean": m_clean["medae"],
        "r2_clean": m_clean["r2"],
        "median_bias_clean": m_clean["median_bias"],
        "mae_debiased_clean": m_clean.get("mae_debiased", float("nan")),
    }
    for i, (month, mm) in enumerate(sorted(res.get("month", {}).items()), start=1):
        row[f"bias_m{i}"] = mm["median_bias"]
        row[f"month_m{i}"] = month
    if "cold_start" in res:
        row["mae_clean_cold"] = res["cold_start"]["cold_clean"]["mae"]
        row["mae_clean_warm"] = res["cold_start"]["warm_clean"]["mae"]
    return row


def run_experiment(
    cfg: ExperimentConfig,
    data: BacktestData | None = None,
    folds: list[str] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Run ``cfg`` over its folds (optionally restricted to ``folds``) and write artifacts."""
    data = data or load_backtest_data()
    fold_names = [f for f in cfg.folds if folds is None or f in folds]
    run_at = datetime.now().isoformat(timespec="seconds")
    cold_seeds: list[int | None] = list(cfg.cold_start_seeds) or [None]
    cb_seeds: list[int | None] = list(cfg.seeds) or [None]
    if cfg.model in ("baseline", "level_baseline"):
        cb_seeds = [None]  # deterministic

    fold_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    runs = [(f, c, s) for f in get_folds(fold_names) for c in cold_seeds for s in cb_seeds]
    for fold, cold_seed, seed in runs:
        key = fold.name + ("" if cold_seed is None else f"_s{cold_seed}") + ("" if seed is None else f"_r{seed}")
        print(f"[{cfg.name}] {key} ...", flush=True)
        res, preds, importance = run_fold(cfg, data, fold, cold_seed=cold_seed, seed=seed)
        fold_results[key] = res
        summary_rows.append(_summary_row(cfg, key, res, run_at))
        m = res["metrics"]
        print(
            f"[{cfg.name}] {key}: MAE clean {m['clean']['mae']:.2f}  all {m['all']['mae']:.2f}"
            f"  MAPE clean {m['clean']['mape']:.2f}%  day-debiased {m['clean']['mae_debiased']:.2f}  bias/month "
            + " ".join(f"{k[-2:]}:{v['median_bias']:+.3f}" for k, v in sorted(res["month"].items()))
            + f"  ({res['fit_seconds']}s)",
            flush=True,
        )
        if write:
            config.PREDS_DIR.mkdir(parents=True, exist_ok=True)
            preds.to_csv(config.PREDS_DIR / f"{cfg.name}_{key}.csv", index=False)
            first_seed = seed is None or seed == cb_seeds[0]
            if importance is not None and fold.name in cfg.importance_folds and first_seed:
                path = config.BACKTEST_DIR / f"{cfg.exp_id.lower()}_{key.lower()}_importance.csv"
                importance.to_csv(path, index=False)

    report = {
        "exp_id": cfg.exp_id,
        "name": cfg.name,
        "run_at": run_at,
        "config": asdict(cfg),
        "outlier_threshold": config.OUTLIER_LOG_RATIO_THRESHOLD,
        "seed": config.SEED if not cfg.seeds else list(cfg.seeds),
        "folds": fold_results,
        "mean": _mean_metrics(fold_results),
        "by_fold": fold_seed_stats(fold_results),
    }
    if write:
        config.BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        (config.BACKTEST_DIR / f"{cfg.name}.json").write_text(json.dumps(report, indent=2, default=str))
        _append_summary(pd.DataFrame(summary_rows))
    return report


SEED_STAT_METRICS: dict[str, tuple[str, str]] = {
    "mae_clean": ("clean", "mae"),
    "mae_all": ("all", "mae"),
    "mape_clean": ("clean", "mape"),
    "rmse_all": ("all", "rmse"),
    "rmse_clean": ("clean", "rmse"),
    "median_bias_clean": ("clean", "median_bias"),
    "mae_debiased_clean": ("clean", "mae_debiased"),
}


def _mean_sd(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}


def fold_seed_stats(fold_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per fold: mean and sd (ddof=1) across runs (seeds / city draws) of the key metrics
    and of the clean median bias per test month."""
    by_fold: dict[str, list[dict[str, Any]]] = {}
    for res in fold_results.values():
        by_fold.setdefault(res["fold"], []).append(res)
    out: dict[str, dict[str, Any]] = {}
    for fold, runs in by_fold.items():
        stats: dict[str, Any] = {"n_runs": len(runs)}
        for name, (part, metric) in SEED_STAT_METRICS.items():
            if all(metric in r["metrics"][part] for r in runs):
                stats[name] = _mean_sd([r["metrics"][part][metric] for r in runs])
        months = sorted({m for r in runs for m in r.get("month", {})})
        stats["month_bias"] = {m: _mean_sd([r["month"][m]["median_bias"] for r in runs]) for m in months}
        if all("cold_start" in r for r in runs):
            stats["mae_clean_cold"] = _mean_sd([r["cold_start"]["cold_clean"]["mae"] for r in runs])
            stats["mae_clean_warm"] = _mean_sd([r["cold_start"]["warm_clean"]["mae"] for r in runs])
        out[fold] = stats
    return out


def _mean_metrics(fold_results: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for part in ["all", "clean"]:
        frames = pd.DataFrame([r["metrics"][part] for r in fold_results.values()])
        out[part] = frames.mean(numeric_only=True).to_dict()
    return out


def _append_summary(rows: pd.DataFrame) -> None:
    path = config.BACKTEST_DIR / "summary.csv"
    if path.exists():
        rows = pd.concat([pd.read_csv(path), rows], ignore_index=True)
    rows.to_csv(path, index=False)


def latest_summary() -> pd.DataFrame:
    """``summary.csv`` keeping only the most recent run of each experiment x fold."""
    df = pd.read_csv(config.BACKTEST_DIR / "summary.csv")
    return df.sort_values("run_at").drop_duplicates(["experiment", "fold"], keep="last").reset_index(drop=True)
