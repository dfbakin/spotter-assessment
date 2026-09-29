"""Phase 2b: feature-family ablation and Optuna tuning of the cross-sectional CatBoost ``f(x)``.

The model is ``log(rpm) = L(date) + f(x)`` (:class:`freight.models.LevelCatBoostRate`) on the
feature set selected in phase 2a, ``cb_level_noqs``. ``L`` has no hyper-parameters; only the
trees of ``f`` are tuned here.

Selection instrument: **day-debiased MAE** (:func:`freight.evaluate.day_debiased_mae`). On
F1/F2 most of the plain-MAE error is the level extrapolation of ``L`` (median bias per test
month up to -2.3% / +2.6%), which tree hyper-parameters cannot change and which would swamp
tree-quality differences. Removing each test day's median log error isolates the
cross-sectional error the trees own. Plain MAE (clean and all) is always recorded next to it.

Protocol (forward-time folds, ``config.FOLD_DATES``): everything is SELECTED on F1 + F2 only.
F3 is scored once, with the final choice, through the ``cb_level_tuned`` experiment.

* :func:`run_ablation` -- mask one feature family at a time (default params, seed 0).
* :func:`create_study` / :func:`make_objective` -- TPE search; F1 is reported as step 0 so the
  median pruner can stop a trial before its F2 fit. CatBoost's defaults are enqueued as
  trial 0 so the baseline lives inside the study.
* :func:`reevaluate` / :func:`choose` -- top trials and the defaults re-run with 3 seeds.
* :func:`write_best_params` -- ``artifacts/tuning/best_params.json`` (read by the
  ``cb_level_tuned`` experiment and the final fit).
* :func:`final_comparison` -- default vs tuned (and references) on F1/F2/F3 from the
  backtest JSONs + prediction files.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostError

from freight import config
from freight.backtest import EXPERIMENTS, BacktestData, ExperimentConfig, _feature_view, run_fold
from freight.evaluate import add_slice_columns, day_debiased_mae, slice_table
from freight.split import get_folds

BASE_EXPERIMENT: str = "cb_level_noqs"
TUNING_SEED: int = 0
REEVAL_SEEDS: tuple[int, ...] = config.LEVEL_SEEDS
STUDY_NAME: str = "cb_level_noqs_f12_day_debiased"
STUDY_PATH: Path = config.TUNING_DIR / "optuna_study.db"
# Ablation rule: a family is dropped for tuning only if masking it improves the F1+F2
# day-debiased MAE by more than this many dollars.
ABLATION_DROP_GAIN: float = 1.0
# Re-evaluation rule: candidates whose 3-seed F1+F2 day-debiased MAE is within this many
# dollars of the best count as tied; among them the fastest to fit wins.
TIE_TOLERANCE: float = 0.5
# Feature families removed for tuning after the ablation (none: see artifacts/tuning/ablation.csv).
TUNING_EXCLUDE: tuple[str, ...] = ()

# CatBoost's CPU behaviour for every experiment so far (DEFAULT_CATBOOST_PARAMS plus CatBoost's
# own defaults: MVS bootstrap with subsample 0.8, 254 borders, one_hot_max_size 2, symmetric
# trees, random_strength 1), written in the search-space names so it can be enqueued as trial 0.
DEFAULT_TRIAL_PARAMS: dict[str, Any] = {
    "depth": 6,
    "learning_rate": 0.05,
    "iterations": 2000,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "border_count": 254,
    "one_hot_max_size": 2,
    "bootstrap_type": "MVS",
    "grow_policy": "SymmetricTree",
    "subsample": 0.8,
}

SCORE_KEYS: tuple[str, ...] = (
    "mae_debiased",
    "mae_clean",
    "mae_all",
    "mape_clean",
    "rmse_all",
    "median_bias_clean",
    "bias_m1",
    "bias_m2",
    "fit_seconds",
)


# --------------------------------------------------------------------------- scoring
def base_config(exclude: tuple[str, ...] = ()) -> ExperimentConfig:
    """``cb_level_noqs`` with extra feature columns masked."""
    base = EXPERIMENTS[BASE_EXPERIMENT]
    return replace(base, exclude_features=tuple(dict.fromkeys((*base.exclude_features, *exclude))))


def tree_features(data: BacktestData, exclude: tuple[str, ...] = ()) -> list[str]:
    """Columns the trees see for :func:`base_config` ``(exclude)``."""
    return list(_feature_view(base_config(exclude), data)[0].columns)


def score_fold(
    data: BacktestData,
    fold_name: str,
    params: dict[str, Any],
    seed: int,
    exclude: tuple[str, ...] = (),
) -> dict[str, float]:
    """Fit ``cb_level_noqs`` (+ ``params`` over the defaults, ``exclude`` masked) on one fold.

    Uses the backtest harness unchanged (same outlier handling, clean mask and level fit).
    Returns plain and day-debiased metrics, median bias per test month and fit seconds.
    """
    cfg = replace(base_config(exclude), name="tuning", catboost_params=dict(params), seeds=(seed,))
    res, preds, _ = run_fold(cfg, data, get_folds([fold_name])[0], seed=seed)
    m = res["metrics"]
    out = {
        "mae_debiased": m["clean"]["mae_debiased"],
        "mae_clean": m["clean"]["mae"],
        "mae_all": m["all"]["mae"],
        "mape_clean": m["clean"]["mape"],
        "rmse_all": m["all"]["rmse"],
        "median_bias_clean": m["clean"]["median_bias"],
        "fit_seconds": float(res["fit_seconds"]),
    }
    for i, (_, mm) in enumerate(sorted(res["month"].items()), start=1):
        out[f"bias_m{i}"] = mm["median_bias"]
    del preds
    gc.collect()
    return {k: float(v) for k, v in out.items()}


# --------------------------------------------------------------------------- ablation
def run_ablation(
    data: BacktestData,
    families: dict[str, tuple[str, ...]] | None = None,
    folds: tuple[str, ...] = tuple(config.FOLD_DATES),
    seed: int = TUNING_SEED,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Default params, one seed: the full ``cb_level_noqs`` set and each family masked.

    Long table: variant x fold with plain and day-debiased metrics. F3 is computed for
    information only; decisions use F1 + F2 (:func:`summarise_ablation`).
    """
    families = dict(config.ABLATION_FAMILIES if families is None else families)
    variants: dict[str, tuple[str, ...]] = {"full": (), **families}
    rows = []
    for name, cols in variants.items():
        n_features = len(tree_features(data, cols))
        for fold in folds:
            s = score_fold(data, fold, {}, seed, exclude=cols)
            rows.append({"variant": name, "dropped": " ".join(cols), "n_features": n_features, "fold": fold, "seed": seed, **s})
            log(
                f"[ablation] {name:8s} {fold}: day-debiased {s['mae_debiased']:.2f}  MAE clean {s['mae_clean']:.2f}"
                f"  all {s['mae_all']:.2f}  ({s['fit_seconds']:.0f}s)"
            )
    return pd.DataFrame(rows)


def summarise_ablation(runs: pd.DataFrame, gain: float = ABLATION_DROP_GAIN) -> pd.DataFrame:
    """One row per variant: per-fold and F1+F2-mean metrics, deltas vs ``full`` and the decision.

    ``decision`` = ``drop`` if masking the family lowers the F1+F2 day-debiased MAE by more
    than ``gain`` dollars, else ``keep``.
    """
    rows = []
    for variant, grp in runs.groupby("variant", sort=False):
        g = grp.set_index("fold")
        row: dict[str, Any] = {"variant": variant, "dropped": g["dropped"].iloc[0], "n_features": int(g["n_features"].iloc[0])}
        for metric in ("mae_debiased", "mae_clean", "mae_all"):
            for fold in g.index:
                row[f"{metric}_{fold}"] = g.loc[fold, metric]
            row[f"{metric}_f12"] = g.loc[list(config.TUNING_FOLDS), metric].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    full = out.set_index("variant").loc["full"]
    for metric in ("mae_debiased", "mae_clean", "mae_all"):
        out[f"delta_{metric}_f12"] = out[f"{metric}_f12"] - full[f"{metric}_f12"]
    out["decision"] = np.where(out["delta_mae_debiased_f12"] < -gain, "drop", "keep")
    out.loc[out["variant"] == "full", "decision"] = ""
    return out


# --------------------------------------------------------------------------- optuna
def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """CatBoost search space (loss stays RMSE on log rpm; no early stopping).

    ``iterations`` is a tuned fixed number, so the final refit uses exactly the same count.
    MVS (CatBoost's CPU default bootstrap) is added to the brief's {Bayesian, Bernoulli} so
    the defaults are a point of the space. ``one_hot_max_size`` 4 one-hot encodes equipment
    (3 levels); 2 leaves it to CTRs like the other categoricals.
    """
    params: dict[str, Any] = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "iterations": trial.suggest_int("iterations", 500, 4000, step=100),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        "border_count": trial.suggest_categorical("border_count", [128, 254]),
        "one_hot_max_size": trial.suggest_categorical("one_hot_max_size", [2, 4]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise"]),
    }
    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 3.0)
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    if params["grow_policy"] == "Depthwise":
        params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 100, log=True)
    return params


def create_study(path: Path = STUDY_PATH, study_name: str = STUDY_NAME) -> optuna.Study:
    """SQLite-backed study (resumes if it exists); enqueues the defaults as the first trial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{path}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0),
        load_if_exists=True,
    )
    if not study.trials:
        study.enqueue_trial(DEFAULT_TRIAL_PARAMS, user_attrs={"is_default": True})
    return study


def make_objective(data: BacktestData, exclude: tuple[str, ...] = TUNING_EXCLUDE) -> Callable[[optuna.Trial], float]:
    """Mean F1/F2 day-debiased MAE (seed 0). F1 is reported as step 0 and may prune the trial."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        is_default = bool(trial.user_attrs.get("is_default")) or trial.params == DEFAULT_TRIAL_PARAMS
        trial.set_user_attr("is_default", is_default)
        scores: dict[str, dict[str, float]] = {}
        for step, fold in enumerate(config.TUNING_FOLDS):
            s = score_fold(data, fold, params, TUNING_SEED, exclude)
            scores[fold] = s
            for key, value in s.items():
                trial.set_user_attr(f"{fold}_{key}", value)
            if step == 0:
                trial.report(s["mae_debiased"], step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        for key in ("mae_clean", "mae_all", "fit_seconds"):
            trial.set_user_attr(f"f12_{key}", float(np.mean([s[key] for s in scores.values()])))
        return float(np.mean([s["mae_debiased"] for s in scores.values()]))

    return objective


def _progress(t0: float) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        f1 = trial.user_attrs.get("F1_mae_debiased", float("nan"))
        done = trial.state == optuna.trial.TrialState.COMPLETE
        val = f"F1+F2 {trial.value:.2f}" if done and trial.value is not None else ""
        try:
            best = f"best {study.best_value:.2f} (#{study.best_trial.number})"
        except ValueError:
            best = "best -"
        p = trial.params
        print(
            f"[tune] #{trial.number:3d} {trial.state.name:8s} F1 {f1:6.2f} {val:14s} | {best} | "
            f"d{p.get('depth')} lr{p.get('learning_rate', 0):.3f} it{p.get('iterations')} "
            f"{p.get('grow_policy', '')[:5]} {p.get('bootstrap_type', '')[:5]} | {(time.time() - t0) / 60:.1f} min",
            flush=True,
        )

    return callback


def run_study(
    study: optuna.Study,
    data: BacktestData,
    n_trials: int,
    timeout_s: float,
    exclude: tuple[str, ...] = TUNING_EXCLUDE,
) -> optuna.Study:
    """Sequential optimisation (``n_jobs=1``) until ``n_trials`` trials exist or ``timeout_s`` passes."""
    started = [t for t in study.trials if t.state != optuna.trial.TrialState.WAITING]
    remaining = max(0, n_trials - len(started))
    if remaining == 0:
        return study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        make_objective(data, exclude),
        n_trials=remaining,
        timeout=timeout_s,
        n_jobs=1,
        gc_after_trial=True,
        catch=(CatBoostError,),
        callbacks=[_progress(time.time())],
    )
    return study


def export_trials(study: optuna.Study, path: Path) -> pd.DataFrame:
    """All trials (params, user attrs, state, duration) to CSV."""
    df = study.trials_dataframe(
        attrs=("number", "value", "datetime_start", "datetime_complete", "duration", "params", "user_attrs", "state")
    )
    if "duration" in df.columns:
        df["duration_s"] = pd.to_timedelta(df["duration"]).dt.total_seconds().round(1)
        df = df.drop(columns="duration")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def study_stats(study: optuna.Study) -> dict[str, Any]:
    states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
    starts = [t.datetime_start for t in study.trials if t.datetime_start]
    ends = [t.datetime_complete for t in study.trials if t.datetime_complete]
    wall = (max(ends) - min(starts)).total_seconds() / 60 if starts and ends else float("nan")
    busy = sum((t.duration.total_seconds() for t in study.trials if t.duration), 0.0) / 60
    return {"n_trials": len(study.trials), "states": states, "wall_clock_minutes": round(wall, 1), "trial_minutes": round(busy, 1)}


# --------------------------------------------------------------------------- re-evaluation
def reevaluation_candidates(study: optuna.Study, top_k: int = 5) -> list[optuna.trial.FrozenTrial]:
    """The default-params trial plus the ``top_k`` best non-default complete trials."""
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    default = [t for t in complete if t.user_attrs.get("is_default")]
    ranked = sorted((t for t in complete if not t.user_attrs.get("is_default")), key=lambda t: t.value)
    return default[:1] + ranked[:top_k]


def reevaluate(
    data: BacktestData,
    candidates: list[optuna.trial.FrozenTrial],
    seeds: tuple[int, ...] = REEVAL_SEEDS,
    exclude: tuple[str, ...] = TUNING_EXCLUDE,
    reuse_study_seed: bool = True,
    out_path: Path | None = None,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Candidates x seeds x (F1, F2). The study's own seed-0 scores are reused when
    ``reuse_study_seed`` (CatBoost on CPU with a fixed thread count is deterministic).
    Writes ``out_path`` after every candidate so an interruption loses little."""
    rows: list[dict[str, Any]] = []
    for t in candidates:
        label = "default" if t.user_attrs.get("is_default") else f"trial_{t.number}"
        for seed in seeds:
            for fold in config.TUNING_FOLDS:
                cached = reuse_study_seed and seed == TUNING_SEED and f"{fold}_mae_debiased" in t.user_attrs
                if cached:
                    s = {k: float(t.user_attrs[f"{fold}_{k}"]) for k in SCORE_KEYS if f"{fold}_{k}" in t.user_attrs}
                else:
                    s = score_fold(data, fold, dict(t.params), seed, exclude)
                source = "study" if cached else "refit"
                rows.append({"candidate": label, "trial": t.number, "seed": seed, "fold": fold, "source": source, **s})
                timing = "from study" if cached else f"{s['fit_seconds']:.0f}s"
                log(
                    f"[reeval] {label:10s} seed {seed} {fold}: day-debiased {s['mae_debiased']:.2f}"
                    f"  MAE clean {s['mae_clean']:.2f}  ({timing})"
                )
        if out_path is not None:
            pd.DataFrame(rows).to_csv(out_path, index=False)
    return pd.DataFrame(rows)


def summarise_reeval(runs: pd.DataFrame, candidates: list[optuna.trial.FrozenTrial] | None = None) -> pd.DataFrame:
    """Per candidate: mean and sd over seeds of the F1+F2 mean (and per fold), fit seconds."""
    rows = []
    params = {t.number: t.params for t in candidates or []}
    for (label, number), grp in runs.groupby(["candidate", "trial"], sort=False):
        per_seed = grp.groupby("seed")
        row: dict[str, Any] = {"candidate": label, "trial": int(number), "n_seeds": grp["seed"].nunique()}
        for metric in ("mae_debiased", "mae_clean", "mae_all"):
            f12 = per_seed[metric].mean()
            row[f"{metric}_f12_mean"] = f12.mean()
            row[f"{metric}_f12_sd"] = f12.std(ddof=1) if len(f12) > 1 else 0.0
            for fold, g in grp.groupby("fold"):
                row[f"{metric}_{fold}_mean"] = g[metric].mean()
                row[f"{metric}_{fold}_sd"] = g[metric].std(ddof=1) if len(g) > 1 else 0.0
        row["fit_seconds_f12"] = per_seed["fit_seconds"].sum().mean() / len(config.TUNING_FOLDS)
        if number in params:
            row["params"] = json.dumps(params[number], sort_keys=True)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mae_debiased_f12_mean").reset_index(drop=True)


def choose(summary: pd.DataFrame, tol: float = TIE_TOLERANCE) -> pd.Series:
    """Best mean F1+F2 day-debiased MAE; among candidates within ``tol`` of it, the fastest."""
    best = summary["mae_debiased_f12_mean"].min()
    tied = summary[summary["mae_debiased_f12_mean"] <= best + tol]
    return tied.sort_values(["fit_seconds_f12", "mae_debiased_f12_mean"]).iloc[0]


def write_best_params(
    path: Path,
    chosen: pd.Series,
    trial: optuna.trial.FrozenTrial,
    study: optuna.Study,
    summary: pd.DataFrame,
    features: list[str],
    exclude: tuple[str, ...] = TUNING_EXCLUDE,
) -> dict[str, Any]:
    """``best_params.json``: the chosen CatBoost params, their F1+F2 scores and provenance."""
    per_fold = {
        fold: {
            metric: {"mean": float(chosen[f"{metric}_{fold}_mean"]), "sd": float(chosen[f"{metric}_{fold}_sd"])}
            for metric in ("mae_debiased", "mae_clean", "mae_all")
        }
        for fold in config.TUNING_FOLDS
    }
    per_fold["F1+F2"] = {
        metric: {"mean": float(chosen[f"{metric}_f12_mean"]), "sd": float(chosen[f"{metric}_f12_sd"])}
        for metric in ("mae_debiased", "mae_clean", "mae_all")
    }
    blob = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "base_experiment": BASE_EXPERIMENT,
        "excluded_families": list(exclude),
        "tree_features": features,
        "params": dict(trial.params),
        "source": {
            "kind": "default" if trial.user_attrs.get("is_default") else "optuna_trial",
            "trial_number": trial.number,
            "study_name": study.study_name,
            "storage": str(STUDY_PATH.relative_to(config.ROOT)),
            **study_stats(study),
        },
        "objective": "mean over F1, F2 of the day-debiased clean MAE ($), CatBoost seed 0",
        "selection": {
            "rule": (
                f"re-evaluate the defaults and the top trials with seeds {list(REEVAL_SEEDS)} on F1+F2; "
                f"best mean day-debiased MAE, ties within ${TIE_TOLERANCE} broken by fit time"
            ),
            "seeds": list(REEVAL_SEEDS),
            "candidates": summary[["candidate", "trial", "mae_debiased_f12_mean", "mae_debiased_f12_sd", "mae_clean_f12_mean", "fit_seconds_f12"]]
            .round(3)
            .to_dict(orient="records"),
            "chosen": chosen["candidate"],
        },
        "scores": per_fold,
        "fit_seconds_per_fold": round(float(chosen["fit_seconds_f12"]), 1),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2))
    return blob


# --------------------------------------------------------------------------- final comparison
FINAL_EXPERIMENTS: tuple[str, ...] = ("yardstick_level", "cb_level", "cb_level_noqs", "cb_level_tuned")


def _experiment_runs(name: str) -> dict[str, tuple[dict[str, Any], pd.DataFrame]]:
    """``run key -> (result dict, predictions)`` from a backtest JSON and its prediction files."""
    report = json.loads((config.BACKTEST_DIR / f"{name}.json").read_text())
    out = {}
    for key, res in report["folds"].items():
        preds = pd.read_csv(config.PREDS_DIR / f"{name}_{key}.csv", parse_dates=["date"])
        out[key] = (res, preds)
    return out


def final_comparison(experiments: tuple[str, ...] = FINAL_EXPERIMENTS) -> dict[str, pd.DataFrame]:
    """Experiment x fold (mean / sd over seeds): plain MAE clean/all, MAPE, RMSE, day-debiased
    MAE (recomputed from the prediction files, so older runs get it too) and median bias per
    test month; plus F3 slices by equipment and distance band (mean / sd over seeds)."""
    table, months, slices = [], [], []
    for name in experiments:
        if not (config.BACKTEST_DIR / f"{name}.json").exists():
            print(f"  (skipping {name}: no backtest JSON)")
            continue
        runs = _experiment_runs(name)
        per_run = []
        for key, (res, preds) in runs.items():
            m = res["metrics"]
            row = {
                "fold": res["fold"],
                "key": key,
                "mae_clean": m["clean"]["mae"],
                "mae_all": m["all"]["mae"],
                "mape_clean": m["clean"]["mape"],
                "rmse_all": m["all"]["rmse"],
                "mae_debiased": day_debiased_mae(preds["actual"], preds["pred"], preds["date"], preds["clean"]),
            }
            for i, (month, mm) in enumerate(sorted(res["month"].items()), start=1):
                row[f"month_m{i}"] = month
                row[f"bias_m{i}"] = mm["median_bias"]
                months.append({"experiment": name, "run": key, "fold": res["fold"], "month": month, "median_bias_clean": mm["median_bias"], "mae_clean": mm["mae"]})
            per_run.append(row)
            if res["fold"] == config.HOLDOUT_FOLD:
                framed = add_slice_columns(preds)
                for by in ("equipment", "distance_band"):
                    t = slice_table(framed, by).rename(columns={by: "group"})
                    slices.append(t.assign(experiment=name, slice=by, run=key))
        df = pd.DataFrame(per_run)
        for fold, grp in df.groupby("fold"):
            out: dict[str, Any] = {"experiment": name, "fold": fold, "n_runs": len(grp)}
            for metric in ("mae_clean", "mae_all", "mape_clean", "rmse_all", "mae_debiased", "bias_m1", "bias_m2"):
                out[f"{metric}_mean"] = grp[metric].mean()
                out[f"{metric}_sd"] = grp[metric].std(ddof=1) if len(grp) > 1 else 0.0
            out["month_m1"], out["month_m2"] = grp["month_m1"].iloc[0], grp["month_m2"].iloc[0]
            table.append(out)
    slice_df = pd.concat(slices, ignore_index=True) if slices else pd.DataFrame()
    if not slice_df.empty:
        slice_df = (
            slice_df.groupby(["experiment", "slice", "group"], sort=False)
            .agg(n=("n", "first"), mae_mean=("mae", "mean"), mae_sd=("mae", "std"), mape_mean=("mape", "mean"), median_bias_mean=("median_bias", "mean"), n_runs=("run", "nunique"))
            .reset_index()
        )
        # natural order: equipment alphabetically, distance bands by their lower edge
        lower = pd.to_numeric(slice_df["group"].astype(str).str.extract(r"^(\d+)")[0], errors="coerce")
        slice_df = (
            slice_df.assign(_exp=slice_df["experiment"].map({n: i for i, n in enumerate(experiments)}), _lo=lower)
            .sort_values(["_exp", "slice", "_lo", "group"], ascending=[True, False, True, True])
            .drop(columns=["_exp", "_lo"])
            .reset_index(drop=True)
        )
    return {"comparison": pd.DataFrame(table), "month_bias": pd.DataFrame(months), "slices_f3": slice_df}
