"""Phase 2b: Optuna tuning of the cb_level_noqs CatBoost on F1+F2 (day-debiased MAE).

Steps (default: ``all`` = study + reeval):
    study    TPE (seed 0) + median pruner (F1 = step 0), sequential; stops at --n-trials trials
             in the study or after --timeout-min minutes. Resumes the SQLite study if present.
             -> artifacts/tuning/optuna_study.db, trials.csv
    reeval   defaults + top --top-k trials with seeds 0, 1, 2 on F1+F2; choose the best mean
             day-debiased MAE (ties within $0.5 -> fastest).
             -> reeval_runs.csv, reeval.csv, best_params.json
    summary  after `scripts/backtest.py tuned`: default vs tuned (and references) on F1/F2/F3.
             -> final_comparison.csv, final_month_bias.csv, final_slices_f3.csv

    .venv/bin/python scripts/tune.py                 # study + reeval
    .venv/bin/python scripts/tune.py study --n-trials 80 --timeout-min 75
    .venv/bin/python scripts/tune.py summary
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from freight import config
from freight.backtest import load_backtest_data
from freight.tuning import (
    TUNING_EXCLUDE,
    choose,
    create_study,
    export_trials,
    final_comparison,
    reevaluate,
    reevaluation_candidates,
    run_study,
    study_stats,
    summarise_reeval,
    tree_features,
    write_best_params,
)

OUT = config.TUNING_DIR


def step_study(data, n_trials: int, timeout_min: float) -> None:  # noqa: ANN001 - BacktestData
    study = create_study()
    t0 = time.time()
    run_study(study, data, n_trials=n_trials, timeout_s=60 * timeout_min, exclude=TUNING_EXCLUDE)
    export_trials(study, OUT / "trials.csv")
    print(f"\nstudy: {study_stats(study)}  (this call {(time.time() - t0) / 60:.1f} min)")
    print(f"best trial #{study.best_trial.number}: F1+F2 day-debiased {study.best_value:.2f}  params {study.best_trial.params}")


def step_reeval(data, top_k: int) -> None:  # noqa: ANN001 - BacktestData
    study = create_study()
    candidates = reevaluation_candidates(study, top_k=top_k)
    runs = reevaluate(data, candidates, exclude=TUNING_EXCLUDE, out_path=OUT / "reeval_runs.csv")
    runs.to_csv(OUT / "reeval_runs.csv", index=False)
    summary = summarise_reeval(runs, candidates)
    summary.to_csv(OUT / "reeval.csv", index=False)
    chosen = choose(summary)
    trial = next(t for t in candidates if t.number == int(chosen["trial"]))
    blob = write_best_params(
        config.BEST_PARAMS_PATH, chosen, trial, study, summary, tree_features(data, TUNING_EXCLUDE), TUNING_EXCLUDE
    )
    cols = [
        "candidate",
        "mae_debiased_f12_mean",
        "mae_debiased_f12_sd",
        "mae_clean_f12_mean",
        "mae_clean_f12_sd",
        "mae_all_f12_mean",
        "fit_seconds_f12",
    ]
    with pd.option_context("display.width", 200, "display.precision", 2):
        print("\nRe-evaluation, F1+F2 mean over seeds 0-2 (sd across seeds)")
        print(summary[cols].to_string(index=False))
    print(f"\nchosen: {chosen['candidate']}  params {blob['params']}")
    print(f"wrote {config.BEST_PARAMS_PATH}")


def step_summary() -> None:
    out = final_comparison()
    out["comparison"].to_csv(OUT / "final_comparison.csv", index=False)
    out["month_bias"].to_csv(OUT / "final_month_bias.csv", index=False)
    out["slices_f3"].to_csv(OUT / "final_slices_f3.csv", index=False)
    comp = out["comparison"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 2):
        cols = [
            "experiment",
            "fold",
            "n_runs",
            "mae_clean_mean",
            "mae_clean_sd",
            "mae_all_mean",
            "mape_clean_mean",
            "rmse_all_mean",
            "mae_debiased_mean",
            "bias_m1_mean",
            "bias_m2_mean",
        ]
        print("\nFinal comparison (mean / sd over seeds)")
        print(comp[cols].to_string(index=False))
        print("\nF3 slices (clean rows, mean over seeds)")
        print(out["slices_f3"].to_string(index=False))
    print(f"\nwrote final_comparison.csv, final_month_bias.csv, final_slices_f3.csv to {OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", nargs="?", default="all", choices=["all", "study", "reeval", "summary"])
    parser.add_argument("--n-trials", type=int, default=80, help="stop when the study holds this many trials")
    parser.add_argument("--timeout-min", type=float, default=75.0, help="wall-clock budget of this study call")
    parser.add_argument("--top-k", type=int, default=5, help="non-default trials to re-evaluate")
    args = parser.parse_args()

    if args.step == "summary":
        step_summary()
        return
    data = load_backtest_data()
    if args.step in ("all", "study"):
        step_study(data, args.n_trials, args.timeout_min)
    if args.step in ("all", "reeval"):
        step_reeval(data, args.top_k)


if __name__ == "__main__":
    main()
