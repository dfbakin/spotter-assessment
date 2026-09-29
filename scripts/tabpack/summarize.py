"""Summary tables of the TabPack backtest runs (run with the MAIN venv from the project root).

    .venv/bin/python scripts/tabpack/summarize.py

Reads every ``artifacts/tabpack/runs/<exp>_<fold>_s<seed>.json`` (FINAL runs excluded) and the
CatBoost references ``artifacts/backtest/preds/{cb_level_noqs,cb_time}_<fold>_r<seed>.csv``
(3 seeds each, re-scored here with the same metric code). Writes:

* ``artifacts/tabpack/runs_summary.csv``  one row per run: metrics, month bias, runtime, epochs,
  ensemble size, device;
* ``artifacts/tabpack/table_a.csv``       per experiment x fold: mean and sd over seeds.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from freight import config
from freight.backtest import month_metrics
from freight.evaluate import all_and_clean, day_debiased_mae

OUT = config.ARTIFACTS_DIR / "tabpack"
REFERENCES = {"cb_level_noqs": (0, 1, 2), "cb_time": (0, 1, 2)}
FOLDS = ("F1", "F2", "F3")
EXP_ORDER = ["cb_time", "cb_level_noqs", "tp_standalone", "tp_level", "tp_level_lanetopk", "tp_level_lanefull"]


def reference_rows() -> list[dict]:
    rows = []
    for exp, seeds in REFERENCES.items():
        for fold in FOLDS:
            for seed in seeds:
                path = config.PREDS_DIR / f"{exp}_{fold}_r{seed}.csv"
                if not path.exists():
                    continue
                p = pd.read_csv(path, parse_dates=["date"])
                res = all_and_clean(p["actual"], p["pred"].to_numpy(), p["clean"])
                months = sorted(month_metrics(p).items())
                rows.append(
                    {
                        "experiment": exp,
                        "fold": fold,
                        "seed": seed,
                        "source": f"artifacts/backtest/preds/{path.name}",
                        "mae_clean": res["clean"]["mae"],
                        "mae_all": res["all"]["mae"],
                        "mape_clean": res["clean"]["mape"],
                        "mae_debiased": day_debiased_mae(p["actual"], p["pred"], p["date"], p["clean"]),
                        "bias_m1": months[0][1]["median_bias"],
                        "bias_m2": months[1][1]["median_bias"],
                        "month_m1": months[0][0],
                        "month_m2": months[1][0],
                    }
                )
    return rows


def tabpack_rows() -> list[dict]:
    rows = []
    for path in sorted((OUT / "runs").glob("*.json")):
        r = json.loads(path.read_text())
        if r["fold"] == "FINAL" or "test" not in r:
            continue
        months = sorted(r["month"].items())
        rt = r["runtime"]
        rows.append(
            {
                "experiment": r["experiment"],
                "fold": r["fold"],
                "seed": r["seed"],
                "source": f"artifacts/tabpack/runs/{path.name}",
                "mae_clean": r["test"]["mae_clean"],
                "mae_all": r["test"]["mae_all"],
                "mape_clean": r["test"]["mape_clean"],
                "mae_debiased": r["test"]["mae_debiased"],
                "bias_m1": months[0][1]["median_bias"],
                "bias_m2": months[1][1]["median_bias"],
                "month_m1": months[0][0],
                "month_m2": months[1][0],
                "val_mae_clean": r["val"]["mae_clean"],
                "val_rmse_log": r["val"]["rmse_log_clean"],
                "device": rt["device"],
                "wall_min": rt["wall_min"],
                "epochs_run": rt["epochs_run"],
                "s_per_epoch": rt["s_per_epoch_after_first"],
                "n_models": rt["n_models_config"],
                "n_models_finished": rt["n_models_finished"],
                "best_epoch_median": (rt.get("model_best_epoch") or {}).get("median"),
                "ensemble_size": rt["ensemble_size"],
                "ensemble_n_unique": rt["ensemble_n_unique_models"],
                "n_train": r["export"]["n"]["train"],
                "n_val": r["export"]["n"]["val"],
                "n_test": r["export"]["n"]["test"],
            }
        )
    return rows


def main() -> None:
    runs = pd.DataFrame(reference_rows() + tabpack_rows())
    runs["exp_rank"] = runs["experiment"].map({e: i for i, e in enumerate(EXP_ORDER)})
    runs = runs.sort_values(["exp_rank", "fold", "seed"]).drop(columns="exp_rank").reset_index(drop=True)
    runs.to_csv(OUT / "runs_summary.csv", index=False)

    metrics = ["mae_clean", "mae_all", "mape_clean", "mae_debiased", "bias_m1", "bias_m2"]
    table = []
    for (exp, fold), g in runs.groupby(["experiment", "fold"], sort=False):
        row = {"experiment": exp, "fold": fold, "n_seeds": len(g)}
        for m in metrics:
            row[m] = g[m].mean()
            row[f"{m}_sd"] = g[m].std(ddof=1) if len(g) > 1 else np.nan
        row["month_m1"], row["month_m2"] = g["month_m1"].iloc[0], g["month_m2"].iloc[0]
        table.append(row)
    table_a = pd.DataFrame(table)
    # F1+F2 mean (selection folds) per experiment, when both folds are present.
    f12 = (
        table_a[table_a["fold"].isin(["F1", "F2"])]
        .groupby("experiment")
        .filter(lambda g: len(g) == 2)
        .groupby("experiment", sort=False)[["mae_clean", "mae_all", "mape_clean", "mae_debiased"]]
        .mean()
        .reset_index()
        .assign(fold="F1+F2")
    )
    table_a = pd.concat([table_a, f12], ignore_index=True)
    table_a["exp_rank"] = table_a["experiment"].map({e: i for i, e in enumerate(EXP_ORDER)})
    table_a["fold_rank"] = table_a["fold"].map({"F1": 0, "F2": 1, "F1+F2": 2, "F3": 3})
    table_a = table_a.sort_values(["exp_rank", "fold_rank"]).drop(columns=["exp_rank", "fold_rank"]).reset_index(drop=True)
    table_a.round(4).to_csv(OUT / "table_a.csv", index=False)

    def fmt(r: pd.Series) -> str:
        sd = r.get("mae_clean_sd")
        mae = f"{r['mae_clean']:.2f}" + (f" ± {sd:.2f}" if pd.notna(sd) else "")
        bias = (
            f"{100 * r['bias_m1']:+.2f}% / {100 * r['bias_m2']:+.2f}%" if pd.notna(r.get("bias_m1")) else "–"
        )
        n = int(r["n_seeds"]) if pd.notna(r.get("n_seeds")) else "–"
        return (
            f"| {r['experiment']} | {r['fold']} | {n} | {mae} | {r['mae_all']:.2f} | {r['mape_clean']:.2f}% "
            f"| {r['mae_debiased']:.2f} | {bias} |"
        )

    print("| experiment | fold | n | MAE clean | MAE all | MAPE | debiased | bias m1 / m2 |")
    print("|---|---|---|---|---|---|---|---|")
    for _, r in table_a.iterrows():
        print(fmt(r))
    tp = runs[runs["experiment"].str.startswith("tp_")]
    if len(tp):
        print()
        print(tp[["experiment", "fold", "seed", "device", "wall_min", "epochs_run", "s_per_epoch", "n_models_finished",
                  "best_epoch_median", "ensemble_size", "ensemble_n_unique", "val_mae_clean"]].to_string(index=False))


if __name__ == "__main__":
    main()
