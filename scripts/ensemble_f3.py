"""For information: the final 5-seed ensemble scored on the F3 holdout (Jan-Aug -> Sep-Oct).

Runs the final config (cb_level_tuned = cb_level_noqs trees + best_params.json) through the
backtest harness (freight.backtest.run_fold) with CatBoost seeds 0-4, averages the seeds'
log-predictions (geometric mean, as in freight.final) and scores it like any run. No decision
depends on it. Writes artifacts/final/ensemble_f3.json.
    .venv/bin/python scripts/ensemble_f3.py
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from freight import config
from freight.backtest import EXPERIMENTS, load_backtest_data, month_metrics, run_fold
from freight.evaluate import all_and_clean, day_debiased_mae
from freight.final import FINAL_DIR, FINAL_EXPERIMENT, FINAL_SEEDS, ensemble_log_mean
from freight.split import get_folds

FOLD = config.HOLDOUT_FOLD


def main() -> None:
    cfg = EXPERIMENTS[FINAL_EXPERIMENT]
    data = load_backtest_data()
    fold = get_folds([FOLD])[0]
    singles, logs, base = {}, [], None
    for seed in FINAL_SEEDS:
        res, preds, _ = run_fold(cfg, data, fold, seed=seed)
        singles[str(seed)] = {
            "mae_clean": res["metrics"]["clean"]["mae"],
            "mae_all": res["metrics"]["all"]["mae"],
            "mae_debiased": res["metrics"]["clean"]["mae_debiased"],
            "mape_clean": res["metrics"]["clean"]["mape"],
        }
        print(f"{FOLD} seed {seed}: MAE clean {singles[str(seed)]['mae_clean']:.3f}", flush=True)
        logs.append(np.log(preds["pred"].to_numpy()))
        base = preds if base is None else base
    ens = base.copy()
    ens["pred"] = np.exp(ensemble_log_mean(np.vstack(logs)))
    metrics = all_and_clean(ens["actual"], ens["pred"], ens["clean"])
    single_clean = np.array([s["mae_clean"] for s in singles.values()])
    out = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "experiment": FINAL_EXPERIMENT,
        "fold": FOLD,
        "seeds": list(FINAL_SEEDS),
        "ensemble": {
            "mae_clean": metrics["clean"]["mae"],
            "mae_all": metrics["all"]["mae"],
            "mape_clean": metrics["clean"]["mape"],
            "rmse_all": metrics["all"]["rmse"],
            "median_bias_clean": metrics["clean"]["median_bias"],
            "mae_debiased": day_debiased_mae(ens["actual"], ens["pred"], ens["date"], ens["clean"]),
            "month": month_metrics(ens),
        },
        "single_seed": singles,
        "single_seed_mae_clean_mean": float(single_clean.mean()),
        "single_seed_mae_clean_sd": float(single_clean.std(ddof=1)),
    }
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "ensemble_f3.json").write_text(json.dumps(out, indent=2) + "\n")
    e = out["ensemble"]
    print(
        f"{FOLD} 5-seed ensemble: MAE clean {e['mae_clean']:.2f}  all {e['mae_all']:.2f}  MAPE {e['mape_clean']:.2f}%"
        f"  debiased {e['mae_debiased']:.2f}  | single seeds {single_clean.mean():.2f} ± {single_clean.std(ddof=1):.2f}"
    )


if __name__ == "__main__":
    main()
