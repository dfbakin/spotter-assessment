"""Score one finished TabPack run in dollars and write its artifacts (run with the MAIN venv).

    .venv/bin/python scripts/tabpack/evaluate_run.py --run tabpack/local/runs/tp_level_F1_s0 \
        --data tabpack/data/freight-level-nolane-F1 --experiment tp_level --fold F1 --seed 0

Reads ``<run>/ensemble_greedy_predictions.npz`` (written by run_tabpack.py; the online greedy
ensemble's predictions on the target scale), ``<run>/driver.json`` and ``<data>/meta.csv``;
the rate is ``distance * exp(L + f_hat)`` (L = 0 for the standalone model). Metrics use
``freight.evaluate`` / ``freight.backtest.month_metrics``, exactly as the CatBoost backtests.

Writes (``--out``, default artifacts/tabpack):
* ``runs/<name>.json``   metrics (test + inner val), month bias, slices, runtime, epochs, ensemble;
* ``preds/<name>.csv``   per-row test predictions (``f`` = TabPack output on the log scale);
* ``configs/<name>.json`` the TabPack config.json of the run.
For ``--fold FINAL`` (unlabelled test rows) only predictions, config and runtime are written.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from freight import config
from freight.backtest import month_metrics
from freight.evaluate import all_and_clean, day_debiased_mae, standard_slices

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--experiment", required=True)
ap.add_argument("--fold", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", default=str(config.ARTIFACTS_DIR / "tabpack"))
args = ap.parse_args()

run, ds, out = Path(args.run), Path(args.data), Path(args.out)
name = f"{args.experiment}_{args.fold}_s{args.seed}"
meta = pd.read_csv(ds / "meta.csv", parse_dates=["date"])
preds = np.load(run / "ensemble_greedy_predictions.npz")
driver = json.loads((run / "driver.json").read_text())
export = json.loads((ds / "export.json").read_text())
tp_config = json.loads((run / "config.json").read_text())


def frame_for(part: str) -> pd.DataFrame:
    m = meta[meta["part"] == part].reset_index(drop=True)
    f = np.asarray(preds[part], dtype=float)
    assert len(f) == len(m), (part, len(f), len(m))
    m["f"] = f
    m["pred"] = m["distance"] * np.exp(m["L"] + f)
    return m


def score(m: pd.DataFrame) -> dict:
    res = all_and_clean(m["actual"], m["pred"].to_numpy(), m["clean"])
    clean = m["clean"].to_numpy(dtype=bool)
    return {
        "n": int(len(m)),
        "n_clean": int(clean.sum()),
        "mae_clean": res["clean"]["mae"],
        "mae_all": res["all"]["mae"],
        "mape_clean": res["clean"]["mape"],
        "rmse_clean": res["clean"]["rmse"],
        "median_bias_clean": res["clean"]["median_bias"],
        "mae_debiased": day_debiased_mae(m["actual"], m["pred"], m["date"], m["clean"]),
        "rmse_log_clean": float(np.sqrt(np.mean((m["target"][clean] - m["f"][clean]) ** 2))),
    }


ens = driver.get("ensemble_greedy", {})
runtime = {
    "device": driver["device"],
    "gpu": driver.get("gpu"),
    "torch": driver["torch"],
    "wall_s": driver["wall_s"],
    "wall_min": round(driver["wall_s"] / 60, 2),
    "epochs_run": driver.get("epochs_run", len(driver["epoch_end_times_s"])),
    "s_per_epoch_after_first": driver["s_per_epoch_after_first"],
    "n_models_config": driver.get("n_models_config", tp_config["n_models"]),
    "n_models_finished": driver["n_models_finished"],
    "model_best_epoch": driver.get("model_best_epoch"),
    "ensemble_size": ens.get("size", len(ens.get("ids", []))),
    "ensemble_n_unique_models": ens.get("n_unique_models", len(set(ens.get("ids", [])))),
    "ensemble_member_epochs": ens.get("epochs"),
    "peak_rss_mb": driver["peak_rss_mb"],
    "cuda_max_reserved_mb": driver.get("cuda_max_reserved_mb"),
}

test = frame_for("test")
for sub in ("runs", "preds", "configs"):
    (out / sub).mkdir(parents=True, exist_ok=True)
shutil.copyfile(run / "config.json", out / "configs" / f"{name}.json")

cols = ["load_id", "source", "date", "equipment", "distance", "actual", "clean", "city_status", "lane_train_count", "L", "f", "pred"]
csv = test[cols].copy()
csv["date"] = csv["date"].dt.strftime("%Y-%m-%d")
csv["L"] = csv["L"].round(7)
csv["f"] = csv["f"].round(7)
csv["pred"] = csv["pred"].round(4)
csv.to_csv(out / "preds" / f"{name}.csv", index=False)

result = {
    "name": name,
    "experiment": args.experiment,
    "fold": args.fold,
    "seed": args.seed,
    "dataset": ds.name,
    "export": {k: export[k] for k in ("train_window", "val_window", "test_window", "n", "n_train_flagged_dropped",
                                        "n_test_flagged", "lane_mode", "lane_topk", "cat_cols", "num_cols",
                                        "cat_cardinality_train", "level_coef") if k in export},
    "tabpack_config": {k: tp_config[k] for k in ("n_models", "batch_size", "patience") } | {
        "d_block": tp_config["model"]["d_block"],
        "embeddings": tp_config["model"].get("num_embeddings", {}).get("type"),
        "optimizer": tp_config["optimizer"]["type"],
        "ensemble": tp_config["online_ensembles"]["greedy"]["options"] | {"patience": tp_config["online_ensembles"]["greedy"]["patience"]},
    },
    "runtime": runtime,
}
if args.fold != "FINAL":
    val = frame_for("val")
    result["test"] = score(test)
    result["val"] = score(val)
    result["month"] = month_metrics(test)
    result["slices"] = {k: v.to_dict(orient="records") for k, v in standard_slices(test).items()}
(out / "runs" / f"{name}.json").write_text(json.dumps(result, indent=2, default=str))

if args.fold != "FINAL":
    t = result["test"]
    months = " ".join(f"{k}:{v['median_bias']:+.4f}" for k, v in sorted(result["month"].items()))
    print(
        f"[{name}] test MAE clean {t['mae_clean']:.2f} all {t['mae_all']:.2f} MAPE {t['mape_clean']:.2f}% "
        f"debiased {t['mae_debiased']:.2f} | bias {months} | val MAE {result['val']['mae_clean']:.2f} | "
        f"{runtime['device']} {runtime['wall_min']} min, {runtime['epochs_run']} epochs, ens {runtime['ensemble_size']}"
    )
else:
    print(f"[{name}] wrote {len(csv)} predictions; {runtime['device']} {runtime['wall_min']} min")
