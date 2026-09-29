"""Phase-2a diagnostics for the market-level model (no CatBoost training).

Reads the experiment JSONs written by ``scripts/backtest.py level`` and writes, under
``artifacts/backtest``:

* ``level_summary.csv``      experiment x fold: mean / sd across seeds of MAE (clean, all),
                             MAPE clean, RMSE all, median bias, bias per test month;
* ``level_month_bias.csv``   long table: experiment, run key, seed, test month, median bias;
* ``level_coefficients.json`` L coefficients per fold (F1, F2, F3) and on ALL Jan-Oct clean
                             rows, each fold's out-of-sample daily level error, and the K=2 vs
                             K=1 backfitting comparison;
* ``level_index.png``        daily index (training labels) vs the all-data L, extended over
                             Nov-Dec with the validation market table.

It also prints the report tables. Run after the backtests:
    .venv/bin/python scripts/level_diagnostics.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freight import config
from freight.backtest import LEVEL_EXPERIMENTS, load_backtest_data
from freight.level import LEVEL_TERMS, MarketLevel, daily_mean
from freight.models import log_rpm
from freight.outliers import RpmReference, flag_corrupted
from freight.split import get_folds

TABLE_EXPERIMENTS = [
    "yardstick",
    "yardstick_level",
    "cb_time",
    "cb_level",
    "cb_level_mi",
    "cb_level_noqs",
    "cb_level_notrend",
    "cb_level_noramp",
]
TERM_VARIANTS: dict[str, tuple[str, ...]] = {
    "full": LEVEL_TERMS,
    "no_trend": tuple(t for t in LEVEL_TERMS if t != "t"),
    "no_ramp": tuple(t for t in LEVEL_TERMS if t != "qe_ramp"),
}


def _load(name: str) -> dict[str, Any] | None:
    path = config.BACKTEST_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------- tables
def summary_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, month_rows = [], []
    for name in LEVEL_EXPERIMENTS:
        rep = _load(name)
        if rep is None or "by_fold" not in rep:
            print(f"  (skipping {name}: no phase-2a JSON)")
            continue
        for fold, st in rep["by_fold"].items():
            row: dict[str, Any] = {"experiment": name, "fold": fold, "n_runs": st["n_runs"]}
            for metric in ["mae_clean", "mae_all", "mape_clean", "rmse_all", "median_bias_clean"]:
                row[f"{metric}_mean"] = st[metric]["mean"]
                row[f"{metric}_sd"] = st[metric]["sd"]
            for i, (month, mb) in enumerate(sorted(st["month_bias"].items()), start=1):
                row[f"month_m{i}"] = month
                row[f"bias_m{i}_mean"] = mb["mean"]
                row[f"bias_m{i}_sd"] = mb["sd"]
            for extra in ["mae_clean_cold", "mae_clean_warm"]:
                if extra in st:
                    row[f"{extra}_mean"] = st[extra]["mean"]
                    row[f"{extra}_sd"] = st[extra]["sd"]
            rows.append(row)
        for key, res in rep["folds"].items():
            for month, mm in sorted(res.get("month", {}).items()):
                month_rows.append(
                    {
                        "experiment": name,
                        "run": key,
                        "fold": res["fold"],
                        "cb_seed": res.get("cb_seed"),
                        "cold_seed": res.get("cold_seed"),
                        "month": month,
                        "median_bias_clean": mm["median_bias"],
                        "mae_clean": mm["mae"],
                        "n_clean": mm["n"],
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(month_rows)


def _pm(mean: float, sd: float, digits: int = 1) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def print_main_table(summary: pd.DataFrame) -> None:
    print("\nMAE clean (mean ± sd over seeds), F3 MAE all, F3 MAPE clean, selection = mean(F1, F2) MAE clean")
    header = ["experiment", "seeds", "F1", "F2", "F3", "F3 all", "F3 MAPE", "sel F1+F2"]
    print(" | ".join(header))
    for name in TABLE_EXPERIMENTS:
        sub = summary[summary["experiment"] == name].set_index("fold")
        if sub.empty:
            continue
        cells = [name, str(int(sub["n_runs"].max()))]
        for fold in ["F1", "F2", "F3"]:
            cells.append(_pm(sub.loc[fold, "mae_clean_mean"], sub.loc[fold, "mae_clean_sd"]) if fold in sub.index else "-")
        if "F3" in sub.index:
            cells.append(_pm(sub.loc["F3", "mae_all_mean"], sub.loc["F3", "mae_all_sd"]))
            cells.append(_pm(sub.loc["F3", "mape_clean_mean"], sub.loc["F3", "mape_clean_sd"], 2) + "%")
        sel = sub.loc[[f for f in ["F1", "F2"] if f in sub.index], "mae_clean_mean"].mean()
        cells.append(f"{sel:.1f}")
        print(" | ".join(cells))


def print_month_bias(summary: pd.DataFrame, names: list[str]) -> None:
    print("\nMedian bias by test month, clean rows (mean over seeds, %)")
    for name in names:
        sub = summary[summary["experiment"] == name].sort_values("fold")
        parts = []
        for _, r in sub.iterrows():
            parts.append(
                f"{r['fold']}: {r['month_m1'][-2:]} {100 * r['bias_m1_mean']:+.2f} / {r['month_m2'][-2:]} {100 * r['bias_m2_mean']:+.2f}"
            )
        print(f"  {name:18s} " + "   ".join(parts))


# --------------------------------------------------------------------------- level fits
def _clean_training(frame: pd.DataFrame, mask: pd.Series) -> tuple[pd.DataFrame, RpmReference]:
    train = frame[mask]
    reference = RpmReference().fit(train)
    return train[~flag_corrupted(reference, train)], reference


def level_coefficients(data) -> dict[str, Any]:  # noqa: ANN001 - BacktestData
    frame, market = data.frame, data.market_table
    out: dict[str, Any] = {"terms": list(LEVEL_TERMS), "t_origin": config.DATE_ORIGIN, "folds": {}, "variants": {}}
    windows: dict[str, tuple[pd.Series, pd.Series | None]] = {
        f.name: (f.train_mask(frame), f.test_mask(frame)) for f in get_folds()
    }
    windows["ALL"] = (pd.Series(True, index=frame.index), None)

    for variant, terms in TERM_VARIANTS.items():
        for name, (train_mask, test_mask) in windows.items():
            fit_rows, reference = _clean_training(frame, train_mask)
            residual = log_rpm(fit_rows) - np.log(reference.reference_rpm(fit_rows).to_numpy())
            level = MarketLevel(market, terms).fit(fit_rows, residual)
            entry: dict[str, Any] = level.coefficients()
            if test_mask is not None:
                # Out-of-sample daily level error: L predicted for test days vs the realised
                # daily index of clean test rows, both against the fold's training reference.
                test = frame[test_mask]
                test = test[~flag_corrupted(reference, test)]
                idx = daily_mean(test["date"], log_rpm(test) - np.log(reference.reference_rpm(test).to_numpy()))
                err = pd.Series(level.predict_dates(idx.index) - idx["value"].to_numpy(), index=idx.index)
                entry["oos_level_error_pred_minus_actual"] = {
                    "mean": float(err.mean()),
                    "sd": float(err.std(ddof=1)),
                    "by_month": {str(k): float(v) for k, v in err.groupby(err.index.to_period("M")).mean().items()},
                }
            if variant == "full":
                out["folds"][name] = entry
            else:
                out["variants"].setdefault(variant, {})[name] = entry
    out["backfit"] = backfit_comparison()
    return out


def backfit_comparison() -> dict[str, Any]:
    """K=2 (cb_level_k2, seed 0) vs K=1 (cb_level, seed 0): coefficients and metrics."""
    k1, k2 = _load("cb_level"), _load("cb_level_k2")
    if k1 is None or k2 is None:
        return {}
    out: dict[str, Any] = {}
    for fold in ["F1", "F2", "F3"]:
        a, b = k1["folds"].get(f"{fold}_r0"), k2["folds"].get(f"{fold}_r0")
        if a is None or b is None:
            continue
        out[fold] = {
            "k1_level": a["level"][-1],
            "k2_levels": b["level"],
            "k1": {"mae_clean": a["metrics"]["clean"]["mae"], "mae_all": a["metrics"]["all"]["mae"], "month": a["month"]},
            "k2": {"mae_clean": b["metrics"]["clean"]["mae"], "mae_all": b["metrics"]["all"]["mae"], "month": b["month"]},
        }
    return out


def print_coefficients(coefs: dict[str, Any]) -> None:
    print("\nL coefficients (weighted daily OLS; t in days since 2025-01-01)")
    print("  window  const    b_mi    b_t(/day)  b_t(%/30d)  b_qe    R2     oos err mean±sd (by month)")
    for name, e in coefs["folds"].items():
        oos = e.get("oos_level_error_pred_minus_actual")
        oos_s = ""
        if oos:
            by = " ".join(f"{k[-2:]}:{100 * v:+.2f}" for k, v in oos["by_month"].items())
            oos_s = f"{100 * oos['mean']:+.2f}±{100 * oos['sd']:.2f}% ({by})"
        print(
            f"  {name:6s} {e['const']:+.4f} {e['mi_day_mean']:.4f} {e['t']:.6f}   {100 * 30 * e['t']:.2f}%      "
            f"{e['qe_ramp']:.4f} {e['r2_daily_weighted']:.3f}  {oos_s}"
        )
    for fold, cmp in coefs.get("backfit", {}).items():
        l1, l2 = cmp["k1_level"], cmp["k2_levels"][-1]
        print(
            f"  backfit {fold}: K=1 b_mi {l1['mi_day_mean']:.4f} b_t {l1['t']:.6f} b_qe {l1['qe_ramp']:.4f}"
            f" | K=2 b_mi {l2['mi_day_mean']:.4f} b_t {l2['t']:.6f} b_qe {l2['qe_ramp']:.4f}"
            f" | MAE clean {cmp['k1']['mae_clean']:.2f} -> {cmp['k2']['mae_clean']:.2f}"
            f" | all {cmp['k1']['mae_all']:.2f} -> {cmp['k2']['mae_all']:.2f}"
        )


# --------------------------------------------------------------------------- plot
def plot_index(data) -> None:  # noqa: ANN001 - BacktestData
    frame, market = data.frame, data.market_table
    fit_rows, reference = _clean_training(frame, pd.Series(True, index=frame.index))
    residual = log_rpm(fit_rows) - np.log(reference.reference_rpm(fit_rows).to_numpy())
    level = MarketLevel(market).fit(fit_rows, residual)
    idx = level.daily_
    all_days = pd.date_range(config.DATE_ORIGIN, "2025-12-31", freq="D")
    L = pd.Series(level.predict_dates(all_days), index=all_days)
    future = L.index >= pd.Timestamp("2025-11-01")

    plt.style.use(config.MPL_STYLE_PATH)
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.axvspan(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"), color="#F2F2F2", lw=0, zorder=0)
    ax.plot(idx.index, 100 * idx["value"], color="#616161", lw=1.0, label="daily index (clean labels)")
    ax.plot(L.index[~future], 100 * L[~future], color="#1565C0", lw=2.0, label="fitted L(date)")
    ax.plot(L.index[future], 100 * L[future], color="#1565C0", lw=2.0, ls="--", label="L extended (validation inputs)")
    ax.text(pd.Timestamp("2025-11-03"), ax.get_ylim()[1], "Nov-Dec: no labels", va="top", ha="left", fontsize=8, color="#616161")
    c = level.coef_
    ax.set_title(
        f"Daily price level vs L = {c['const']:+.3f} + {c['mi_day_mean']:.3f}·mi + {1e4 * c['t']:.2f}e-4·t + {c['qe_ramp']:.3f}·qe_ramp"
        f"   (R² {level.r2_:.2f})",
        loc="left",
    )
    ax.set_ylabel("log(rpm / lane×equip. median) ×100")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=3)
    ax.margins(x=0.01)
    config.BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.BACKTEST_DIR / "level_index.png")
    plt.close(fig)


def main() -> None:
    summary, month_bias = summary_tables()
    config.BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.BACKTEST_DIR / "level_summary.csv", index=False)
    month_bias.to_csv(config.BACKTEST_DIR / "level_month_bias.csv", index=False)
    print_main_table(summary)
    print_month_bias(summary, ["yardstick", "yardstick_level", "cb_time", "cb_level", "cb_level_mi"])
    cold = summary[summary["experiment"] == "cb_level_cold"]
    for _, r in cold.iterrows():
        print(
            f"\ncb_level_cold {r['fold']} ({int(r['n_runs'])} city draws): cold {_pm(r['mae_clean_cold_mean'], r['mae_clean_cold_sd'])}"
            f"  warm {_pm(r['mae_clean_warm_mean'], r['mae_clean_warm_sd'])}  overall {_pm(r['mae_clean_mean'], r['mae_clean_sd'])}"
        )

    data = load_backtest_data()
    coefs = level_coefficients(data)
    (config.BACKTEST_DIR / "level_coefficients.json").write_text(json.dumps(coefs, indent=2))
    print_coefficients(coefs)
    plot_index(data)
    print(f"\nwrote level_summary.csv, level_month_bias.csv, level_coefficients.json, level_index.png to {config.BACKTEST_DIR}")


if __name__ == "__main__":
    main()
