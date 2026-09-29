"""December chart-lane PREVIEW (sanity check, not the deliverable).

Fits cb_level (market level L + cross-sectional CatBoost, CatBoost seed 0) on ALL clean
Jan-Oct rows and predicts the 31 December rows of Lexington -> Fort Wayne (360 mi, Dry Van,
32,000 lb) with Option-A market inputs (market_index / quote_signal := that day's market-wide
mean, so mi_dev = qs_dev = 0). A second curve replaces every December mi_day_mean by the
December average ("neutral market") to separate the market-index path from the calendar
terms (trend, quarter-end ramp, day-of-week / day-of-month effects of the trees).

Writes ``artifacts/preview/december_preview.csv`` and ``december_preview.png``. Nothing is
written to ``data/``.
    .venv/bin/python scripts/december_preview.py
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freight import config
from freight.backtest import EXPERIMENTS, _feature_view, catboost_params, load_backtest_data
from freight.data import load_december, load_validation
from freight.features import build_features, city_coordinates, prepare_chart_rows
from freight.level import qe_ramp
from freight.models import LevelCatBoostRate
from freight.outliers import RpmReference, flag_corrupted

SEED = 0


def main() -> None:
    cfg = EXPERIMENTS["cb_level"]
    data = load_backtest_data()
    frame, market = data.frame, data.market_table

    reference = RpmReference().fit(frame)
    fit_rows = frame[~flag_corrupted(reference, frame)]
    X, cats = _feature_view(cfg, data)
    model = LevelCatBoostRate(market, cfg.level_terms, cfg.backfit_rounds, catboost_params(cfg, SEED))
    model.fit(X.loc[fit_rows.index], fit_rows, cats, reference)
    print("L coefficients (all Jan-Oct):", {k: round(v, 6) for k, v in model.level_.coef_.items()})

    coords = city_coordinates(frame, load_validation())
    rows = prepare_chart_rows(load_december(), market, coords)
    Xd, _ = build_features(rows, market, use_time=True)
    Xd = Xd[X.columns]
    L, f = model.predict_components(Xd, rows)
    distance = rows["distance"].to_numpy(dtype=float)
    pred_a = distance * np.exp(L + f)

    # Neutral market: every December day gets the December-average mi_day_mean. With Option A
    # mi_dev stays 0 and mi_day_mean is not a tree feature in cb_level, so only L changes.
    dec = (market.index >= "2025-12-01") & (market.index <= "2025-12-31")
    neutral_market = market.copy()
    neutral_market.loc[dec, "mi_day_mean"] = market.loc[dec, "mi_day_mean"].mean()
    L_neutral = replace(model.level_, market_table=neutral_market).predict(rows)
    pred_n = distance * np.exp(L_neutral + f)

    dates = pd.to_datetime(rows["date"]).dt.normalize()
    out = pd.DataFrame(
        {
            "date": dates.dt.date.astype(str),
            "dow": dates.dt.day_name().str[:3],
            "pred_optionA": pred_a.round(2),
            "pred_neutral": pred_n.round(2),
            "L": L.round(6),
            "L_neutral": L_neutral.round(6),
            "f": f.round(6),
            "mi_day_mean": market["mi_day_mean"].reindex(dates.to_numpy()).to_numpy().round(5),
            "qe_ramp": qe_ramp(dates).round(4),
        }
    )
    config.PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.PREVIEW_DIR / "december_preview.csv", index=False)

    # ---- numbers for the report
    for col in ["pred_optionA", "pred_neutral"]:
        s = out[col]
        print(f"{col}: min {s.min():.0f}  max {s.max():.0f}  mean {s.mean():.0f}")
    tree_dollars = distance * np.exp(L_neutral.mean() + f)
    by_dow = pd.Series(tree_dollars, index=out["dow"]).groupby(level=0).mean().round(0)
    print("tree part (f) at constant L, mean $ by weekday:", by_dow.to_dict(), f"range {np.ptp(tree_dollars):.0f}")
    ramp_pct = 100 * (np.exp(model.level_.coef_["qe_ramp"]) - 1)
    first_half = out.loc[out["qe_ramp"] == 0, "pred_neutral"].mean()
    last_week = out.loc[dates.dt.day >= 25, "pred_neutral"].mean()
    print(f"quarter-end ramp at Dec 31: +{ramp_pct:.2f}% of level; neutral Dec 1-15 mean {first_half:.0f} vs Dec 25-31 mean {last_week:.0f}")
    trend = 100 * (np.exp(30 * model.level_.coef_["t"]) - 1)
    print(f"trend: +{trend:.2f}% per 30 days")
    lane = frame[(frame["pickup"] == "Lexington") & (frame["delivery"] == "Fort Wayne") & (frame["equipment"] == "Dry Van")]
    lane = lane[~flag_corrupted(reference, lane)]
    recent = lane[lane["date"] >= "2025-09-01"]
    print(
        f"history Lexington->Fort Wayne Dry Van: {len(lane)} clean rows; Sep-Oct n={len(recent)} "
        f"rate median {recent['posted_rate'].median():.0f} (per 360 mi: {(recent['posted_rate'] / recent['distance'] * 360).median():.0f})"
    )
    print("\n".join(f"{r.date[5:]} {r.dow}: {r.pred_optionA:.0f}, {r.pred_neutral:.0f}" for r in out.itertuples()))

    # ---- plot
    plt.style.use(config.MPL_STYLE_PATH)
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.plot(dates, pred_a, color="#1565C0", lw=2.0, marker="o", ms=3, label="Option A (daily market index)")
    ax.plot(dates, pred_n, color="#616161", lw=2.0, ls="--", label="neutral market (Dec-mean index)")
    ax.axvspan(pd.Timestamp("2025-12-15"), pd.Timestamp("2025-12-31"), color="#F2F2F2", lw=0, zorder=0)
    ax.text(pd.Timestamp("2025-12-16"), ax.get_ylim()[0], "quarter-end ramp", va="bottom", fontsize=8, color="#616161")
    ax.set_title("PREVIEW: Lexington → Fort Wayne, Dry Van, 360 mi, 32,000 lb (cb_level, seed 0)", loc="left")
    ax.set_ylabel("predicted rate (USD)")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%a"))
    ax.legend(loc="upper left")
    fig.savefig(config.PREVIEW_DIR / "december_preview.png")
    plt.close(fig)
    print(f"\nwrote {config.PREVIEW_DIR / 'december_preview.csv'} and december_preview.png")


if __name__ == "__main__":
    main()
