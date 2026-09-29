"""Sanity checks of the final predictions (no CatBoost training if the models are saved).

Writes ``artifacts/final/sanity.md``, ``level_continuity.png`` and ``december_components.png``:

1. validation predictions: finite / positive; median rpm by equipment and distance band vs
   the Sep-Oct clean actuals;
2. level continuity: daily mean log(rpm / lane x equipment reference) -- actual labels
   Jan-Oct, predictions Nov-Dec -- with the fitted level L;
3. the 8 cities unseen in training: row counts, rpm vs warm rows, per-city relative rpm;
4. agreement with yardstick_level (reference x exp(L)) fitted on all Jan-Oct;
5. edge cases: capped / flipped / missing weight, missing market_index;
6. the December Lexington -> Fort Wayne Dry Van validation loads vs the chart curve;
7. December chart summary: range, weekly wave, late-December ramp, Option A vs neutral.
    .venv/bin/python scripts/sanity.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freight import config
from freight.features import distance_band, lane_key
from freight.final import (
    DECEMBER_OUT,
    FINAL_DIR,
    VALIDATION_OUT,
    december_components,
    december_rows,
    load_or_fit,
    validation_rows,
)
from freight.models import LaneLevelBaseline
from freight.outliers import flag_corrupted

UNSEEN_CITIES = ("Allentown", "Charlotte", "Chicago", "Jackson", "Knoxville", "Laredo", "Norfolk", "San Diego")
BLUE, GREEN, RED, AMBER, GRAY = "#1565C0", "#2E7D32", "#C62828", "#F9A825", "#616161"


# --------------------------------------------------------------------------- helpers
def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Minimal GitHub markdown table (no tabulate dependency)."""

    def cell(v: object) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:+.{digits}f}%"


class Report:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def h(self, text: str) -> None:
        self.parts.append(f"\n## {text}\n")

    def p(self, text: str) -> None:
        self.parts.append(text + "\n")

    def t(self, df: pd.DataFrame, floatfmt: str = ".3f") -> None:
        self.parts.append(md_table(df, floatfmt) + "\n")

    def text(self) -> str:
        return "\n".join(self.parts)


# --------------------------------------------------------------------------- checks
def check_predictions(rep: Report, val: pd.DataFrame, sepoct: pd.DataFrame) -> None:
    rep.h("1. Validation predictions")
    written = pd.read_csv(VALIDATION_OUT).set_index("load_id")["predicted_rate"]
    diff = float(np.abs(written.reindex(val["load_id"]).to_numpy() - val["pred"].to_numpy()).max())
    n_nan = int(val["pred"].isna().sum())
    rep.p(
        f"- {len(val):,} rows; NaN {n_nan}; non-positive {int((val['pred'] <= 0).sum())}; min ${val['pred'].min():.2f}, "
        f"median ${val['pred'].median():.2f}, max ${val['pred'].max():.2f}; max |CSV − model| = {diff:.4f} (rounding)."
    )
    rep.p(
        "- Median rpm: validation PREDICTIONS (Nov–Dec) vs Sep–Oct CLEAN actuals. The ratio mixes the level change "
        "Oct → Nov/Dec with small composition differences."
    )
    for by in ("equipment", "band"):
        a = val.groupby(by)["pred_rpm"].agg(n_val="size", pred_rpm_median="median")
        b = sepoct.groupby(by)["rpm"].agg(n_sepoct="size", actual_rpm_median="median")
        tab = a.join(b).reset_index()
        tab["ratio"] = tab["pred_rpm_median"] / tab["actual_rpm_median"]
        if by == "band":
            tab = tab.sort_values(by, key=lambda s: s.str.extract(r"^(\d+)")[0].astype(int))
        rep.t(tab)
    rel_val = float(np.median(val["pred_rpm"] / val["ref_rpm"]))
    rel_hist = float(np.median(sepoct["rpm"] / sepoct["ref_rpm"]))
    rep.p(
        f"- Composition-free: median rpm / lane reference = {rel_val:.4f} (Nov–Dec predicted) vs {rel_hist:.4f} "
        f"(Sep–Oct actual), i.e. {pct(rel_val / rel_hist - 1)}."
    )
    print(f"[1] NaN {n_nan}, min {val['pred'].min():.2f}; rel-to-reference Nov-Dec {rel_val:.4f} vs Sep-Oct {rel_hist:.4f}")


def daily_index(dates: pd.Series, log_ratio: np.ndarray) -> pd.Series:
    frame = pd.DataFrame({"date": pd.to_datetime(dates).dt.normalize().to_numpy(), "v": log_ratio})
    return frame.groupby("date")["v"].mean()


def check_level(rep: Report, model, clean_train: pd.DataFrame, val: pd.DataFrame) -> None:  # noqa: ANN001
    rep.h("2. Level continuity")
    actual = daily_index(clean_train["date"], np.log(clean_train["rpm"] / clean_train["ref_rpm"]).to_numpy())
    pred = daily_index(val["date"], np.log(val["pred_rpm"] / val["ref_rpm"]).to_numpy())
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    L = pd.Series(model.level.predict_dates(days), index=days)
    idx = pd.concat([actual, pred])

    def month_mean(s: pd.Series, m: str) -> float:
        return float(s[s.index.to_period("M").astype(str) == m].mean())

    rows = []
    for m in ["2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]:
        src = "predicted" if m >= "2025-11" else "actual"
        rows.append({"month": m, "index source": src, "mean index": month_mean(idx, m), "mean L": month_mean(L, m)})
    tab = pd.DataFrame(rows)
    rep.p("Daily mean of log(rpm / RpmReference lane × equipment rpm): clean actual labels Jan–Oct, predictions Nov–Dec.")
    rep.t(tab, ".4f")
    oct_, nov, dec = (month_mean(idx, m) for m in ("2025-10", "2025-11", "2025-12"))
    rep.p(
        f"- Nov predicted level {nov:+.4f} vs Oct actual {oct_:+.4f} (Δ {nov - oct_:+.4f} log ≈ {pct(np.exp(nov - oct_) - 1)}); "
        f"Dec {dec:+.4f} (Δ vs Oct {pct(np.exp(dec - oct_) - 1)})."
    )
    last_oct, first_nov = actual.loc["2025-10-25":"2025-10-31"].mean(), pred.loc["2025-11-01":"2025-11-07"].mean()
    rep.p(f"- Seam: last Oct week actual {last_oct:+.4f} vs first Nov week predicted {first_nov:+.4f} (Δ {first_nov - last_oct:+.4f}).")

    # weekly wave: day-of-week profile of the index net of its monthly mean
    def dow_profile(s: pd.Series) -> pd.Series:
        dev = s - s.groupby(s.index.to_period("M")).transform("mean")
        return dev.groupby(dev.index.dayofweek).mean()

    insample = daily_index(clean_train["date"], np.log(model.predict_rows(clean_train) / clean_train["distance"] / clean_train["ref_rpm"]).to_numpy())
    L_jo = L.loc[actual.index.min():actual.index.max()]
    prof = pd.DataFrame(
        {
            "Jan–Oct actual": dow_profile(actual),
            "Jan–Oct in-sample pred": dow_profile(insample),
            "Jan–Oct L": dow_profile(L_jo),
            "Nov–Dec predicted": dow_profile(pred),
        }
    )
    prof.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    rep.p("Weekly wave: day-of-week mean of the index net of its monthly mean (log units):")
    rep.t(prof.reset_index(names="dow"), "+.4f")
    amp = prof.apply(np.ptp)
    mi = model.market_table["mi_day_mean"]
    mi_dev = mi - mi.groupby(mi.index.to_period("M")).transform("mean")
    mi_amp = mi_dev.groupby([mi.index >= "2025-11-01", mi.index.dayofweek]).mean().groupby(level=0).agg(np.ptp)
    amp_a, amp_p = float(amp.iloc[0]), float(amp.iloc[3])
    rep.p(
        f"- Weekly amplitude (max − min): actual {amp_a:.4f}, in-sample prediction {amp.iloc[1]:.4f}, L alone "
        f"{amp.iloc[2]:.4f}, Nov–Dec prediction {amp_p:.4f}. The market-index wave itself is unchanged "
        f"(mi_day_mean amplitude {mi_amp.loc[False]:.4f} Jan–Oct vs {mi_amp.loc[True]:.4f} Nov–Dec), so the Nov–Dec wave is "
        f"L's: its single mi_day_mean slope overstates the within-week swing by ≈ {amp.iloc[2] - amp_a:.4f} log, i.e. about "
        f"±{50 * (amp.iloc[2] - amp_a):.2f}% at the weekly peak and trough. In-sample the trees absorb this through dow × dom; "
        f"out of sample they do not."
    )

    # quarter-end ramp: days 25+ minus days 1-15, per quarter-end month
    ramp_rows = []
    for m, src in [("2025-03", "actual"), ("2025-06", "actual"), ("2025-09", "actual"), ("2025-12", "predicted")]:
        s = idx[idx.index.to_period("M").astype(str) == m]
        ramp_rows.append({"month": m, "source": src, "days 25+ minus days 1–15": float(s[s.index.day >= 25].mean() - s[s.index.day <= 15].mean())})
    for m in ("2025-10", "2025-11"):
        s = idx[idx.index.to_period("M").astype(str) == m]
        ramp_rows.append({"month": m, "source": "actual" if m < "2025-11" else "predicted", "days 25+ minus days 1–15": float(s[s.index.day >= 25].mean() - s[s.index.day <= 15].mean())})
    rep.p("Late-month lift (quarter-end months should show the ramp; Oct/Nov are controls):")
    rep.t(pd.DataFrame(ramp_rows), "+.4f")
    print(f"[2] level Oct actual {oct_:+.4f}  Nov pred {nov:+.4f}  Dec pred {dec:+.4f}; weekly amp {amp_a:.4f} vs {amp_p:.4f}")

    # plot
    plt.style.use(config.MPL_STYLE_PATH)
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.plot(actual.index, 100 * actual.to_numpy(), color=BLUE, lw=1.0, label="actual labels (Jan–Oct, clean rows)")
    ax.plot(pred.index, 100 * pred.to_numpy(), color=RED, lw=1.2, label="predictions (Nov–Dec validation loads)")
    ax.plot(L.index, 100 * L.to_numpy(), color=GRAY, lw=1.0, ls="--", label="market level L(date)")
    ax.axvline(pd.Timestamp("2025-11-01"), color=GRAY, lw=0.8, ls=":")
    ax.text(pd.Timestamp("2025-11-03"), ax.get_ylim()[1], "validation", va="top", fontsize=8, color=GRAY)
    ax.set_title("Daily price level: mean log(rpm / lane × equipment reference)", loc="left")
    ax.set_ylabel("log ratio × 100 (≈ %)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(loc="lower right")
    fig.savefig(FINAL_DIR / "level_continuity.png")
    plt.close(fig)


def check_unseen(rep: Report, val: pd.DataFrame, seen: set[str]) -> None:
    rep.h("3. Cities unseen in training")
    unseen = sorted((set(val["pickup"]) | set(val["delivery"])) - seen)
    rep.p(f"- Unseen cities found: {', '.join(unseen)} ({'matches' if tuple(unseen) == UNSEEN_CITIES else 'DIFFERS FROM'} the expected 8).")
    cold = val["cold"]
    rep.p(f"- Validation rows touching them: {int(cold.sum()):,} of {len(val):,} ({100 * cold.mean():.1f}%).")
    tab = (
        val.groupby(["band", "cold"])["pred_rpm"].median().unstack().rename(columns={False: "warm median rpm", True: "cold median rpm"})
    )
    tab["n cold"] = val[cold].groupby("band").size()
    tab["cold / warm"] = tab["cold median rpm"] / tab["warm median rpm"]
    tab = tab.reset_index().sort_values("band", key=lambda s: s.str.extract(r"^(\d+)")[0].astype(int))
    rep.t(tab[["band", "n cold", "warm median rpm", "cold median rpm", "cold / warm"]])

    # per-city relative rpm: row rpm / median predicted rpm of warm rows of the same equipment x band
    warm_med = val[~cold].groupby(["equipment", "band"])["pred_rpm"].median()
    val = val.assign(rel=val["pred_rpm"].to_numpy() / warm_med.reindex(pd.MultiIndex.from_frame(val[["equipment", "band"]])).to_numpy())
    per_city = []
    for city in sorted(set(val["pickup"]) | set(val["delivery"])):
        rows = val[(val["pickup"] == city) | (val["delivery"] == city)]
        out_rows, in_rows = val[val["pickup"] == city], val[val["delivery"] == city]
        per_city.append(
            {
                "city": city,
                "cold": city in unseen,
                "n": len(rows),
                "rel rpm": float(rows["rel"].median()),
                "rel rpm outbound": float(out_rows["rel"].median()) if len(out_rows) else np.nan,
                "rel rpm inbound": float(in_rows["rel"].median()) if len(in_rows) else np.nan,
            }
        )
    pc = pd.DataFrame(per_city)
    warm_pc = pc[~pc["cold"]]
    lo, hi = warm_pc["rel rpm"].min(), warm_pc["rel rpm"].max()
    q05, q95 = warm_pc["rel rpm"].quantile([0.05, 0.95])
    cold_pc = pc[pc["cold"]].copy()
    cold_pc["inside warm range"] = cold_pc["rel rpm"].between(lo, hi)
    rep.p(
        f"Per-city median of rpm / (warm median rpm of the same equipment × band). Warm cities span {lo:.3f}–{hi:.3f} "
        f"(5–95%: {q05:.3f}–{q95:.3f}); the 8 cold cities:"
    )
    rep.t(cold_pc.drop(columns="cold"))
    flagged = cold_pc.loc[~cold_pc["inside warm range"], "city"].tolist()
    rep.p(f"- Cold cities outside the warm-city range: {flagged or 'none'}.")
    print(f"[3] {int(cold.sum())} cold rows; cold rel rpm {cold_pc['rel rpm'].min():.3f}..{cold_pc['rel rpm'].max():.3f}; warm {lo:.3f}..{hi:.3f}; flagged {flagged}")


def check_yardstick(rep: Report, model, clean_train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:  # noqa: ANN001
    rep.h("4. Agreement with yardstick_level (lane × equipment reference × exp(L), all Jan–Oct)")
    yard = LaneLevelBaseline(model.market_table).fit(clean_train, model.reference)
    val = val.assign(yard=yard.predict(val))
    val["ratio"] = val["pred"] / val["yard"]
    corr = float(np.corrcoef(val["pred"], val["yard"])[0, 1])
    corr_log = float(np.corrcoef(np.log(val["pred_rpm"]), np.log(val["yard"] / val["distance"]))[0, 1])
    n10 = int((np.abs(val["ratio"] - 1) > 0.10).sum())
    n5 = int((np.abs(val["ratio"] - 1) > 0.05).sum())
    rep.p(
        f"- Pearson r (rates) {corr:.4f}; r (log rpm) {corr_log:.4f}; median ratio model / yardstick {val['ratio'].median():.4f}; "
        f"5–95% of ratio {val['ratio'].quantile(0.05):.3f}–{val['ratio'].quantile(0.95):.3f}."
    )
    rep.p(f"- |ratio − 1| > 10%: {n10} rows ({100 * n10 / len(val):.2f}%); > 5%: {n5} rows ({100 * n5 / len(val):.1f}%).")
    by = val.assign(big=np.abs(val["ratio"] - 1) > 0.10).groupby(["ref_level", "cold"]).agg(
        n=("big", "size"), n_gt10=("big", "sum"), median_ratio=("ratio", "median")
    )
    rep.p("By reference level (which history the yardstick used) and cold status:")
    rep.t(by.reset_index())
    top = val.assign(abs_log=np.abs(np.log(val["ratio"]))).nlargest(5, "abs_log")
    cols = ["load_id", "date", "lane", "equipment", "distance", "weight", "weight_at_cap", "weight_missing", "mi_dev", "cold", "ref_level", "pred", "yard", "ratio"]
    top = top.assign(date=top["date"].dt.date.astype(str))[cols]
    rep.p("Five largest disagreements:")
    rep.t(top, ".3f")
    print(f"[4] r {corr:.4f}  median ratio {val['ratio'].median():.4f}  >10%: {n10}")
    return val


def check_edges(rep: Report, val: pd.DataFrame, clean_train: pd.DataFrame, yard_train: np.ndarray) -> None:
    rep.h("5. Edge cases")
    hist = clean_train.assign(ratio=clean_train["posted_rate"].to_numpy() / yard_train)
    groups = {
        "all rows": (np.ones(len(val), bool), np.ones(len(hist), bool)),
        "weight at cap (47,500)": (val["weight_at_cap"] == 1, hist["weight_at_cap"] == 1),
        "weight at floor (5,000)": (val["weight_at_floor"] == 1, hist["weight_at_floor"] == 1),
        "weight negative (flipped)": (val["weight_was_negative"] == 1, hist["weight_was_negative"] == 1),
        "weight missing": (val["weight_missing"] == 1, hist["weight_missing"] == 1),
        "market_index missing": (val["mi_missing"] == 1, hist["mi_missing"] == 1),
    }
    rows = []
    for name, (vm, hm) in groups.items():
        vm, hm = np.asarray(vm), np.asarray(hm)
        rows.append(
            {
                "group": name,
                "n val": int(vm.sum()),
                "pred rpm / ref rpm": float(np.median(val.loc[vm, "pred_rpm"] / val.loc[vm, "ref_rpm"])),
                "pred / yardstick": float(np.median(val.loc[vm, "ratio"])),
                "n Jan–Oct clean": int(hm.sum()),
                "actual / yardstick (in-sample)": float(np.median(hist.loc[hm, "ratio"])),
            }
        )
    tab = pd.DataFrame(rows)
    rep.p(
        "Median ratios per group. 'pred / yardstick' removes lane and level, leaving the trees' per-load adjustment; "
        "the last column is the same quantity for actual Jan–Oct labels."
    )
    rep.t(tab, ".4f")
    print("[5]", {r["group"]: round(r["pred / yardstick"], 4) for r in rows})


def check_december_lane(rep: Report, model, val: pd.DataFrame, comps: pd.DataFrame) -> None:  # noqa: ANN001
    rep.h("6. December cross-check: Lexington → Fort Wayne Dry Van validation loads vs the chart")
    lane = val[(val["pickup"] == "Lexington") & (val["delivery"] == "Fort Wayne") & (val["equipment"] == "Dry Van")]
    lane = lane[lane["date"].dt.month == 12].copy()
    chart = comps.set_index("date")["pred_optionA"]
    lane["chart"] = lane["date"].dt.date.astype(str).map(chart).to_numpy()
    # counterfactual: the same load with the chart's inputs (360 mi, 32,000 lb, Option-A market)
    cf = lane.copy()
    cf["distance"], cf["weight"] = 360.0, 32_000.0
    cf["market_index"] = model.market_table["mi_day_mean"].reindex(cf["date"].dt.normalize().to_numpy()).to_numpy()
    for flag in ("weight_missing", "weight_was_negative", "weight_at_cap", "weight_at_floor", "mi_missing"):
        cf[flag] = 0
    lane["counterfactual"] = model.predict_rows(cf)
    lane["pred per 360 mi"] = lane["pred"] * 360.0 / lane["distance"]
    lane["ratio per-mile vs chart"] = lane["pred per 360 mi"] / lane["chart"]
    out = lane.assign(date=lane["date"].dt.date.astype(str))[
        ["load_id", "date", "distance", "weight", "mi_dev", "pred", "pred per 360 mi", "chart", "ratio per-mile vs chart", "counterfactual"]
    ]
    rep.t(out, ".3f")
    gap = float(np.abs(lane["counterfactual"] - lane["chart"]).max())
    rep.p(
        f"- Setting each load's inputs to the chart's (360 mi, 32,000 lb, market_index := day mean) reproduces the chart "
        f"value to within ${gap:.4f}: the validation and chart paths are the same model. The remaining per-mile gaps are "
        f"the loads' own weight and market_index deviation."
    )
    print(f"[6] {len(lane)} loads; per-mile ratios {np.round(lane['ratio per-mile vs chart'].to_numpy(), 3)}; counterfactual gap {gap:.4f}")


def check_december_chart(rep: Report, model, comps: pd.DataFrame, clean_train: pd.DataFrame) -> None:  # noqa: ANN001
    rep.h("7. December chart (Lexington → Fort Wayne, Dry Van, 360 mi, 32,000 lb)")
    yard_chart = LaneLevelBaseline(model.market_table).fit(clean_train, model.reference).predict(december_rows(model))
    written = pd.read_csv(DECEMBER_OUT)
    s = written["predicted_rate"]
    a, n = comps["pred_optionA"], comps["pred_neutral"]
    dates = pd.to_datetime(comps["date"])
    rep.p(f"- Option A (deliverable): min ${s.min():.2f}, max ${s.max():.2f}, mean ${s.mean():.2f}, range ${s.max() - s.min():.2f}.")
    rep.p(f"- Neutral market (Dec-mean index): min ${n.min():.2f}, max ${n.max():.2f}, mean ${n.mean():.2f}.")
    diff = a - n
    rep.p(
        f"- Option A − neutral: mean ${diff.mean():+.2f}, min ${diff.min():+.2f}, max ${diff.max():+.2f} "
        f"(the market-index path; |max| {pct(float(np.abs(a / n - 1).max()))})."
    )
    wave = (a / n - 1).groupby(dates.dt.dayofweek).mean()
    wave.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    rep.p(
        "- Weekly wave (Option A / neutral − 1 by weekday, i.e. the market-index effect): "
        + ", ".join(f"{k} {pct(v)}" for k, v in wave.items())
        + f"; amplitude {pct(float(np.ptp(wave)))} ≈ ${float(np.ptp(wave)) * a.mean():.0f}."
    )
    tree_dollars = 360.0 * np.exp(comps["L_neutral"].mean() + comps["tree_f"])
    tree_dow = tree_dollars.groupby(dates.dt.dayofweek).mean()
    rep.p(
        f"- Tree part at constant L (dow / dom effects of f): range ${np.ptp(tree_dollars):.2f} over the month; "
        f"weekday means ${tree_dow.min():.2f}–${tree_dow.max():.2f}. Seed sd of f ≤ {comps['tree_f_seed_sd'].max():.4f} log."
    )
    b_qe, b_t = model.level.coef_["qe_ramp"], model.level.coef_["t"]
    early, late = n[dates.dt.day <= 15].mean(), n[dates.dt.day >= 25].mean()
    rep.p(
        f"- Late-December ramp: qe_ramp coefficient {b_qe:.4f} → {pct(np.exp(b_qe) - 1)} of level on Dec 31; neutral mean "
        f"Dec 1–15 ${early:.2f} vs Dec 25–31 ${late:.2f} ({pct(late / early - 1)}, includes the trend "
        f"{pct(np.exp(30 * b_t) - 1)} per 30 days)."
    )
    lane = clean_train[(clean_train["pickup"] == "Lexington") & (clean_train["delivery"] == "Fort Wayne") & (clean_train["equipment"] == "Dry Van")]
    recent = lane[lane["date"] >= "2025-09-01"]
    per360 = (recent["posted_rate"] / recent["distance"] * 360).median()
    rep.p(
        f"- History: {len(lane)} clean Lexington → Fort Wayne Dry Van loads in Jan–Oct; Sep–Oct n = {len(recent)}, median rate "
        f"per 360 mi ${per360:.2f}; chart Dec mean is {pct(s.mean() / per360 - 1)} above it."
    )
    rows = december_rows(model)
    by_w = {}
    for w in (float(lane["weight"].median()), float(recent["weight"].median()), 32_000.0):
        by_w[w] = float(model.predict_rows(rows.assign(weight=w)).mean())
    rep.p(
        "- The gap is mostly weight: the lane's loads are lighter than the chart's 32,000 lb (median "
        f"{lane['weight'].median():,.0f} lb Jan–Oct, {recent['weight'].median():,.0f} lb Sep–Oct). Chart mean at those weights: "
        + ", ".join(f"{w:,.0f} lb ${v:.2f}" for w, v in by_w.items())
        + f". yardstick_level (no weight) on the chart rows: mean ${float(yard_chart.mean()):.2f}."
    )
    # f's day-of-month pattern vs L's in-sample daily residuals in each month
    resid = daily_index(clean_train["date"], np.log(clean_train["rpm"] / clean_train["ref_rpm"]).to_numpy())
    resid = resid - model.level.predict_dates(resid.index)
    tf = comps["tree_f"].to_numpy() - comps["tree_f"].mean()
    corrs = {}
    for mon in ("2025-03", "2025-06", "2025-09", "2025-10"):
        r = resid.reindex(pd.date_range(f"{mon}-01", periods=31, freq="D")).to_numpy()
        ok = ~np.isnan(r)
        corrs[mon] = float(np.corrcoef(tf[ok], r[ok])[0, 1])
    rep.p(
        "- Tree part across December: days 1–15 "
        f"{tf[:15].mean():+.4f}, days 25–31 {tf[24:].mean():+.4f} (log). Its day-of-month pattern correlates with L's in-sample "
        "daily residuals of the quarter-end months ("
        + ", ".join(f"{k[-2:]}: {v:+.2f}" for k, v in corrs.items())
        + "; Oct is a control): f partly corrects the shape of the linear quarter-end ramp through dom, trimming its last days."
    )
    rep.p("Daily values (Option A, neutral):")
    rep.t(comps.assign(date=comps["date"].str[5:])[["date", "dow", "pred_optionA", "pred_neutral", "L", "mi_day_mean", "qe_ramp", "tree_f"]], ".4f")
    print(f"[7] Dec chart min {s.min():.2f} max {s.max():.2f} mean {s.mean():.2f}; A-neutral max |{np.abs(diff).max():.2f}|; history per-360 {per360:.2f}")

    plt.style.use(config.MPL_STYLE_PATH)
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.plot(dates, a, color=BLUE, lw=2.0, marker="o", ms=3, label="Option A (daily market index) — deliverable")
    ax.plot(dates, n, color=GRAY, lw=1.6, ls="--", label="neutral market (December-mean index)")
    ax.axvspan(pd.Timestamp("2025-12-15"), pd.Timestamp("2025-12-31"), color="#F2F2F2", lw=0, zorder=0)
    ax.text(pd.Timestamp("2025-12-16"), ax.get_ylim()[0], "quarter-end ramp", va="bottom", fontsize=8, color=GRAY)
    ax.set_title("Lexington → Fort Wayne, Dry Van, 360 mi, 32,000 lb (final model, 5 seeds)", loc="left")
    ax.set_ylabel("predicted rate (USD)")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%a"))
    ax.legend(loc="upper left")
    fig.savefig(FINAL_DIR / "december_components.png")
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main() -> None:
    model = load_or_fit()
    frame, ref = model.data.frame, model.reference
    clean_train = frame[~flag_corrupted(ref, frame)].copy()
    clean_train["rpm"] = clean_train["posted_rate"] / clean_train["distance"]
    clean_train["ref_rpm"] = ref.reference_rpm(clean_train).to_numpy()
    clean_train["band"] = distance_band(clean_train["distance"])
    sepoct = clean_train[clean_train["date"] >= "2025-09-01"]

    val = validation_rows()
    val["pred"] = model.predict_rows(val)
    val["pred_rpm"] = val["pred"] / val["distance"]
    val["ref_rpm"] = ref.reference_rpm(val).to_numpy()
    val["ref_level"] = ref.reference_level(val).to_numpy()
    val["band"] = distance_band(val["distance"])
    val["lane"] = lane_key(val)
    seen = set(frame["pickup"]) | set(frame["delivery"])
    val["cold"] = ~(val["pickup"].isin(seen) & val["delivery"].isin(seen))
    mi_day = model.market_table["mi_day_mean"].reindex(val["date"].dt.normalize().to_numpy()).to_numpy()
    val["mi_dev"] = val["market_index"].fillna(pd.Series(mi_day, index=val.index)) - mi_day

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    rep = Report()
    rep.p("# Sanity checks of the final predictions\n")
    card = model.card
    rep.p(
        f"Model card: {card['created']}; seeds {card['catboost']['seeds']}; {card['training']['n_rows_used']:,} training rows "
        f"({card['training']['n_excluded_corrupted']} corrupted excluded). Generated by `scripts/sanity.py`."
    )
    check_predictions(rep, val, sepoct)
    check_level(rep, model, clean_train, val)
    check_unseen(rep, val, seen)
    val = check_yardstick(rep, model, clean_train, val)
    yard_train = LaneLevelBaseline(model.market_table).fit(clean_train, ref).predict(clean_train)
    check_edges(rep, val, clean_train, yard_train)
    comps = december_components(model)
    check_december_lane(rep, model, val, comps)
    check_december_chart(rep, model, comps, clean_train)
    (FINAL_DIR / "sanity.md").write_text(rep.text())
    print(f"wrote {FINAL_DIR / 'sanity.md'}, level_continuity.png, december_components.png")


if __name__ == "__main__":
    main()
