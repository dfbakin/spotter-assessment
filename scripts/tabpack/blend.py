"""Post-hoc CatBoost + TabPack blends (no training; run with the MAIN venv from the project root).

    .venv/bin/python scripts/tabpack/blend.py            # backtest blends -> artifacts/tabpack/blend*
    .venv/bin/python scripts/tabpack/blend.py --final    # final-fit blend -> artifacts/tabpack/final/

Blend, in log space: ``log p = w * log p_CB + (1 - w) * log p_TP`` where p_CB is the seed mean
(log scale, as the final model averages seeds) of ``cb_level_noqs`` (seeds 0-2,
``artifacts/backtest/preds``) and p_TP the seed mean of ``tp_level`` (``artifacts/tabpack/preds``).

Protocol: w in {0, 0.1, ..., 1} is chosen by the mean clean MAE over F1 + F2 only; the chosen w
is then applied to F3 once. The debiased grid, the 50/50 blend and the blends with the other
TabPack experiments (tp_level_lanetopk, tp_level_lanefull, tp_standalone) are reported for
information only. Seed noise: every (CatBoost seed, TabPack seed) pair is blended
separately; sampling noise: a day-block bootstrap of MAE(blend) - MAE(CatBoost).

Backtest outputs (artifacts/tabpack/):
* ``blend_grid.csv``      blend x fold x w: clean / all MAE, MAPE, debiased MAE, month bias;
* ``blend.json``          chosen w, F1+F2 and F3 results, seed-pair spread, bootstrap CIs, the
                          final-fit gate (blend beats CatBoost on F1+F2 clean MAE by > $0.5);
* ``blend_slices.csv``    slices (equipment, distance band, month, lane frequency) per fold and model;
* ``blend_preds_F3.csv``  per-row F3 predictions of CatBoost, TabPack and the chosen blend.

``--final`` (only meaningful when the gate passes): combines the saved final CatBoost models
(``freight.final.load_final``) with the ``tp_level_FINAL_s*`` TabPack predictions at the chosen w
and writes validation / December-chart CSVs under artifacts/tabpack/final/. The submission files
in the project root are NOT touched.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from freight import config
from freight.backtest import load_backtest_data, month_metrics
from freight.evaluate import all_and_clean, day_debiased_mae, day_debiased_predictions, standard_slices

OUT = config.ARTIFACTS_DIR / "tabpack"
CB_EXP, CB_SEEDS = "cb_level_noqs", (0, 1, 2)
TP_MAIN = "tp_level"
# Information-only TabPack variants: experiment -> column prefix.
TP_EXTRA = {"tp_level_lanetopk": "tplane", "tp_level_lanefull": "tpfull", "tp_standalone": "tpstd"}
FOLDS = ("F1", "F2", "F3")
SELECT_FOLDS = ("F1", "F2")
GRID = [round(0.1 * i, 1) for i in range(11)]
GATE_DOLLARS = 0.5
N_BOOT = 2000
LANE_BANDS = [(0, 0, "unseen"), (1, 4, "1-4"), (5, 19, "5-19"), (20, 49, "20-49"), (50, 10**9, "50+")]


# --------------------------------------------------------------------------- loading
def load_cb(fold: str) -> pd.DataFrame:
    frames = []
    for s in CB_SEEDS:
        p = pd.read_csv(config.PREDS_DIR / f"{CB_EXP}_{fold}_r{s}.csv", parse_dates=["date"])
        frames.append(p.set_index("load_id"))
    base = frames[0][["date", "pickup", "delivery", "equipment", "distance", "actual", "clean", "city_status"]].copy()
    for s, p in zip(CB_SEEDS, frames):
        assert p.index.equals(base.index)
        base[f"cb_s{s}"] = np.log(p["pred"].to_numpy(dtype=float))
    base["cb"] = base[[f"cb_s{s}" for s in CB_SEEDS]].mean(axis=1)
    return base


def tp_seeds(exp: str, fold: str) -> list[int]:
    return sorted(int(p.stem.rsplit("_s", 1)[1]) for p in (OUT / "preds").glob(f"{exp}_{fold}_s*.csv"))


def load_tp(exp: str, fold: str, prefix: str) -> pd.DataFrame | None:
    seeds = tp_seeds(exp, fold)
    if not seeds:
        return None
    out = None
    for s in seeds:
        p = pd.read_csv(OUT / "preds" / f"{exp}_{fold}_s{s}.csv").set_index("load_id")
        col = pd.DataFrame(
            {
                f"{prefix}_s{s}": p["L"] + p["f"] + np.log(p["distance"]),
                f"{prefix}_actual": p["actual"],
                f"{prefix}_clean": p["clean"],
            }
        )
        out = col if out is None else out.join(col[[f"{prefix}_s{s}"]])
    out[prefix] = out[[f"{prefix}_s{s}" for s in seeds]].mean(axis=1)
    out.attrs["seeds"] = seeds
    return out


def fold_frame(fold: str, lane_counts: pd.Series) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    base = load_cb(fold)
    seeds: dict[str, list[int]] = {}
    for exp, prefix in ((TP_MAIN, "tp"), *TP_EXTRA.items()):
        tp = load_tp(exp, fold, prefix)
        if tp is None:
            continue
        assert len(tp) == len(base) and tp.index.isin(base.index).all(), (exp, fold)
        tp = tp.reindex(base.index)
        assert np.allclose(tp[f"{prefix}_actual"], base["actual"]), (exp, fold, "actual mismatch")
        assert (tp[f"{prefix}_clean"].astype(bool) == base["clean"].astype(bool)).all(), (exp, fold, "clean mismatch")
        base = base.join(tp.drop(columns=[f"{prefix}_actual", f"{prefix}_clean"]))
        seeds[exp] = tp.attrs["seeds"]
    lane = base["pickup"].astype(str) + "→" + base["delivery"].astype(str)
    base["lane_train_count"] = lane.map(lane_counts).fillna(0).astype(int).to_numpy()
    band = pd.Series("", index=base.index)
    for lo, hi, label in LANE_BANDS:
        band[(base["lane_train_count"] >= lo) & (base["lane_train_count"] <= hi)] = label
    base["lane_band"] = band
    return base.reset_index(), seeds


# --------------------------------------------------------------------------- metrics
def score(frame: pd.DataFrame, logp: np.ndarray) -> dict[str, float]:
    pred = np.exp(np.asarray(logp, dtype=float))
    res = all_and_clean(frame["actual"], pred, frame["clean"])
    p = frame[["date", "actual", "clean"]].assign(pred=pred)
    months = sorted(month_metrics(p).items())
    return {
        "mae_clean": res["clean"]["mae"],
        "mae_all": res["all"]["mae"],
        "mape_clean": res["clean"]["mape"],
        "mae_debiased": day_debiased_mae(frame["actual"], pred, frame["date"], frame["clean"]),
        "median_bias_clean": res["clean"]["median_bias"],
        "bias_m1": months[0][1]["median_bias"],
        "bias_m2": months[1][1]["median_bias"],
    }


def blend_log(frame: pd.DataFrame, w: float, cb: str = "cb", tp: str = "tp") -> np.ndarray:
    return w * frame[cb].to_numpy() + (1.0 - w) * frame[tp].to_numpy()


def day_sums(
    frame: pd.DataFrame, logp_a: np.ndarray, logp_b: np.ndarray, debiased: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Per test day (clean rows): sum of |err_a| - |err_b| and the row count.

    ``debiased``: errors after removing each day's median log error (``day_debiased_mae``); the
    shift is per day, so resampling whole days keeps it consistent.
    """
    clean = frame["clean"].to_numpy(dtype=bool)
    y = frame["actual"].to_numpy(dtype=float)
    pa, pb = np.exp(logp_a), np.exp(logp_b)
    if debiased:
        pa = day_debiased_predictions(y, pa, frame["date"], clean)
        pb = day_debiased_predictions(y, pb, frame["date"], clean)
    diff = np.abs(pa - y) - np.abs(pb - y)
    d = pd.DataFrame({"date": frame["date"].to_numpy()[clean], "diff": diff[clean]})
    g = d.groupby("date")["diff"].agg(["sum", "size"])
    return g["sum"].to_numpy(), g["size"].to_numpy(dtype=float)


def bootstrap_delta(parts: list[tuple[np.ndarray, np.ndarray]], seed: int = 0) -> dict[str, float]:
    """Day-block bootstrap of the mean over folds of MAE_a - MAE_b (days resampled per fold)."""
    rng = np.random.default_rng(seed)
    point = float(np.mean([s.sum() / n.sum() for s, n in parts]))
    boots = np.zeros(N_BOOT)
    for s, n in parts:
        idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
        boots += s[idx].sum(axis=1) / n[idx].sum(axis=1)
    boots /= len(parts)
    return {
        "delta": point,
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
        "p_delta_ge_0": float(np.mean(boots >= 0)),
    }


def slice_rows(frame: pd.DataFrame, fold: str, model: str, logp: np.ndarray) -> list[dict]:
    p = frame.assign(pred=np.exp(logp))
    tables = standard_slices(p)
    from freight.evaluate import slice_table

    tables["lane_band"] = slice_table(p, "lane_band")
    rows = []
    for name, t in tables.items():
        for _, r in t.iterrows():
            rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "slice": name,
                    "group": r[name],
                    "n": int(r["n"]),
                    "mae": r["mae"],
                    "mape": r["mape"],
                    "median_bias": r["median_bias"],
                }
            )
    return rows


# --------------------------------------------------------------------------- backtest
def backtest() -> None:
    data = load_backtest_data()
    from freight.split import get_folds

    frames, seeds = {}, {}
    for fold in get_folds(list(FOLDS)):
        train = data.frame[fold.train_mask(data.frame)]
        counts = (train["pickup"].astype(str) + "→" + train["delivery"].astype(str)).value_counts()
        frames[fold.name], seeds[fold.name] = fold_frame(fold.name, counts)

    have_main = [f for f in FOLDS if "tp" in frames[f]]
    if not set(SELECT_FOLDS) <= set(have_main):
        raise SystemExit(f"tp_level predictions missing for {sorted(set(SELECT_FOLDS) - set(have_main))}")

    # ---- grids
    grid_rows = []
    for blend, tp_col in (("cb+tp_level", "tp"), *((f"cb+{e}", c) for e, c in TP_EXTRA.items())):
        for fold in FOLDS:
            if tp_col not in frames[fold]:
                continue
            for w in GRID:
                grid_rows.append({"blend": blend, "fold": fold, "w": w, **score(frames[fold], blend_log(frames[fold], w, tp=tp_col))})
    grid = pd.DataFrame(grid_rows)
    grid.round(5).to_csv(OUT / "blend_grid.csv", index=False)

    main = grid[grid["blend"] == "cb+tp_level"]
    f12 = main[main["fold"].isin(SELECT_FOLDS)].groupby("w")[["mae_clean", "mae_debiased", "mae_all", "mape_clean"]].mean()
    w_star = float(f12["mae_clean"].round(6).idxmin())
    w_star_debiased = float(f12["mae_debiased"].round(6).idxmin())
    cb_f12 = float(f12.loc[1.0, "mae_clean"])
    blend_f12 = float(f12.loc[w_star, "mae_clean"])
    gain_f12 = cb_f12 - blend_f12

    # ---- seed noise: every (CatBoost seed, TabPack seed) pair at w*
    pair_rows = []
    for i in CB_SEEDS:
        for j in seeds["F1"].get(TP_MAIN, []):
            if not all(j in seeds[f].get(TP_MAIN, []) for f in SELECT_FOLDS):
                continue
            vals, cbv = [], []
            for f in SELECT_FOLDS:
                fr = frames[f]
                vals.append(score(fr, blend_log(fr, w_star, cb=f"cb_s{i}", tp=f"tp_s{j}"))["mae_clean"])
                cbv.append(score(fr, fr[f"cb_s{i}"].to_numpy())["mae_clean"])
            pair_rows.append({"cb_seed": i, "tp_seed": j, "blend_f12": float(np.mean(vals)), "cb_f12": float(np.mean(cbv))})
    pairs = pd.DataFrame(pair_rows)
    if len(pairs):
        pairs["gain"] = pairs["cb_f12"] - pairs["blend_f12"]

    # ---- sampling noise: day-block bootstrap of MAE(blend at w*) - MAE(CatBoost)
    boot_f12 = bootstrap_delta(
        [day_sums(frames[f], blend_log(frames[f], w_star), frames[f]["cb"].to_numpy()) for f in SELECT_FOLDS]
    )
    boot_f3 = (
        bootstrap_delta([day_sums(frames["F3"], blend_log(frames["F3"], w_star), frames["F3"]["cb"].to_numpy())])
        if "tp" in frames["F3"]
        else None
    )

    # Information only: the cross-sectional (day-debiased) gain at the debiased-optimal w.
    boot_debiased = {
        "w": w_star_debiased,
        "F1+F2": bootstrap_delta([day_sums(frames[f], blend_log(frames[f], w_star_debiased), frames[f]["cb"].to_numpy(),
                                           debiased=True) for f in SELECT_FOLDS]),
        "F3": (bootstrap_delta([day_sums(frames["F3"], blend_log(frames["F3"], w_star_debiased),
                                         frames["F3"]["cb"].to_numpy(), debiased=True)]) if "tp" in frames["F3"] else None),
    }

    def at(blend: str, fold: str, w: float) -> dict | None:
        r = grid[(grid["blend"] == blend) & (grid["fold"] == fold) & (np.isclose(grid["w"], w))]
        return None if r.empty else r.drop(columns=["blend", "fold"]).iloc[0].to_dict()

    result = {
        "protocol": "w chosen by mean clean MAE over F1+F2 (grid 0..1 step 0.1); F3 scored once at the chosen w",
        "catboost": {"experiment": CB_EXP, "seeds": list(CB_SEEDS), "seed_average": "mean of log predictions"},
        "tabpack_seeds": {f: seeds[f] for f in FOLDS},
        "f12_grid": f12.reset_index().round(4).to_dict(orient="records"),
        "chosen_w": w_star,
        "chosen_w_by_debiased_info": w_star_debiased,
        "f12": {"catboost": cb_f12, "tabpack": float(f12.loc[0.0, "mae_clean"]), "blend": blend_f12, "gain": gain_f12,
                "blend_50_50": float(f12.loc[0.5, "mae_clean"])},
        "per_fold_at_w": {f: {"catboost": at("cb+tp_level", f, 1.0), "tabpack": at("cb+tp_level", f, 0.0),
                              "blend": at("cb+tp_level", f, w_star), "blend_50_50": at("cb+tp_level", f, 0.5)}
                          | {f"{e.removeprefix('tp_level_') if e != 'tp_standalone' else 'standalone'}_{k}": at(f"cb+{e}", f, w)
                             for e in TP_EXTRA for k, w in (("alone", 0.0), ("blend_at_w", w_star), ("blend_50_50", 0.5))}
                          for f in FOLDS},
        "seed_pairs_f12": pairs.round(4).to_dict(orient="records") if len(pairs) else [],
        "seed_pairs_gain_summary": (
            {"mean": float(pairs["gain"].mean()), "sd": float(pairs["gain"].std(ddof=1)) if len(pairs) > 1 else 0.0,
             "min": float(pairs["gain"].min()), "max": float(pairs["gain"].max())} if len(pairs) else None
        ),
        "bootstrap_blend_minus_cb": {"F1+F2": boot_f12, "F3": boot_f3, "n_boot": N_BOOT, "unit": "test days, per fold"},
        "bootstrap_debiased_blend_minus_cb_info": boot_debiased | {"n_boot": N_BOOT, "unit": "test days, per fold"},
        "final_fit_gate": {"rule": f"blend beats {CB_EXP} on F1+F2 mean clean MAE by more than ${GATE_DOLLARS}",
                           "gain": gain_f12, "passes": bool(gain_f12 > GATE_DOLLARS)},
    }
    for e in TP_EXTRA:
        g = grid[grid["blend"] == f"cb+{e}"]
        if not g.empty and set(SELECT_FOLDS) <= set(g["fold"]):
            gf12 = g[g["fold"].isin(SELECT_FOLDS)].groupby("w")[["mae_clean", "mae_debiased"]].mean()
            result[f"{e}_f12_grid_info"] = gf12.reset_index().round(4).to_dict(orient="records")
    (OUT / "blend.json").write_text(json.dumps(result, indent=2, default=float))

    # ---- slices and per-row F3 predictions
    slice_out = []
    for f in FOLDS:
        fr = frames[f]
        models = {"catboost": fr["cb"].to_numpy()}
        if "tp" in fr:
            models["tp_level"] = fr["tp"].to_numpy()
            models[f"blend_w{w_star:.1f}"] = blend_log(fr, w_star)
        for e, c in TP_EXTRA.items():
            if c in fr:
                models[e] = fr[c].to_numpy()
        for m, lp in models.items():
            slice_out += slice_rows(fr, f, m, lp)
    pd.DataFrame(slice_out).round(5).to_csv(OUT / "blend_slices.csv", index=False)
    if "tp" in frames["F3"]:
        fr = frames["F3"]
        rows = fr[["load_id", "date", "equipment", "distance", "actual", "clean", "lane_train_count"]].copy()
        rows["date"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d")
        rows["pred_catboost"] = np.exp(fr["cb"]).round(4)
        rows["pred_tp_level"] = np.exp(fr["tp"]).round(4)
        rows[f"pred_blend_w{w_star:.1f}"] = np.exp(blend_log(fr, w_star)).round(4)
        rows.to_csv(OUT / "blend_preds_F3.csv", index=False)

    # ---- console summary
    print("F1+F2 grid (cb+tp_level): w, mean clean MAE, mean debiased MAE")
    for w, r in f12.iterrows():
        print(f"  w={w:.1f}  clean {r['mae_clean']:.2f}  debiased {r['mae_debiased']:.2f}")
    print(f"chosen w = {w_star} (debiased would pick {w_star_debiased}); F1+F2 CatBoost {cb_f12:.2f} -> blend {blend_f12:.2f} "
          f"(gain {gain_f12:+.2f}); gate (> ${GATE_DOLLARS}) {'PASSES' if gain_f12 > GATE_DOLLARS else 'fails'}")
    print(f"bootstrap F1+F2 MAE(blend)-MAE(CB): {boot_f12}")
    print(f"info: bootstrap debiased MAE(blend w={w_star_debiased})-MAE(CB): {boot_debiased}")
    if len(pairs):
        print(f"seed pairs gain: {result['seed_pairs_gain_summary']}")
    if "tp" in frames["F3"]:
        f3 = result["per_fold_at_w"]["F3"]
        print(f"F3: CatBoost {f3['catboost']['mae_clean']:.2f} | TabPack {f3['tabpack']['mae_clean']:.2f} | "
              f"blend w={w_star} {f3['blend']['mae_clean']:.2f} | 50/50 {f3['blend_50_50']['mae_clean']:.2f} | bootstrap {boot_f3}")


# --------------------------------------------------------------------------- final
def _rel(path) -> str:
    try:
        return str(path.relative_to(config.ROOT))
    except ValueError:
        return str(path)


def final() -> None:
    from freight import final as fin

    blend = json.loads((OUT / "blend.json").read_text())
    w = float(blend["chosen_w"])
    gate = blend["final_fit_gate"]
    print(f"chosen w = {w}; gate passes = {gate['passes']} (gain {gate['gain']:+.2f})")
    fdir = OUT / "final"
    seeds = sorted(int(p.stem.rsplit("_s", 1)[1]) for p in (fdir / "preds").glob("tp_level_FINAL_s*.csv"))
    if not seeds:
        raise SystemExit("no tp_level_FINAL predictions; run `run_all.sh final` first")
    tp = None
    for s in seeds:
        p = pd.read_csv(fdir / "preds" / f"tp_level_FINAL_s{s}.csv").set_index("load_id")
        col = p[["source", "date", "distance", "L", "city_status"]].copy() if tp is None else None
        if tp is None:
            tp = col
        tp[f"f_s{s}"] = p["f"]
    tp["f_tp"] = tp[[f"f_s{s}" for s in seeds]].mean(axis=1)

    model = fin.load_final()
    val_rows = fin.validation_rows()
    chart_rows = fin.december_rows(model)
    chart_ids = "DEC-" + pd.to_datetime(chart_rows["date"]).dt.strftime("%Y-%m-%d")
    parts = []
    for rows, ids in ((val_rows, val_rows["load_id"].to_numpy()), (chart_rows, chart_ids.to_numpy())):
        L, f_cb = model.components(rows)
        parts.append(pd.DataFrame({"load_id": ids, "L_cb": L, "f_cb": f_cb, "distance_cb": rows["distance"].to_numpy(float)}))
    cb = pd.concat(parts, ignore_index=True).set_index("load_id")
    both = tp.join(cb, how="inner")
    assert len(both) == len(tp) == len(cb), (len(both), len(tp), len(cb))
    # The preds CSV stores L rounded to 7 decimals (evaluate_run.py), so |dL| <= 5e-8 is rounding.
    max_dl = float(np.abs(both["L"] - both["L_cb"]).max())
    assert max_dl < 1e-6, f"TabPack export L differs from the final model's L (max |dL| = {max_dl:.2e})"
    # Use the final model's full-precision L for every prediction.
    both["L"] = both["L_cb"]
    both["pred_catboost"] = both["distance"] * np.exp(both["L"] + both["f_cb"])
    both["pred_tabpack"] = both["distance"] * np.exp(both["L"] + both["f_tp"])
    both["pred_blend"] = both["distance"] * np.exp(both["L"] + w * both["f_cb"] + (1 - w) * both["f_tp"])

    # Sanity: the recomputed CatBoost predictions reproduce the submitted file (2 decimals).
    submitted = pd.read_csv(fin.VALIDATION_OUT).set_index("load_id")["predicted_rate"]
    v = both[both["source"] == "validation"]
    max_dev = float(np.abs(v["pred_catboost"].round(2) - submitted.reindex(v.index)).max())

    template = pd.read_csv(config.TEMPLATE_PATH)
    raw_dec = fin.raw_december()
    dec = both[both["source"] == "december_chart"].sort_values("date")
    written = {}
    for kind in ("tabpack", "blend"):
        vp = pd.DataFrame({"load_id": v.index, "predicted_rate": v[f"pred_{kind}"].to_numpy()})
        path_v = fdir / f"validation_predictions_{kind}.csv"
        fin.format_validation(template, vp, decimals=2).to_csv(path_v, index=False)
        path_d = fdir / f"december_chart_predictions_{kind}.csv"
        fin.format_december(raw_dec, dec[f"pred_{kind}"].to_numpy(), decimals=2).to_csv(path_d, index=False)
        written[kind] = [_rel(path_v), _rel(path_d)]
    comp = both.reset_index()[["load_id", "source", "date", "distance", "L", "f_cb", "f_tp", "pred_catboost", "pred_tabpack", "pred_blend"]]
    comp.round(6).to_csv(fdir / "final_components.csv", index=False)

    ratio = v["pred_blend"] / v["pred_catboost"]
    summary = {
        "chosen_w": w,
        "gate": gate,
        "tabpack_seeds": seeds,
        "catboost_seeds": list(model.seeds),
        "catboost_reproduces_submission_max_abs_dev": max_dev,
        "export_L_max_abs_dev_vs_final_model": max_dl,
        "validation": {
            "n": int(len(v)),
            "corr_tabpack_vs_catboost": float(np.corrcoef(v["pred_tabpack"], v["pred_catboost"])[0, 1]),
            "median_ratio_tabpack_over_catboost": float(np.median(v["pred_tabpack"] / v["pred_catboost"])),
            "median_ratio_blend_over_catboost": float(np.median(ratio)),
            "p99_abs_log_ratio_blend_vs_catboost": float(np.quantile(np.abs(np.log(ratio)), 0.99)),
            "n_blend_differs_gt_5pct": int((np.abs(ratio - 1) > 0.05).sum()),
            # Cities never seen in Jan-Oct ("cold") exist only in Nov-Dec, so this is the only
            # TabPack-vs-CatBoost comparison on them (no labels: agreement, not accuracy).
            "by_city_status": {
                status: {
                    "n": int(len(g)),
                    "median_ratio_tabpack_over_catboost": float(np.median(g["pred_tabpack"] / g["pred_catboost"])),
                    "p90_abs_log_ratio_tabpack_vs_catboost": float(
                        np.quantile(np.abs(np.log(g["pred_tabpack"] / g["pred_catboost"])), 0.9)
                    ),
                }
                for status, g in v.groupby("city_status")
            },
        },
        "december_chart": {
            kind: {"min": float(dec[f"pred_{kind}"].min()), "max": float(dec[f"pred_{kind}"].max()),
                   "mean": float(dec[f"pred_{kind}"].mean())}
            for kind in ("catboost", "tabpack", "blend")
        },
        "files": written | {"components": _rel(fdir / "final_components.csv")},
        "note": "The submission files in the project root were not modified.",
    }
    (fdir / "final_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    final() if a.final else backtest()
