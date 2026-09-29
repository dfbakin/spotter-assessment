"""Export one backtest fold (or the final Jan-Oct fit) to TabPack's on-disk dataset format.

Run with the MAIN project venv from the project root:

    .venv/bin/python scripts/tabpack/export_fold.py --fold F1 --mode level --lane drop \
        --out tabpack/data/freight-level-nolane-F1

Modes
* ``level`` (tp_level, tp_level_lanetopk): mirrors ``freight.backtest.run_fold`` +
  ``LevelCatBoostRate`` (K = 1) of ``cb_level_noqs`` / ``cb_level_tuned``:
  RpmReference on ALL window rows -> corrupted rows dropped -> L = MarketLevel fitted on the
  clean window rows (residual log(rpm / reference rpm)) -> target y = log(rpm) - L(date);
  features = the 24 tree features of ``artifacts/tuning/best_params.json`` (``--lane`` decides
  what happens to ``lane``).
* ``standalone`` (tp_standalone): target y = log(rpm) (no level model, L := 0); features = the
  phase-1 ``cb_time`` feature set (raw market columns, calendar incl. days_to_quarter_end,
  t_days, geography, distance, weight + flags, cities, equipment) with ``lane`` dropped.

Splits: inner TIME validation = the last ``--val-days`` days of the training window (clean
rows only); train = the clean rows before it; test = the fold's full test window (corrupted
rows kept and flagged in meta.csv, scored on clean rows).

``--fold FINAL`` (level mode only): train on all clean Jan-Oct rows (validation = last 28
days of October); test = the 12,000 Nov-Dec validation rows + the 31 December chart rows
(Option A market inputs, as in ``freight.final``) with a dummy target of 0.

Layout written (TabPack ``lib.data``): x_num.npy (float32, NaN kept), x_cat.npy (str), y.npy
(float32), splits/default/{train,val,test}.npy (int32), info.json; plus meta.csv (row-aligned
with the split order: part, load_id, date, distance, actual, L, target, clean, ...) and
export.json (provenance). A copy of export.json goes to ``--provenance`` if given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from freight import config
from freight.backtest import EXPERIMENTS, _feature_view, load_backtest_data
from freight.data import clean, load_december, load_validation
from freight.features import build_features, city_coordinates, prepare_chart_rows
from freight.level import MarketLevel
from freight.models import log_rpm
from freight.outliers import RpmReference, flag_corrupted
from freight.split import get_folds

ap = argparse.ArgumentParser()
ap.add_argument("--fold", required=True, choices=["F1", "F2", "F3", "FINAL"])
ap.add_argument("--mode", required=True, choices=["level", "standalone"])
ap.add_argument("--out", required=True)
ap.add_argument("--val-days", type=int, default=28)
ap.add_argument("--lane", choices=["drop", "topk", "full"], default="drop")
ap.add_argument("--lane-topk", type=int, default=155, help="topk: keep the K most frequent inner-train lanes")
ap.add_argument("--provenance", default=None, help="also copy export.json to this path")
args = ap.parse_args()
if args.fold == "FINAL" and args.mode != "level":
    raise SystemExit("--fold FINAL is only defined for --mode level")
if args.mode == "standalone" and args.lane != "drop":
    raise SystemExit("standalone mode drops the lane by definition")

data = load_backtest_data()
frame = data.frame

# ------------------------------------------------------------------ rows and outlier flags
if args.fold == "FINAL":
    # Exactly freight.final._training_rows: one reference on all Jan-Oct rows.
    train_rows = frame
    train_end = pd.Timestamp(frame["date"].max()).normalize()
    window = [str(pd.Timestamp(frame["date"].min()).date()), str(train_end.date())]
    reference = RpmReference().fit(train_rows)
    train_flagged = flag_corrupted(reference, train_rows, config.OUTLIER_LOG_RATIO_THRESHOLD)
    fit_rows = train_rows[~train_flagged]
    validation = clean(load_validation())
    coords = city_coordinates(frame, load_validation())
    chart = prepare_chart_rows(load_december(), data.market_table, coords).copy()
    chart["load_id"] = "DEC-" + pd.to_datetime(chart["date"]).dt.strftime("%Y-%m-%d")
    validation["source"] = "validation"
    chart["source"] = "december_chart"
    test_rows = pd.concat([validation, chart], ignore_index=True)
    test_flagged = pd.Series(False, index=test_rows.index)
    test_window = [str(pd.Timestamp(test_rows["date"].min()).date()), str(pd.Timestamp(test_rows["date"].max()).date())]
else:
    fold = get_folds([args.fold])[0]
    train_rows = frame[fold.train_mask(frame)]
    test_rows = frame[fold.test_mask(frame)].copy()
    test_rows["source"] = "test"
    train_end = fold.train_end
    window = [str(fold.train_start.date()), str(fold.train_end.date())]
    test_window = [str(fold.test_start.date()), str(fold.test_end.date())]
    # As run_fold (no cold start): one reference on the full training window flags both the
    # training rows (dropped) and the test rows (evaluation clean mask).
    reference = RpmReference().fit(train_rows)
    test_flagged = reference.log_ratio(test_rows).abs() > config.OUTLIER_LOG_RATIO_THRESHOLD
    train_flagged = flag_corrupted(reference, train_rows)
    fit_rows = train_rows[~train_flagged]

# ------------------------------------------------------------------ feature matrices
base_cfg = EXPERIMENTS["cb_level_tuned" if args.mode == "level" else "cb_time"]
X_all, cat_features = _feature_view(base_cfg, data)
if args.mode == "level":
    best = json.loads(config.BEST_PARAMS_PATH.read_text())
    assert list(X_all.columns) == best["tree_features"], (list(X_all.columns), best["tree_features"])
feature_cols = list(X_all.columns)
X_fit = X_all.loc[fit_rows.index]
if args.fold == "FINAL":
    X_test = build_features(test_rows, data.market_table, use_time=True)[0][feature_cols]
else:
    X_test = X_all.loc[test_rows.index]

# ------------------------------------------------------------------ targets
y_fit = log_rpm(fit_rows)
level_coef = None
if args.mode == "level":
    # L exactly as LevelCatBoostRate.fit, round 1 (= freight.final._fit_level for FINAL).
    resid = y_fit - np.log(reference.reference_rpm(fit_rows).to_numpy())
    level = MarketLevel(data.market_table, base_cfg.level_terms).fit(fit_rows, resid)
    level_coef = level.coefficients()
    L_fit = level.predict(fit_rows)
    L_test = level.predict(test_rows)
else:
    L_fit = np.zeros(len(fit_rows))
    L_test = np.zeros(len(test_rows))
t_fit = y_fit - L_fit
if args.fold == "FINAL":
    t_test = np.zeros(len(test_rows))  # dummy target (unlabelled rows)
    actual_test = np.full(len(test_rows), np.nan)
else:
    t_test = log_rpm(test_rows) - L_test
    actual_test = test_rows[config.TARGET].to_numpy(dtype=float)

# ------------------------------------------------------------------ splits
val_start = train_end - pd.Timedelta(days=args.val_days - 1)
is_val = (pd.to_datetime(fit_rows["date"]).dt.normalize() >= val_start).to_numpy()
parts = {
    "train": (fit_rows[~is_val], X_fit[~is_val], t_fit[~is_val], L_fit[~is_val], fit_rows[config.TARGET].to_numpy(float)[~is_val]),
    "val": (fit_rows[is_val], X_fit[is_val], t_fit[is_val], L_fit[is_val], fit_rows[config.TARGET].to_numpy(float)[is_val]),
    "test": (test_rows, X_test, t_test, L_test, actual_test),
}

cat_cols = list(cat_features)
lane_kept: list[str] | None = None
if "lane" in cat_cols:
    if args.lane == "drop":
        cat_cols.remove("lane")
    elif args.lane == "topk":
        counts = parts["train"][1]["lane"].value_counts()
        # Deterministic order: count desc, then name.
        counts = counts.sort_index().sort_values(ascending=False, kind="mergesort")
        lane_kept = list(counts.index[: args.lane_topk])
        keep = set(lane_kept)
        for key, (rows, Xp, t, L, act) in parts.items():
            Xp = Xp.copy()
            Xp["lane"] = Xp["lane"].where(Xp["lane"].isin(keep), "__other__")
            parts[key] = (rows, Xp, t, L, act)
num_cols = [c for c in feature_cols if c not in cat_features]

# Lane frequency in the inner-train rows (for the lane slices; independent of --lane).
train_lane_counts = (
    parts["train"][0]["pickup"].astype(str) + "→" + parts["train"][0]["delivery"].astype(str)
).value_counts()

out = Path(args.out)
(out / "splits" / "default").mkdir(parents=True, exist_ok=True)
x_num, x_cat, y, meta = [], [], [], []
train_cities = set(train_rows["pickup"]) | set(train_rows["delivery"])
start = 0
for key, (rows, Xp, t, L, act) in parts.items():
    n = len(rows)
    np.save(out / "splits" / "default" / f"{key}.npy", np.arange(start, start + n, dtype=np.int32))
    start += n
    x_num.append(Xp[num_cols].to_numpy(dtype=np.float32))
    x_cat.append(Xp[cat_cols].astype(str).to_numpy())
    y.append(np.asarray(t, dtype=np.float32))
    if key == "test":
        clean_mask = ~test_flagged.to_numpy() if args.fold != "FINAL" else np.zeros(n, dtype=bool)
    else:
        clean_mask = np.ones(n, dtype=bool)
    cold = ~(rows["pickup"].isin(train_cities) & rows["delivery"].isin(train_cities))
    lane = rows["pickup"].astype(str) + "→" + rows["delivery"].astype(str)
    meta.append(
        pd.DataFrame(
            {
                "part": key,
                "source": rows["source"].to_numpy() if "source" in rows else key,
                "load_id": rows["load_id"].to_numpy(),
                "date": pd.to_datetime(rows["date"]).dt.normalize().to_numpy(),
                "pickup": rows["pickup"].to_numpy(),
                "delivery": rows["delivery"].to_numpy(),
                "equipment": rows["equipment"].to_numpy(),
                "distance": rows["distance"].to_numpy(dtype=float),
                "actual": np.asarray(act, dtype=float),
                "L": np.asarray(L, dtype=float),
                "target": np.asarray(t, dtype=float),
                "clean": clean_mask,
                "city_status": np.where(cold, "cold", "warm"),
                "lane_train_count": lane.map(train_lane_counts).fillna(0).astype(int).to_numpy(),
            }
        )
    )

np.save(out / "x_num.npy", np.concatenate(x_num))
np.save(out / "x_cat.npy", np.concatenate(x_cat).astype(np.str_))
np.save(out / "y.npy", np.concatenate(y))
pd.concat(meta, ignore_index=True).to_csv(out / "meta.csv", index=False)
(out / "info.json").write_text(json.dumps({"name": out.name, "task": {"type": "regression", "score": "rmse"}}, indent=2))
Xall = pd.concat([p[1] for p in parts.values()])
summary = {
    "fold": args.fold,
    "mode": args.mode,
    "base_feature_config": base_cfg.name,
    "train_window": window,
    "val_window": [str(val_start.date()), str(train_end.date())],
    "test_window": test_window,
    "n": {k: len(v[0]) for k, v in parts.items()},
    "n_train_window": len(train_rows),
    "n_train_flagged_dropped": int(train_flagged.sum()),
    "n_test_flagged": int(test_flagged.sum()),
    "n_test_cold": int((meta[-1]["city_status"] == "cold").sum()),
    "lane_mode": args.lane if "lane" in cat_features else "absent",
    "lane_topk": args.lane_topk if args.lane == "topk" else None,
    "lane_topk_min_count": int(train_lane_counts.loc[lane_kept].min()) if lane_kept else None,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "cat_cardinality_train": {c: int(parts["train"][1][c].nunique()) for c in cat_cols},
    "level_terms": list(base_cfg.level_terms) if args.mode == "level" else None,
    "level_coef": level_coef,
    "target_std": {k: float(np.std(v[2])) for k, v in parts.items()},
    "nan_counts": {c: int(Xall[c].isna().sum()) for c in num_cols if Xall[c].isna().any()},
}
text = json.dumps(summary, indent=2)
(out / "export.json").write_text(text)
if args.provenance:
    Path(args.provenance).parent.mkdir(parents=True, exist_ok=True)
    Path(args.provenance).write_text(text)
print(text)
