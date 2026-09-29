"""Project-wide constants: paths, seeds, fold calendar and feature groups.

Everything that a reviewer might want to tweak lives here so the pipeline modules
stay free of magic numbers.
"""

from __future__ import annotations

import math
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT / "data"
# File names on disk use hyphens even though the brief/README spell them with underscores.
TRAIN_PATH: Path = DATA_DIR / "train-test.csv"
VALIDATION_PATH: Path = DATA_DIR / "validation.csv"
DECEMBER_PATH: Path = DATA_DIR / "december-chart-inputs.csv"
TEMPLATE_PATH: Path = DATA_DIR / "validation-predictions-template.csv"

ARTIFACTS_DIR: Path = ROOT / "artifacts"
BACKTEST_DIR: Path = ARTIFACTS_DIR / "backtest"
PREDS_DIR: Path = BACKTEST_DIR / "preds"
PREVIEW_DIR: Path = ARTIFACTS_DIR / "preview"
TUNING_DIR: Path = ARTIFACTS_DIR / "tuning"
# Chosen CatBoost hyper-parameters of f(x) (phase 2b, scripts/tune.py); committed, and read
# by the cb_level_tuned experiment and the final fit.
BEST_PARAMS_PATH: Path = TUNING_DIR / "best_params.json"
MPL_STYLE_PATH: Path = ROOT / "reports" / "styles" / "report.mplstyle"

# --------------------------------------------------------------------------- runtime
SEED: int = 42
# 8-core laptop with little free RAM: leave two cores for the OS / other work.
THREAD_COUNT: int = 6

# --------------------------------------------------------------------------- data rules
TARGET: str = "posted_rate"
DATE_ORIGIN: str = "2025-01-01"
WEIGHT_CAP: float = 47_500.0  # sentinel: 1,191 rows sit exactly on it, 1,204 after abs() (13 rows are -47,500); censored upper tail
WEIGHT_FLOOR: float = 5_000.0  # sentinel: 31 rows sit exactly on it, 32 after abs() (1 row is -5,000); censored lower tail

# Distance bands (miles) used by the coarse fallback of the rpm reference and by slice tables.
DISTANCE_BAND_EDGES: tuple[float, ...] = (0, 150, 300, 500, 750, 1000, 1500, 2000, 2500, float("inf"))

# |log(rpm / reference rpm)| above this marks a corrupted target. Chosen in the empty gap of
# the bimodal |log-ratio| histogram (scripts/outlier_threshold.py ->
# artifacts/backtest/outlier_log_ratio_hist.png). With the fold-trained two-stage reference,
# genuine rows reach at most 0.23 and corrupted rows start at 0.73 in every fold's train and
# test window; log(1.6) = 0.470 sits in the middle of that gap, so any value in 0.25..0.70
# yields identical flags.
OUTLIER_LOG_RATIO_THRESHOLD: float = math.log(1.6)
# Minimum group size before a lane x equipment (or lane) median is trusted.
REFERENCE_MIN_COUNT: int = 3

# --------------------------------------------------------------------------- folds
# Expanding-window, forward-time folds with 2-month test windows. Training always starts on
# DATE_ORIGIN. F3 is the reported holdout; hyper-parameter tuning may only look at F1 + F2.
FOLD_DATES: dict[str, tuple[str, str, str, str]] = {
    # name: (train_start, train_end, test_start, test_end) -- all bounds inclusive
    "F1": ("2025-01-01", "2025-04-30", "2025-05-01", "2025-06-30"),
    "F2": ("2025-01-01", "2025-06-30", "2025-07-01", "2025-08-31"),
    "F3": ("2025-01-01", "2025-08-31", "2025-09-01", "2025-10-31"),
}
TUNING_FOLDS: tuple[str, ...] = ("F1", "F2")
HOLDOUT_FOLD: str = "F3"

# --------------------------------------------------------------------------- features
CAT_FEATURES: list[str] = ["pickup", "delivery", "lane", "equipment"]
GEO_FEATURES: list[str] = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "haversine_mi",
    "bearing_deg",
    "delta_lat",
    "delta_lon",
]
DISTANCE_FEATURES: list[str] = ["distance", "log_distance", "circuity"]
WEIGHT_FEATURES: list[str] = [
    "weight",
    "weight_missing",
    "weight_was_negative",
    "weight_at_cap",
    "weight_at_floor",
]
MARKET_RAW_FEATURES: list[str] = ["market_index", "mi_missing", "quote_signal"]
# Derived from the daily market table (input columns only, trailing windows).
MARKET_DERIVED_FEATURES: list[str] = ["mi_day_mean", "mi_dev", "mi_ma7", "mi_ma28"]
CALENDAR_FEATURES: list[str] = [
    "dow",
    "dom",
    "days_to_month_end",
    "days_to_quarter_end",
    "is_weekend",
]
TIME_FEATURES: list[str] = ["t_days"]
# Built but only used when an experiment opts in (ExperimentConfig.extra_features).
# quote_signal carries a month-level offset that steps at month boundaries (sd of monthly
# means 0.12 vs within-month daily sd 0.03) -- a hidden month identifier. qs_dev removes the
# day's market-wide mean and keeps only the per-load part.
OPTIONAL_FEATURES: list[str] = ["qs_dev"]

# Absolute calendar positions that would act as an extrapolated time axis for Nov-Dec
# (never seen in training). Tests assert none of these appear in the feature matrix.
FORBIDDEN_FEATURES: tuple[str, ...] = (
    "month",
    "day_of_year",
    "dayofyear",
    "doy",
    "week",
    "week_of_year",
    "weekofyear",
    "quarter",
)

# --------------------------------------------------------------------------- level model
# Features of the cross-sectional CatBoost in log(rpm) = L(date) + f(x) (freight.level).
# The daily price LEVEL is owned by L (mi_day_mean, trend, quarter-end ramp), so f only gets
# per-load / per-lane information plus in-range cyclic calendar (dow, dom):
# market readings enter as deviations from their day mean (mi_dev, qs_dev).
LEVEL_CB_FEATURES: list[str] = (
    CAT_FEATURES
    + GEO_FEATURES
    + DISTANCE_FEATURES
    + WEIGHT_FEATURES
    + ["mi_dev", "qs_dev", "mi_missing", "dow", "dom"]
)
# Level carriers / month proxies that must never reach f (asserted in tests). Raw
# market_index = mi_day_mean + mi_dev carries the day level; days_to_month_end + dom encodes
# the month length (Feb vs 30- vs 31-day months), a weak month identifier.
# mi_day_mean may be re-added deliberately as an ablation (cb_level_mi).
LEVEL_CB_EXCLUDED: tuple[str, ...] = (
    "market_index",
    "quote_signal",
    "mi_day_mean",
    "mi_ma7",
    "mi_ma28",
    "t_days",
    "days_to_month_end",
    "days_to_quarter_end",
)
LEVEL_SEEDS: tuple[int, ...] = (0, 1, 2)

# Feature families of the cb_level_noqs trees, masked one at a time in the phase-2b ablation
# (freight.tuning.run_ablation). equipment, distance and log_distance are never masked
# (distance also converts rpm back to dollars).
ABLATION_FAMILIES: dict[str, tuple[str, ...]] = {
    "city": ("pickup", "delivery"),
    "lane": ("lane",),
    "coords": ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "delta_lat", "delta_lon", "bearing_deg"),
    "crowfly": ("haversine_mi", "circuity"),
    "weight": tuple(WEIGHT_FEATURES),
    "dow_dom": ("dow", "dom"),
    "mi_dev": ("mi_dev", "mi_missing"),
}

ALL_FEATURES: list[str] = (
    CAT_FEATURES
    + GEO_FEATURES
    + DISTANCE_FEATURES
    + WEIGHT_FEATURES
    + MARKET_RAW_FEATURES
    + MARKET_DERIVED_FEATURES
    + CALENDAR_FEATURES
)
