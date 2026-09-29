"""Metrics and slice tables.

MAE in dollars is the primary metric (Spotter's metric is undisclosed; MAE is the most
interpretable and matches a median-targeting model). Every table is reported on ALL test
rows and on CLEAN rows (corrupted targets removed): the corruption is unpredictable noise,
so the clean numbers measure the model, and the gap measures the noise tax.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from freight.features import distance_band


def regression_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    """MAE, RMSE, MAPE (%), MedAE, R^2 and median bias (median of pred/actual - 1)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if len(y) == 0:
        return {k: float("nan") for k in ["n", "mae", "rmse", "mape", "medae", "r2", "median_bias"]}
    err = p - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": float(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(100 * np.mean(np.abs(err) / np.abs(y))),
        "medae": float(np.median(np.abs(err))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "median_bias": float(np.median(p / y) - 1),
    }


def day_bias(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    dates: pd.Series | np.ndarray,
    clean: np.ndarray | pd.Series | None = None,
) -> pd.Series:
    """Per-day log bias ``b_d = median over the day's clean rows of log(pred / actual)``.

    Indexed by (normalised) date. Days without a clean row do not appear.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    keep = np.ones(len(y), dtype=bool) if clean is None else np.asarray(clean, dtype=bool)
    d = pd.to_datetime(pd.Series(np.asarray(dates))).dt.normalize()
    frame = pd.DataFrame({"date": d[keep].to_numpy(), "log_ratio": np.log(p[keep] / y[keep])})
    return frame.groupby("date")["log_ratio"].median()


def day_debiased_predictions(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    dates: pd.Series | np.ndarray,
    clean: np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    """``pred * exp(-b_d)``: each day's predictions shifted so their clean median log error is 0.

    Rows on a day without clean rows are left unchanged.
    """
    b = day_bias(y_true, y_pred, dates, clean)
    d = pd.to_datetime(pd.Series(np.asarray(dates))).dt.normalize()
    shift = d.map(b).fillna(0.0).to_numpy(dtype=float)
    return np.asarray(y_pred, dtype=float) * np.exp(-shift)


def day_debiased_mae(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    dates: pd.Series | np.ndarray,
    clean: np.ndarray | pd.Series | None = None,
) -> float:
    """MAE ($) on clean rows after removing each test day's median log error.

    A SELECTION INSTRUMENT, not a performance number. In ``log(rpm) = L(date) + f(x)`` any
    error shared by all loads of a day belongs to the market level ``L`` (whose extrapolation
    dominates plain MAE on F1/F2 and cannot be fixed by tree hyper-parameters). Removing it
    isolates the cross-sectional error the trees are responsible for. Always report plain
    MAE next to it.
    """
    y = np.asarray(y_true, dtype=float)
    keep = np.ones(len(y), dtype=bool) if clean is None else np.asarray(clean, dtype=bool)
    adj = day_debiased_predictions(y, y_pred, dates, keep)
    return float(np.mean(np.abs(adj[keep] - y[keep]))) if keep.any() else float("nan")


def all_and_clean(y_true: pd.Series, y_pred: np.ndarray, clean: pd.Series) -> dict[str, dict[str, float]]:
    """Metrics on every test row and on the clean subset."""
    clean = np.asarray(clean, dtype=bool)
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    return {"all": regression_metrics(y, p), "clean": regression_metrics(y[clean], p[clean])}


def slice_table(frame: pd.DataFrame, by: str, clean_only: bool = True) -> pd.DataFrame:
    """Metrics per group of ``by``; ``frame`` needs ``actual``, ``pred``, ``clean`` columns."""
    data = frame[frame["clean"]] if clean_only else frame
    rows = []
    for key, grp in data.groupby(by, observed=True):
        rows.append({by: key, **regression_metrics(grp["actual"], grp["pred"])})
    return pd.DataFrame(rows)


def add_slice_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the standard slicing columns: distance band and calendar month."""
    out = frame.copy()
    out["distance_band"] = distance_band(out["distance"])
    out["month"] = pd.to_datetime(out["date"]).dt.to_period("M").astype(str)
    return out


def standard_slices(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Clean-row slice tables by equipment, distance band, month and (if present) city status."""
    frame = add_slice_columns(frame)
    slices = {by: slice_table(frame, by) for by in ["equipment", "distance_band", "month"]}
    if "city_status" in frame.columns:
        slices["city_status"] = slice_table(frame, "city_status")
    return slices
