"""Loading and row-preserving cleaning of the raw CSV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from freight import config


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def load_train(path: Path = config.TRAIN_PATH) -> pd.DataFrame:
    """Labelled loads, 2025-01-01 .. 2025-10-31."""
    return _read(path)


def load_validation(path: Path = config.VALIDATION_PATH) -> pd.DataFrame:
    """Unlabelled loads to predict, 2025-11-01 .. 2025-12-31."""
    return _read(path)


def load_december(path: Path = config.DECEMBER_PATH) -> pd.DataFrame:
    """The fixed Lexington -> Fort Wayne December chart inputs (no market columns)."""
    return _read(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Repair input columns without ever dropping rows.

    * Negative weights are sign flips (their rate per mile is normal), so we take ``abs``.
    * Missing weight stays NaN: CatBoost routes NaN natively, and an imputed value would
      pretend to know something we do not. A flag keeps the missingness visible.
    * Weights sitting exactly on the cap / floor are censored sentinels; flags let the model
      treat them separately from genuine 47,500 lb loads.
    * ``mi_missing`` records a missing market_index before it gets filled from the daily
      market table in :mod:`freight.features`.
    """
    out = df.copy()
    weight = out["weight"]
    out["weight_was_negative"] = (weight < 0).astype(np.int8)
    out["weight"] = weight.abs()
    out["weight_missing"] = out["weight"].isna().astype(np.int8)
    out["weight_at_cap"] = (out["weight"] == config.WEIGHT_CAP).astype(np.int8)
    out["weight_at_floor"] = (out["weight"] == config.WEIGHT_FLOOR).astype(np.int8)
    if "market_index" in out.columns:
        out["mi_missing"] = out["market_index"].isna().astype(np.int8)
    return out


def add_rpm(df: pd.DataFrame) -> pd.DataFrame:
    """Add rate per mile, the scale on which lanes and outliers are comparable."""
    out = df.copy()
    out["rpm"] = out[config.TARGET] / out["distance"]
    return out
