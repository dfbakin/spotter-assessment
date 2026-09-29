"""Feature matrix construction.

Design notes
------------
* City IDs alone cannot serve 8 cities that only appear in validation (12% of its rows), so
  geography is also described by coordinates, great-circle distance, bearing and deltas.
  Coordinates are distorted but consistent per city, so they still work as a smooth location
  code for trees.
* No ``month``, ``day_of_year`` or ``week_of_year``: validation is Nov-Dec, which the model
  never sees in training. Such columns would force trees to extrapolate an absolute time
  axis (every Nov-Dec value falls in the last leaf of any split learned on Jan-Oct). Only
  *cyclic, in-range* calendar effects are used (day of week, day of month, days to
  month/quarter end), whose full range is covered by training data. An explicit, monotone
  time coordinate ``t_days`` is available behind ``use_time=True`` so its value can be
  tested as a separate experiment.
* Every feature is computable from input columns only; nothing here touches the target.
* Rows without market columns (the December chart file) get "Option A" market inputs:
  market_index := the date's mi_day_mean and quote_signal := the date's qs_day_mean from
  the market table, i.e. a typical load on that day (mi_dev = qs_dev = 0, mi_missing = 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from freight import config
from freight.data import clean as clean_rows
from freight.market import fill_market_index, join_market

EARTH_RADIUS_MI: float = 3958.8


def haversine_miles(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> np.ndarray:
    """Great-circle distance in miles."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return np.asarray(2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a)))


def initial_bearing(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> np.ndarray:
    """Initial compass bearing (degrees, 0 = north) from pickup to delivery.

    Direction matters in freight (headhaul vs backhaul pricing), which symmetric distance
    cannot express.
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlmb = np.radians(lon2) - np.radians(lon1)
    x = np.sin(dlmb) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlmb)
    return np.asarray((np.degrees(np.arctan2(x, y)) + 360.0) % 360.0)


def distance_band(distance: pd.Series) -> pd.Series:
    """Discretise miles into the bands of ``config.DISTANCE_BAND_EDGES`` (string labels)."""
    edges = list(config.DISTANCE_BAND_EDGES)
    labels = [
        f"{int(lo)}-{int(hi)}" if np.isfinite(hi) else f"{int(lo)}+"
        for lo, hi in zip(edges[:-1], edges[1:])
    ]
    return pd.cut(distance, bins=edges, labels=labels, right=False).astype(str)


def lane_key(df: pd.DataFrame) -> pd.Series:
    """Directional lane identifier ``pickup→delivery``."""
    return df["pickup"].astype(str) + "→" + df["delivery"].astype(str)


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """City -> (lat, lon) from pickup and delivery columns of input frames.

    Coordinates are distorted in the data but constant per city; a city with conflicting
    coordinates raises, so a silent mix-up cannot happen.
    """
    parts = []
    for f in frames:
        for side in ("pickup", "delivery"):
            parts.append(
                f[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
            )
    coords = pd.concat(parts, ignore_index=True).drop_duplicates()
    dup = coords["city"].duplicated(keep=False)
    if dup.any():
        raise ValueError(f"cities with conflicting coordinates: {sorted(coords.loc[dup, 'city'].unique())[:5]}")
    return coords.set_index("city")


def attach_coordinates(df: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Add pickup/delivery lat/lon columns from a :func:`city_coordinates` table."""
    out = df.copy()
    for side in ("pickup", "delivery"):
        unknown = set(out[side]) - set(coords.index)
        if unknown:
            raise KeyError(f"no coordinates for {side} cities {sorted(unknown)}")
        out[f"{side}_lat"] = out[side].map(coords["lat"]).astype(float)
        out[f"{side}_lon"] = out[side].map(coords["lon"]).astype(float)
    return out


def apply_option_a_market(df: pd.DataFrame, market_table: pd.DataFrame) -> pd.DataFrame:
    """Fill ABSENT market columns with the date's market-wide means ("Option A").

    ``market_index`` := ``mi_day_mean`` and ``quote_signal`` := ``qs_day_mean`` of the row's
    date, i.e. the load is priced as a typical load of that day. Columns that already exist
    are left untouched (their NaNs are handled by :func:`build_features`).
    """
    out = df.copy()
    dates = pd.to_datetime(out["date"]).dt.normalize().to_numpy()
    for col, source in (("market_index", "mi_day_mean"), ("quote_signal", "qs_day_mean")):
        if col not in out.columns:
            values = market_table[source].reindex(dates).to_numpy(dtype=float)
            if np.isnan(values).any():
                raise ValueError(f"market table lacks {source} for some dates; cannot fill {col}")
            out[col] = values
    return out


def prepare_chart_rows(raw: pd.DataFrame, market_table: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Turn raw December-chart rows (no coordinates, no market columns) into a cleaned frame.

    Adds coordinates, Option-A market inputs, then :func:`freight.data.clean` (weight flags).
    """
    rows = raw.drop(columns=[c for c in ("predicted_rate",) if c in raw.columns])
    rows = attach_coordinates(rows, coords)
    rows = apply_option_a_market(rows, market_table)
    return clean_rows(rows)


def _calendar(dates: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(dates).dt.normalize()
    month_end = dates + pd.offsets.MonthEnd(0)
    quarter_end = dates + pd.offsets.QuarterEnd(0)
    return pd.DataFrame(
        {
            "dow": dates.dt.dayofweek.astype(np.int8),
            "dom": dates.dt.day.astype(np.int8),
            "days_to_month_end": (month_end - dates).dt.days.astype(np.int16),
            "days_to_quarter_end": (quarter_end - dates).dt.days.astype(np.int16),
            "is_weekend": (dates.dt.dayofweek >= 5).astype(np.int8),
        },
        index=dates.index,
    )


def build_features(
    df: pd.DataFrame,
    market_table: pd.DataFrame,
    use_time: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the model matrix from a *cleaned* frame (see :func:`freight.data.clean`).

    Returns ``(X, cat_features)``. ``X`` keeps ``df``'s index. The only column allowed to
    contain NaN is ``weight`` (CatBoost handles missing numerics natively). Columns in
    ``config.OPTIONAL_FEATURES`` are always built; experiments opt into them. Rows without
    ``market_index`` / ``quote_signal`` columns get Option-A values
    (:func:`apply_option_a_market`).
    """
    required = {"weight_missing", "weight_was_negative", "weight_at_cap", "weight_at_floor"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"build_features expects a cleaned frame; missing {sorted(missing)}")

    df = apply_option_a_market(df, market_table)

    X = pd.DataFrame(index=df.index)
    # categorical
    X["pickup"] = df["pickup"].astype(str)
    X["delivery"] = df["delivery"].astype(str)
    X["lane"] = lane_key(df)
    X["equipment"] = df["equipment"].astype(str)

    # geography
    for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[col] = df[col].astype(float)
    hav = haversine_miles(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
    X["haversine_mi"] = hav
    X["bearing_deg"] = initial_bearing(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
    X["delta_lat"] = df["delivery_lat"] - df["pickup_lat"]
    X["delta_lon"] = df["delivery_lon"] - df["pickup_lon"]

    # distance
    X["distance"] = df["distance"].astype(float)
    X["log_distance"] = np.log(df["distance"].astype(float))
    # Road miles per crow-fly mile; guard against coincident (clipped) coordinates.
    X["circuity"] = df["distance"] / np.clip(hav, 1.0, None)

    # weight (NaN kept on purpose)
    for col in config.WEIGHT_FEATURES:
        X[col] = df[col]

    # market
    with_market = join_market(df, market_table)
    X["market_index"] = fill_market_index(df, market_table)
    X["mi_missing"] = df["market_index"].isna().astype(np.int8)
    X["quote_signal"] = df["quote_signal"].astype(float)
    X["mi_day_mean"] = with_market["mi_day_mean"]
    X["mi_dev"] = X["market_index"] - X["mi_day_mean"]
    X["mi_ma7"] = with_market["mi_ma7"]
    X["mi_ma28"] = with_market["mi_ma28"]

    # calendar (cyclic, within-range effects only -- see module docstring)
    X = pd.concat([X, _calendar(df["date"])], axis=1)

    # optional (config.OPTIONAL_FEATURES): per-load quote signal net of the day's mean
    X["qs_dev"] = X["quote_signal"] - with_market["qs_day_mean"]

    if use_time:
        origin = pd.Timestamp(config.DATE_ORIGIN)
        X["t_days"] = (pd.to_datetime(df["date"]).dt.normalize() - origin).dt.days.astype(float)

    cat_features = list(config.CAT_FEATURES)
    return X, cat_features


def select_features(
    X: pd.DataFrame, cat_features: list[str], exclude: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Drop an ablated feature family while keeping the categorical list consistent."""
    exclude = list(exclude or [])
    unknown = set(exclude) - set(X.columns)
    if unknown:
        raise KeyError(f"cannot exclude unknown features {sorted(unknown)}")
    keep = [c for c in X.columns if c not in exclude]
    return X[keep], [c for c in cat_features if c in keep]
