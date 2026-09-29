"""Shared fixtures: small synthetic frames (fast) and the real training data (cached)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freight import config
from freight.data import clean, load_train, load_validation
from freight.market import build_market_table


def make_loads(n: int = 400, seed: int = 0, start: str = "2025-01-01", days: int = 120) -> pd.DataFrame:
    """Synthetic loads with the raw schema (including sentinel weights and NaNs)."""
    rng = np.random.default_rng(seed)
    cities = {
        "A": (40.0, -75.0),
        "B": (41.0, -80.0),
        "C": (35.0, -90.0),
        "D": (33.0, -97.0),
        "E": (45.0, -93.0),
    }
    names = list(cities)
    pickup = rng.choice(names, n)
    delivery = np.array([rng.choice([c for c in names if c != p]) for p in pickup])
    dist = np.array(
        [300 + 50 * abs(names.index(p) - names.index(d)) * 3 for p, d in zip(pickup, delivery)], dtype=float
    )
    weight = rng.uniform(8_000, 45_000, n).round()
    weight[:5] = -weight[:5]
    weight[5:8] = config.WEIGHT_CAP
    weight[8:10] = config.WEIGHT_FLOOR
    weight[10:13] = np.nan
    mi = rng.normal(0.95, 0.1, n)
    mi[13:16] = np.nan
    rpm = 2.2 * rng.lognormal(0, 0.02, n)
    rate = rpm * dist
    rate[16:20] *= 3.5  # corrupted up
    rate[20:24] *= 0.3  # corrupted down
    return pd.DataFrame(
        {
            "load_id": [f"TR-{i:06d}" for i in range(n)],
            "pickup": pickup,
            "delivery": delivery,
            "pickup_lat": [cities[c][0] for c in pickup],
            "pickup_lon": [cities[c][1] for c in pickup],
            "delivery_lat": [cities[c][0] for c in delivery],
            "delivery_lon": [cities[c][1] for c in delivery],
            "distance": dist,
            "equipment": rng.choice(["Dry Van", "Reefer", "Flatbed"], n),
            "weight": weight,
            "date": pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days, n), unit="D"),
            "market_index": mi,
            "quote_signal": rng.normal(2.0, 0.3, n),
            "posted_rate": rate,
        }
    )


@pytest.fixture
def synthetic() -> pd.DataFrame:
    return make_loads()


@pytest.fixture(scope="session")
def real_train() -> pd.DataFrame:
    return clean(load_train())


@pytest.fixture(scope="session")
def real_market(real_train: pd.DataFrame) -> pd.DataFrame:
    return build_market_table(real_train, load_validation())
