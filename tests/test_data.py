from __future__ import annotations

import numpy as np
import pandas as pd

from freight import config
from freight.data import clean


def test_clean_never_drops_rows(synthetic: pd.DataFrame) -> None:
    out = clean(synthetic)
    assert len(out) == len(synthetic)
    assert out.index.equals(synthetic.index)


def test_clean_does_not_mutate_input(synthetic: pd.DataFrame) -> None:
    before = synthetic.copy()
    clean(synthetic)
    pd.testing.assert_frame_equal(synthetic, before)


def test_negative_weights_are_flipped_and_flagged(synthetic: pd.DataFrame) -> None:
    out = clean(synthetic)
    neg = synthetic["weight"] < 0
    assert neg.sum() == 5
    assert (out["weight"].dropna() >= 0).all()
    np.testing.assert_allclose(out.loc[neg, "weight"], -synthetic.loc[neg, "weight"])
    assert (out["weight_was_negative"] == neg.astype(int)).all()


def test_weight_nan_kept_and_flagged(synthetic: pd.DataFrame) -> None:
    out = clean(synthetic)
    assert out["weight"].isna().sum() == synthetic["weight"].isna().sum() == 3
    assert (out["weight_missing"] == out["weight"].isna().astype(int)).all()


def test_sentinel_flags(synthetic: pd.DataFrame) -> None:
    out = clean(synthetic)
    assert out["weight_at_cap"].sum() == (out["weight"] == config.WEIGHT_CAP).sum() == 3
    assert out["weight_at_floor"].sum() == (out["weight"] == config.WEIGHT_FLOOR).sum() == 2


def test_market_index_missing_flag(synthetic: pd.DataFrame) -> None:
    out = clean(synthetic)
    assert out["mi_missing"].sum() == 3
    assert out["market_index"].isna().sum() == 3  # filling happens in features, not here


def test_real_data_cleaning_counts(real_train: pd.DataFrame) -> None:
    assert len(real_train) == 48_000
    assert real_train["weight_was_negative"].sum() == 292
    assert real_train["weight_missing"].sum() == 300
    # 1,191 raw rows sit on +47,500 and 13 more on -47,500 (sign-flipped sentinel), so the
    # cap flag -- computed after abs() -- covers 1,204 rows; likewise 31 + 1 on the floor.
    assert real_train["weight_at_cap"].sum() == 1_204
    assert real_train["weight_at_floor"].sum() == 32
    assert real_train["mi_missing"].sum() == 374
