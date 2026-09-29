from __future__ import annotations

import pandas as pd
import pytest

from freight import config
from freight.split import Fold, cold_start_holdout, get_folds, iter_folds


def test_folds_are_forward_in_time(real_train: pd.DataFrame) -> None:
    for fold, train_mask, test_mask in iter_folds(real_train):
        train_dates = real_train.loc[train_mask, "date"]
        test_dates = real_train.loc[test_mask, "date"]
        assert train_mask.sum() > 0 and test_mask.sum() > 0
        assert train_dates.max() < test_dates.min(), fold.name
        assert not (train_mask & test_mask).any(), fold.name


def test_folds_expand_and_start_on_origin() -> None:
    folds = get_folds()
    assert [f.name for f in folds] == list(config.FOLD_DATES)
    for prev, nxt in zip(folds, folds[1:]):
        assert nxt.train_end > prev.train_end
    assert all(f.train_start == pd.Timestamp(config.DATE_ORIGIN) for f in folds)
    assert folds[-1].test_end == pd.Timestamp("2025-10-31")


def test_fold_rejects_overlap() -> None:
    with pytest.raises(ValueError):
        Fold("bad", *(pd.Timestamp(d) for d in ["2025-01-01", "2025-05-01", "2025-05-01", "2025-06-30"]))


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_cold_start_removes_every_training_row_of_held_out_cities(real_train: pd.DataFrame, seed: int) -> None:
    fold = get_folds(["F2"])[0]
    train = real_train[fold.train_mask(real_train)]
    test = real_train[fold.test_mask(real_train)]
    split = cold_start_holdout(train, test, n_cities=8, seed=seed)

    assert len(split.cities) == 8
    kept = train[split.train_keep]
    cities = set(split.cities)
    assert not kept["pickup"].isin(cities).any()
    assert not kept["delivery"].isin(cities).any()
    # every removed row touches a held-out city
    removed = train[~split.train_keep]
    assert (removed["pickup"].isin(cities) | removed["delivery"].isin(cities)).all()
    # cold test mask = test rows touching a held-out city, and there are some
    cold = test[split.test_cold]
    assert len(cold) > 0
    assert (cold["pickup"].isin(cities) | cold["delivery"].isin(cities)).all()
    warm = test[~split.test_cold]
    assert not (warm["pickup"].isin(cities) | warm["delivery"].isin(cities)).any()


def test_cold_start_is_seeded(real_train: pd.DataFrame) -> None:
    fold = get_folds(["F3"])[0]
    train, test = real_train[fold.train_mask(real_train)], real_train[fold.test_mask(real_train)]
    a = cold_start_holdout(train, test, seed=0).cities
    b = cold_start_holdout(train, test, seed=0).cities
    c = cold_start_holdout(train, test, seed=1).cities
    assert a == b and a != c
