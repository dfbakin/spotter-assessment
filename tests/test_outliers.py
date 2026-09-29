from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from freight import config
from freight.data import clean
from freight.outliers import RpmReference, fit_and_flag, flag_corrupted
from freight.split import get_folds


def test_fit_takes_training_rows_explicitly() -> None:
    params = list(inspect.signature(RpmReference.fit).parameters)
    assert params == ["self", "train_rows"]
    assert list(inspect.signature(fit_and_flag).parameters)[:2] == ["train_rows", "test_rows"]


def test_reference_depends_only_on_training_rows(real_train: pd.DataFrame) -> None:
    """Scrambling test-window labels must not change the reference or the training flags."""
    fold = get_folds(["F2"])[0]
    train = real_train[fold.train_mask(real_train)]
    test = real_train[fold.test_mask(real_train)]

    ref_a, flags_a, _ = fit_and_flag(train, test)
    scrambled = test.copy()
    scrambled["posted_rate"] = scrambled["posted_rate"].sample(frac=1.0, random_state=0).to_numpy() * 3
    ref_b, flags_b, _ = fit_and_flag(train, scrambled)

    pd.testing.assert_series_equal(flags_a, flags_b)
    probe = test.drop(columns="posted_rate")
    pd.testing.assert_series_equal(ref_a.reference_rpm(probe), ref_b.reference_rpm(probe))
    assert ref_a.n_train_ == len(train)


def test_synthetic_corruption_is_recovered(synthetic: pd.DataFrame) -> None:
    df = clean(synthetic)
    ref = RpmReference().fit(df)
    flags = flag_corrupted(ref, df)
    expected = np.zeros(len(df), dtype=bool)
    expected[16:24] = True
    np.testing.assert_array_equal(flags.to_numpy(), expected)


def test_real_flag_rate_is_about_1_4_percent(real_train: pd.DataFrame) -> None:
    ref = RpmReference().fit(real_train)
    a = ref.log_ratio(real_train).abs()
    t = config.OUTLIER_LOG_RATIO_THRESHOLD
    assert 0.012 < (a > t).mean() < 0.016
    # the threshold sits in an empty gap: nothing within +-0.2 of it
    assert ((a > t - 0.2) & (a < t + 0.2)).sum() == 0


def test_unseen_lane_falls_back(real_train: pd.DataFrame) -> None:
    ref = RpmReference().fit(real_train)
    row = real_train.head(1).copy()
    row["pickup"] = "Nowhere"
    assert ref.reference_level(row).iloc[0] == "equipment_band"
    assert np.isfinite(ref.reference_rpm(row)).all()
