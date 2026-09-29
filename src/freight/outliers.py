"""Detection of corrupted targets.

About 1.4% of labels have a rate per mile (rpm) that is x2.5..5.5 or x0.2..0.45 of normal,
scattered randomly over time and equipment -- data-entry style corruption, not market moves.
Genuine rpm noise inside a lane x equipment x week cell is only ~2% CV, so the ratio to a
robust lane reference separates the two populations with a wide empty gap.

Leakage rule: the reference is *fitted on the fold's training rows only* (``fit`` takes them
explicitly). It is then *applied* to test rows solely to build the evaluation "clean" mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from freight import config
from freight.features import distance_band, lane_key


@dataclass
class RpmReference:
    """Hierarchical median rpm: lane x equipment -> lane -> equipment x distance band.

    Medians (not means) so that corrupted rows inside the training data cannot drag the
    reference toward themselves. Small cells (< ``min_count`` rows) fall back to a coarser
    level instead of trusting one or two possibly corrupted loads.

    ``fit`` is two-stage. A median of a 3-5 row lane cell is still broken when 2 of its rows
    are corrupted (it happens a handful of times per fold), so stage 1 screens the training
    rows against the coarse equipment x distance-band median -- which alone already
    separates the populations (genuine |log-ratio| <= 0.25, corrupted >= 0.75) -- and stage 2
    fits the fine lane-level medians on the screened rows only.
    """

    min_count: int = config.REFERENCE_MIN_COUNT
    screen_threshold: float = config.OUTLIER_LOG_RATIO_THRESHOLD
    lane_equipment_: pd.Series = field(default_factory=pd.Series, repr=False)
    lane_: pd.Series = field(default_factory=pd.Series, repr=False)
    equipment_band_: pd.Series = field(default_factory=pd.Series, repr=False)
    equipment_: pd.Series = field(default_factory=pd.Series, repr=False)
    n_train_: int = 0
    n_screened_: int = 0

    @staticmethod
    def _keys(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "lane": lane_key(df),
                "equipment": df["equipment"].astype(str),
                "band": distance_band(df["distance"]),
            },
            index=df.index,
        )

    @staticmethod
    def _median(rpm: pd.Series, keys: pd.DataFrame, by: list[str], min_count: int) -> pd.Series:
        grouped = rpm.groupby([keys[k] for k in by]).agg(["median", "size"])
        return grouped.loc[grouped["size"] >= min_count, "median"]

    def fit(self, train_rows: pd.DataFrame) -> "RpmReference":
        """Fit on labelled TRAINING rows (needs ``posted_rate`` and ``distance``)."""
        rpm = train_rows[config.TARGET] / train_rows["distance"]
        keys = self._keys(train_rows)

        # Stage 1: coarse, robust screen.
        coarse = self._median(rpm, keys, ["equipment", "band"], 1)
        coarse_ref = coarse.reindex(pd.MultiIndex.from_frame(keys[["equipment", "band"]])).to_numpy()
        keep = np.abs(np.log(rpm.to_numpy() / coarse_ref)) <= self.screen_threshold
        rpm, keys = rpm[keep], keys[keep]

        # Stage 2: hierarchical medians on screened rows.
        self.equipment_ = self._median(rpm, keys, ["equipment"], 1)
        self.equipment_band_ = self._median(rpm, keys, ["equipment", "band"], 1)
        self.lane_equipment_ = self._median(rpm, keys, ["lane", "equipment"], self.min_count)
        # Lane level pools equipment types, so express rpm relative to its equipment's
        # premium first; otherwise a Reefer-heavy lane would inflate its Dry Van reference.
        rpm_eq_adjusted = rpm / self._equipment_factor(keys)
        self.lane_ = self._median(rpm_eq_adjusted, keys, ["lane"], self.min_count)
        self.n_train_ = len(train_rows)
        self.n_screened_ = int((~keep).sum())
        return self

    def reference_rpm(self, df: pd.DataFrame) -> pd.Series:
        """Reference rpm per row, using the finest level that has enough training data."""
        keys = self._keys(df)
        le_idx = pd.MultiIndex.from_frame(keys[["lane", "equipment"]])
        eb_idx = pd.MultiIndex.from_frame(keys[["equipment", "band"]])

        ref = pd.Series(self.lane_equipment_.reindex(le_idx).to_numpy(), index=df.index)
        lane_ref = pd.Series(self.lane_.reindex(keys["lane"]).to_numpy(), index=df.index)
        ref = ref.fillna(lane_ref * self._equipment_factor(keys))

        band_ref = pd.Series(self.equipment_band_.reindex(eb_idx).to_numpy(), index=df.index)
        ref = ref.fillna(band_ref)
        ref = ref.fillna(keys["equipment"].map(self.equipment_).astype(float))
        return ref.fillna(float(self.equipment_.median()))

    def _equipment_factor(self, keys: pd.DataFrame) -> pd.Series:
        """Equipment premium relative to the median equipment type (1.0 if unseen)."""
        factor = self.equipment_ / float(self.equipment_.median())
        return keys["equipment"].map(factor).astype(float).fillna(1.0)

    def reference_level(self, df: pd.DataFrame) -> pd.Series:
        """Which hierarchy level served each row (for diagnostics)."""
        keys = self._keys(df)
        le = pd.MultiIndex.from_frame(keys[["lane", "equipment"]]).isin(self.lane_equipment_.index)
        ln = keys["lane"].isin(self.lane_.index).to_numpy()
        level = np.where(le, "lane_equipment", np.where(ln, "lane", "equipment_band"))
        return pd.Series(level, index=df.index)

    def log_ratio(self, df: pd.DataFrame) -> pd.Series:
        """log(rpm / reference rpm) for labelled rows."""
        rpm = df[config.TARGET] / df["distance"]
        return np.log(rpm / self.reference_rpm(df))


def flag_corrupted(
    reference: RpmReference,
    rows: pd.DataFrame,
    threshold: float = config.OUTLIER_LOG_RATIO_THRESHOLD,
) -> pd.Series:
    """Boolean mask of rows whose |log(rpm / reference)| exceeds ``threshold``."""
    return reference.log_ratio(rows).abs() > threshold


def fit_and_flag(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame | None = None,
    threshold: float = config.OUTLIER_LOG_RATIO_THRESHOLD,
) -> tuple[RpmReference, pd.Series, pd.Series | None]:
    """Fit the reference on ``train_rows`` only, then flag train (and optionally test) rows.

    Returns ``(reference, train_flags, test_flags)``. ``test_flags`` is meant exclusively
    for the evaluation clean mask; it must never feed back into model fitting.
    """
    reference = RpmReference().fit(train_rows)
    train_flags = flag_corrupted(reference, train_rows, threshold)
    test_flags = None if test_rows is None else flag_corrupted(reference, test_rows, threshold)
    return reference, train_flags, test_flags
