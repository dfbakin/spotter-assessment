"""Inspect the |log(rpm / reference)| distribution and justify the outlier threshold.

For every fold the reference is fitted on that fold's training rows only, then applied to
its training and test rows. Prints the largest "genuine" and smallest "corrupted" value on
each side of the gap and the flag counts; saves a histogram of the full-period training
distribution to artifacts/backtest/outlier_log_ratio_hist.png and a JSON with the numbers.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from freight import config
from freight.data import clean, load_train
from freight.outliers import RpmReference
from freight.split import get_folds


def gap_bounds(abs_ratio: np.ndarray, threshold: float) -> tuple[float, float]:
    """Largest value below and smallest value above the threshold."""
    return float(abs_ratio[abs_ratio <= threshold].max()), float(abs_ratio[abs_ratio > threshold].min())


def main() -> None:
    t = config.OUTLIER_LOG_RATIO_THRESHOLD
    train = clean(load_train())
    summary: dict[str, dict] = {"threshold": t}

    for fold in get_folds():
        tr, te = train[fold.train_mask(train)], train[fold.test_mask(train)]
        ref = RpmReference().fit(tr)
        for part, rows in [("train", tr), ("test", te)]:
            a = ref.log_ratio(rows).abs().to_numpy()
            lo, hi = gap_bounds(a, t)
            key = f"{fold.name}_{part}"
            summary[key] = {"n": len(rows), "flagged": int((a > t).sum()), "genuine_max": lo, "corrupted_min": hi}
            print(f"{key:9s} n={len(rows):6d} flagged={int((a > t).sum()):4d} ({100 * (a > t).mean():.2f}%)  gap=({lo:.3f}, {hi:.3f})")

    full_ref = RpmReference().fit(train)
    a = full_ref.log_ratio(train).abs().to_numpy()
    lo, hi = gap_bounds(a, t)
    summary["all_train"] = {"n": len(train), "flagged": int((a > t).sum()), "genuine_max": lo, "corrupted_min": hi}
    print(f"all_train n={len(train)} flagged={int((a > t).sum())} gap=({lo:.3f}, {hi:.3f})")

    config.BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    (config.BACKTEST_DIR / "outlier_threshold.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.hist(a, bins=np.arange(0, 2.0, 0.025), color="#2a6f97", edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.axvline(t, color="#333333", linewidth=1.2, linestyle="--")
    ax.axvspan(lo, hi, color="#999999", alpha=0.12, linewidth=0)
    ax.text(t + 0.02, ax.get_ylim()[1] * 0.3, f"threshold log(1.6) = {t:.3f}", fontsize=8, color="#333333")
    ax.set_xlabel("|log(rpm / lane reference rpm)|  (Jan-Oct training rows)")
    ax.set_ylabel("rows (log scale)")
    ax.set_title(
        f"Bimodal residual: {int((a > t).sum())} of {len(a):,} rows flagged; empty gap {lo:.2f}-{hi:.2f}",
        loc="left",
        fontsize=10,
    )
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = config.BACKTEST_DIR / "outlier_log_ratio_hist.png"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
