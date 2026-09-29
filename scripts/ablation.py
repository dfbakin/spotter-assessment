"""Phase 2b feature-family ablation of the cb_level_noqs trees (default CatBoost params, seed 0).

Masks one family of config.ABLATION_FAMILIES at a time and scores F1, F2 (decision) and F3
(information only) with plain and day-debiased MAE. A family is dropped for tuning only if
masking it improves the F1+F2 day-debiased MAE by more than $1.

Writes artifacts/tuning/ablation_runs.csv (variant x fold) and ablation.csv (one row per
variant, deltas vs the full set, decision).

    .venv/bin/python scripts/ablation.py
"""

from __future__ import annotations

import pandas as pd

from freight import config
from freight.backtest import load_backtest_data
from freight.tuning import ABLATION_DROP_GAIN, run_ablation, summarise_ablation


def main() -> None:
    data = load_backtest_data()
    runs = run_ablation(data)
    config.TUNING_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.TUNING_DIR / "ablation_runs.csv", index=False)
    summary = summarise_ablation(runs)
    summary.to_csv(config.TUNING_DIR / "ablation.csv", index=False)

    cols = ["variant", "n_features"]
    for m in ("mae_debiased", "mae_clean"):
        cols += [f"{m}_F1", f"{m}_F2", f"{m}_f12", f"delta_{m}_f12", f"{m}_F3"]
    cols += ["decision"]
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 2):
        print("\nAblation (default params, seed 0); decision on F1+F2 day-debiased MAE, F3 for information")
        print(summary[cols].to_string(index=False))
    drops = summary.loc[summary["decision"] == "drop", "variant"].tolist()
    print(f"\nfamilies improving F1+F2 day-debiased MAE by > ${ABLATION_DROP_GAIN:.0f}: {drops or 'none -> keep the full set'}")


if __name__ == "__main__":
    main()
