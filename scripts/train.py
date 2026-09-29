"""Fit the final model on all clean Jan-Oct rows and save it.

L (market level) + CatBoost f with seeds 0-4 (see freight.final). Writes the CatBoost models
to artifacts/models/ (git-ignored) and artifacts/final/model_card.json.
    .venv/bin/python scripts/train.py
"""

from __future__ import annotations

import time

from freight.final import MODEL_CARD_PATH, MODELS_DIR, fit_final, save_final


def main() -> None:
    t0 = time.perf_counter()
    model = fit_final()
    save_final(model)
    card = model.card
    tr, lv = card["training"], card["level"]
    print(
        f"training rows: {tr['n_rows_used']:,} used of {tr['n_rows_window']:,} "
        f"({tr['n_excluded_corrupted']} corrupted excluded), {tr['date_range'][0]} .. {tr['date_range'][1]}"
    )
    print("L coefficients:", {k: round(v, 6) for k, v in lv["coefficients"].items()}, f"R2 {lv['r2_daily_weighted']:.3f}")
    print(f"in-sample RMSE log(rpm): {tr['in_sample_rmse_log_rpm']:.4f}")
    print(f"wrote {MODELS_DIR}/ and {MODEL_CARD_PATH}  ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
