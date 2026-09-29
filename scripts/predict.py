"""Write the deliverables from the saved final model (fits and saves it first if missing).

    validation_predictions.csv                 load_id,predicted_rate (template order, 2 decimals)
    december_chart_predictions.csv             data/december-chart-inputs.csv with predicted_rate
    artifacts/final/december_components.csv    Option A vs neutral market, L, tree part per day
    .venv/bin/python scripts/predict.py
"""

from __future__ import annotations

import pandas as pd

from freight.final import load_or_fit, write_deliverables


def main() -> None:
    model = load_or_fit()
    paths = write_deliverables(model)
    val = pd.read_csv(paths["validation"])
    dec = pd.read_csv(paths["december"])
    print(
        f"validation: {len(val):,} rows, predicted_rate min {val['predicted_rate'].min():.2f} "
        f"median {val['predicted_rate'].median():.2f} max {val['predicted_rate'].max():.2f}"
    )
    s = dec["predicted_rate"]
    print(f"December chart: min {s.min():.2f}  max {s.max():.2f}  mean {s.mean():.2f}")
    for name, path in paths.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
