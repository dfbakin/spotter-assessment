"""Run named backtest experiments over forward-time folds.

Examples:
    .venv/bin/python scripts/backtest.py cb_base
    .venv/bin/python scripts/backtest.py yardstick cb_base --folds F1 F2
    .venv/bin/python scripts/backtest.py all
    .venv/bin/python scripts/backtest.py level     # phase-2a group (market level + CatBoost)
"""

from __future__ import annotations

import argparse

from freight.backtest import EXPERIMENT_GROUPS, EXPERIMENTS, load_backtest_data, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "experiments",
        nargs="+",
        help=f"experiment names, a group ({', '.join(EXPERIMENT_GROUPS)}) or 'all'; known: {', '.join(EXPERIMENTS)}",
    )
    parser.add_argument("--folds", nargs="+", default=None, help="restrict to these folds (e.g. F1 F2)")
    args = parser.parse_args()

    if args.experiments == ["all"]:
        names = list(EXPERIMENTS)
    else:
        names = [n for arg in args.experiments for n in EXPERIMENT_GROUPS.get(arg, (arg,))]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        parser.error(f"unknown experiment(s) {unknown}; known: {list(EXPERIMENTS)}")

    data = load_backtest_data()  # built once, shared by all experiments
    for name in names:
        run_experiment(EXPERIMENTS[name], data=data, folds=args.folds)


if __name__ == "__main__":
    main()
