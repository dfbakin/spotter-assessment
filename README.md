# Freight rate prediction: Spotter MLE assessment

This repository predicts `posted_rate`, the USD price of a truck load, for 12,000 loads in November–December 2025. It
learns from 48,000 labelled loads for January–October 2025 and also produces the fixed December lane chart. The
evaluation metric is not disclosed, so several are reported. Author: Denis Bakin.

## Headline results

On the F3 holdout (Sep–Oct 2025, trained on Jan–Aug, scored once after every choice was made on earlier folds):

| Model | MAE, clean rows | MAE, all rows | MAPE |
|---|---:|---:|---:|
| Lane median × distance (yardstick) | $76.2 | $131.7 | 3.21% |
| CatBoost with a time coordinate (3 seeds) | $57.5 ± 1.9 | $113.3 | 2.35% |
| **Market level L(date) + CatBoost (chosen, 3 seeds)** | **$37.5 ± 0.1** | **$93.5** | **1.61%** |
| Chosen model, 5-seed ensemble (the submitted configuration) | $37.2 | $93.2 | 1.60% |

"Clean" excludes the ~1.5% corrupted test labels. December chart (Lexington → Fort Wayne, Dry Van, 360 mi, 32,000 lb):
$831.84 on 1 Dec to $883.92 on 31 Dec, mean $852.88, with a weekly wave, a gentle trend and a quarter-end ramp.

## Approach

- **Forward-time validation only.** Expanding folds with two-month test windows: F1 (Jan–Apr → May–Jun) and F2
  (Jan–Jun → Jul–Aug) make every choice; F3 (Jan–Aug → Sep–Oct) is scored once. A cold-start slice removes 8 whole
  cities from training, because 8 validation cities never appear in the labelled data.
- **Explicit data fixes.** 677 corrupted labels (rate ×2.5–5.5 or ×0.2–0.45 of a robust lane reference) are dropped
  from training. Sign-flipped, capped and missing weights and missing `market_index` values are repaired and flagged.
- **Level decomposition.** Plain CatBoost predicts the first test month at ~1% MAPE but under-predicts the second by
  up to 6.6%, because trees cannot extrapolate a rising price level. The model therefore splits
  `log(rate / distance) = L(date) + f(x)`:
  - `L` is a four-term OLS on the day's mean market index, a time trend and a quarter-end ramp, all known in advance;
  - `f` is CatBoost for the cross-section, with `L` as its baseline and `quote_signal` dropped.
- **Tuning changed little.** A feature-family ablation and 51 Optuna trials gained $0.34 on the selection metric,
  inside the $0.5 tie rule, at ~3× the fit time, so the CatBoost defaults were kept.
- **TabPack check (not in the submission).** TabPack (Gorishniy et al., ICML 2026, arXiv:2607.05380), a packed
  ensemble of 64 MLPs with sampled hyper-parameters and greedy online ensemble selection, ran in the paper's
  configuration under the same protocol (3 seeds, same folds): in CatBoost's slot on `log(rpm) − L`, standalone
  without `L`, and with lane identity. In CatBoost's slot its cross-section is slightly better (F3 day-debiased MAE
  $31.86 vs $32.56), but its level offset is worse, so F3 clean MAE is $39.89 vs $37.45. Standalone it fails like
  plain trees (F3 $60.36, second month −3.0%). The blend weight chosen on F1+F2 is w = 1.0 (CatBoost alone), so the
  $0.5 gate fails and the submission is unchanged: a stronger learner was worth about $1, the decomposition about
  $20. Details in `artifacts/RESULTS.md` §7.
- **Final fit.** All 47,323 clean Jan–Oct loads, 5 CatBoost seeds averaged on the log scale. December inputs use each
  day's mean `market_index` from `validation.csv` (inputs, not labels), with the weights frozen at 31 October.

## Deliverables

| File | What |
|---|---|
| `validation_predictions.csv` | `load_id,predicted_rate` for all 12,000 validation loads |
| `december_chart_predictions.csv` | the December chart inputs with `predicted_rate` filled (31 rows) |
| `scorer_results/candidate_december.png` | the December chart drawn by the organisers' `score.py` |
| `reports/_output/report.pdf` | the report: validation and split approach, results, December chart |
| `reports/_output/deck.pdf` | the 9 slides used for the 3-minute video walkthrough (`deck-long.pdf`: the full 22-page deck, for interviews) |
| `notebooks/analysis.ipynb` | the analysis notebook: EDA findings, measurements and plots |

## Setup

Tested with Python 3.12.3 on Linux (8 cores, CPU only).

```bash
make setup    # creates .venv, installs requirements.txt and the package (src/freight) in editable mode
```

**Data.** The organisers' files are not redistributed (`data/` is git-ignored). Put them in `data/` under the names as
distributed, with hyphens:

```
data/train-test.csv
data/validation.csv
data/december-chart-inputs.csv
data/validation-predictions-template.csv
```

## Reproduce the deliverables

```bash
make all      # about 6 minutes
```

`make all` runs, in order:

1. `make test`: the 59 unit tests;
2. `make train`: the final fit, written to `artifacts/models/` and `artifacts/final/model_card.json`;
3. `make predict`: the two CSV deliverables;
4. `make score`: the organisers' scorer;
5. `make sanity`: the checks in `artifacts/final/sanity.md`.

It reads the committed `artifacts/tuning/best_params.json` and does not need the research chain. The scorer step is
exactly:

```bash
.venv/bin/python score.py --predictions validation_predictions.csv --december-predictions december_chart_predictions.csv
```

## Reproduce the research

```bash
make research   # a few hours on an 8-core laptop; the Optuna study alone is budgeted at 75 minutes
```

This runs the backtests (phase 1 and the level experiments), the feature-family ablation, the Optuna study, the final
F1/F2/F3 comparison and the 5-seed ensemble on F3. Outputs go to `artifacts/backtest/` and `artifacts/tuning/`;
`artifacts/RESULTS.md` traces every reported number to its file. Individual steps have their own targets (see the
`Makefile`).

## Reproduce the TabPack check

```bash
make tabpack    # bash scripts/tabpack/run_all.sh: the 36 backtest runs, then the tables and the blend with its gate
```

It needs the TabPack reference implementation cloned into `tabpack/` (git-ignored, not redistributed) with its own
venv, `tabpack/.venv`; the main `.venv` runs the export, evaluation and blend scripts. A GPU is strongly recommended:
the reported runs used the paper configuration on a rented vast.ai RTX 5090 (36 runs in 38 minutes), because the
laptop GPU could not run it. The script's defaults are the laptop configuration and fall back to the CPU, which works
but is much slower. Runs are resumable. The setup, the environment knobs and the remote-GPU recipe are in
`scripts/tabpack/README.md`; the outputs are committed in `artifacts/tabpack/`. `make all` does not need it.

## Report, slides and notebook

```bash
make notebook   # re-executes notebooks/analysis.ipynb in place (about 10 s; reads data and artifacts only)
make report     # reports/_output/report.pdf  (needs Quarto >= 1.7 and xelatex)
make deck       # reports/_output/deck.pdf
```

See `reports/README.md` for the document sources and the house style.

## Repository structure

```
.
├── Makefile                  entry points (setup, all, research, tabpack, report, deck, notebook)
├── score.py                  the organisers' scorer
├── src/freight/              the package
│   ├── config.py             paths, folds, thresholds, feature groups
│   ├── data.py               loading and row-preserving cleaning (weight fixes + flags)
│   ├── outliers.py           RpmReference: robust lane reference, corrupted-label flags
│   ├── market.py             daily market table from input columns only
│   ├── features.py           feature matrix, Option A market inputs for the chart
│   ├── split.py              forward-time folds, cold-start city holdout
│   ├── level.py              MarketLevel: L(date) = OLS on [1, mi_day_mean, t, qe_ramp]
│   ├── models.py             yardsticks, CatBoostRate, LevelCatBoostRate (L as baseline)
│   ├── evaluate.py           metrics, month bias, day-debiased MAE, slices
│   ├── backtest.py           experiment configs and the fold runner
│   ├── tuning.py             ablation, Optuna study, re-evaluation, final comparison
│   └── final.py              final fit, deliverables, December components
├── scripts/                  thin CLI wrappers called by the Makefile
│   └── tabpack/              TabPack check: fold export, runs, evaluation, blend (README: setup, remote GPU)
├── tests/                    pytest suite (59 tests)
├── notebooks/analysis.ipynb  the analysis notebook
├── artifacts/                backtest, tuning and final-fit outputs (RESULTS.md indexes them)
│   └── tabpack/              TabPack runs, tables, blend grid and gate, final-fit comparison
├── reports/                  report.qmd, deck.qmd, TikZ/pgfplots figures, house style, _output/ PDFs
├── validation_predictions.csv
├── december_chart_predictions.csv
└── scorer_results/candidate_december.png
```

## Organisers' instructions

The assessment brief is not redistributed. The original task:

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as
   `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`. In this repository the files on
disk use hyphens, and the completed December file is `december_chart_predictions.csv` at the repository root (see
the scorer command above).

**Submit:** a GitHub repository with code, dependencies and run instructions; `validation_predictions.csv`; a PDF or
DOCX report with the validation and data split approach and `candidate_december.png`; a 2–3 minute Loom video.
