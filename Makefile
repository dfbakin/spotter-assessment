PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup test outliers backtest backtest-level level-diagnostics december-preview ablation tune backtest-tuned ensemble-f3 research tabpack train predict score sanity report deck notebook all

setup:
	python3 -m venv .venv
	$(PIP) install -r requirements.txt
	$(PIP) install -e . --no-deps

test:
	$(PY) -m pytest

# Diagnostic: |log-ratio| histogram and per-fold gap bounds behind the outlier threshold.
outliers:
	$(PY) scripts/outlier_threshold.py

# E0-E6 (yardstick, cb_base, cb_lograte, cb_time, cb_keep_outliers, cb_raw_market,
# cb_cold_start), diagnostics E7-E11 (quarter-end / quote_signal time-proxy checks) and the
# phase-2a level experiments E12-E19 (see backtest-level).
backtest: outliers
	$(PY) scripts/backtest.py all

# Phase 2a: log(rpm) = L(date) + f(x) -- market level (OLS on [1, mi_day_mean, t, qe_ramp])
# as the CatBoost baseline. Runs yardstick(_level), cb_time (3 seeds), cb_level (+ _mi,
# _noqs, _notrend, _noramp, _cold, _k2), then the diagnostics and the December preview.
backtest-level:
	$(PY) scripts/backtest.py level
	$(MAKE) level-diagnostics december-preview

# Summary tables, L coefficients per fold, level_index.png (no CatBoost training).
level-diagnostics:
	$(PY) scripts/level_diagnostics.py

# Sanity preview of the December chart lane (artifacts/preview/, not the deliverable).
december-preview:
	$(PY) scripts/december_preview.py

# Phase 2b: mask one feature family of the cb_level_noqs trees at a time (default params,
# seed 0; F1+F2 decide, F3 for information) -> artifacts/tuning/ablation.csv.
ablation:
	$(PY) scripts/ablation.py

# Phase 2b: Optuna (TPE, median pruner, <= 80 trials / 75 min) on F1+F2 day-debiased MAE,
# then defaults + top-5 trials re-run with seeds 0-2 -> artifacts/tuning/best_params.json.
tune:
	$(PY) scripts/tune.py

# Phase 2b: score the chosen params once on F1/F2/F3 (cb_level_tuned vs cb_level_noqs, 3 seeds)
# and write the comparison tables (artifacts/tuning/final_*.csv).
backtest-tuned:
	$(PY) scripts/backtest.py tuned
	$(PY) scripts/tune.py summary

# For information: the final 5-seed ensemble scored on F3 -> artifacts/final/ensemble_f3.json
# (5 sequential fits, ~3 min; no decision depends on it).
ensemble-f3:
	$(PY) scripts/ensemble_f3.py

# The whole research chain in order. WARNING: takes a few hours on an 8-core laptop (the Optuna
# study alone is budgeted at 75 min per call); `make all` does not need it -- it only reads the
# committed artifacts/tuning/best_params.json.
research:
	$(MAKE) backtest
	$(MAKE) backtest-level
	$(MAKE) ablation
	$(MAKE) tune
	$(MAKE) backtest-tuned
	$(MAKE) ensemble-f3

# TabPack check (artifacts/tabpack/, RESULTS.md section 7): the 36 backtest runs (4 experiments x 3 folds
# x 3 seeds, resumable), then the summary tables and the blend with its gate. Needs the TabPack clone in
# tabpack/ (git-ignored) with its own venv tabpack/.venv, and ideally a GPU: the reported runs used the
# paper configuration on a rented RTX 5090 (38 min); the script's defaults are the laptop setup (CPU or a
# small GPU, much slower). Not part of `make all` or `make research`; the submission does not use it.
# Setup, environment knobs and the remote-GPU recipe: scripts/tabpack/README.md.
tabpack:
	bash scripts/tabpack/run_all.sh

# ---- final model and deliverables
# Fit L + CatBoost (seeds 0-4) on all clean Jan-Oct rows -> artifacts/models/ (git-ignored),
# artifacts/final/model_card.json. ~3 min.
train:
	$(PY) scripts/train.py

# validation_predictions.csv, december_chart_predictions.csv (repo root) and
# artifacts/final/december_components.csv; fits first if the saved models are missing.
predict:
	$(PY) scripts/predict.py

# The organisers' scorer: validates both files, writes scorer_results/candidate_december.png.
score:
	$(PY) score.py --predictions validation_predictions.csv --december-predictions december_chart_predictions.csv

# Sanity checks of the predictions -> artifacts/final/sanity.md, level_continuity.png,
# december_components.png.
sanity:
	$(PY) scripts/sanity.py

report:
	cd reports && quarto render report.qmd

deck:
	cd reports && quarto render deck.qmd

# Execute notebooks/analysis.ipynb in place (reads artifacts + data only, no training; ~10 s).
notebook:
	.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb

# Tests, final fit, deliverables, scorer, sanity checks (~6 min). Explicit sub-makes keep the
# order even under `make -j`.
all:
	$(MAKE) test
	$(MAKE) train
	$(MAKE) predict
	$(MAKE) score
	$(MAKE) sanity
