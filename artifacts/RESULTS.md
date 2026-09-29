# Results log

This is a running log of the key backtest tables. The notebook and the report are written from it. Every number
comes from the artifact file named under each table (paths are relative to `artifacts/`).

- **Model.** `log(rpm) = L(date) + f(x)`, where L is the market level (OLS on [1, mi_day_mean, t, qe_ramp]) and f is
  a cross-sectional CatBoost.
- **Folds.** Expanding windows with 2-month test periods:
  - F1: Jan–Apr → May–Jun
  - F2: Jan–Jun → Jul–Aug
  - F3: Jan–Aug → Sep–Oct
- **Protocol.** Selection uses F1+F2 only. F3 is scored once with the final choice.
- **Metrics.**
  - MAE is in $. "Clean" means corrupted test labels are excluded; "all" means every test row.
  - Month bias = median(pred / actual) − 1 on clean rows, per test month.
  - "±" is the sd across CatBoost seeds.

## 1. Phase 1: plain CatBoost (single seed)

| experiment | F1 | F2 | F3 | F3 MAPE | F3 bias Sep / Oct |
|---|---|---|---|---|---|
| yardstick (lane × equipment median rpm × distance) | 148.8 | 83.8 | 76.2 | 3.21% | −0.65% / −1.05% |
| cb_base (no time coordinate) | 80.9 | 69.0 | 88.5 | 4.52% | −2.47% / −6.49% |
| cb_lograte | 63.6 | 58.4 | 91.4 | 4.43% | |
| cb_time (+ t_days) | 87.6 | 66.8 | 53.8 | 2.17% | |
| cb_keep_outliers | 151.8 | 61.5 | 110.0 | 4.66% | |
| cb_raw_market | 55.7 | 64.6 | 72.8 | 3.18% | |

Source: MAE clean from `backtest/summary.csv` (the latest run of E0–E5 with fold keys F1/F2/F3). The cb_base month
bias comes from `backtest/cb_base.json` → `folds.F3.slices.month`.

The trees predict the first test month well and under-predict the second by 5–7%. They cannot extrapolate the rising
price level, and month proxies make it worse.

## 2. Phase 2a: market level L + cross-sectional trees

MAE clean (mean ± sd over seeds; n = number of seeds):

| experiment | n | F1 | F2 | F3 | F1+F2 | F3 all | F3 MAPE |
|---|---|---|---|---|---|---|---|
| yardstick_level (reference × exp(L)) | 1 | 82.2 | 83.3 | 74.1 | 82.7 | 129.6 | 3.13% |
| cb_time | 3 | 83.1 ± 2.8 | 67.1 ± 3.0 | 57.5 ± 1.9 | 75.1 | 113.3 | 2.35% |
| cb_level | 3 | 49.4 ± 0.3 | 60.6 ± 0.5 | 34.8 ± 0.1 | 55.0 | 90.9 | 1.50% |
| cb_level_mi | 3 | 43.9 ± 1.1 | 62.6 ± 0.7 | 49.3 ± 0.3 | 53.2 | 105.2 | 2.02% |
| **cb_level_noqs** (selected) | 2 | 50.6 ± 0.1 | 50.5 ± 0.5 | 37.5 ± 0.1 | **50.6** | 93.5 | 1.61% |
| cb_level_notrend | 2 | 62.2 ± 0.0 | 52.0 ± 0.8 | 83.5 ± 0.5 | 57.1 | 138.9 | 3.55% |
| cb_level_noramp | 2 | 46.8 ± 0.2 | 87.5 ± 0.6 | 40.1 ± 0.0 | 67.1 | 96.2 | 1.71% |
| cb_level_k2 (K = 2 backfitting) | 1 | 40.5 | 75.6 | 34.8 | 58.0 | 90.9 | 1.50% |

Median bias by test month for cb_level_noqs:

| fold | first month | second month |
|---|---|---|
| F1 | −1.05% | −2.12% |
| F2 | +0.52% | +2.56% |
| F3 | −0.68% | −1.12% |

- Cold start: cb_level_cold, F3, 2 city draws, gives 35.6 overall; cold rows 40.3 vs warm rows 34.4.
- Source: `backtest/level_summary.csv`, `backtest/level_month_bias.csv`.

With L owning the level, the second-month collapse disappears. Most of the remaining F1/F2 error is L's level
extrapolation. The trend and the quarter-end ramp both matter (notrend F3 83.5; noramp F2 87.5).

## 3. Phase 2b: feature-family ablation (cb_level_noqs, default params, seed 0)

The selection instrument is the **day-debiased MAE**: clean MAE after removing each test day's median log error. This
removes L's day-level error and leaves the cross-sectional error that the trees own. It is used for selection only
and is not a performance number.

| masked family | # features | debiased F1 | debiased F2 | debiased F1+F2 | Δ vs full | plain F1+F2 | Δ plain | debiased F3 (info) |
|---|---|---|---|---|---|---|---|---|
| none (full) | 24 | 34.42 | 31.43 | 32.92 | – | 50.45 | – | 32.50 |
| city: pickup, delivery | 22 | 34.60 | 31.13 | 32.87 | −0.06 | 50.54 | +0.09 | 32.31 |
| lane | 23 | 34.46 | 31.39 | 32.92 | −0.00 | 50.57 | +0.11 | 32.39 |
| coordinates + deltas + bearing | 17 | 36.49 | 33.18 | 34.84 | +1.91 | 51.76 | +1.31 | 34.14 |
| haversine + circuity | 22 | 34.47 | 31.20 | 32.83 | −0.09 | 50.44 | −0.02 | 32.43 |
| weight + 4 flags | 19 | 60.81 | 58.22 | 59.51 | **+26.60** | 70.49 | +20.03 | 59.44 |
| dow, dom | 22 | 35.05 | 31.40 | 33.23 | +0.30 | 50.82 | +0.37 | 33.07 |
| mi_dev + mi_missing | 22 | 34.90 | 31.78 | 33.34 | +0.41 | 50.58 | +0.13 | 32.98 |

Source: `tuning/ablation.csv` (per-fold runs in `tuning/ablation_runs.csv`, log in `tuning/ablation.log`).

- No family improves F1+F2 debiased MAE by more than $1, so the full 24-feature set is kept for tuning.
- Weight is by far the most valuable family (+$27 debiased when masked).
- Geometry via coordinates is second (+$1.9).
- City, lane and crow-fly distance are mutually redundant (Δ ≈ 0 each).
- The full-set run reproduces phase-2a cb_level_noqs F1_r0 exactly (MAE clean 50.735). This confirms CatBoost is
  deterministic for a fixed seed.

## 4. Phase 2b: Optuna tuning of f (cb_level_noqs, all 24 features)

- **Study.** `cb_level_noqs_f12_day_debiased`: TPE (seed 0) with a median pruner (F1 is step 0). The objective is
  the mean F1+F2 day-debiased clean MAE with CatBoost seed 0. The CatBoost defaults were enqueued as trial 0.
- **Trials.** 51 in total: 32 COMPLETE, 18 PRUNED and 1 FAIL. Trial #40 was orphaned by a power outage and marked
  failed on resume. The study's reported wall clock (909.7 min) and trial time (909.4 min) include the outage:
  trial #40 alone accounts for 834.2 min, so the other 50 trials took 75.2 min, the configured budget.
- **Best in the study.** Trial #36 scored 32.55 (seed 0), against 32.92 for the defaults (trial 0).

Re-evaluation of the defaults and the top 5 trials with seeds 0–2 on F1+F2. Debiased values are mean ± sd over
seeds; fit time is seconds per fold.

| candidate | depth | lr | iterations | bootstrap | debiased F1+F2 | plain clean F1+F2 | fit s |
|---|---|---|---|---|---|---|---|
| trial_36 | 8 | 0.021 | 3500 | Bernoulli | **32.57 ± 0.06** | 50.53 | 65.5 |
| trial_41 | 8 | 0.021 | 3100 | Bernoulli | 32.57 ± 0.05 | 50.58 | 64.7 |
| trial_46 | 8 | 0.026 | 2400 | Bernoulli | 32.63 ± 0.07 | 50.61 | 44.4 |
| trial_27 | 6 | 0.059 | 3800 | MVS | 32.64 ± 0.08 | 50.61 | 50.3 |
| trial_35 | 8 | 0.021 | 3800 | MVS | 32.65 ± 0.06 | 50.58 | 81.7 |
| default | 6 | 0.05 | 2000 | MVS | 32.91 ± 0.05 | 50.63 | **23.7** |

Source: `tuning/reeval.csv` (per-seed runs in `tuning/reeval_runs.csv`); trial states and durations in
`tuning/trials.csv`; the decision in `tuning/best_params.json` → `selection`.

- **Decision: keep the defaults.** The best tuned candidate gains only $0.34 day-debiased and $0.10 plain clean
  F1+F2. Both are inside the $0.5 tie rule, and tuned fits are ~3× slower (65.5 vs 23.7 s per fold).
- All top trials go deeper (8), use a slower learning rate and a lower random_strength (≈ 0.1). The search found
  no configuration that changes the error meaningfully, because L's level error dominates plain MAE.

## 5. Final comparison on F1 / F2 / F3

The final configuration is `cb_level_tuned`: the cb_level_noqs trees with `best_params.json`. Because the chosen
params are the defaults, it reproduces cb_level_noqs run for run (identical rows in the source files). MAE is in $,
mean ± sd over seeds.

| model | n | F1 | F2 | F3 | F3 all | F3 MAPE | F3 debiased |
|---|---|---|---|---|---|---|---|
| yardstick_level | 1 | 82.2 | 83.3 | 74.1 | 129.6 | 3.13% | 71.7 |
| cb_level (with qs_dev; not selected) | 3 | 49.4 ± 0.3 | 60.6 ± 0.5 | 34.8 ± 0.1 | 90.9 | 1.50% | 29.4 |
| **final config** (= cb_level_noqs, default params) | 3 | 50.7 ± 0.2 | 50.5 ± 0.4 | **37.5 ± 0.1** | 93.5 | 1.61% | 32.6 |
| final config, 5-seed ensemble (information only) | 5 | – | – | 37.19 | 93.24 | 1.60% | 32.35 |

- Ensemble reference: the single seeds 0–4 on F3 score 37.38 ± 0.13. The ensemble is $0.19 better, which is the
  expected small variance reduction; no decision depends on it.
- Seeds 0–2 reproduce the recorded F3 runs exactly.
- Source: `tuning/final_comparison.csv`; ensemble in `final/ensemble_f3.json` (`scripts/ensemble_f3.py`).

Median bias by test month, final config (mean over 3 seeds; the ensemble's F3 values in brackets):

| fold | first month | second month |
|---|---|---|
| F1 (May / Jun) | −1.07% | −2.14% |
| F2 (Jul / Aug) | +0.52% | +2.55% |
| F3 (Sep / Oct) | −0.68% [−0.69%] | −1.12% [−1.10%] |

Source: `tuning/final_month_bias.csv` (per seed), `final/ensemble_f3.json` → `ensemble.month`.

F3 slices, clean rows, mean over 3 seeds:

| equipment | n | MAE | MAPE | median bias |
|---|---|---|---|---|
| Dry Van | 5,275 | 33.4 | 1.52% | −0.73% |
| Flatbed | 1,746 | 47.6 | 1.97% | −1.46% |
| Reefer | 2,358 | 39.1 | 1.55% | −1.00% |

| distance band | 0–150 | 150–300 | 300–500 | 500–750 | 750–1000 | 1000–1500 | 1500–2000 | 2000–2500 | 2500+ |
|---|---|---|---|---|---|---|---|---|---|
| MAE ($) | 6.0 | 11.5 | 16.4 | 23.3 | 30.0 | 41.3 | 55.6 | 67.2 | 79.9 |
| MAPE | 1.85% | 1.84% | 1.66% | 1.63% | 1.56% | 1.61% | 1.55% | 1.53% | 1.48% |
| median bias | −1.57% | −1.29% | −1.12% | −0.96% | −0.87% | −0.84% | −0.79% | −0.78% | −0.76% |

Source: `tuning/final_slices_f3.csv`.

- F3 MAE is half the level-aware yardstick's. The error is a flat ~1.6% of the rate across distance, so dollar MAE
  grows with distance. Flatbed is the hardest equipment type.
- The residual month bias is L's extrapolation error (≤ 2.6% on F1/F2, ~1% on F3).
- cb_level would have been $2.6 better on F3, but it lost the F1+F2 selection (55.0 vs 50.6). Per the protocol,
  F3 does not reverse the choice.

## 6. Final fit, predictions and December chart

**Final fit** (source: `final/model_card.json`, written by `scripts/train.py`):

- **Training rows.** 47,323 of the 48,000 Jan–Oct rows (2025-01-01 .. 2025-10-31). 677 corrupted targets are
  excluded: |log(rpm / RpmReference)| > log 1.6 = 0.470, with the reference fitted on all Jan–Oct.
- **L.** const −0.1584, mi_day_mean +0.1192, t +0.000168 per day (+0.51% per 30 days), qe_ramp +0.0377 (+3.85% on
  a quarter's last day). Weighted R² is 0.959 over 304 days. This is identical to the all-row fit in
  `backtest/level_coefficients.json` → `folds.ALL`.
- **f.** CatBoost defaults: depth 6, lr 0.05, 2000 iterations, l2 3, MVS with subsample 0.8, 254 borders. It uses
  the 24 tree features of `best_params.json`. Seeds 0–4 are averaged on the log scale; each fit takes 35–37 s. The
  in-sample RMSE of log(rpm) is 0.0167.
- **Persistence.** Saved models reload bit-exactly (max |Δf| = 0).
- **Scorer.** `score.py` validates both files (12,000 predictions; 31 December rows) and writes
  `scorer_results/candidate_december.png`.

**Prediction sanity** (source: `final/sanity.md`, `final/level_continuity.png`; `scripts/sanity.py`):

| check | result |
|---|---|
| validation file | 12,000 rows, 0 NaN, all > 0; $188.86 – $7,033.20, median $2,068.84 |
| median rpm vs Sep–Oct actuals | ratios by equipment 1.002–1.010 and by distance band 0.997–1.013; rpm / lane reference 1.0143 predicted vs 1.0007 actual (+1.36%) |
| level continuity | daily level index: Oct actual +0.61%, Nov predicted +0.46%, Dec predicted +2.32%, where L's monthly means are +0.41% / +0.47% / +2.20%; the Dec late-month lift is +3.41%, vs +3.14% / +2.60% / +1.94% actual in Mar / Jun / Sep |
| weekly wave | amplitude 2.29% predicted vs 1.93% actual. The input wave is unchanged; L's single mi_day_mean slope overstates the within-week swing by ≈ ±0.16% |
| 8 unseen cities | 1,447 rows (12.1%). Cold / warm median rpm by band is 0.955–1.017; per-city relative rpm is 0.982–1.031, inside the warm cities' 0.969–1.036 |
| vs yardstick_level | r = 0.998 (rates), median ratio 1.0016; 43 rows (0.36%) differ by > 10%, the largest being a thin 4-row Flatbed lane (−12%) |
| edge cases (pred / yardstick vs history) | capped weight 1.033 vs 1.029; floor 0.960 vs 0.963; flipped 1.001 vs 1.004; missing weight 0.997 vs 1.000; missing market_index 1.004 vs 1.000 |
| Dec Lexington → Fort Wayne loads (3) | 3.5–4.8% below the chart per mile (lighter loads, 16.7–26.1k lb, plus their own market_index deviation). With the chart's inputs, each reproduces the chart value exactly |

- Nothing implausible was found.
- The one systematic quirk is the slightly amplified weekly wave (±0.16%), which is far below the ~1.6% error level.

**December chart** (Lexington → Fort Wayne, Dry Van, 360 mi, 32,000 lb; source: `december_chart_predictions.csv`,
`final/december_components.csv`, `final/sanity.md` §7):

- **Range.** Option A (the deliverable) runs from $831.84 (Mon Dec 1) to $883.92 (Wed Dec 31), mean $852.88.
- **Shape.** A weekly wave from the market index (Thu +1.2% / Sun −1.1% vs neutral, ≈ $19 amplitude) rides on
  the trend and the quarter-end ramp. On the neutral curve, Dec 25–31 is +2.88% above Dec 1–15.
- **Option A vs neutral.** Neutral means every December mi_day_mean is set to the December average. The two curves
  stay within ±$11.4 (±1.3%) of each other, with near-equal means ($852.88 vs $852.84).
- **vs lane history.** The chart sits 5.7% above the lane's Sep–Oct median per 360 mi ($807.01, n = 4). Most of the
  gap is weight: the lane's loads are lighter (median 24.7k lb in Sep–Oct), and at that weight the chart mean
  would be $825.31.

## 7. TabPack

TabPack trains a pack of 64 MLPs in parallel. Each MLP has its own sampled hyper-parameters
and early stopping, and an online greedy ensemble combines them. It was run under the same
forward-time protocol as CatBoost, with one difference: its early-stopping and ensemble
validation set is the last 28 days of each training window, which the network does not train
on. Selection is on F1+F2 only, and F3 is reported once. All TabPack paths below are relative
to `artifacts/tabpack/`; the scripts and the remote recipe are described in
`scripts/tabpack/README.md`.

### Environment and cost

- **Hardware.** A rented vast.ai RTX 5090 (32 GB, Blackwell sm_120; driver 580.95.05). The
  laptop (MX350, 7 GB RAM) crashed twice under the load.
- **TabPack venv.** torch 2.7.1+cu128 (CUDA 12.8). This is the lock's torch version, rebuilt
  for sm_120; the locked PyPI build (CUDA 12.6) has no Blackwell kernels. Python 3.12.14;
  numpy 2.3.1, scipy 1.16.0 and scikit-learn 1.7.0, as in the lock.
- **Main venv.** catboost 1.2.10 and pandas 2.3.3, the same as the laptop; all 59 tests pass
  there.
- Source: `runs/*.json` → `runtime` (device, gpu, torch, wall time, epochs).

| experiment | runs | wall min, mean (range) | epochs, mean | s / epoch | peak CUDA alloc / reserved |
|---|---|---|---|---|---|
| tp_level | 9 | 0.71 (0.55–0.99) | 75 | 0.32–0.78 | 4.5 / 11.3 GB |
| tp_standalone | 9 | 0.98 (0.77–1.33) | 108 | 0.35–0.85 | 5.9 / 16.7 GB |
| tp_level_lanetopk | 9 | 0.73 (0.49–1.12) | 73 | 0.32–0.85 | 4.9 / 12.0 GB |
| tp_level_lanefull | 9 | 1.33 (0.79–2.11) | 75 | 0.67–1.24 | 19.6 / 28.8 GB |
| tp_level, FINAL fit | 3 | 1.13 (1.02–1.23) | 64 | 0.97–1.05 | 5.4 / 12.9 GB |

- **Wall clock.** The 36 backtest runs, with exports and evaluation, took 38 minutes
  (`run_all.log`, 16:46:57 → 17:25:05), and the final fit took another 4 minutes.
- **Epoch cost.** Seconds per epoch grow with the training window: 0.44 on F1, 0.67 on F2 and
  0.90 on F3 on average.
- **Memory.** Peak host RSS is at most 1.91 GB.
- **Ensemble stop.** A run stops once the online ensemble stops improving (patience 32), so
  50–64 of the 64 models had early-stopped by then. This is stock TabPack behaviour.
- **Sources.** Peak CUDA memory is in the git-ignored `tabpack/local/runs/<run>/driver.json`;
  the reserved figure is also in `runs/*.json`.

### Configuration

The configuration mirrors the paper's `tabpack/experiments/tabpack-cosine/make.py`:

- **Pack.** 64 models with `d_block` 384, batch 512, `amp_dtype` bfloat16, and Muon's
  Newton–Schulz iteration in bfloat16 (stock TabPack).
- **Model.** Periodic (`CosineEmbeddingsPack`) numerical embeddings and the MuonAdamW
  optimiser. Per model: 1–3 blocks, dropout, `d_embedding`, lr, weight decay and Muon lr are
  sampled.
- **Early stopping.** Patience 16 per model. The greedy online ensemble has patience 32 and
  at most 32 members.
- **Evaluation.** The eval batch is 32,768, the stock default; no out-of-memory fallback
  occurred.
- **Seeds.** 0, 1 and 2 for every experiment.

Source: `configs/*.json`.

| experiment | target | features |
|---|---|---|
| tp_standalone | log(rpm), no L | phase-1 `cb_time` set without `lane` (raw market columns, calendar, `t_days`, geography, weight, cities) |
| tp_level | log(rpm) − L(date) | the 24 tree features minus `lane` |
| tp_level_lanetopk | as tp_level | + `lane` as the top 155 inner-train lanes + other |
| tp_level_lanefull | as tp_level | + `lane` one-hot over every inner-train lane (3,718 / 3,922 / 3,984 on F1 / F2 / F3) |

### Table A: per experiment and fold

MAE clean is the mean ± sd over 3 seeds of the single-seed scores. The other columns are
seed means. The bias columns are the median bias in the two test months.

| experiment | fold | MAE clean | MAE all | MAPE | debiased | bias m1 / m2 |
|---|---|---|---|---|---|---|
| cb_time | F1 / F2 / F3 | 83.09 ± 2.81 / 67.15 ± 3.01 / 57.49 ± 1.89 | 136.28 / 118.78 / 113.29 | 3.75 / 3.21 / 2.35% | 78.56 / 70.01 / 31.80 | −0.46/−7.23, −0.34/−1.04, −1.50/−2.70% |
| cb_level_noqs | F1 | 50.75 ± 0.23 | 104.42 | 1.99% | 34.43 | −1.07% / −2.14% |
| | F2 | 50.52 ± 0.37 | 102.48 | 2.14% | 31.39 | +0.52% / +2.55% |
| | F1+F2 | 50.63 | 103.45 | 2.06% | 32.91 | – |
| | F3 | 37.45 ± 0.10 | 93.50 | 1.61% | 32.56 | −0.68% / −1.12% |
| tp_standalone | F1 | 39.65 ± 2.66 | 93.46 | 1.63% | 36.56 | −0.28% / −1.49% |
| | F2 | 66.74 ± 0.66 | 118.37 | 3.26% | 67.33 | −1.08% / −1.68% |
| | F1+F2 | 53.20 | 105.92 | 2.44% | 51.94 | – |
| | F3 | 60.36 ± 3.66 | 116.16 | 2.48% | 29.23 | −1.49% / −2.98% |
| tp_level | F1 | 45.73 ± 0.45 | 99.45 | 1.83% | 34.17 | −0.76% / −1.96% |
| | F2 | 67.42 ± 2.17 | 119.14 | 2.87% | 30.69 | +1.67% / +3.60% |
| | F1+F2 | 56.57 | 109.29 | 2.35% | 32.43 | – |
| | F3 | 39.89 ± 0.21 | 95.90 | 1.71% | 31.86 | −1.07% / −1.35% |
| tp_level_lanetopk | F1 | 45.29 ± 0.36 | 99.02 | 1.82% | 34.25 | −0.73% / −1.93% |
| | F2 | 67.38 ± 3.42 | 119.09 | 2.89% | 30.79 | +1.70% / +3.61% |
| | F1+F2 | 56.33 | 109.06 | 2.35% | 32.52 | – |
| | F3 | 40.22 ± 0.07 | 96.22 | 1.72% | 31.87 | −1.09% / −1.35% |
| tp_level_lanefull | F1 | 45.56 ± 0.20 | 99.30 | 1.81% | 34.09 | −0.73% / −1.93% |
| | F2 | 66.57 ± 2.26 | 118.28 | 2.87% | 30.68 | +1.67% / +3.58% |
| | F1+F2 | 56.06 | 108.79 | 2.34% | 32.38 | – |
| | F3 | 40.13 ± 0.20 | 96.13 | 1.71% | 31.69 | −1.10% / −1.35% |

Source: `table_a.csv` (per run in `runs_summary.csv`). The CatBoost rows are re-scored from
`backtest/preds` with the same metric code and match §2 and §5.

- **Level vs cross-section.** On debiased MAE, tp_level beats CatBoost on every fold (−0.26,
  −0.70, −0.70). On plain MAE it wins only on F1, because on F2 its month bias sits about
  1.1 pp above CatBoost's, which already shares L's +0.5 / +2.6% error. A plausible cause,
  not tested, is that TabPack's f never trains on the last 4 weeks of the window.
- **Standalone.** Without L, TabPack is the best model on F1 (39.65) and the worst on F2 and
  F3. Its F2 debiased MAE (67.33) shows that the cross-section itself breaks when the trend is
  learned from `t_days`. This is the lesson of §1 → §2 again, now for a different learner.
- **Lane variants.** Top-155 and full one-hot lanes change F1+F2 by −0.24 and −0.51, inside
  the seed sd on F2 (2.2–3.4), and F3 by +0.33 and +0.24. Lane adds nothing, as in the §3
  ablation.
- **Seed noise (level experiments).** The clean-MAE sd is small on F1 and F3 (≤ 0.45) but
  large on F2 (2.2–3.4), where the level offset varies by seed. The debiased sd stays at or
  below 0.67. tp_standalone is noisier (clean sd up to 3.66).

### Table B: blend grid, `w · log p_CB + (1 − w) · log p_TP` (CatBoost = cb_level_noqs, TabPack = tp_level, seed means)

| w | F1 clean | F2 clean | **F1+F2 clean** | F1+F2 debiased | F3 clean (info) | F3 debiased (info) |
|---|---|---|---|---|---|---|
| 0.0 (TabPack) | 45.65 | 67.20 | 56.43 | 32.12 | 39.80 | 31.76 |
| 0.2 | 46.36 | 63.08 | 54.72 | 31.92 | 39.01 | 31.63 |
| 0.4 | 47.21 | 59.30 | 53.25 | **31.88** | 38.36 | 31.61 |
| 0.5 | 47.68 | 57.54 | 52.61 | 31.92 | 38.09 | 31.67 |
| 0.6 | 48.20 | 55.88 | 52.04 | 32.00 | 37.85 | 31.74 |
| 0.8 | 49.32 | 52.89 | 51.11 | 32.25 | 37.49 | 31.99 |
| 0.9 | 49.92 | 51.58 | 50.75 | 32.43 | 37.38 | 32.17 |
| **1.0 (CatBoost)** | 50.55 | 50.37 | **50.46** | 32.64 | **37.30** | 32.37 |

Source: `blend_grid.csv` (every w in steps of 0.1), `blend.json`.

**Selection and gate**

- **Chosen w.** w = 1.0, i.e. CatBoost alone (`blend.json` → `chosen_w`). F1+F2 clean MAE
  falls monotonically as w rises.
- **Gate.** The blend gains $0.00 over CatBoost on F1+F2, so the final-fit gate (> $0.5)
  **fails** (`final_fit_gate`).
- **Bootstrap.** At the chosen w the day-block bootstrap and the seed-pair spread are
  trivially 0 (`bootstrap_blend_minus_cb`, `seed_pairs_gain_summary`).
- **Reference values.** At w = 1.0 the 3-seed CatBoost log-mean scores 50.46 on F1+F2 and
  37.30 on F3, against 50.63 and 37.45 for the single-seed means in Table A.

**F3 comparison** (scored once): CatBoost 37.30, tp_level 39.80, blend at the chosen w 37.30,
50/50 blend 38.09 (`per_fold_at_w.F3`).

**Cross-sectional gain (information only).**

- **Debiased optimum.** Debiased MAE would pick w = 0.4.
- **Gain at w = 0.4.** Against CatBoost alone, debiased MAE changes by −$0.76 on F1+F2
  (day-block bootstrap 95% CI [−0.98, −0.56]) and by −$0.76 on F3 ([−0.99, −0.51])
  (`bootstrap_debiased_blend_minus_cb_info`).
- **Reading.** The cross-sectional gain is real and holds on F3, but small. Plain MAE
  cannot see it, because L's shared extrapolation error dominates and TabPack adds its own
  level offset.

**Other partners (information only; not pre-specified, never selectable).**

- **Lane partners.** The lanetopk and lanefull partners give the same picture: best clean w
  = 1.0, debiased optimum at 0.3–0.4 (`blend_grid.csv`).
- **Standalone partner.**
  - At w = 0.6, cb + tp_standalone scores 42.52 on F1+F2, $7.9 better than CatBoost.
  - On F3 it scores 43.20, against 37.30.
  - The F1+F2 gain is two opposite level errors cancelling: CatBoost's +0.5 / +2.6% on F2
    against standalone's −1.1 / −1.7%. It is not a better model, and F3 shows it does not
    carry forward.

### Slices (clean MAE, $)

| fold | lane band (inner-train count) | n | CatBoost | tp_level | tp_level_lanefull | tp_standalone |
|---|---|---|---|---|---|---|
| F1 | unseen | 172 | 50.60 | 45.38 | 45.90 | 44.38 |
| F1 | 1–4 | 3,466 | 52.54 | 46.69 | 46.58 | 40.10 |
| F1 | 5–19 | 5,923 | 49.34 | 44.99 | 44.76 | 38.21 |
| F2 | unseen | 52 | 53.43 | 72.04 | 68.30 | 61.15 |
| F2 | 5–19 | 7,505 | 49.72 | 66.33 | 65.52 | 65.40 |
| F3 | unseen | 21 | 36.74 | 38.71 | 38.73 | 55.87 |
| F3 | 1–4 | 867 | 39.55 | 42.16 | 42.41 | 58.18 |
| F3 | 5–19 | 7,198 | 37.51 | 39.84 | 40.12 | 60.80 |
| F3 | 20–49 | 1,293 | 34.62 | 37.98 | 38.28 | 57.95 |

Source: `blend_slices.csv` (also equipment, distance band and month; models catboost,
tp_level, blend_w1.0 and the three variants).

- **Lane frequency.** No lane-frequency band reverses the fold-level ranking. Rare and
  unseen lanes follow the overall pattern, and the full-lane one-hot does not help them.
- **City status.** Every F1–F3 test row is "warm" (`city_status` slice): the 8 cities never
  seen in training occur only in Nov–Dec, so the backtests cannot score cold cities.

### Final fit

- **TabPack runs.** tp_level was trained on all clean Jan–Oct rows (43,038 train, plus the
  last 28 days of October, 4,285 rows, as validation), with seeds 0–2. Test rows: the 12,000
  Nov–Dec rows plus 31 December chart rows. Files: `final/runs/`, `final/preds/`,
  `final/configs/`.
- **Blend and gate.** `blend.py --final` blends at the chosen w = 1.0. It records the gate as
  failed and does not refuse; it writes only under `final/` (`final/final_summary.json`).
- **Blend output = submission.** The blended files (`final/validation_predictions_blend.csv`,
  `final/december_chart_predictions_blend.csv`) are byte-identical to the submitted root CSVs.
  The recomputed CatBoost predictions reproduce the submission exactly (max |Δ| = 0).
- **TabPack alone vs CatBoost on validation.** Correlation 0.99992. The median TabPack /
  CatBoost ratio is 1.0042: TabPack is 0.42% higher, uniformly across Nov (1.0044) and Dec
  (1.0040).
- **Cold cities.** On the 1,447 cold-city rows the median ratio is the same (1.0042), but the
  disagreement is wider: p90 |log ratio| is 1.81% on cold rows against 1.10% on warm ones
  (`final_summary.json` → `validation.by_city_status`). Without labels this measures
  agreement, not accuracy.
- **December chart.** TabPack alone runs from $835.02 to $888.76, mean $857.28. The blend
  equals the submitted curve: $831.84 to $883.92, mean $852.88 (`final_summary.json` →
  `december_chart`).

The submission stays CatBoost (`cb_level_noqs`). TabPack's cross-section is slightly better
($0.76 debiased). That gain does not survive its level offset on plain MAE, so the
pre-registered rule (> $0.5 on F1+F2 clean MAE) is not met.
