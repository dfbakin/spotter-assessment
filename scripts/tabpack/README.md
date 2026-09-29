# TabPack experiments

TabPack (a pack of MLPs trained in parallel, with sampled hyper-parameters, early stopping and
an online greedy ensemble) was tried under the same forward-time protocol as the CatBoost
model. Four experiments and a post-hoc blend were run, every experiment with seeds 0, 1 and 2,
on a rented RTX 5090 with the paper's full-size configuration (see "Remote GPU" below):

| name | target | features | seeds |
|---|---|---|---|
| `tp_standalone` | log(rpm), no level model | the phase-1 `cb_time` set without `lane` (raw market columns, calendar incl. `days_to_quarter_end`, `t_days`, geography, distance, weight + flags, cities, equipment) | 3 |
| `tp_level` | log(rpm) − L(date), L fitted as in `LevelCatBoostRate` | the 24 tree features of `artifacts/tuning/best_params.json` minus `lane` | 3 |
| `tp_level_lanetopk` | as `tp_level` | as `tp_level`, with `lane` kept as the 155 most frequent inner-train lanes + `__other__` | 3 |
| `tp_level_lanefull` | as `tp_level` | all 24 features, `lane` one-hot over every inner-train lane (`--lane full`) | 3 |
| blend | `w · log p_CB + (1 − w) · log p_TP` | CatBoost = seed mean of `cb_level_noqs` (3 seeds), TabPack = seed mean of `tp_level` (3 seeds) | – |

Protocol, per fold (F1: Jan–Apr → May–Jun, F2: Jan–Jun → Jul–Aug, F3: Jan–Aug → Sep–Oct):

- As in `freight.backtest.run_fold`, one `RpmReference` fitted on the whole training window
  flags the corrupted rows. They are dropped from train and validation and are excluded from
  the clean test metrics.
- For the level experiments, L is fitted on all clean rows of the training window, exactly as
  `LevelCatBoostRate` does (K = 1).
- TabPack's inner validation, used for early stopping and for picking the greedy ensemble, is
  the last 28 days of the training window, split by time. The network is trained on the clean
  rows before it. CatBoost's `f` sees the whole window, so TabPack has four fewer weeks of
  data for `f`.
- The test split is the fold's whole test window. Metrics come from `freight.evaluate`: MAE
  on clean rows, MAE on all rows, MAPE, day-debiased MAE, and median bias per test month.
- Blend weights are chosen on the F1+F2 mean clean MAE only. The chosen `w` is applied to F3
  once.

The results and their interpretation are in `artifacts/tabpack/` (see "Outputs" below).

## Environments

Two interpreters are involved. Nothing is shared between them except files on disk.

1. **Main project venv `.venv`** (CatBoost, pandas, the editable `freight` package) runs
   `export_fold.py`, `evaluate_run.py`, `summarize.py` and `blend.py` from the project root.
2. **TabPack venv `tabpack/.venv`** runs `run_tabpack.py` from the `tabpack/` root, because
   `lib.utils.init()` refuses any other working directory. TabPack is an external clone and
   is git-ignored. Recreate it as follows:

   ```sh
   git clone <tabpack repo> tabpack    # the paper's reference implementation
   curl -LsSf https://astral.sh/uv/install.sh -o uv-install.sh
   UV_NO_MODIFY_PATH=1 UV_INSTALL_DIR=$HOME/.local/bin sh uv-install.sh
   cd tabpack
   ~/.local/bin/uv export --frozen --no-dev --no-hashes --no-emit-project --no-header --no-annotate \
       | grep -iv '^torch==\|^nvidia\|^triton' > /tmp/req_notorch.txt
   ~/.local/bin/uv venv .venv --python 3.12.9
   ~/.local/bin/uv pip install --python .venv/bin/python torch==2.7.1 \
       --index-url https://download.pytorch.org/whl/cpu
   ~/.local/bin/uv pip install --python .venv/bin/python -r /tmp/req_notorch.txt
   ~/.local/bin/uv pip install --python .venv/bin/python --no-deps -e .
   ```

   Do not run `uv sync` or `uv run` in `tabpack/`. They would re-sync to the lock file and pull
   the multi-GB CUDA build of torch. Call `.venv/bin/python` directly. The full notes, with
   versions, a smoke test and the measured costs, are in `tabpack/SETUP_LOCAL.md`. The
   essentials are repeated here because that file is not committed.

### Local fixes applied by `run_tabpack.py`

- `lib.utils.configure_torch()` pins torch to 1 thread. The driver resets it to 4, the
  number of physical cores.
- `vendor/muon.py` runs the Newton–Schulz orthogonalisation in bfloat16. This CPU (AVX2
  only) and the Pascal GPU have no fast bf16: on the CPU it is about 33× slower than float32.
  The driver swaps in the same iteration in float32. `amp_dtype` also stays unset, so
  training runs in float32.
- The stock code never writes the online ensemble's per-row predictions. The driver captures
  the `OnlineEnsemble` objects and saves `ensemble_greedy_predictions.npz` (val and test, on
  the target scale).
- A watchdog kills the run (exit 99) when host RSS exceeds 2.2 GB. The machine has 7.6 GB
  and only about 2–3 GB free.

### Remote GPU (vast.ai RTX 5090): the configuration of the reported runs

The laptop (MX350, 7 GB RAM) could not run the plan, so all reported runs come from a rented
vast.ai instance: RTX 5090 (32 GB, Blackwell sm_120), 128 cores, Ubuntu 24.04, `uv` installed.
Its storage is not durable, so results are copied back after every few runs.

```sh
# 1. Copy the project (data/ included; venvs, git, models and most backtest preds excluded).
#    summarize.py / blend.py need the cb_level_noqs and cb_time reference predictions.
rsync -az -e "ssh -p <port>" \
  --include='/artifacts/backtest/preds/' \
  --include='/artifacts/backtest/preds/cb_level_noqs_F[123]_r[012].csv' \
  --include='/artifacts/backtest/preds/cb_time_F[123]_r[012].csv' \
  --exclude='/artifacts/backtest/preds/*' \
  --exclude='/.venv' --exclude='/tabpack/.venv' --exclude='/tabpack/local' --exclude='/.git' \
  --exclude='/artifacts/models' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='/reports/_output' --exclude='*.pdf' ./ root@<host>:/root/spotter_assessment/

# 2. Main venv (on the box, project root).
uv venv .venv --python /usr/bin/python3.12
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m pytest -q

# 3. TabPack venv: the locked set minus torch, plus torch 2.7.1 built for CUDA 12.8.
#    The lock's PyPI torch 2.7.1 (CUDA 12.6) has no sm_120 kernels and fails on Blackwell.
cd tabpack
uv export --frozen --no-dev --no-hashes --no-emit-project --no-header --no-annotate \
    | grep -iv '^torch==\|^nvidia\|^triton' > /root/tp_req_notorch.txt
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r /root/tp_req_notorch.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.get_arch_list())'
cd ..

# 4. The plan, paper configuration, detached from the SSH session (resumable; re-run to resume).
export DEVICE=gpu GPU_PY=$PWD/tabpack/.venv/bin/python N_MODELS=64 D_BLOCK=384 AMP=bfloat16 \
       NS_DTYPE=bfloat16 BATCH_SIZE=512 EVAL_BATCH=32768 THREADS=8 RSS_LIMIT_MB=16000
setsid nohup bash -c 'bash scripts/tabpack/run_all.sh plan; bash scripts/tabpack/run_all.sh final' \
    > run_all.out 2>&1 < /dev/null &

# 5. Copy the results back (from the laptop), after every few runs and at the end.
rsync -az -e "ssh -p <port>" root@<host>:/root/spotter_assessment/artifacts/tabpack/ artifacts/tabpack/
rsync -az -e "ssh -p <port>" --include='*/' --include='*.log' --include='report.json' \
    --include='driver.json' --include='config.json' --exclude='*' \
    root@<host>:/root/spotter_assessment/tabpack/local/runs/ tabpack/local/runs/
```

- **Measured cost** (paper configuration):
  - 0.3–1.2 s per epoch and 0.5–2.1 min per run.
  - The 36 backtest runs, with exports and evaluation, took 38 minutes, and the 3 final runs
    took 4 more.
  - Peak CUDA memory: allocated ≤ 5.9 GB, and reserved ≤ 16.7 GB, except
    `tp_level_lanefull` (19.6 / 28.8 GB). Reserved exceeds allocated because of PyTorch's
    caching allocator. The eval batch was 32,768 rows, and no out-of-memory retry occurred.
  - Host RSS ≤ 1.91 GB.
  - Per-run figures are in `runs/*.json` → `runtime`, and the summary is in
    `artifacts/RESULTS.md` §7.
- Do not use `uv sync` or `uv run` in `tabpack/`. They would re-pin the CUDA 12.6 torch.
- On this image `uv` runs with `UV_NO_CACHE=1`, and the PyTorch index served at about
  1–6 MB/s, so step 3 took about 20 minutes.
- `blend.py --final` loads the saved final CatBoost models from `artifacts/models/`, which
  are not copied. Run it on the laptop after copying `artifacts/tabpack/final/` back. On the
  box, the `final` step's blend call fails, which is harmless.
- The shim directory of the local setup does not exist on the box. `PYTHONPATH` still names
  it, which is harmless, and `DEVICE=auto` also detects the GPU there.

### Optional GPU (MX350, 2 GB) through a symlink shim

The GPU runs are about 5× faster than the CPU. No CUDA torch is installed for this project.
Instead, the existing `~/.venv` (torch 2.6.0+cu124, whose kernels include sm_60 and so run
on the MX350's sm_61) is used **read-only**. It gets a shim directory of symlinks to the
pure-Python dependencies in `tabpack/.venv` (delu, loguru, optuna and its dependencies,
rtdl_*, tabm):

```sh
bash scripts/tabpack/make_gpu_shim.sh     # creates tabpack/local/gpu_shim and checks CUDA
```

`~/.venv` must provide torch (CUDA), numpy, scipy and scikit-learn. To use your own GPU
venv instead, note that torch ≤ 2.7 CUDA 12.6 wheels still ship Pascal kernels:
`uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126`. Point
`GPU_PY` at that interpreter.

The GPU is optional. Without the shim, `run_all.sh` falls back to the CPU and to 32 models
(about 70 s per full-pack epoch on 4 cores, about 4× the GPU wall time per run). The
64-model configuration also runs on the CPU (set `N_MODELS=64`), but takes about 2× longer.

## Configuration

The reported runs use the paper's configuration (`tabpack/experiments/tabpack-cosine/make.py`):
64 models, `d_block` 384, batch 512, `amp_dtype` bfloat16, and Muon's Newton–Schulz iteration
in bfloat16 as in the vendored code (`NS_DTYPE=bfloat16`). The eval batch is 32,768, the stock
default. All other settings are the `write_config.py` defaults below. The paper's `make.py`
also sets `save_all_predictions`; it is not needed here, because the driver saves the
ensemble's predictions.

The `write_config.py` / `run_all.sh` defaults are the laptop configuration:

- **Pack and model.** 64 models, `d_block` 256, batch 512, eval batch 1024, float32. Each
  MLP has 1–3 blocks and dropout {0 or U(0, 0.5)}. Numerical features get periodic
  (`CosineEmbeddingsPack`) embeddings, with `d_embedding` 8–32 and `init_scale`
  log-U(0.01, 10).
- **Optimiser.** Muon + AdamW (`MuonAdamWPack`), with lr, weight decay and Muon lr sampled
  per model.
- **Early stopping and ensemble.** Unlimited epochs with patience 16 per model. The greedy
  online ensemble has patience 32 and at most 32 members. These are the paper's settings (see
  `tabpack/experiments/tabpack-cosine-macbook/make.py`). The only differences from the paper
  are `d_block` 256 instead of 384 and bf16 switched off (`D_BLOCK`, `AMP` and `NS_DTYPE`
  restore them).
- **Preprocessing.** Noisy-quantile numerical features. Binary columns become categoricals.
  Categoricals are ordinal- and then one-hot-encoded, and an unseen category maps to the
  all-zero vector.
- **VRAM.** 64 models × `d_block` 384 is at the MX350's VRAM limit, so the laptop default is
  256. On the 5090 the paper's 384 is used.
- **Full lane.** One-hot over all 3,984 lanes runs out of the MX350's memory, hence the
  top-155 variant. On the 5090, `tp_level_lanefull` runs as well.

## Commands

From the project root:

```sh
bash scripts/tabpack/make_gpu_shim.sh          # optional (GPU)
bash scripts/tabpack/run_all.sh                # 36 backtest runs, then summarize + blend
bash scripts/tabpack/run_all.sh run tp_level F1 0   # a single run: <experiment> <fold> <seed>
bash scripts/tabpack/run_all.sh tables         # rebuild the tables / blends from finished runs
bash scripts/tabpack/run_all.sh final          # final fit, 3 seeds, + blend.py --final (see below)
```

- `run_all.sh` runs strictly one TabPack process at a time. A run whose
  `artifacts/tabpack/runs/<name>.json` already exists is skipped, so an interrupted plan
  resumes where it stopped. The plan's order is also the drop order when time is short.
  Seed 0 comes first, in this order: `tp_level`, `tp_standalone`, `tp_level` seed 1,
  `tp_level_lanetopk`, `tp_level_lanefull`. Then come the remaining seeds (0–2 for every
  experiment, 36 runs).
- Environment knobs:
  - `DEVICE=gpu|cpu|auto`. Auto picks the GPU when `GPU_PY` imports torch with CUDA plus the
    TabPack dependencies.
  - `GPU_PY`, `N_MODELS`.
  - `D_BLOCK` (default 256), `AMP=none|bfloat16`, `NS_DTYPE=float32|bfloat16`.
  - `BATCH_SIZE` (512), `EVAL_BATCH` (1024), `THREADS` (4).
  - `RSS_LIMIT_MB` (2200, the driver's host-RAM watchdog), `RUN_TIMEOUT` (seconds per run,
    default 3600).
  - `FINAL_SEEDS` (default `0 1 2`).

  The defaults reproduce the laptop setup. The remote recipe above sets the paper
  configuration.

The steps of one run, which `run_all.sh run` performs:

```sh
.venv/bin/python scripts/tabpack/export_fold.py --fold F1 --mode level --lane drop \
    --out tabpack/data/freight-level-nolane-F1 --provenance artifacts/tabpack/exports/level-nolane-F1.json
python3 scripts/tabpack/write_config.py --data tabpack/data/freight-level-nolane-F1 \
    --exp tabpack/local/runs/tp_level_F1_s0 --seed 0
cd tabpack && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=local/gpu_shim:$PWD/src \
    ~/.venv/bin/python ../scripts/tabpack/run_tabpack.py local/runs/tp_level_F1_s0 --force && cd ..
#   (CPU: cd tabpack && .venv/bin/python ../scripts/tabpack/run_tabpack.py local/runs/tp_level_F1_s0 --force)
.venv/bin/python scripts/tabpack/evaluate_run.py --run tabpack/local/runs/tp_level_F1_s0 \
    --data tabpack/data/freight-level-nolane-F1 --experiment tp_level --fold F1 --seed 0
```

- **Export modes.** `export_fold.py` takes `--mode level|standalone` and
  `--lane drop|topk|full` (with `--lane-topk 155`). With `--fold FINAL` it trains on all
  clean Jan–Oct rows, with the last 28 days of October as validation. Its test split is the
  12,000 Nov–Dec validation rows plus the 31 December chart rows (Option A market inputs, as
  in `freight.final`), with a dummy target.
- **Final fit and gate.** `run_all.sh final` runs `tp_level` on that export with
  `FINAL_SEEDS` (0, 1, 2). It then blends the saved final CatBoost models at the chosen `w`
  (`blend.py --final`). The gate requires the chosen blend to beat `cb_level_noqs` on the
  F1+F2 mean clean MAE by more than $0.5.
  - The gate is recorded, not enforced. `blend.py --final` writes its files whatever the
    verdict and stores it in `final/final_summary.json` → `gate`.
  - A failed gate means the blend is not a candidate for submission.
  - It never touches the submission files in the project root.

## Outputs

Committed, under `artifacts/tabpack/`:

| path | content |
|---|---|
| `runs/<exp>_<fold>_s<seed>.json` | one per run, written as soon as it finishes: test and inner-val metrics, month bias, slices, runtime, device, epochs, ensemble size and members, export provenance |
| `preds/<exp>_<fold>_s<seed>.csv` | per-row test predictions: `L`, TabPack output `f` (log scale), `pred` = distance · exp(L + f) |
| `configs/<exp>_<fold>_s<seed>.json` | the exact TabPack `config.json` of each run |
| `exports/<variant>-<fold>.json` | export provenance: windows, row counts, dropped rows, columns, cardinalities, L coefficients |
| `runs_summary.csv`, `table_a.csv` | per run, and per experiment × fold (mean ± sd over seeds), with the `cb_level_noqs` / `cb_time` references re-scored from `artifacts/backtest/preds` |
| `blend_grid.csv`, `blend.json` | blend grid per fold and w (tp_level, plus the other experiments for information); the chosen w, F3 result, seed-pair spread, day-block bootstrap, final-fit gate, and for information a day-block bootstrap of the debiased MAE at the debiased-optimal w |
| `blend_slices.csv`, `blend_preds_F3.csv` | slices (equipment, distance band, month, lane frequency) per fold and model; F3 per-row predictions |
| `final/` | final-fit TabPack runs (`runs/`, `preds/`, `configs/`), TabPack and blended validation / December predictions, `final_components.csv`, `final_summary.json` (with the gate verdict) |
| `run_all.log` | the plan's log |

Git-ignored: `tabpack/data/freight-*` (exported datasets, rebuilt in seconds) and
`tabpack/local/runs/` (raw TabPack run directories with checkpoints and logs).
