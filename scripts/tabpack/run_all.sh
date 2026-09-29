#!/bin/bash
# TabPack experiments under the project's forward-time protocol. Run from anywhere:
#
#   bash scripts/tabpack/run_all.sh                  # backtest plan (36 runs), then tables + blends
#   bash scripts/tabpack/run_all.sh run tp_level F1 0   # one run: <experiment> <fold> <seed>
#   bash scripts/tabpack/run_all.sh tables           # summary tables + blends from finished runs
#   bash scripts/tabpack/run_all.sh final            # final fit (tp_level on Jan-Oct, FINAL_SEEDS) + blend --final
#
# Runs are strictly sequential (one TabPack process at a time; ~1.4 GB host RAM on GPU). A run
# whose artifacts/tabpack/runs/<name>.json exists is skipped, so an interrupted plan resumes
# where it stopped. Every run writes its own JSON / CSV as soon as it finishes.
#
# Environment knobs (defaults reproduce the local MX350 / CPU configuration; see README.md):
#   DEVICE=gpu|cpu|auto  auto: GPU if GPU_PY imports torch with CUDA and the TabPack deps
#                        (PYTHONPATH = tabpack/local/gpu_shim, if present, + tabpack/src)
#   GPU_PY               interpreter for GPU runs (default ~/.venv/bin/python, the local shim setup)
#   N_MODELS             default 64 on the GPU, 32 on the CPU
#   D_BLOCK=256          MLP width (the paper uses 384)
#   AMP=none|bfloat16    autocast dtype (the paper uses bfloat16; needs bf16 hardware)
#   NS_DTYPE=float32|bfloat16   Muon Newton-Schulz dtype (stock TabPack: bfloat16)
#   BATCH_SIZE=512  EVAL_BATCH=1024  THREADS=4  RSS_LIMIT_MB=2200  RUN_TIMEOUT=3600 (s per run)
#   FINAL_SEEDS="0 1 2"  seeds of the final Jan-Oct fit
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$ROOT/scripts/tabpack"
MAIN_PY="$ROOT/.venv/bin/python"
TP="$ROOT/tabpack"
OUT="$ROOT/artifacts/tabpack"
SHIM="$TP/local/gpu_shim"
GPU_PY="${GPU_PY:-$HOME/.venv/bin/python}"
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}"
D_BLOCK="${D_BLOCK:-256}"
AMP="${AMP:-none}"
NS_DTYPE="${NS_DTYPE:-float32}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_BATCH="${EVAL_BATCH:-1024}"
THREADS="${THREADS:-4}"
RSS_LIMIT_MB="${RSS_LIMIT_MB:-2200}"
FINAL_SEEDS="${FINAL_SEEDS:-0 1 2}"
mkdir -p "$OUT/runs" "$OUT/exports" "$TP/local/runs"

DEVICE="${DEVICE:-auto}"
if [ "$DEVICE" = auto ]; then
  if CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$SHIM:$TP/src" "$GPU_PY" -c \
      'import torch,sys,delu,tabm; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    DEVICE=gpu
  else
    DEVICE=cpu
  fi
fi
if [ "$DEVICE" = gpu ]; then N_MODELS="${N_MODELS:-64}"; else N_MODELS="${N_MODELS:-32}"; fi

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/run_all.log"; }

variant_of() {
  case $1 in
    tp_standalone) echo standalone ;;
    tp_level) echo level-nolane ;;
    tp_level_lanetopk) echo level-lanetopk ;;
    tp_level_lanefull) echo level-lanefull ;;
    *) echo "unknown experiment $1" >&2; exit 2 ;;
  esac
}
export_args() {
  case $1 in
    standalone) echo "--mode standalone --lane drop" ;;
    level-nolane) echo "--mode level --lane drop" ;;
    level-lanetopk) echo "--mode level --lane topk --lane-topk 155" ;;
    level-lanefull) echo "--mode level --lane full" ;;
  esac
}

run_one() {  # <experiment> <fold> <seed>
  local exp=$1 fold=$2 seed=$3
  local name="${exp}_${fold}_s${seed}" variant; variant=$(variant_of "$exp")
  local data="$TP/data/freight-$variant-$fold" rundir="$TP/local/runs/$name"
  local out="$OUT"; [ "$fold" = FINAL ] && out="$OUT/final"
  if [ -f "$out/runs/$name.json" ]; then log "skip $name (done)"; return 0; fi
  if [ ! -f "$data/export.json" ]; then
    log "export $variant $fold"
    # shellcheck disable=SC2046
    (cd "$ROOT" && "$MAIN_PY" "$HERE/export_fold.py" --fold "$fold" $(export_args "$variant") \
        --out "$data" --provenance "$OUT/exports/$variant-$fold.json" > "$data.export.log" 2>&1) \
      || { log "EXPORT FAILED $variant $fold (see $data.export.log)"; return 1; }
  fi
  python3 "$HERE/write_config.py" --data "$data" --exp "$rundir" --seed "$seed" --n-models "$N_MODELS" \
      --d-block "$D_BLOCK" --amp "$AMP" --batch-size "$BATCH_SIZE" --eval-batch-size "$EVAL_BATCH" > /dev/null
  log "run $name on $DEVICE (n_models $N_MODELS, d_block $D_BLOCK, batch $BATCH_SIZE, amp $AMP, ns $NS_DTYPE)"
  local rc
  local drv=(--force --threads "$THREADS" --ns-dtype "$NS_DTYPE" --rss-limit-mb "$RSS_LIMIT_MB")
  if [ "$DEVICE" = gpu ]; then
    (cd "$TP" && CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$SHIM:$TP/src" timeout "$RUN_TIMEOUT" \
        "$GPU_PY" "$HERE/run_tabpack.py" "$rundir" "${drv[@]}" > "$rundir.log" 2>&1)
    rc=$?
  else
    (cd "$TP" && timeout "$RUN_TIMEOUT" .venv/bin/python "$HERE/run_tabpack.py" "$rundir" "${drv[@]}" \
        > "$rundir.log" 2>&1)
    rc=$?
  fi
  if [ $rc -ne 0 ]; then log "RUN FAILED $name exit=$rc (see $rundir.log)"; return 1; fi
  (cd "$ROOT" && "$MAIN_PY" "$HERE/evaluate_run.py" --run "$rundir" --data "$data" --experiment "$exp" \
      --fold "$fold" --seed "$seed" --out "$out" 2>&1 | tee -a "$OUT/run_all.log") \
    || { log "EVAL FAILED $name"; return 1; }
}

tables() {
  (cd "$ROOT" && "$MAIN_PY" "$HERE/summarize.py" && "$MAIN_PY" "$HERE/blend.py") 2>&1 | tee -a "$OUT/run_all.log"
}

# Priority order (the drop order when time runs short is from the tail): every experiment's
# seed 0 first (tp_level, then standalone, tp_level seed 1, the lane variants), then the
# remaining seeds. 4 experiments x 3 folds x seeds 0-2 = 36 runs.
PLAN=()
for spec in "tp_level 0" "tp_standalone 0" "tp_level 1" "tp_level_lanetopk 0" "tp_level_lanefull 0" \
    "tp_level 2" "tp_standalone 1" "tp_standalone 2" "tp_level_lanetopk 1" "tp_level_lanetopk 2" \
    "tp_level_lanefull 1" "tp_level_lanefull 2"; do
  read -r _exp _seed <<< "$spec"
  for _fold in F1 F2 F3; do PLAN+=("$_exp $_fold $_seed"); done
done

case "${1:-plan}" in
  plan)
    log "plan start: device $DEVICE, n_models $N_MODELS, d_block $D_BLOCK, batch $BATCH_SIZE, amp $AMP, ns $NS_DTYPE, ${#PLAN[@]} runs"
    for spec in "${PLAN[@]}"; do
      # shellcheck disable=SC2086
      run_one $spec
      (cd "$ROOT" && "$MAIN_PY" "$HERE/summarize.py" > /dev/null 2>&1)
    done
    tables ;;
  run) run_one "$2" "$3" "$4" ;;
  tables) tables ;;
  final)
    for seed in $FINAL_SEEDS; do run_one tp_level FINAL "$seed"; done
    (cd "$ROOT" && "$MAIN_PY" "$HERE/blend.py" --final) 2>&1 | tee -a "$OUT/run_all.log" ;;
  *) echo "usage: $0 [plan|run <exp> <fold> <seed>|tables|final]"; exit 2 ;;
esac
