#!/bin/bash
# Create the (optional) GPU shim: a directory of symlinks to the pure-Python dependencies of
# tabpack/.venv, so that TabPack can run with the CUDA torch of another interpreter
# (here ~/.venv: torch 2.6.0+cu124, whose kernels include sm_60 -> runs on the MX350, sm_61).
# ~/.venv is only READ; nothing is installed into it.
#
#   bash scripts/tabpack/make_gpu_shim.sh            # -> tabpack/local/gpu_shim
#   GPU_PY=/path/to/python bash scripts/tabpack/make_gpu_shim.sh
#
# The interpreter must provide torch (CUDA), numpy, scipy and scikit-learn; the shim adds
# delu, loguru, optuna (+ sqlalchemy, alembic, colorlog, mako, yaml, greenlet), rtdl_* and tabm.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SITE="$ROOT/tabpack/.venv/lib/python3.12/site-packages"
SHIM="$ROOT/tabpack/local/gpu_shim"
GPU_PY="${GPU_PY:-$HOME/.venv/bin/python}"

[ -d "$SITE" ] || { echo "missing $SITE (create tabpack/.venv first, see tabpack/SETUP_LOCAL.md)"; exit 1; }
mkdir -p "$SHIM"
for name in alembic colorlog delu greenlet loguru mako optuna PyYAML.libs \
            rtdl_num_embeddings.py rtdl_revisiting_models.py sqlalchemy tabm.py _yaml yaml; do
  if [ -e "$SITE/$name" ]; then
    ln -sfn "$SITE/$name" "$SHIM/$name"
  else
    echo "note: $name not found in $SITE (skipped)"
  fi
done
echo "shim: $SHIM"
# Check that the interpreter sees a CUDA device and can import TabPack through the shim.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$SHIM:$ROOT/tabpack/src" "$GPU_PY" - <<'EOF'
import torch, delu, optuna, tabm, rtdl_num_embeddings  # noqa: F401
import project.tabpack  # noqa: F401
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')
EOF
