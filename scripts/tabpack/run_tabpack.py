"""Thin TabPack driver. Run from the tabpack/ root (``lib.utils.init()`` insists on it):

    CPU:  .venv/bin/python ../scripts/tabpack/run_tabpack.py <exp_dir> --force
    GPU:  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=local/gpu_shim:$PWD/src \
          ~/.venv/bin/python ../scripts/tabpack/run_tabpack.py <exp_dir> --force

``<exp_dir>`` holds a config.json from write_config.py. On top of the stock
``python -m project.tabpack <exp>`` it:

* resets torch's thread count (``lib.utils.configure_torch`` pins it to 1 thread);
* runs Muon's Newton-Schulz iteration in float32 (``--ns-dtype``): the vendored code casts to
  bfloat16, ~33x slower than fp32 on this AVX2-only CPU and no faster on the Pascal MX350;
* captures the online-ensemble objects and saves the ensemble's per-row val/test predictions
  to ``ensemble_greedy_predictions.npz`` (the stock code never fills
  ``online_ensemble_predictions.npz``); predictions are on the target scale (de-standardised);
* records wall time, epochs run, ensemble size, per-epoch times and peak memory in
  ``driver.json``; a watchdog kills the run (exit 99) above ``--rss-limit-mb`` host RSS.
"""

import argparse
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np

import lib.experiment
import lib.utils
import project.ensemble_utils as eu
import project.tabpack as tp
import vendor.muon

_CAPTURED: dict = {}
_orig_update = tp.update_online_ensembles


def _update_and_capture(online_ensembles, **kwargs):
    _CAPTURED['ens'] = online_ensembles
    return _orig_update(online_ensembles, **kwargs)


tp.update_online_ensembles = _update_and_capture


def _make_ns(dtype_name: str):
    import torch

    dtype = getattr(torch, dtype_name)
    orig = vendor.muon.zeropower_via_newtonschulz5

    def ns(G, steps: int):
        if dtype is torch.bfloat16:
            return orig(G, steps)
        a, b, c = (3.4445, -4.7750, 2.0315)  # identical to the vendored iteration
        X = G.to(dtype)
        if G.size(-2) > G.size(-1):
            X = X.mT
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(-2) > G.size(-1):
            X = X.mT
        return X

    return ns


def _rss_mb() -> float:
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2**20


def _watchdog(limit_mb: float, state: dict) -> None:
    while True:
        rss = _rss_mb()
        state['max_rss_mb'] = max(state.get('max_rss_mb', 0.0), rss)
        if rss > limit_mb:
            print(f'\n[watchdog] RSS {rss:.0f} MB > {limit_mb:.0f} MB: aborting', flush=True)
            os._exit(99)
        time.sleep(0.25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('exp')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--rss-limit-mb', type=float, default=2200.0)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--ns-dtype', choices=['float32', 'bfloat16'], default='float32')
    args = ap.parse_args()

    lib.utils.init()
    import torch

    torch.set_num_threads(args.threads)
    vendor.muon.zeropower_via_newtonschulz5 = _make_ns(args.ns_dtype)
    state: dict = {}
    threading.Thread(target=_watchdog, args=(args.rss_limit_mb, state), daemon=True).start()

    exp = Path(args.exp)
    t0 = time.perf_counter()
    lib.experiment.run(tp.main, None, exp, force=args.force)
    wall = time.perf_counter() - t0

    out: dict = {
        'threads': args.threads,
        'ns_dtype': args.ns_dtype,
        'device': str(lib.utils.get_device()),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'torch': torch.__version__,
        'wall_s': round(wall, 1),
        'peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        'watchdog_max_rss_mb': round(state.get('max_rss_mb', 0.0), 1),
    }
    if torch.cuda.is_available():
        out['cuda_max_allocated_mb'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        out['cuda_max_reserved_mb'] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    numlog = np.load(exp / 'numlog.npz')
    ep_t = numlog['epochs.time']
    report = lib.experiment.load_report(exp)
    epoch_size = int(report['epoch_size'])
    out['epochs_run'] = int(len(ep_t))
    out['epoch_size_steps'] = epoch_size
    out['epoch_end_times_s'] = [round(float(x), 1) for x in ep_t]
    out['s_per_epoch_after_first'] = (
        round(float(np.diff(ep_t).mean()), 2) if len(ep_t) > 1 else round(float(ep_t[0]), 2)
    )
    # Best (early-stopping) epoch of every finished model.
    best_epochs = [
        int(e['report']['best_step']) / epoch_size
        for e in report.get('experiments', [])
        if 'report' in e and 'best_step' in e['report']
    ]
    if best_epochs:
        out['model_best_epoch'] = {
            'min': round(min(best_epochs), 1),
            'median': round(float(np.median(best_epochs)), 1),
            'max': round(max(best_epochs), 1),
        }
    ens = _CAPTURED.get('ens') or {}
    for name, e in ens.items():
        preds = {
            part: eu.compute_ensemble_prediction(v, e.weights, e._prediction_type)
            for part, v in e.predictions.items()
        }
        np.savez(exp / f'ensemble_{name}_predictions.npz', **preds)
        out[f'ensemble_{name}'] = {
            'ids': e.ids.tolist(),
            'steps': e.steps.tolist(),
            'epochs': [round(s / epoch_size, 1) for s in e.steps.tolist()],
            'size': int(len(e.ids)),
            'n_unique_models': int(len(np.unique(e.ids))),
            'val_score': e._score,
        }
    out['report_time_s'] = report.get('time')
    out['n_models_config'] = json.loads((exp / 'config.json').read_text())['n_models']
    out['n_models_finished'] = report.get('n_models')
    out['online_ensemble_metrics'] = (
        report.get('online_ensembles', {}).get('greedy', {}).get('report', {}).get('metrics')
    )
    (exp / 'driver.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith('ensemble_')}, indent=2))


if __name__ == '__main__':
    sys.exit(main())
