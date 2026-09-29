"""Write a TabPack config.json (TabPack-dagger search space with cost knobs).

Plain-stdlib script (any python3):

    python3 scripts/tabpack/write_config.py --data <dataset_dir> --exp <exp_dir> [--seed 0] [knobs]

The defaults are the configuration used for the freight experiments (feasible on a 2 GB MX350):
64 models, d_block 256, batch 512, float32, periodic ("cosine") numerical embeddings, Muon, and
the paper's early-stopping / ensemble settings (patience 16; greedy online ensemble with
patience 32 and at most 32 members). They follow experiments/tabpack-cosine-macbook/make.py
except d_block (384 there) and amp_dtype (off: no bf16 hardware on this CPU / Pascal GPU).
For a CPU-only run use --n-models 32 (about 70 s per full-pack epoch on 4 cores).
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--exp', required=True)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--n-models', type=int, default=64)
ap.add_argument('--d-block', type=int, default=256)
ap.add_argument('--max-n-blocks', type=int, default=3)
ap.add_argument('--d-emb-max', type=int, default=32)
ap.add_argument('--batch-size', type=int, default=512)
ap.add_argument('--eval-batch-size', type=int, default=1024)
ap.add_argument('--n-epochs', type=int, default=-1)
ap.add_argument('--patience', type=int, default=16)
ap.add_argument('--ens-patience', type=int, default=32)
ap.add_argument('--ens-max-size', type=int, default=32)
ap.add_argument('--timeout', type=int, default=None)
ap.add_argument('--amp', choices=['none', 'bfloat16'], default='none')
ap.add_argument('--optimizer', choices=['MuonAdamWPack', 'AdamWPack'], default='MuonAdamWPack')
ap.add_argument('--no-embeddings', action='store_true')
ap.add_argument('--num-policy', default='noisy-quantile')
args = ap.parse_args()

space_model: dict = {
    'n_blocks': ['_tune_', 'int', 1, args.max_n_blocks],
    'dropout': ['_tune_', '?uniform', 0.0, 0.0, 0.5],
}
model: dict = {'activation': 'ReLU', 'd_block': args.d_block}
if not args.no_embeddings:
    model['num_embeddings'] = {'type': 'CosineEmbeddingsPack'}
    space_model['num_embeddings'] = {
        'd_embedding': ['_tune_', 'int', 8, args.d_emb_max, 4],
        'init_scale': ['_tune_', 'loguniform', 0.01, 10.0],
    }
space_opt: dict = {
    'lr': ['_tune_', 'loguniform', 0.0001, 0.005],
    'weight_decay': ['_tune_', 'loguniform', 0.001, 1.0],
}
if args.optimizer == 'MuonAdamWPack':
    space_opt['muon_lr'] = ['_tune_', 'loguniform', 0.001, 0.1]

config: dict = {
    'seed': args.seed,
    'data': {
        'path': str(Path(args.data).resolve()),
        'extract_bin_from_num': True,
        'num_policy': args.num_policy,
        'bin_policy': 'convert-to-cat',
        'cat_policy': 'ordinal',
        'cache': False,
    },
    'n_models': args.n_models,
    'model': model,
    'optimizer': {'type': args.optimizer, 'shared_step': True},
    'batch_size': args.batch_size,
    'eval_batch_size': args.eval_batch_size,
    'n_epochs': args.n_epochs,
    'patience': args.patience,
    'online_ensembles': {
        'greedy': {
            'type': 'greedy',
            'update_type': 'latest',
            'include_current_ensemble_in_pool': True,
            'patience': args.ens_patience,
            'options': {'max_ensemble_size': args.ens_max_size},
        }
    },
    'sampler': {'type': 'RandomSampler', 'space': {'model': space_model, 'optimizer': space_opt}},
    'track_online_ensemble_history': True,
    'track_experiments': True,
}
if args.amp != 'none':
    config['amp_dtype'] = args.amp
if args.timeout is not None:
    config['timeout'] = args.timeout

exp = Path(args.exp)
exp.mkdir(parents=True, exist_ok=True)
(exp / 'config.json').write_text(json.dumps(config, indent=4))
print(f'wrote {exp / "config.json"}')
