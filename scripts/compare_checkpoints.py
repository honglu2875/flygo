#!/usr/bin/env python3
"""Compare frozen checkpoints on one common validation set; leave final test labels closed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release
from flygo.play import load_player
from flygo.qualify import sha256
from flygo.runtime import cpu_profile, pin
from flygo.storage import GIB, StorageBudget


def predict_metrics(model, arrays, indices, batch_size):
    rows = []
    for begin in range(0, len(indices), batch_size):
        selected = indices[begin:begin + batch_size]
        result = model.infer(arrays['features'][selected])
        logits = np.where(arrays['legal'][selected], result['logits'], -1e30)
        logits -= logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        target = arrays['raw_policy'][selected]
        cross_entropy = -(target * logp).sum(axis=1, dtype=np.float64)
        entropy = -(target * np.log(np.maximum(target, 1e-30))).sum(axis=1, dtype=np.float64)
        rows.append(np.stack([cross_entropy - entropy,
                              np.square(result['value'] - arrays['raw_value'][selected]),
                              logits.argmax(axis=1) == target.argmax(axis=1)], axis=1))
    return np.concatenate(rows)


def summarize(metrics, games, *, seed=709, bootstrap=1000):
    if not len(metrics):
        return {'positions': 0, 'games': 0}
    unique, membership = np.unique(games, return_inverse=True)
    counts = np.bincount(membership)
    totals = np.stack([np.bincount(membership, weights=metrics[:, i]) for i in range(3)], axis=1)
    rng = np.random.default_rng(seed)
    sample = rng.integers(len(unique), size=(bootstrap, len(unique)))
    distribution = totals[sample].sum(axis=1) / counts[sample].sum(axis=1)[:, None]
    interval = np.quantile(distribution, [.025, .975], axis=0)
    names = ('policy_kl', 'value_mse', 'teacher_top1_agreement')
    return dict(positions=len(metrics), games=len(unique), **{
        name: {'mean': float(metrics[:, i].mean()), 'game_bootstrap_95': interval[:, i].tolist()}
        for i, name in enumerate(names)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    parser.add_argument('--release', default='pilot-v1')
    parser.add_argument('--checkpoints', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--cpus')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    cpus = list(map(int, args.cpus.split(','))) if args.cpus else cpu_profile()['research_cpus'][:args.threads]
    if not 0 < args.threads <= len(cpus) or args.batch_size <= 0:
        parser.error('Positive thread and batch sizes must fit the CPU allocation')
    pin(cpus)
    args.output.mkdir(parents=True, exist_ok=False)
    with StorageBudget(args.root).reserve(files=32 * 1024**2, heap=24 * GIB, purpose='common held-out validation comparison'):
        manifest, arrays, indexes = load_release(args.root, args.root / 'releases' / args.release / 'manifest.json',cache=True)
        indices = indexes['validation']
        ply, value, opponent = (arrays[key][indices] for key in ('ply', 'raw_value', 'opponent_index'))
        slices = {
            'natural': np.ones(len(indices), bool),
            'novel_d4_input': np.isin(indices, indexes['validation_novel']),
            'opening': ply < 16, 'middle': (ply >= 16) & (ply < 50), 'late': ply >= 50,
            'near_even': np.abs(value) < .5,
            'decisive': (np.abs(value) >= .5) & (np.abs(value) < .98),
            'saturated': np.abs(value) >= .98,
            'teacher_pass': arrays['raw_policy'][indices].argmax(axis=1) == 81,
            'opponent_lower': opponent < 3,
            'opponent_middle': (opponent >= 3) & (opponent < 6),
            'opponent_upper': opponent >= 6,
        }
        graph = Path(json.loads((args.root / 'runs/m4/graph.json').read_text())['path'])
        records = []
        for checkpoint in args.checkpoints:
            started = time.perf_counter()
            model, metadata = load_player(checkpoint, graph, threads=args.threads)
            if metadata['dataset_id'] != manifest['dataset_id']:
                raise ValueError('Checkpoint and evaluation release must match for this comparison')
            metrics = predict_metrics(model, arrays, indices, args.batch_size)
            groups = arrays['game_index'][indices]
            record = dict(checkpoint=str(checkpoint), sha256=sha256(checkpoint),
                          model=metadata['model_config'], dataset_id=metadata['dataset_id'],
                          optimizer_step=int(model.checkpoint_arrays()['optimizer_step']),
                          seconds=time.perf_counter() - started,
                          slices={name: summarize(metrics[mask], groups[mask]) for name, mask in slices.items()})
            records.append(record)
            atomic_json(args.output / 'result.json', dict(status='complete' if len(records) == len(args.checkpoints) else 'running',
                release=args.release, dataset_id=manifest['dataset_id'], cpus=cpus, records=records,
                uncertainty='95% percentile bootstrap by complete game, 1000 resamples; not uncertainty across training seeds',
                test_set='not evaluated'))
            print(json.dumps({k: v for k, v in record.items() if k != 'slices'} | {'natural': record['slices']['natural']}), flush=True)
            del model


if __name__ == '__main__':
    main()
