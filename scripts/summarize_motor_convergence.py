#!/usr/bin/env python3
"""Audit convex-decoder endpoints and compare all paired frozen-core fits.

Run ``owner`` beside each completed bank, then ``combine`` on the three small
owner reports. No motor bank or checkpoint needs to be copied to the main host.
Family totals retain exact paired means and conditional bootstrap intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB, StorageBudget

try:
    from readout_probe import position_metrics
except ModuleNotFoundError:
    from flygo.readout_probe import position_metrics


METRICS = ('policy_kl', 'policy_entropy', 'teacher_entropy', 'top_probability', 'teacher_top1')


def read(path):
    return json.loads(path.read_text())


def check_file(path, digest):
    if sha256(path) != digest:
        raise ValueError('Changed evidence: ' + str(path))


def family_totals(metrics, membership, count):
    return np.stack([np.bincount(membership, weights=metrics[name], minlength=count)
                     for name in METRICS], axis=1).tolist()


def owner(root, study, seed):
    plan_path = study / 'plan.json'
    plan = read(plan_path)
    case = next(case for case in plan['banks'] if case['seed'] == seed)
    directory = study / f'seed-{seed}'
    report = read(directory / 'result.json')
    if report['status'] != 'complete' or read(directory / 'status.json')['state'] != 'complete':
        raise ValueError('All fits must complete before analysis')
    source = read(directory / 'source.json')
    if report['source'] != source or source['plan_sha256'] != sha256(plan_path):
        raise ValueError('Endpoint provenance differs')
    for name, key in [('worker.py', 'worker_sha256'), ('readout_convex.py', 'convex_sha256'),
                      ('readout_probe.py', 'probe_sha256')]:
        check_file(study / name, source[key])
    bank_dir = root / plan['source_bank'] / f'seed-{seed}'
    check_file(bank_dir / 'bank.json', case['bank_sha256'])
    bank = read(bank_dir / 'bank.json')
    source_plan = bank_dir.parent / 'plan.json'
    check_file(source_plan, plan['source_plan_sha256'])
    if source['bank_sha256'] != case['bank_sha256'] or not bank['source_parameters_unchanged']:
        raise ValueError('Frozen core or bank identity differs')
    manifest = read(root / 'releases' / read(source_plan)['release'] / 'manifest.json')
    if manifest['dataset_id'] != bank['dataset_id']:
        raise ValueError('Dataset identity differs')
    baseline = {}
    family_sets = {}
    for split in ('train', 'validation'):
        if bank['selection'][split]['positions'] != plan['training_positions' if split == 'train' else 'validation_positions']:
            raise ValueError('Bank size differs from the fitting contract')
        check_file(bank_dir / (split + '-motors.npy'), bank['selection'][split]['motors_sha256'])
        labels_path = bank_dir / (split + '-labels.npz')
        check_file(labels_path, bank['selection'][split]['labels_sha256'])
        with np.load(labels_path, allow_pickle=False) as data:
            metrics = position_metrics(data['baseline'], data['legal'], data['policy'])
            families = np.asarray([manifest['records'][int(i)]['opening_family'] for i in data['game_index']])
        family_sets[split] = set(families)
        if sorted(family_sets[split]) != bank['selection'][split]['families']:
            raise ValueError('Opening-family metadata differs')
        for key in METRICS:
            np.testing.assert_allclose(metrics[key].mean(), report['baseline'][split][key], rtol=0, atol=1e-12)
        if split == 'validation':
            unique, membership = np.unique(families, return_inverse=True)
            counts = np.bincount(membership)
            baseline = family_totals(metrics, membership, len(unique))
    if family_sets['train'] & family_sets['validation']:
        raise ValueError('Training and validation families overlap')
    expected = [(variant, ridge) for variant in plan['variants'] for ridge in plan['ridge']]
    if [(r['variant'], r['ridge']) for r in report['trials']] != expected:
        raise ValueError('Missing, duplicate or reordered fits')
    totals = []
    receipts = []
    for trial in report['trials']:
        target = directory / trial['variant'] / ('ridge-' + format(trial['ridge'], '.4g'))
        if read(target / 'result.json') != trial or trial['status'] != 'complete':
            raise ValueError('Endpoint record differs')
        for name, key in [('checkpoint.npz', 'checkpoint_sha256'), ('position-metrics.npz', 'position_metrics_sha256')]:
            check_file(target / name, trial[key])
        check_file(target.parent / 'basis.npz', trial['basis_sha256'])
        history = read(target / 'curve.json')
        if (len(history) != trial['objective_calls'] or trial['fit_label_exposures'] != len(history)*plan['training_positions']
                or trial['iterations'] > plan['optimizer']['maxiter']
                or trial['solver_evaluations'] + 2 != trial['objective_calls']):
            raise ValueError('Solver budget or label accounting differs')
        np.testing.assert_allclose([history[0]['objective'], history[-1]['objective']],
                                   [trial['initial_objective'], trial['objective']], rtol=0, atol=1e-12)
        gap = trial['optimization_gap_bound']
        if not np.isfinite(gap) or gap < 0:
            raise ValueError('Invalid optimization error bound')
        if trial['sufficiently_converged'] != (gap <= plan['acceptance']['max_gradient_gap_bound']):
            raise ValueError('Convergence classification differs')
        with np.load(target / 'checkpoint.npz', allow_pickle=False) as data:
            penalty = trial['ridge']/2*sum(float(np.sum(data[key]**2)) for key in ('weight', 'bias'))
        np.testing.assert_allclose(penalty, trial['penalty'], rtol=0, atol=1e-12)
        with np.load(target / 'position-metrics.npz', allow_pickle=False) as data:
            for split in ('train', 'validation'):
                for key in METRICS:
                    values = data[split + '/' + key]
                    expected_size = plan['training_positions' if split == 'train' else 'validation_positions']
                    if values.shape != (expected_size,) or not np.isfinite(values).all():
                        raise ValueError('Invalid per-position metrics')
                    np.testing.assert_allclose(values.mean(), trial['metrics'][split][key], rtol=0, atol=1e-12)
            metrics = {key: data['validation/' + key] for key in METRICS}
            totals.append(family_totals(metrics, membership, len(unique)))
        ce = trial['metrics']['train']['policy_kl'] + trial['metrics']['train']['teacher_entropy']
        np.testing.assert_allclose(trial['objective'], ce + penalty, rtol=0, atol=1e-11)
        receipts.append(dict(variant=trial['variant'], ridge=trial['ridge'],
                             curve_sha256=sha256(target / 'curve.json'), result_sha256=sha256(target / 'result.json')))
    return dict(status='complete', seed=seed, plan_sha256=sha256(plan_path), source=source, bank=bank,
                analysis_sha256=sha256(Path(__file__)),
                result=report, result_sha256=sha256(directory / 'result.json'), receipts=receipts,
                families=unique.tolist(), counts=counts.tolist(), baseline_totals=baseline, trial_totals=totals)


def combine(plan_path, reports):
    plan = read(plan_path)
    seeds = [case['seed'] for case in plan['banks']]
    if sorted(r['seed'] for r in reports) != sorted(seeds):
        raise ValueError('Exactly the registered seeds are required')
    reports = sorted(reports, key=lambda r: seeds.index(r['seed']))
    first = reports[0]
    for report in reports:
        if (report['status'] != 'complete' or report['plan_sha256'] != sha256(plan_path)
                or report['analysis_sha256'] != sha256(Path(__file__))):
            raise ValueError('Unqualified or changed analysis source')
        if report['families'] != first['families'] or report['counts'] != first['counts']:
            raise ValueError('Validation families are not paired')
        for split in ('train', 'validation'):
            if report['bank']['selection'][split]['indices_sha256'] != first['bank']['selection'][split]['indices_sha256']:
                raise ValueError('Selected positions differ between cores')
        for key in ('worker_sha256', 'convex_sha256', 'probe_sha256', 'scipy', 'numpy', 'lbfgsb_sha256', 'lbfgsb_python_sha256'):
            if report['source'][key] != first['source'][key]:
                raise ValueError('Fitting implementations differ')
    counts = np.asarray(first['counts'])
    samples = np.random.default_rng(709).integers(len(counts), size=(1000, len(counts)))

    def summarize(totals):
        totals = np.asarray(totals)
        means = totals.sum(axis=1)/counts.sum()
        average = totals.mean(axis=0)
        distribution = average[samples].sum(axis=1)/counts[samples].sum(axis=1)[:, None]
        intervals = np.quantile(distribution, [.025, .975], axis=0)
        return {key: dict(mean=float(means[:, i].mean()), per_seed=means[:, i].tolist(),
                          seed_sd=float(means[:, i].std(ddof=1)), family_bootstrap_95=intervals[:, i].tolist())
                for i, key in enumerate(METRICS)}

    baseline = np.asarray([r['baseline_totals'] for r in reports])
    values = np.asarray([r['trial_totals'] for r in reports])
    keys = [(v, ridge) for v in plan['variants'] for ridge in plan['ridge']]
    records = []
    for i, (variant, ridge) in enumerate(keys):
        trials = [r['result']['trials'][i] for r in reports]
        bias = values[:, keys.index(('bias-only', ridge))]
        records.append(dict(variant=variant, ridge=ridge, validation=summarize(values[:, i]),
            minus_original=summarize(values[:, i]-baseline), minus_bias_only=summarize(values[:, i]-bias),
            train={key:float(np.mean([t['metrics']['train'][key] for t in trials])) for key in METRICS},
            converged=[t['sufficiently_converged'] for t in trials],
            gap_bounds=[t['optimization_gap_bound'] for t in trials],
            solver_success=[t['optimizer_success'] for t in trials],
            iterations=[t['iterations'] for t in trials],
            fit_label_exposures=sum(t['fit_label_exposures'] for t in trials)))
    return dict(status='complete', created=time.time(), plan=plan, plan_sha256=sha256(plan_path), seeds=seeds,
        source_sha256=sha256(Path(__file__)), baseline=summarize(baseline), records=records,
        owner_reports=reports, complete_fits=len(keys)*len(seeds),
        converged_fits=sum(sum(r['converged']) for r in records),
        fit_label_exposures=sum(r['fit_label_exposures'] for r in records),
        uncertainty='1000 paired opening-family bootstrap resamples, seed 709, after averaging across fitted seeds; conditional on those weights. Training-seed SD is separate. Validation-informed diagnostic, not independent confirmation.',
        scope=plan['scope'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['owner', 'combine'])
    parser.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--owners', type=Path, nargs='*')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pin([56, 57, 58, 59])
    args.output.resolve().relative_to(args.root.resolve())
    if args.output.exists():
        raise ValueError('Analysis reports are immutable')
    with StorageBudget(args.root).reserve(files=16 << 20, heap=3*GIB, purpose='audit convex decoder endpoints'):
        result = (owner(args.root, args.study, args.seed) if args.mode == 'owner'
                  else combine(args.study / 'plan.json', [read(p) for p in args.owners]))
        atomic_json(args.output, result)
        print(json.dumps(dict(status=result['status'], output=str(args.output))))


if __name__ == '__main__':
    main()
