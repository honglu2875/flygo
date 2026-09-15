#!/usr/bin/env python3
"""Reduce complete, source-verified signal audits without accessing new labels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.storage import StorageBudget


GROUPS = ('all', 'visual_sensors', 'context_sensors', 'motors', 'other')
CORE_PARAMETERS = ('edge', 'bias', 'leak', 'input_gain')
COMPARISONS = {*CORE_PARAMETERS, 'readout_gain'}


def checked(rows, expected):
    if len(rows) != expected or any(not row['passed'] or row['mismatches'] for row in rows):
        raise ValueError('Missing or failed numerical comparisons')
    if any(not math.isfinite(row[key]) or row[key] < 0
           for row in rows for key in ('max_abs', 'relative_l2')):
        raise ValueError('Invalid numerical comparison errors')
    return dict(comparisons=len(rows), max_abs=max(row['max_abs'] for row in rows),
                max_relative_l2=max(row['relative_l2'] for row in rows))


def signal(row):
    visual = row['visual_std']; total = row['std']; active = row['positive_fraction']
    denominator = visual['count'] * visual['rms'] ** 2
    return dict(nodes=visual['count'], rms_visual_std=visual['rms'], rms_total_std=total['rms'],
                visual_std_median=visual['quantiles'][2], visual_std_p90=visual['quantiles'][3],
                visual_std_p99=visual['quantiles'][4], visual_std_max=visual['quantiles'][5],
                leading_visual_variance_share=visual['quantiles'][5] ** 2 / denominator if denominator else None,
                positive_node_position_fraction=active['mean'], nodes_ever_positive=active['nonzero'])


def condense(report, plan):
    if (report['status'] != 'complete' or report['quantiles'] != [0., .1, .5, .9, .99, 1.]
            or [r['step'] for r in report['records']] != plan['checkpoint_steps']):
        raise ValueError('Incomplete audit or different summary conventions')
    records = []
    for row in report['records']:
        if not row['parameters_unchanged']:
            raise ValueError('Audit changed scientific parameters')
        expected = [name for name in plan['gradient_cases']
                    if row['step'] or name != 'fitted-linear-policy']
        if ([g['objective'] for g in row['gradients']] != expected
                or [s['step'] for s in row['signals']] != list(range(plan['passes'] + 1))):
            raise ValueError('Missing objective or recurrent pass')
        gradients = []
        for gradient in row['gradients']:
            if (set(gradient['comparisons']) != COMPARISONS
                    or [s['step'] for s in gradient['cotangents']] != list(range(plan['passes'] + 1))):
                raise ValueError('Incomplete adjoint evidence')
            gradients.append(dict(objective=gradient['objective'], loss=gradient['loss'],
                qualification=checked(list(gradient['comparisons'].values()), len(COMPARISONS)),
                core_gradients={key: gradient['native_gradients'][key] for key in CORE_PARAMETERS},
                edge_path_gradients=gradient['edge_path_gradients'],
                trace=[dict(step=s['step'], groups={key: s['groups'][key]['rms']['rms'] for key in GROUPS})
                       for s in gradient['cotangents']],
                input_drive_groups={key: gradient['input_drive_groups'][key] for key in GROUPS}))
        records.append(dict(step=row['step'], checkpoint_sha256=row['checkpoint_sha256'],
            forward_qualification=checked(row['forward_reconstruction'], plan['passes']),
            signals=[dict(step=s['step'], groups={key: signal(s['groups'][key]) for key in GROUPS})
                     for s in row['signals']],
            final_superclasses={key: signal(value) for key, value in row['signals'][-1]['groups'].items()
                                if key.startswith('superclass/')},
            weights={key: row['weights'][key] for key in GROUPS},
            jacobian_row_bound=[dict(step=s['step'], groups={key: s['groups'][key] for key in GROUPS})
                                for s in row['jacobian_row_bound']],
            gradients=gradients, adam_moments={key: row['adam_moments'][key] for key in CORE_PARAMETERS},
            parameter_changes={key: row['parameter_changes'][key] for key in CORE_PARAMETERS}))
    return dict(seed=report['seed'], source=report['source'], records=records, seconds=report['seconds'],
                potential_visual_path_edges=report['visual_path_edges'],
                truncated_motor_eye_distance=report['motor_eye_distance'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    parser.add_argument('--study', type=Path, default=Path('runs/signal-flow-v2'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root = args.root; base = root / args.study
    args.output.resolve().relative_to(root.resolve())
    if args.output.exists():
        raise ValueError('Analysis reports are immutable')
    plan = json.loads((base / 'plan.json').read_text())
    sources = json.loads((base / 'source.json').read_text())['files']
    collection = json.loads((base / 'collection-v1.json').read_text())
    if collection['status'] != 'complete':
        raise ValueError('Incomplete owner collection')
    for name, digest in sources.items():
        if sha256(base / name) != digest:
            raise ValueError('Frozen audit source changed: ' + name)
    with StorageBudget(root).reserve(files=4 << 20, heap=64 << 20, purpose='summarize signal-flow audit'):
        records = []; receipts = []
        for case in plan['cases']:
            seed = case['seed']; path = base / f'seed-{seed}' / 'result.json'
            receipt = next(r for r in collection['records'] if r['seed'] == seed)
            report = json.loads(path.read_text()); source = report['source']
            if (sha256(path) != receipt['result_sha256'] or receipt['host'] != case['host']
                    or report['seed'] != seed or source['plan_sha256'] != sources['plan.json']
                    or source['worker_sha256'] != sources['worker.py']
                    or source['helper_sha256'] != sources['signal_flow.py']
                    or source['bank_sha256'] != case['bank_sha256']):
                raise ValueError('Owner report or implementation identity differs')
            if records and source['runtime_sha256'] != records[0]['source']['runtime_sha256']:
                raise ValueError('Scientific runtime differs between seeds')
            records.append(condense(report, plan)); receipts.append(receipt)
        atomic_json(args.output, dict(status='complete', name=plan['name'], plan=plan, records=records,
            owner_receipts=receipts, plan_sha256=sha256(base / 'plan.json'), source_sha256=sha256(Path(__file__)),
            interpretation=[
                'Training-only diagnostic: 128 fixed views for signals, first 32 for gradients. No parameters updated; no new decoder fits or validation/test access.',
                'Visual response is current-eye rate minus neutral-gray-eye rate with the same context. Its centered variance is not an additive share of total variance; neutral combinations may be out of distribution.',
                'Positive activity is a node-position fraction, not the count of motors that vary across the complete dataset. A zero derivative here does not mean a neuron is permanently dead.',
                'The fitted raw-motor residual yields correct core cotangents through a fixed inverse readout-gain conversion. Its API readout-gain derivative belongs to that local surrogate, not the composite raw-residual objective; exclude it from scientific gradient comparisons.',
                'All five native-versus-reference VJP comparisons are retained, including that auxiliary gain derivative. Task heads and gains are held fixed; core gradients and state cotangents are interpreted.',
                'Raw single-objective gradients exclude joint-loss clipping and moment history. sqrt(vhat)/(sqrt(vhat)+epsilon) describes the saved Adam denominator, not an actual proposed update.',
                'Path distances are truncated at seven hops; -1 is outside that horizon, not globally unreachable. A potential path ignores signs, cancellation and ReLU gates.',
                'Incoming L2 and effective fan-in are descriptive. They imply variance transfer only under additional input-covariance assumptions. Row sums bound a local Jacobian, not long-horizon stability.',
                'No causal optimizer/propagation effect, biological function, generalization gain, playing strength or matched-CNN benefit is established.']))
        print(json.dumps(dict(status='complete', seeds=len(records), output=str(args.output))))


if __name__ == '__main__':
    main()
