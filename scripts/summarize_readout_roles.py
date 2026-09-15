#!/usr/bin/env python3
"""Close the four-arm readout study from aligned, owner-verified CPU evidence."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.feather as feather

from compare_checkpoints import summarize
from summarize_attachment_confirmation import concentration
from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.readout import load_head_mask
from flygo.runtime import pin
from flygo.storage import GIB, StorageBudget


METRICS = ('policy_kl', 'value_mse', 'teacher_top1_agreement')
SLICES = ('natural', 'current_reduced_source_novel')


def read(path):
    return json.loads(path.read_text())


def complete(path):
    report = read(path)
    if report.get('status') != 'complete':
        raise ValueError('Incomplete evidence: ' + str(path))
    return report


def aligned_metrics(path, indices, family_names, expected_families=None):
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['indices'], indices)
        games = data['game_index']; metrics = data['metrics']
    if (games.shape != indices.shape or games.dtype.kind not in 'iu'
            or np.any(games < 0) or np.any(games >= len(family_names))
            or metrics.shape != (len(indices), 3) or not np.isfinite(metrics).all()):
        raise ValueError('Invalid validation metrics or game identities')
    families = family_names[games]
    if expected_families is not None:
        np.testing.assert_array_equal(families, expected_families)
    return metrics, families


def paired_results(matrices, families, novel, seeds, arms, contrasts, *, seed=709, bootstrap=1000):
    """Average paired predictions, retaining seed variation outside the family CI."""
    if set(matrices) != {(s, a) for s in seeds for a in arms} or len(seeds) < 2:
        raise ValueError('Every registered arm must occur once per paired seed')
    if novel.dtype != np.bool_ or novel.shape != families.shape or not novel.any():
        raise ValueError('Expected a nonempty aligned source-novel slice')
    if any(m.shape != (len(families), 3) or not np.isfinite(m).all() for m in matrices.values()):
        raise ValueError('Invalid paired metric matrices')

    def slices(matrix):
        result = {}
        for name, mask in zip(SLICES, (slice(None), novel)):
            row = summarize(matrix[mask], families[mask], seed=seed, bootstrap=bootstrap)
            row['families'] = row.pop('games')
            for key in METRICS:
                row[key]['family_bootstrap_95'] = row[key].pop('game_bootstrap_95')
            result[name] = row
        return result

    def combine(values):
        variation = {}
        for name, mask in zip(SLICES, (slice(None), novel)):
            means = np.stack([value[mask].mean(axis=0) for value in values])
            variation[name] = {key: dict(per_seed=means[:, i].tolist(),
                sample_sd=float(means[:, i].std(ddof=1))) for i, key in enumerate(METRICS)}
        return dict(**slices(np.mean(values, axis=0)), seed_variation=variation)

    records = [dict(seed=s, arm=a, **slices(matrices[s, a])) for s in seeds for a in arms]
    paired = []; pooled = []
    for contrast in contrasts:
        candidate, reference = contrast['candidate'], contrast['reference']
        values = [matrices[s, candidate] - matrices[s, reference] for s in seeds]
        paired.extend(dict(seed=s, **contrast, **slices(value)) for s, value in zip(seeds, values))
        pooled.append(dict(**contrast, **combine(values)))
    return dict(records=records, contrasts=paired,
                pooled_records=[dict(arm=a, **combine([matrices[s, a] for s in seeds])) for a in arms],
                pooled_contrasts=pooled, seeds=seeds, contrast_direction='candidate minus reference')


def diagnostics(root, base, endpoint, job, wave, analysis, annotation, expected_selection, expected_motors, expected_count):
    digest = endpoint['checkpoint_sha256']; signal = complete(base / 'signal/result.json')
    selection = read(base / 'signal/selection.json')
    if (selection != expected_selection or signal['dataset_id'] != endpoint['dataset_id']
            or signal['plan_sha256'] != wave['sha256'] or len(selection['position_indices']) != analysis['diagnostics']['positions']):
        raise ValueError('Motor probe source or selection differs')
    if [r['case'] for r in signal['records']] != ['initial', 'current']:
        raise ValueError('Expected initial and final current-input motor probes')
    for row, receipt in zip(signal['records'], endpoint['checkpoint_receipts']):
        if (row['checkpoint_sha256'] != receipt['sha256'] or row['optimizer_step'] != receipt['step']
                or sha256(base / 'signal' / (row['case'] + '-responses.npz')) != row['response_file_sha256']):
            raise ValueError('Motor response provenance differs')
    concentrated, motors = concentration(base / 'signal/current-responses.npz', 'current', annotation, expected_motors)
    count = complete(base / 'count/result.json')
    expected_plan = dict(release=job['release'], positions=64, batches=[1, 32], sample_seed=917271,
        cases=[dict(name=job['run_id'], checkpoint=f'runs/{job["run_id"]}/checkpoints/step-{analysis["endpoint_update"]:08d}.npz')])
    if (count['dataset_id'] != endpoint['dataset_id'] or count['graph_id'] != endpoint['graph_id']
            or count['native_sha256'] != analysis['native_sha256'] or count['plan'] != expected_plan
            or count['indices_sha256'] != sha256(base / 'count/indices.npy') or len(count['results']) != 1):
        raise ValueError('Arithmetic source or sample contract differs')
    counted_indices = np.load(base / 'count/indices.npy', allow_pickle=False)
    if counted_indices.shape != (64,):
        raise ValueError('Wrong arithmetic sample size')
    if expected_count is not None:
        np.testing.assert_array_equal(counted_indices, expected_count)
    counted = count['results'][0]
    if counted['checkpoint_sha256'] != digest or counted['head_mask'] != endpoint['head_mask']:
        raise ValueError('Counted decoder differs')
    if [r['batch_size'] for r in counted['records']] != [1, 32]:
        raise ValueError('Both declared prediction batches are required')
    for record in counted['records']:
        expected_heads = 2 * sum(endpoint['head_mask'][key] for key in ('policy_coefficients', 'value_coefficients'))
        if record['nominal']['parts']['heads'] != expected_heads or 'external_visual_encoding' not in record['nominal']['parts']:
            raise ValueError('Decoder or rendering work omitted')
        batch = record['batch_size']
        if [r['offset'] for r in record['batches']] != list(range(0, 64, batch)):
            raise ValueError('Incomplete counted batches')
        measured = np.asarray([r['cached_zero_skipped_flops_per_position'] for r in record['batches']])
        if not np.isfinite(measured).all() or np.any(measured <= 0):
            raise ValueError('Invalid complete prediction counts')
        np.testing.assert_allclose(measured.mean(), record['cached_zero_skipped_flops_per_position']['mean'], rtol=0, atol=1e-8)
    return dict(visual_signal=signal['records'][-1], visual_concentration=concentrated,
                counted_inference=counted, count_record_sha256=sha256(base / 'count/result.json'),
                signal_record_sha256=sha256(base / 'signal/result.json'),
                selection_sha256=sha256(base / 'signal/selection.json')), motors, counted_indices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); root = args.root; analysis = read(args.plan); pin([56, 57, 58, 59])
    args.output.resolve().relative_to(root.resolve())
    if args.output.exists():
        raise ValueError('Analysis reports are immutable')
    if (tuple(analysis['metrics']) != METRICS or tuple(analysis['slices']) != SLICES
            or analysis['bootstrap'] != dict(unit='opening_family', resamples=1000, seed=709, interval=[.025, .975])
            or analysis['diagnostics'] != dict(positions=256, selection_seed=972091, motor_std_floor=1e-6, leading_cell_counts=[1, 8])):
        raise ValueError('This analyzer implements only the declared readout analysis rules')
    jobs = []; plans = []
    for wave in analysis['wave_plans']:
        path = root / wave['path']
        if sha256(path) != wave['sha256']:
            raise ValueError('Deployed wave plan differs')
        plan = read(path); plans.append(plan)
        if plan['updates'] != analysis['endpoint_update'] or plan['updates'] * plan['batch_size'] != analysis['labeled_exposures_per_endpoint']:
            raise ValueError('Registered exposure horizon differs')
        jobs.extend((dict(job, release=plan['release']), wave, plan) for job in plan['jobs'])
    if sorted((job['seed'], job['arm']) for job, _, _ in jobs) != sorted(
            (seed, arm) for seed in analysis['seeds'] for arm in analysis['arms']):
        raise ValueError('The complete four-arm paired study is required')
    base_contract = {k: v for k, v in plans[0].items() if k not in ('jobs', 'name', 'wave')}
    if any({k: v for k, v in plan.items() if k not in ('jobs', 'name', 'wave')} != base_contract for plan in plans):
        raise ValueError('The two waves changed a shared scientific contract')
    with StorageBudget(root).reserve(files=16 << 20, heap=3 * GIB, purpose='paired readout role analysis'):
        manifest = read(root / 'releases' / plans[0]['release'] / 'manifest.json'); dataset = manifest['dataset_id']
        family_names = np.asarray([r['opening_family'] for r in manifest['records']])
        path = root / analysis['novelty_masks']; novelty = complete(path.with_name('result.json'))
        if sha256(path) != analysis['novelty_masks_sha256'] or novelty['masks_sha256'] != sha256(path) or novelty['dataset_id'] != dataset:
            raise ValueError('Source-novelty receipt differs')
        with np.load(path, allow_pickle=False) as data:
            indices = data['indices']; novel = data['current']
        if len(indices) != analysis['validation_positions'] or int(novel.sum()) != analysis['current_source_novel_positions']:
            raise ValueError('Novelty slice differs')
        graph = Path(read(root / 'runs/m4/graph.json')['path'])
        annotation = feather.read_table(graph / 'annotations.feather', columns=['bodyId', 'type', 'superclass', 'subclass'])
        selection = read(root / 'runs/retinal-allocation-signal-v1/selection.json')
        records = []; matrices = {}; families = motors = counted_indices = shared = None
        for job, wave, plan in jobs:
            run = job['run_id']; seed = job['seed']; arm = job['arm']; endpoint_dir = root / wave['endpoints']
            collection = complete(endpoint_dir / 'result.json')
            receipt = next(r for r in collection['records'] if r['run_id'] == run)
            endpoint_path = root / receipt['path']; endpoint = complete(endpoint_path)
            if sha256(endpoint_path) != receipt['sha256'] or endpoint['source_sha256'] != collection['helper_sha256']:
                raise ValueError('Owner audit bytes differ')
            mask_path = root / job['head_mask']
            if sha256(mask_path) != job['head_mask_sha256']:
                raise ValueError('Registered head mask changed')
            mask = load_head_mask(mask_path, graph_id=endpoint['graph_id'], attachment_sha256=plan['input_map_sha256'])
            if (endpoint['run_id'] != run or endpoint['seed'] != seed or endpoint['mode'] != 'current'
                    or endpoint['dataset_id'] != dataset or endpoint['head_mask'] != mask.contract
                    or endpoint['plan_sha256'] != wave['sha256'] or endpoint['native_sha256'] != analysis['native_sha256']
                    or endpoint['numerical_runtime'] != analysis['numerical_runtime']
                    or endpoint['labeled_training_exposures'] != analysis['labeled_exposures_per_endpoint']
                    or [r['step'] for r in endpoint['checkpoint_receipts']] != [0, analysis['endpoint_update']]
                    or any(r['replica_status'] != 'verified' for r in endpoint['checkpoint_receipts'])):
                raise ValueError('Endpoint contract differs')
            contract = {k: endpoint[k] for k in ('dataset_id', 'graph_id', 'training_contract', 'input_contract')}
            contract['model'] = {k: v for k, v in endpoint['model_config'].items() if k != 'seed'}
            if shared is None: shared = contract
            elif contract != shared: raise ValueError('A shared input, model or optimizer factor changed')
            base = root / wave['followup'] / run; status = read(base / 'status.json')
            if status['state'] != 'complete' or status['checkpoint_sha256'] != endpoint['checkpoint_sha256']:
                raise ValueError('Followup evidence is incomplete')
            report = complete(base / 'validation/result.json')
            if len(report['records']) != 1: raise ValueError('Expected exactly one final checkpoint')
            row = report['records'][0]; digest = endpoint['checkpoint_sha256']
            if (row['sha256'] != digest or row['dataset_id'] != dataset or row['optimizer_step'] != analysis['endpoint_update']
                    or row['model'] != endpoint['model_config']):
                raise ValueError('Validated endpoint differs')
            metrics_path = base / 'validation' / f'step-{analysis["endpoint_update"]:08d}-{digest[:12]}-metrics.npz'
            metrics, families = aligned_metrics(metrics_path, indices, family_names, families)
            np.testing.assert_allclose(metrics.mean(axis=0), [row['slices']['natural'][key]['mean'] for key in METRICS], rtol=0, atol=1e-12)
            matrices[seed, arm] = metrics
            diagnostic, motors, counted_indices = diagnostics(root, base, endpoint, job, wave, analysis, annotation, selection, motors, counted_indices)
            records.append(dict(endpoint, arm=arm, **diagnostic, endpoint_record_sha256=sha256(endpoint_path),
                metrics_sha256=sha256(metrics_path), validation_record_sha256=sha256(base / 'validation/result.json')))
        paired = paired_results(matrices, families, novel, analysis['seeds'], analysis['arms'], analysis['contrasts'])
        costs = {(r['seed'], r['arm']): r['counted_inference']['records'] for r in records}
        cost_contrasts = []
        for contrast in analysis['contrasts']:
            for batch_index, batch in enumerate((1, 32)):
                differences = [costs[s, contrast['candidate']][batch_index]['cached_zero_skipped_flops_per_position']['mean']
                             - costs[s, contrast['reference']][batch_index]['cached_zero_skipped_flops_per_position']['mean'] for s in analysis['seeds']]
                cost_contrasts.append(dict(**contrast, batch_size=batch, per_seed=differences,
                    mean=float(np.mean(differences)), sample_sd=float(np.std(differences, ddof=1))))
        helpers = ('compare_checkpoints.py', 'summarize_attachment_confirmation.py', 'summarize_attachments.py')
        atomic_json(args.output, dict(status='complete', created=time.time(), plan=analysis, records=records,
            statistics=paired, cost_contrasts=cost_contrasts, dataset_id=dataset, probe_selection=selection,
            labeled_training_exposures=len(records) * analysis['labeled_exposures_per_endpoint'],
            novelty=novelty, source_sha256=sha256(Path(__file__)), plan_sha256=sha256(args.plan),
            helper_sha256={name: sha256(Path(__file__).with_name(name)) for name in helpers},
            annotation_sha256=sha256(graph / 'annotations.feather'),
            uncertainty=analysis['uncertainty'], limitations=[
                'Three fresh paired head/sampler seeds; initial core strengths are shared. Short, fixed 32k-exposure screen.',
                'Opening-family intervals condition on fitted weights. All four contrasts are reported; intervals are not multiplicity-corrected.',
                'Current-source novelty precedes retinal rendering; rendered collision absence is not certified.',
                'Motor variance and decoder influence do not establish biological function or predictive usefulness.',
                'Complete unpruned prediction arithmetic includes the renderer and masked heads; excludes backward, optimizer, memory, nonlinear functions and search. Trace timing is not production latency.',
                'Earlier candidate discovery, input selection, numerical qualification and diagnostic work remain development costs. No CNN advantage, final-test result, match result or TPU use is established.']))
        print(json.dumps(dict(status='complete', records=len(records), output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
