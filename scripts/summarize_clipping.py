#!/usr/bin/env python3
"""Close a registered optimizer study; keep confidence, seed and family effects separate."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.feather as feather

from compare_checkpoints import summarize
from summarize_readout_roles import (METRICS, SLICES, aligned_metrics, complete,
                                    diagnostics, paired_results, read)
from flygo.data.corpus import atomic_json
from flygo.fly import PARAMETERS, RustFly
from flygo.qualify import sha256
from flygo.readout import load_head_mask
from flygo.runtime import pin
from flygo.storage import GIB, StorageBudget


CONFIDENCE = ('policy_entropy', 'mean_top_probability')
VALUE_CORE_FACTOR = 'supervised value-to-circuit gradient scale only'


def contrasts(analysis):
    return analysis['contrasts'] if 'contrasts' in analysis else [analysis['contrast']]


def optimizer_contract(analysis, arm):
    """Return the permitted varying field and the exact arm's optimizer settings."""
    if analysis.get('factor') == VALUE_CORE_FACTOR:
        return 'value_core_scale', analysis['clip_mode'], float(np.float32(analysis['value_core_arms'][arm]))
    return 'clip_mode', arm, 1.0


def validate_contract(analysis):
    """Keep the statistical rules fixed while accepting registered seeds and horizons."""
    seeds = analysis['seeds']; update = analysis['endpoint_update']
    if (len(seeds) < 2 or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds) or type(update) is not int or update <= 0 or update % 10
            or tuple(analysis['metrics']) != METRICS or tuple(analysis['confidence_metrics']) != CONFIDENCE
            or tuple(analysis['slices']) != SLICES
            or analysis['bootstrap'] != dict(unit='opening_family', resamples=1000, seed=709, interval=[.025, .975])):
        raise ValueError('Unsupported clipping analysis contract')
    if analysis.get('factor') == VALUE_CORE_FACTOR:
        scales = {k: float(np.float32(v)) for k, v in analysis['value_core_arms'].items()}
        if (analysis['arms'] != ['full', 'reduced', 'policy-only']
                or scales != dict(full=1., reduced=float(np.float32(.1)), **{'policy-only': 0.})
                or analysis['clip_mode'] != 'parameter-group'
                or contrasts(analysis) != [dict(candidate=a, reference='full') for a in ('reduced', 'policy-only')]):
            raise ValueError('Unsupported value core analysis contract')
    elif (analysis.get('factor', 'optimizer clipping scope only') != 'optimizer clipping scope only'
            or analysis['arms'] != ['global', 'parameter-group']
            or contrasts(analysis) != [dict(candidate='parameter-group', reference='global')]):
        raise ValueError('Unsupported clipping analysis contract')


def paired_confidence(matrices, families, novel, seeds, arms, contrast, *, bootstrap=1000, seed=709):
    """The same position weighting and paired-family bootstrap, for two confidence columns."""
    if set(matrices) != {(s, a) for s in seeds for a in arms} or len(seeds) < 2:
        raise ValueError('Every registered seed and arm is required')
    if novel.dtype != np.bool_ or novel.shape != families.shape or not novel.any():
        raise ValueError('Invalid source-novel selection')
    if any(m.shape != (len(families), 2) or not np.isfinite(m).all() for m in matrices.values()):
        raise ValueError('Invalid confidence metrics')

    def combine(values):
        result = {}; variation = {}
        for name, mask in zip(SLICES, (slice(None), novel)):
            selected = [value[mask] for value in values]
            row = summarize(np.mean(selected, axis=0), families[mask], names=CONFIDENCE,
                            bootstrap=bootstrap, seed=seed)
            row['families'] = row.pop('games')
            means = np.stack([value.mean(axis=0) for value in selected])
            variation[name] = {}
            for i, metric in enumerate(CONFIDENCE):
                row[metric]['family_bootstrap_95'] = row[metric].pop('game_bootstrap_95')
                variation[name][metric] = dict(per_seed=means[:, i].tolist(),
                    sample_sd=float(means[:, i].std(ddof=1)) if len(values) > 1 else None)
            result[name] = row
        return dict(**result, seed_variation=variation)

    comparisons = [contrast] if isinstance(contrast, dict) else contrast
    if (not comparisons or any(c['candidate'] not in arms or c['reference'] not in arms
                              or c['candidate'] == c['reference'] for c in comparisons)):
        raise ValueError('Invalid confidence contrasts')
    paired = []; pooled = []
    for comparison in comparisons:
        differences = [matrices[s, comparison['candidate']] - matrices[s, comparison['reference']] for s in seeds]
        paired.extend(dict(seed=s, **comparison, **combine([d])) for s, d in zip(seeds, differences))
        pooled.append(dict(**comparison, **combine(differences)))
    return dict(records=[dict(seed=s, arm=a, **combine([matrices[s, a]])) for s in seeds for a in arms],
        pooled_records=[dict(arm=a, **combine([matrices[s, a] for s in seeds])) for a in arms],
        contrasts=paired, pooled_contrasts=pooled, seeds=seeds,
        contrast_direction='candidate minus reference')


def clipping_history(log, mode, updates, clip):
    """Audit logged pre-update factors; aggregate all-update clipping is a distinct counter."""
    rows = [row for row in log if row.get('kind') == 'train']
    if [row['step'] for row in rows] != [1, *range(10, updates + 1, 10)]:
        raise ValueError('Registered training log steps are incomplete')
    norms = []; factors = []
    for row in rows:
        record = row['clipping']
        if record['mode'] != mode or set(record['group_norms']) != set(PARAMETERS) or set(record['factors']) != set(PARAMETERS):
            raise ValueError('Clipping scope or group identities differ')
        n = np.asarray([record['group_norms'][key] for key in PARAMETERS])
        f = np.asarray([record['factors'][key] for key in PARAMETERS], dtype=np.float32)
        if not np.isfinite(n).all() or np.any(n < 0) or not np.isfinite(row['gradient_norm']):
            raise ValueError('Invalid pre-update norm')
        np.testing.assert_allclose(np.linalg.norm(n), row['gradient_norm'], rtol=2e-15, atol=1e-15)
        expected = np.minimum(1., clip / np.maximum(n if mode == 'parameter-group' else row['gradient_norm'], 1e-30)).astype(np.float32)
        np.testing.assert_array_equal(f, np.broadcast_to(expected, f.shape))
        if record['clipped'] != bool(np.any(f < 1)) or row['clipped'] != record['clipped']:
            raise ValueError('Clipping event does not match applied factors')
        norms.append(n); factors.append(f)
    norms = np.stack(norms); factors = np.stack(factors)
    fraction = rows[-1]['clipping_fraction']
    if not 0 <= fraction <= 1 or abs(fraction * updates - round(fraction * updates)) > 1e-8:
        raise ValueError('Invalid all-update clipping counter')
    seconds = np.asarray([row['seconds'] for row in rows])
    if not np.isfinite(seconds).all() or np.any(seconds <= 0):
        raise ValueError('Invalid sampled training durations')
    return dict(mode=mode, logged_steps=[row['step'] for row in rows], all_update_clipped_fraction=fraction,
        groups={name: dict(logged_clipped_fraction=float((factors[:, i] < 1).mean()),
            factor_quantiles=np.quantile(factors[:, i], [0, .5, 1]).tolist(),
            pre_update_norm_quantiles=np.quantile(norms[:, i], [0, .5, 1]).tolist()) for i, name in enumerate(PARAMETERS)},
        logged_update_seconds_median=float(np.median(seconds)), final_pre_update=rows[-1]['clipping'],
        scope='Group distributions cover logged steps 1,10,20,... only. The all-update counter covers the entire run. Update timing includes forward/backward and is not isolated optimizer overhead.')


def decision(statistics, index=0):
    contrast = statistics['pooled_contrasts'][index]
    variation = contrast['seed_variation']['natural']
    policy = variation['policy_kl']['per_seed']
    value = contrast['natural']['value_mse']['mean']
    return dict(policy_improves_each_seed=all(delta < 0 for delta in policy),
                mean_value_nonworse=value <= 0,
                provisional_candidate=all(delta < 0 for delta in policy) and value <= 0,
                paired_policy_differences=policy, mean_value_difference=value,
                rule='Natural validation KL improves in every paired seed, with nonworse mean value MSE; fixed endpoints only.')


def study_decision(statistics):
    if len(statistics['pooled_contrasts']) == 1:
        return decision(statistics)
    return dict(contrasts=[dict(candidate=row['candidate'], reference=row['reference'], **decision(statistics, i))
                for i, row in enumerate(statistics['pooled_contrasts'])],
                rule='Evaluate both preregistered candidates independently. Report every seed and value tradeoff; any candidate still needs fresh-seed confirmation.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); root = args.root; pin([56, 57, 58, 59])
    args.output.resolve().relative_to(root.resolve())
    if args.output.exists(): raise ValueError('Analysis reports are immutable')
    analysis = read(args.plan)
    validate_contract(analysis)
    if (analysis.get('analysis_implementation_sha256')
            and sha256(Path(__file__)) != analysis['analysis_implementation_sha256']):
        raise ValueError('Registered analysis implementation changed')
    for helper, digest in analysis['statistics_implementations'].items():
        if sha256(Path(__file__).with_name(helper)) != digest: raise ValueError('Registered statistics source changed')
    plan_path = root / analysis['trial_plan']; plan = read(plan_path)
    if sha256(plan_path) != analysis['trial_plan_sha256']: raise ValueError('Trial contract changed')
    if (plan['updates'] != analysis['endpoint_update']
            or plan['updates'] * plan['batch_size'] != analysis['labeled_exposures_per_endpoint']):
        raise ValueError('Exposure horizon differs')
    jobs = plan['jobs']; seeds = analysis['seeds']; arms = analysis['arms']
    if sorted((j['seed'], j['arm']) for j in jobs) != sorted((s, a) for s in seeds for a in arms):
        raise ValueError('Each registered seed/arm must occur once')
    collection = complete(root / analysis['endpoints'] / 'result.json')
    if {r['run_id'] for r in collection['records']} != {j['run_id'] for j in jobs}:
        raise ValueError('Endpoint evidence is incomplete')
    wave = dict(sha256=sha256(plan_path))
    with StorageBudget(root).reserve(files=8 << 20, heap=3 * GIB, purpose='paired clipping endpoint analysis'):
        manifest = read(root / 'releases' / plan['release'] / 'manifest.json')
        dataset = manifest['dataset_id']
        if dataset != analysis['dataset_id']: raise ValueError('Dataset differs')
        family_names = np.asarray([r['opening_family'] for r in manifest['records']])
        novelty_path = root / analysis['novelty_masks']; novelty = complete(novelty_path.with_name('result.json'))
        if sha256(novelty_path) != analysis['novelty_masks_sha256'] or novelty['masks_sha256'] != sha256(novelty_path) or novelty['dataset_id'] != dataset:
            raise ValueError('Novelty selection differs')
        with np.load(novelty_path, allow_pickle=False) as data: indices = data['indices']; novel = data['current']
        if len(indices) != analysis['validation_positions'] or int(novel.sum()) != analysis['current_source_novel_positions']:
            raise ValueError('Validation slice differs')
        graph = Path(read(root / 'runs/m4/graph.json')['path'])
        annotation = feather.read_table(graph / 'annotations.feather', columns=['bodyId', 'type', 'superclass', 'subclass'])
        selection = read(root / 'runs/retinal-allocation-signal-v1/selection.json')
        records = []; matrices = {}; confidence = {}; initializations = {}
        families = motors = counted_indices = shared = None
        for job in jobs:
            run = job['run_id']; seed = job['seed']; arm = job['arm']
            varied, clip_mode, scale = optimizer_contract(analysis, arm)
            receipt = next(r for r in collection['records'] if r['run_id'] == run)
            endpoint_path = root / receipt['path']; endpoint = complete(endpoint_path)
            if sha256(endpoint_path) != receipt['sha256'] or endpoint['source_sha256'] != collection['helper_sha256']:
                raise ValueError('Owner endpoint audit changed')
            mask_path = root / job['head_mask']
            if sha256(mask_path) != job['head_mask_sha256']: raise ValueError('Head mask changed')
            mask = load_head_mask(mask_path, graph_id=endpoint['graph_id'], attachment_sha256=plan['input_map_sha256'])
            if (endpoint['run_id'] != run or endpoint['seed'] != seed or endpoint['mode'] != 'current'
                    or endpoint['dataset_id'] != dataset or endpoint['head_mask'] != mask.contract
                    or endpoint['plan_sha256'] != sha256(plan_path) or endpoint['native_sha256'] != analysis['native_sha256']
                    or endpoint['numerical_runtime'] != RustFly.numerical_runtime
                    or endpoint['training_contract'].get('clip_mode', 'global') != clip_mode
                    or endpoint['training_contract'].get('value_core_scale', 1.) != scale
                    or endpoint['labeled_training_exposures'] != analysis['labeled_exposures_per_endpoint']
                    or [r['step'] for r in endpoint['checkpoint_receipts']] != [0, analysis['endpoint_update']]
                    or any(r['replica_status'] != 'verified' for r in endpoint['checkpoint_receipts'])):
                raise ValueError('Endpoint contract differs')
            initialization = endpoint['initialization']
            if seed in initializations and initialization != initializations[seed]:
                raise ValueError('Paired initial parameters, moments, ports or sampler differ')
            initializations[seed] = initialization
            contract = {k: endpoint[k] for k in ('dataset_id', 'graph_id', 'input_contract', 'head_mask')}
            contract['model'] = {k: v for k, v in endpoint['model_config'].items() if k != 'seed'}
            contract['training'] = {k: v for k, v in endpoint['training_contract'].items() if k != varied}
            if shared is not None and contract != shared: raise ValueError('Another scientific factor changed')
            shared = contract
            base = root / analysis['followup'] / run; status = read(base / 'status.json')
            if status['state'] != 'complete' or status['checkpoint_sha256'] != endpoint['checkpoint_sha256']:
                raise ValueError('Followup is incomplete')
            validation = complete(base / 'validation/result.json')
            if len(validation['records']) != 1: raise ValueError('Expected one final validation')
            row = validation['records'][0]; digest = endpoint['checkpoint_sha256']
            if (row['sha256'] != digest or row['dataset_id'] != dataset or row['optimizer_step'] != analysis['endpoint_update']
                    or row['model'] != endpoint['model_config'] or row['optimizer_clip_mode'] != clip_mode):
                raise ValueError('Validation endpoint differs')
            path = base / 'validation' / f'step-{analysis["endpoint_update"]:08d}-{digest[:12]}-metrics.npz'
            values, families = aligned_metrics(path, indices, family_names, families)
            with np.load(path, allow_pickle=False) as data: conf = data['confidence']
            if conf.shape != (len(indices), 2) or not np.isfinite(conf).all(): raise ValueError('Invalid confidence data')
            for measured, names, field in ((values, METRICS, 'slices'), (conf, CONFIDENCE, 'confidence_slices')):
                np.testing.assert_allclose(measured.mean(axis=0), [row[field]['natural'][k]['mean'] for k in names], rtol=0, atol=1e-12)
            matrices[seed, arm] = values; confidence[seed, arm] = conf
            diagnostic, motors, counted_indices = diagnostics(root, base, endpoint, dict(job, release=plan['release']),
                wave, analysis, annotation, selection, motors, counted_indices)
            log_path = root / 'runs' / run / 'metrics.jsonl'
            if sha256(log_path) != endpoint['metrics_log_sha256']: raise ValueError('Training log changed')
            history = clipping_history([json.loads(line) for line in log_path.read_text().splitlines()], clip_mode, plan['updates'], plan['clip'])
            if history['final_pre_update'] != endpoint['final_activity']['clipping']:
                raise ValueError('Pre-update diagnostic and training factors differ')
            records.append(dict(endpoint, arm=arm, **diagnostic, clipping_history=history,
                endpoint_record_sha256=sha256(endpoint_path), metrics_sha256=sha256(path),
                validation_record_sha256=sha256(base / 'validation/result.json')))
        comparisons = contrasts(analysis)
        statistics = paired_results(matrices, families, novel, seeds, arms, comparisons)
        costs = {(r['seed'], r['arm']): r['counted_inference']['records'] for r in records}
        differences = []
        for comparison in comparisons:
            for index, batch in enumerate((1, 32)):
                values = [costs[s, comparison['candidate']][index]['cached_zero_skipped_flops_per_position']['mean']
                        - costs[s, comparison['reference']][index]['cached_zero_skipped_flops_per_position']['mean'] for s in seeds]
                label = comparison if len(comparisons) > 1 else {}
                differences.append(dict(**label, batch_size=batch, per_seed=values, mean=float(np.mean(values)), sample_sd=float(np.std(values, ddof=1))))
        atomic_json(args.output, dict(status='complete', created=time.time(), plan=analysis, dataset_id=dataset,
            records=records, statistics=statistics, confidence=paired_confidence(confidence, families, novel, seeds, arms, comparisons),
            decision=study_decision(statistics), cost_contrasts=differences, initialization_pairing='exact array fingerprints and sampler state',
            labeled_training_exposures=len(records) * analysis['labeled_exposures_per_endpoint'],
            novelty=novelty, probe_selection=selection, source_sha256=sha256(Path(__file__)), plan_sha256=sha256(args.plan),
            endpoint_collection_sha256=sha256(root / analysis['endpoints'] / 'result.json'),
            annotation_sha256=sha256(graph / 'annotations.feather'), uncertainty=analysis['uncertainty'],
            limitations=[analysis['scope'], analysis['arithmetic'],
                'Family intervals condition on fitted weights; seed differences and SD are separate. No multiplicity correction.',
                'Core initialization is shared; head and training sampler seeds vary. Previous development and qualification exposures are separate.',
                'Motor variance does not establish biological function. Confidence alone does not measure policy quality.',
                'No optimizer-only overhead is inferred from training or validation wall times.']))
        print(json.dumps(dict(status='complete', output=str(args.output), decision=study_decision(statistics))), flush=True)


if __name__ == '__main__': main()
