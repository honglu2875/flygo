#!/usr/bin/env python3
"""Close paired attachment confirmations using owner-verified endpoint records."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.feather as feather

from summarize_attachments import family_summary
from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB, StorageBudget


MODES = ('history', 'current', 'neutral')
METRICS = ('policy_kl', 'value_mse', 'teacher_top1_agreement')
CONTRASTS = (('neutral', 'history'), ('neutral', 'current'), ('history', 'current'))


def read(path):
    return json.loads(path.read_text())


def complete(report):
    if report.get('status') != 'complete':
        raise ValueError('Evidence is incomplete')
    return report


def slices(metrics, families, common):
    return dict(natural=family_summary(metrics, families),
                common_reduced_source_novel=family_summary(metrics[common], families[common]))


def seed_variation(matrices, common):
    """Describe fitted-seed variation separately from conditional family intervals."""
    result = {}
    for name, mask in [('natural', slice(None)), ('common_reduced_source_novel', common)]:
        values = np.stack([matrix[mask].mean(axis=0) for matrix in matrices])
        result[name] = {key: dict(per_seed=values[:, i].tolist(),
                                  sample_sd=float(values[:, i].std(ddof=1)))
                        for i, key in enumerate(METRICS)}
    return result


def concentration(path, mode, annotation, expected_motors):
    with np.load(path, allow_pickle=False) as data:
        motors = data['motors']
        if expected_motors is not None:
            np.testing.assert_array_equal(motors, expected_motors)
        # Match the seed-1 diagnostic exactly, including FP32 subtraction.
        y = (data[mode] - data['neutral']).astype(np.float64)
    if y.shape != (256, 2129) or not np.isfinite(y).all() or len(np.unique(motors)) != 2129:
        raise ValueError('Invalid or unaligned motor responses')
    y -= y.mean(axis=0)
    variance = np.square(y).sum(axis=0)
    use = np.sqrt(variance / (len(y) - 1)) > 1e-6
    unit = y[:, use] / np.sqrt(variance[use])
    gram = unit @ unit.T
    total = float(variance.sum())
    rank_denominator = float(np.square(gram).sum())
    result = dict(varying_cells=int(use.sum()), std_floor=1e-6,
                  standardized_participation_rank=(float(np.trace(gram)**2 / rank_denominator)
                                                   if rank_denominator else None),
                  top_cells=[dict(node=int(motors[i]),
                                  visual_variance_fraction=float(variance[i] / total) if total else None,
                                  **{key: annotation[key][int(motors[i])].as_py()
                                     for key in annotation.column_names})
                             for i in np.argsort(variance)[::-1][:5]])
    return result, motors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pin([56, 57, 58, 59])
    root = args.root
    args.output.resolve().relative_to(root.resolve())
    if args.output.exists():
        raise ValueError('Analysis reports are immutable')
    study = root / 'runs/attachment-confirmation-v1'
    plan = read(study / 'plan.json')
    plan_sha = sha256(study / 'plan.json')
    adopted_path = root / plan['seed1_reference']
    if sha256(adopted_path) != plan['seed1_analysis_sha256']:
        raise ValueError('The adopted exploratory analysis changed')
    adopted = complete(read(adopted_path))
    for key in ('release', 'updates', 'batch_size', 'passes', 'groups', 'rate',
                'warmup_steps', 'epsilon', 'clip', 'rate_scales', 'eval_every', 'eval_positions'):
        # The original registered plan names its release through dataset_id.
        if key != 'release' and plan[key] != adopted['plan'][key]:
            raise ValueError('Confirmation differs from its adopted contract: ' + key)
    jobs = [dict(job, seed=1) for job in adopted['plan']['jobs']] + plan['jobs']
    if sorted((job['seed'], job['mode']) for job in jobs) != sorted(
            (seed, mode) for seed in (1, 2, 3) for mode in MODES):
        raise ValueError('Each of three modes must occur once per declared seed')

    with StorageBudget(root).reserve(files=8 << 20, heap=3 * GIB,
                                     purpose='paired attachment confirmation analysis'):
        manifest = read(root / 'releases' / plan['release'] / 'manifest.json')
        dataset_id = manifest['dataset_id']
        if dataset_id != adopted['plan']['dataset_id']:
            raise ValueError('Release identity differs')
        family_names = np.asarray([row['opening_family'] for row in manifest['records']])
        novelty_dir = root / 'runs/attachment-input-novelty-v1'
        novelty = complete(read(novelty_dir / 'result.json'))
        if novelty['dataset_id'] != dataset_id or sha256(novelty_dir / 'masks.npz') != novelty['masks_sha256']:
            raise ValueError('Novelty evidence differs')
        with np.load(novelty_dir / 'masks.npz', allow_pickle=False) as data:
            indices = data['indices']
            common = data['current'] & data['neutral']
        graph = Path(read(root / 'runs/m4/graph.json')['path'])
        annotation = feather.read_table(graph / 'annotations.feather',
                                       columns=['bodyId', 'type', 'superclass', 'subclass'])
        metrics_by_case, records, diagnostic_records = {}, [], []
        expected_families = expected_motors = expected_selection = None
        for job in jobs:
            seed, mode, run_id = job['seed'], job['mode'], job['run_id']
            if seed == 1:
                endpoint = next(row for row in adopted['records'] if row['mode'] == mode)
                validation = root / 'runs/attachment-validation-v1' / run_id / 'validation'
                signal_dir = root / 'runs/attachment-signal-v1'
                count_path = root / 'runs/attachment-count-v1/result.json'
                endpoint_sha = plan['seed1_analysis_sha256']
            else:
                endpoint_path = root / 'runs/attachment-confirmation-endpoints-v1' / (run_id + '.json')
                endpoint = complete(read(endpoint_path))
                endpoint_sha = sha256(endpoint_path)
                if (endpoint['plan_sha256'] != plan_sha or endpoint['seed'] != seed
                        or endpoint['mode'] != mode or endpoint['dataset_id'] != dataset_id
                        or any(row['replica_status'] != 'verified' for row in endpoint['checkpoint_receipts'])):
                    raise ValueError('Endpoint contract or recovery receipts differ')
                validation = root / 'runs/attachment-confirmation-validation-v1' / run_id / 'validation'
                diagnostic_dir = root / 'runs/attachment-confirmation-diagnostics-v1' / run_id
                signal_dir = diagnostic_dir / 'signal'
                count_path = diagnostic_dir / 'count/result.json'
            if (endpoint['labeled_training_exposures'] != plan['updates'] * plan['batch_size']
                    or endpoint['input_contract']['mode'] != mode
                    or endpoint['input_contract']['attachment_sha256'] != adopted['records'][0]['input_contract']['attachment_sha256']):
                raise ValueError('Input attachment or exposure horizon differs')
            report = complete(read(validation / 'result.json'))
            if len(report['records']) != 1:
                raise ValueError('Expected exactly one final validation checkpoint')
            validated = report['records'][0]
            checkpoint_sha = endpoint['checkpoint_sha256']
            if (validated['sha256'] != checkpoint_sha or validated['dataset_id'] != dataset_id
                    or validated['optimizer_step'] != plan['updates'] or validated['model']['seed'] != seed
                    or validated['model']['steps'] != plan['passes'] or validated['model']['groups'] != plan['groups']):
                raise ValueError('Validation checkpoint identity differs')
            metric_path = validation / (f"step-{plan['updates']:08d}-{checkpoint_sha[:12]}-metrics.npz")
            with np.load(metric_path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['indices'], indices)
                metrics = data['metrics']
                families = family_names[data['game_index']]
            if metrics.shape != (len(indices), 3) or not np.isfinite(metrics).all():
                raise ValueError('Invalid validation metrics')
            if expected_families is None:
                expected_families = families
            else:
                np.testing.assert_array_equal(families, expected_families)
            np.testing.assert_allclose(metrics.mean(axis=0),
                [validated['slices']['natural'][key]['mean'] for key in METRICS], rtol=0, atol=1e-12)
            case_slices = slices(metrics, families, common)
            if seed == 1:
                if sha256(metric_path) != endpoint['metrics_sha256'] or any(
                        case_slices[key] != endpoint[key] for key in case_slices):
                    raise ValueError('Exploratory metrics or analysis no longer reproduce')
            metrics_by_case[seed, mode] = metrics
            records.append(dict(endpoint, seed=seed, mode=mode, run_id=run_id,
                                seed_role='exploratory' if seed == 1 else 'confirmation',
                                endpoint_record_sha256=endpoint_sha, metrics_sha256=sha256(metric_path),
                                validation_record_sha256=sha256(validation / 'result.json'), **case_slices))

            signal = complete(read(signal_dir / 'result.json'))
            selection = read(signal_dir / 'selection.json')
            if expected_selection is None:
                expected_selection = selection
            elif selection != expected_selection:
                raise ValueError('Motor probes selected different training positions')
            if signal['dataset_id'] != dataset_id or len(selection['position_indices']) != 256:
                raise ValueError('Motor probe dataset or position count differs')
            response = next(row for row in signal['records'] if row['case'] == mode)
            if response['checkpoint_sha256'] != checkpoint_sha or response['optimizer_step'] != plan['updates']:
                raise ValueError('Motor probe checkpoint differs')
            for row in signal['records']:
                if row['case'] in ('initial', mode):
                    if sha256(signal_dir / (row['case'] + '-responses.npz')) != row['response_file_sha256']:
                        raise ValueError('Motor response bytes changed')
            concentrated = None
            if mode != 'neutral':
                concentrated, expected_motors = concentration(signal_dir / (mode + '-responses.npz'),
                                                              mode, annotation, expected_motors)
            count = complete(read(count_path))
            if count['dataset_id'] != dataset_id or count['indices_sha256'] != adopted['counted_inference']['indices_sha256']:
                raise ValueError('Arithmetic probe dataset or positions differ')
            counted = next(row for row in count['results'] if row['checkpoint_sha256'] == checkpoint_sha)
            diagnostic_records.append(dict(seed=seed, mode=mode, checkpoint_sha256=checkpoint_sha,
                visual_signal=response, visual_concentration=concentrated,
                signal_record_sha256=sha256(signal_dir / 'result.json'),
                selection_sha256=sha256(signal_dir / 'selection.json'),
                counted_inference=counted, count_record_sha256=sha256(count_path)))

        contrasts = [dict(seed=seed, reference=reference, candidate=candidate,
                          **slices(metrics_by_case[seed, candidate] - metrics_by_case[seed, reference],
                                   families, common))
                     for seed in (1, 2, 3) for reference, candidate in CONTRASTS]
        pooled = []
        for seeds in ((2, 3), (1, 2, 3)):
            aggregate_records, aggregate_contrasts = [], []
            for mode in MODES:
                matrices = [metrics_by_case[seed, mode] for seed in seeds]
                aggregate_records.append(dict(mode=mode, **slices(np.mean(matrices, axis=0), families, common),
                                              seed_variation=seed_variation(matrices, common)))
            for reference, candidate in CONTRASTS:
                matrices = [metrics_by_case[seed, candidate] - metrics_by_case[seed, reference] for seed in seeds]
                aggregate_contrasts.append(dict(reference=reference, candidate=candidate,
                    **slices(np.mean(matrices, axis=0), families, common),
                    seed_variation=seed_variation(matrices, common)))
            pooled.append(dict(seeds=list(seeds), includes_exploratory_seed1=1 in seeds,
                               records=aggregate_records, contrasts=aggregate_contrasts))
        atomic_json(args.output, dict(status='complete', created=time.time(), plan=plan,
            dataset_id=dataset_id, records=records, contrasts=contrasts, pooled=pooled,
            contrast_direction='candidate minus reference', diagnostics=diagnostic_records,
            uncertainty='1000 paired opening-family bootstrap resamples of per-position metrics averaged across the named fitted seeds. Intervals condition on those weights; they do not estimate training-seed uncertainty. Per-seed means and sample SD are separate.',
            novelty=novelty, probe_selection=expected_selection,
            source_sha256=sha256(Path(__file__)),
            family_helper_sha256=sha256(Path(__file__).with_name('summarize_attachments.py')),
            bootstrap_helper_sha256=sha256(Path(__file__).with_name('compare_checkpoints.py')),
            annotation_sha256=sha256(graph / 'annotations.feather'), plan_sha256=plan_sha,
            limitations=[
                'Two new head/sampler seeds plus the adopted exploratory seed; initial core strengths are not independently randomized.',
                'Fixed 32k-exposure horizon. No checkpoint selection, final-test evaluation, matches or TPU use.',
                'Reduced-source novelty precedes retinal rendering; FP32 rendering collisions are not certified absent.',
                'Motor perturbations hold context fixed and can be out of distribution. No action groups were fitted.',
                'Raw motor response concentration describes these models and probes, not physiological recordings.',
                'Counted inference uses the original unpruned runtime and excludes backward/optimizer/memory work and search. No new equi-FLOP CNN comparison is established.']))
        print(json.dumps(dict(status='complete', output=str(args.output), records=len(records))), flush=True)


if __name__ == '__main__':
    main()
