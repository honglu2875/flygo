#!/usr/bin/env python3
"""Audit retinal sampling and model-dependent motor/descending correlations.

This is a diagnostic, not a grouping fit or a learnability comparison. It uses
one position per training opening family, fixes input/head seeds, and varies
synaptic magnitudes explicitly. No production model or topology is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from flygo import _native
from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release
from flygo.fly import FlyConfig, RustFly, firing_rate, initialize, load_graph
from flygo.play import load_player
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB, StorageBudget


STD_FLOOR = 1e-6  # Model rate units; retain raw amplitudes as well as correlations.
CANDIDATES = ('descending_neuron', 'cb_motor', 'vnc_motor')


def residualize(response, nuisance):
    """Remove a fitted intercept and declared common input covariates in FP64."""
    y = np.asarray(response, np.float64)
    design = np.column_stack([np.ones(len(y)), np.asarray(nuisance, np.float64)])
    coefficient, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coefficient, int(rank)


def correlations(response, floor=STD_FLOOR):
    """Signed Pearson matrix; silent/negligible columns explicitly remain NaN."""
    y = np.asarray(response, np.float64)
    if y.ndim != 2 or len(y) < 3 or not np.isfinite(y).all() or floor <= 0:
        raise ValueError('Expected finite observations by cells and a positive variance floor')
    centered = y - y.mean(axis=0)
    norm = np.linalg.norm(centered, axis=0)
    std = norm / np.sqrt(len(y) - 1)
    valid = std > floor
    unit = centered[:, valid] / norm[valid]
    matrix = np.full((y.shape[1], y.shape[1]), np.nan, np.float32)
    matrix[np.ix_(valid, valid)] = np.clip(unit.T @ unit, -1, 1).astype(np.float32)
    gram = centered @ centered.T
    trace = float(np.trace(gram))
    square = float(np.square(gram).sum())
    eigen = np.linalg.eigvalsh(gram)
    summary = dict(cells=y.shape[1], observations=len(y), std_floor=floor,
                   varying_cells=int(valid.sum()), sample_rank_ceiling=len(y)-1,
                   std_quantiles=np.quantile(std, [0, .1, .5, .9, 1]).tolist(),
                   rms_input_variation=float(np.sqrt(np.square(std).mean())),
                   covariance_participation_rank=trace*trace/square if square > 0 else 0.,
                   first_pc_variance_fraction=float(max(eigen[-1], 0)/trace) if trace > 0 else None)
    return matrix, valid, summary


def pair_summary(matrix, labels):
    row, col = np.triu_indices(len(matrix), 1)
    value = matrix[row, col]
    finite = np.isfinite(value)
    same = np.asarray(labels)[row] == np.asarray(labels)[col]
    def stats(mask):
        x = value[finite & mask]
        return dict(pairs=len(x), mean=float(x.mean()) if len(x) else None,
                    quantiles=np.quantile(x, [0, .1, .5, .9, 1]).tolist() if len(x) else [],
                    positive_over_point5_fraction=float(np.mean(x > .5)) if len(x) else None)
    return dict(within_annotation=stats(same), between_annotations=stats(~same))


def matrix_agreement(first, second):
    """Compare off-diagonal coefficients on the intersection of varying cells."""
    if first.shape != second.shape:
        raise ValueError('Candidate identities must agree before comparing correlations')
    row, col = np.triu_indices(len(first), 1)
    a, b = first[row, col].astype(np.float64), second[row, col].astype(np.float64)
    use = np.isfinite(a) & np.isfinite(b)
    a, b = a[use], b[use]
    a -= a.mean() if len(a) else 0
    b -= b.mean() if len(b) else 0
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dict(pairs=len(a), coefficient_correlation=float(a @ b/norm) if norm > 0 else None)


def jitter_edges(graph, edge, seed, sigma):
    """Preserve signs, edges and each destination's initial absolute weight sum."""
    if seed == 0:
        return edge.copy()
    magnitude = np.logaddexp(0., edge.astype(np.float64))
    target = np.bincount(graph['dst'], weights=magnitude, minlength=len(graph['type_id']))
    rng = np.random.default_rng(seed)
    perturbed = magnitude * np.exp(sigma * rng.normal(size=len(edge)))
    total = np.bincount(graph['dst'], weights=perturbed, minlength=len(target))
    scale = np.divide(target, total, out=np.ones_like(target), where=total > 0)
    perturbed *= scale[graph['dst']]
    return np.log(np.expm1(perturbed)).astype(np.float32)


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.parameters().items()):
        digest.update(name.encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def policy_summary(logits, legal):
    masked = np.where(legal, logits.astype(np.float64), -np.inf)
    masked -= masked.max(axis=1, keepdims=True)
    probability = np.exp(masked)
    probability /= probability.sum(axis=1, keepdims=True)
    logp = np.log(np.maximum(probability, np.finfo(np.float64).tiny))
    entropy = -(probability * logp).sum(axis=1)
    legal_count = legal.sum(axis=1)
    relative = np.divide(entropy, np.log(legal_count), out=np.zeros_like(entropy), where=legal_count > 1)
    return dict(mean_entropy=float(entropy.mean()), mean_effective_actions=float(np.exp(entropy).mean()),
                mean_entropy_over_legal_uniform=float(relative.mean()),
                mean_max_probability=float(probability.max(axis=1).mean()),
                mean_legal_actions=float(legal_count.mean()))


def anatomy(graph_path, graph, columns_path, output):
    import pyarrow.feather as feather
    annotation_path = graph_path/'annotations.feather'
    table = feather.read_table(annotation_path)
    body = np.load(graph_path/'body_id.npy', mmap_mode='r')
    np.testing.assert_array_equal(table['bodyId'].to_numpy(), body)
    labels = {key: np.asarray([str(v) if v is not None else '<unknown>' for v in table[key].to_pylist()])
              for key in ('superclass', 'class', 'subclass', 'type', 'rootSide', 'somaSide', 'exitNerve', 'somaNeuromere')}
    candidate = np.flatnonzero(np.isin(labels['superclass'], CANDIDATES))
    if not 1 <= len(candidate) <= 3000:
        raise ValueError('Motor/descending population exceeds the bounded correlation profile')
    with np.load(columns_path, allow_pickle=False) as data:
        receipt = json.loads((columns_path.parent/'result.json').read_text())
        if (receipt['graph_id'] != graph['manifest']['graph_id'] or
                receipt['columns_sha256'] != sha256(columns_path) or
                receipt['annotation_sha256'] != sha256(annotation_path)):
            raise ValueError('Retinal atlas provenance mismatch')
        np.testing.assert_array_equal(data['body_id'], body)
        photo = data['photoreceptors']
        expected = graph['sensory'][labels['class'][graph['sensory']] == 'visual']
        np.testing.assert_array_equal(photo, expected)
        coord = data['coordinates'][photo]
        valid = np.isfinite(coord).all(axis=1) & (data['confidence'][photo] >= .5) & (data['synapses'][photo] >= 4)
        eyes = {}
        for side in ('L', 'R'):
            choose = valid & (labels['rootSide'][photo] == side)
            unique, counts = np.unique(coord[choose], axis=0, return_counts=True)
            eyes[side] = dict(visual_cells=int(np.sum(labels['rootSide'][photo] == side)),
                             selected_cells=int(choose.sum()), distinct_columns=len(unique),
                             mean_columns_per_12_patches=len(unique)/12,
                             columns_required_for_12_separate_9x9_patches=972,
                             cells_per_column_quantiles=np.quantile(counts, [0, .5, 1]).tolist(),
                             selected_types=dict(Counter(labels['type'][photo[choose]])))
        np.savez(output/'atlas.npz', body_id=body, candidates=candidate, photoreceptors=photo,
                 photo_coordinates=coord, photo_qualified=valid,
                 **{key: value[candidate] for key, value in labels.items()})
    side = np.where(np.isin(labels['somaSide'], ['L', 'R']), labels['somaSide'], labels['rootSide'])
    group = np.asarray(['/'.join(row) for row in zip(labels['superclass'][candidate], labels['subclass'][candidate], side[candidate])])
    report = dict(annotation_sha256=sha256(annotation_path), columns_sha256=sha256(columns_path),
                  atlas_sha256=sha256(output/'atlas.npz'), annotation_columns=table.column_names,
                  sensory_cells=len(graph['sensory']), visual_cells=len(photo), qualified_visual_cells=int(valid.sum()),
                  eyes=eyes, candidate_cells=len(candidate),
                  candidate_superclasses=dict(Counter(labels['superclass'][candidate])),
                  candidate_subclasses={kind: dict(Counter(labels['subclass'][labels['superclass'] == kind])) for kind in CANDIDATES},
                  side_rule='somaSide when L/R, otherwise rootSide; photoreceptors use rootSide',
                  geometry_scope='Inferred optic-column coordinates, not measured optical axes or 3D eye coordinates')
    return candidate, photo, labels['superclass'][candidate], group, report


def probes(root, count, seed):
    manifest, arrays, indexes = load_release(root, root/'releases/v0-1m/manifest.json', cache=True)
    families = defaultdict(list)
    offset = 0
    for record in manifest['records']:
        if record['split'] == 'train':
            families[record['opening_family']].append((offset, record['rows']))
        offset += record['rows']
    if count > len(families):
        raise ValueError('Insufficient distinct training opening families')
    rng = np.random.default_rng(seed)
    names = sorted(families)
    chosen = rng.choice(len(names), size=count, replace=False)
    indices = []
    for family in chosen:
        games = families[names[family]]
        start, length = games[int(rng.integers(len(games)))]
        indices.append(start + int(rng.integers(length)))
    indices = np.asarray(indices, np.int64)
    if not np.isin(indices, indexes['train']).all():
        raise AssertionError('Probe escaped the training split')
    x = arrays['features'][indices].copy()
    # No target policy or target value enters probe selection or correlations.
    nuisance = np.column_stack([x[:, :, :, 0].mean(axis=(1, 2)), x[:, :, :, 1].mean(axis=(1, 2)),
                               x[:, 0, 0, 8], x[:, 0, 0, 10], x[:, :, :, 11].mean(axis=(1, 2)),
                               arrays['ply'][indices]/324])
    return x, arrays['legal'][indices].copy(), nuisance, dict(dataset_id=manifest['dataset_id'], seed=seed,
        position_indices=indices.tolist(), opening_families=[names[i] for i in chosen],
        rule='One unaugmented position per distinct training opening family; no target-based selection',
        nuisance_columns=['own_stone_fraction', 'opponent_stone_fraction', 'black_to_play',
                          'consecutive_passes_over_2', 'legal_point_fraction', 'ply_over_324'])


def run_case(model, x, legal, nuisance, candidate, classes, groups, passes, batch, output, name):
    before = parameter_hash(model)
    model.config = replace(model.config, steps=max(passes))
    if model.config.readout_mean_scale != 1:
        raise ValueError('Qualify the probe separately for nondefault mean conditioning')
    params = model.parameters(); ports = model.ports
    selected = np.flatnonzero(ports['output_group'] >= 0)
    scale = ports['output_scale'][selected] * params['readout_gain'][selected]
    responses = np.empty((len(passes), len(x), len(candidate)), np.float32)
    pools = np.empty((len(passes), len(x), model.config.groups), np.float32)
    for start in range(0, len(x), batch):
        end = min(start+batch, len(x))
        result = model.infer(x[start:end], trace=True)
        for index, step in enumerate(passes):
            rate = firing_rate(result['states'][step], model.config.rate_softness)
            responses[index, start:end] = rate[candidate].T
            pool = np.zeros((model.config.groups, end-start), np.float32)
            np.add.at(pool, ports['output_group'][selected], scale[:, None]*rate[selected])
            pools[index, start:end] = pool.T
        reconstructed = pools[-1, start:end] @ params['policy_weight'].reshape(model.config.actions, model.config.groups).T + params['policy_bias']
        np.testing.assert_allclose(reconstructed, result['logits'], rtol=3e-4, atol=3e-6)
    if parameter_hash(model) != before:
        raise AssertionError('Inference mutated model parameters')
    response_path = output/(name+'-responses.npz')
    np.savez(response_path, passes=np.asarray(passes), rates=responses, pools=pools)
    records = []
    for index, step in enumerate(passes):
        raw, valid, summary = correlations(responses[index])
        residual, rank = residualize(responses[index], nuisance)
        conditioned, residual_valid, residual_summary = correlations(residual)
        _, _, pooled_summary = correlations(pools[index])
        half = len(x)//2
        first, _, _ = correlations(responses[index, :half])
        second, _, _ = correlations(responses[index, half:])
        logits = pools[index] @ params['policy_weight'].reshape(model.config.actions, model.config.groups).T + params['policy_bias']
        records.append(dict(passes=step, raw=summary, conditioned=residual_summary, nuisance_rank=rank,
            by_superclass={kind: dict(cells=int(np.sum(classes == kind)), varying_cells=int(valid[classes == kind].sum()),
                                      conditioned_varying_cells=int(residual_valid[classes == kind].sum())) for kind in CANDIDATES},
            annotation_pairs=pair_summary(raw, groups), conditioned_annotation_pairs=pair_summary(conditioned, groups),
            split_half_raw_agreement=matrix_agreement(first, second), pooled=pooled_summary,
            policy=policy_summary(logits, legal)))
        # Bound storage: retain full correlation matrices at the largest pass only.
        if step == max(passes):
            np.savez(output/(name+'-correlations.npz'), raw=raw, conditioned=conditioned,
                     candidate=candidate, raw_valid=valid, conditioned_valid=residual_valid)
    return dict(name=name, model_config=asdict(model.config), parameter_sha256=before,
                response_sha256=sha256(response_path), correlation_sha256=sha256(output/(name+'-correlations.npz')),
                records=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--positions', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--probe-seed', type=int, default=918417)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--cpus', default=','.join(map(str, range(32, 56))))
    args = p.parse_args(); cpus = list(map(int, args.cpus.split(','))); pin(cpus)
    if not 64 <= args.positions <= 512 or not 1 <= args.batch_size <= 32:
        p.error('Use 64..512 probes and batch 1..32 in this bounded pilot')
    args.output.resolve().relative_to(args.root.resolve())
    started = time.time()
    with StorageBudget(args.root).reserve(files=768<<20, heap=12*GIB, purpose='biological ports correlation pilot'):
        args.output.mkdir(parents=True, exist_ok=False)
        graph_path = Path(json.loads((args.root/'runs/m4/graph.json').read_text())['path'])
        graph = load_graph(graph_path)
        port_dir = args.root/'ports/retinal-overlay-v1'
        candidate, photo, classes, groups, atlas = anatomy(graph_path, graph, port_dir/'columns.npz', args.output)
        x, legal, nuisance, probe_record = probes(args.root, args.positions, args.probe_seed)
        np.savez(args.output/'probes.npz', features=x, legal=legal, nuisance=nuisance)
        config = FlyConfig(steps=8, threads=len(cpus), seed=1)
        base_ports, base_params = initialize(graph, config)
        # The original spatial overlay still has random feature channels: label it accurately.
        overlay_path = port_dir/'spatial-g656-s1.npz'
        if sha256(overlay_path) != json.loads(overlay_path.with_suffix('.json').read_text())['sha256']:
            raise ValueError('Spatial overlay hash mismatch')
        with np.load(overlay_path) as data:
            overlay = {key: data[key].copy() for key in base_ports}
        visual = np.zeros(len(graph['type_id']), bool); visual[photo] = True
        report = dict(schema_version=1, status='running', graph_id=graph['manifest']['graph_id'], anatomy=atlas,
                      probes=probe_record, probes_sha256=sha256(args.output/'probes.npz'), cpus=cpus,
                      script_sha256=sha256(Path(__file__)), native_sha256=sha256(Path(_native.__file__)),
                      fly_interface_sha256=sha256(Path(__file__).resolve().parents[1]/'python/flygo/fly.py'),
                      overlay_sha256=sha256(overlay_path), cases=[],
                      interpretation='Descriptive model responses, not physiological recordings or a learning comparison. '
                      'No topology changes, optimizer updates, or final-test selection. Absolute response amplitudes accompany '
                      'correlations. Jitter preserves signs and destination incoming mass; input and head seed stay fixed. '
                      'Existing overlays retain random feature channels and unqualified photoreceptor assignments. '
                      'Do not freeze action groups from these pilots before biological input alignment.')
        atomic_json(args.output/'result.json', report)
        try:
            for mode, seed in [('all-sensory', 0), ('visual-only', 0), ('overlay-visual-only', 0),
                               ('overlay-visual-only', 1), ('overlay-visual-only', 2)]:
                ports = {key: value.copy() for key, value in (overlay if mode.startswith('overlay') else base_ports).items()}
                if mode != 'all-sensory':
                    ports['input_index'][~visual] = -1
                params = dict(base_params, edge=jitter_edges(graph, base_params['edge'], seed, .1))
                model = RustFly(graph, config, ports=ports, params=params)
                name = mode+'-j'+str(seed)
                result = run_case(model, x, legal, nuisance, candidate, classes, groups, [4, 8], args.batch_size, args.output, name)
                result.update(weight_jitter_seed=seed, log_weight_jitter_std=.1 if seed else 0,
                              direct_input_cells=int(np.sum(ports['input_index'] >= 0)))
                report['cases'].append(result)
                atomic_json(args.output/'result.json', report)
                print(json.dumps(dict(case=name, seconds=time.time()-started, records=result['records'])), flush=True)
                del model, params
            if args.checkpoint:
                model, metadata = load_player(args.checkpoint, graph_path, threads=len(cpus))
                if metadata['dataset_id'] != probe_record['dataset_id']:
                    raise ValueError('Checkpoint and probe release differ')
                trained_passes = model.config.steps
                result = run_case(model, x, legal, nuisance, candidate, classes, groups, [4, 8], args.batch_size, args.output, 'trained')
                result.update(checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
                              trained_passes=trained_passes, training_contract=metadata.get('training_contract'))
                report['cases'].append(result)
                del model
            reference = np.load(args.output/'overlay-visual-only-j0-correlations.npz')
            agreements = []
            for seed in (1, 2):
                with np.load(args.output/f'overlay-visual-only-j{seed}-correlations.npz') as other:
                    agreements.append(dict(weight_jitter_seed=seed, **{key: matrix_agreement(reference[key], other[key])
                                                                      for key in ('raw', 'conditioned')}))
            reference.close()
            report.update(status='complete', weight_jitter_agreement=agreements, seconds=time.time()-started)
        except BaseException as error:
            report.update(status='failed', error=repr(error), seconds=time.time()-started)
            raise
        finally:
            atomic_json(args.output/'result.json', report)
        print(json.dumps(dict(status=report['status'], seconds=report['seconds'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
