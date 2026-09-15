#!/usr/bin/env python3
"""Read-only policy/value gradient decomposition on frozen fly checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from flygo.attachments import runtime_hashes
from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release
from flygo.fly import PARAMETERS
from flygo.play import load_player
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.signal_flow import pooled_cotangent
from flygo.storage import GIB, StorageBudget


SHARED = ('edge', 'leak', 'bias', 'input_gain', 'readout_gain')


def fingerprints(arrays):
    return {name: dict(shape=list(value.shape), dtype=value.dtype.str,
                       sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
            for name, value in arrays.items()}


def products(policy, value):
    p, v = np.asarray(policy, np.float64).ravel(), np.asarray(value, np.float64).ravel()
    if p.shape != v.shape or not np.isfinite(p).all() or not np.isfinite(v).all():
        raise ValueError('Expected aligned finite gradients')
    return dict(policy_square=float(p @ p), value_square=float(v @ v), dot=float(p @ v))


def geometry(inner):
    pp, vv, pv = (inner[k] for k in ('policy_square', 'value_square', 'dot'))
    return dict(**inner, policy_norm=float(np.sqrt(pp)), value_norm=float(np.sqrt(vv)),
                cosine=pv / np.sqrt(pp * vv) if pp > 0 and vv > 0 else None,
                value_to_policy_norm=float(np.sqrt(vv / pp)) if pp > 0 else None,
                value_squared_norm_share=vv / (pp + vv) if pp + vv > 0 else None,
                joint_sgd_policy_dot=pp + pv, joint_sgd_value_dot=vv + pv,
                scope='Raw gradient inner products. A negative joint_sgd_policy_dot means infinitesimal joint SGD increases this batch policy loss; it is not an Adam update or a validation prediction.')


def separate_gradients(core, encoded, legal, policy, target_value, output=None):
    """Keep the native policy loss; cancel value exactly, then use its score VJP."""
    output = core.infer(encoded, trace=True) if output is None else output
    joint_loss, joint = core.loss_and_grad(encoded, legal, policy, target_value)
    policy_loss, policy_grad = core.loss_and_grad(encoded, legal, policy, output['value'].copy())
    if policy_loss['value_loss'] != 0 or policy_loss['policy_loss'] != joint_loss['policy_loss']:
        raise ValueError('Policy-only call did not cancel the value residual exactly')

    def objective(embedding, linear_score):
        # This API supplies the pre-tanh score and accepts d(loss)/d(score).
        # Use the already traced native tanh value, avoiding a second tanh
        # implementation. The all-group decomposition check audits this VJP.
        prediction = output['value']
        if linear_score.shape != prediction.shape or not np.isfinite(linear_score).all():
            raise ValueError('Expected one finite linear value score per position')
        delta = prediction - target_value
        score = (np.float32(2) * delta / np.float32(len(prediction))) * (np.float32(1) - prediction * prediction)
        return dict(value_loss=float(np.mean(delta * delta, dtype=np.float64))), np.zeros_like(embedding), score

    value_loss, value_grad = core.embedding_loss_and_grad(encoded, objective)
    np.testing.assert_allclose(value_loss['value_loss'], joint_loss['value_loss'], rtol=0, atol=1e-12)
    for name in ('value_weight', 'value_bias'):
        if np.count_nonzero(policy_grad[name]):
            raise ValueError('Policy-only objective changed value-head gradients')
    for name in ('policy_weight', 'policy_bias'):
        if np.count_nonzero(value_grad[name]):
            raise ValueError('Value-only objective changed policy-head gradients')
    return joint_loss, dict(joint=joint, policy=policy_grad, value=value_grad)


def decomposition_check(gradients, rtol, atol):
    checks = {}
    for name in PARAMETERS:
        actual = gradients['joint'][name].astype(np.float64)
        separate = gradients['policy'][name].astype(np.float64) + gradients['value'][name].astype(np.float64)
        error = separate - actual
        checks[name] = dict(passed=bool(np.all(np.abs(error) <= atol + rtol * np.abs(actual))),
            max_abs=float(np.max(np.abs(error), initial=0)),
            relative_l2=float(np.linalg.norm(error) / max(np.linalg.norm(actual), 1e-30)))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--host', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root = args.root; out = args.output
    plan = json.loads(args.plan.read_text()); pin(plan['cpus'])
    out.resolve().relative_to(root.resolve())
    if runtime_hashes() != plan['runtime_sha256'] or sha256(Path(__file__)) != plan['worker_sha256']:
        raise ValueError('Frozen worker or learner source changed')
    cases = [case for case in plan['cases'] if case['host'] == args.host]
    if len(cases) != 3:
        raise ValueError('Each owner has one shared initial core and two trained endpoints')
    selection_path = root / plan['selection']
    if sha256(selection_path) != plan['selection_sha256']:
        raise ValueError('Training probe selection changed')
    selection = json.loads(selection_path.read_text()); count = plan['positions']
    with StorageBudget(root).reserve(files=768 << 20, heap=12 * GIB, purpose='read-only native policy/value gradient audit'):
        out.mkdir(parents=True, exist_ok=False); started = time.time()
        manifest, arrays, indexes = load_release(root, root / 'releases' / plan['release'] / 'manifest.json', cache=True)
        indices = np.asarray(selection['position_indices'][:count], np.int64)
        if (manifest['dataset_id'] != plan['dataset_id'] or selection['dataset_id'] != plan['dataset_id']
                or len(indices) != count or not np.isin(indices, indexes['train']).all()):
            raise ValueError('Expected only the registered training positions')
        x = arrays['features'][indices].copy(); legal = arrays['legal'][indices].copy()
        policy = arrays['raw_policy'][indices].copy(); target_value = arrays['raw_value'][indices].copy()
        np.savez(out / 'selection.npz', indices=indices, features=x, legal=legal, policy=policy, value=target_value)
        del arrays, indexes
        graph_path = Path(json.loads((root / 'runs/m4/graph.json').read_text())['path'])
        records = []
        atomic_json(out / 'source.json', dict(plan_sha256=sha256(args.plan), worker_sha256=sha256(Path(__file__)),
            runtime_sha256=runtime_hashes(), selection_sha256=sha256(out / 'selection.npz'),
            graph_id=plan['graph_id'], numpy=np.__version__))
        for case in cases:
            name = case['run_id'] + f'-step-{case["step"]:08d}'
            directory = out / name; directory.mkdir()
            atomic_json(out / 'status.json', dict(state='gradients', case=name, updated=time.time()))
            checkpoint = root / 'runs' / case['run_id'] / 'checkpoints' / f'step-{case["step"]:08d}.npz'
            receipt = json.loads(checkpoint.with_suffix('.json').read_text())
            if (receipt['replica_status'] != 'verified' or receipt['sha256'] != case['checkpoint_sha256']
                    or sha256(checkpoint) != case['checkpoint_sha256']):
                raise ValueError('Expected the registered checkpoint and its verified replica')
            player, metadata = load_player(checkpoint, graph_path, threads=len(plan['cpus']))
            if (metadata['dataset_id'] != plan['dataset_id'] or metadata['graph_id'] != plan['graph_id']
                    or player.config.seed != case['seed'] or player.config.steps != 8 or player.config.rate_softness
                    or player.config.groups != 2129 or player.config.readout_mean_scale != 1
                    or player.mode != 'current' or player.core.clip_mode != case['arm']
                    or metadata['input_contract']['attachment_sha256'] != plan['input_map_sha256']
                    or player.head_mask is None or player.head_mask.contract != plan['head_mask']):
                raise ValueError('Unexpected model, head, input or source contract')
            saved_state = player.checkpoint_arrays()
            if int(saved_state['optimizer_step']) != case['step']:
                raise ValueError('Checkpoint optimizer step differs')
            before = fingerprints(saved_state); del saved_state
            encoded = player.adapter.encode(x, 'current'); output = player.core.infer(encoded, trace=True)
            loss, gradients = separate_gradients(player.core, encoded, legal, policy, target_value, output)
            checks = decomposition_check(gradients, **plan['native_comparison'])
            np.savez(directory / 'gradients.npz', **{mode + '/' + key: value for mode, values in gradients.items() for key, value in values.items()})
            if not all(row['passed'] for row in checks.values()):
                atomic_json(directory / 'failure.json', dict(status='failed', checks=checks))
                raise ValueError('Native joint gradient does not reproduce its decomposition; preserve the failed audit')
            groups = {key: geometry(products(gradients['policy'][key], gradients['value'][key])) for key in PARAMETERS}
            shared = geometry({key: sum(groups[name][key] for name in SHARED) for key in ('policy_square', 'value_square', 'dot')})
            diagonal = {}
            with np.load(checkpoint, allow_pickle=False) as saved:
                for key in SHARED:
                    second = saved['second/' + key].astype(np.float64)
                    if np.any(second < 0) or not np.isfinite(second).all():
                        raise ValueError('Invalid saved second moment')
                    correction = 1 - plan['beta2'] ** case['step'] if case['step'] else 1.
                    scale = np.sqrt(second / correction) + plan['epsilon']
                    diagonal[key] = geometry(products(gradients['policy'][key] / scale, gradients['value'][key] / scale))
                    for field in ('joint_sgd_policy_dot', 'joint_sgd_value_dot'):
                        del diagonal[key][field]
                    diagonal[key]['scope'] = 'Angles and norms after dividing by the saved bias-corrected Adam denominator. This omits first-moment history and the next second-moment update; it is not an actual Adam step.'
            params = player.parameters(); selected = np.flatnonzero(player.ports['output_group'] >= 0)
            motors = selected[np.argsort(player.ports['output_group'][selected])]
            np.testing.assert_array_equal(player.ports['output_group'][motors], np.arange(player.config.groups))
            np.testing.assert_array_equal(player.ports['output_scale'][motors], np.ones(len(motors)))
            cotangents = {}
            for label in ('policy', 'value'):
                _, _, pooled, _ = pooled_cotangent('original-' + label, output, params, motors, policy, target_value, dict(legal=legal))
                cotangents[label] = pooled * params['readout_gain'][motors] * (output['states'][-1][motors].T > 0)
            current = np.maximum(output['states'][-1][motors].T, 0)
            neutral_output = player.core.infer(player.adapter.encode(x, 'neutral'), trace=True)
            neutral = np.maximum(neutral_output['states'][-1][motors].T, 0)
            variance = np.var(current.astype(np.float64) - neutral, axis=0)
            top = np.argsort(-variance, kind='stable')[:5]
            motor_square = {label: np.sum(values.astype(np.float64) ** 2, axis=0) for label, values in cotangents.items()}
            motor = dict(geometry=geometry(products(cotangents['policy'], cotangents['value'])),
                leading_visual=[dict(motor_index=int(i), graph_index=int(motors[i]),
                    variance_fraction=float(variance[i] / variance.sum()) if variance.sum() else None,
                    **{label + '_cotangent_square_fraction': float(values[i] / values.sum()) if values.sum() else None
                       for label, values in motor_square.items()}) for i in top])
            for field in ('joint_sgd_policy_dot', 'joint_sgd_value_dot'):
                del motor['geometry'][field]
            motor['geometry']['scope'] = 'Angles and norms of final motor-state cotangents. Motor states are intermediate features, not independently trainable parameters.'
            np.savez(directory / 'motors.npz', motors=motors, current=current, neutral=neutral,
                     policy_cotangent=cotangents['policy'], value_cotangent=cotangents['value'],
                     logits=output['logits'], value=output['value'])
            after = fingerprints(player.checkpoint_arrays())
            if after != before or sha256(checkpoint) != case['checkpoint_sha256']:
                raise ValueError('Read-only audit changed parameters, moments or checkpoint bytes')
            record = dict(case=case, status='complete', loss=loss, checks=checks, groups=groups, shared=shared,
                saved_adam_diagonal=diagonal, motor=motor, unchanged_state=before,
                gradients_sha256=sha256(directory / 'gradients.npz'), motors_sha256=sha256(directory / 'motors.npz'))
            atomic_json(directory / 'result.json', record); records.append(record)
            del player, gradients, output, neutral_output, params
        atomic_json(out / 'result.json', dict(status='complete', created=time.time(), seconds=time.time() - started,
            records=records, plan_sha256=sha256(args.plan), source_sha256=sha256(out / 'source.json'),
            update_exposures=0, diagnostic_backward_positions=len(records) * 3 * count,
            diagnostic_forward_positions=len(records) * 6 * count, scope=plan['scope']))
        atomic_json(out / 'status.json', dict(state='complete', updated=time.time()))
        print(json.dumps(dict(status='complete', cases=len(records), seconds=time.time() - started)), flush=True)


if __name__ == '__main__': main()
