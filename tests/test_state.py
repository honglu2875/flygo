"""Explicit memory must compose through both task heads and the complete state."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from dataclasses import replace
import unittest
import numpy as np

from flygo.fly import RustFly, PARAMETERS
from flygo.readout import HeadMask
from tests.test_fly import fixture


def dense_forward(graph, ports, params, config, x, initial):
    """Independent float64, dense recurrence for directional/coordinate differences."""
    p = {k: np.asarray(v, np.float64) for k, v in params.items()}
    n = len(graph['type_id'])
    matrix = np.zeros((n, n), np.float64)
    np.add.at(matrix, (graph['dst'], graph['src']),
              graph['sign'][graph['src']] * np.logaddexp(0, p['edge']))
    alpha = (.01 + .98 / (1 + np.exp(-p['leak'][graph['type_id']])))[:, None]
    drive = np.zeros((n, len(x)), np.float64)
    for node, feature in enumerate(ports['input_index']):
        if feature >= 0:
            drive[node] = p['input_gain'][feature] * x[:, feature]
    def rate(h):
        s = config.rate_softness
        return np.maximum(h, 0) + (s * np.log1p(np.exp(-np.abs(h) / s)) if s else 0)
    h = np.asarray(initial, np.float64)
    for _ in range(config.steps):
        h = (1-alpha)*h + alpha*(matrix@rate(h) + p['bias'][graph['type_id'], None] + drive)
    pooled = np.zeros((config.groups, len(x)), np.float64)
    for node, group in enumerate(ports['output_group']):
        if group >= 0:
            pooled[group] += ports['output_scale'][node] * p['readout_gain'][node] * rate(h[node])
    pooled -= (1-config.readout_mean_scale) * pooled.mean(axis=0, keepdims=True)
    return dict(logits=pooled.T@p['policy_weight'].reshape(config.actions, config.groups).T+p['policy_bias'],
                value=np.tanh(pooled.T@p['value_weight']+p['value_bias'][0]), next_state=h)


def teacher_cotangents(output, legal, policy, value, scale=1.):
    logits = np.where(legal, output['logits'].astype(np.float64), -np.inf)
    shifted = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(shifted)
    prob /= prob.sum(axis=1, keepdims=True)
    dl = ((prob*policy.sum(axis=1, keepdims=True, dtype=np.float64)-policy)/len(value)*scale).astype(np.float32)
    dv = (2*(output['value']-value)/len(value)*scale).astype(np.float32)
    return dl, dv


class StateInterface(unittest.TestCase):
    def test_reset_chunk_composition_batch_independence_and_owned_arrays(self):
        graph, config, ports, params, x, *_ = fixture()
        for softness in (0., .05):
            cfg = replace(config, rate_softness=softness)
            model = RustFly(graph, cfg, ports=ports, params=params)
            start = model.initial_state(len(x))
            original = start.copy()
            explicit = model.infer_state(x, start, trace=True)
            legacy = model.infer(x, trace=True)
            for key in ('logits', 'value', 'states'):
                np.testing.assert_array_equal(explicit[key], legacy[key])
            # A carried state must bypass the cached reset message.
            initial = np.array([[.3, -.2], [.15, .4], [-.13, .25], [.5, -.1]], np.float32)
            first = model.infer_state(x, initial)
            next_copy = first['next_state'].copy()
            second = model.infer_state(x, first['next_state'])
            whole = RustFly(graph, replace(cfg, steps=2*cfg.steps), ports=ports, params=params)
            uninterrupted = whole.infer_state(x, initial, trace=True)
            for key in ('logits', 'value', 'next_state'):
                np.testing.assert_array_equal(second[key], uninterrupted[key])
            self.assertFalse(np.array_equal(first['next_state'], explicit['next_state']))
            np.testing.assert_array_equal(start, original)
            np.testing.assert_array_equal(first['next_state'], next_copy)
            # Interleaved games and independent batch columns cannot share hidden state.
            for b in range(len(x)):
                one = model.infer_state(x[b:b+1], initial[:, b:b+1])
                np.testing.assert_array_equal(one['next_state'], first['next_state'][:, b:b+1])
                np.testing.assert_array_equal(one['logits'], first['logits'][b:b+1])
            np.testing.assert_array_equal(model.infer(x)['logits'], legacy['logits'])

    def test_composed_vjp_all_groups_and_initial_state_against_dense_differences(self):
        graph, config, ports, params, x, *_ = fixture()
        initial = np.array([[.3, -.2], [.15, .4], [-.13, .25], [.5, -.1]], np.float32)
        xs = [x, x[::-1].copy()*.7]
        dl = [np.array([[.2, -.1, .3], [-.4, .3, .1]], np.float32),
              np.array([[-.1, .4, .2], [.1, -.2, .5]], np.float32)]
        dv = [np.array([.3, -.2], np.float32), np.array([-.15, .35], np.float32)]
        terminal = np.linspace(-.2, .3, initial.size, dtype=np.float32).reshape(initial.shape)
        for softness in (0., .05):
            cfg = replace(config, rate_softness=softness, readout_mean_scale=.4)
            # Generic VJPs must remain literal even when the supervised objective differs.
            model = RustFly(graph, cfg, ports=ports, params=params, value_core_scale=0.)
            first = model.infer_state(xs[0], initial)
            second = model.infer_state(xs[1], first['next_state'])
            g2, dh = model.state_vjp(xs[1], first['next_state'], revision=second['revision'],
                                   dlogits=dl[1], dvalue=dv[1], dstate=terminal)
            g1, dh = model.state_vjp(xs[0], initial, revision=first['revision'],
                                   dlogits=dl[0], dvalue=dv[0], dstate=dh)
            total = {k: g1[k]+g2[k] for k in PARAMETERS}
            def objective(p, h):
                loss = 0.
                for t in range(2):
                    o = dense_forward(graph, ports, p, cfg, xs[t], h)
                    h = o['next_state']
                    loss += np.sum(o['logits']*dl[t]) + np.sum(o['value']*dv[t])
                return loss + np.sum(h*terminal)
            epsilon = 1e-5
            for key in PARAMETERS:
                expected = np.zeros_like(params[key], dtype=np.float64)
                for i in range(expected.size):
                    p = {k: v.astype(np.float64) for k, v in params.items()}
                    p[key][i] += epsilon
                    plus = objective(p, initial)
                    p[key][i] -= 2*epsilon
                    expected[i] = (plus-objective(p, initial))/(2*epsilon)
                np.testing.assert_allclose(total[key], expected, rtol=4e-4, atol=2e-7, err_msg=key)
            expected = np.zeros(initial.shape)
            for index in np.ndindex(initial.shape):
                h = initial.astype(np.float64)
                h[index] += epsilon
                plus = objective(params, h)
                h[index] -= 2*epsilon
                expected[index] = (plus-objective(params, h))/(2*epsilon)
            np.testing.assert_allclose(dh, expected, rtol=4e-4, atol=2e-7)

    def test_jax_two_ply_objective_all_gradients_and_three_free_updates(self):
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            self.skipTest('Install CPU JAX for independent parity')
        from flygo.jax.model import forward, teacher_loss, adam
        jax.config.update('jax_enable_x64', True)
        graph, config, ports, baseline, x, legal, policy, value = fixture()
        initial = np.array([[.3, -.2], [.15, .4], [-.13, .25], [.5, -.1]], np.float32)
        xs = [x, x[::-1].copy()*.7]
        mask = HeadMask(np.array([[1, 0], [0, 1], [1, 1]], np.uint8),
                        np.array([0, 1], np.uint8), '{"fixture":"state-vjp"}')
        for softness, clip_mode in ((0., 'global'), (.05, 'parameter-group')):
            cfg = replace(config, rate_softness=softness, readout_mean_scale=.4)
            params = {k: v.copy() for k, v in baseline.items()}
            for k, enabled in mask.parameter_masks().items():
                params[k] *= enabled
            model = RustFly(graph, cfg, ports=ports, params=params, head_mask=mask,
                            clip_mode=clip_mode, value_core_scale=.1)
            kwargs = dict(steps=cfg.steps, groups=cfg.groups, actions=cfg.actions,
                          rate_softness=softness, readout_mean_scale=cfg.readout_mean_scale,
                          head_mask=mask.parameter_masks())
            def objective(p, h):
                loss = jnp.float32(0)
                for current in xs:
                    out = forward(p, graph, ports, current, initial_state=h, **kwargs)
                    loss += teacher_loss(out, legal, policy, value)[0]/len(xs)
                    h = out['states'][-1]
                return loss
            gradient = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))
            first = {k: jnp.zeros_like(v) for k, v in params.items()}
            second = {k: jnp.zeros_like(v) for k, v in params.items()}
            for update in range(1, 4):
                _, (expected, expected_initial) = gradient(params, initial)
                inputs, outputs = [], []
                h = initial
                for current in xs:
                    inputs.append(h)
                    out = model.infer_state(current, h, trace=True)
                    reference = forward(params, graph, ports, current, initial_state=h, **kwargs)
                    for key in ('logits', 'value', 'states'):
                        np.testing.assert_allclose(out[key], reference[key], rtol=3e-4, atol=1e-7)
                    outputs.append(out)
                    h = out['next_state']
                total = {k: np.zeros_like(v) for k, v in params.items()}
                dh = None
                for t in reversed(range(len(xs))):
                    dl, dv = teacher_cotangents(outputs[t], legal, policy, value, 1/len(xs))
                    grad, dh = model.state_vjp(xs[t], inputs[t], revision=outputs[t]['revision'],
                                              dlogits=dl, dvalue=dv, dstate=dh)
                    for k in PARAMETERS:
                        total[k] += grad[k]
                for k in PARAMETERS:
                    np.testing.assert_allclose(total[k], expected[k], rtol=4e-4, atol=2e-7, err_msg=k)
                np.testing.assert_allclose(dh, expected_initial, rtol=4e-4, atol=2e-7)
                params, first, second, norm = adam(params, expected, first, second, update,
                    rate=.02, clip=.03, epsilon=1e-6, clip_mode=clip_mode)
                result = model.apply_gradients(total, revision=outputs[0]['revision'],
                    rate=.02, clip=.03, epsilon=1e-6)
                self.assertEqual(result['step'], update)
                self.assertAlmostEqual(result['gradient_norm'], float(norm), delta=2e-6)
                saved = model.checkpoint_arrays()
                for prefix, want in [('param/', params), ('first/', first), ('second/', second)]:
                    for k in PARAMETERS:
                        np.testing.assert_allclose(saved[prefix+k], want[k], rtol=5e-4, atol=2e-6, err_msg=prefix+k)

    def test_malformed_inputs_and_stale_revisions_reject_before_mutation(self):
        graph, cfg, ports, params, x, *_ = fixture()
        model = RustFly(graph, cfg, ports=ports, params=params)
        initial = model.initial_state(len(x))
        output = model.infer_state(x, initial)
        revision = output['revision']
        saved = model.checkpoint_arrays()
        for bad in (initial.T, initial.ravel(), initial[:1], initial*np.nan, initial*np.inf):
            with self.assertRaises(ValueError):
                model.infer_state(x, bad)
            with self.assertRaises(ValueError):
                model.state_vjp(x, initial, revision=revision, dstate=bad)
        for extra in (dict(dlogits=np.zeros((3, 2))), dict(dvalue=np.zeros((2, 1))),
                      dict(dvalue=np.full(2, np.nan)), dict(dlogits=np.full((2, 3), np.inf))):
            with self.assertRaises(ValueError):
                model.state_vjp(x, initial, revision=revision, **extra)
        grad, _ = model.state_vjp(x, initial, revision=revision, dstate=np.ones_like(initial))
        invalid = {k: v.copy() for k, v in grad.items()}
        invalid['edge'][0] = np.nan
        with self.assertRaises(ValueError):
            model.apply_gradients(invalid, revision=revision)
        for k, v in saved.items():
            np.testing.assert_array_equal(model.checkpoint_arrays()[k], v)
        model.apply_gradients(grad, revision=revision)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            model.state_vjp(x, initial, revision=revision)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            model.apply_gradients(grad, revision=revision)
        new_revision = model.infer_state(x, initial)['revision']
        model.restore_arrays(saved)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            model.state_vjp(x, initial, revision=new_revision)
        np.testing.assert_array_equal(model.infer_state(x, initial)['next_state'], output['next_state'])


if __name__ == '__main__':
    unittest.main()
