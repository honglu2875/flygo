"""Independent objective derivatives, head preservation and checkpoint semantics."""
import contextlib
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from flygo.fly import RustFly, PARAMETERS
from flygo.objectives import (OBJECTIVE_VERSION, value_core_scale, saved_value_core_scale,
                              check_objective_restore, check_objective_resume)
from test_fly import fixture

SHARED = PARAMETERS[:5]


class ValueCore(unittest.TestCase):
    def test_heads_losses_and_predictions_stay_equal_and_zero_scale_is_policy_only(self):
        graph, config, ports, params, *batch = fixture()
        batch[2] *= np.float32(.99992)
        for softness in (0., .01):
            cfg = replace(config, rate_softness=softness, readout_mean_scale=.5)
            full = RustFly(graph, cfg, ports=ports, params=params)
            expected = full.infer(batch[0], trace=True)
            losses, joint = full.loss_and_grad(*batch)
            _, policy_only = full.loss_and_grad(*batch[:3], expected['value'].copy())
            for scale in (0., .1, 1., 2.):
                model = RustFly(graph, cfg, ports=ports, params=params, value_core_scale=scale)
                actual_loss, actual = model.loss_and_grad(*batch)
                self.assertEqual(actual_loss, losses)
                for key, value in model.infer(batch[0], trace=True).items():
                    np.testing.assert_array_equal(value, expected[key])
                for name in PARAMETERS[5:]:
                    np.testing.assert_array_equal(actual[name], joint[name])
                for name in SHARED:
                    target = policy_only[name] + np.float32(scale) * (joint[name] - policy_only[name])
                    np.testing.assert_allclose(actual[name], target, rtol=3e-4, atol=3e-6, err_msg=name)
                    if scale == 0: np.testing.assert_array_equal(actual[name], policy_only[name])
                self.assertGreater(np.linalg.norm(actual['value_weight']), 0)

    def test_derivatives_match_weighted_shared_loss_and_unweighted_head_loss(self):
        graph, config, ports, params, *batch = fixture()
        scale = value_core_scale(.1)
        model = RustFly(graph, config, ports=ports, params=params, value_core_scale=scale)
        _, gradient = model.loss_and_grad(*batch)
        for name in PARAMETERS:
            for index in range(len(params[name])):
                measured = []
                step = np.float32(.002)
                for sign in (-1, 1):
                    perturbed = {key: value.copy() for key, value in params.items()}
                    perturbed[name][index] += sign * step
                    candidate = RustFly(graph, config, ports=ports, params=perturbed)
                    parts, _ = candidate.loss_and_grad(*batch)
                    coefficient = scale if name in SHARED else 1.
                    measured.append(parts['policy_loss'] + coefficient * parts['value_loss'])
                derivative = (measured[1] - measured[0]) / (2 * float(step))
                np.testing.assert_allclose(gradient[name][index], derivative, rtol=.02, atol=2e-5,
                                           err_msg=f'{name}[{index}]')

    def test_default_and_explicit_one_preserve_exact_updates_and_legacy_arrays(self):
        graph, config, ports, params, *batch = fixture()
        for mode in ('global', 'parameter-group'):
            left = RustFly(graph, config, ports=ports, params=params, clip_mode=mode)
            right = RustFly(graph, config, ports=ports, params=params, clip_mode=mode, value_core_scale=1.)
            for _ in range(3): self.assertEqual(left.train_step(*batch), right.train_step(*batch))
            self.assertNotIn('objective_value_core_scale', left.checkpoint_arrays())
            for key, value in left.checkpoint_arrays().items():
                np.testing.assert_array_equal(value, right.checkpoint_arrays()[key])
            with self.assertRaises(AttributeError): left.value_core_scale = .1

    def test_invalid_scales_and_wrong_array_restore_cannot_change_state(self):
        graph, config, ports, params, *batch = fixture()
        for bad in (-1, np.nan, np.inf, 1e100, 1e-100, None, [], 1j, 'invalid'):
            with self.assertRaises(ValueError): value_core_scale(bad)
        models = [RustFly(graph, config, ports=ports, params=params, value_core_scale=s) for s in (0., .1, 1.)]
        for model in models:
            before = model.checkpoint_arrays()
            for other in models:
                if other is model: continue
                with self.assertRaisesRegex(ValueError, 'scale differs'): model.restore_arrays(other.checkpoint_arrays())
            for tag in (np.asarray(.1), np.asarray([.1], np.float32), np.asarray(np.nan, np.float32)):
                with self.assertRaises(ValueError): check_objective_restore(.1, {'objective_value_core_scale': tag})
            for key, value in before.items(): np.testing.assert_array_equal(value, model.checkpoint_arrays()[key])
        check_objective_resume(1., {})
        self.assertEqual(saved_value_core_scale({}), 1.)
        with self.assertRaisesRegex(ValueError, 'differs'): check_objective_resume(.1, {})
        with self.assertRaisesRegex(ValueError, 'Inconsistent'):
            saved_value_core_scale(dict(value_core_scale=.1, objective_version=OBJECTIVE_VERSION,
                                        training_contract=dict(value_core_scale=1.)))

    def test_checkpoint_replays_next_update_and_binds_metadata(self):
        from flygo.checkpoint import save_checkpoint, load_checkpoint
        from flygo.data.loader import Sampler
        graph, config, ports, params, *batch = fixture(); graph['manifest'] = dict(graph_id='fixture')
        model = RustFly(graph, config, ports=ports, params=params, value_core_scale=.1)
        sampler = Sampler({}, {}, 19); model.train_step(*batch)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path = root / 'checkpoint.npz'
            metadata = dict(dataset_id='fixture', training_contract=dict(value_core_scale=model.value_core_scale))
            with patch('flygo.checkpoint.StorageBudget.reserve', return_value=contextlib.nullcontext()):
                save_checkpoint(model, sampler, path, metadata, root=root)
            expected = model.train_step(*batch); arrays = model.checkpoint_arrays()
            restored = RustFly(graph, config, ports=ports, params=params, value_core_scale=.1)
            info = load_checkpoint(path, restored, sampler)
            self.assertEqual(info['objective_version'], OBJECTIVE_VERSION)
            self.assertEqual(saved_value_core_scale(info), restored.value_core_scale)
            self.assertEqual(restored.train_step(*batch), expected)
            for key, value in arrays.items(): np.testing.assert_array_equal(value, restored.checkpoint_arrays()[key])
            wrong = RustFly(graph, config, ports=ports, params=params)
            with self.assertRaisesRegex(ValueError, 'scale differs'): load_checkpoint(path, wrong)

    def test_qualification_does_not_substitute_another_gradient_rule(self):
        from flygo.attachments import require_qualification, runtime_hashes
        _, config, *_ = fixture(); contract = dict(mode='current')
        optimizer = dict(rate=.03, epsilon=1e-6, clip=1., rate_scales={'bias':.01})
        record = dict(input_contract=contract, head_mask=None, batch_size=2, updates=3,
                      model=asdict(config), optimizer=optimizer)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'qualification.json'
            path.write_text(json.dumps(dict(status='complete', runtime_sha256=runtime_hashes(), records=[record])))
            require_qualification(path, contract, config, batch_size=2, **optimizer)
            with self.assertRaisesRegex(ValueError, 'No matching'):
                require_qualification(path, contract, config, batch_size=2, value_core_scale=.1, **optimizer)

    def test_rust_jax_gradients_updates_masks_and_recovery(self):
        try: import jax
        except ImportError: self.skipTest('JAX optional dependency')
        from flygo.jax.learner import JaxFly
        from flygo.readout import HeadMask
        graph, config, ports, params, *small = fixture()
        batch = [np.concatenate([a] * jax.device_count()) for a in small]
        masks = [None, HeadMask(np.asarray([[1,0],[1,1],[0,1]], bool), np.asarray([1,0], bool),
                                json.dumps(dict(kind='value-core-test')))]
        for scale, mask in [(0., masks[1]), (.1, masks[0]), (.1, masks[1])]:
            rust = RustFly(graph, config, ports=ports, params=params, head_mask=mask, value_core_scale=scale)
            reference = JaxFly(graph, config, ports=ports, params=params, head_mask=mask, value_core_scale=scale)
            settings = dict(rate=.003, clip=.01, epsilon=1e-6, rate_scales={'bias':.01})
            for _ in range(3):
                left_loss, left = rust.loss_and_grad(*batch); right_loss, right = reference.loss_and_grad(*batch)
                np.testing.assert_allclose(list(left_loss.values()), list(right_loss.values()), rtol=3e-4, atol=3e-6)
                for name in PARAMETERS: np.testing.assert_allclose(right[name], left[name], rtol=3e-4, atol=3e-6, err_msg=name)
                rust.train_step(*batch, **settings); reference.train_step(*batch, **settings)
                for key, value in rust.checkpoint_arrays().items():
                    np.testing.assert_allclose(reference.checkpoint_arrays()[key], value, rtol=3e-4, atol=3e-6, err_msg=key)
            for model in (rust, reference):
                saved = model.checkpoint_arrays(); model.train_step(*batch, **settings); expected = model.checkpoint_arrays()
                model.restore_arrays(saved); model.train_step(*batch, **settings)
                for key, value in expected.items(): np.testing.assert_array_equal(value, model.checkpoint_arrays()[key])


if __name__ == '__main__': unittest.main()
