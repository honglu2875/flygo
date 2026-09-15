"""Decomposition must preserve native objectives and all model/optimizer state."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_loss_balance import decomposition_check, fingerprints, geometry, products, separate_gradients
from flygo.fly import PARAMETERS, RustFly
from test_fly import fixture


class LossBalance(unittest.TestCase):
    def test_raw_geometry_distinguishes_conflict_dominance_and_silent_groups(self):
        opposing = geometry(products(np.array([1., 0.]), np.array([-2., 0.])))
        self.assertEqual(opposing['cosine'], -1.)
        self.assertEqual(opposing['value_to_policy_norm'], 2.)
        self.assertEqual(opposing['joint_sgd_policy_dot'], -1.)
        self.assertEqual(opposing['joint_sgd_value_dot'], 2.)
        orthogonal = geometry(products(np.array([1., 0.]), np.array([0., 1.])))
        self.assertEqual(orthogonal['cosine'], 0.)
        self.assertIsNone(geometry(products(np.zeros(3), np.ones(3)))['cosine'])
        with self.assertRaises(ValueError): products(np.array([np.nan]), np.ones(1))
        with self.assertRaises(ValueError): products(np.zeros(2), np.zeros(3))

    def test_native_joint_loss_decomposes_with_masked_nonunit_mass_and_unchanged_state(self):
        graph, config, ports, params, x, legal, policy, value = fixture()
        policy *= np.float32(.99992)
        for mode in ('global', 'parameter-group'):
            model = RustFly(graph, config, ports=ports, params=params, clip_mode=mode)
            # Check the diagnostic after Adam state exists, not just at initialization.
            for _ in range(2): model.train_step(x, legal, policy, value, rate=.01)
            before = fingerprints(model.checkpoint_arrays())
            _, gradients = separate_gradients(model, x, legal, policy, value)
            checked = decomposition_check(gradients, rtol=3e-4, atol=3e-6)
            self.assertEqual(set(checked), set(PARAMETERS))
            self.assertTrue(all(row['passed'] for row in checked.values()), checked)
            self.assertEqual(before, fingerprints(model.checkpoint_arrays()))
            gradients['joint']['bias'][0] += 1
            self.assertFalse(decomposition_check(gradients, rtol=3e-4, atol=3e-6)['bias']['passed'])


if __name__ == '__main__': unittest.main()
