import unittest
import numpy as np
from flygo.readout_probe import initialize, forward, loss_and_grad, transform, normalizer, adam


class ReadoutProbeTests(unittest.TestCase):
    def test_identical_initial_policies_for_linear_and_nonlinear_heads(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(7, 11)); baseline = rng.normal(size=(7, 5))
        for hidden in (0, 13):
            np.testing.assert_array_equal(forward(initialize(11, 5, hidden=hidden), x, baseline), baseline)

    def test_train_statistics_and_competitive_gating(self):
        training = np.array([[2, 0, 1, 5], [2, 2, 3, 5]], np.float64)
        mean, scale, std = normalizer(training)
        np.testing.assert_array_equal(std, [0, 1, 1, 0])
        probe = mean+scale*np.array([[3, 3, 2, -8]], np.float64)
        gated = transform(probe, 'gated-linear', mean, scale, top_k=2)
        np.testing.assert_allclose(gated, [[3, 3, 0, 0]])
        np.testing.assert_array_equal(mean, [2, 1, 2, 5])
        self.assertEqual(np.count_nonzero(gated), 2)

    def test_directional_derivatives_include_target_mass_and_mask(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(7, 11)); baseline = rng.normal(size=(7, 5))
        legal = rng.uniform(size=(7, 5)) > .2; legal[:, 0] = True
        q = rng.uniform(size=(7, 5))*legal
        q = q/q.sum(axis=1, keepdims=True)*rng.uniform(.9, 1.1, size=(7, 1))
        for hidden in (0, 13):
            p = initialize(11, 5, hidden=hidden)
            p['weight'][:] = rng.normal(0, .1, p['weight'].shape)
            _, g = loss_and_grad(p, x, baseline, legal, q)
            direction = {k:rng.normal(size=v.shape) for k,v in p.items()}
            step = 1e-6
            plus = {k:v+step*direction[k] for k,v in p.items()}
            minus = {k:v-step*direction[k] for k,v in p.items()}
            numeric = (loss_and_grad(plus, x, baseline, legal, q)[0]-loss_and_grad(minus, x, baseline, legal, q)[0])/(2*step)
            analytic = sum(np.sum(g[k]*direction[k]) for k in p)
            self.assertAlmostEqual(numeric, analytic, places=7)

    def test_serialized_moments_reproduce_next_update(self):
        rng = np.random.default_rng(15)
        x = rng.normal(size=(7, 11)); baseline = rng.normal(size=(7, 5))
        q = np.full((7, 5), .2); legal = np.ones_like(q, bool)
        p = initialize(11, 5, hidden=13)
        m = {k:np.zeros_like(v) for k,v in p.items()}; v = {k:np.zeros_like(z) for k,z in p.items()}
        for step in range(1, 4):
            _, g = loss_and_grad(p, x, baseline, legal, q)
            adam(p, g, m, v, step, rate=.3)
        copies = [{k:a.copy() for k,a in group.items()} for group in (p, m, v)]
        for pp, mm, vv in [(p,m,v), copies]:
            _, g = loss_and_grad(pp, x, baseline, legal, q)
            adam(pp, g, mm, vv, 4, rate=.3)
        for a,b in zip((p,m,v),copies):
            for key in a:np.testing.assert_array_equal(a[key],b[key])


if __name__ == '__main__': unittest.main()
