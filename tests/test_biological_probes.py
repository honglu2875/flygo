"""Scientific failure modes in the read-only correlation diagnostic."""
import unittest

import numpy as np

from scripts.probe_biological_ports import (
    correlations, jitter_edges, matrix_agreement, pair_summary,
    policy_summary, residualize,
)


class BiologicalProbes(unittest.TestCase):
    def test_silent_cells_are_missing_and_opponent_responses_keep_their_sign(self):
        signal = np.arange(-2, 3, dtype=float)
        matrix, valid, summary = correlations(np.column_stack([signal, 2*signal, -signal, np.ones(5)]))
        np.testing.assert_array_equal(valid, [True, True, True, False])
        self.assertTrue(np.isnan(matrix[-1]).all())
        self.assertEqual(matrix[0, 1], 1)
        self.assertEqual(matrix[0, 2], -1)
        self.assertAlmostEqual(summary['covariance_participation_rank'], 1)
        pairs = pair_summary(matrix, ['a', 'a', 'b', 'b'])
        self.assertEqual(pairs['within_annotation']['pairs'], 1)
        self.assertEqual(pairs['between_annotations']['pairs'], 2)
        self.assertEqual(pairs['between_annotations']['mean'], -1)
        self.assertAlmostEqual(matrix_agreement(matrix, matrix)['coefficient_correlation'], 1)

    def test_shared_input_can_hide_opponent_selectivity(self):
        a = np.tile([-1., -1., 1., 1.], 8)
        b = np.tile([-1., 1., -1., 1.], 8)
        response = np.column_stack([100*a+b, 100*a-b, 100*a])
        raw, _, _ = correlations(response)
        self.assertGreater(raw[0, 1], .99)
        residual, rank = residualize(response, a[:, None])
        conditioned, valid, _ = correlations(residual)
        self.assertEqual(rank, 2)
        np.testing.assert_array_equal(valid, [True, True, False])
        self.assertAlmostEqual(conditioned[0, 1], -1)

    def test_tiny_numerical_variation_does_not_become_a_functional_group(self):
        x = np.arange(10, dtype=float)
        _, valid, summary = correlations(np.column_stack([x, x*1e-9]))
        np.testing.assert_array_equal(valid, [True, False])
        self.assertEqual(summary['varying_cells'], 1)
        with self.assertRaises(ValueError):
            correlations(np.array([[0., np.nan], [1., 2.], [2., 3.]]))

    def test_strength_jitter_preserves_incoming_mass_without_adding_edges(self):
        graph = dict(dst=np.array([1, 1, 2, 2, 2]), type_id=np.zeros(4, int))
        magnitude = np.array([.2, .3, .1, .4, .8])
        edge = np.log(np.expm1(magnitude)).astype(np.float32)
        np.testing.assert_array_equal(jitter_edges(graph, edge, 0, .1), edge)
        changed = jitter_edges(graph, edge, 1, .1)
        self.assertEqual(changed.shape, edge.shape)
        self.assertFalse(np.array_equal(changed, edge))
        recovered = np.logaddexp(0., changed.astype(np.float64))
        self.assertTrue(np.all(recovered > 0))
        np.testing.assert_allclose(np.bincount(graph['dst'], weights=recovered),
                                   np.bincount(graph['dst'], weights=magnitude), rtol=2e-7)

    def test_sharpness_respects_legality_and_single_action_positions(self):
        legal = np.array([[1, 1, 0], [1, 0, 0]], bool)
        report = policy_summary(np.array([[0., 0., 100.], [0., 100., 100.]]), legal)
        self.assertAlmostEqual(report['mean_entropy'], np.log(2)/2)
        self.assertAlmostEqual(report['mean_effective_actions'], 1.5)
        self.assertAlmostEqual(report['mean_max_probability'], .75)


if __name__ == '__main__':
    unittest.main()
