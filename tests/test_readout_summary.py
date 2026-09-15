"""A paired comparison must preserve identities, family dependence and seed variation."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_readout_roles import aligned_metrics, paired_results


class ReadoutSummary(unittest.TestCase):
    def test_pair_direction_and_seed_uncertainty_are_separate(self):
        families = np.array(['a', 'a', 'b', 'b'])
        novel = np.array([True, False, False, True])
        matrices = {}
        for seed, delta in zip((4, 5, 6), (1., 3., 5.)):
            matrices[seed, 'dense'] = np.full((4, 3), seed * 10.)
            matrices[seed, 'side'] = matrices[seed, 'dense'] + delta
        result = paired_results(matrices, families, novel, [4, 5, 6], ['dense', 'side'],
                                [dict(candidate='side', reference='dense')], bootstrap=100)
        contrast = result['pooled_contrasts'][0]
        self.assertEqual(contrast['natural']['policy_kl']['mean'], 3.)
        # Every family has the same seed-averaged difference: its conditional
        # interval has zero width, even though the fitted seeds vary by SD 2.
        self.assertEqual(contrast['natural']['policy_kl']['family_bootstrap_95'], [3., 3.])
        self.assertEqual(contrast['seed_variation']['natural']['policy_kl'],
                         dict(per_seed=[1., 3., 5.], sample_sd=2.))
        self.assertEqual(contrast['current_reduced_source_novel']['positions'], 2)
        self.assertEqual(contrast['current_reduced_source_novel']['families'], 2)
        del matrices[6, 'side']
        with self.assertRaisesRegex(ValueError, 'Every registered arm'):
            paired_results(matrices, families, novel, [4, 5, 6], ['dense', 'side'], [])

    def test_position_weighting_and_family_resampling_are_not_interchanged(self):
        families = np.array(['large', 'large', 'large', 'small'])
        values = np.repeat(np.array([[0.], [0.], [0.], [4.]]), 3, axis=1)
        matrices = {(s, 'dense'): values for s in (4, 5, 6)}
        result = paired_results(matrices, families, np.ones(4, bool), [4, 5, 6], ['dense'], [], bootstrap=1000)
        natural = result['pooled_records'][0]['natural']
        self.assertEqual(natural['policy_kl']['mean'], 1.)  # position-weighted, not family mean 2
        self.assertEqual(natural['policy_kl']['family_bootstrap_95'], [0., 4.])
        self.assertEqual(natural['families'], 2)

    def test_reordered_positions_or_changed_families_cannot_be_paired(self):
        indices = np.array([21, 23, 28]); games = np.array([0, 0, 1]); names = np.array(['a', 'b'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.npz'
            def write(**changes):
                fields = dict(indices=indices, game_index=games, metrics=np.zeros((3, 3)))
                fields.update(changes); np.savez(path, **fields)
            write(); _, families = aligned_metrics(path, indices, names)
            np.testing.assert_array_equal(families, ['a', 'a', 'b'])
            write(indices=indices[::-1])
            with self.assertRaises(AssertionError): aligned_metrics(path, indices, names)
            write(game_index=np.array([0, 1, 1]))
            with self.assertRaises(AssertionError): aligned_metrics(path, indices, names, families)
            write(game_index=np.array([0, 0, 2]))
            with self.assertRaisesRegex(ValueError, 'game identities'): aligned_metrics(path, indices, names)
            write(metrics=np.full((3, 3), np.nan))
            with self.assertRaisesRegex(ValueError, 'Invalid validation'): aligned_metrics(path, indices, names)


if __name__ == '__main__':
    unittest.main()
