"""Keep position weighting, pairing and fitted-seed uncertainty distinct."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import summarize_motor_convergence as summary


class MotorConvergenceSummary(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.plan_path = Path(temporary.name) / 'plan.json'
        self.plan_path.write_text(json.dumps(dict(banks=[dict(seed=s) for s in (4, 5, 6)],
            variants=['bias-only', 'scaled-linear'], ridge=[.01], scope='synthetic paired test')))
        self.reports = []
        for seed, delta in zip((4, 5, 6), (1., 3., 5.)):
            base = np.array([[1.]*5, [9.]*5])  # Family sizes 1 and 3, means 1 and 3.
            totals = base + np.array([1, 3])[:, None]*delta
            trials = [dict(metrics={'train':dict.fromkeys(summary.METRICS, 1.)},
                           sufficiently_converged=True, optimization_gap_bound=0.,
                           optimizer_success=True, iterations=3, fit_label_exposures=40) for _ in range(2)]
            self.reports.append(dict(status='complete', seed=seed, plan_sha256=summary.sha256(self.plan_path),
                analysis_sha256=summary.sha256(Path(summary.__file__)), families=['a', 'b'], counts=[1, 3],
                bank={'selection':{s:dict(indices_sha256='paired') for s in ('train', 'validation')}},
                source=dict.fromkeys(('worker_sha256', 'convex_sha256', 'probe_sha256', 'scipy', 'numpy',
                                      'lbfgsb_sha256', 'lbfgsb_python_sha256'), 'same'),
                baseline_totals=base.tolist(), trial_totals=[base.tolist(), totals.tolist()],
                result={'trials':trials}))

    def test_position_weighting_pair_direction_and_seed_variation(self):
        result = summary.combine(self.plan_path, self.reports[::-1])
        self.assertEqual(result['baseline']['policy_kl']['mean'], 2.5)
        contrast = result['records'][1]['minus_bias_only']['policy_kl']
        self.assertEqual(contrast['mean'], 3.)
        self.assertEqual(contrast['per_seed'], [1., 3., 5.])
        self.assertEqual(contrast['seed_sd'], 2.)
        np.testing.assert_array_equal(contrast['family_bootstrap_95'], [3., 3.])
        self.assertEqual(result['fit_label_exposures'], 240)

    def test_different_positions_or_family_multiplicities_break_pairing(self):
        bad = copy.deepcopy(self.reports)
        bad[1]['counts'] = [2, 2]
        with self.assertRaisesRegex(ValueError, 'not paired'):
            summary.combine(self.plan_path, bad)
        bad = copy.deepcopy(self.reports)
        bad[1]['bank']['selection']['validation']['indices_sha256'] = 'other'
        with self.assertRaisesRegex(ValueError, 'positions differ'):
            summary.combine(self.plan_path, bad)

    def test_missing_seed_or_changed_solver_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'registered seeds'):
            summary.combine(self.plan_path, self.reports[:-1])
        bad = copy.deepcopy(self.reports)
        bad[2]['source']['lbfgsb_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'implementations differ'):
            summary.combine(self.plan_path, bad)


if __name__ == '__main__':
    unittest.main()
