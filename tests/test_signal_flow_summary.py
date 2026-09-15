"""Prevent partial or failed audits from becoming a positive research report."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_signal_flow import checked, condense, signal


class SignalSummaryTests(unittest.TestCase):
    def test_failed_missing_and_nonfinite_checks_cannot_pass(self):
        good = dict(passed=True, mismatches=0, max_abs=1e-8, relative_l2=1e-6)
        self.assertEqual(checked([good], 1)['comparisons'], 1)
        bad = [[], [dict(good, passed=False)], [dict(good, mismatches=1)],
               [dict(good, max_abs=float('nan'))], [dict(good, relative_l2=-1)]]
        for rows in bad:
            with self.assertRaises(ValueError):
                checked(rows, 1)

    def test_missing_gradient_objective_cannot_be_summarized(self):
        plan = dict(checkpoint_steps=[0], gradient_cases=['original-policy', 'original-value'], passes=1)
        report = dict(status='complete', quantiles=[0., .1, .5, .9, .99, 1.], records=[
            dict(step=0, parameters_unchanged=True, gradients=[dict(objective='original-policy')],
                 signals=[dict(step=0), dict(step=1)])])
        with self.assertRaises(ValueError):
            condense(report, plan)

    def test_leading_share_uses_sum_of_variances_and_handles_zero_signal(self):
        # Two response standard deviations, 3 and 4: leading share is 16/25.
        visual = dict(count=2, rms=(25 / 2) ** .5, quantiles=[3, 3.1, 3.5, 3.9, 3.99, 4])
        result = signal(dict(visual_std=visual, std=dict(rms=6),
                             positive_fraction=dict(mean=.75, nonzero=2)))
        self.assertAlmostEqual(result['leading_visual_variance_share'], 16 / 25)
        visual.update(rms=0, quantiles=[0] * 6)
        result = signal(dict(visual_std=visual, std=dict(rms=6),
                             positive_fraction=dict(mean=.75, nonzero=2)))
        self.assertIsNone(result['leading_visual_variance_share'])


if __name__ == '__main__':
    unittest.main()
