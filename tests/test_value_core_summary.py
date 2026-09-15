"""Three-arm comparisons retain every seed, family weighting and value tradeoff."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_clipping import (CONFIDENCE, VALUE_CORE_FACTOR, optimizer_contract,
                               paired_confidence, study_decision, validate_contract)
from summarize_readout_roles import METRICS, paired_results
import queue_cpu


class ValueCoreSummary(unittest.TestCase):
    def test_only_the_registered_scale_comparisons_are_accepted(self):
        path = Path(__file__).resolve().parents[1] / 'configs/group-clipping-analysis-v1.json'
        plan = json.loads(path.read_text()); plan.pop('contrast')
        plan.update(factor=VALUE_CORE_FACTOR, arms=['full', 'reduced', 'policy-only'],
                    value_core_arms={'full': 1., 'reduced': .1, 'policy-only': 0.},
                    clip_mode='parameter-group', seeds=[16, 17, 18], endpoint_update=4000,
                    contrasts=[dict(candidate=a, reference='full') for a in ('reduced', 'policy-only')])
        validate_contract(plan)
        for changes in [dict(clip_mode='global'), dict(value_core_arms={'full': 1., 'reduced': .01, 'policy-only': 0.}),
                        dict(contrasts=plan['contrasts'][:1]), dict(contrasts=list(reversed(plan['contrasts']))),
                        dict(seeds=[16, 16, 18]), dict(factor='unregistered propagation change')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_contract(dict(plan, **changes))
        self.assertEqual(optimizer_contract(plan, 'reduced'),
                         ('value_core_scale', 'parameter-group', float(np.float32(.1))))
        self.assertEqual(optimizer_contract({}, 'global'), ('clip_mode', 'global', 1.))

    def test_both_candidates_keep_all_seed_and_family_contrasts(self):
        seeds = [16, 17, 18]; arms = ['full', 'reduced', 'policy-only']
        comparisons = [dict(candidate=a, reference='full') for a in arms[1:]]
        families = np.array(['large', 'large', 'large', 'small'])
        novel = np.array([True, False, False, True])
        matrices = {}
        for i, seed in enumerate(seeds):
            matrices[seed, 'full'] = np.ones((4, 3))
            matrices[seed, 'reduced'] = matrices[seed, 'full'] + [-.1*(i+1), -.05, .01]
            matrices[seed, 'policy-only'] = matrices[seed, 'full'] + [-.2*(i+1), .1, .02]
            matrices[seed, 'policy-only'][-1, 0] -= .4
        result = paired_results(matrices, families, novel, seeds, arms, comparisons)
        confidence = paired_confidence({k: v[:, :2] for k, v in matrices.items()},
                                       families, novel, seeds, arms, comparisons)
        self.assertEqual(len(result['records']), 9)
        self.assertEqual(len(confidence['contrasts']), 6)
        self.assertEqual(len(confidence['pooled_contrasts']), 2)
        for left, right in zip(confidence['pooled_contrasts'], result['pooled_contrasts']):
            for metric, original in zip(CONFIDENCE, METRICS):
                self.assertEqual(left['natural'][metric], right['natural'][original])
                self.assertEqual(left['seed_variation']['natural'][metric],
                                 right['seed_variation']['natural'][original])
        decisions = study_decision(result)['contrasts']
        self.assertTrue(decisions[0]['provisional_candidate'])
        self.assertFalse(decisions[1]['provisional_candidate'])
        self.assertTrue(decisions[1]['policy_improves_each_seed'])
        self.assertGreater(decisions[1]['mean_value_difference'], 0)
        missing = {k: v for k, v in matrices.items() if k != (18, 'policy-only')}
        with self.assertRaises(ValueError): paired_results(missing, families, novel, seeds, arms, comparisons)


class QueueObservation(unittest.TestCase):
    def test_transient_observation_does_not_start_or_discard_a_job(self):
        for failure in [subprocess.TimeoutExpired('ssh', 30), subprocess.CalledProcessError(255, 'ssh')]:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); config = root / 'config.json'
                config.write_text(json.dumps(dict(root=str(root), source='qualified', affinity=[116], timeout_seconds=3600)))
                (root / 'registration.json').write_text(json.dumps(dict(jobs=[dict(run_id='trial', host=1, wait_for=['previous'])])))
                observed = [failure, [dict(state='complete', ready=True)]]
                with patch.object(queue_cpu, 'pin'), patch.object(queue_cpu.time, 'sleep'), \
                        patch.object(queue_cpu, 'dependencies', side_effect=observed) as check, \
                        patch.object(queue_cpu.subprocess, 'run') as launch:
                    queue_cpu.worker(config)
                self.assertEqual(check.call_count, 2)
                self.assertEqual(launch.call_count, 1)
                self.assertEqual(json.loads((root / 'status.json').read_text())['launched'], ['trial'])


if __name__ == '__main__': unittest.main()
