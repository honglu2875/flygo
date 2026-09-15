"""Clipping conclusions must retain paired seeds and observed update semantics."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from attachment_endpoint import initial_state
from summarize_clipping import CONFIDENCE, PARAMETERS, clipping_history, decision, paired_confidence, validate_contract
from summarize_readout_roles import METRICS, paired_results


class ClippingSummary(unittest.TestCase):
    def test_new_registered_horizon_retains_decision_direction_and_sampling_rules(self):
        plan = json.loads((Path(__file__).resolve().parents[1] / 'configs/group-clipping-analysis-v1.json').read_text())
        validate_contract(plan)
        longer = dict(plan, seeds=[13, 14, 15], endpoint_update=4000)
        validate_contract(longer)
        for changes in [dict(seeds=[13, 13, 15]), dict(seeds=[13]), dict(endpoint_update=3999),
                        dict(contrast=dict(candidate='global', reference='parameter-group')),
                        dict(bootstrap=dict(plan['bootstrap'], unit='position'))]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_contract(dict(longer, **changes))

    def test_confidence_uses_the_registered_pairing_and_family_weighting(self):
        families = np.array(['large', 'large', 'large', 'small'])
        novel = np.array([True, False, False, True])
        seeds = [10, 11, 12]; arms = ['global', 'parameter-group']
        contrast = dict(candidate=arms[1], reference=arms[0])
        matrices = {}
        for seed, delta in zip(seeds, [1., 3., 5.]):
            matrices[seed, arms[0]] = np.zeros((4, 2))
            matrices[seed, arms[1]] = np.array([[0., 1.], [0., 1.], [0., 1.], [4., 3.]]) + delta
        result = paired_confidence(matrices, families, novel, seeds, arms, contrast)
        padded = {key: np.column_stack([value, np.zeros(4)]) for key, value in matrices.items()}
        registered = paired_results(padded, families, novel, seeds, arms, [contrast])
        for left, right in zip(result['pooled_contrasts'], registered['pooled_contrasts']):
            for metric, original in zip(CONFIDENCE, METRICS):
                self.assertEqual(left['natural'][metric], right['natural'][original])
                self.assertEqual(left['seed_variation']['natural'][metric], right['seed_variation']['natural'][original])
        self.assertEqual(result['pooled_contrasts'][0]['natural']['policy_entropy']['mean'], 4.)
        self.assertEqual(result['pooled_contrasts'][0]['seed_variation']['natural']['policy_entropy']['sample_sd'], 2.)
        del matrices[12, 'global']
        with self.assertRaises(ValueError): paired_confidence(matrices, families, novel, seeds, arms, contrast)

    def test_a_good_mean_does_not_hide_a_failed_seed_or_value_tradeoff(self):
        def evaluate(policy, value):
            return decision(dict(pooled_contrasts=[dict(natural=dict(value_mse=dict(mean=value)),
                seed_variation=dict(natural=dict(policy_kl=dict(per_seed=policy))))]))
        self.assertTrue(evaluate([-.1, -.2, -.3], 0.)['provisional_candidate'])
        self.assertFalse(evaluate([-.1, -.2, .01], -.2)['provisional_candidate'])
        self.assertFalse(evaluate([-.1, -.2, -.3], .001)['provisional_candidate'])
        self.assertFalse(evaluate([-.1, -.2, 0.], -.2)['provisional_candidate'])

    def test_actual_factors_define_group_clipping(self):
        norms = dict.fromkeys(PARAMETERS, 0.)
        norms['bias'] = norms['input_gain'] = .8
        clipping = dict(mode='parameter-group', group_norms=norms,
                        factors=dict.fromkeys(PARAMETERS, 1.), clipped=False)
        rows = [dict(kind='train', step=s, clipping=clipping, gradient_norm=np.hypot(.8, .8),
                     clipped=False, clipping_fraction=0., seconds=.5) for s in (1, 10)]
        report = clipping_history(rows, 'parameter-group', 10, 1.)
        self.assertEqual(report['all_update_clipped_fraction'], 0.)
        self.assertEqual(report['groups']['bias']['logged_clipped_fraction'], 0.)
        corrupt = copy.deepcopy(rows); corrupt[-1]['clipping']['factors']['edge'] = .5
        with self.assertRaises(AssertionError): clipping_history(corrupt, 'parameter-group', 10, 1.)
        with self.assertRaises(ValueError): clipping_history(rows[:1], 'parameter-group', 10, 1.)

    def test_initial_pairing_covers_parameters_moments_ports_and_sampler(self):
        arrays = dict(metadata=np.array([1], np.uint8), optimizer_step=np.array(0, np.uint64),
                      **{'param/edge': np.array([1.], np.float32), 'first/edge': np.array([0.], np.float32),
                         'port/input_index': np.array([0, -1], np.int32)})
        metadata = dict(sampler={'seed': 10, 'offset': 0})
        reference = initial_state(arrays, metadata)
        tagged = dict(arrays, metadata=np.array([2], np.uint8), optimizer_clip_mode=np.array(1, np.uint8),
                      objective_value_core_scale=np.array(.1, np.float32))
        self.assertEqual(reference, initial_state(tagged, metadata))
        self.assertEqual(reference['arrays']['optimizer_step']['shape'], [])
        for key in ('param/edge', 'first/edge', 'port/input_index'):
            changed = dict(arrays); changed[key] = arrays[key] + 1
            self.assertNotEqual(reference, initial_state(changed, metadata))
        self.assertNotEqual(reference, initial_state(arrays, dict(sampler={'seed': 11, 'offset': 0})))


if __name__ == '__main__': unittest.main()
