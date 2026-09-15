"""Policy influence must be invariant to common logits and allow cancellation."""
import unittest
import numpy as np
from scripts.probe_readout_contributions import energy, policy_change, relative_logits


class ReadoutContributions(unittest.TestCase):
    def test_common_shift_and_illegal_actions_do_not_change_policy(self):
        scores=np.array([[.2,-.5,.7],[-.3,.6,.1]],np.float64)
        legal=np.array([[True,False,True],[False,True,True]])
        shifted=scores+np.array([[123.4],[-75.3]])
        shifted[~legal]=1e9
        np.testing.assert_allclose(relative_logits(scores,legal),relative_logits(shifted,legal),atol=1e-13)
        change=policy_change(scores,shifted,legal)
        self.assertAlmostEqual(change['mean_kl'],0,places=13)
        self.assertEqual(change['top1_change_fraction'],0)

    def test_cancellation_can_make_residual_energy_exceed_total(self):
        total=np.array([[1.,-1.],[2.,-2.]])
        one_motor=3*total
        legal=np.ones_like(total,dtype=bool)
        self.assertEqual(energy(total-one_motor,legal)/energy(total,legal),4.)


if __name__=='__main__':unittest.main()
