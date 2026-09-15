"""Policy influence must be invariant to common logits and allow cancellation."""
import unittest
import numpy as np
from scripts.probe_readout_contributions import energy, policy_change, relative_logits, sequential_decode


class ReadoutContributions(unittest.TestCase):
    def test_fp64_reconstruction_preserves_small_cancelling_signals_and_applies_masks(self):
        from flygo.readout import HeadMask
        raw=np.ones((1,3),np.float32);scale=np.ones(3,np.float32)
        policy=np.array([[1e8,1,-1e8]],np.float32);bias=np.zeros(1,np.float32)
        value=np.array([1e8,1,-1e8],np.float32)
        old=sequential_decode(raw,scale,policy,bias,value,bias)
        precise=sequential_decode(raw,scale,policy,bias,value,bias,accumulator_dtype=np.float64)
        self.assertEqual(float(old[0][0,0]),0)
        self.assertEqual(float(precise[0][0,0]),1)
        self.assertAlmostEqual(float(precise[1][0]),float(np.tanh(np.float32(1))),places=7)
        mask=HeadMask(np.array([[1,0,1]],np.uint8),np.array([0,1,0],np.uint8),'{"kind":"fixture"}')
        restricted=sequential_decode(raw,scale,policy,bias,value,bias,head_mask=mask,accumulator_dtype=np.float64)
        self.assertEqual(float(restricted[0][0,0]),0)
        np.testing.assert_array_equal(restricted[1],precise[1])

    def test_reconstruction_matches_actual_masked_native_predictions_after_learning(self):
        from flygo.fly import RustFly
        from test_fly import fixture
        from test_head_masks import mask_fixture
        graph,cfg,ports,params,*batch=fixture();mask=mask_fixture();motors=np.array([1,2])
        ports['output_group'][:]=-1;ports['output_group'][motors]=[0,1]
        model=RustFly(graph,cfg,ports=ports,params=params,head_mask=mask)
        model.train_step(*batch)
        output=model.infer(batch[0],trace=True);params=model.parameters()
        raw=np.maximum(output['states'][-1][motors].T,0)
        scale=ports['output_scale'][motors]*params['readout_gain'][motors]
        logits,value=sequential_decode(raw,scale,params['policy_weight'].reshape(cfg.actions,cfg.groups),params['policy_bias'],
            params['value_weight'],params['value_bias'],head_mask=mask,accumulator_dtype=np.float64)
        np.testing.assert_array_equal(logits,output['logits'])
        np.testing.assert_allclose(value,output['value'],rtol=0,atol=1.2e-7)

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
