"""Endpoint summaries cannot certify a changed mask or disabled optimizer state."""
import unittest
from scripts.attachment_endpoint import head_state
from flygo.fly import RustFly
from test_fly import fixture
from test_head_masks import mask_fixture


class ReadoutEndpoint(unittest.TestCase):
    def test_frozen_weights_and_moments_are_checked_after_training(self):
        graph,cfg,ports,params,*batch=fixture();mask=mask_fixture()
        model=RustFly(graph,cfg,ports=ports,params=params,head_mask=mask)
        initial=model.checkpoint_arrays();model.train_step(*batch);final=model.checkpoint_arrays()
        self.assertEqual(head_state(initial,final,mask.contract),mask.contract)
        for key in ('param/value_weight','first/value_weight','second/value_weight'):
            changed={k:v.copy() for k,v in final.items()};changed[key][0]+=1
            with self.assertRaises(ValueError):head_state(initial,changed,mask.contract)
        with self.assertRaisesRegex(ValueError,'head mask differs'):head_state(initial,final,None)

    def test_dense_legacy_endpoint_has_no_implicit_mask(self):
        graph,cfg,ports,params,*batch=fixture();model=RustFly(graph,cfg,ports=ports,params=params)
        initial=model.checkpoint_arrays();model.train_step(*batch)
        self.assertIsNone(head_state(initial,model.checkpoint_arrays(),None))
        with self.assertRaisesRegex(ValueError,'head mask differs'):
            head_state(initial,model.checkpoint_arrays(),mask_fixture().contract)


if __name__=='__main__':unittest.main()
