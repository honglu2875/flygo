"""Interchangeable learner updates, recovery and read-only diagnostics."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import unittest
import numpy as np

from test_fly import fixture
from flygo.fly import RustFly


class Learner(unittest.TestCase):
    def test_group_rate_scaling_and_parity(self):
        try:
            import jax
        except ImportError:
            self.skipTest('JAX optional dependency')
        from flygo.jax.learner import JaxFly
        graph,cfg,ports,params,*small=fixture()
        batch=[np.concatenate([a]*jax.device_count()) for a in small]
        ordinary=RustFly(graph,cfg,ports=ports,params=params)
        identity=RustFly(graph,cfg,ports=ports,params=params)
        ordinary.train_step(*batch)
        identity.train_step(*batch,rate_scales={})
        for key,want in ordinary.checkpoint_arrays().items():np.testing.assert_array_equal(identity.checkpoint_arrays()[key],want)
        rust=RustFly(graph,cfg,ports=ports,params=params)
        model=JaxFly(graph,cfg,ports=ports,params=params)
        for _ in range(3):
            scales=dict(bias=0,edge=.2,input_gain=2,policy_weight=.03904344047215152,
                        value_weight=.03904344047215152)
            rust.train_step(*batch,rate_scales=scales)
            model.train_step(*batch,rate_scales=scales)
            for key,want in rust.checkpoint_arrays().items():
                np.testing.assert_allclose(model.checkpoint_arrays()[key],want,rtol=3e-4,atol=2e-6,err_msg=key)
        np.testing.assert_array_equal(rust.parameters()['bias'],params['bias'])
        self.assertGreater(np.abs(rust.checkpoint_arrays()['first/bias']).sum(),0)

    def test_diagnostics_do_not_change_training_state(self):
        from flygo.diagnostics import measure
        graph,cfg,ports,params,*batch=fixture()
        model=RustFly(graph,cfg,ports=ports,params=params)
        before=model.parameters()
        metrics=model.train_step(*batch)
        checkpoint=model.checkpoint_arrays()
        result=measure(model,batch,before,metrics,clip=1)
        self.assertGreater(result['groups']['edge']['update_norm'],0)
        self.assertEqual(result['diagnostic_backward_positions'],len(batch[0]))
        for key,array in checkpoint.items():
            np.testing.assert_array_equal(model.checkpoint_arrays()[key],array)

    def test_sharded_updates_padding_and_restore(self):
        try:
            import jax
        except ImportError:
            self.skipTest('JAX optional dependency')
        from flygo.jax.learner import JaxFly
        graph,cfg,ports,params,*small=fixture()
        batch=[np.concatenate([a]*jax.device_count()) for a in small]
        rust=RustFly(graph,cfg,ports=ports,params=params)
        model=JaxFly(graph,cfg,ports=ports,params=params)
        initial=model.infer(batch[0],trace=True)
        for key,want in rust.infer(batch[0],trace=True).items():
            np.testing.assert_allclose(initial[key],want,rtol=3e-4,atol=3e-6)
        for _ in range(3):
            model.train_step(*batch,clip=.01)
            rust.train_step(*batch,clip=.01)
            for key,want in rust.checkpoint_arrays().items():
                np.testing.assert_allclose(model.checkpoint_arrays()[key],want,rtol=3e-4,atol=2e-6,err_msg=key)
        output=model.infer(batch[0])
        padded=model.infer(batch[0][:-1])
        np.testing.assert_allclose(padded['logits'],output['logits'][:-1],atol=1e-7)
        saved=model.checkpoint_arrays()
        model.train_step(*batch)
        expected=model.checkpoint_arrays()
        model.restore_arrays(saved)
        model.train_step(*batch)
        for key,want in expected.items():
            np.testing.assert_array_equal(model.checkpoint_arrays()[key],want)
        malformed={**saved,'second/edge':-np.ones_like(saved['second/edge'])}
        with self.assertRaisesRegex(ValueError,'Negative second moment'):
            model.restore_arrays(malformed)
        for key,want in expected.items():
            np.testing.assert_array_equal(model.checkpoint_arrays()[key],want)


if __name__=='__main__':unittest.main()
