"""Tiny-gradient scaling, explicit epsilon, and unchanged checkpoint semantics."""
from dataclasses import replace
import unittest
import numpy as np

from flygo.fly import RustFly
from flygo.optimizer import check_epsilon_resume,validate_epsilon
from test_fly import fixture


class AdamEpsilon(unittest.TestCase):
    def test_first_update_has_epsilon_outside_root(self):
        graph,cfg,ports,params,*batch=fixture()
        model=RustFly(graph,cfg,ports=ports,params=params)
        _,gradient=model.loss_and_grad(*batch)
        norm=np.sqrt(sum(np.square(g.astype(np.float64)).sum() for g in gradient.values()))
        self.assertLess(norm,1.)
        model.train_step(*batch,epsilon=1e-4)
        for name,value in model.parameters().items():
            g=gradient[name].astype(np.float64)
            expected=params[name]-.003*g/(np.abs(g)+1e-4)
            np.testing.assert_allclose(value,expected,rtol=2e-6,atol=2e-7,err_msg=name)

    def test_default_and_explicit_legacy_epsilon_are_identical(self):
        graph,cfg,ports,params,*batch=fixture()
        a=RustFly(graph,cfg,ports=ports,params=params)
        b=RustFly(graph,cfg,ports=ports,params=params)
        for _ in range(3):
            a.train_step(*batch)
            b.train_step(*batch,epsilon=1e-8)
        for key,value in a.checkpoint_arrays().items():
            np.testing.assert_array_equal(value,b.checkpoint_arrays()[key])

    def test_invalid_epsilon_is_atomic_and_resume_checks_legacy(self):
        graph,cfg,ports,params,*batch=fixture()
        model=RustFly(graph,cfg,ports=ports,params=params)
        before=model.checkpoint_arrays()
        for epsilon in (0.,-1.,np.nan,np.inf,1e-100,1e100):
            with self.assertRaisesRegex(ValueError,'epsilon'):
                model.train_step(*batch,epsilon=epsilon)
        for key,value in before.items():np.testing.assert_array_equal(value,model.checkpoint_arrays()[key])
        check_epsilon_resume(1e-8,{})
        check_epsilon_resume(1e-6,{'epsilon':float(np.float32(1e-6))})
        with self.assertRaisesRegex(ValueError,'differs'):check_epsilon_resume(1e-6,{})
        with self.assertRaisesRegex(ValueError,'differs'):check_epsilon_resume(1e-8,{'epsilon':1e-6})

    def test_scaled_smooth_rust_jax_updates_and_restore(self):
        try:import jax
        except ImportError:self.skipTest('JAX optional dependency')
        from flygo.jax.learner import JaxFly
        graph,cfg,ports,params,*small=fixture()
        batch=[np.concatenate([a]*jax.device_count()) for a in small]
        for softness in (0.,.01):
            for epsilon in (1e-6,1e-4):
                config=replace(cfg,rate_softness=softness)
                rust=RustFly(graph,config,ports=ports,params=params)
                model=JaxFly(graph,config,ports=ports,params=params)
                settings=dict(rate=.03,rate_scales={'bias':.01},clip=.01,epsilon=epsilon)
                for _ in range(3):
                    rust.train_step(*batch,**settings);model.train_step(*batch,**settings)
                    for key,value in rust.checkpoint_arrays().items():
                        np.testing.assert_allclose(model.checkpoint_arrays()[key],value,rtol=3e-4,atol=2e-6,err_msg=key)
                for learner in (rust,model):
                    saved=learner.checkpoint_arrays()
                    learner.train_step(*batch,**settings);expected=learner.checkpoint_arrays()
                    learner.restore_arrays(saved);learner.train_step(*batch,**settings)
                    for key,value in expected.items():np.testing.assert_array_equal(learner.checkpoint_arrays()[key],value)
                    for bad in (0.,-1.,np.nan,np.inf,1e-100,1e100):
                        with self.assertRaisesRegex(ValueError,'epsilon'):learner.train_step(*batch,epsilon=bad)


if __name__=='__main__':unittest.main()
