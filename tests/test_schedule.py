"""Scheduled updates must agree across backends and survive a restart boundary."""
import unittest
import numpy as np

from flygo.schedule import Schedule
from flygo.fly import RustFly
from test_fly import fixture


class LearningRates(unittest.TestCase):
    def test_boundaries_and_legacy_constant(self):
        fixed=Schedule(.003)
        for update in (1,100,10000):self.assertEqual(fixed.rate(update),.003)
        fixed.check_resume({})
        schedule=Schedule(.003,warmup_steps=4,decay_until=12)
        for update,expected in ((1,.00075),(4,.003),(8,.00165),(12,.0003),(100,.0003)):
            self.assertAlmostEqual(schedule.rate(update),expected,places=15)
        for options in (dict(peak=0),dict(peak=float('nan')),dict(peak=.1,decay_until=1),
                        dict(peak=.1,warmup_steps=10,decay_until=10),dict(peak=.1,final_ratio=0)):
            with self.assertRaises(ValueError):Schedule(**options)
        with self.assertRaises(ValueError):schedule.rate(0)
        with self.assertRaisesRegex(ValueError,'no schedule'):schedule.check_resume({})
        schedule.check_resume({'schedule':schedule.contract()})
        with self.assertRaisesRegex(ValueError,'differs'):
            fixed.check_resume({'schedule':schedule.contract()})

    def test_resume_crosses_warmup_and_decay_end_bitwise(self):
        graph,cfg,ports,params,*batch=fixture()
        schedule=Schedule(.003,warmup_steps=3,decay_until=6)
        original=RustFly(graph,cfg,ports=ports,params=params)
        for update in (1,2):original.train_step(*batch,rate=schedule.rate(update))
        restored=RustFly(graph,cfg,ports=ports,params=params)
        restored.restore_arrays(original.checkpoint_arrays())
        for update in range(3,9):
            original.train_step(*batch,rate=schedule.rate(update))
            step=int(restored.checkpoint_arrays()['optimizer_step'])+1
            restored.train_step(*batch,rate=schedule.rate(step))
            for key,array in original.checkpoint_arrays().items():
                self.assertEqual(array.tobytes(),restored.checkpoint_arrays()[key].tobytes(),key)

    def test_rust_jax_scheduled_scaled_updates(self):
        try:import jax
        except ImportError:self.skipTest('JAX optional dependency')
        from flygo.jax.learner import JaxFly
        graph,cfg,ports,params,*small=fixture()
        batch=[np.concatenate([a]*jax.device_count()) for a in small]
        rust=RustFly(graph,cfg,ports=ports,params=params)
        model=JaxFly(graph,cfg,ports=ports,params=params)
        schedule=Schedule(.003,warmup_steps=2,decay_until=4)
        for update in range(1,6):
            rate=schedule.rate(update)
            for learner in (rust,model):learner.train_step(*batch,rate=rate,rate_scales={'bias':.01})
            for key,array in rust.checkpoint_arrays().items():
                np.testing.assert_allclose(model.checkpoint_arrays()[key],array,rtol=3e-4,atol=2e-6,err_msg=key)


if __name__=='__main__':unittest.main()
