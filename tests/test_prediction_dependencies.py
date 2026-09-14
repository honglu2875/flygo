"""Readout pruning must preserve prediction bytes across depth, updates and restoration."""
from dataclasses import replace
import unittest
import numpy as np

from flygo.fly import FlyConfig,RustFly,initialize


def fixture():
    # A sensory chain/cycle feeding two readouts plus a disconnected recurrent pair.
    graph=dict(indptr=np.array([0,0,2,3,4,5,6,7],np.int32),
        src=np.array([0,2,1,2,1,6,5],np.int32),
        dst=np.array([1,1,2,3,4,5,6],np.int32),
        type_id=np.array([0,1,0,1,0,1,0],np.int32),
        sign=np.array([1,-1,1,1,1,-1,1],np.float32),
        strength=np.array([.4,.2,-.1,.2,-.1,.3,-.2],np.float32),
        sensory=np.array([0,5],np.int32))
    config=FlyConfig(steps=4,features=2,groups=2,actions=3,threads=2)
    _,params=initialize(graph,config)
    ports=dict(input_index=np.array([0,-1,-1,-1,-1,1,-1],np.int32),
        output_group=np.array([-1,-1,-1,0,1,-1,-1],np.int32),
        output_scale=np.array([0,0,0,1,1,0,0],np.float32))
    x=np.random.default_rng(87219).normal(size=(32,2)).astype(np.float32)
    legal=np.ones((32,3),np.uint8);policy=np.tile([.2,.3,.5],(32,1)).astype(np.float32)
    value=np.linspace(-.5,.5,32,dtype=np.float32)
    return graph,config,ports,params,(x,legal,policy,value)


class PredictionDependencies(unittest.TestCase):
    def test_exact_predictions_across_depth_batch_dynamics_update_and_restore(self):
        graph,config,ports,params,batch=fixture()
        for softness in (0.,.01):
            for conditioning in (1.,.3):
                model=RustFly(graph,replace(config,rate_softness=softness,readout_mean_scale=conditioning),
                              ports=ports,params=params)
                saved=model.checkpoint_arrays()
                for phase in range(3):
                    # Changing depth on one model also exercises dependency-cache replacement.
                    for depth in (1,8,2,32,4,1):
                        model.config=replace(model.config,steps=depth)
                        dependencies=model.prediction_dependencies()
                        self.assertEqual(len(dependencies),depth)
                        self.assertEqual(dependencies[-1],dict(neurons=2,edges=2))
                        self.assertTrue(all(row['neurons']<=5 for row in dependencies))
                        for size in (1,3,32):
                            x=batch[0][:size]
                            # Alternate which execution prepares the parameter cache first.
                            pruned=model.infer(x,prune=True)
                            full=model.infer(x)
                            traced=model.infer(x,trace=True)
                            for name,value in full.items():
                                self.assertEqual(value.tobytes(),pruned[name].tobytes())
                                self.assertEqual(value.tobytes(),traced[name].tobytes())
                    if phase==0:model.train_step(*batch,rate=.03,epsilon=1e-6)
                    if phase==1:model.restore_arrays(saved)
                for name,value in saved.items():
                    self.assertEqual(value.tobytes(),model.checkpoint_arrays()[name].tobytes())

    def test_pruning_rejects_full_traces_invalid_inputs_and_depths(self):
        graph,config,ports,params,batch=fixture()
        model=RustFly(graph,config,ports=ports,params=params)
        with self.assertRaisesRegex(ValueError,'full-state trace'):
            model.infer(batch[0],trace=True,prune=True)
        for bad in (batch[0][:0],np.full_like(batch[0],np.nan),np.full_like(batch[0],np.inf)):
            with self.assertRaisesRegex(ValueError,'finite feature-major'):
                model.infer(bad,prune=True)
        for depth in (0,1025):
            model.config=replace(model.config,steps=depth)
            with self.assertRaisesRegex(ValueError,'step count'):model.infer(batch[0],prune=True)
            with self.assertRaisesRegex(ValueError,'step count'):model.prediction_dependencies()


if __name__=='__main__':unittest.main()
