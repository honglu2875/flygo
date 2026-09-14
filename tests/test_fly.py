"""Independent derivatives and restoration checks on a directed recurrent fixture."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')

import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np

from flygo.fly import FlyConfig,RustFly,PARAMETERS,initialize


def fixture():
    graph=dict(indptr=np.array([0,1,2,4,5],np.int32),src=np.array([2,0,1,3,2],np.int32),
               dst=np.array([0,1,2,2,3],np.int32),type_id=np.array([0,1,0,1],np.int32),
               sign=np.array([1,1,-1,1],np.float32),strength=np.array([-.1,.3,.2,.4,-.2],np.float32),
               sensory=np.array([0],np.int32))
    config=FlyConfig(steps=3,features=2,groups=2,actions=3,threads=2)
    ports,params=initialize(graph,config)
    params['policy_weight']*=10;params['value_weight']*=10
    x=np.array([[.3,.7],[.4,.2]],np.float32)
    legal=np.array([[1,0,1],[1,1,1]],np.uint8)
    policy=np.array([[.3,0,.7],[.5,.3,.2]],np.float32)
    value=np.array([.2,-.1],np.float32)
    return graph,config,ports,params,x,legal,policy,value


class FlyCore(unittest.TestCase):
    def test_invalid_parameter_restore_is_atomic_for_every_group(self):
        graph,cfg,ports,params,*batch=fixture()
        model=RustFly(graph,cfg,ports=ports,params=params)
        model.train_step(*batch)
        saved=model.checkpoint_arrays()
        expected=model.infer(batch[0])
        for name in PARAMETERS:
            for bad_value in (np.nan,np.inf,-np.inf):
                bad={k:v.copy() for k,v in saved.items()}
                bad['param/'+name][-1]=bad_value
                with self.assertRaisesRegex(ValueError,'Invalid model parameter'):
                    model.restore_arrays(bad)
            bad={k:v.copy() for k,v in saved.items()}
            bad['param/'+name]=bad['param/'+name][:-1]
            with self.assertRaisesRegex(ValueError,'Invalid model parameter'):
                model.restore_arrays(bad)
        for key,value in model.checkpoint_arrays().items():
            np.testing.assert_array_equal(value,saved[key])
        for key,value in model.infer(batch[0]).items():
            np.testing.assert_array_equal(value,expected[key])

    def test_checkpoint_survives_failed_replica(self):
        from flygo.checkpoint import save_checkpoint, load_checkpoint
        from flygo.data.loader import Sampler

        graph,config,ports,params,x,legal,policy,value=fixture()
        graph['manifest']={'graph_id':'fixture'}
        model=RustFly(graph,config,ports=ports,params=params)
        model.train_step(x,legal,policy,value)
        sampler=Sampler({}, {}, seed=19)
        expected_sampler=sampler.state()
        expected=model.checkpoint_arrays()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            path=root/'checkpoints/step-1.npz'
            # Admission has its own pressure tests; this tiny fixture should also
            # run on machines without the production fleet's RAM headroom.
            with patch('flygo.checkpoint.StorageBudget'), \
                    patch('flygo.checkpoint.replicate_file',side_effect=RuntimeError('offline peer')):
                receipt=save_checkpoint(model,sampler,path,{'dataset_id':'fixture-data'},root=root,peer='offline')
            self.assertEqual(receipt['replica_status'],'retry')
            self.assertEqual(json.loads(path.with_suffix('.json').read_text()),receipt)
            other=RustFly(graph,config,ports=ports,params=params)
            restored_sampler=Sampler({}, {}, seed=2)
            load_checkpoint(path,other,restored_sampler,dataset_id='fixture-data')
            self.assertEqual(restored_sampler.state(),expected_sampler)
            for key,array in other.checkpoint_arrays().items():
                np.testing.assert_array_equal(array,expected[key])
            with self.assertRaisesRegex(ValueError,'dataset differs'):
                load_checkpoint(path,other,dataset_id='different-data')
            with path.open('ab') as stream:
                stream.write(b'corrupted')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                load_checkpoint(path,other)

    def test_batch_reset_and_restore(self):
        graph,config,ports,params,x,legal,policy,value=fixture()
        model=RustFly(graph,config,ports=ports,params=params)
        output=model.infer(x,trace=True)
        for b in range(len(x)):
            one=model.infer(x[b:b+1],trace=True)
            np.testing.assert_allclose(one['logits'][0],output['logits'][b],atol=2e-7)
            for a,c in zip(one['states'],output['states']):
                np.testing.assert_array_equal(a[:,0],c[:,b])
        np.testing.assert_array_equal(model.infer(x)['logits'],output['logits'])
        model.train_step(x,legal,policy,value)
        after_first=model.infer(x)
        saved=model.checkpoint_arrays()
        expected=model.train_step(x,legal,policy,value)
        other=RustFly(graph,config,ports=ports,params=params)
        other.infer(x)  # Prime transforms before restore, exercising invalidation.
        other.restore_arrays(saved)
        np.testing.assert_array_equal(other.infer(x)['logits'],after_first['logits'])
        self.assertEqual(other.train_step(x,legal,policy,value),expected)
        np.testing.assert_array_equal(other.infer(x)['logits'],model.infer(x)['logits'])
        for key,array in other.checkpoint_arrays().items():
            np.testing.assert_array_equal(array,model.checkpoint_arrays()[key])

    def test_finite_differences(self):
        graph,config,ports,params,x,legal,policy,value=fixture()
        model=RustFly(graph,config,ports=ports,params=params)
        _,grad=model.loss_and_grad(x,legal,policy,value)
        for name in PARAMETERS:
            for index in range(len(params[name])):
                losses=[]
                for sign in (-1,1):
                    perturbed={k:v.copy() for k,v in params.items()}
                    perturbed[name][index]+=sign*0.003
                    candidate=RustFly(graph,config,ports=ports,params=perturbed)
                    result,_=candidate.loss_and_grad(x,legal,policy,value)
                    losses.append(result['policy_loss']+result['value_loss'])
                numerical=(losses[1]-losses[0])/0.006
                self.assertAlmostEqual(float(grad[name][index]),numerical,delta=2e-5,msg=f'{name}[{index}]')

    def test_invalid_graph_and_targets(self):
        graph,config,ports,params,x,legal,policy,value=fixture()
        with self.assertRaises(ValueError):
            RustFly({**graph,'src':np.array([9,0,1,3,2],np.int32)},config)
        overlap={**ports,'output_group':ports['output_group'].copy()}
        overlap['output_group'][0]=0
        with self.assertRaises(ValueError):
            RustFly(graph,config,ports=overlap)
        model=RustFly(graph,config)
        policy[0,1]=.1
        with self.assertRaises(ValueError):
            model.loss_and_grad(x,legal,policy,value)

    def test_jax_all_groups_and_three_adam_updates(self):
        try:
            import jax
        except ImportError:
            self.skipTest('Install the jax extra for independent parity')
        from flygo.jax.model import forward,loss,adam
        jax.config.update('jax_enable_x64',True)
        graph,config,ports,params,x,legal,policy,value=fixture()
        model=RustFly(graph,config,ports=ports,params=params)
        kwargs=dict(steps=config.steps,groups=config.groups,actions=config.actions)
        reference=forward(params,graph,ports,x,**kwargs)
        actual=model.infer(x,trace=True)
        for name in ('logits','value','states'):
            np.testing.assert_allclose(actual[name],np.asarray(reference[name]),rtol=2e-5,atol=2e-7)
        derivative=jax.jit(jax.value_and_grad(lambda p:loss(p,graph,ports,x,legal.astype(bool),policy,value,**kwargs),has_aux=True))
        first={k:np.zeros_like(v) for k,v in params.items()};second={k:np.zeros_like(v) for k,v in params.items()}
        for step in range(1,4):
            (_,losses),grad=derivative(params)
            actual_loss,actual_grad=model.loss_and_grad(x,legal,policy,value)
            np.testing.assert_allclose([actual_loss['policy_loss'],actual_loss['value_loss']],losses,atol=2e-7)
            for name in PARAMETERS:
                np.testing.assert_allclose(actual_grad[name],grad[name],rtol=2e-4,atol=2e-7,err_msg=name)
            params,first,second,norm=adam(params,grad,first,second,step,rate=.003,clip=.01)
            result=model.train_step(x,legal,policy,value,rate=.003,clip=.01)
            self.assertAlmostEqual(result['gradient_norm'],float(norm),delta=2e-7)
            state=model.checkpoint_arrays()
            for prefix,expected in [('param/',params),('first/',first),('second/',second)]:
                for name in PARAMETERS:
                    np.testing.assert_allclose(state[prefix+name],expected[name],rtol=3e-4,atol=1e-6,err_msg=prefix+name)


if __name__=='__main__':
    unittest.main()
