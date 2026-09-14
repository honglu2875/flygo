"""A smooth neuron rule needs independent derivatives and explicit checkpoint identity."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from flygo.fly import FlyConfig,RustFly,firing_rate,PARAMETERS
from test_fly import fixture


class SmoothDynamics(unittest.TestCase):
    def test_nonlinearity_zero_and_extremes(self):
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:self.skipTest('JAX optional dependency')
        from flygo.jax.numerics import firing_rate as reference
        x=np.array([-.3,-.05,0,.05,.3],np.float32)
        for softness in (.01,.05):
            want=softness*np.logaddexp(0,x.astype(float)/softness)
            np.testing.assert_allclose(firing_rate(x,softness),want,rtol=2e-6,atol=1e-8)
            np.testing.assert_allclose(reference(jnp.asarray(x),softness),want,rtol=2e-6,atol=1e-8)
            derivative=jax.grad(lambda v:reference(v,softness).sum())
            np.testing.assert_allclose(derivative(jnp.asarray(x)),1/(1+np.exp(-x.astype(float)/softness)),rtol=2e-6,atol=1e-8)
            huge=np.array([-np.finfo(np.float32).max,np.finfo(np.float32).max],np.float32)
            np.testing.assert_array_equal(firing_rate(huge,softness),[0,huge[1]])
            np.testing.assert_array_equal(reference(jnp.asarray(huge),softness),[0,huge[1]])
        for bad in (-.1,float('inf'),float('nan'),1e-99):
            with self.assertRaises(ValueError):FlyConfig(rate_softness=bad)

    def test_all_derivatives_and_three_updates_with_negative_states(self):
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:self.skipTest('JAX optional dependency')
        from flygo.jax.model import forward,loss,adam
        jax.config.update('jax_enable_x64',True)
        for softness in (.01,.05):
            graph,cfg,ports,params,*batch=fixture();cfg=replace(cfg,rate_softness=softness)
            params['bias']=np.array([-.12,.08],np.float32)
            model=RustFly(graph,cfg,ports=ports,params=params)
            states=model.infer(batch[0],trace=True)['states']
            self.assertTrue(any(np.any(s<0) for s in states))
            self.assertTrue(any(np.any(s>0) for s in states))
            _,gradient=model.loss_and_grad(*batch)
            for name in PARAMETERS:
                for index in range(len(params[name])):
                    values=[]
                    for direction in (-1,1):
                        perturbed={k:v.copy() for k,v in params.items()}
                        perturbed[name][index]+=direction*.001
                        candidate=RustFly(graph,cfg,ports=ports,params=perturbed)
                        prediction=candidate.infer(batch[0])
                        # Differences in tiny derivatives can be smaller than
                        # one FP32 loss ULP. Evaluate the independent scalar
                        # objective in FP64, keeping the actual Rust forward.
                        logits=np.where(batch[1],prediction['logits'].astype(float),-1e30)
                        logits-=logits.max(axis=1,keepdims=True)
                        logp=logits-np.log(np.exp(logits).sum(axis=1,keepdims=True))
                        policy_loss=-(batch[2]*logp).sum(axis=1).mean()
                        value_loss=np.square(prediction['value'].astype(float)-batch[3]).mean()
                        values.append(policy_loss+value_loss)
                    self.assertAlmostEqual(float(gradient[name][index]),(values[1]-values[0])/.002,
                                           delta=5e-5,msg=f'{softness}/{name}/{index}')
            kwargs=dict(steps=cfg.steps,groups=cfg.groups,actions=cfg.actions,rate_softness=softness)
            derivative=jax.jit(jax.value_and_grad(lambda p:loss(p,graph,ports,*batch,**kwargs),has_aux=True))
            jp=jax.tree.map(jnp.asarray,params);first=jax.tree.map(jnp.zeros_like,jp);second=jax.tree.map(jnp.zeros_like,jp)
            for step in range(1,4):
                actual=model.infer(batch[0],trace=True);expected=forward(jp,graph,ports,batch[0],**kwargs)
                for key in actual:np.testing.assert_allclose(actual[key],expected[key],rtol=2e-5,atol=2e-7)
                (_,losses),gradient=derivative(jp)
                native_loss,native_gradient=model.loss_and_grad(*batch)
                np.testing.assert_allclose(list(native_loss.values()),losses,rtol=2e-5,atol=2e-7)
                for key in PARAMETERS:np.testing.assert_allclose(native_gradient[key],gradient[key],rtol=2e-4,atol=2e-7,err_msg=key)
                jp,first,second,_=adam(jp,gradient,first,second,step,rate=.003,clip=.01)
                model.train_step(*batch,rate=.003,clip=.01)
                saved=model.checkpoint_arrays()
                for prefix,group in [('param/',jp),('first/',first),('second/',second)]:
                    for key in PARAMETERS:np.testing.assert_allclose(saved[prefix+key],group[key],rtol=3e-4,atol=1e-6,err_msg=prefix+key)

    def test_distinct_smooth_checkpoint_and_legacy_baseline(self):
        from flygo.checkpoint import save_checkpoint,load_checkpoint
        from flygo.data.loader import Sampler
        graph,cfg,ports,params,*batch=fixture();graph['manifest']={'graph_id':'fixture'}
        smooth=RustFly(graph,replace(cfg,rate_softness=.05),ports=ports,params=params)
        smooth.train_step(*batch)
        with tempfile.TemporaryDirectory() as directory,patch('flygo.checkpoint.StorageBudget'):
            root=Path(directory);path=root/'smooth.npz'
            save_checkpoint(smooth,Sampler({},{}),path,dict(dataset_id='fixture'),root=root)
            restored=RustFly(graph,replace(cfg,rate_softness=.05),ports=ports,params=params)
            metadata=load_checkpoint(path,restored)
            self.assertEqual(metadata['model_version'],'leaky-rate-softplus-v1')
            smooth.train_step(*batch);restored.train_step(*batch)
            for key,value in smooth.checkpoint_arrays().items():
                self.assertEqual(value.tobytes(),restored.checkpoint_arrays()[key].tobytes())
            baseline=RustFly(graph,cfg,ports=ports,params=params)
            with self.assertRaisesRegex(ValueError,'Unsupported checkpoint model'):load_checkpoint(path,baseline)
            different=RustFly(graph,replace(cfg,rate_softness=.01),ports=ports,params=params)
            with self.assertRaisesRegex(ValueError,'numerical model configuration'):load_checkpoint(path,different)
            legacy=root/'legacy.npz'
            save_checkpoint(baseline,Sampler({},{}),legacy,dict(dataset_id='fixture'),root=root)
            with np.load(legacy) as data:arrays={k:data[k].copy() for k in data}
            info=json.loads(arrays['metadata'].tobytes());info['model_config'].pop('rate_softness');info.pop('model_version')
            arrays['metadata']=np.frombuffer(json.dumps(info).encode(),np.uint8);np.savez(legacy,**arrays)
            receipt=json.loads(legacy.with_suffix('.json').read_text());receipt['sha256']=hashlib.sha256(legacy.read_bytes()).hexdigest()
            legacy.with_suffix('.json').write_text(json.dumps(receipt))
            load_checkpoint(legacy,baseline)


if __name__=='__main__':unittest.main()
