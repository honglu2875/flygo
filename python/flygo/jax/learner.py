"""Data-parallel JAX implementation of the small RustFly model interface.

Every controller calls the same methods with the same globally indexed batch.
Only addressable examples are transferred; parameters and graph are replicated.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding,PartitionSpec as P
from jax.experimental import multihost_utils as mh
import numpy as np

from .model import forward,loss,adam
from .sparse import build_layout
from ..fly import FlyConfig,initialize


class JaxLearner:
    """Shared sharding, optimizer and checkpoint mechanics for a pure model."""
    numerical_runtime='jax-highest-fp32-norm-v1'

    def __init__(self,graph,config,*,ports,params,compute_graph,forward_function,loss_function,kwargs,mesh=None):
        self.graph,self.config=graph,config
        self.ports=ports
        self.mesh=mesh or jax.make_mesh((jax.device_count(),),('data',))
        self.replicated=NamedSharding(self.mesh,P())
        self.data=NamedSharding(self.mesh,P('data'))
        self._graph=self._put(compute_graph)
        self._ports=self._put(self.ports)
        self._params=self._put(params)
        self._first=jax.tree.map(jnp.zeros_like,self._params)
        self._second=jax.tree.map(jnp.zeros_like,self._params)
        self.step=0

        def mapped(fun,inputs,outputs):
            return jax.jit(jax.shard_map(fun,mesh=self.mesh,in_specs=inputs,
                                       out_specs=outputs,check_vma=False))
        def infer(p,g,a,x):
            result=forward_function(p,g,a,x,**kwargs)
            return dict(logits=result['logits'],value=result['value'])
        self._infer=mapped(infer,(P(),P(),P(),P('data')),dict(logits=P('data'),value=P('data')))
        self._trace=mapped(lambda p,g,a,x:forward_function(p,g,a,x,**kwargs),(P(),P(),P(),P('data')),
                           dict(logits=P('data'),value=P('data'),states=P(None,None,'data')))
        def derivative(p,g,a,b):
            (_,parts),gradient=jax.value_and_grad(loss_function,has_aux=True)(p,g,a,*b,**kwargs)
            return jax.lax.pmean(parts,'data'),jax.lax.pmean(gradient,'data')
        self._gradient=mapped(derivative,(P(),P(),P(),(P('data'),)*4),(P(),P()))
        def update(p,first,second,g,a,b,corrections,rate,clip):
            parts,gradient=derivative(p,g,a,b)
            p,first,second,norm=adam(p,gradient,first,second,0,rate=rate,clip=clip,
                                    corrections=corrections,norm_dtype=jnp.float32)
            return p,first,second,(*parts,norm)
        self._update=mapped(update,(P(),P(),P(),P(),P(),(P('data'),)*4,P(),P(),P()),(P(),P(),P(),P()))

    def _put(self,tree,sharding=None):
        sharding=self.replicated if sharding is None else sharding
        def put(array):
            array=np.asarray(array)
            return jax.make_array_from_callback(array.shape,sharding,lambda index:array[index])
        return jax.tree.map(put,tree)

    @staticmethod
    def _host(tree):
        return jax.tree.map(lambda a:np.asarray(a.addressable_shards[0].data).copy(),tree)

    @staticmethod
    def collective_any(value):
        return bool(np.any(mh.process_allgather(np.asarray(bool(value)))))

    @staticmethod
    def verify_checkpoint_copies(receipt):
        digest=np.frombuffer(bytes.fromhex(receipt['sha256']),np.uint8)
        hashes=np.asarray(mh.process_allgather(digest,tiled=False))
        if not np.all(hashes==hashes[0]):
            raise AssertionError('Replicated checkpoint contents differ between controllers')
        if jax.process_count()>1:
            return {**receipt,'replica_status':'verified',
                    'peer':f'{jax.process_count()} SPMD controllers; independent saves with identical SHA-256'}
        return receipt

    def _features(self,features):
        array=np.asarray(features,np.float32)
        if array.ndim<2 or int(np.prod(array.shape[1:]))!=self.config.features:
            raise ValueError('Incorrect feature shape')
        return array.reshape(len(array),self.config.features)

    def infer(self,features,*,trace=False):
        x=self._features(features);count=len(x)
        if not count:
            raise ValueError('Inference needs a nonempty batch')
        padding=(-count)%self.mesh.size
        if padding:x=np.pad(x,((0,padding),(0,0)))
        fun=self._trace if trace else self._infer
        result=jax.block_until_ready(fun(self._params,self._graph,self._ports,self._put(x,self.data)))
        result=mh.process_allgather(result,tiled=True)
        result['logits']=result['logits'][:count];result['value']=result['value'][:count]
        if trace:result['states']=[state[:,:count] for state in result['states']]
        return result

    def _batch(self,features,legal,policy,value):
        x=self._features(features);count=len(x)
        if count<1 or count%self.mesh.size:
            raise ValueError('Training batch must divide evenly across the data mesh')
        legal=np.asarray(legal,bool);policy=np.asarray(policy,np.float32);value=np.asarray(value,np.float32)
        if legal.shape!=(count,self.config.actions) or policy.shape!=legal.shape or value.shape!=(count,):
            raise ValueError('Incorrect target shapes')
        if (not np.isfinite(x).all() or not np.isfinite(policy).all() or not np.isfinite(value).all()
                or np.any(policy<0) or np.any(policy[~legal]!=0) or np.any(~legal.any(axis=1))
                or not np.allclose(policy.sum(axis=1),1,atol=1e-5) or np.any(np.abs(value)>1)):
            raise ValueError('Invalid training features or targets')
        return self._put((x,legal,policy,value),self.data)

    def loss_and_grad(self,*batch):
        parts,gradient=jax.block_until_ready(self._gradient(self._params,self._graph,self._ports,self._batch(*batch)))
        return dict(policy_loss=float(parts[0]),value_loss=float(parts[1])),self._host(gradient)

    def train_step(self,*batch,rate=.003,clip=1.0,rate_scales=None):
        if not np.isfinite(rate) or rate<=0 or not np.isfinite(clip) or clip<=0:
            raise ValueError('Finite positive learning rate and clip required')
        if rate_scales is not None:
            if set(rate_scales)-set(self._params) or any(not np.isfinite(v) or v<0 for v in rate_scales.values()):
                raise ValueError('Invalid parameter-group learning-rate multipliers')
            rate={k:np.float32(rate)*np.float32(rate_scales.get(k,1.0)) for k in self._params}
        else:rate=np.asarray(rate,np.float32)
        if any(not np.isfinite(v) for v in jax.tree.leaves(rate)):
            raise ValueError('Learning rate exceeds FP32 range')
        corrections=self._put(np.asarray([1-.9**(self.step+1),1-.999**(self.step+1)],np.float32))
        self._params,self._first,self._second,parts=jax.block_until_ready(self._update(
            self._params,self._first,self._second,self._graph,self._ports,self._batch(*batch),
            corrections,self._put(rate),self._put(np.asarray(clip,np.float32))))
        self.step+=1
        metrics=dict(policy_loss=float(parts[0]),value_loss=float(parts[1]),gradient_norm=float(parts[2]),step=self.step)
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError('Non-finite learner metrics')
        return metrics

    def parameters(self):
        return self._host(self._params)

    def checkpoint_arrays(self):
        return {**{prefix+k:v for prefix,group in [('param/',self._params),('first/',self._first),('second/',self._second)]
                   for k,v in self._host(group).items()},'optimizer_step':np.asarray(self.step,np.uint64)}

    def restore_arrays(self,arrays):
        step=np.asarray(arrays['optimizer_step'])
        if step.shape!=() or step.dtype.kind not in 'ui' or not 0<=int(step)<2**64:
            raise ValueError('Invalid optimizer step')
        groups=[]
        for prefix in ('param/','first/','second/'):
            group={name:np.asarray(arrays[prefix+name],np.float32) for name in self._params}
            if any(a.shape!=self._params[k].shape or not np.isfinite(a).all() for k,a in group.items()):
                raise ValueError('Invalid checkpoint parameter or moment arrays')
            if prefix=='second/' and any(np.any(a<0) for a in group.values()):
                raise ValueError('Negative second moment')
            groups.append(group)
        self._params,self._first,self._second=[self._put(group) for group in groups]
        self.step=int(step)


class JaxFly(JaxLearner):
    numerical_runtime='jax-highest-fp32-norm-buckets-v1'

    @property
    def model_version(self):
        return self.config.model_version

    def __init__(self,graph,config=FlyConfig(),*,ports=None,params=None,mesh=None):
        initial_ports,initial_params=initialize(graph,config)
        compute_graph={k:graph[k] for k in ('src','dst','type_id','sign')}
        compute_graph['layout']=build_layout(graph['src'],graph['dst'],len(graph['type_id']))
        super().__init__(graph,config,ports=initial_ports if ports is None else ports,
            params=initial_params if params is None else params,compute_graph=compute_graph,
            forward_function=forward,loss_function=loss,
            kwargs=dict(steps=config.steps,groups=config.groups,actions=config.actions,
                        rate_softness=config.rate_softness,readout_mean_scale=config.readout_mean_scale),mesh=mesh)
