#!/usr/bin/env python3
"""Qualify full-circuit port variants against an independent JAX CPU reference."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release,Sampler
from flygo.fly import FlyConfig,RustFly,initialize,load_graph
from flygo.ports import load_ports
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB,StorageBudget


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--ports',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpus',default='117,118,119')
    args=p.parse_args();cpus=list(map(int,args.cpus.split(',')));pin(cpus)
    os.environ['JAX_PLATFORMS']='cpu'
    import jax
    import jax.numpy as jnp
    from flygo.jax.model import forward,loss,adam
    jax.config.update('jax_enable_x64',True)
    args.output.resolve().relative_to(args.root.resolve())
    with StorageBudget(args.root).reserve(files=64*(1<<20),heap=16*GIB,purpose='full-graph port derivative qualification'):
        args.output.mkdir(parents=True,exist_ok=False);begin=time.time();records=[]
        root=args.root;graph=load_graph(Path(json.loads((root/'runs/m4/graph.json').read_text())['path']))
        manifest,arrays,indexes=load_release(root,root/'releases/v0-1m/manifest.json',cache=True)
        jgraph={key:jnp.asarray(graph[key]) for key in ('src','dst','type_id','sign')}
        for path in args.ports:
            receipt=json.loads(path.with_suffix('.json').read_text())
            cfg=FlyConfig(steps=4,threads=len(cpus),seed=receipt['seed'],groups=receipt['groups'])
            ports,_=load_ports(path,graph_id=graph['manifest']['graph_id'],features=cfg.features,groups=cfg.groups,seed=cfg.seed)
            _,params=initialize(graph,cfg);rust=RustFly(graph,cfg,ports=ports,params=params)
            sampler=Sampler(arrays,indexes,cfg.seed);jports=jax.tree.map(jnp.asarray,ports)
            jp=jax.tree.map(jnp.asarray,params);first=jax.tree.map(jnp.zeros_like,jp);second=jax.tree.map(jnp.zeros_like,jp)
            kwargs=dict(steps=cfg.steps,groups=cfg.groups,actions=cfg.actions)
            infer=jax.jit(lambda p,g,a,x:forward(p,g,a,x,**kwargs))
            derivative=jax.jit(jax.value_and_grad(lambda p,g,a,*batch:loss(p,g,a,*batch,**kwargs),has_aux=True))
            update=jax.jit(lambda p,g,m,v,step:adam(p,g,m,v,step,rate=np.float32(.01),clip=np.float32(1),norm_dtype=jnp.float64))
            errors={}
            def check(actual,expected,key,rtol=3e-3,atol=5e-6):
                a,b=np.asarray(actual),np.asarray(expected)
                try:
                    if not np.isfinite(a).all() or not np.isfinite(b).all():raise AssertionError('Nonfinite comparison: '+key)
                    np.testing.assert_allclose(a,b,rtol=rtol,atol=atol,err_msg=key)
                except AssertionError as error:
                    atomic_json(args.output/'status.json',dict(state='failed',port=str(path),stage=key,error=str(error)))
                    raise
                errors[key]=float(np.max(np.abs(a-b)))
            for step in range(3):
                batch=sampler.batch(1)
                output=jax.block_until_ready(infer(jp,jgraph,jports,batch[0]));native=rust.infer(batch[0],trace=True)
                for key in native:check(native[key],output[key],f'{step}/forward/{key}',3e-4,3e-6)
                (_,losses),gradient=jax.block_until_ready(derivative(jp,jgraph,jports,*batch))
                native_loss,native_gradient=rust.loss_and_grad(*batch)
                for key,value in zip(('policy_loss','value_loss'),losses):
                    check(native_loss[key],value,f'{step}/loss/{key}',3e-4,3e-6)
                for key,value in native_gradient.items():check(value,gradient[key],f'{step}/gradient/{key}')
                jp,first,second,_=jax.block_until_ready(update(jp,gradient,first,second,step+1))
                rust.train_step(*batch,rate=.01)
                saved=rust.checkpoint_arrays()
                for prefix,group in [('param/',jp),('first/',first),('second/',second)]:
                    for key,value in group.items():check(saved[prefix+key],value,f'{step}/'+prefix+key)
                atomic_json(args.output/'status.json',dict(state='qualifying',port=str(path),completed_updates=step+1))
            records.append(dict(ports=str(path),ports_sha256=receipt['sha256'],readout_neurons=int((ports['output_group']>=0).sum()),
                                updates=3,batch_size=1,errors=errors))
            atomic_json(args.output/'result.json',dict(status='complete' if len(records)==len(args.ports) else 'running',
                graph_id=graph['manifest']['graph_id'],dataset_id=manifest['dataset_id'],records=records,cpus=cpus,
                script_sha256=sha256(Path(__file__)),seconds=time.time()-begin,
                scope='Full states, losses, every gradient and three free-running parameter/moment updates on real V0 inputs; independent E-by-B JAX CPU reference, no TPU use.'))
            del rust,jp,first,second,gradient,native_gradient,output,native,infer,derivative,update
            jax.clear_caches()
        atomic_json(args.output/'status.json',dict(state='complete',variants=len(records)))
        print(json.dumps(dict(status='passed',output=str(args.output),variants=len(records))))


if __name__=='__main__':main()
