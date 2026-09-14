#!/usr/bin/env python3
"""Qualify full-circuit adapters and rate rules against an independent JAX CPU reference."""
from dataclasses import asdict
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
from flygo.optimizer import validate_epsilon
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB,StorageBudget


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--ports',type=Path,nargs='+',help='Omit for the original seeded random adapters')
    p.add_argument('--input-map',type=Path,help='Visual/context attachment; replaces --ports')
    p.add_argument('--input-modes',nargs='+',choices=('history','current','neutral'),default=['history','current','neutral'])
    p.add_argument('--passes',type=int,default=4)
    p.add_argument('--readout-mean-scale',type=float,default=1.0)
    p.add_argument('--batch-size',type=int,default=1,help='Bounded full-reference batch, 1..32')
    p.add_argument('--rate-softness',type=float,default=0.0)
    p.add_argument('--epsilon',type=float,default=1e-8)
    p.add_argument('--rate',type=float,default=.01)
    p.add_argument('--rate-scales',type=json.loads,default={})
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpus',default='117,118,119')
    args=p.parse_args();cpus=list(map(int,args.cpus.split(',')));pin(cpus)
    if args.input_map and args.ports:p.error('An input map replaces --ports')
    if not 1<=args.passes<=1024:p.error('Passes must be in 1..1024')
    if not 1<=args.batch_size<=32:p.error('Full CPU reference supports bounded batches 1..32')
    validate_epsilon(args.epsilon)
    if not np.isfinite(args.rate) or args.rate<=0:p.error('Rate must be finite and positive')
    os.environ['JAX_PLATFORMS']='cpu'
    import jax
    import jax.numpy as jnp
    from flygo.jax.model import forward,loss,adam
    jax.config.update('jax_enable_x64',True)
    args.output.resolve().relative_to(args.root.resolve())
    with StorageBudget(args.root).reserve(files=64*(1<<20),heap=max(16,12+2*args.batch_size)*GIB,purpose='full-graph port derivative qualification'):
        args.output.mkdir(parents=True,exist_ok=False);begin=time.time();records=[]
        root=args.root;graph=load_graph(Path(json.loads((root/'runs/m4/graph.json').read_text())['path']))
        manifest,arrays,indexes=load_release(root,root/'releases/v0-1m/manifest.json',cache=True)
        jgraph={key:jnp.asarray(graph[key]) for key in ('src','dst','type_id','sign')}
        adapter=None;runtime=None
        if args.input_map:
            from flygo.attachments import load_attachment,input_contract,runtime_hashes
            adapter,visual_ports,visual_receipt=load_attachment(args.input_map,
                graph_id=graph['manifest']['graph_id'],dataset_id=manifest['dataset_id'])
            runtime=runtime_hashes()
        variants=args.input_modes if adapter else args.ports or [None]
        for path in variants:
            visual_contract=input_contract(visual_receipt,path) if adapter else None
            receipt=(dict(visual_receipt,seed=1) if adapter else
                json.loads(path.with_suffix('.json').read_text()) if path else dict(seed=1,groups=656,sha256=None))
            cfg=FlyConfig(steps=args.passes,threads=len(cpus),seed=receipt['seed'],groups=receipt['groups'],
                features=adapter.features if adapter else 972,rate_softness=args.rate_softness,readout_mean_scale=args.readout_mean_scale)
            ports,params=initialize(graph,cfg)
            if adapter:ports=visual_ports
            elif path:ports,_=load_ports(path,graph_id=graph['manifest']['graph_id'],features=cfg.features,groups=cfg.groups,seed=cfg.seed)
            rust=RustFly(graph,cfg,ports=ports,params=params)
            sampler=Sampler(arrays,indexes,cfg.seed);jports=jax.tree.map(jnp.asarray,ports)
            jp=jax.tree.map(jnp.asarray,params);first=jax.tree.map(jnp.zeros_like,jp);second=jax.tree.map(jnp.zeros_like,jp)
            kwargs=dict(steps=cfg.steps,groups=cfg.groups,actions=cfg.actions,rate_softness=cfg.rate_softness,readout_mean_scale=cfg.readout_mean_scale)
            infer=jax.jit(lambda p,g,a,x:forward(p,g,a,x,**kwargs))
            derivative=jax.jit(jax.value_and_grad(lambda p,g,a,*batch:loss(p,g,a,*batch,**kwargs),has_aux=True))
            rates={k:np.float32(args.rate)*np.float32(args.rate_scales.get(k,1)) for k in params}
            update=jax.jit(lambda p,g,m,v,step:adam(p,g,m,v,step,rate=rates,clip=np.float32(1),norm_dtype=jnp.float64,epsilon=np.float32(args.epsilon)))
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
                batch=sampler.batch(args.batch_size)
                if adapter:batch=(adapter.encode(batch[0],path),*batch[1:])
                output=jax.block_until_ready(infer(jp,jgraph,jports,batch[0]));native=rust.infer(batch[0],trace=True)
                for key in native:check(native[key],output[key],f'{step}/forward/{key}',3e-4,3e-6)
                (_,losses),gradient=jax.block_until_ready(derivative(jp,jgraph,jports,*batch))
                native_loss,native_gradient=rust.loss_and_grad(*batch)
                for key,value in zip(('policy_loss','value_loss'),losses):
                    check(native_loss[key],value,f'{step}/loss/{key}',3e-4,3e-6)
                for key,value in native_gradient.items():check(value,gradient[key],f'{step}/gradient/{key}')
                jp,first,second,_=jax.block_until_ready(update(jp,gradient,first,second,step+1))
                rust.train_step(*batch,rate=args.rate,rate_scales=args.rate_scales or None,epsilon=args.epsilon)
                saved=rust.checkpoint_arrays()
                for prefix,group in [('param/',jp),('first/',first),('second/',second)]:
                    for key,value in group.items():check(saved[prefix+key],value,f'{step}/'+prefix+key)
                atomic_json(args.output/'status.json',dict(state='qualifying',port=str(path),completed_updates=step+1))
            records.append(dict(ports=str(args.input_map or path) if path else None,ports_sha256=receipt['sha256'],input_contract=visual_contract,model=asdict(cfg),readout_neurons=int((ports['output_group']>=0).sum()),
                                updates=3,batch_size=args.batch_size,optimizer=dict(rate=args.rate,rate_scales=args.rate_scales,epsilon=args.epsilon,clip=1),errors=errors))
            atomic_json(args.output/'result.json',dict(status='complete' if len(records)==len(variants) else 'running',
                graph_id=graph['manifest']['graph_id'],dataset_id=manifest['dataset_id'],records=records,cpus=cpus,
                script_sha256=sha256(Path(__file__)),runtime_sha256=runtime,seconds=time.time()-begin,
                scope='Full states, losses, every gradient and three free-running parameter/moment updates on real V0 inputs; independent E-by-B JAX CPU reference, no TPU use.'))
            del rust,jp,first,second,gradient,native_gradient,output,native,infer,derivative,update
            jax.clear_caches()
        atomic_json(args.output/'status.json',dict(state='complete',variants=len(records)))
        print(json.dumps(dict(status='passed',output=str(args.output),variants=len(records))))


if __name__=='__main__':main()
