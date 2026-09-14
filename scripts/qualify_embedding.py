#!/usr/bin/env python3
"""Full-CNS spherical embedding states, objective gradients and Adam qualification."""
import argparse
from dataclasses import asdict
import json,os,time
from pathlib import Path
import numpy as np
import flygo
from flygo import _native
from flygo.contrastive import BranchObjective
from flygo.data.branches import load_bank,quartet_views
from flygo.data.corpus import atomic_json
from flygo.fly import FlyConfig,RustFly,initialize,load_graph,firing_rate
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB,StorageBudget


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epsilon',type=float,default=1e-8)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--cpus',default=','.join(map(str,range(92,116))))
    args=p.parse_args();cpus=list(map(int,args.cpus.split(',')));pin(cpus)
    from flygo.optimizer import validate_epsilon
    validate_epsilon(args.epsilon)
    if not 4<=args.batch_size<=32 or args.batch_size%4:p.error('Use 1–8 complete quartets, batch 4..32')
    args.output.resolve().relative_to(args.root.resolve())
    os.environ['JAX_PLATFORMS']='cpu'
    import jax
    import jax.numpy as jnp
    from flygo.jax.model import forward,adam
    jax.config.update('jax_enable_x64',True)
    with StorageBudget(args.root).reserve(files=64<<20,heap=max(28,12+2*args.batch_size)*GIB,purpose='full-CNS embedding CPU qualification'):
        args.output.mkdir(parents=True,exist_ok=False);started=time.time();errors={}
        report,arrays,pairs,renderer,ports,geometry=load_bank(args.bank)
        graph=load_graph(Path(json.loads((args.root/'runs/m4/graph.json').read_text())['path']))
        if graph['manifest']['graph_id']!=report['graph_id']:raise ValueError('Graph differs from bank')
        cfg=FlyConfig(steps=8,features=len(geometry['sensors']),groups=len(geometry['motors']),actions=1,threads=len(cpus))
        _,params=initialize(graph,cfg);rust=RustFly(graph,cfg,ports=ports,params=params)
        objective=BranchObjective();rng=np.random.default_rng(918424)
        jg={k:jnp.asarray(graph[k]) for k in ('src','dst','type_id','sign')}
        ja=jax.tree.map(jnp.asarray,ports);jp=jax.tree.map(jnp.asarray,params)
        first=jax.tree.map(jnp.zeros_like,jp);second=jax.tree.map(jnp.zeros_like,jp)
        kwargs=dict(steps=cfg.steps,groups=cfg.groups,actions=cfg.actions,return_embedding=True)
        infer=jax.jit(lambda p,g,a,x:forward(p,g,a,x,**kwargs))
        def loss(p,g,a,x):
            out=forward(p,g,a,x,**kwargs)
            # Match the declared FP64 small objective, retaining FP32 recurrence.
            return objective.jax_loss(out['embedding'].astype(jnp.float64),out['score'].astype(jnp.float64))
        derivative=jax.jit(jax.value_and_grad(loss))
        scales=dict(bias=.01,readout_gain=0.,policy_weight=0.,policy_bias=0.,value_bias=0.)
        rates={k:np.float32(.01)*np.float32(scales.get(k,1.)) for k in params}
        update=jax.jit(lambda p,g,m,v,s:adam(p,g,m,v,s,rate=rates,clip=np.float32(1.),norm_dtype=jnp.float64,
                                           epsilon=np.float32(args.epsilon)))
        receipt=dict(status='running',graph_id=report['graph_id'],bank_sha256=sha256(args.bank/'result.json'),
            model=asdict(cfg),objective=asdict(objective),native_sha256=sha256(Path(_native.__file__)),
            python_sha256={name:sha256(Path(flygo.__file__).parent/name) for name in
                           ('fly.py','vision.py','contrastive.py','data/branches.py','jax/model.py')},
            script_sha256=sha256(Path(__file__)),cpus=cpus,errors=errors)
        receipt['optimizer']=dict(rate=.01,clip=1.,epsilon=args.epsilon,rate_scales=scales)
        receipt['batch_size']=args.batch_size
        def check(a,b,key,rtol=3e-3,atol=5e-6):
            a,b=np.asarray(a),np.asarray(b)
            if not np.isfinite(a).all() or not np.isfinite(b).all():raise AssertionError('Nonfinite '+key)
            np.testing.assert_allclose(a,b,rtol=rtol,atol=atol,err_msg=key)
            errors[key]=float(np.max(np.abs(a-b)))
        try:
            choices=np.flatnonzero(arrays['split']=='train')
            for step in range(3):
                count=args.batch_size//4
                x=quartet_views(renderer,arrays['features'][choices[step*count:(step+1)*count]],rng)
                result=jax.block_until_ready(infer(jp,jg,ja,x));actual=rust.infer(x,trace=True)
                check(actual['states'],result['states'],f'{step}/states',3e-4,3e-6)
                check(rust.embedding(x),result['embedding'],f'{step}/embedding',3e-4,3e-6)
                np.testing.assert_array_equal(rust.embedding(x),firing_rate(actual['states'][-1])[geometry['motors']].T)
                jl,jgrad=jax.block_until_ready(derivative(jp,jg,ja,x))
                native_loss,native_grad=rust.embedding_loss_and_grad(x,objective)
                check(native_loss['loss'],jl,f'{step}/loss',3e-4,3e-6)
                for key,g in native_grad.items():check(g,jgrad[key],f'{step}/gradient/{key}')
                jp,first,second,_=jax.block_until_ready(update(jp,jgrad,first,second,step+1))
                rust.train_embedding(x,objective,rate=.01,rate_scales=scales,epsilon=args.epsilon)
                saved=rust.checkpoint_arrays()
                for prefix,values in [('param/',jp),('first/',first),('second/',second)]:
                    for k,v in values.items():check(saved[prefix+k],v,f'{step}/'+prefix+k)
                receipt['completed_updates']=step+1
                atomic_json(args.output/'result.json',receipt)
                print(json.dumps(dict(completed_updates=step+1,seconds=time.time()-started)),flush=True)
            receipt.update(status='passed',scope=f'Full CPU reference, B{args.batch_size}/K8, every state/gradient and three free-running Adam updates. No TPU claim.')
        except BaseException as e:
            receipt.update(status='failed',error=repr(e));raise
        finally:
            receipt['seconds']=time.time()-started;atomic_json(args.output/'result.json',receipt)


if __name__=='__main__':main()
