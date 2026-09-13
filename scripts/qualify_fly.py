#!/usr/bin/env python3
"""Full fixed-graph CPU parity and measured inference/training costs."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time

os.environ['JAX_PLATFORMS']='cpu'
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np

from flygo.data.corpus import atomic_json
from flygo.fly import FlyConfig,RustFly,load_graph,initialize,PARAMETERS
from flygo.runtime import cpu_profile,pin
from flygo.storage import GIB,StorageBudget


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    parser.add_argument('--threads',type=int,default=24)
    args=parser.parse_args()
    cpus=cpu_profile()['research_cpus'][:args.threads]
    pin(cpus)
    root=args.root
    out=root/'runs/m4'
    graph_path=Path(json.loads((out/'graph.json').read_text())['path'])
    with StorageBudget(root).reserve(files=256*1024**2,heap=32*GIB,purpose='full-graph Rust/JAX CPU parity and benchmark'):
        graph=load_graph(graph_path)
        config=FlyConfig(steps=2,threads=args.threads)
        ports,params=initialize(graph,config)
        model=RustFly(graph,config,ports=ports,params=params)
        rng=np.random.default_rng(18)
        x=(rng.random((1,972))<.12).astype(np.float32)
        legal=np.ones((1,82),bool);legal[:,2::7]=False
        policy=legal.astype(np.float32);policy/=policy.sum(axis=-1,keepdims=True)
        value=np.array([.2],np.float32)
        import jax
        from flygo.jax.model import forward,loss
        jax.config.update('jax_enable_x64',True)
        jgraph={key:jax.numpy.asarray(graph[key]) for key in ('src','dst','type_id','sign')}
        jports={key:jax.numpy.asarray(array) for key,array in ports.items()}
        jparams={key:jax.numpy.asarray(array) for key,array in params.items()}
        jf=jax.jit(forward,static_argnames=('steps','groups','actions'))
        started=time.time()
        expected=jax.block_until_ready(jf(jparams,jgraph,jports,x,steps=2,groups=config.groups,actions=config.actions))
        actual=model.infer(x,trace=True)
        errors={}
        for name in ('logits','value','states'):
            a,b=np.asarray(actual[name]),np.asarray(expected[name])
            np.testing.assert_allclose(a,b,rtol=3e-4,atol=3e-6,err_msg=name)
            errors[name]=float(np.max(np.abs(a-b)))
        print('Full graph forward parity',errors,flush=True)
        def objective(p,g,ports,x,legal,policy,value):
            return loss(p,g,ports,x,legal,policy,value,steps=2,groups=config.groups,actions=config.actions)
        jl=jax.jit(jax.value_and_grad(objective,has_aux=True))
        (_,losses),gradient=jax.block_until_ready(jl(jparams,jgraph,jports,x,legal,policy,value))
        rloss,rgrad=model.loss_and_grad(x,legal,policy,value)
        gradient_errors={}
        for name in PARAMETERS:
            expected_grad=np.asarray(gradient[name]);actual_grad=rgrad[name]
            np.testing.assert_allclose(actual_grad,expected_grad,rtol=3e-3,atol=5e-6,err_msg=name)
            gradient_errors[name]=dict(max_absolute_error=float(np.max(np.abs(actual_grad-expected_grad))),
                nonzero=int(np.count_nonzero(actual_grad)),size=len(actual_grad))
        atomic_json(out/'parity.json',dict(status='passed',graph_id=graph['manifest']['graph_id'],neurons=len(graph['type_id']),
            edges=len(graph['src']),steps=2,batch=1,cpus=cpus,forward=errors,gradients=gradient_errors,
            rust_loss=rloss,jax_loss=[float(v) for v in losses],elapsed_seconds=time.time()-started,
            caveat='JAX reference materializes E*B messages; full parity uses B=1; no TPU execution'))
        del gradient,rgrad,expected,actual,jparams,jgraph,jports,jf,jl
        jax.clear_caches()
        benchmarks=[]
        for steps in (2,8):
            # Reuse owned parameters and graph across batches, excluding setup from timings.
            model.config=replace(config,steps=steps)
            for batch in (1,8,32,128):
                features=(rng.random((batch,972))<.12).astype(np.float32)
                model.infer(features)
                samples=[]
                for _ in range(3):
                    start=time.perf_counter();model.infer(features);samples.append(time.perf_counter()-start)
                record=dict(kind='forward',steps=steps,batch=batch,seconds=samples,
                            median_seconds=float(np.median(samples)),positions_per_second=batch/float(np.median(samples)))
                benchmarks.append(record);atomic_json(out/'benchmark.json',dict(cpus=cpus,threads=args.threads,records=benchmarks))
                print(json.dumps(record),flush=True)
        model.config=replace(config,steps=8)
        features=(rng.random((8,972))<.12).astype(np.float32)
        legal=np.ones((8,82),np.uint8);policy=np.full((8,82),1/82,np.float32);value=np.linspace(-.8,.8,8,dtype=np.float32)
        start=time.perf_counter();result=model.train_step(features,legal,policy,value)
        benchmarks.append(dict(kind='train_step',steps=8,batch=8,seconds=time.perf_counter()-start,metrics=result))
        atomic_json(out/'benchmark.json',dict(cpus=cpus,threads=args.threads,records=benchmarks))
        print('Finished',flush=True)


if __name__=='__main__':
    main()
