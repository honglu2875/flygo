"""Full graph, explicit data-parallel TPU qualification against the Rust equations."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils as mh
import numpy as np

from .model import forward, loss, adam
from .numerics import softplus
from ..data.corpus import atomic_json
from ..fly import FlyConfig, RustFly, initialize, load_graph, PARAMETERS
from ..runtime import pin


def benchmark(config, report, out, mesh):
    root = Path(config['root'])
    graph_path = Path(json.loads((root / 'runs/m4/graph.json').read_text())['path'])
    graph = load_graph(graph_path)
    cfg = FlyConfig(steps=config['steps'], threads=len(config['cpus']))
    ports, params = initialize(graph, cfg)
    replicated = NamedSharding(mesh, P())
    data = NamedSharding(mesh, P('data'))
    def put(tree):
        # device_put accepts only addressable shardings in multi-controller JAX.
        return jax.tree.map(lambda a: jax.make_array_from_callback(
            a.shape,replicated,lambda index:a[index]),tree)
    jgraph = put({k:graph[k] for k in ('src','dst','type_id','sign')})
    if config.get('kernel')=='buckets':
        from .sparse import build_layout,LAYOUT_VERSION
        layout=build_layout(graph['src'],graph['dst'],len(graph['type_id']),
                            tile_edges=config.get('tile_edges',32768),tile_rows=config.get('tile_rows',128))
        report['layout']=dict(version=LAYOUT_VERSION,bytes=sum(a.nbytes for a in jax.tree.leaves(layout)),
                             forward_tiles=[list(b['edge_ids'].shape) for b in layout['forward']['buckets']],
                             transpose_tiles=[list(b['edge_ids'].shape) for b in layout['transpose']['buckets']])
        jgraph['layout']=put(layout)
        del layout
    jports = put(ports)
    jparams = put(params)
    kwargs = dict(steps=cfg.steps, groups=cfg.groups, actions=cfg.actions)
    report.update(status='qualifying', graph_id=graph['manifest']['graph_id'], records=[])
    atomic_json(out / 'status.json', report)

    def mapped(fun, in_specs, out_specs):
        return jax.jit(jax.shard_map(fun, mesh=mesh, in_specs=in_specs,
                                    out_specs=out_specs, check_vma=False))

    def inference(p, g, a, x):
        result = forward(p, g, a, x, **kwargs)
        return result['logits'], result['value']
    jf = mapped(inference, (P(),P(),P(),P('data')), (P('data'),P('data')))
    trace = mapped(lambda p,g,a,x: forward(p,g,a,x,**kwargs),
                   (P(),P(),P(),P('data')),
                   dict(logits=P('data'), value=P('data'), states=P(None,None,'data')))

    def derivative(p, g, a, batch):
        losses, gradient = jax.value_and_grad(loss, has_aux=True)(p,g,a,*batch,**kwargs)
        return lax.pmean(losses, 'data'), lax.pmean(gradient, 'data')
    jd = mapped(derivative, (P(),P(),P(),(P('data'),)*4), (P(),P()))

    def update(p, first, second, g, a, batch, corrections):
        losses, gradient = derivative(p,g,a,batch)
        p, first, second, norm = adam(p,gradient,first,second,0,
                                    norm_dtype=jnp.float32,corrections=corrections)
        return p, first, second, (losses, norm)
    jt = mapped(update, (P(),P(),P(),P(),P(),(P('data'),)*4,P()), (P(),P(),P(),P()))

    def batch(per_device):
        count = per_device * jax.device_count()
        rng = np.random.default_rng(819)
        x = (rng.random((count,cfg.features)) < .15).astype(np.float32)
        legal = rng.random((count,cfg.actions)) > .2
        legal[:,-1] = True
        target = (rng.random(legal.shape)*legal).astype(np.float32)
        target /= target.sum(axis=-1,keepdims=True)
        value = rng.uniform(-.8,.8,count).astype(np.float32)
        arrays = (x,legal,target,value)
        # Device mesh order need not equal hostname/process order. The callback
        # assigns each globally indexed position to its actual addressable shard.
        return arrays, tuple(jax.make_array_from_callback(a.shape,data,lambda index,a=a:a[index]) for a in arrays)

    full, local = batch(1)
    compiled = trace.lower(jparams,jgraph,jports,local[0]).compile()
    if config['mode']=='audit':
        (out/'forward-hlo.txt').write_text(compiled.as_text())
    expected = jax.block_until_ready(compiled(jparams,jgraph,jports,local[0]))
    states = mh.process_allgather(expected, tiled=True)
    # process_allgather returns the global arrays, including states' last-axis shards.
    reference = RustFly(graph,cfg,ports=ports,params=params) if jax.process_index()==0 else None
    errors = {}
    failed = []
    if reference is not None:
        actual = reference.infer(full[0],trace=True)
        for key in ('logits','value','states'):
            errors[key] = float(np.max(np.abs(np.asarray(actual[key])-np.asarray(states[key]))))
            try:
                np.testing.assert_allclose(actual[key],np.asarray(states[key]),rtol=3e-4,atol=3e-6,err_msg=key)
            except AssertionError as error:
                failed.append(str(error))
        errors['by_step']=[dict(max_absolute=float(np.max(np.abs(np.asarray(a)-b))),
                               rms=float(np.sqrt(np.mean(np.square(np.asarray(a)-b),dtype=np.float64))))
                           for a,b in zip(actual['states'],states['states'])]
        atomic_json(out/'forward-parity.json',dict(status='failed' if failed else 'passed',errors=errors,failures=failed))
    passed=bool(mh.broadcast_one_to_all(np.asarray(not failed),is_source=jax.process_index()==0))
    if config['mode']=='audit':
        weights=jax.jit(lambda p,g:g['sign'][g['src']]*softplus(p['edge']))(jparams,jgraph)
        if reference is not None:
            host=np.asarray(weights.addressable_shards[0].data)
            want=graph['sign'][graph['src']]*np.logaddexp(np.float32(0),params['edge'])
            errors['weights']=dict(max_absolute=float(np.max(np.abs(host-want))),
                                    max_relative=float(np.max(np.abs((host-want)/want))))
            atomic_json(out/'arithmetic-audit.json',dict(status='diagnostic',errors=errors,parity_passed=passed))
        report['audit']='completed; not training qualification'
        return
    if not passed:
        raise AssertionError('Full-state forward parity failed; see process-0 forward-parity.json')
    del compiled
    del expected, states
    def check_close(actual, expected, name, destination, *, rtol=3e-3, atol=5e-6):
        destination[name]=max(destination.get(name,0),float(np.max(np.abs(actual-expected))))
        try:
            np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol,err_msg=name)
        except AssertionError as error:
            failed.append(str(error))

    def collective_check(stage):
        if reference is not None:
            atomic_json(out/(stage+'-parity.json'),dict(status='failed' if failed else 'passed',
                        gradients=gradient_errors,updates=update_errors,failures=failed))
        if not bool(mh.broadcast_one_to_all(np.asarray(not failed),is_source=jax.process_index()==0)):
            raise AssertionError(stage+' parity failed; see process-0 evidence')

    first = jax.tree.map(jnp.zeros_like,jparams)
    second = jax.tree.map(jnp.zeros_like,jparams)
    gradient_errors = {}
    update_errors = {}
    started = time.time()
    for step in range(1,4):
        losses, gradient = jax.block_until_ready(jd(jparams,jgraph,jports,local))
        host_grad = jax.tree.map(lambda a: np.asarray(a.addressable_shards[0].data),gradient)
        if reference is not None:
            rloss, rgrad = reference.loss_and_grad(*full)
            check_close(np.asarray([rloss['policy_loss'],rloss['value_loss']]),
                        np.asarray([float(a) for a in losses[1]]),'loss',gradient_errors,rtol=3e-4,atol=3e-6)
            for key in PARAMETERS:
                check_close(rgrad[key],host_grad[key],key,gradient_errors)
        collective_check('gradient-'+str(step))
        corrections=put(np.asarray([1-.9**step,1-.999**step],np.float32))
        jparams,first,second,metrics = jax.block_until_ready(jt(jparams,first,second,jgraph,jports,local,corrections))
        if reference is not None:
            reference.train_step(*full)
            arrays=reference.checkpoint_arrays()
            for prefix,group in [('param/',jparams),('first/',first),('second/',second)]:
                for key,array in group.items():
                    actual=arrays[prefix+key]
                    expected=np.asarray(array.addressable_shards[0].data)
                    check_close(actual,expected,prefix+key,update_errors)
        collective_check('update-'+str(step))
        mh.sync_global_devices('qualified-update-'+str(step))
    report['parity'] = dict(status='passed', steps=cfg.steps, batch=len(full[0]),
                            updates=3, forward=errors, gradients=gradient_errors, updates_max_error=update_errors,
                            seconds=time.time()-started, numerical_runtime='tpu-fp32-norm-v1')
    atomic_json(out / 'status.json',report)
    print('Parity passed', flush=True)
    del gradient, host_grad, reference, first, second, trace, jd
    gc.collect()

    for per_device in config['per_device_batches']:
        full, inputs = batch(per_device)
        count=len(full[0])
        for kind in ('forward','train'):
            report.update(status='compiling', case=dict(kind=kind,per_device_batch=per_device,global_batch=count))
            atomic_json(out/'status.json',report)
            p=put(params)
            first=jax.tree.map(jnp.zeros_like,p);second=jax.tree.map(jnp.zeros_like,p)
            corrections=put(np.asarray([.1,.001],np.float32))
            arguments=(p,jgraph,jports,inputs[0]) if kind=='forward' else (p,first,second,jgraph,jports,inputs,corrections)
            fun=jf if kind=='forward' else jt
            begin=time.perf_counter()
            executable=fun.lower(*arguments).compile()
            compile_seconds=time.perf_counter()-begin
            memory=executable.memory_analysis()
            memory={k:int(getattr(memory,k)) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
            required=memory['argument_size_in_bytes']+memory['output_size_in_bytes']+memory['temp_size_in_bytes']-memory['alias_size_in_bytes']
            if required > 25*(1<<30):
                record=dict(**report['case'],status='skipped_memory',compile_seconds=compile_seconds,memory=memory)
            else:
                result=jax.block_until_ready(executable(*arguments))
                pin(config['cpus'])
                samples=[]
                for repetition in range(config['repetitions']):
                    if kind=='train':
                        p,first,second,_=result
                        corrections=put(np.asarray([1-.9**(repetition+2),1-.999**(repetition+2)],np.float32))
                        arguments=(p,first,second,jgraph,jports,inputs,corrections)
                    mh.sync_global_devices(f'{kind}-{per_device}-{repetition}')
                    begin=time.perf_counter()
                    result=jax.block_until_ready(executable(*arguments))
                    samples.append(time.perf_counter()-begin)
                times=np.asarray(mh.process_allgather(np.asarray(samples),tiled=False))
                seconds=float(np.median(times.max(axis=0)))
                record=dict(**report['case'],status='passed',compile_seconds=compile_seconds,
                            memory=memory,host_seconds=times.tolist(),median_seconds=seconds,
                            positions_per_second=count/seconds,
                            device_memory=[d.memory_stats() for d in jax.local_devices()])
                del result
            report['records'].append(record)
            atomic_json(out/'status.json',report)
            print(json.dumps(record),flush=True)
            del executable,arguments,p,first,second
            gc.collect()
        # Bound executable/cache lifetime between shape specializations.
        jax.clear_caches()
    atomic_json(out/'result.json',report)
