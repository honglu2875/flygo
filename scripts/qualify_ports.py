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
from flygo.optimizer import CLIP_MODES,validate_epsilon
from flygo.objectives import value_core_scale
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB,StorageBudget


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--ports',type=Path,nargs='+',help='Omit for the original seeded random adapters')
    p.add_argument('--input-map',type=Path,help='Visual/context attachment; replaces --ports')
    p.add_argument('--head-mask',type=Path,help='Immutable external head mask; requires --input-map')
    p.add_argument('--seed',type=int,help='Learner seed for a fixed visual/context attachment')
    p.add_argument('--input-modes',nargs='+',choices=('history','current','neutral'),default=['history','current','neutral'])
    p.add_argument('--passes',type=int,default=4)
    p.add_argument('--readout-mean-scale',type=float,default=1.0)
    p.add_argument('--batch-size',type=int,default=1,help='Bounded full-reference batch, 1..32')
    p.add_argument('--rate-softness',type=float,default=0.0)
    p.add_argument('--epsilon',type=float,default=1e-8)
    p.add_argument('--rate',type=float,default=.01)
    p.add_argument('--rate-scales',type=json.loads,default={})
    p.add_argument('--clip-mode',choices=CLIP_MODES,default='global')
    p.add_argument('--value-core-scale',type=value_core_scale,default=1.0)
    p.add_argument('--protocol',choices=('aligned-peak','free-warmup'))
    p.add_argument('--numerical-plan',type=Path,help='Registered transition protocol; required with --protocol')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpus',default='117,118,119')
    args=p.parse_args();cpus=list(map(int,args.cpus.split(',')));pin(cpus)
    if bool(args.protocol)!=bool(args.numerical_plan):p.error('Protocol and numerical plan must be supplied together')
    numerical_plan=None;selected_protocol=None
    used_rates=[args.rate]*3
    if args.numerical_plan:
        from flygo.schedule import Schedule
        numerical_plan=json.loads(args.numerical_plan.read_text())
        if (args.seed not in numerical_plan['seeds'] or args.passes!=numerical_plan['steps']
                or args.batch_size!=numerical_plan['batch_size'] or args.rate!=numerical_plan['rate']
                or args.epsilon!=numerical_plan['epsilon'] or args.rate_scales!=numerical_plan['rate_scales']
                or args.input_modes!=[numerical_plan['input_mode']] or cpus!=numerical_plan['cpus']
                or args.input_map!=args.root/numerical_plan['input_map']
                or args.head_mask!=args.root/numerical_plan['head_mask']
                or args.clip_mode not in numerical_plan.get('clip_modes',numerical_plan['arms']) or numerical_plan['clip']!=1
                or args.value_core_scale not in [value_core_scale(s) for s in numerical_plan.get('value_core_scales',[1.0])]
                or args.rate_softness!=numerical_plan['rate_softness']
                or args.readout_mean_scale!=numerical_plan['readout_mean_scale']
                or numerical_plan['platform']!='cpu' or numerical_plan['updates']!=3
                or numerical_plan['tolerances']!={
                    'states_outputs_losses':dict(rtol=3e-4,atol=3e-6),
                    'gradients_parameters_moments':dict(rtol=3e-3,atol=5e-6)}):
            p.error('Arguments differ from the registered clipping transition protocol')
        schedule=Schedule(args.rate,numerical_plan['warmup_steps'])
        selected_protocol=next(x for x in numerical_plan['protocols'] if x['name']==args.protocol)
        used_rates=[args.rate if args.protocol=='aligned-peak' else schedule.rate(i+1) for i in range(3)]
        if not np.allclose(used_rates,selected_protocol['rates'],rtol=0,atol=1e-18):p.error('Registered rates differ')
    if args.input_map and args.ports:p.error('An input map replaces --ports')
    if args.head_mask and not args.input_map:p.error('Head masks require --input-map')
    if args.seed is not None and (not args.input_map or args.seed<0):
        p.error('A nonnegative learner seed requires --input-map')
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
        adapter=None;runtime=None;head_mask=None
        if args.input_map:
            from flygo.attachments import load_attachment,input_contract,runtime_hashes
            adapter,visual_ports,visual_receipt=load_attachment(args.input_map,
                graph_id=graph['manifest']['graph_id'],dataset_id=manifest['dataset_id'])
            runtime=runtime_hashes()
            if numerical_plan and (runtime!=numerical_plan['runtime_sha256']
                    or graph['manifest']['graph_id']!=numerical_plan['graph_id']
                    or manifest['dataset_id']!=numerical_plan['dataset_id']
                    or visual_receipt['sha256']!=numerical_plan['input_map_sha256']
                    or sha256(args.head_mask)!=numerical_plan['head_mask_sha256']):
                raise ValueError('Runtime, corpus or attachment differs from the frozen protocol')
            if args.head_mask:
                from flygo.readout import load_head_mask
                head_mask=load_head_mask(args.head_mask,graph_id=graph['manifest']['graph_id'],
                    attachment_sha256=visual_receipt['sha256'])
        variants=args.input_modes if adapter else args.ports or [None]
        for path in variants:
            visual_contract=input_contract(visual_receipt,path) if adapter else None
            receipt=(dict(visual_receipt,seed=args.seed if args.seed is not None else 1) if adapter else
                json.loads(path.with_suffix('.json').read_text()) if path else dict(seed=1,groups=656,sha256=None))
            cfg=FlyConfig(steps=args.passes,threads=len(cpus),seed=receipt['seed'],groups=receipt['groups'],
                features=adapter.features if adapter else 972,rate_softness=args.rate_softness,readout_mean_scale=args.readout_mean_scale)
            ports,params=initialize(graph,cfg)
            if adapter:ports=visual_ports
            elif path:ports,_=load_ports(path,graph_id=graph['manifest']['graph_id'],features=cfg.features,groups=cfg.groups,seed=cfg.seed)
            rust=RustFly(graph,cfg,ports=ports,params=params,head_mask=head_mask,clip_mode=args.clip_mode,
                         value_core_scale=args.value_core_scale)
            sampler=Sampler(arrays,indexes,cfg.seed);jports=jax.tree.map(jnp.asarray,ports)
            jp=jax.tree.map(jnp.asarray,params);first=jax.tree.map(jnp.zeros_like,jp);second=jax.tree.map(jnp.zeros_like,jp)
            kwargs=dict(steps=cfg.steps,groups=cfg.groups,actions=cfg.actions,rate_softness=cfg.rate_softness,readout_mean_scale=cfg.readout_mean_scale,
                head_mask=None if head_mask is None else head_mask.parameter_masks())
            infer=jax.jit(lambda p,g,a,x:forward(p,g,a,x,**kwargs))
            derivative=jax.jit(jax.value_and_grad(lambda p,g,a,*batch:loss(p,g,a,*batch,**kwargs,
                value_core_scale=args.value_core_scale),has_aux=True))
            multipliers={k:np.float32(args.rate_scales.get(k,1)) for k in params}
            update=jax.jit(lambda p,g,m,v,step,rate:adam(p,g,m,v,step,
                rate={k:rate*scale for k,scale in multipliers.items()},clip=np.float32(1),norm_dtype=jnp.float64,
                epsilon=np.float32(args.epsilon),clip_mode=args.clip_mode,return_stats=True))
            errors={};deltas=[];aligned=[];clipping=[]
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
                before=rust.checkpoint_arrays()
                if args.protocol=='aligned-peak':
                    jp={k:jnp.asarray(before['param/'+k]) for k in params}
                    first={k:jnp.asarray(before['first/'+k]) for k in params}
                    second={k:jnp.asarray(before['second/'+k]) for k in params}
                    for prefix,group in [('param/',jp),('first/',first),('second/',second)]:
                        for key,value in group.items():np.testing.assert_array_equal(before[prefix+key],np.asarray(value))
                    aligned.append(dict(step=step,arrays=3*len(params),exact=True))
                reference_before=jax.tree.map(np.asarray,jp)
                batch=sampler.batch(args.batch_size)
                if adapter:batch=(adapter.encode(batch[0],path),*batch[1:])
                output=jax.block_until_ready(infer(jp,jgraph,jports,batch[0]));native=rust.infer(batch[0],trace=True)
                for key in native:check(native[key],output[key],f'{step}/forward/{key}',3e-4,3e-6)
                (_,losses),gradient=jax.block_until_ready(derivative(jp,jgraph,jports,*batch))
                native_loss,native_gradient=rust.loss_and_grad(*batch)
                for key,value in zip(('policy_loss','value_loss'),losses):
                    check(native_loss[key],value,f'{step}/loss/{key}',3e-4,3e-6)
                for key,value in native_gradient.items():check(value,gradient[key],f'{step}/gradient/{key}')
                jp,first,second,norm,norms,factors=jax.block_until_ready(update(jp,gradient,first,second,step+1,np.float32(used_rates[step])))
                metrics=rust.train_step(*batch,rate=used_rates[step],rate_scales=args.rate_scales or None,epsilon=args.epsilon)
                check(metrics['gradient_norm'],norm,f'{step}/clipping/norm')
                for key in params:
                    check(metrics['clipping']['group_norms'][key],norms[key],f'{step}/group_norm/{key}')
                    check(metrics['clipping']['factors'][key],factors[key],f'{step}/clip_factor/{key}')
                clipping.append(dict(step=step,gradient_norm=metrics['gradient_norm'],**metrics['clipping']))
                saved=rust.checkpoint_arrays()
                for key in params:
                    a=saved['param/'+key].astype(np.float64)-before['param/'+key].astype(np.float64)
                    b=np.asarray(jp[key]).astype(np.float64)-reference_before[key].astype(np.float64)
                    err=a-b
                    deltas.append(dict(step=step,parameter=key,max_absolute=float(np.max(np.abs(err))),
                        rms=float(np.sqrt(np.mean(err*err))),reference_l2=float(np.linalg.norm(b)),
                        relative_l2=float(np.linalg.norm(err)/max(np.linalg.norm(b),1e-30))))
                for prefix,group in [('param/',jp),('first/',first),('second/',second)]:
                    for key,value in group.items():check(saved[prefix+key],value,f'{step}/'+prefix+key)
                atomic_json(args.output/'status.json',dict(state='qualifying',port=str(path),completed_updates=step+1))
            records.append(dict(ports=str(args.input_map or path) if path else None,ports_sha256=receipt['sha256'],input_contract=visual_contract,model=asdict(cfg),readout_neurons=int((ports['output_group']>=0).sum()),
                                updates=3,batch_size=args.batch_size,optimizer=dict(rate=args.rate,rate_scales=args.rate_scales,epsilon=args.epsilon,clip=1,clip_mode=args.clip_mode,value_core_scale=args.value_core_scale),errors=errors,
                                numerical_protocol=args.protocol,actual_rates=used_rates,aligned_checkpoints=aligned,
                                update_differences=deltas,clipping=clipping,
                                **({} if head_mask is None else {'head_mask':head_mask.contract})))
            atomic_json(args.output/'result.json',dict(status='complete' if len(records)==len(variants) else 'running',
                graph_id=graph['manifest']['graph_id'],dataset_id=manifest['dataset_id'],records=records,cpus=cpus,
                script_sha256=sha256(Path(__file__)),runtime_sha256=runtime,seconds=time.time()-begin,
                numerical_plan_sha256=None if args.numerical_plan is None else sha256(args.numerical_plan),
                numerical_protocol=args.protocol,registered_protocol=selected_protocol,
                scope='Full states, losses, all gradients, parameters, moments and actual clipping factors on V0 inputs; independent E-by-B JAX CPU reference. '
                      'Aligned peak copies native parameters/moments before each transition; free warmup uses independent trajectories under actual rates. '
                      'Without a registered protocol, three independent peak-rate updates are used. No scientific endpoint or TPU use.'))
            del rust,jp,first,second,gradient,native_gradient,output,native,infer,derivative,update
            jax.clear_caches()
        atomic_json(args.output/'status.json',dict(state='complete',variants=len(records)))
        print(json.dumps(dict(status='passed',output=str(args.output),variants=len(records))))


if __name__=='__main__':main()
