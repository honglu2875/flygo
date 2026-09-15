"""Offline teacher-policy/value distillation with interchangeable Rust/JAX learners."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np

from .checkpoint import save_checkpoint,load_checkpoint
from .data.corpus import atomic_json
from .data.loader import load_release,Sampler
from .fly import FlyConfig,RustFly,load_graph
from .runtime import cpu_profile,pin
from .schedule import Schedule
from .optimizer import (DEFAULT_EPSILON,validate_epsilon,check_epsilon_resume,
                        CLIP_MODES,DEFAULT_CLIP_MODE,check_clipping_resume)
from .storage import GIB,StorageBudget,StoragePressure


def evaluate(model,arrays,indices,*,batch_size=32,limit=512,transform=None):
    indices=indices[:limit]
    if not len(indices):
        return dict(positions=0)
    total_policy=total_value=total_entropy=correct=0.0
    for begin in range(0,len(indices),batch_size):
        selected=indices[begin:begin+batch_size]
        features=arrays['features'][selected]
        result=model.infer(transform(features) if transform else features)
        logits=np.where(arrays['legal'][selected],result['logits'],-1e30)
        logits-=logits.max(axis=1,keepdims=True)
        logp=logits-np.log(np.exp(logits).sum(axis=1,keepdims=True))
        target=arrays['raw_policy'][selected]
        total_policy+=float(-(target*logp).sum(dtype=np.float64))
        total_entropy+=float(-(target*np.log(np.maximum(target,1e-30))).sum(dtype=np.float64))
        total_value+=float(np.square(result['value']-arrays['raw_value'][selected]).sum(dtype=np.float64))
        correct+=int((logits.argmax(axis=1)==target.argmax(axis=1)).sum())
    n=len(indices)
    return dict(positions=n,policy_cross_entropy=total_policy/n,policy_kl=(total_policy-total_entropy)/n,
                value_mse=total_value/n,teacher_top1_agreement=correct/n)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    parser.add_argument('--graph',type=Path)
    parser.add_argument('--ports',type=Path,help='Qualified sensory/readout artifact; defaults to seeded random ports')
    parser.add_argument('--input-map',type=Path,help='Qualified spherical visual/context attachment')
    parser.add_argument('--head-mask',type=Path,help='Qualified immutable external policy/value mask')
    parser.add_argument('--input-mode',choices=('history','current','neutral'),default='history')
    parser.add_argument('--qualification',type=Path,help='Full-circuit numerical gate required for an input map')
    parser.add_argument('--release',default='pilot-v1')
    parser.add_argument('--run-id',default='fly-baseline-k8-seed1')
    parser.add_argument('--steps',type=int,default=1000,help='Total optimizer steps, including restored steps')
    parser.add_argument('--passes',type=int,default=8)
    parser.add_argument('--readout-mean-scale',type=float,default=1.0,help='Fixed common readout component scale in (0,1]; one preserves the baseline')
    parser.add_argument('--rate-softness',type=float,default=0.0,help='Fixed smooth firing-rate scale; zero uses the original ReLU')
    parser.add_argument('--groups',type=int,default=656,help='Number of disjoint readout pools')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--threads',type=int,default=24)
    parser.add_argument('--cpus',help='Explicit allocation; defaults to spare physical cores')
    parser.add_argument('--rate',type=float,default=.003)
    parser.add_argument('--warmup-steps',type=int,default=0)
    parser.add_argument('--decay-until',type=int,default=0,help='Absolute update at cosine floor; zero keeps a constant peak')
    parser.add_argument('--final-rate-ratio',type=float,default=.1)
    parser.add_argument('--rate-scales',type=json.loads,default={},help='JSON parameter-group multipliers, e.g. {"bias":0.1}')
    parser.add_argument('--backend',choices=('cpu','tpu'),default='cpu')
    parser.add_argument('--model',choices=('fly','cnn'),default='fly')
    parser.add_argument('--channels',type=int,default=64,help='CNN control width')
    parser.add_argument('--blocks',type=int,default=10,help='CNN control residual blocks')
    parser.add_argument('--clip',type=float,default=1.0)
    parser.add_argument('--clip-mode',choices=CLIP_MODES,default=DEFAULT_CLIP_MODE)
    parser.add_argument('--epsilon',type=float,default=DEFAULT_EPSILON,help='Adam denominator epsilon, outside the square root')
    parser.add_argument('--diagnostics-every',type=int,default=0,help='Zero disables extra gradient/activity measurements')
    parser.add_argument('--diagnostic-batch-size',type=int,default=32,help='Bound extra full-state/gradient measurements independently of the training batch')
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--eval-every',type=int,default=100)
    parser.add_argument('--eval-batch-size',type=int,default=32)
    parser.add_argument('--eval-positions',type=int,default=2048)
    parser.add_argument('--checkpoint-every',type=int,default=250)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--peer',default='cubic27@t1v-n-a09f5679-w-1')
    args=parser.parse_args(argv)
    if args.model=='cnn' and args.clip_mode!=DEFAULT_CLIP_MODE:
        parser.error('This clipping study is scoped to the fly model')
    if min(args.steps,args.passes,args.groups,args.batch_size,args.threads,args.eval_every,
           args.eval_batch_size,args.eval_positions,args.checkpoint_every,args.diagnostic_batch_size)<=0:
        parser.error('Step, batch, thread and interval counts must be positive')
    if not np.isfinite(args.rate) or args.rate<=0 or not np.isfinite(args.clip) or args.clip<=0 or args.diagnostics_every<0:
        parser.error('Finite positive learning rate/clip and a nonnegative diagnostic interval are required')
    if args.model=='cnn' and (args.ports or args.input_map or args.head_mask or args.diagnostics_every or args.rate_softness or args.readout_mean_scale!=1):
        parser.error('Fly ports, firing rates and activity diagnostics apply only to the fly model')
    if args.input_map and (args.ports or not args.qualification):
        parser.error('An input map replaces --ports and requires --qualification')
    if args.head_mask and not args.input_map:
        parser.error('Head-mask training requires a qualified visual/context input map')
    if args.input_map and args.backend!='cpu':
        parser.error('Visual/context training requires a separate actual TPU qualification before TPU use')
    if not args.input_map and args.input_mode!='history':
        parser.error('An input mode requires --input-map')
    try:Schedule(args.rate,args.warmup_steps,args.decay_until,args.final_rate_ratio)
    except ValueError as error:parser.error(str(error))
    try:validate_epsilon(args.epsilon)
    except ValueError as error:parser.error(str(error))
    cpus=[int(x) for x in args.cpus.split(',')] if args.cpus else cpu_profile()['research_cpus'][:args.threads]
    if args.threads>len(cpus):
        parser.error('Thread count exceeds allocated physical CPUs')
    pin(cpus)
    root=args.root;run=root/'runs'/args.run_id
    run.mkdir(parents=True,exist_ok=bool(args.resume))
    owns_distributed=False
    try:
        if args.backend=='tpu':
            os.environ['JAX_PLATFORMS']='tpu'
            import jax
            if not jax.distributed.is_initialized():
                jax.distributed.initialize(initialization_timeout=90)
                owns_distributed=True
        elif args.model=='cnn':os.environ['JAX_PLATFORMS']='cpu'
        run_training(args,cpus,run)
    except BaseException as error:
        previous=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
        atomic_json(run/'status.json',dict(state='failed',step=previous.get('step'),
                    pid=os.getpid(),error=repr(error),updated=time.time()))
        raise
    finally:
        if owns_distributed:jax.distributed.shutdown()


def run_training(args,cpus,run):
    root=args.root
    schedule=Schedule(args.rate,args.warmup_steps,args.decay_until,args.final_rate_ratio)
    if args.model=='cnn':
        from .jax.cnn import CNNConfig,JaxCNN
        config=CNNConfig(channels=args.channels,blocks=args.blocks,threads=args.threads,seed=args.seed)
    else:config=FlyConfig(steps=args.passes,groups=args.groups,threads=args.threads,seed=args.seed,rate_softness=args.rate_softness,readout_mean_scale=args.readout_mean_scale)
    graph_path=(args.graph or Path(json.loads((root/'runs/m4/graph.json').read_text())['path'])) if args.model=='fly' else None
    started=time.time()
    budget=StorageBudget(root)
    with budget.reserve(files=64*1024**2,heap=24*GIB,purpose='offline training '+args.run_id), \
            (run/'metrics.jsonl').open('a') as log:
        atomic_json(run/'status.json',dict(state='loading_data',pid=os.getpid(),cpus=cpus,updated=time.time()))
        manifest,arrays,indexes=load_release(root,root/'releases'/args.release/'manifest.json',cache=True)
        sampler=Sampler(arrays,indexes,args.seed)
        graph=load_graph(graph_path) if args.model=='fly' else None
        ports=None;port_contract=None;visual_contract=None;transform=None;head_mask=None
        if args.ports:
            from .ports import load_ports
            ports,port_contract=load_ports(args.ports,graph_id=graph['manifest']['graph_id'],
                features=config.features,groups=config.groups,seed=config.seed)
        if args.input_map:
            from dataclasses import replace
            from .attachments import load_attachment,input_contract,require_qualification
            adapter,ports,port_contract=load_attachment(args.input_map,
                graph_id=graph['manifest']['graph_id'],dataset_id=manifest['dataset_id'])
            if config.groups!=port_contract['groups']:
                raise ValueError('Readout group count differs from the visual/context attachment')
            config=replace(config,features=adapter.features)
            visual_contract=input_contract(port_contract,args.input_mode)
            if args.head_mask:
                from .readout import load_head_mask
                head_mask=load_head_mask(args.head_mask,graph_id=graph['manifest']['graph_id'],
                    attachment_sha256=port_contract['sha256'])
                head_mask.validate_shape(actions=config.actions,groups=config.groups)
            require_qualification(args.qualification,visual_contract,config,batch_size=args.batch_size,
                rate=args.rate,epsilon=args.epsilon,clip=args.clip,rate_scales=args.rate_scales,head_mask=head_mask,
                clip_mode=args.clip_mode)
            transform=lambda features:adapter.encode(features,args.input_mode)
        if args.model=='cnn':model=JaxCNN(config)
        elif args.backend=='tpu':
            from .jax.learner import JaxFly
            model=JaxFly(graph,config,ports=ports,clip_mode=args.clip_mode)
        else:
            model=RustFly(graph,config,ports=ports,head_mask=head_mask,clip_mode=args.clip_mode)
        diagnostic_size=min(args.batch_size,args.diagnostic_batch_size)
        if args.diagnostics_every and hasattr(model,'mesh') and diagnostic_size%model.mesh.size:
            raise ValueError('Diagnostic batch must divide evenly across the data mesh')
        any_host=getattr(model,'collective_any',bool)
        step=0
        if args.resume:
            previous=load_checkpoint(args.resume,model,sampler,dataset_id=manifest['dataset_id'],
                numerical_runtime=getattr(model,'numerical_runtime','rust-fp32-f64-norm-v1'))
            if previous.get('input_contract')!=visual_contract:
                raise ValueError('Checkpoint external input encoding differs')
            schedule.check_resume(previous.get('training_contract',{}))
            check_epsilon_resume(args.epsilon,previous.get('training_contract',{}))
            check_clipping_resume(args.clip_mode,previous.get('training_contract',{}))
            step=int(model.checkpoint_arrays()['optimizer_step'])
        atomic_json(run/'config.json',dict(model=asdict(config),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    dataset_id=manifest['dataset_id'],graph_id=model.graph['manifest']['graph_id'],cpus=cpus,port_contract=port_contract,input_contract=visual_contract,
                    split_counts={k:len(v) for k,v in indexes.items()},
                    **({} if head_mask is None else {'head_mask':head_mask.contract})))
        # Common fixed random slices, independent of model/sampler seed. Avoid
        # repeatedly reporting only the first few complete games in file order.
        evaluation={key:np.random.default_rng(912099).choice(indexes[key],
                    size=min(args.eval_positions,len(indexes[key])),replace=False)
                    for key in ('train','validation','validation_novel')}
        def validation(step):
            result=dict(kind='validation',step=step,unix=time.time(),
                natural=evaluate(model,arrays,evaluation['validation'],limit=args.eval_positions,batch_size=args.eval_batch_size,transform=transform),
                novel=evaluate(model,arrays,evaluation['validation_novel'],limit=args.eval_positions,batch_size=args.eval_batch_size,transform=transform),
                train=evaluate(model,arrays,evaluation['train'],limit=args.eval_positions,batch_size=args.eval_batch_size,transform=transform))
            if args.input_map and step in (0,args.steps):
                result['input_perturbations']={mode:evaluate(model,arrays,evaluation['validation'],
                    limit=args.eval_positions,batch_size=args.eval_batch_size,
                    transform=lambda x,mode=mode:adapter.encode(x,mode))
                    for mode in ('history','current','neutral') if mode!=args.input_mode}
            log.write(json.dumps(result)+'\n');log.flush();print(json.dumps(result),flush=True)
            return result
        validation(step)
        checkpoints=sorted((run/'checkpoints').glob('step-*.npz'))
        def checkpoint(metrics):
            path=run/'checkpoints'/f'step-{step:08d}.npz'
            error=None
            try:
                receipt=save_checkpoint(model,sampler,path,dict(dataset_id=manifest['dataset_id'],metrics=metrics,
                    port_contract=port_contract,input_contract=visual_contract,training_contract=dict(batch_size=args.batch_size,rate=args.rate,
                        clip=args.clip,clip_mode=args.clip_mode,rate_scales=args.rate_scales,schedule=schedule.contract(),epsilon=args.epsilon)),
                    root=root,peer=args.peer or None)
            except Exception as failure:
                error=failure
            if any_host(error is not None):
                raise RuntimeError('A controller could not publish its checkpoint: '+str(error or 'peer failure')) from error
            if hasattr(model,'verify_checkpoint_copies'):
                receipt=model.verify_checkpoint_copies(receipt)
                atomic_json(path.with_suffix('.json'),receipt)
            checkpoints.append(path)
            atomic_json(run/'latest.json',receipt)
            # Keep two recovery points. Selected models live in a separate directory.
            while len(checkpoints)>2:
                old=checkpoints.pop(0)
                old.unlink();old.with_suffix('.json').unlink(missing_ok=True)

        metrics={'step':step}
        clipped_updates=observed_updates=0
        if not args.resume:checkpoint(metrics)
        while step<args.steps and not any_host((run/'stop').exists()):
            if step%25==0:
                pressure=None
                try:
                    budget.check()
                except StoragePressure as error:
                    pressure=str(error)
                if any_host(pressure is not None):
                    atomic_json(run/'status.json',dict(state='paused_storage',reason=pressure or 'Peer storage pressure',step=step,updated=time.time()))
                    time.sleep(20);continue
            sample_begin=time.perf_counter();batch=sampler.batch(args.batch_size)
            if transform:batch=(transform(batch[0]),*batch[1:])
            sample_seconds=time.perf_counter()-sample_begin
            diagnostic=args.diagnostics_every and (step+1)%args.diagnostics_every==0
            before=model.parameters() if diagnostic else None
            rate=schedule.rate(step+1)
            begin=time.perf_counter();metrics=model.train_step(*batch,rate=rate,clip=args.clip,
                                                              rate_scales=args.rate_scales or None,epsilon=args.epsilon)
            elapsed=time.perf_counter()-begin;step=metrics['step']
            clipped=metrics['clipping']['clipped']
            observed_updates+=1;clipped_updates+=int(clipped)
            record=dict(kind='train',**metrics,learning_rate=rate,clipped=clipped,
                        clipping_fraction=clipped_updates/observed_updates,
                        clipping_scope='updates since this process started',
                        seconds=elapsed,positions_per_second=args.batch_size/elapsed,
                        sample_seconds=sample_seconds,
                        sample_and_update_positions_per_second=args.batch_size/(sample_seconds+elapsed),
                        training_exposures=step*args.batch_size,unix=time.time())
            if step%10==0 or step==1:
                log.write(json.dumps(record)+'\n');log.flush()
                atomic_json(run/'status.json',dict(state='training',pid=os.getpid(),updated=time.time(),**record))
            if diagnostic:
                from .diagnostics import measure
                diagnostic_batch=tuple(array[:diagnostic_size] for array in batch)
                report=measure(model,diagnostic_batch,before,metrics,clip=args.clip,
                               training_batch_size=args.batch_size)
                log.write(json.dumps(report,allow_nan=False)+'\n');log.flush()
                del before
            if step%args.eval_every==0 or step==args.steps:
                validation(step)
            if step%args.checkpoint_every==0 or step==args.steps:
                checkpoint(metrics)
        if step and (not checkpoints or checkpoints[-1].name!=f'step-{step:08d}.npz'):
            checkpoint(metrics)
        atomic_json(run/'status.json',dict(state='complete' if step==args.steps else 'stopped',step=step,
                    training_exposures=step*args.batch_size,numerical_runtime=getattr(model,'numerical_runtime','rust-fp32-f64-norm-v1'),
                    updated=time.time(),elapsed_seconds=time.time()-started))


if __name__=='__main__':
    main()
