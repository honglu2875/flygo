"""Offline teacher-policy/value distillation; the complete train step runs in Rust."""
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
from .storage import GIB,StorageBudget,StoragePressure


def evaluate(model,arrays,indices,*,batch_size=32,limit=512):
    indices=indices[:limit]
    if not len(indices):
        return dict(positions=0)
    total_policy=total_value=total_entropy=correct=0.0
    for begin in range(0,len(indices),batch_size):
        selected=indices[begin:begin+batch_size]
        result=model.infer(arrays['features'][selected])
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
    parser.add_argument('--release',default='pilot-v1')
    parser.add_argument('--run-id',default='fly-baseline-k8-seed1')
    parser.add_argument('--steps',type=int,default=1000,help='Total optimizer steps, including restored steps')
    parser.add_argument('--passes',type=int,default=8)
    parser.add_argument('--groups',type=int,default=656,help='Number of disjoint readout pools')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--threads',type=int,default=24)
    parser.add_argument('--cpus',help='Explicit allocation; defaults to spare physical cores')
    parser.add_argument('--rate',type=float,default=.003)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--eval-every',type=int,default=100)
    parser.add_argument('--checkpoint-every',type=int,default=250)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--peer',default='cubic27@t1v-n-a09f5679-w-1')
    args=parser.parse_args(argv)
    if min(args.steps,args.passes,args.groups,args.batch_size,args.threads,args.eval_every,args.checkpoint_every)<=0:
        parser.error('Step, batch, thread and interval counts must be positive')
    cpus=[int(x) for x in args.cpus.split(',')] if args.cpus else cpu_profile()['research_cpus'][:args.threads]
    if args.threads>len(cpus):
        parser.error('Thread count exceeds allocated physical CPUs')
    pin(cpus)
    root=args.root;run=root/'runs'/args.run_id
    run.mkdir(parents=True,exist_ok=bool(args.resume))
    try:
        run_training(args,cpus,run)
    except BaseException as error:
        previous=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
        atomic_json(run/'status.json',dict(state='failed',step=previous.get('step'),
                    pid=os.getpid(),error=repr(error),updated=time.time()))
        raise


def run_training(args,cpus,run):
    root=args.root
    config=FlyConfig(steps=args.passes,groups=args.groups,threads=args.threads,seed=args.seed)
    graph_path=args.graph or Path(json.loads((root/'runs/m4/graph.json').read_text())['path'])
    started=time.time()
    budget=StorageBudget(root)
    with budget.reserve(files=64*1024**2,heap=24*GIB,purpose='offline training '+args.run_id), \
            (run/'metrics.jsonl').open('a') as log:
        atomic_json(run/'status.json',dict(state='loading_data',pid=os.getpid(),cpus=cpus,updated=time.time()))
        manifest,arrays,indexes=load_release(root,root/'releases'/args.release/'manifest.json',cache=True)
        sampler=Sampler(arrays,indexes,args.seed)
        model=RustFly(load_graph(graph_path),config)
        step=0
        if args.resume:
            load_checkpoint(args.resume,model,sampler,dataset_id=manifest['dataset_id'])
            step=int(model.checkpoint_arrays()['optimizer_step'])
        atomic_json(run/'config.json',dict(model=asdict(config),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    dataset_id=manifest['dataset_id'],graph_id=model.graph['manifest']['graph_id'],cpus=cpus,
                    split_counts={k:len(v) for k,v in indexes.items()}))
        # Common fixed random slices, independent of model/sampler seed. Avoid
        # repeatedly reporting only the first few complete games in file order.
        evaluation={key:np.random.default_rng(912099).choice(indexes[key],
                    size=min(2048,len(indexes[key])),replace=False)
                    for key in ('train','validation','validation_novel')}
        def validation(step):
            result=dict(kind='validation',step=step,unix=time.time(),
                natural=evaluate(model,arrays,evaluation['validation'],limit=2048),
                novel=evaluate(model,arrays,evaluation['validation_novel'],limit=2048),
                train=evaluate(model,arrays,evaluation['train'],limit=2048))
            log.write(json.dumps(result)+'\n');log.flush();print(json.dumps(result),flush=True)
            return result
        validation(step)
        checkpoints=sorted((run/'checkpoints').glob('step-*.npz'))
        def checkpoint(metrics):
            path=run/'checkpoints'/f'step-{step:08d}.npz'
            receipt=save_checkpoint(model,sampler,path,dict(dataset_id=manifest['dataset_id'],metrics=metrics),root=root,peer=args.peer or None)
            checkpoints.append(path)
            atomic_json(run/'latest.json',receipt)
            # Keep two recovery points. Selected models live in a separate directory.
            while len(checkpoints)>2:
                old=checkpoints.pop(0)
                old.unlink();old.with_suffix('.json').unlink(missing_ok=True)

        metrics={'step':step}
        if not args.resume:checkpoint(metrics)
        while step<args.steps and not (run/'stop').exists():
            if step%25==0:
                try:
                    budget.check()
                except StoragePressure as error:
                    atomic_json(run/'status.json',dict(state='paused_storage',reason=str(error),step=step,updated=time.time()))
                    time.sleep(20);continue
            batch=sampler.batch(args.batch_size)
            begin=time.perf_counter();metrics=model.train_step(*batch,rate=args.rate)
            elapsed=time.perf_counter()-begin;step=metrics['step']
            record=dict(kind='train',**metrics,seconds=elapsed,positions_per_second=args.batch_size/elapsed,unix=time.time())
            if step%10==0 or step==1:
                log.write(json.dumps(record)+'\n');log.flush()
                atomic_json(run/'status.json',dict(state='training',pid=os.getpid(),updated=time.time(),**record))
            if step%args.eval_every==0 or step==args.steps:
                validation(step)
            if step%args.checkpoint_every==0 or step==args.steps:
                checkpoint(metrics)
        if step and (not checkpoints or checkpoints[-1].name!=f'step-{step:08d}.npz'):
            checkpoint(metrics)
        atomic_json(run/'status.json',dict(state='complete' if step==args.steps else 'stopped',step=step,
                    updated=time.time(),elapsed_seconds=time.time()-started))


if __name__=='__main__':
    main()
