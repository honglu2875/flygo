#!/usr/bin/env python3
"""Fit a small fixed set of real training positions; diagnose optimization separately from generalization."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np

from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release
from flygo.fly import FlyConfig,RustFly,load_graph
from flygo.runtime import pin
from flygo.storage import StorageBudget,GIB
from flygo.train import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    parser.add_argument('--release',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpus',default='117,118,119')
    parser.add_argument('--passes',type=int,default=4)
    parser.add_argument('--groups',type=int,default=656)
    parser.add_argument('--positions',type=int,default=16)
    parser.add_argument('--updates',type=int,default=500)
    args=parser.parse_args()
    if min(args.passes,args.groups,args.positions,args.updates)<1:parser.error('Counts must be positive')
    cpus=list(map(int,args.cpus.split(',')));pin(cpus)
    args.output.mkdir(parents=True,exist_ok=False)
    with StorageBudget(args.root).reserve(files=16*1024**2,heap=8*GIB,purpose='real-data fitting diagnostic'):
        atomic_json(args.output/'status.json',dict(state='loading_data',pid=os.getpid(),updated=time.time()))
        manifest,arrays,indexes=load_release(args.root,args.root/'releases'/args.release/'manifest.json',cache=True)
        selected=np.random.default_rng(3400).choice(indexes['train'],size=args.positions,replace=False)
        config=FlyConfig(steps=args.passes,groups=args.groups,threads=len(cpus),seed=1)
        graph=load_graph(Path(json.loads((args.root/'runs/m4/graph.json').read_text())['path']))
        model=RustFly(graph,config)
        batch=tuple(arrays[name][selected] for name in ('features','legal','raw_policy','raw_value'))
        initial_edges=model.parameters()['edge']
        losses=[];started=time.time()
        atomic_json(args.output/'config.json',dict(dataset_id=manifest['dataset_id'],model=asdict(config),
            cpus=cpus,indices=selected.tolist(),updates=args.updates,augmentation=False,
            gate=dict(policy_kl_below=.1,value_mse_below=.05),scope='Training-only diagnostic, no generalization or Go-strength claim'))
        for step in range(args.updates+1):
            if step%25==0 or step==args.updates:
                metrics=evaluate(model,arrays,selected,batch_size=args.positions,limit=args.positions)
                record=dict(step=step,**metrics);losses.append(record)
                atomic_json(args.output/'status.json',dict(state='training',pid=os.getpid(),updated=time.time(),**record))
                print(json.dumps(record),flush=True)
                if step>=50 and metrics['policy_kl']<.1 and metrics['value_mse']<.05:break
            if step<args.updates:model.train_step(*batch,rate=.003)
        final=losses[-1]
        passed=final['policy_kl']<.1 and final['value_mse']<.05
        atomic_json(args.output/'result.json',dict(status='complete',fit_gate='passed' if passed else 'not_met',
            graph_id=graph['manifest']['graph_id'],dataset_id=manifest['dataset_id'],losses=losses,
            changed_edges=int(np.count_nonzero(model.parameters()['edge']!=initial_edges)),
            seconds=time.time()-started,model=asdict(config)))
        atomic_json(args.output/'status.json',dict(state='complete',fit_gate='passed' if passed else 'not_met',
                   updated=time.time(),step=final['step']))


if __name__=='__main__':main()
