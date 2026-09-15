#!/usr/bin/env python3
"""Check paired initialization and fresh-process recovery for an input-map change."""
from dataclasses import asdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from flygo.attachments import load_attachment, input_contract, runtime_hashes
from flygo.checkpoint import load_checkpoint, save_checkpoint
from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release, Sampler
from flygo.fly import FlyConfig, RustFly, load_graph
from flygo.play import load_player
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.schedule import Schedule
from flygo.storage import GIB, StorageBudget


def hashes(arrays):
    result={}
    for key,value in arrays.items():
        array=np.asarray(value)
        result[key]=dict(shape=list(array.shape),dtype=str(array.dtype),
                         sha256=hashlib.sha256(array.tobytes()).hexdigest())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True,help='Verified initial checkpoint hashes and metadata')
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-only',action='store_true')
    args=p.parse_args();root=args.root;out=args.output
    out.resolve().relative_to(root.resolve())
    plan=json.loads(args.plan.read_text());reference=json.loads(args.reference.read_text())
    job=next(j for j in plan['jobs'] if j['seed']==args.seed)
    pin(job['cpus'])
    with StorageBudget(root).reserve(files=8<<20,heap=24*GIB,purpose='attachment IO qualification'):
        if not args.resume_only:out.mkdir(parents=True,exist_ok=False)
        graph_path=Path(json.loads((root/'runs/m4/graph.json').read_text())['path'])
        graph=load_graph(graph_path)
        manifest,arrays,indexes=load_release(root,root/'releases'/plan['release']/'manifest.json',cache=True)
        adapter,ports,receipt=load_attachment(root/plan['input_map'],
            graph_id=graph['manifest']['graph_id'],dataset_id=manifest['dataset_id'])
        assert receipt['sha256']==plan['input_map_sha256']
        config=FlyConfig(steps=job['passes'],groups=job['groups'],features=adapter.features,
                        seed=job['seed'],threads=len(job['cpus']))
        model=RustFly(graph,config,ports=ports);sampler=Sampler(arrays,indexes,args.seed)
        contract=input_contract(receipt,job['mode'])
        schedule=Schedule(plan['rate'],plan['warmup_steps'],0,.1)
        training=dict(batch_size=plan['batch_size'],rate=plan['rate'],clip=plan['clip'],
                      rate_scales=plan['rate_scales'],schedule=schedule.contract(),epsilon=plan['epsilon'])
        checkpoint=out/'continuation.npz'
        if args.resume_only:
            previous=load_checkpoint(checkpoint,model,sampler,dataset_id=manifest['dataset_id'])
            assert previous['input_contract']==contract and previous['training_contract']==training
        else:
            actual={**model.checkpoint_arrays(),**{'port/'+k:v for k,v in ports.items()}}
            assert hashes(actual)==reference['arrays'],'Initial parameters, moments or ports differ'
            before=reference['metadata']
            old_config={**before['model_config'],'threads':config.threads}
            assert old_config==asdict(config) and sampler.state()==before['sampler']
            assert before['dataset_id']==manifest['dataset_id'] and before['graph_id']==graph['manifest']['graph_id']
            assert training==before['training_contract'],'Reference training contract differs'
            batch=sampler.batch(plan['batch_size'])
            model.train_step(adapter.encode(batch[0],job['mode']),*batch[1:],rate=schedule.rate(1),
                clip=plan['clip'],rate_scales=plan['rate_scales'],epsilon=plan['epsilon'])
            save_checkpoint(model,sampler,checkpoint,dict(dataset_id=manifest['dataset_id'],
                input_contract=contract,training_contract=training),root=root)
        batch=sampler.batch(plan['batch_size'])
        batch_hash=hashes({str(i):v for i,v in enumerate(batch)})
        if args.resume_only:
            player,metadata=load_player(checkpoint,graph_path,threads=len(job['cpus']))
            assert metadata['input_contract']==contract
        predictions={}
        for size in (1,plan['batch_size']):
            direct=model.infer(adapter.encode(batch[0][:size],job['mode']),trace=True)
            if args.resume_only:
                restored=player.infer(batch[0][:size],trace=True)
                for key in direct:np.testing.assert_array_equal(direct[key],restored[key])
            predictions[str(size)]=hashes(direct)
        loss=model.train_step(adapter.encode(batch[0],job['mode']),*batch[1:],rate=schedule.rate(2),
            clip=plan['clip'],rate_scales=plan['rate_scales'],epsilon=plan['epsilon'])
        record=dict(batch=batch_hash,predictions=predictions,loss=loss,
                    arrays=hashes(model.checkpoint_arrays()),sampler=sampler.state())
        if args.resume_only:
            assert record==json.loads((out/'expected.json').read_text()),'Fresh continuation differs'
            atomic_json(out/'fresh-process.json',dict(status='passed',pid=os.getpid(),updated=time.time(),
                                                    runtime_sha256=runtime_hashes()))
        else:
            atomic_json(out/'expected.json',record)
            inference=root/plan['inference_environment']
            assert (inference/'snapshot.json').is_file(),'Inference source is not immutable'
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--resume-only'],check=True,
                           env={**os.environ,'PYTHONPATH':str(inference/'site-packages')})
            fresh=json.loads((out/'fresh-process.json').read_text());assert fresh['pid']!=os.getpid()
            for key in ('native','fly.py','optimizer.py','train.py','vision.py','jax/model.py'):
                assert runtime_hashes()[key]==fresh['runtime_sha256'][key],'Numerical implementation changed'
            atomic_json(out/'result.json',dict(status='passed',seed=args.seed,created=time.time(),
                reference_sha256=sha256(args.reference),reference_checkpoint=reference['receipt'],
                source_sha256=sha256(Path(__file__)),plan_sha256=sha256(args.plan),runtime_sha256=runtime_hashes(),
                input_contract=contract,model_config=asdict(config),training_contract=training,
                initial_arrays=len(reference['arrays']),fresh_process=fresh,
                scope='Paired initial arrays/ports/sampler, B1/B32 full states and predictions through the Go player, '
                      'and exact next batch/loss/parameters/moments after fresh-process checkpoint recovery. '
                      'Two engineering updates on training examples, not a scientific endpoint. CPU only.'))
            print(json.dumps(dict(status='passed',seed=args.seed,output=str(out))),flush=True)


if __name__=='__main__':main()
