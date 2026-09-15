#!/usr/bin/env python3
"""Compare a candidate global optimizer with a separately loaded legacy runtime."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from flygo.attachments import load_attachment,input_contract,runtime_hashes
from flygo.checkpoint import load_checkpoint,save_checkpoint
from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release,Sampler
from flygo.fly import FlyConfig,RustFly,load_graph
from flygo.play import load_player
from flygo.qualify import sha256
from flygo.readout import load_head_mask
from flygo.runtime import pin
from flygo.schedule import Schedule
from flygo.storage import StorageBudget,GIB
from qualify_attachment_io import hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('legacy','candidate'),required=True)
    args=p.parse_args();root=args.root;plan=json.loads(args.plan.read_text());out=args.output/args.phase
    pin(plan['cpus']);out.resolve().relative_to(root.resolve())
    if runtime_hashes()!=plan['runtime_sha256'][args.phase]:raise ValueError('Wrong runtime for regression phase')
    with StorageBudget(root).reserve(files=(1 if args.phase=='legacy' else 0)*GIB+(16<<20),heap=24*GIB,
                                    purpose='exact global optimizer and legacy checkpoint regression'):
        out.mkdir(parents=True,exist_ok=False);started=time.time()
        graph_path=Path(json.loads((root/'runs/m4/graph.json').read_text())['path']);graph=load_graph(graph_path)
        manifest,arrays,indexes=load_release(root,root/'releases/v0-1m/manifest.json',cache=True)
        adapter,ports,receipt=load_attachment(root/plan['input_map'],graph_id=graph['manifest']['graph_id'],dataset_id=manifest['dataset_id'])
        if (manifest['dataset_id']!=plan['dataset_id'] or graph['manifest']['graph_id']!=plan['graph_id']
                or receipt['sha256']!=plan['input_map_sha256'] or sha256(root/plan['head_mask'])!=plan['head_mask_sha256']):
            raise ValueError('Regression data or attachments differ')
        mask=load_head_mask(root/plan['head_mask'],graph_id=graph['manifest']['graph_id'],attachment_sha256=receipt['sha256'])
        schedule=Schedule(plan['rate'],plan['warmup_steps']);cases=[]
        for seed in plan['seeds']:
            cfg=FlyConfig(steps=plan['steps'],features=adapter.features,groups=receipt['groups'],seed=seed,threads=len(plan['cpus']))
            for protocol in ('aligned-peak','free-warmup'):
                # Both native runtimes run freely. Checkpoint alignment belongs
                # to the independent JAX test, not this exact regression.
                model=RustFly(graph,cfg,ports=ports,head_mask=mask);sampler=Sampler(arrays,indexes,seed)
                row=dict(seed=seed,rate_protocol=protocol,model=asdict(cfg),initial=hashes(model.checkpoint_arrays()),updates=[])
                def transition(batch,step):
                    x=adapter.encode(batch[0],plan['input_mode'])
                    prediction={str(size):hashes(model.infer(x[:size],trace=True)) for size in (1,32)}
                    losses,grad=model.loss_and_grad(x,*batch[1:])
                    rate=plan['rate'] if protocol=='aligned-peak' else schedule.rate(step)
                    metrics=model.train_step(x,*batch[1:],rate=rate,clip=plan['clip'],epsilon=plan['epsilon'],rate_scales=plan['rate_scales'])
                    return dict(batch=hashes({str(i):v for i,v in enumerate(batch)}),predictions=prediction,
                        losses=losses,gradients=hashes(grad),metrics={k:metrics[k] for k in ('policy_loss','value_loss','gradient_norm','step')},
                        arrays=hashes(model.checkpoint_arrays()),sampler=sampler.state())
                for step in range(1,4):row['updates'].append(transition(sampler.batch(plan['batch_size']),step))
                if protocol=='free-warmup':
                    checkpoint=args.output/'legacy'/f'seed-{seed}.npz'
                    if args.phase=='legacy':
                        save_checkpoint(model,sampler,checkpoint,dict(dataset_id=manifest['dataset_id'],
                            input_contract=input_contract(receipt,plan['input_mode']),
                            training_contract=dict(batch_size=plan['batch_size'],rate=plan['rate'],clip=plan['clip'],
                                epsilon=plan['epsilon'],rate_scales=plan['rate_scales'],schedule=schedule.contract())),root=root)
                    else:
                        before=hashes(model.checkpoint_arrays());sampling=sampler.state()
                        metadata=load_checkpoint(checkpoint,model,sampler,dataset_id=manifest['dataset_id'],numerical_runtime=model.numerical_runtime)
                        if 'optimizer_clip_mode' in metadata:raise AssertionError('Expected an actual legacy checkpoint')
                        if before!=hashes(model.checkpoint_arrays()) or sampling!=sampler.state():
                            raise AssertionError('Legacy restored state differs from the independently repeated trajectory')
                        player,_=load_player(checkpoint,graph_path,threads=len(plan['cpus']))
                        raw=arrays['features'][indexes['train'][:32]]
                        expected=player.infer(raw,trace=True)
                        for key,value in model.infer(adapter.encode(raw,plan['input_mode']),trace=True).items():
                            np.testing.assert_array_equal(value,expected[key])
                        del player
                    row['continuation']=transition(sampler.batch(plan['batch_size']),4)
                cases.append(row)
                atomic_json(out/f'seed-{seed}-{protocol}.json',row)
                del model
        if args.phase=='candidate':
            expected=json.loads((args.output/'legacy/result.json').read_text())
            if expected['cases']!=cases:raise AssertionError('Candidate global arithmetic differs from the legacy runtime; compare retained per-case reports')
        atomic_json(out/'result.json',dict(status='passed',phase=args.phase,cases=cases,runtime_sha256=runtime_hashes(),
            helper_sha256=sha256(Path(__file__)),plan_sha256=sha256(args.plan),seconds=time.time()-started,
            labeled_update_exposures=len(plan['seeds'])*7*plan['batch_size'],
            scope='Exact states, losses, gradients, parameters, moments, sampler and updates at peak/warmup rates. '
                  'Candidate additionally restores actual legacy checkpoints and checks Go-player inference. Engineering only; no TPU.'))
        print(json.dumps(dict(status='passed',phase=args.phase,cases=len(cases))),flush=True)


if __name__=='__main__':main()
