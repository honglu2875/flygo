#!/usr/bin/env python3
"""Check actual masked-head initialization, Go inference, arithmetic and fresh recovery."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from flygo.attachments import load_attachment,input_contract,require_qualification,runtime_hashes
from flygo.checkpoint import save_checkpoint,load_checkpoint
from flygo.cost import fly_cost
from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release,Sampler
from flygo.fly import FlyConfig,RustFly,load_graph
from flygo.play import load_player
from flygo.qualify import sha256
from flygo.readout import load_head_mask
from flygo.runtime import pin
from flygo.schedule import Schedule
from flygo.storage import GIB,StorageBudget
from qualify_attachment_io import hashes
from count_inference import count_edges


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--arm',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-only',action='store_true')
    args=p.parse_args();root=args.root;out=args.output;plan=json.loads(args.plan.read_text())
    settings=dict(steps=8,batch_size=32,updates=3,rate=.03,epsilon=1e-6,clip=1,rate_scales={'bias':.01})
    if any(plan[key]!=value for key,value in settings.items()):raise ValueError('Qualification plan settings differ')
    out.resolve().relative_to(root.resolve());pin(list(range(92,116)))
    arm=next(a for a in plan['arms'] if a['arm']==args.arm)
    if args.seed not in [j['seed'] for j in plan['jobs']]:raise ValueError('Unregistered qualification seed')
    with StorageBudget(root).reserve(files=16<<20,heap=24*GIB,purpose='masked head IO qualification'):
        if not args.resume_only:out.mkdir(parents=True,exist_ok=False)
        graph_path=Path(json.loads((root/'runs/m4/graph.json').read_text())['path']);graph=load_graph(graph_path)
        manifest,arrays,indexes=load_release(root,root/'releases/v0-1m/manifest.json',cache=True)
        adapter,ports,receipt=load_attachment(root/plan['input_map'],graph_id=graph['manifest']['graph_id'],dataset_id=manifest['dataset_id'])
        if receipt['sha256']!=plan['input_map_sha256'] or sha256(root/arm['path'])!=arm['sha256']:
            raise ValueError('Input or head artifact differs from the plan')
        mask=load_head_mask(root/arm['path'],graph_id=graph['manifest']['graph_id'],attachment_sha256=receipt['sha256'])
        cfg=FlyConfig(steps=8,features=adapter.features,groups=receipt['groups'],seed=args.seed,threads=24)
        contract=input_contract(receipt,'current');schedule=Schedule(.03,100,0,.1)
        training=dict(batch_size=32,rate=.03,clip=1,rate_scales={'bias':.01},epsilon=1e-6,schedule=schedule.contract())
        qualification=args.plan.parent/f'seed-{args.seed}'/args.arm/'result.json'
        require_qualification(qualification,contract,cfg,batch_size=32,rate=.03,epsilon=1e-6,clip=1,rate_scales={'bias':.01},head_mask=mask)
        model=RustFly(graph,cfg,ports=ports,head_mask=mask);sampler=Sampler(arrays,indexes,args.seed)
        checkpoint=out/'continuation.npz';control=None
        def update(m,batch,step):return m.train_step(adapter.encode(batch[0],'current'),*batch[1:],
            rate=schedule.rate(step),clip=1,rate_scales={'bias':.01},epsilon=1e-6)
        def common_state(m):
            return hashes({**{k:v for k,v in m.checkpoint_arrays().items() if not k.startswith('readout_mask/')},
                           **{'port/'+k:v for k,v in ports.items()}})
        def compare_control(batch,step,metrics):
            expected=update(control,batch,step);left,right=common_state(model),common_state(control)
            differing=[key for key in left if left[key]!=right[key]]
            if expected!=metrics or differing:
                atomic_json(out/'control-mismatch.json',dict(step=step,masked=metrics,unmasked=expected,
                    differing_arrays=differing,masked_arrays=left,unmasked_arrays=right))
                raise AssertionError('All-enabled update differs from the unmasked control; see control-mismatch.json')
        if args.resume_only:
            metadata=load_checkpoint(checkpoint,model,sampler,dataset_id=manifest['dataset_id'])
            if metadata['input_contract']!=contract or metadata['training_contract']!=training:
                raise ValueError('Saved training or input contract differs')
        else:
            initial=dict(arrays=common_state(model),sampler=sampler.state(),model_config=asdict(cfg),dataset_id=manifest['dataset_id'])
            atomic_json(out/'initial.json',initial)
            if args.arm=='dense':control=RustFly(graph,cfg,ports=ports)
            for step in range(1,4):
                batch=sampler.batch(32)
                if control is not None:
                    encoded=adapter.encode(batch[0],'current')
                    for key,value in model.infer(encoded,trace=True).items():np.testing.assert_array_equal(value,control.infer(encoded,trace=True)[key])
                    loss,grad=model.loss_and_grad(encoded,*batch[1:]);other_loss,other_grad=control.loss_and_grad(encoded,*batch[1:])
                    if loss!=other_loss:raise AssertionError('All-enabled loss differs from the unmasked control')
                    for key,value in grad.items():np.testing.assert_array_equal(value,other_grad[key])
                metrics=update(model,batch,step)
                if control is not None:
                    compare_control(batch,step,metrics)
            save_checkpoint(model,sampler,checkpoint,dict(dataset_id=manifest['dataset_id'],input_contract=contract,
                training_contract=training),root=root)
        batch=sampler.batch(32);predictions={};arithmetic={};degree=np.bincount(graph['src'],minlength=len(graph['type_id']))
        if args.resume_only:player,_=load_player(checkpoint,graph_path,threads=24)
        for size in (1,32):
            features=batch[0][:size];encoded=adapter.encode(features,'current')
            model.infer(encoded)  # Prime the fixed-parameter first-message cache.
            direct=model.infer(encoded,trace=True)
            if args.resume_only:
                for key,value in direct.items():np.testing.assert_array_equal(value,player.infer(features,trace=True)[key])
            predictions[str(size)]=hashes(direct)
            nominal=fly_cost(len(degree),len(graph['src']),int(np.count_nonzero(ports['input_index']>=0)),
                passes=8,groups=cfg.groups,batch_size=size,readout_neurons=cfg.groups,head_mask=mask)
            external=9*len(adapter.renderer.index)+324+1/size
            passes=count_edges(direct['states'],degree)
            warm=nominal['arithmetic_flops']-nominal['parts']['sparse_aggregation']+external+2*sum(x['edge_multiply_adds_per_position'] for x in passes[1:])
            arithmetic[str(size)]=dict(nominal=nominal,external_visual_encoding=external,passes=passes,warm_flops_per_position=warm)
        metrics=update(model,batch,4)
        if control is not None:
            compare_control(batch,4,metrics)
        record=dict(batch=hashes({str(i):v for i,v in enumerate(batch)}),predictions=predictions,arithmetic=arithmetic,
            loss=metrics,arrays=hashes(model.checkpoint_arrays()),sampler=sampler.state())
        if args.resume_only:
            if record!=json.loads((out/'expected.json').read_text()):raise AssertionError('Fresh-process state, prediction, arithmetic, sampler or next update differs')
            atomic_json(out/'fresh-process.json',dict(status='passed',pid=os.getpid(),runtime_sha256=runtime_hashes()))
        else:
            atomic_json(out/'expected.json',record)
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--resume-only'],check=True)
            fresh=json.loads((out/'fresh-process.json').read_text())
            if fresh['pid']==os.getpid() or fresh['runtime_sha256']!=runtime_hashes():raise AssertionError('Recovery must use a fresh process and identical runtime')
            atomic_json(out/'result.json',dict(status='passed',created=time.time(),seed=args.seed,arm=args.arm,
                source_sha256=sha256(Path(__file__)),helper_sha256={name:sha256(Path(__file__).with_name(name)) for name in ('qualify_attachment_io.py','count_inference.py')},
                plan_sha256=sha256(args.plan),qualification_sha256=sha256(qualification),runtime_sha256=runtime_hashes(),
                head_mask=mask.contract,input_contract=contract,model=asdict(cfg),training_contract=training,
                initial_sha256=sha256(out/'initial.json'),expected_sha256=sha256(out/'expected.json'),fresh_process=fresh,
                checkpoint=json.loads(checkpoint.with_suffix('.json').read_text()),arithmetic=arithmetic,
                engineering_updates=9 if control is not None else 5,
                scope='Common initial arrays/sampler are retained for cross-arm pairing. Exact B1/B32 Go-player states/predictions, arithmetic and next update after fresh recovery. Dense arm additionally matches an unmasked control through four updates. Every update uses B32 training examples; none is a scientific endpoint. CPU only.'))
            print(json.dumps(dict(status='passed',seed=args.seed,arm=args.arm,output=str(out))),flush=True)


if __name__=='__main__':main()
