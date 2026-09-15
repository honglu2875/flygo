#!/usr/bin/env python3
"""Run the registered clipping transitions and fresh-process recovery for one seed."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import StorageBudget,GIB


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    args=p.parse_args();root=args.root;base=args.plan.parent
    pin([117,118,119]);plan=json.loads(args.plan.read_text())
    if args.seed not in plan['seeds']:raise ValueError('Unregistered qualification seed')
    scripts=Path(__file__).parent;out=base/f'seed-{args.seed}'
    with StorageBudget(root).reserve(files=4<<20,heap=GIB,purpose='clipping qualification coordinator'):
        out.mkdir(parents=True,exist_ok=False);records=[]
        def run(command,arm,protocol):
            log=out/(arm+'-'+protocol+'.log')
            with log.open('xb') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
            row=dict(arm=arm,protocol=protocol,returncode=result.returncode,command=command,log_sha256=sha256(log))
            records.append(row)
            atomic_json(out/'status.json',dict(state='qualifying',records=records,updated=time.time()))
            return row
        for arm in plan['arms']:
            paths=[]
            for protocol in plan['protocols']:
                target=out/arm/protocol['name']
                command=['taskset','--cpu-list',','.join(map(str,plan['cpus'])),sys.executable,'-B',str(scripts/'qualify_ports.py'),
                    '--root',str(root),'--input-map',str(root/plan['input_map']),'--head-mask',str(root/plan['head_mask']),
                    '--input-modes',plan['input_mode'],'--seed',str(args.seed),'--passes',str(plan['steps']),
                    '--batch-size',str(plan['batch_size']),'--rate',str(plan['rate']),'--epsilon',str(plan['epsilon']),
                    '--rate-scales',json.dumps(plan['rate_scales']),'--clip-mode',arm,
                    '--cpus',','.join(map(str,plan['cpus'])),'--protocol',protocol['name'],
                    '--numerical-plan',str(args.plan),'--output',str(target)]
                row=run(command,arm,protocol['name'])
                if row['returncode']==0:
                    path=target/'result.json';report=json.loads(path.read_text())
                    if (report['status']!='complete' or report['numerical_protocol']!=protocol['name']
                            or report['numerical_plan_sha256']!=sha256(args.plan)):
                        raise ValueError('Unexpected numerical report')
                    row['result_sha256']=sha256(path);paths.append(path)
            if len(paths)!=len(plan['protocols']):continue
            reports=[json.loads(path.read_text()) for path in paths]
            for key in ('runtime_sha256','dataset_id','graph_id'):
                if any(r[key]!=reports[0][key] for r in reports):raise ValueError('Protocol sources differ')
            combined={k:reports[0][k] for k in ('runtime_sha256','dataset_id','graph_id','cpus')}
            combined.update(status='complete',created=time.time(),numerical_protocol='combined-v2',
                records=[r['records'][0] for r in reports],numerical_plan_sha256=sha256(args.plan),
                constituent_reports=[dict(path=str(path),sha256=sha256(path)) for path in paths],
                source_sha256=sha256(Path(__file__)),qualifier_sha256=sha256(scripts/'qualify_ports.py'),
                scope='Three matched-checkpoint peak transitions and three independent actual-warmup updates, including clipping statistics. Engineering only; no TPU.')
            atomic_json(out/arm/'result.json',combined)
        for arm in plan['arms']:
            if not (out/arm/'result.json').is_file():continue
            target=base/'io'/f'seed-{args.seed}'/arm
            command=['taskset','--cpu-list','92-115',sys.executable,'-B',str(scripts/'qualify_head_io.py'),
                '--root',str(root),'--plan',str(base/'io-plan.json'),'--seed',str(args.seed),'--arm',arm,'--output',str(target)]
            row=run(command,arm,'exact-recovery')
            if row['returncode']==0:row['result_sha256']=sha256(target/'result.json')
        passed=len(records)==len(plan['arms'])*(len(plan['protocols'])+1) and all(x['returncode']==0 for x in records)
        atomic_json(out/'worker-result.json',dict(status='passed' if passed else 'failed',records=records,
            created=time.time(),seed=args.seed,source_sha256=sha256(Path(__file__))))
        atomic_json(out/'status.json',dict(state='complete' if passed else 'failed',updated=time.time()))
        print(json.dumps(dict(seed=args.seed,status='passed' if passed else 'failed',checks=len(records))),flush=True)
        if not passed:raise SystemExit(1)


if __name__=='__main__':main()
