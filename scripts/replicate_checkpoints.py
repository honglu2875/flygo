#!/usr/bin/env python3
"""Star-topology checkpoint replicas: the first host already has trusted SSH to all peers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import uuid

from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.replication import replicate_file
from flygo.runtime import pin
from flygo.storage import StorageBudget,GIB

SSH=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10']


def pull(root,host,run_id):
    peer=f'cubic27@t1v-n-a09f5679-w-{host}'
    directory=root/'runs'/run_id/'checkpoints'
    code='import pathlib,json; p=pathlib.Path('+repr(str(directory))+');print(json.dumps([dict(path=str(f),receipt=json.loads(f.with_suffix(".json").read_text())) for f in p.glob("step-*.npz") if f.with_suffix(".json").exists()]))'
    records=json.loads(subprocess.check_output(SSH+[peer,'python3 -c '+shlex.quote(code)],text=True,timeout=30))
    copied=[]
    for record in records:
        path=Path(record['path']);receipt=record['receipt']
        if path.exists():
            if sha256(path)!=receipt['sha256']:raise ValueError('Checkpoint replica conflict')
            expected={**receipt,'peer':'t1v-n-a09f5679-w-0','origin_host':f't1v-n-a09f5679-w-{host}','replica_status':'verified'}
            receipt_path=path.with_suffix('.json')
            if not receipt_path.exists():
                atomic_json(receipt_path,expected)
            if receipt.get('replica_status')!='verified':
                code='from pathlib import Path;p=Path('+repr(str(receipt_path))+');t=p.with_suffix(".json.replicating");t.write_text('+repr(json.dumps(expected,indent=2)+'\n')+');t.replace(p)'
                subprocess.run(SSH+[peer,'python3 -c '+shlex.quote(code)],check=True,timeout=30)
            continue
        with StorageBudget(root).reserve(files=receipt['bytes'],heap=4*1024**2,purpose='checkpoint pull '+run_id):
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.incoming')
            try:
                subprocess.run(['rsync','-a','-e',shlex.join(SSH),peer+':'+str(path),str(temp)],check=True,timeout=120)
                if sha256(temp)!=receipt['sha256']:raise ValueError('Checkpoint pull hash mismatch')
                os.link(temp,path)
            finally:temp.unlink(missing_ok=True)
            receipt={**receipt,'peer':'t1v-n-a09f5679-w-0','origin_host':f't1v-n-a09f5679-w-{host}','replica_status':'verified'}
            atomic_json(path.with_suffix('.json'),receipt)
            code='from pathlib import Path;p=Path('+repr(str(path.with_suffix('.json')))+');t=p.with_suffix(".json.replicating");t.write_text('+repr(json.dumps(receipt,indent=2)+'\n')+');t.replace(p)'
            subprocess.run(SSH+[peer,'python3 -c '+shlex.quote(code)],check=True,timeout=30)
            copied.append(str(path))
    return copied


def push_local(root,run_id):
    copied=[]
    for path in sorted((root/'runs'/run_id/'checkpoints').glob('step-*.npz')):
        receipt_path=path.with_suffix('.json')
        if not receipt_path.exists():continue
        receipt=json.loads(receipt_path.read_text())
        if receipt.get('replica_status')=='verified':continue
        peer='cubic27@t1v-n-a09f5679-w-1'
        replicate_file(path,root,peer)
        receipt.pop('replica_error',None)
        receipt.update(peer=peer,replica_status='verified',origin_host='t1v-n-a09f5679-w-0')
        atomic_json(receipt_path,receipt)
        try:
            replicate_file(receipt_path,root,peer)
        except (OSError,RuntimeError,subprocess.SubprocessError) as error:
            receipt.update(replica_status='retry',replica_error=str(error))
            atomic_json(receipt_path,receipt)
            raise
        copied.append(str(path))
    return copied


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--jobs',type=Path,help='Explicit jobs.json from deploy_research.py')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();pin([116])
    out=args.output or args.root/'runs/m5/replica-coordinator';out.mkdir(parents=True,exist_ok=True)
    jobs=json.loads(args.jobs.read_text()) if args.jobs else [dict(host=h,run_id=f'fly-baseline-k{k}-seed1') for h,k in [(1,2),(2,4),(3,16)]]
    while not (out/'stop').exists():
        results=[]
        for job in jobs:
            host,run_id=job['host'],job['run_id']
            try:
                copied=pull(args.root,host,run_id) if host else push_local(args.root,run_id)
                results.append(dict(host=host,run_id=run_id,copied=copied,status='passed'))
            except Exception as error:results.append(dict(host=host,run_id=run_id,status='retry',error=repr(error)))
        atomic_json(out/'status.json',dict(pid=os.getpid(),updated=time.time(),hosts=results))
        if args.once:break
        time.sleep(15)


if __name__=='__main__':main()
