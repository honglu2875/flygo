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
from flygo.replication import replicate_file,replicate_stream
from flygo.runtime import pin
from flygo.storage import StorageBudget,GIB

SSH=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10']


def relay(root,host,replica_host,run_id,*,timeout=180):
    """Copy between owned workers through root's pipes; reserve no payload file on root."""
    if (host not in (1,2,3) or replica_host not in (1,2,3) or host==replica_host
            or Path(run_id).name!=run_id or run_id in ('.','..')):
        raise ValueError('Relay needs distinct owned workers and a plain run ID')
    source=f'cubic27@t1v-n-a09f5679-w-{host}'
    destination=f'cubic27@t1v-n-a09f5679-w-{replica_host}'
    directory=root/'runs'/run_id/'checkpoints'
    inventory='''import pathlib,json
p=pathlib.Path(DIRECTORY)
print(json.dumps([dict(name=f.name,receipt=json.loads(f.with_suffix('.json').read_text()))
 for f in sorted(p.glob('step-*.npz')) if f.with_suffix('.json').is_file()]))
'''.replace('DIRECTORY',repr(str(directory)))
    records=json.loads(subprocess.check_output(SSH+[source,'python3 -c '+shlex.quote(inventory)],text=True,timeout=30))
    copied=[]
    for record in records:
        path=directory/record['name'];receipt=record['receipt']
        if path.parent!=directory or int(path.stem.removeprefix('step-'))!=receipt['step']:
            raise ValueError('Checkpoint inventory escaped its declared run')
        if receipt.get('replica_status')=='verified' and receipt.get('peer')==destination:
            continue
        producer_code='''import pathlib,sys,hashlib
p=pathlib.Path(PATH);h=hashlib.sha256();count=0
assert p.stat().st_size==SIZE,'Source checkpoint size changed'
with p.open('rb') as stream:
 for chunk in iter(lambda:stream.read(1024*1024),b''):
  sys.stdout.buffer.write(chunk);h.update(chunk);count+=len(chunk)
sys.stdout.buffer.flush()
assert count==SIZE and h.hexdigest()==DIGEST,'Source checkpoint checksum failed'
'''
        for key,value in dict(PATH=str(path),SIZE=receipt['bytes'],DIGEST=receipt['sha256']).items():
            producer_code=producer_code.replace(key,repr(value))
        with StorageBudget(root).reserve(files=0,heap=8<<20,purpose='checkpoint relay '+run_id):
            producer=subprocess.Popen(SSH+[source,'python3 -c '+shlex.quote(producer_code)],
                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                replicate_stream(producer.stdout,root,path.relative_to(root),destination,
                    size=receipt['bytes'],digest=receipt['sha256'],timeout=timeout)
                producer.stdout.close();producer.stdout=None
                _,errors=producer.communicate(timeout=timeout)
                if producer.returncode:
                    raise RuntimeError('Checkpoint source failed: '+errors.decode(errors='replace')[-2000:])
            finally:
                if producer.stdout is not None:producer.stdout.close();producer.stdout=None
                if producer.poll() is None:producer.kill()
                producer.communicate(timeout=10)
        verified={**receipt,'origin_host':source,'peer':destination,'replica_status':'verified',
                  'replica_method':'SHA-256 verified worker copy streamed through root; no root payload file'}
        verified.pop('replica_error',None)
        # Publish the destination receipt before marking the source recoverable.
        for peer in (destination,source):
            code='''import pathlib,json,os,uuid
p=pathlib.Path(PATH);record=RECORD
if p.exists():
 old=json.loads(p.read_text());assert old['sha256']==record['sha256'] and old['bytes']==record['bytes'],'Receipt conflict'
temporary=p.with_name(p.name+'.'+uuid.uuid4().hex+'.replicating')
temporary.write_text(json.dumps(record,indent=2)+chr(10));os.replace(temporary,p)
latest=p.parent.parent/'latest.json'
if latest.exists() and json.loads(latest.read_text()).get('sha256')==record['sha256']:
 temporary=latest.with_name(latest.name+'.'+uuid.uuid4().hex+'.replicating')
 temporary.write_text(json.dumps(record,indent=2)+chr(10));os.replace(temporary,latest)
'''.replace('PATH',repr(str(path.with_suffix('.json')))).replace('RECORD',repr(verified))
            subprocess.run(SSH+[peer,'python3 -c '+shlex.quote(code)],check=True,timeout=30)
        copied.append(str(path))
    return copied


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
                copied=(relay(args.root,host,job['replica_host'],run_id) if 'replica_host' in job else
                        pull(args.root,host,run_id) if host else push_local(args.root,run_id))
                results.append(dict(host=host,run_id=run_id,copied=copied,status='passed'))
            except Exception as error:results.append(dict(host=host,run_id=run_id,status='retry',error=repr(error)))
        atomic_json(out/'status.json',dict(pid=os.getpid(),updated=time.time(),hosts=results))
        if args.once:break
        time.sleep(15)


if __name__=='__main__':main()
