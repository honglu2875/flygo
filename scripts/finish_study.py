#!/usr/bin/env python3
"""Finish one frozen CPU study: wait for each learner, then validate and play fresh panels."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


def running_train(run_id):
    """Check exact argv fields, including after a trainer publishes its final status."""
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            command=(path/'cmdline').read_bytes().decode().split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            continue
        if 'flygo.train' in command and '--run-id' in command:
            index=command.index('--run-id')
            if index+1<len(command) and command[index+1]==run_id:
                return int(path.name)
    return None


def worker(args):
    from flygo.data.corpus import atomic_json
    from flygo.qualify import sha256
    from flygo.runtime import pin
    config=json.loads(args.config.read_text())
    job=config['job'];plan=config['plan'];root=Path(config['root'])
    source=root/'runs'/job['run_id'];out=args.config.parent
    # Waiting uses an explicitly shared housekeeping core; heavy children only
    # acquire the training lane after the owning trainer has exited.
    pin([116])
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            checkpoint=source/'checkpoints'/f"step-{plan['updates']:08d}.npz"
            while not (out/'stop').exists():
                status=json.loads((source/'status.json').read_text())
                if status.get('state') in ('failed','stopped'):
                    raise RuntimeError('Source trainer did not complete: '+str(status))
                complete=status.get('state')=='complete' and status.get('step')==plan['updates']
                if complete and not running_train(job['run_id']):
                    receipt=json.loads(checkpoint.with_suffix('.json').read_text())
                    if receipt.get('replica_status')=='verified':
                        break
                atomic_json(out/'status.json',dict(state='waiting_training_and_replica',pid=os.getpid(),
                            source_step=status.get('step'),updated=time.time()))
                time.sleep(20)
            else:
                atomic_json(out/'status.json',dict(state='stopped',updated=time.time()))
                return
            original=json.loads((source/'config.json').read_text())
            expected=dict(passes=job['passes'],groups=job['groups'],seed=job['seed'],
                          steps=plan['updates'],batch_size=plan['batch_size'],rate=job.get('rate',plan['rate']),release=plan['release'],
                          rate_scales=job.get('rate_scales',plan.get('rate_scales',{})))
            if any(original['arguments'].get(k,{} if k=='rate_scales' else None)!=v for k,v in expected.items()):
                raise ValueError('Source trial differs from frozen study contract')
            if sha256(checkpoint)!=receipt['sha256']:
                raise ValueError('Final checkpoint checksum failed')
            atomic_json(out/'selection.json',dict(checkpoint=str(checkpoint),receipt=receipt,
                        rule='Final checkpoint at the predeclared exposure horizon; no validation selection'))
            cpus=','.join(map(str,job['cpus']));threads=str(len(job['cpus']))
            common=['--root',str(root),'--threads',threads,'--cpus',cpus]
            stages=[('validation',[sys.executable,str(out/'compare_checkpoints.py'),*common,
                    '--release',plan['release'],'--checkpoints',str(checkpoint),'--output',str(out/'validation')])]
            for name,simulations in [('prior',0),('puct',16),('gumbel',16)]:
                stages.append((name,[sys.executable,'-m','flygo.evaluate',*common,'--checkpoint',str(checkpoint),
                              '--opponent','0','--games','16','--seed','229101','--simulations',str(simulations),
                              '--search','gumbel' if name=='gumbel' else 'puct','--output',str(out/name)]))
            for name,command in stages:
                if (out/'stop').exists():
                    atomic_json(out/'status.json',dict(state='stopped',updated=time.time()))
                    return
                result=out/name/'result.json'
                if result.exists():
                    previous=json.loads(result.read_text())
                    if previous.get('status') in ('complete','passed'):
                        continue
                    raise ValueError('Incomplete stage retained; use a new attempt: '+name)
                atomic_json(out/'status.json',dict(state=name,pid=os.getpid(),cpus=job['cpus'],updated=time.time()))
                with (out/(name+'.log')).open('a') as log:
                    subprocess.run(['taskset','--cpu-list',cpus,*command],stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,check=True)
            atomic_json(out/'status.json',dict(state='complete',source=job['run_id'],
                        checkpoint_sha256=receipt['sha256'],updated=time.time()))
        except BaseException as error:
            atomic_json(out/'status.json',dict(state='failed',pid=os.getpid(),error=repr(error),updated=time.time()))
            raise


def launch(args):
    from cluster import snapshot,SSH,PYTHON,remote
    from flygo.data.corpus import atomic_json
    from flygo.replication import replicate_bundle
    root=args.root
    plan=json.loads(args.plan.read_text())
    out=root/'runs'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    environment=snapshot(root)
    atomic_json(out/'plan.json',plan)
    helper=Path(__file__).resolve()
    compare=helper.with_name('compare_checkpoints.py')
    records=[]
    for job in plan['jobs']:
        directory=out/job['run_id'];directory.mkdir()
        config=dict(schema_version=1,root=str(root),plan={k:v for k,v in plan.items() if k!='jobs'},job=job,
                    environment=str(environment),helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
                    comparator_sha256=hashlib.sha256(compare.read_bytes()).hexdigest())
        atomic_json(directory/'config.json',config)
        shutil.copyfile(helper,directory/'worker.py')
        shutil.copyfile(compare,directory/'compare_checkpoints.py')
        records.append(dict(**job,config=str(directory/'config.json')))
    def deploy(host):
        peer=f'cubic27@t1v-n-a09f5679-w-{host}'
        jobs=[r for r in records if r['host']==host]
        if host:
            files=[p for p in environment.rglob('*') if p.is_file()]
            files.extend(p for j in jobs for p in Path(j['config']).parent.iterdir() if p.is_file())
            replicate_bundle(files,root,peer)
        result=[]
        for job in jobs:
            directory=Path(job['config']).parent
            code=f'''import subprocess,os,pathlib,json
p=pathlib.Path({str(directory)!r})
env=dict(os.environ,PYTHONPATH={str(environment/'site-packages')!r},PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
with (p/'worker.log').open('a') as log:
 child=subprocess.Popen([{PYTHON!r},str(p/'worker.py'),'worker','--config',str(p/'config.json')],env=env,
      stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid)))'''
            record={**job,**json.loads(remote(f't1v-n-a09f5679-w-{host}',code))}
            result.append(record)
        return result
    with ThreadPoolExecutor(4) as pool:
        launched=sum(pool.map(deploy,sorted({j['host'] for j in records})),[])
    atomic_json(out/'jobs.json',launched)
    print(json.dumps(dict(output=str(out),jobs=launched),indent=2))


def collect(args):
    from cluster import SSH,remote
    from flygo.data.corpus import atomic_json
    from flygo.storage import StorageBudget,GIB
    root=args.root;out=root/'runs'/args.run_id
    jobs=json.loads((out/'jobs.json').read_text())
    results=[]
    for job in jobs:
        directory=Path(job['config']).parent
        host=job['host']
        if host:
            with StorageBudget(root).reserve(files=64*(1<<20),heap=1<<20,purpose='finished study evidence copy'):
                subprocess.run(['rsync','-a','--checksum','-e',shlex.join(SSH),
                                f'cubic27@t1v-n-a09f5679-w-{host}:{directory}/',str(directory)+'/'],check=True)
        status=json.loads((directory/'status.json').read_text()) if (directory/'status.json').exists() else {}
        records={name:json.loads((directory/name/'result.json').read_text()) for name in ('validation','prior','puct','gumbel')
                 if (directory/name/'result.json').exists()}
        results.append(dict(run=job['run_id'],host=host,status=status,results=records))
    atomic_json(out/'summary.json',dict(updated=time.time(),records=results))
    print(json.dumps([dict(run=r['run'],state=r['status'].get('state'),stages=list(r['results'])) for r in results],indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('launch','worker','collect'))
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--run-id',default='prototype-followup-v1')
    p.add_argument('--plan',type=Path,default=Path('configs/prototype-v1.json'))
    p.add_argument('--config',type=Path)
    args=p.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in ('.','..'):
        p.error('Run ID must be a plain path component')
    {'launch':launch,'worker':worker,'collect':collect}[args.action](args)


if __name__=='__main__':
    main()
