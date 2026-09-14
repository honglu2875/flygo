#!/usr/bin/env python3
"""Run a frozen TPU study with one immutable SPMD cohort at a time."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from cluster import HOSTS,remote,snapshot
from flygo.data.corpus import atomic_json
from flygo.runtime import pin
from flygo.storage import GIB,StorageBudget


def inspect(root,run_id):
    code=f'''import pathlib,json
p=pathlib.Path({str(root/'runs'/run_id)!r})
s=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {{}}
c=json.loads((p/'config.json').read_text()) if (p/'config.json').exists() else {{}}
job=json.loads((p/'job.json').read_text()) if (p/'job.json').exists() else {{}}
pid=job.get('pid');command=[]
if pid:
 try:command=pathlib.Path('/proc',str(pid),'cmdline').read_bytes().decode().split('\\0')
 except (FileNotFoundError,ProcessLookupError):pass
print(json.dumps(dict(exists=p.exists(),status=s.get('status'),error=s.get('error'),
                     live=str(p/'worker.py') in command,training=c.get('training'))))'''
    with ThreadPoolExecutor(4) as pool:
        return [json.loads(s) for s in pool.map(lambda host:remote(host,code),HOSTS)]


def cases_for(plan):
    """Explicit cases support later controlled studies without a second runner."""
    if 'cases' in plan:
        cases=[dict(case,training=dict(case['training'])) for case in plan['cases']]
    else:
        cases=[]
        for family in ('cnn','fly'):
            for seed in plan['seeds']:
                for rate in plan['rates']:
                    encoded=f'{rate:g}'
                    tag=encoded[2:] if encoded.startswith('0.') else encoded.replace('.','p')
                    run_id=f"tpu-{'cnn-' if family=='cnn' else ''}v0-b{plan['batch_size']}-lr{tag}-s{seed}"
                    training=dict(model=family,release=plan['release'],passes=plan['fly']['passes'],groups=plan['fly']['groups'],
                        seed=seed,rate=rate,clip=plan['clip'],batch_size=plan['batch_size'],updates=plan['updates'],
                        eval_every=plan['evaluation']['every'],eval_batch_size=plan['batch_size'],
                        eval_positions=plan['evaluation']['positions'],checkpoint_every=64)
                    if family=='cnn':training.update(channels=plan['cnn']['channels'],blocks=plan['cnn']['blocks'])
                    cases.append(dict(run_id=run_id,training=training))
    seen=set()
    for case in cases:
        name=case['run_id']
        if not name or name in ('.','..') or Path(name).name!=name or name in seen:
            raise ValueError('Every cohort needs a unique plain run ID')
        seen.add(name)
    if not cases:raise ValueError('A study needs at least one cohort')
    return cases


def worker(path):
    job=json.loads(path.read_text());root=Path(job['root']);out=path.parent
    pin([116])
    with (out/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results=[]
        try:
            for case in job['cases']:
                if (out/'stop').exists():
                    atomic_json(out/'status.json',dict(state='stopped',results=results));return
                run_id=case['run_id'];plan=case['training']
                state=inspect(root,run_id)
                if any(s['exists'] for s in state):
                    if not all(s['exists'] for s in state):raise ValueError('Partially deployed cohort needs inspection: '+run_id)
                    for s in state:
                        actual=s['training'] or {}
                        defaults=dict(model='fly')
                        if any(actual.get(k,defaults.get(k))!=v for k,v in plan.items()):
                            raise ValueError('Existing trial differs from registered settings: '+run_id)
                else:
                    command=[sys.executable,'-B',str(out/'code/tpu.py'),'launch','--root',str(root),'--run-id',run_id,
                             '--mode','train','--train-plan',case['plan_path'],'--source',job['source'],
                             '--runtime',job['runtime'],'--worker',str(out/'code/tpu_worker.py')]
                    # Deployment hashes/copies on spare cores; the coordinator
                    # returns to the single housekeeping core while waiting.
                    with (out/(run_id+'.launch.log')).open('a') as log:
                        subprocess.run(['taskset','--cpu-list','117,118,119',*command],stdout=log,stderr=subprocess.STDOUT,check=True)
                started=time.time()
                while True:
                    state=inspect(root,run_id)
                    atomic_json(out/'status.json',dict(state='running',pid=os.getpid(),run=run_id,
                        hosts=state,completed=len(results),updated=time.time()))
                    if any(s['status']=='failed' for s in state):raise RuntimeError('Retained failed cohort: '+run_id)
                    if all(s['status']=='passed' and not s['live'] for s in state):break
                    if time.time()-started>7200:raise TimeoutError('Bounded cohort exceeded two hours: '+run_id)
                    time.sleep(20)
                learner=root/'runs'/(run_id+'-learner')
                records=[json.loads(line) for line in (learner/'metrics.jsonl').read_text().splitlines()]
                result=dict(run_id=run_id,training=plan,
                    validation=[r for r in records if r['kind']=='validation'][-1],
                    status=json.loads((learner/'status.json').read_text()),
                    checkpoint=json.loads((learner/'latest.json').read_text()),
                    source=json.loads((root/'runs'/run_id/'config.json').read_text())['source'])
                if result['status']['state']!='complete' or result['status']['step']!=plan['updates']:
                    raise ValueError('Incomplete exposure horizon: '+run_id)
                results.append(result)
                atomic_json(out/'results.json',dict(status='running',results=results))
            atomic_json(out/'results.json',dict(status='complete',results=results))
            atomic_json(out/'status.json',dict(state='complete',completed=len(results),updated=time.time()))
        except BaseException as error:
            atomic_json(out/'status.json',dict(state='failed',error=repr(error),completed=len(results),updated=time.time()))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/matched-control-v1.json'))
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--source',type=Path,help='Reuse a published source environment across related studies')
    p.add_argument('--runtime',type=Path,help='Reuse an already qualified TPU dependency runtime')
    p.add_argument('--worker',type=Path)
    args=p.parse_args()
    if args.worker:return worker(args.worker)
    pin([117,118,119])
    plan=json.loads(args.config.read_text());root=args.root;out=root/'runs'/plan['name']
    if Path(plan['name']).name!=plan['name']:raise ValueError('Study name must be a path component')
    cases=cases_for(plan)
    for gate in ('tpu-cnn-parity-v3','tpu-v0-trainer-restore-v1',*plan.get('require_cohorts',[])):
        if not all(s['status']=='passed' and not s['live'] for s in inspect(root,gate)):
            raise ValueError('Numerical gate is not complete: '+gate)
    from tpu import environment
    source=args.source or snapshot(root);runtime=args.runtime or environment(root,base=source)
    if not (source/'snapshot.json').is_file() or not (runtime/'runtime.json').is_file():
        raise ValueError('Study source and runtime must be published immutable environments')
    with StorageBudget(root).reserve(files=1<<20,heap=GIB,purpose='freeze matched TPU study coordinator'):
        (out/'code').mkdir(parents=True,exist_ok=False)
        for name in ('tpu_study.py','tpu.py','tpu_worker.py','cluster.py'):
            shutil.copyfile(Path(__file__).with_name(name),out/'code'/name)
        atomic_json(out/'registration.json',plan)
        for case in cases:
            path=out/(case['run_id']+'.json');atomic_json(path,case['training'])
            case['plan_path']=str(path)
        path=out/'queue.json';atomic_json(path,dict(root=str(root),source=str(source),runtime=str(runtime),cases=cases))
        with (out/'coordinator.log').open('a') as log:
            process=subprocess.Popen(['taskset','--cpu-list','116',sys.executable,'-B',str(out/'code/tpu_study.py'),
                '--worker',str(path)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'PYTHONPATH':str(source/'site-packages'),'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1'})
        atomic_json(out/'coordinator.json',dict(pid=process.pid,source=str(source),runtime=str(runtime)))
        print(json.dumps(dict(output=str(out),pid=process.pid,cases=len(cases))))


if __name__=='__main__':main()
