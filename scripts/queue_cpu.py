#!/usr/bin/env python3
"""Freeze a CPU study and start each host only after all its dependencies finish."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from cluster import PYTHON,remote
from flygo.data.corpus import atomic_json
from flygo.runtime import pin
from flygo.storage import GIB,StorageBudget


def dependencies(root,host,names):
    code=f'''import fcntl,json,pathlib
root=pathlib.Path({str(root)!r});records=[]
for name in {names!r}:
 p=root/'runs'/name
 s=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {{}}
 ready=s.get('state')=='complete'
 if ready:
  with (p/'worker.lock').open('a') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:ready=False
 records.append(dict(dependency=name,state=s.get('state'),ready=ready))
print(json.dumps(records))'''
    return json.loads(remote(f't1v-n-a09f5679-w-{host}',code))


def worker(path):
    config=json.loads(path.read_text());out=path.parent;root=Path(config['root'])
    plan=json.loads((out/'registration.json').read_text());launched=[]
    try:
        pin([116])
        started=time.time()
        while len(launched)<len(plan['jobs']):
            waiting=[]
            for job in plan['jobs']:
                if job['run_id'] in launched:continue
                try:
                    states=dependencies(root,job['host'],job.get('wait_for',[]))
                except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
                    if isinstance(error,subprocess.CalledProcessError) and error.returncode!=255:
                        raise
                    waiting.append(dict(run=job['run_id'],ready=False,observation_error=repr(error)))
                    atomic_json(out/'status.json',dict(state='observation_retry',launched=launched,
                                waiting=waiting,updated=time.time()))
                    continue
                if any(row['state'] in ('failed','stopped') for row in states):
                    raise RuntimeError('Retained failed dependency: '+str(states))
                ready=all(row['ready'] for row in states)
                waiting.append(dict(run=job['run_id'],ready=ready,dependencies=states))
                if ready:
                    command=['taskset','--cpu-list',','.join(map(str,config['affinity'])),PYTHON,'-B',
                        str(out/'code/deploy_research.py'),'--root',str(root),'--config',
                        str(out/f"host-{job['host']}.json"),'--source',config['source']]
                    with (out/(job['run_id']+'.launch.log')).open('a') as log:
                        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                    launched.append(job['run_id'])
                atomic_json(out/'status.json',dict(state='waiting',launched=launched,waiting=waiting,updated=time.time()))
            if (out/'stop').exists():raise RuntimeError('Queue stop requested; running learners keep their own stop controls')
            if time.time()-started>config['timeout_seconds']:raise TimeoutError('CPU dependencies exceeded the queue bound')
            if len(launched)<len(plan['jobs']):time.sleep(20)
        atomic_json(out/'status.json',dict(state='complete',launched=launched,updated=time.time()))
    except BaseException as error:
        atomic_json(out/'status.json',dict(state='failed',error=repr(error),launched=launched,updated=time.time()))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--plan',type=Path)
    p.add_argument('--source',type=Path,help='Qualified immutable learner environment')
    p.add_argument('--run-id')
    p.add_argument('--timeout-seconds',type=int,default=14400)
    p.add_argument('--worker',type=Path)
    args=p.parse_args()
    if args.worker:return worker(args.worker)
    if not args.plan or not args.source or not args.run_id:p.error('--plan, --source and --run-id are required')
    if Path(args.run_id).name!=args.run_id or args.run_id in ('.','..'):p.error('Use a plain queue run ID')
    if not 60<=args.timeout_seconds<=43200:p.error('Choose a queue bound between one minute and twelve hours')
    if not (args.source/'snapshot.json').is_file():p.error('Source must be a published immutable environment')
    plan=json.loads(args.plan.read_text());hosts=[j['host'] for j in plan['jobs']]
    if not hosts or len(set(hosts))!=len(hosts) or any(h not in range(4) for h in hosts):
        p.error('Use one case per host; queue another study for another wave')
    for job in plan['jobs']:
        for name in job.get('wait_for',[]):
            if Path(name).is_absolute() or any(part in ('.','..') for part in name.split('/')):
                p.error('Dependencies must name run directories under the storage root')
    affinity=sorted(os.sched_getaffinity(0));pin([117,118,119]);out=args.root/'runs'/args.run_id
    with StorageBudget(args.root).reserve(files=1<<20,heap=GIB,purpose='freeze CPU dependency queue'):
        (out/'code').mkdir(parents=True,exist_ok=False)
        for name in ('queue_cpu.py','deploy_research.py','cluster.py','replicate_checkpoints.py'):
            shutil.copyfile(Path(__file__).with_name(name),out/'code'/name)
        atomic_json(out/'registration.json',plan)
        for job in plan['jobs']:
            atomic_json(out/f"host-{job['host']}.json",dict(plan,name=plan['name']+f"-host{job['host']}",jobs=[job]))
        atomic_json(out/'config.json',dict(root=str(args.root),source=str(args.source),affinity=affinity,
                                          timeout_seconds=args.timeout_seconds))
        env=dict(os.environ,PYTHONPATH=str(args.source/'site-packages'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
        with (out/'worker.log').open('a') as log:
            process=subprocess.Popen(['taskset','--cpu-list','116',PYTHON,'-B',
                str(out/'code/queue_cpu.py'),'--worker',str(out/'config.json')],
                env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(out/'job.json',dict(pid=process.pid))
        print(json.dumps(dict(output=str(out),pid=process.pid)))


if __name__=='__main__':main()
