#!/usr/bin/env python3
"""Deploy one immutable environment and frozen corpus for an explicit CPU experiment plan."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess

from cluster import snapshot, SSH, PYTHON
from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release
from flygo.data.cache import CACHE_VERSION
from flygo.replication import replicate_bundle
from flygo.runtime import cpu_profile,pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    parser.add_argument('--resume-launch',action='store_true',help='Adopt matching live trials from a partially completed launch')
    args = parser.parse_args()
    plan = json.loads(args.config.read_text())
    root = args.root
    allowed = set(cpu_profile()['research_cpus'])
    pin([117,118,119])
    allocations = {}
    run_ids=set()
    for job in plan['jobs']:
        run_id=job.get('run_id',f"{plan['name']}-k{job['passes']}-seed{job['seed']}")
        if run_id in run_ids or Path(run_id).name!=run_id:
            raise ValueError('Every trial must have a unique plain run ID')
        run_ids.add(run_id)
        cpus = set(job['cpus'])
        if not cpus or not cpus <= allowed or len(cpus) != len(job['cpus']):
            raise ValueError('Every job must use a unique subset of spare physical CPUs')
        if cpus & allocations.setdefault(job['host'], set()):
            raise ValueError('The experiment plan overlaps CPU allocations on a host')
        if job['host'] not in range(4):
            raise ValueError('Host index must be in 0..3')
        allocations[job['host']] |= cpus

    environment = snapshot(root)
    release = root / 'releases' / plan['release']
    manifest = json.loads((release / 'manifest.json').read_text())
    if json.loads((release / 'replication.json').read_text())['status'] != 'passed':
        raise ValueError('The frozen corpus must have a verified second copy')
    graph_record = root / 'runs/m4/graph.json'
    graph_path = Path(json.loads(graph_record.read_text())['path'])
    files = {graph_record, *(root / record['path'] for record in manifest['records'])}
    for directory in (environment, graph_path, release):
        files.update(path for path in directory.rglob('*') if path.is_file())
    if plan.get('feature_cache',False):
        load_release(root,release/'manifest.json',cache=True)
        cache=root/'cache/features'/CACHE_VERSION/manifest['dataset_id']
        files.update(path for path in cache.iterdir() if path.is_file())
    output = root / 'runs' / plan['name']
    if output.exists():
        if not args.resume_launch or json.loads((output/'plan.json').read_text())!=plan:
            raise ValueError('Existing plan is immutable; use --resume-launch only with the identical plan')
    else:
        output.mkdir(parents=True)
        atomic_json(output / 'plan.json', plan)

    def deploy(host):
        if host:
            replicate_bundle(sorted(files), root, f'cubic27@t1v-n-a09f5679-w-{host}')
        results = []
        for job in (job for job in plan['jobs'] if job['host'] == host):
            run_id = job.get('run_id',f"{plan['name']}-k{job['passes']}-seed{job['seed']}")
            if args.resume_launch:
                code='''import pathlib,json,os
p=pathlib.Path(ROOT)/'runs'/RUN_ID
status=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {}
config=json.loads((p/'config.json').read_text()) if (p/'config.json').exists() else {}
pid=status.get('pid');live=bool(pid and pathlib.Path('/proc',str(pid)).exists())
command=pathlib.Path('/proc',str(pid),'cmdline').read_bytes().decode().split('\\0') if live else []
print(json.dumps(dict(exists=p.exists(),status=status,config=config,live=live,command=command)))
'''.replace('ROOT',repr(str(root))).replace('RUN_ID',repr(run_id))
                inspect=[PYTHON,'-c',code]
                if host:inspect=SSH+[f'cubic27@t1v-n-a09f5679-w-{host}',shlex.join(inspect)]
                old=json.loads(subprocess.check_output(inspect,text=True,timeout=30))
                if old['exists']:
                    config=old['config'];arguments=config.get('arguments',{})
                    expected=dict(run_id=run_id,release=plan['release'],passes=job['passes'],
                        groups=job.get('groups',plan.get('groups',656)),seed=job['seed'],
                        steps=plan['updates'],batch_size=plan['batch_size'],rate=plan['rate'],
                        threads=len(job['cpus']),eval_every=plan['eval_every'],checkpoint_every=plan['checkpoint_every'])
                    valid=(config.get('dataset_id')==manifest['dataset_id'] and config.get('cpus')==job['cpus']
                           and all(arguments.get(key)==value for key,value in expected.items()))
                    if not valid or not (old['status'].get('state')=='complete' or
                            (old['live'] and 'flygo.train' in old['command'] and run_id in old['command'])):
                        raise ValueError('Cannot adopt a mismatched or stopped trial: '+run_id)
                    record={**job,'run_id':run_id,'environment':str(environment),
                            'dataset_id':manifest['dataset_id'],'adopted_pid':old['status'].get('pid')}
                    atomic_json(output/(run_id+'.job.json'),record)
                    results.append(record);print(json.dumps(record),flush=True)
                    continue
            command = ['env', 'PYTHONPATH=' + str(environment / 'site-packages'),
                       'PYTHONDONTWRITEBYTECODE=1', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1',
                       PYTHON, '-u', '-m', 'flygo.train', '--root', str(root),
                       '--release', plan['release'], '--run-id', run_id,
                       '--passes', str(job['passes']), '--seed', str(job['seed']),
                       '--groups',str(job.get('groups',plan.get('groups',656))),
                       '--steps', str(plan['updates']), '--batch-size', str(plan['batch_size']),
                       '--rate', str(plan['rate']), '--threads', str(len(job['cpus'])),
                       '--cpus', ','.join(map(str, job['cpus'])), '--peer', '',
                       '--eval-every', str(plan['eval_every']),
                       '--checkpoint-every', str(plan['checkpoint_every'])]
            if host:
                command = SSH + [f'cubic27@t1v-n-a09f5679-w-{host}', shlex.join(command)]
            else:
                # The coordinator is pinned to spare housekeeping cores. Give
                # its local child the plan's validated allocation before Python
                # checks inherited affinity; taskset affects only this child.
                command=['taskset','--cpu-list',','.join(map(str,job['cpus'])),*command]
            with (output / (run_id + '.log')).open('a') as log:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            record = {**job,'run_id':run_id,'coordinator_pid':process.pid,
                      'environment':str(environment),'dataset_id':manifest['dataset_id']}
            atomic_json(output / (run_id + '.job.json'), record)
            results.append(record)
            print(json.dumps(record), flush=True)
        return results

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = sum(pool.map(deploy, sorted(allocations)), [])
    atomic_json(output / 'jobs.json', records)
    coordinator=output/'replica-coordinator.json'
    if coordinator.exists():
        pid=json.loads(coordinator.read_text())['pid']
        try:
            command=Path('/proc',str(pid),'cmdline').read_bytes().decode().split('\0')
        except FileNotFoundError:
            command=[]
        if str(output/'jobs.json') in command and any('replicate_checkpoints.py' in part for part in command):
            return
    command = ['taskset','--cpu-list','116',str(root / 'venv/bin/python'), '-B', str(Path(__file__).with_name('replicate_checkpoints.py')),
               '--root', str(root), '--jobs', str(output / 'jobs.json'),
               '--output', str(output / 'replicas')]
    with (output / 'replicas.log').open('a') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, 'OPENBLAS_NUM_THREADS': '1'})
    atomic_json(coordinator, dict(pid=process.pid))


if __name__ == '__main__':
    main()
