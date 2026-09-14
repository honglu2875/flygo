#!/usr/bin/env python3
"""Launch one immutable SPMD qualification worker per TPU host; inspect with status."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def environment(root,base=None):
    from cluster import snapshot
    from flygo.storage import GIB, StorageBudget
    base = snapshot(root) if base is None else base
    cpu = root / 'venv/lib/python3.12/site-packages'
    reference = Path('/home/cubic27/go/.venv/lib/python3.12/site-packages')
    packages = ['jax', 'jaxlib', 'ml_dtypes', 'opt_einsum', 'scipy', 'scipy.libs']
    sources = [cpu / name for name in packages]
    sources += [p for p in cpu.glob('*.dist-info')
                if p.name.split('-')[0] in ('jax', 'jaxlib', 'ml_dtypes', 'opt_einsum', 'scipy')]
    for name in ('libtpu', 'requests', 'urllib3', 'certifi', 'charset_normalizer', 'idna'):
        sources.extend([reference / name, *reference.glob(name + '-*.dist-info')])
    digest = hashlib.sha256(b'flygo-tpu-runtime-v1')
    for source in sorted(sources):
        if not source.is_dir():
            raise FileNotFoundError(source)
        for path in sorted(source.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                digest.update(str(path.relative_to(source.parent)).encode())
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 << 20), b''):
                        digest.update(chunk)
    target = root / 'environments' / ('tpu-' + digest.hexdigest()[:20])
    if (target / 'runtime.json').exists():
        return target
    with StorageBudget(root).reserve(files=4 * GIB, heap=GIB, purpose='isolated TPU runtime'):
        shutil.copytree(base, target, dirs_exist_ok=True)
        for source in sources:
            shutil.copytree(source, target / 'site-packages' / source.name,
                            ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
        (target / 'runtime.json').write_text(json.dumps(dict(
            runtime_id=digest.hexdigest(), flygo_environment=base.name,
            jax='0.11.1', jaxlib='0.11.1', libtpu='0.0.46.1',
            sources=[str(p) for p in sources]), indent=2) + '\n')
    return target


def launch(args):
    from cluster import HOSTS, SSH, PYTHON, remote, snapshot
    from flygo.storage import GIB, StorageBudget
    from flygo.data.corpus import atomic_json
    run = args.root / 'runs' / args.run_id
    if run.exists():
        raise FileExistsError('Use a new run ID; prior attempts are retained: ' + str(run))
    source = args.source or snapshot(args.root)
    runtime = args.runtime or environment(args.root,base=source)
    training=json.loads(args.train_plan.read_text()) if args.train_plan else None
    # A training-plan port artifact is part of the declared model and must be
    # copied before all controllers start, just like qualification ports.
    if training and training.get('ports'):
        planned_ports=Path(training['ports'])
        if args.ports and args.ports!=planned_ports:
            raise ValueError('Training-plan and command-line port artifacts differ')
        args.ports=planned_ports
    run.mkdir(parents=True)
    worker = run / 'worker.py'
    shutil.copyfile(args.worker or REPO / 'scripts/tpu_worker.py', worker)
    config = dict(schema_version=1, mode=args.mode, cpus=args.cpus,
                  per_device_batches=args.batches, steps=args.steps, repetitions=args.repetitions,
                  kernel=args.kernel,
                  tile_edges=args.tile_edges,tile_rows=args.tile_rows,
                  release=args.release,groups=args.groups,seed=args.seed,
                  restore_from=str(args.restore_from) if args.restore_from else None,
                  training=training,
                  ports=str(args.ports) if args.ports else None,rate_scales=args.rate_scales,
                  control_reference=str(args.control_reference) if args.control_reference else None,
                  runtime=str(runtime), source=str(source), worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
                  root=str(args.root), run_id=args.run_id, expected_processes=4, expected_devices=16,
                  precision='float32; highest matmul precision; FP32 global gradient norm',
                  started=time.time())
    atomic_json(run / 'config.json', config)

    def deploy(index):
        host = HOSTS[index]
        if index:
            code = f'''import sys
sys.path.insert(0,{str(args.root / 'environments/68683f4941d89ec42a34/site-packages')!r})
from flygo.storage import StorageBudget,GIB
StorageBudget({str(args.root)!r}).check()
from pathlib import Path
Path({str(runtime.parent)!r}).mkdir(parents=True,exist_ok=True)
Path({str(run)!r}).mkdir(parents=True,exist_ok=False)
'''
            remote(host, code)
            with StorageBudget(args.root).reserve(files=0, heap=GIB, purpose='TPU runtime transfer'):
                for artifact, destination in ((runtime, runtime.parent), (source, source.parent), (worker, run), (run / 'config.json', run)):
                    subprocess.run(['rsync', '-a', '--checksum', '-e', shlex.join(SSH), str(artifact),
                                    f'cubic27@{host}:{destination}/'], check=True)
            if args.ports:
                from flygo.replication import replicate_bundle
                replicate_bundle([args.ports,args.ports.with_suffix('.json')],args.root,f'cubic27@{host}')
            if args.control_reference:
                from flygo.replication import replicate_bundle
                replicate_bundle(sorted(args.control_reference.glob('*')),args.root,f'cubic27@{host}')
        return host

    def start(host):
        code = f'''import os,subprocess,json,pathlib,time
run=pathlib.Path({str(run)!r})
env=dict(os.environ,PYTHONPATH={str(source / 'site-packages')+':'+str(runtime / 'site-packages')!r},PYTHONDONTWRITEBYTECODE='1',
         JAX_PLATFORMS='tpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
         TMPDIR={str(args.root / 'tmp')!r},JAX_COMPILATION_CACHE_DIR=str(run/'compile-cache'))
env.pop('LD_PRELOAD',None)
pathlib.Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
with (run/'worker.log').open('a') as log:
 p=subprocess.Popen([{PYTHON!r},str(run/'worker.py'),'--config',str(run/'config.json')],
                    env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(host={host!r},pid=p.pid,started=time.time())
(run/'job.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
'''
        return json.loads(remote(host, code))

    with ThreadPoolExecutor(4) as pool:
        hosts = list(pool.map(deploy, range(4)))
        jobs = list(pool.map(start, hosts))
    atomic_json(run / 'jobs.json', jobs)
    print(json.dumps(dict(run=str(run), jobs=jobs), indent=2))


def status(args):
    from cluster import HOSTS, remote
    code = f'''import pathlib,json
p=pathlib.Path({str(args.root / 'runs' / args.run_id)!r})
s=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {{}}
summary={{k:v for k,v in s.items() if k not in ('config','affinity','records','parity','local_devices','layout')}}
if s.get('learner_run'):
 learner=p.parent/s['learner_run']/'status.json'
 if learner.exists():summary['learner']=json.loads(learner.read_text())
summary['completed_cases']=len(s.get('records',[]))
summary['parity']=s.get('parity',{{}}).get('status')
summary['thread_count']=len(s.get('affinity',[]))
summary['affinity_masks']=list({{tuple(x) for x in s.get('affinity',[])}})
if s.get('status')=='failed' and (p/'worker.log').exists():summary['log']=(p/'worker.log').read_text()[-3000:]
print(json.dumps(summary))'''
    with ThreadPoolExecutor(4) as pool:
        for host, result in zip(HOSTS, pool.map(lambda h: remote(h, code), HOSTS)):
            print(host, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('launch', 'status'))
    p.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    p.add_argument('--run-id', required=True)
    p.add_argument('--mode', choices=('probe', 'audit', 'benchmark','qualify','restore','train','control'), default='probe')
    p.add_argument('--cpus', type=lambda v: [int(x) for x in v.split(',')], default=[56,57,58,59,117,118,119])
    p.add_argument('--batches', type=lambda v: [int(x) for x in v.split(',')], default=[1,4,16,64,128])
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--groups',type=int,default=656)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--release',default='v0-1m')
    p.add_argument('--restore-from',type=Path)
    p.add_argument('--train-plan',type=Path)
    p.add_argument('--ports',type=Path)
    p.add_argument('--rate-scales',type=json.loads,default={})
    p.add_argument('--control-reference',type=Path)
    p.add_argument('--source',type=Path,help='Immutable source snapshot for a multi-trial study')
    p.add_argument('--runtime',type=Path,help='Previously qualified immutable dependency runtime')
    p.add_argument('--worker',type=Path,help='Frozen worker script for a multi-trial study')
    p.add_argument('--kernel',choices=('reference','buckets'),default='reference')
    p.add_argument('--tile-edges',type=int,default=32768)
    p.add_argument('--tile-rows',type=int,default=128)
    p.add_argument('--repetitions', type=int, default=3)
    args = p.parse_args()
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        p.error('Run ID must be one path component')
    if min(args.batches) < 1 or min(args.steps,args.repetitions,args.tile_edges,args.tile_rows) < 1:
        p.error('Positive batch sizes, steps and repetitions required')
    if args.mode=='restore' and not args.restore_from:
        p.error('--restore-from is required for fresh-process recovery')
    if args.mode=='train' and not args.train_plan:
        p.error('--train-plan is required for learning')
    if args.mode=='control' and not args.control_reference:
        p.error('--control-reference is required for CNN qualification')
    (launch if args.action == 'launch' else status)(args)


if __name__ == '__main__':
    main()
