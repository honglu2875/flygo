#!/usr/bin/env python3
"""Small four-host data launcher. Snapshots are immutable; status is read-only."""
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
if __name__ == '__main__':
    # Status/stop need only the standard-library runtime helpers, even before build.
    sys.path.insert(0, str(REPO / 'python'))
from flygo.runtime import cpu_profile
from flygo.storage import GIB, StorageBudget

HOSTS = [f't1v-n-a09f5679-w-{i}' for i in range(4)]
SSH = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
       '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4']
PYTHON = '/home/cubic27/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12'


def remote(host, code, *, timeout=30):
    return subprocess.check_output(SSH + ['cubic27@' + host, 'python3 -c ' + shlex.quote(code)], text=True, timeout=timeout)


def snapshot(root):
    installed = root / 'venv/lib/python3.12/site-packages'
    native = next((installed / 'flygo').glob('_native*.so'))
    digest = hashlib.sha256(native.read_bytes())
    for path in sorted((REPO / 'python/flygo').rglob('*.py')):
        digest.update(str(path.relative_to(REPO)).encode()); digest.update(path.read_bytes())
    key = digest.hexdigest()[:20]
    target = root / 'environments' / key
    if target.exists():
        return target
    with StorageBudget(root).reserve(files=GIB, heap=GIB, purpose='freeze data worker environment'):
        site = target / 'site-packages'
        site.mkdir(parents=True)
        for name in ('numpy', 'numpy.libs', 'flygo'):
            shutil.copytree(installed / name, site / name, ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(REPO / 'python/flygo', site / 'flygo', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
        (target / 'entry.py').write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).parent/'site-packages'))\nfrom flygo.data.service import main\nmain()\n")
        (target / 'snapshot.json').write_text(json.dumps(dict(snapshot=key, created=time.time(),
                native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(), numpy='2.5.3'), indent=2)+'\n')
        if (root / 'venv/build.json').exists():
            shutil.copyfile(root / 'venv/build.json', target / 'build.json')
    return target


def start(root, run_id, selection):
    candidates = json.loads((root / 'runs/m2/candidates.json').read_text())
    teacher = json.loads(selection.read_text())['selected_teacher']
    teacher = {**teacher, 'role': 'raw policy/value teacher; selected by 2026-09-13 9x9 panel'}
    contract = dict(schema_version=1, rules='positional-area-multisuicide-v1', board_size=9, komi=7.5,
                    teacher=teacher, opponents=candidates['opponents'], engine_sha256=candidates['engine']['sha256'],
                    visits=16, max_moves=324, opening_moves=8,
                    behavior='root order after 8-ply 0.75 visits + 0.25 raw policy exploration',
                    opponent_assignment='uniform checkpoint strata; equal planned games per stratum',
                    labels='raw policy and raw value all plies; searched teacher labels only expert turns',
                    neural_history='ignorePreRootHistory=false, ignoreAllHistory=false; complete causal prefix',
                    search_determinism='concurrent KataGo search can differ across runs; published rows immutable')
    profile = cpu_profile()
    environment = snapshot(root)
    run = root / 'runs' / run_id
    run.mkdir(parents=True, exist_ok=False)
    (run / 'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    def deploy(index):
        host = HOSTS[index]
        config = dict(storage_root=str(root), run_id=run_id, host_index=index, concurrent_games=4,
                      research_cpus=profile['research_cpus'], contract=contract,
                      workers=[dict(cpus=cpus, opponent_index=i) for i,cpus in enumerate(profile['workers'])])
        # Inventory is checked before transferring or starting anything on a host.
        check = json.loads(remote(host, 'import json,os,pathlib; print(json.dumps(dict(cpus=len(os.sched_getaffinity(0)),python=pathlib.Path('+repr(PYTHON)+').is_file())))'))
        if check != dict(cpus=240, python=True):
            raise RuntimeError(f'Host prerequisites differ: {host}: {check}')
        admission = json.loads(remote(host, 'import os,json,pathlib; p=pathlib.Path('+repr(str(root))+');'
            'p.mkdir(parents=True,exist_ok=True); s=os.statvfs(p); '
            'm={x.split(":")[0]:int(x.split()[1])*1024 for x in pathlib.Path("/proc/meminfo").read_text().splitlines()};'
            'print(json.dumps(dict(free=s.f_bavail*s.f_frsize,available=m["MemAvailable"])))'))
        if admission['free'] < 66*GIB or admission['available'] < 164*GIB:
            raise RuntimeError(f'Insufficient shared RAM headroom for deployment: {host}: {admission}')
        if index:
            remote(host, 'from pathlib import Path;Path('+repr(str(root / 'environments'))+').mkdir(parents=True,exist_ok=True)')
            transport = shlex.join(SSH)
            for source, destination in ((environment, root / 'environments'), (root / 'artifacts', root)):
                subprocess.run(['rsync', '-a', '--ignore-existing', '-e', transport,
                                str(source), f'cubic27@{host}:{destination}/'], check=True)
        path = run / 'config.json'
        remote(host, 'from pathlib import Path; p=Path('+repr(str(path))+');p.parent.mkdir(parents=True,exist_ok=True);p.write_text('+repr(json.dumps(config,indent=2)+'\n')+')')
        log = (run / f'host-{index}.log').open('a')
        command = shlex.join([PYTHON, str(environment / 'entry.py'), 'serve', str(path)])
        process = subprocess.Popen(SSH + ['cubic27@'+host, command], stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        return dict(host=host, ssh_pid=process.pid, environment=str(environment), config=str(path))
    with ThreadPoolExecutor(max_workers=4) as pool:
        launched = list(pool.map(deploy, range(4)))
    (run / 'cluster.json').write_text(json.dumps(launched,indent=2)+'\n')
    print(json.dumps(launched,indent=2))


def status(root, run_id, *, as_json=False):
    code = '''import pathlib,json,os,hashlib
root=pathlib.Path(ROOT);p=root/'runs'/RUN_ID
config=json.loads((p/'config.json').read_text())
contract_id=hashlib.sha256(json.dumps(config['contract'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
memory={line.split(':')[0]:int(line.split()[1])*1024 for line in pathlib.Path('/proc/meminfo').read_text().splitlines()}
fs=os.statvfs(root)
print(json.dumps(dict(supervisor=json.loads((p/'supervisor.json').read_text()) if (p/'supervisor.json').exists() else None,
 workers=[json.loads(x.read_text()) for x in sorted(p.glob('worker-*/status.json'))],
 concurrent_games=config['concurrent_games'],
 published_games=sum(1 for _ in (root/'corpora'/contract_id).glob('host-'+str(config['host_index'])+'/worker-*/*.npz')),
 ram_available=memory['MemAvailable'],shm_free=fs.f_bavail*fs.f_frsize,
 learners=[dict(run=x.parent.name,**json.loads(x.read_text())) for x in sorted((root/'runs').glob('*/status.json'))])))
'''.replace('ROOT',repr(str(root))).replace('RUN_ID',repr(run_id))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda h: dict(host=h, **json.loads(remote(h, code))), HOSTS))
    if as_json:
        print(json.dumps(results,indent=2))
        return
    print(f'{run_id}: 64 physical cores/host, 8 workers/host. RAM figures are GiB.')
    print('Host  Workers       Published games  Session positions  Free shm  Available RAM')
    for record in results:
        workers=record['workers']
        active=sum(w['state']=='generating' for w in workers)
        states=','.join(sorted({w['state'] for w in workers if w['state']!='generating'}))
        print(f"w{record['host'][-1]}    {active}/{len(workers)} {states:<9} {record['published_games']:>10,}"
              f" {sum(w['positions'] for w in workers):>18,} {record['shm_free']/GIB:>9.1f} {record['ram_available']/GIB:>14.1f}")
        for learner in record['learners']:
            if learner.get('state') not in ('complete','stopped'):
                detail=(f" at update {learner['step']}" if learner.get('step') is not None
                        else ': '+learner['reason'] if learner.get('reason') else '')
                print(f"      {learner['run']}: {learner.get('state')}{detail}")
    print('Session positions reset when a worker restarts; published game counts include earlier work.')


def restart(root, run_id, concurrent_games):
    """Drain each host, preserve its immutable games, and change only runtime concurrency."""
    run = root / 'runs' / run_id
    previous = json.loads((run / 'cluster.json').read_text())
    revision = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    archive = run / 'restarts' / revision
    archive.mkdir(parents=True, exist_ok=False)
    (archive / 'cluster-before.json').write_text(json.dumps(previous, indent=2) + '\n')

    def host_restart(record):
        host = record['host']
        code = '''import pathlib,json,hashlib,time,fcntl
run=pathlib.Path(RUN)
archive=run/'restarts'/REVISION
archive.mkdir(parents=True,exist_ok=True)
config=json.loads((run/'config.json').read_text())
old=config['concurrent_games']
(archive/'config-before.json').write_text(json.dumps(config,indent=2)+'\\n')
corpus=pathlib.Path(config['storage_root'])/'corpora'
pattern='*/host-'+str(config['host_index'])+'/worker-*/*.npz'
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in corpus.glob(pattern)}
(archive/'games-before.json').write_text(json.dumps(before,sort_keys=True)+'\\n')
(run/'stop').touch()
deadline=time.monotonic()+330
while True:
 state=json.loads((run/'supervisor.json').read_text())
 if state.get('state')=='stopped':break
 if time.monotonic()>deadline:raise TimeoutError('Supervisor did not finish draining; stop file retained')
 time.sleep(5)
with (run/'supervisor.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert all(pathlib.Path(p).is_file() and hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
 config['concurrent_games']=CONCURRENCY
 temporary=run/'config.json.restarting'
 temporary.write_text(json.dumps(config,indent=2)+'\\n')
 temporary.replace(run/'config.json')
 (run/'stop').unlink()
 result=dict(host=HOST,previous_concurrent_games=old,concurrent_games=CONCURRENCY,
             preserved_games=len(before),status='ready',updated=time.time())
 (archive/'result.json').write_text(json.dumps(result,indent=2)+'\\n')
 print(json.dumps(result))
'''.replace('RUN', repr(str(run))).replace('REVISION', repr(revision)).replace('CONCURRENCY', str(concurrent_games)).replace('HOST', repr(host))
        result = json.loads(remote(host, code, timeout=360))
        command = shlex.join([PYTHON, str(Path(record['environment']) / 'entry.py'),
                              'serve', record['config']])
        index = HOSTS.index(host)
        with (run / f'host-{index}.log').open('a') as log:
            process = subprocess.Popen(SSH + ['cubic27@' + host, command], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result.update(ssh_pid=process.pid, environment=record['environment'], config=record['config'])
        print(json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(host_restart, previous))
    (archive / 'result.json').write_text(json.dumps(results, indent=2) + '\n')
    temporary = run / 'cluster.json.restarting'
    temporary.write_text(json.dumps(results, indent=2) + '\n')
    temporary.replace(run / 'cluster.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'status', 'stop', 'restart'])
    parser.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    parser.add_argument('--run-id', default='expert-v1')
    parser.add_argument('--selection', type=Path, default=Path('/dev/shm/flygo/runs/m2/data-qualification/teacher-panel.json'))
    parser.add_argument('--concurrent-games', type=int, default=16, help='Runtime concurrency for restart; cores and corpus contract stay fixed')
    parser.add_argument('--json',action='store_true',help='Detailed machine-readable status')
    args = parser.parse_args()
    if not 1 <= args.concurrent_games <= 32:
        parser.error('Concurrent games must be in 1..32')
    if args.action == 'start':
        start(args.root, args.run_id, args.selection)
    elif args.action == 'status':
        status(args.root,args.run_id,as_json=args.json)
    elif args.action == 'restart':
        restart(args.root,args.run_id,args.concurrent_games)
    else:
        for host in HOSTS:
            remote(host, 'from pathlib import Path; Path('+repr(str(args.root/'runs'/args.run_id/'stop'))+').touch()')
        print('Requested graceful drain of current batches on all four hosts.')


if __name__ == '__main__':
    main()
