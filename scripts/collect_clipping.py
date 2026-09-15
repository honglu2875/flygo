#!/usr/bin/env python3
"""Close a registered clipping study on a worker, relaying archives through root."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.storage import StorageBudget


def read(path):
    return json.loads(path.read_text())


def inside(root, relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise ValueError('Expected an artifact path relative to the root')
    target = root / path
    target.resolve().relative_to(root.resolve())
    return target


def archive_members(root, archive, manifest_path, expected_manifest):
    """Check every member before extracting anything; never trust tar paths."""
    with tarfile.open(archive) as tar:
        entries = tar.getmembers()
        members = {m.name: m for m in entries}
        if len(members) != len(entries) or any(not m.isfile() for m in entries):
            raise ValueError('Archive must contain unique regular files')
        for name in members:
            inside(root, name)
        if manifest_path not in members:
            raise ValueError('Archive manifest is missing')
        payload = tar.extractfile(members[manifest_path]).read()
        if hashlib.sha256(payload).hexdigest() != expected_manifest:
            raise ValueError('Archive manifest changed')
        manifest = json.loads(payload)
        if set(members) != set(manifest['files']) | {manifest_path}:
            raise ValueError('Archive inventory differs from the manifest')
        for name, record in manifest['files'].items():
            member = members[name]
            if member.size != record['bytes']:
                raise ValueError('Archive member size differs: ' + name)
            digest = hashlib.sha256()
            with tar.extractfile(member) as source:
                for chunk in iter(lambda: source.read(1 << 20), b''):
                    digest.update(chunk)
            if digest.hexdigest() != record['sha256']:
                raise ValueError('Archive member checksum differs: ' + name)
        return manifest


def extract_verified(root, archive, manifest_path, expected_manifest):
    manifest = archive_members(root, archive, manifest_path, expected_manifest)
    records = dict(manifest['files'])
    with tarfile.open(archive) as tar:
        member = tar.getmember(manifest_path)
        records[manifest_path] = dict(bytes=member.size, sha256=expected_manifest)
        missing = []
        # Detect conflicts throughout the inventory before publishing new files.
        for name, record in records.items():
            target = inside(root, name)
            if target.exists():
                if not target.is_file() or target.stat().st_size != record['bytes'] or sha256(target) != record['sha256']:
                    raise ValueError('Immutable evidence conflict: ' + name)
            else:
                missing.append(name)
        block = os.statvfs(root).f_frsize
        size = sum(((records[name]['bytes'] + block - 1) // block) * block for name in missing)
        with StorageBudget(root).reserve(files=size, heap=4 << 20, purpose='verified study evidence extraction'):
            for name in missing:
                target = inside(root, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                # Failed writes remain visible; an immutable conflict requires a new attempt.
                with target.open('xb') as output, tar.extractfile(name) as source:
                    for chunk in iter(lambda: source.read(1 << 20), b''):
                        output.write(chunk)
                if sha256(target) != records[name]['sha256']:
                    raise ValueError('Extracted checksum differs: ' + name)
    return dict(files=len(manifest['files']), extracted=len(missing))


def configuration(path):
    config = read(path); root = Path(config['root'])
    path.resolve().relative_to(root.resolve())
    if sha256(Path(__file__)) != config['collector_sha256']:
        raise ValueError('Frozen collection implementation changed')
    analysis_path = inside(root, config['analysis'])
    if sha256(analysis_path) != config['analysis_sha256']:
        raise ValueError('Registered analysis changed')
    analysis = read(analysis_path)
    plan_path = inside(root, analysis['trial_plan'])
    if sha256(plan_path) != analysis['trial_plan_sha256']:
        raise ValueError('Registered trials changed')
    tools = inside(root, config['tools'])
    if (sha256(tools / 'summarize_clipping.py') != analysis['analysis_implementation_sha256']
            or sha256(tools / 'attachment_endpoint.py') != config['endpoint_helper_sha256']):
        raise ValueError('Frozen analysis or endpoint helper changed')
    if config['analysis_host'] != analysis['bulk_analysis_host']:
        raise ValueError('Collection destination differs from registration')
    return config, root, analysis, read(plan_path), tools


def inspect(root, analysis, plan, host):
    from finish_study import dependency_complete, running_train
    rows = []
    for job in plan['jobs']:
        if job['host'] != host:
            continue
        run = job['run_id']; status_path = root / 'runs' / run / 'status.json'
        status = read(status_path) if status_path.exists() else {}
        follow = inside(root, analysis['followup']) / run
        follower_path = follow / 'status.json'
        follower = read(follower_path) if follower_path.exists() else {}
        if status.get('state') in ('failed', 'stopped') or follower.get('state') in ('failed', 'stopped'):
            raise RuntimeError('A registered endpoint failed; preserve the case: ' + run)
        pid = running_train(run)
        # Later registered waves have no trainer status until their dependency
        # releases the CPU lane. An absent unqueued learner is not such a wait.
        if not status and pid is None and not job.get('wait_for'):
            raise RuntimeError('An unqueued registered trainer is missing: ' + run)
        ready = (status.get('state') == 'complete' and status.get('step') == plan['updates']
                 and pid is None and dependency_complete(follow))
        rows.append(dict(run_id=run, step=status.get('step'), trainer_pid=pid,
                         trainer_state=status.get('state'), followup=follower.get('state'), ready=ready))
    if not rows:
        raise ValueError('No registered jobs on this owner')
    return dict(ready=all(row['ready'] for row in rows), records=rows)


def pack(config_path, host):
    config, root, analysis, plan, tools = configuration(config_path)
    os.sched_setaffinity(0, set(config['audit_cpus']))
    if not inspect(root, analysis, plan, host)['ready']:
        raise ValueError('Wait for each actual learner and followup to exit')
    out = inside(root, config['directory']) / f'owner-{host}'
    if out.exists():
        raise ValueError('Owner archives are immutable; inspect the earlier attempt')
    out.mkdir(parents=True)
    files = [inside(root, analysis['trial_plan']), config_path, Path(__file__)]
    endpoints = []
    for job in [job for job in plan['jobs'] if job['host'] == host]:
        run = job['run_id']; train = root / 'runs' / run
        audit = inside(root, analysis['endpoints']) / (run + '.json')
        command = [sys.executable, '-B', str(tools / 'attachment_endpoint.py'), '--root', str(root),
                   '--plan', str(inside(root, analysis['trial_plan'])), '--run-id', run,
                   '--output', str(audit), '--cpus', ','.join(map(str, config['audit_cpus']))]
        with (out / (run + '.log')).open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        endpoints.append(dict(host=host, run_id=run, path=str(audit.relative_to(root)), sha256=sha256(audit)))
        files += [audit, out / (run + '.log')]
        files += [train / name for name in ('config.json', 'status.json', 'metrics.jsonl', 'latest.json')]
        files += [train / 'checkpoints' / f'step-{step:08d}.json' for step in (0, plan['updates'])]
        followup = inside(root, analysis['followup']) / run
        files += [p for p in followup.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and p.suffix in ('.json', '.jsonl', '.npz', '.npy', '.log', '.py')]
    files = sorted(set(files))
    for path in files:
        path.resolve().relative_to(root.resolve())
        if path.is_symlink():
            raise ValueError('Evidence must be regular files')
    size = sum(p.stat().st_size + 4096 for p in files) + (1 << 20)
    with StorageBudget(root).reserve(files=size, heap=32 << 20, purpose='completed owner evidence archive'):
        manifest = dict(owner=host, files={str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in files})
        atomic_json(out / 'manifest.json', manifest)
        archive = out / 'evidence.tar'
        with tarfile.open(archive, 'w') as tar:
            for path in [*files, out / 'manifest.json']:
                tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
        record = dict(owner=host, path=str(archive.relative_to(root)), bytes=archive.stat().st_size,
                      sha256=sha256(archive), files=len(files), endpoints=endpoints,
                      manifest_path=str((out / 'manifest.json').relative_to(root)), manifest_sha256=sha256(out / 'manifest.json'))
        archive_members(root, archive, record['manifest_path'], record['manifest_sha256'])
        atomic_json(out / 'receipt.json', record)
    return record


def verify(config_path, receipt_path, extract=False):
    config, root, _, _, _ = configuration(config_path)
    os.sched_setaffinity(0, set(config['audit_cpus']))
    record = read(receipt_path); archive = inside(root, record['path'])
    if archive.stat().st_size != record['bytes'] or sha256(archive) != record['sha256']:
        raise ValueError('Archive payload differs from owner receipt')
    if extract:
        checked = extract_verified(root, archive, record['manifest_path'], record['manifest_sha256'])
    else:
        manifest = archive_members(root, archive, record['manifest_path'], record['manifest_sha256'])
        checked = dict(files=len(manifest['files']))
    if checked['files'] != record['files']:
        raise ValueError('Peer inventory differs from owner receipt')
    return dict(status='passed', **checked)


def analyze(config_path):
    config, root, analysis, plan, tools = configuration(config_path)
    os.sched_setaffinity(0, set(config['audit_cpus']))
    out = inside(root, config['directory'])
    records = [read(out / f'owner-{h}.json') for h in sorted({j['host'] for j in plan['jobs']})]
    endpoints = [endpoint for record in records for endpoint in record['endpoints']]
    if {row['run_id'] for row in endpoints} != {job['run_id'] for job in plan['jobs']}:
        raise ValueError('Missing owner endpoint evidence')
    for row in endpoints:
        if sha256(inside(root, row['path'])) != row['sha256']:
            raise ValueError('Owner audit checksum differs')
    with StorageBudget(root).reserve(files=1 << 20, heap=16 << 20, purpose='endpoint collection receipt'):
        target = inside(root, analysis['endpoints']) / 'result.json'
        if target.exists():
            raise ValueError('Endpoint collection is immutable')
        atomic_json(target, dict(status='complete', created=time.time(), records=endpoints,
                                helper_sha256=config['endpoint_helper_sha256']))
    result = root / 'runs' / analysis['name'] / 'result.json'
    with (out / 'analysis.log').open('x') as log:
        subprocess.run([sys.executable, '-B', str(tools / 'summarize_clipping.py'), '--root', str(root),
                        '--plan', str(inside(root, config['analysis'])), '--output', str(result)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read(result)
    return dict(status=report['status'], path=str(result.relative_to(root)), bytes=result.stat().st_size,
                sha256=sha256(result), endpoint_collection_sha256=sha256(target),
                decision=report['decision'], cost_contrasts=report['cost_contrasts'],
                pooled_records=report['statistics']['pooled_records'],
                pooled_contrasts=report['statistics']['pooled_contrasts'])


def coordinate(config_path):
    from cluster import HOSTS, SSH, PYTHON, remote
    from flygo.replication import replicate_file, replicate_stream
    config, root, analysis, plan, _ = configuration(config_path)
    os.sched_setaffinity(0, {116, 117, 118, 119})
    out = inside(root, config['directory']); owners = sorted({j['host'] for j in plan['jobs']})
    destination = config['analysis_host']
    if owners != [1, 2, 3] or destination not in owners:
        raise ValueError('This coordinator expects the three registered CPU workers')

    def call(host, action, *arguments, timeout=300):
        command = [PYTHON, '-B', str(Path(__file__)), action, '--config', str(config_path), *arguments]
        wrapper = ('import os,subprocess;env=dict(os.environ,'
                   f'PYTHONPATH={config["pythonpath"]!r},PYTHONDONTWRITEBYTECODE="1",'
                   'OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",JAX_PLATFORMS="cpu");'
                   f'subprocess.run({command!r},env=env,check=True)')
        return json.loads(remote(HOSTS[host], wrapper, timeout=timeout))

    with (out / 'coordinator.lock').open('a') as lock, StorageBudget(root).reserve(
            files=2 << 20, heap=32 << 20, purpose='small clipping collection control records'):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                try:
                    with ThreadPoolExecutor(3) as pool:
                        status = list(pool.map(lambda h: call(h, 'inspect', '--host', str(h)), owners))
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                    if isinstance(error, subprocess.CalledProcessError) and error.returncode != 255:
                        raise
                    # An observation failure is not evidence that a learner stopped.
                    atomic_json(out / 'status.json', dict(state='observation_retry', error=repr(error), updated=time.time()))
                    time.sleep(20)
                    continue
                atomic_json(out / 'status.json', dict(state='waiting', owners=status, updated=time.time()))
                if any('terminal_error' in row for row in status):
                    raise RuntimeError('Registered endpoint failure: ' + repr(status))
                if all(row['ready'] for row in status):
                    break
                time.sleep(20)
            atomic_json(out / 'status.json', dict(state='owner_audits', updated=time.time()))
            with ThreadPoolExecutor(3) as pool:
                receipts = list(pool.map(lambda h: call(h, 'pack', '--host', str(h), timeout=600), owners))
            for record in receipts:
                host = record['owner']; target = destination if host != destination else (host % 3 + 1)
                receipt_path = out / f'owner-{host}.json'; atomic_json(receipt_path, record)
                for h in {target, destination}:
                    replicate_file(receipt_path, root, 'cubic27@' + HOSTS[h])
                producer = subprocess.Popen(SSH + ['cubic27@' + HOSTS[host],
                    shlex.join(['taskset', '-c', '116-119', 'cat', str(inside(root, record['path']))])], stdout=subprocess.PIPE)
                try:
                    copied = replicate_stream(producer.stdout, root, Path(record['path']), 'cubic27@' + HOSTS[target],
                        size=record['bytes'], digest=record['sha256'], timeout=300)
                finally:
                    producer.stdout.close()
                    try:
                        returncode = producer.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        producer.terminate(); producer.wait(); raise
                if returncode:
                    raise RuntimeError('Owner archive producer failed')
                checked = call(target, 'verify', '--receipt', str(receipt_path))
                extracted = call(destination, 'extract', '--receipt', str(receipt_path))
                atomic_json(out / f'transfer-{host}.json', dict(owner=host, peer=target, transfer=copied,
                    peer_verification=checked, analysis_host=destination, extraction=extracted))
            atomic_json(out / 'status.json', dict(state='analysis', updated=time.time()))
            result = call(destination, 'analyze', timeout=600)
            atomic_json(out / 'result.json', dict(**result, created=time.time(), analysis_host=destination,
                scope='All owner endpoints and raw evidence verified on peers; full analysis remains on the registered worker. Root stores small control records only.'))
            for path in sorted(out.glob('*.json')):
                # status.json remains mutable until the coordinator exits.
                if path.name != 'status.json':
                    replicate_file(path, root, 'cubic27@' + HOSTS[destination])
            atomic_json(out / 'status.json', dict(state='complete', updated=time.time(), sha256=result['sha256']))
        except BaseException as error:
            atomic_json(out / 'status.json', dict(state='failed', error=repr(error), updated=time.time()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['coordinate', 'inspect', 'pack', 'verify', 'extract', 'analyze'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--host', type=int)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.action == 'coordinate':
        coordinate(args.config); return
    config, root, analysis, plan, _ = configuration(args.config)
    if args.action == 'inspect':
        os.sched_setaffinity(0, {116})
        try:
            result = inspect(root, analysis, plan, args.host)
        except RuntimeError as error:
            # A scientific failure is terminal for collection, not a transient SSH error.
            result = dict(ready=False, terminal_error=str(error))
    elif args.action == 'pack':
        with StorageBudget(root).reserve(files=1 << 20, heap=4 << 20, purpose='owner audit logs and receipt'):
            result = pack(args.config, args.host)
    elif args.action == 'analyze': result = analyze(args.config)
    else: result = verify(args.config, args.receipt, extract=args.action == 'extract')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
