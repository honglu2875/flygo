#!/usr/bin/env python3
"""Collect immutable endpoint audits on their owners; never copy checkpoint payloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from cluster import PYTHON, remote
from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.replication import replicate_bundle
from flygo.runtime import pin
from flygo.storage import GIB, StorageBudget


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='Reuse identical completed audits after an interrupted collection')
    args = p.parse_args(); root = args.root; out = args.output
    pin([117, 118, 119])
    out.resolve().relative_to(root.resolve()); args.plan.resolve().relative_to(root.resolve())
    plan = json.loads(args.plan.read_text())
    jobs = plan['jobs']
    if (not jobs or len({j['run_id'] for j in jobs}) != len(jobs)
            or any(j['host'] not in (1, 2, 3) or Path(j['run_id']).name != j['run_id']
                   or j['run_id'] in ('.', '..') for j in jobs)):
        raise ValueError('Expected unique, plain run IDs with worker checkpoint owners')
    environment = root / plan['environment']
    if not (environment / 'snapshot.json').is_file():
        raise ValueError('Use the study\'s published source environment')
    helper = Path(__file__).with_name('attachment_endpoint.py')
    contract = dict(plan_sha256=sha256(args.plan), helper_sha256=sha256(helper),
                    collector_sha256=sha256(Path(__file__)), environment=str(environment))
    with StorageBudget(root).reserve(files=16 << 20, heap=GIB, purpose='owner-local endpoint audit collection'):
        if out.exists():
            if not args.resume or json.loads((out / 'contract.json').read_text()) != contract:
                raise ValueError('Retain the previous attempt; only an identical collection can resume')
            if (out / 'result.json').exists():
                raise ValueError('Completed collections are immutable')
        else:
            out.mkdir(parents=True)
            atomic_json(out / 'contract.json', contract)
            (out / helper.name).write_bytes(helper.read_bytes())
            (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        script = out / helper.name
        if sha256(script) != contract['helper_sha256']:
            raise ValueError('Frozen audit source changed')

        def collect(host):
            peer = 'cubic27@t1v-n-a09f5679-w-' + str(host)
            replicate_bundle([script, args.plan, out / 'contract.json'], root, peer)
            records = []
            for job in (j for j in jobs if j['host'] == host):
                target = out / (job['run_id'] + '.json')
                command = [PYTHON, '-B', str(script), '--root', str(root), '--plan', str(args.plan),
                           '--run-id', job['run_id'], '--output', str(target)]
                code = f'''import json, os, pathlib, subprocess
target = pathlib.Path({str(target)!r})
if not target.exists():
    env = dict(os.environ, PYTHONPATH={str(environment / 'site-packages')!r},
               PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', JAX_PLATFORMS='cpu')
    env.pop('LD_PRELOAD', None)
    with target.with_suffix('.log').open('x') as log:
        subprocess.run({command!r}, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
print(target.read_text())
'''
                record = json.loads(remote(peer.split('@')[1], code, timeout=600))
                if (record['status'] != 'complete' or record['run_id'] != job['run_id']
                        or record['plan_sha256'] != contract['plan_sha256']
                        or record['source_sha256'] != contract['helper_sha256']):
                    raise ValueError('Owner record differs from its frozen collection contract')
                if target.exists():
                    if json.loads(target.read_text()) != record:
                        raise ValueError('Owner audit differs from the retained copy')
                else:
                    atomic_json(target, record)
                records.append(dict(host=host, run_id=job['run_id'], path=str(target.relative_to(root)),
                                    sha256=sha256(target), checkpoint_sha256=record['checkpoint_sha256']))
            return records

        with ThreadPoolExecutor(3) as pool:
            records = sum(pool.map(collect, sorted({j['host'] for j in jobs})), [])
        atomic_json(out / 'result.json', dict(status='complete', created=time.time(), **contract, records=records,
            scope='Owner-local initial/final audits; verified recovery copies required. Only small JSON evidence copied to root. No new inference, updates, checkpoint payload transfers, final-test access or TPU.'))
        receipt = replicate_bundle([path for path in out.iterdir() if path.is_file()], root,
                                   'cubic27@t1v-n-a09f5679-w-1')
        atomic_json(out / 'replication.json', receipt)
        print(json.dumps(dict(status='complete', records=len(records), output=str(out))), flush=True)


if __name__ == '__main__':
    main()
