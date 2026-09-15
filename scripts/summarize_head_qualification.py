#!/usr/bin/env python3
"""Verify all paired readout engineering evidence before registering scientific runs."""
import argparse
import json
from pathlib import Path
import time

from flygo.attachments import runtime_hashes
from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.storage import StorageBudget,GIB
from deploy_research import head_io_files


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root;out=args.output
    out.resolve().relative_to(root.resolve())
    if out.exists():raise ValueError('Qualification summaries are immutable')
    preparation=read(root/'ports/readout-masks-v1/result.json')
    source=read(root/'runs/readout-mask-build-v3/source.json')
    plan_path=root/'runs/readout-mask-qualification-v3/plan.json';plan=read(plan_path)
    if root/plan['environment']!=Path(source['environment']):raise ValueError('Numerical source differs')
    runtime=runtime_hashes()
    if runtime['native']!=source['native_sha256']:raise ValueError('Run this report under the qualified source')
    with StorageBudget(root).reserve(files=8<<20,heap=GIB,purpose='readout qualification summary'):
        archives={}
        for version in ('v1','v2','v3'):
            base=root/('runs/readout-mask-archive-'+version);receipt=read(base/'receipt.json')
            if receipt['status']!='complete':raise ValueError('Incomplete owner archive')
            for owner in receipt['records']:
                for row in owner['files']:
                    path=root/row['path']
                    if path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:raise ValueError('Owner archive changed')
            archives[version]=dict(path=str(base.relative_to(root)),receipt_sha256=sha256(base/'receipt.json'))
        rows=[];initials={};training=None
        gate_plan=dict(rate=.03,epsilon=1e-6,clip=1,batch_size=32,warmup_steps=100,rate_scales={'bias':.01},jobs=[])
        for seed in (4,5,6):
            owner=seed-3;archive=root/archives['v3']['path']/f'owner-{owner}'
            for arm in preparation['records']:
                name=arm['arm']
                numerical=archive/f'runs/readout-mask-qualification-v3/seed-{seed}'/name/'result.json'
                io=archive/f'runs/readout-mask-io-v3/seed-{seed}'/name/'result.json'
                n,r=read(numerical),read(io)
                if (n['status']!='complete' or len(n['records'])!=1 or n['runtime_sha256']!=runtime
                        or r['status']!='passed' or r['plan_sha256']!=sha256(plan_path)
                        or r['arm']!=name or r['head_mask']!=arm['contract']
                        or n['records'][0]['head_mask']!=arm['contract']
                        or r['model']['seed']!=seed or r['model']['steps']!=8
                        or r['model']['groups']!=2129 or r['model']['actions']!=82
                        or n['records'][0]['batch_size']!=32 or n['records'][0]['updates']!=3):
                    raise ValueError('Incomplete or mismatched actual head qualification')
                if training is None:training=r['training_contract']
                elif training!=r['training_contract']:raise ValueError('Training contracts differ across arms')
                initial=read(io.with_name('initial.json'))
                if initial!=initials.setdefault(seed,initial):raise ValueError('Cross-arm initial arrays or sampler differ')
                if len(initial['arrays'])!=31 or initial['dataset_id']!=n['dataset_id']:raise ValueError('Initial state coverage differs')
                job=dict(seed=seed,head_mask=arm['path'],qualification=str(numerical.relative_to(root)),io_qualification=str(io.relative_to(root)))
                gate_plan['jobs'].append(job)
                errors=n['records'][0]['errors']
                rows.append(dict(seed=seed,arm=name,owner=owner,numerical=job['qualification'],io=job['io_qualification'],
                    numerical_sha256=sha256(numerical),io_sha256=sha256(io),initial_sha256=r['initial_sha256'],
                    head_mask=arm['contract'],head_mask_path=arm['path'],head_mask_sha256=arm['sha256'],
                    forward_max_absolute={key:max(v for k,v in errors.items() if '/forward/'+key in k) for key in ('states','logits','value')},
                    arithmetic={size:dict(warm_flops_per_position=a['warm_flops_per_position'],
                        decoder_flops=a['nominal']['parts']['heads']) for size,a in r['arithmetic'].items()},
                    engineering_updates=r['engineering_updates']))
        head_io_files(gate_plan,root)
        history=[]
        for version in ('v1','v2'):
            cases=[]
            for seed in (4,5,6):
                base=root/archives[version]['path']/f'owner-{seed-3}'/f'runs/readout-mask-qualification-{version}/seed-{seed}'
                for arm in preparation['records']:
                    case=base/arm['arm'];result=case/'result.json'
                    passed=result.exists() and read(result)['status']=='complete'
                    cases.append(dict(seed=seed,arm=arm['arm'],passed=passed,
                        failure=None if passed else read(case/'status.json')))
            history.append(dict(version=version,passed=sum(c['passed'] for c in cases),cases=cases))
        build=root/'runs/readout-mask-build-v3'
        if (any(c['exit_code'] for c in read(build/'commands.json'))
                or read(build/'python-tests-frozen.json')['status']!='passed'
                or read(build/'go-parity-test.json')['status']!='passed'):
            raise ValueError('Required build or tests did not pass')
        result=dict(status='passed',created=time.time(),source=source,runtime_sha256=runtime,
            source_sha256=sha256(Path(__file__)),deployment_helper_sha256=sha256(Path(__file__).with_name('deploy_research.py')),
            graph_id=preparation['plan']['graph_id'],dataset_id=initials[4]['dataset_id'],
            preparation_sha256=sha256(root/'ports/readout-masks-v1/result.json'),plan_sha256=sha256(plan_path),
            archives=archives,records=rows,training_contract=training,previous_numerical_cohorts=history,
            initial_pairing=dict(seeds=[4,5,6],arrays_per_seed=31,all_four_arms_identical=True),
            unit_tests=dict(rust=50,python_core=110,python_core_note='109 pass in discovery; the optional direct Rust/Python Go test passes separately with its executable configured.',
                rust_log_sha256=sha256(build/'rust-tests.log'),python_log_sha256=sha256(build/'python-tests-frozen.log'),go_parity_log_sha256=sha256(build/'go-parity-test.log')),
            precision=dict(circuit='FP32 states and parameters',decoder='FP64 products and accumulators, FP32 head outputs and gradients',
                policy_loss='FP64 normalization and reduction with actual teacher mass',optimizer='FP32 Adam, fixed-order FP64 global norm',
                reference='Independent CPU JAX FP32/highest reference; stable log-softmax shift'),
            limitations=['Engineering checks only; no scientific masked-head endpoint or advantage over dense/CNN is established.',
                'Arithmetic samples use a step-3 engineering checkpoint, not a trained endpoint distribution or CNN matching budget.',
                'Rust skips disabled head coefficients; masked dense JAX matmul has no claimed executed-work reduction.',
                'The five-cell value group is response-motivated; soma side is a proxy. Neither is a proven biological Go role.',
                'Checkpoint payloads remain on owner workers; the archives contain metadata, not payload replicas.',
                'TPU use remains paused. The new precision and mask variants require separate actual TPU qualification.'])
        atomic_json(out,result)
        print(json.dumps(dict(status='passed',output=str(out),cases=len(rows),paired_seeds=3)))


if __name__=='__main__':main()
