#!/usr/bin/env python3
"""Deploy one immutable environment and frozen corpus for an explicit CPU experiment plan."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from cluster import snapshot, SSH, PYTHON
from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release
from flygo.data.cache import cache_directory
from flygo.replication import replicate_bundle
from flygo.runtime import cpu_profile,pin


def numerical_protocol_files(plan,job,root,report,records):
    """Require both registered checks when a trial uses the combined v2 report."""
    from flygo.qualify import sha256
    from flygo.schedule import Schedule
    from flygo.objectives import value_core_scale
    if not plan.get('numerical_plan'):
        if report.get('numerical_protocol') or len(records)!=1:
            raise ValueError('A revised numerical protocol needs an explicit launch contract')
        return set()
    path=root/plan['numerical_plan'];spec=json.loads(path.read_text())
    digest=sha256(path)
    if (digest!=plan['numerical_plan_sha256'] or report.get('numerical_plan_sha256')!=digest
            or report.get('status')!='complete' or report.get('numerical_protocol')!='combined-v2'
            or len(records)!=2 or records!=report.get('records') or len(report.get('constituent_reports',[]))!=2
            or spec['source']!=plan['environment'] or spec['platform']!='cpu'
            or spec['input_map']!=plan['input_map'] or job['mode']!=spec['input_mode']
            or job['seed'] not in spec['seeds'] or job['arm'] not in spec['arms']):
        raise ValueError('Combined numerical evidence differs from its registered protocol')
    schedule=Schedule(job.get('rate',plan['rate']),job.get('warmup_steps',plan.get('warmup_steps',0)),
        job.get('decay_until',plan.get('decay_until',0)),job.get('final_rate_ratio',plan.get('final_rate_ratio',.1)))
    expected_rates={'aligned-peak':[schedule.peak]*3,
                    'free-warmup':[schedule.rate(i) for i in range(1,4)]}
    if (schedule.warmup_steps!=spec['warmup_steps'] or spec['rate']!=schedule.peak
            or spec['steps']!=job['passes'] or spec['batch_size']!=plan['batch_size']
            or {p['name']:p['rates'] for p in spec['protocols']}!=expected_rates):
        raise ValueError('Numerical trajectories differ from the actual launch schedule')
    parameters=('edge','leak','bias','input_gain','readout_gain','policy_weight','policy_bias','value_weight','value_bias')
    checks={f'{step}/{family}/{name}' for step in range(3)
        for family,names in [('forward',('states','logits','value')),('loss',('policy_loss','value_loss')),
                             *[(f,parameters) for f in ('gradient','param','first','second')]] for name in names}
    clip_mode=job.get('clip_mode',plan.get('clip_mode','global'))
    scale=value_core_scale(job.get('value_core_scale',plan.get('value_core_scale',1.0)))
    if spec.get('measure_clipping'):
        checks.update(f'{step}/{family}/{name}' for step in range(3)
                      for family in ('group_norm','clip_factor') for name in parameters)
        checks.update(f'{step}/clipping/norm' for step in range(3))
        if (spec.get('factor')=='optimizer clipping scope only' and clip_mode!=job['arm']) or (
                spec.get('factor') not in ('optimizer clipping scope only','supervised value-to-circuit gradient scale only')):
            raise ValueError('Clipping evidence differs from the actual trial mode')
    if scale not in [value_core_scale(v) for v in spec.get('value_core_scales',[1.0])]:
        raise ValueError('Value core scale differs from the registered numerical protocol')
    if spec.get('factor')=='supervised value-to-circuit gradient scale only' and (
            clip_mode!=spec['clip_mode'] or scale!=value_core_scale(spec['value_core_arms'][job['arm']])):
        raise ValueError('Value core evidence differs from the actual trial arm')
    expected_optimizer=dict(rate=schedule.peak,epsilon=job.get('epsilon',plan['epsilon']),
        clip=job.get('clip',plan.get('clip',1.)),rate_scales=job.get('rate_scales',plan.get('rate_scales',{})),
        clip_mode=clip_mode,value_core_scale=scale)
    files={path};seen=set()
    for evidence,record in zip(report['constituent_reports'],records):
        constituent=Path(evidence['path']);constituent.resolve().relative_to(root.resolve())
        if sha256(constituent)!=evidence['sha256']:
            raise ValueError('Numerical constituent evidence changed')
        child=json.loads(constituent.read_text());protocol=record.get('numerical_protocol')
        alignment=[dict(step=i,arrays=27,exact=True) for i in range(3)] if protocol=='aligned-peak' else []
        if (protocol not in expected_rates or protocol in seen or child.get('records')!=[record]
                or child.get('status')!='complete' or child.get('numerical_protocol')!=protocol
                or child.get('numerical_plan_sha256')!=digest
                or child.get('script_sha256')!=report.get('qualifier_sha256')
                or child.get('registered_protocol')!=next(p for p in spec['protocols'] if p['name']==protocol)
                or any(child.get(key)!=report.get(key) for key in
                       ('runtime_sha256','dataset_id','graph_id'))
                or record['actual_rates']!=expected_rates[protocol]
                or record['aligned_checkpoints']!=alignment or set(record['errors'])!=checks
                or record['model']!=records[0]['model'] or record['model']['seed']!=job['seed']
                or record['updates']!=3 or record['batch_size']!=plan['batch_size']
                or {'clip_mode':'global','value_core_scale':1.0,**record['optimizer']}!=expected_optimizer):
            raise ValueError('Both complete numerical trajectories must cover the actual trial')
        seen.add(protocol);files.add(constituent)
    return files


def head_io_files(plan,root):
    """Bind expensive masked trials to their actual recovery and pairing evidence."""
    from flygo.attachments import runtime_hashes
    from flygo.qualify import sha256
    from flygo.schedule import Schedule
    from flygo.objectives import value_core_scale
    files=set();initial_by_seed={}
    for job in plan['jobs']:
        if not job.get('head_mask'):continue
        if not job.get('io_qualification'):
            raise ValueError('Masked trials need an actual IO qualification')
        path=root/job['io_qualification'];report=json.loads(path.read_text())
        qualification=root/job['qualification']
        native=json.loads(qualification.read_text())
        record=next((r for r in native['records'] if r['model']['seed']==job['seed']
            and r.get('input_contract')==report.get('input_contract')
            and r.get('head_mask')==report.get('head_mask')),None)
        schedule=Schedule(job.get('rate',plan['rate']),job.get('warmup_steps',plan.get('warmup_steps',0)),
            job.get('decay_until',plan.get('decay_until',0)),job.get('final_rate_ratio',plan.get('final_rate_ratio',.1)))
        training=dict(batch_size=plan['batch_size'],rate=job.get('rate',plan['rate']),
            clip=job.get('clip',plan.get('clip',1.0)),epsilon=job.get('epsilon',plan['epsilon']),
            rate_scales=job.get('rate_scales',plan.get('rate_scales',{})),schedule=schedule.contract(),
            clip_mode=job.get('clip_mode',plan.get('clip_mode','global')),
            value_core_scale=value_core_scale(job.get('value_core_scale',plan.get('value_core_scale',1.0))))
        if (record is None or native.get('status')!='complete' or report.get('status')!='passed' or report['seed']!=job['seed']
                or report['runtime_sha256']!=runtime_hashes() or report['qualification_sha256']!=sha256(qualification)
                or report['head_mask']!=record['head_mask'] or report['input_contract']!=record['input_contract']
                or report['model']!=record['model'] or {'clip_mode':'global','value_core_scale':1.0,**report['training_contract']}!=training
                or report['fresh_process']['status']!='passed'
                or report['fresh_process']['runtime_sha256']!=runtime_hashes()):
            raise ValueError('Head recovery qualification differs from the trial contract')
        if training['clip_mode']!='global' and (report['fresh_process'].get('clip_mode')!=training['clip_mode']
                or report['fresh_process'].get('wrong_mode_restore_rejected') is not True):
            raise ValueError('Group clipping needs exact recovery and mode-mismatch evidence')
        if training['value_core_scale']!=1 and (report['fresh_process'].get('value_core_scale')!=training['value_core_scale']
                or report['fresh_process'].get('wrong_scale_restore_rejected') is not True):
            raise ValueError('Value core scaling needs exact recovery and scale-mismatch evidence')
        for name,key in [('initial.json','initial_sha256'),('expected.json','expected_sha256')]:
            evidence=path.with_name(name)
            if sha256(evidence)!=report[key]:raise ValueError('Head recovery evidence changed')
            files.add(evidence)
        initial=json.loads(path.with_name('initial.json').read_text())
        if initial!=initial_by_seed.setdefault(job['seed'],initial):
            raise ValueError('Paired head trials have different initial arrays or samplers')
        files.add(path)
    return files


def attachment_files(plan,root,environment):
    """Validate the exact learner seed and numerical contract before deployment."""
    if not plan.get('input_map'):
        return set()
    from flygo.attachments import load_attachment,input_contract,require_qualification,runtime_hashes
    from flygo.fly import FlyConfig
    from flygo.qualify import sha256
    code='from flygo.attachments import runtime_hashes;import json;print(json.dumps(runtime_hashes()))'
    qualified=json.loads(subprocess.check_output([sys.executable,'-B','-c',code],text=True,
        env={**os.environ,'PYTHONPATH':str(environment/'site-packages'),'OPENBLAS_NUM_THREADS':'1'},timeout=30))
    if qualified!=runtime_hashes():
        raise ValueError('Controller and selected attachment learner have different numerical sources')
    path=root/plan['input_map']
    if plan.get('input_map_sha256') and sha256(path)!=plan['input_map_sha256']:
        raise ValueError('Input map differs from the registered attachment')
    if plan.get('reference_analysis') and sha256(root/plan['reference_analysis'])!=plan['reference_analysis_sha256']:
        raise ValueError('Adopted paired analysis differs from the registered reference')
    manifest=json.loads((root/'releases'/plan['release']/'manifest.json').read_text())
    graph=json.loads((root/'runs/m4/graph.json').read_text())
    graph_manifest=json.loads((Path(graph['path'])/'manifest.json').read_text())
    adapter,_,receipt=load_attachment(path,graph_id=graph_manifest['graph_id'],dataset_id=manifest['dataset_id'])
    files={path,path.with_suffix('.json')}
    if plan.get('seed1_reference') and sha256(root/plan['seed1_reference'])!=plan['seed1_analysis_sha256']:
        raise ValueError('Adopted exploratory result differs from the registered reference')
    for job in plan['jobs']:
        if job.get('ports') or job.get('model','fly')!='fly' or job.get('backend','cpu')!='cpu':
            raise ValueError('Attachment deployment supports only separately qualified CPU fly trials')
        qualification=root/(job.get('qualification') or plan['qualifications'][str(job['seed'])])
        report=json.loads(qualification.read_text())
        contract=input_contract(receipt,job['mode'])
        config=FlyConfig(steps=job['passes'],groups=job.get('groups',plan['groups']),
            features=adapter.features,threads=len(job['cpus']),seed=job['seed'],
            rate_softness=job.get('rate_softness',plan.get('rate_softness',0.0)),
            readout_mean_scale=job.get('readout_mean_scale',plan.get('readout_mean_scale',1.0)))
        head_mask=None
        if job.get('head_mask'):
            from flygo.readout import load_head_mask
            mask_path=root/job['head_mask']
            if sha256(mask_path)!=job['head_mask_sha256']:
                raise ValueError('Head mask differs from the registered output attachment')
            head_mask=load_head_mask(mask_path,graph_id=graph_manifest['graph_id'],attachment_sha256=receipt['sha256'])
            files.update((mask_path,mask_path.with_suffix('.json')))
        require_qualification(qualification,contract,config,batch_size=plan['batch_size'],
            rate=job.get('rate',plan['rate']),epsilon=job.get('epsilon',plan['epsilon']),
            clip=job.get('clip',plan.get('clip',1.0)),rate_scales=job.get('rate_scales',plan.get('rate_scales',{})),head_mask=head_mask,
            clip_mode=job.get('clip_mode',plan.get('clip_mode','global')),
            value_core_scale=job.get('value_core_scale',plan.get('value_core_scale',1.0)))
        records=[r for r in report['records'] if r['input_contract']==contract
                 and r.get('head_mask')==(None if head_mask is None else head_mask.contract)]
        if (report['dataset_id']!=manifest['dataset_id'] or report['graph_id']!=graph_manifest['graph_id']
                or not records or any(r['model']['seed']!=job['seed'] for r in records)):
            raise ValueError('Attachment qualification does not cover this actual seed and corpus')
        files.update(numerical_protocol_files(plan,job,root,report,records))
        files.add(qualification)
    return files


def main():
    from flygo.objectives import value_core_scale
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    parser.add_argument('--source', type=Path, help='Reuse one immutable source environment across successive study waves')
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
        if 'replica_host' in job and (job['host'] not in (1,2,3)
                or job['replica_host'] not in (1,2,3) or job['replica_host']==job['host']):
            raise ValueError('Checkpoint relay requires two distinct worker hosts')
        allocations[job['host']] |= cpus

    environment = args.source or snapshot(root)
    if not (environment/'snapshot.json').is_file():
        raise ValueError('An explicit source must be a published immutable environment')
    if plan.get('environment') and environment.resolve()!=(root/plan['environment']).resolve():
        raise ValueError('Selected source differs from the registered environment')
    # Operational helpers may need newer transfer code while the scientific
    # learner retains its previously qualified numerical implementation.
    replica_environment=root/plan['replica_environment'] if plan.get('replica_environment') else environment
    if not (replica_environment/'snapshot.json').is_file():
        raise ValueError('Replica source must be a published immutable environment')
    subprocess.run([str(root/'venv/bin/python'),'-B','-c',
        'from flygo.replication import replicate_file, replicate_stream'],check=True,timeout=30,
        env={**os.environ,'PYTHONPATH':str(replica_environment/'site-packages'),'OPENBLAS_NUM_THREADS':'1'})
    release = root / 'releases' / plan['release']
    manifest = json.loads((release / 'manifest.json').read_text())
    if json.loads((release / 'replication.json').read_text())['status'] != 'passed':
        raise ValueError('The frozen corpus must have a verified second copy')
    graph_record = root / 'runs/m4/graph.json'
    graph_path = Path(json.loads(graph_record.read_text())['path'])
    files = {graph_record, *(root / record['path'] for record in manifest['records'])}
    files.update(attachment_files(plan,root,environment))
    files.update(head_io_files(plan,root))
    if plan.get('engineering_qualification'):
        from flygo.qualify import sha256
        evidence=root/plan['engineering_qualification']
        if (sha256(evidence)!=plan['engineering_qualification_sha256']
                or json.loads(evidence.read_text()).get('status')!='passed'):
            raise ValueError('Registered engineering qualification changed or did not pass')
        files.add(evidence)
    if plan.get('io_qualifications'):
        from flygo.qualify import sha256
        for job in plan['jobs']:
            path=root/plan['io_qualifications'][str(job['seed'])]
            report=json.loads(path.read_text())
            if (report['status']!='passed' or report['seed']!=job['seed']
                    or report['plan_sha256']!=sha256(args.config)):
                raise ValueError('Initial pairing and recovery gate does not cover this launch plan')
            files.add(path)
    for job in plan['jobs']:
        if job.get('ports'):
            path=root/job['ports'];files.update((path,path.with_suffix('.json')))
    for directory in (environment, graph_path, release):
        # Interpreter caches are host-local byproducts, outside snapshot identity.
        files.update(path for path in directory.rglob('*')
                     if path.is_file() and '__pycache__' not in path.parts)
    if plan.get('feature_cache',False):
        # Retain the maps and their reader leases through bundle replication.
        cached_data=load_release(root,release/'manifest.json',cache=True)
        cache=cache_directory(root,manifest['dataset_id'])
        files.update(path for path in cache.iterdir() if path.is_file())
    output = root / 'runs' / plan['name']
    if output.exists():
        if not args.resume_launch or json.loads((output/'plan.json').read_text())!=plan:
            raise ValueError('Existing plan is immutable; use --resume-launch only with the identical plan')
    else:
        output.mkdir(parents=True)
        atomic_json(output / 'plan.json', plan)
    helper=output/'replicate_checkpoints.py'
    if not helper.exists():
        shutil.copyfile(Path(__file__).with_name('replicate_checkpoints.py'),helper)
    if not (output/'deploy_research.py').exists():
        shutil.copyfile(Path(__file__),output/'deploy_research.py')

    def deploy(host):
        if host:
            transfer=replicate_bundle(sorted(files), root, f'cubic27@t1v-n-a09f5679-w-{host}')
            atomic_json(output/f'host-{host}-bundle.json',transfer)
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
                        steps=plan['updates'],batch_size=plan['batch_size'],rate=job.get('rate',plan['rate']),
                        clip=job.get('clip',plan.get('clip',1.0)),clip_mode=job.get('clip_mode',plan.get('clip_mode','global')),
                        value_core_scale=value_core_scale(job.get('value_core_scale',plan.get('value_core_scale',1.0))),backend='cpu',
                        diagnostics_every=plan.get('diagnostics_every',0),
                        ports=str(root/job['ports']) if job.get('ports') else None,
                        rate_scales=job.get('rate_scales',plan.get('rate_scales',{})),
                        threads=len(job['cpus']),eval_every=plan['eval_every'],checkpoint_every=plan['checkpoint_every'],
                        eval_positions=plan.get('eval_positions',2048),eval_batch_size=plan.get('eval_batch_size',32))
                    for key,default in (('warmup_steps',0),('decay_until',0),('final_rate_ratio',.1),('diagnostic_batch_size',32),('rate_softness',0.0),('readout_mean_scale',1.0),('epsilon',1e-8)):
                        expected[key]=job.get(key,plan.get(key,default))
                    if plan.get('input_map'):
                        expected.update(input_map=str(root/plan['input_map']),input_mode=job['mode'],
                            qualification=str(root/(job.get('qualification') or plan['qualifications'][str(job['seed'])])))
                    expected['head_mask']=str(root/job['head_mask']) if job.get('head_mask') else None
                    valid=(config.get('dataset_id')==manifest['dataset_id'] and config.get('cpus')==job['cpus']
                           and all(arguments.get(key,dict(clip=1.0,clip_mode='global',value_core_scale=1.0,backend='cpu',diagnostics_every=0,rate_scales={},
                                    warmup_steps=0,decay_until=0,final_rate_ratio=.1,diagnostic_batch_size=32,rate_softness=0.0,readout_mean_scale=1.0,epsilon=1e-8).get(key))==value
                                   for key,value in expected.items()))
                    if not valid or not (old['status'].get('state')=='complete' or
                            (old['live'] and 'flygo.train' in old['command'] and run_id in old['command'])):
                        raise ValueError('Cannot adopt a mismatched or stopped trial: '+run_id)
                    record={**job,'run_id':run_id,'environment':str(environment),
                            'dataset_id':manifest['dataset_id'],'adopted_pid':old['status'].get('pid')}
                    atomic_json(output/(run_id+'.job.json'),record)
                    results.append(record);print(json.dumps(record),flush=True)
                    continue
            command = ['env', 'PYTHONPATH=' + str(environment / 'site-packages'),
                       'PYTHONDONTWRITEBYTECODE=1', 'JAX_PLATFORMS=cpu', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1',
                       PYTHON, '-u', '-m', 'flygo.train', '--root', str(root),
                       '--release', plan['release'], '--run-id', run_id,
                       '--passes', str(job['passes']), '--seed', str(job['seed']),
                       '--groups',str(job.get('groups',plan.get('groups',656))),
                       '--steps', str(plan['updates']), '--batch-size', str(plan['batch_size']),
                       '--rate', str(job.get('rate',plan['rate'])), '--threads', str(len(job['cpus'])),
                       '--clip',str(job.get('clip',plan.get('clip',1.0))),
                       '--rate-scales',json.dumps(job.get('rate_scales',plan.get('rate_scales',{}))),
                       '--diagnostics-every',str(plan.get('diagnostics_every',0)),
                       '--cpus', ','.join(map(str, job['cpus'])), '--peer', '',
                       '--eval-every', str(plan['eval_every']),
                       '--eval-positions',str(plan.get('eval_positions',2048)),
                       '--eval-batch-size',str(plan.get('eval_batch_size',32)),
                       '--checkpoint-every', str(plan['checkpoint_every'])]
            for key in ('warmup_steps','decay_until','final_rate_ratio','diagnostic_batch_size','rate_softness','readout_mean_scale','epsilon','clip_mode'):
                if key in job or key in plan:
                    command+=['--'+key.replace('_','-'),str(job.get(key,plan.get(key)))]
            scale=value_core_scale(job.get('value_core_scale',plan.get('value_core_scale',1.0)))
            if scale!=1:
                command+=['--value-core-scale',str(scale)]
            if job.get('ports'):command+=['--ports',str(root/job['ports'])]
            if plan.get('input_map'):
                command+=['--input-map',str(root/plan['input_map']),'--input-mode',job['mode'],
                          '--qualification',str(root/(job.get('qualification') or plan['qualifications'][str(job['seed'])]))]
            if job.get('head_mask'):command+=['--head-mask',str(root/job['head_mask'])]
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
    command = ['taskset','--cpu-list','116',str(root / 'venv/bin/python'), '-B', str(helper),
               '--root', str(root), '--jobs', str(output / 'jobs.json'),
               '--output', str(output / 'replicas')]
    with (output / 'replicas.log').open('a') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, 'OPENBLAS_NUM_THREADS': '1',
                                        'PYTHONPATH':str(replica_environment/'site-packages')})
    atomic_json(coordinator, dict(pid=process.pid,environment=str(replica_environment)))


if __name__ == '__main__':
    main()
