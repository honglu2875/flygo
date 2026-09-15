#!/usr/bin/env python3
"""Summarize one owned attachment endpoint without moving its checkpoint arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from flygo import _native
from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.optimizer import saved_clipping_mode,check_clipping_restore
from flygo.objectives import value_core_scale,saved_value_core_scale,check_objective_restore
from flygo.runtime import pin
from flygo.storage import GIB,StorageBudget


def initial_state(arrays, metadata):
    """Fingerprint numerical initialization; callers separately validate fixed training tags."""
    records = {}
    for key in sorted(arrays):
        if key in ('metadata', 'optimizer_clip_mode', 'objective_value_core_scale'):
            continue
        value = np.asarray(arrays[key])
        records[key] = dict(shape=list(value.shape), dtype=value.dtype.str,
                            sha256=hashlib.sha256(value.tobytes()).hexdigest())
    return dict(arrays=records, sampler=metadata['sampler'])


def head_state(initial,final,expected):
    """Check the effective head and frozen coefficients over a complete training run."""
    from flygo.readout import HeadMask,check_restore
    before,after=HeadMask.from_checkpoint(initial),HeadMask.from_checkpoint(final)
    contract=None if after is None else after.contract
    if contract!=expected or (None if before is None else before.contract)!=contract:
        raise ValueError('Checkpoint head mask differs from the registered trial')
    if after is None:return None
    check_restore(after,initial);check_restore(after,final)
    for name,enabled in after.parameter_masks().items():
        if not np.array_equal(initial['param/'+name][enabled==0],final['param/'+name][enabled==0]):
            raise ValueError('Disabled head coefficients changed during training')
    return contract


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpus',default='56,57,58,59')
    args=parser.parse_args();pin(list(map(int,args.cpus.split(','))))
    if Path(args.run_id).name!=args.run_id or args.run_id in ('.','..'):
        parser.error('Use a plain run ID')
    args.output.resolve().relative_to(args.root.resolve())
    if args.output.exists():raise ValueError('Endpoint summaries are immutable')
    plan=json.loads(args.plan.read_text());jobs=[job for job in plan['jobs'] if job['run_id']==args.run_id]
    if len(jobs)!=1:raise ValueError('Run must occur exactly once in the registered plan')
    job=jobs[0];run=args.root/'runs'/args.run_id
    status=json.loads((run/'status.json').read_text())
    if status.get('state')!='complete' or status.get('step')!=plan['updates']:
        raise ValueError('The declared learner endpoint is not complete')
    try:command=Path('/proc',str(status.get('pid')),'cmdline').read_bytes().decode().split('\0')
    except FileNotFoundError:command=[]
    if 'flygo.train' in command and args.run_id in command:
        raise ValueError('Wait for the owning learner to exit before closing its endpoint')
    with StorageBudget(args.root).reserve(files=1<<20,heap=2*GIB,purpose='attachment endpoint '+args.run_id):
        config=json.loads((run/'config.json').read_text())
        manifest=json.loads((args.root/'releases'/plan['release']/'manifest.json').read_text())
        expected=dict(seed=job['seed'],steps=plan['updates'],batch_size=plan['batch_size'],
            passes=job['passes'],groups=job['groups'],input_mode=job['mode'],
            input_map=str(args.root/plan['input_map']),rate=plan['rate'],epsilon=plan['epsilon'],
            warmup_steps=plan['warmup_steps'],rate_scales=plan['rate_scales'],clip=plan['clip'])
        clip_mode=job.get('clip_mode',plan.get('clip_mode','global'))
        if config['arguments'].get('clip_mode','global')!=clip_mode:
            raise ValueError('Learner clipping mode differs from its declared optimizer contract')
        scale=value_core_scale(job.get('value_core_scale',plan.get('value_core_scale',1.0)))
        if value_core_scale(config['arguments'].get('value_core_scale',1.0))!=scale:
            raise ValueError('Learner value core scale differs from the registered trial')
        expected_head=None
        if job.get('head_mask'):
            from flygo.readout import load_head_mask
            path=args.root/job['head_mask']
            if sha256(path)!=job['head_mask_sha256']:raise ValueError('Registered head mask changed')
            expected['head_mask']=str(path)
            expected_head=load_head_mask(path,graph_id=config['graph_id'],
                attachment_sha256=plan['input_map_sha256']).contract
            if config.get('head_mask')!=expected_head:raise ValueError('Learner head mask differs')
        if (config['dataset_id']!=manifest['dataset_id'] or config['cpus']!=job['cpus']
                or any(config['arguments'].get(key)!=value for key,value in expected.items())):
            raise ValueError('Learner differs from its declared attachment/optimizer contract')
        paths=[run/'checkpoints'/f'step-{step:08d}.npz' for step in (0,plan['updates'])]
        receipts=[]
        for path in paths:
            receipt=json.loads(path.with_suffix('.json').read_text())
            if receipt['replica_status']!='verified' or sha256(path)!=receipt['sha256']:
                raise ValueError('Checkpoint is corrupt or has no verified recovery copy')
            receipts.append(receipt)
        changes={}
        with np.load(paths[0],allow_pickle=False) as initial,np.load(paths[1],allow_pickle=False) as final:
            metadata=json.loads(final['metadata'].tobytes())
            initial_metadata=json.loads(initial['metadata'].tobytes())
            initialization=initial_state(initial,initial_metadata)
            for metadata_part,arrays_part in [(initial_metadata,initial),(metadata,final)]:
                if saved_clipping_mode(metadata_part)!=clip_mode:
                    raise ValueError('Checkpoint clipping mode differs from the registered trial')
                check_clipping_restore(clip_mode,arrays_part)
                if saved_value_core_scale(metadata_part)!=scale:
                    raise ValueError('Checkpoint value core scale differs from the registered trial')
                check_objective_restore(scale,arrays_part)
            head=head_state(initial,final,expected_head)
            numerical_runtime=metadata.get('numerical_runtime','rust-fp32-f64-norm-v1')
            if numerical_runtime!=initial_metadata.get('numerical_runtime','rust-fp32-f64-norm-v1'):
                raise ValueError('Numerical runtime changed during training')
            if head is not None:
                from flygo.fly import RustFly
                if numerical_runtime!=RustFly.numerical_runtime:raise ValueError('Use the qualified numerical source for this masked endpoint')
            if (metadata['dataset_id']!=manifest['dataset_id'] or metadata['input_contract']!=config['input_contract']
                    or int(initial['optimizer_step'])!=0 or int(final['optimizer_step'])!=plan['updates']):
                raise ValueError('Checkpoint contract or optimizer horizon differs')
            for key in final.files:
                if key.startswith('port/'):
                    np.testing.assert_array_equal(initial[key],final[key])
                if key.startswith('param/'):
                    before,after=initial[key],final[key];delta=after.astype(np.float64)-before
                    if not np.isfinite(delta).all():raise ValueError('Nonfinite parameter change')
                    changes[key[6:]]=dict(scalars=after.size,changed=int(np.count_nonzero(after!=before)),
                        delta_l2=float(np.linalg.norm(delta)),delta_max=float(np.abs(delta).max()))
            attachment=Path(config['arguments']['input_map'])
            if sha256(attachment)!=metadata['input_contract']['attachment_sha256']:
                raise ValueError('Input map differs from the checkpoint')
            with np.load(attachment,allow_pickle=False) as mapping:visual=len(mapping['sensors'])
            gain=final['param/input_gain'];original=initial['param/input_gain']
            for name,selection in [('visual',slice(None,visual)),('context',slice(visual,None))]:
                changes[name+'_input_gain']=dict(changed=int(np.count_nonzero(gain[selection]!=original[selection])),
                    quantiles=np.quantile(gain[selection],[0,.1,.5,.9,1]).tolist())
        log=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        endpoints=[row for row in log if row.get('kind')=='validation']
        diagnostics=[row for row in log if row.get('kind')=='diagnostics']
        if [row['step'] for row in endpoints]!=list(range(0,plan['updates']+1,plan['eval_every'])):
            raise ValueError('Registered fixed-slice evaluations are incomplete')
        result=dict(status='complete',created=time.time(),run_id=args.run_id,mode=job['mode'],seed=job['seed'],
            checkpoint_sha256=receipts[-1]['sha256'],checkpoint_receipts=receipts,
            initialization=initialization,
            dataset_id=manifest['dataset_id'],graph_id=metadata['graph_id'],input_contract=metadata['input_contract'],
            model_config=metadata['model_config'],training_contract=metadata['training_contract'],
            head_mask=head,numerical_runtime=numerical_runtime,
            native_sha256=sha256(Path(_native.__file__)),
            learning_curve=endpoints,parameter_changes=changes,
            labeled_training_exposures=plan['updates']*plan['batch_size'],
            diagnostic_forward_positions=sum(row['diagnostic_forward_positions'] for row in diagnostics),
            diagnostic_backward_positions=sum(row['diagnostic_backward_positions'] for row in diagnostics),
            validation_forward_positions=sum(sum(row[key]['positions'] for key in ('natural','novel','train'))+
                sum(value['positions'] for value in row.get('input_perturbations',{}).values()) for row in endpoints),
            final_activity=diagnostics[-1] if diagnostics else None,
            source_sha256=sha256(Path(__file__)),plan_sha256=sha256(args.plan),
            metrics_log_sha256=sha256(run/'metrics.jsonl'),
            scope='Owner-side endpoint accounting and fixed-slice diagnostics. Full validation is separate. No new inference, learning, grouping, checkpoint transfer, test examples or TPU use.')
        atomic_json(args.output,result)
        print(json.dumps(dict(status='complete',output=str(args.output),run_id=args.run_id)),flush=True)


if __name__=='__main__':main()
