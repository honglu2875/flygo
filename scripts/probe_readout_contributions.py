#!/usr/bin/env python3
"""Audit learned motor-to-head contributions using retained training responses."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB, StorageBudget


def relative_logits(logits, legal):
    """Remove the common shift separately for each position's legal actions."""
    count=legal.sum(axis=1,keepdims=True)
    if np.any(count==0):raise ValueError('Every position needs a legal action')
    mean=np.where(legal,logits,0).sum(axis=1,keepdims=True)/count
    return np.where(legal,logits-mean,0)


def policy_change(reference, candidate, legal):
    def log_policy(logits):
        masked=np.where(legal,logits,-np.inf)
        shifted=masked-masked.max(axis=1,keepdims=True)
        return shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
    before,after=log_policy(reference),log_policy(candidate)
    delta=np.where(legal,before,0)-np.where(legal,after,0)
    kl=(np.exp(before)*delta).sum(axis=1)
    return dict(mean_kl=float(kl.mean()),max_kl=float(kl.max()),
                top1_change_fraction=float(np.mean(before.argmax(axis=1)!=after.argmax(axis=1))))


def sequential_decode(raw, scale, policy, policy_bias, value, value_bias, *, head_mask=None, accumulator_dtype=np.float32):
    """Mirror ordered head accumulation and its FP32 boundary, including masks."""
    pooled=np.float32(0)+raw*scale
    logits=np.broadcast_to(policy_bias,(len(raw),len(policy_bias))).astype(accumulator_dtype).copy()
    linear=np.full(len(raw),value_bias.item(),accumulator_dtype)
    for group in range(pooled.shape[1]):
        selected=slice(None) if head_mask is None else head_mask.policy[:,group].astype(bool)
        rates=pooled[:,group].astype(accumulator_dtype)
        logits[:,selected]+=rates[:,None]*policy[selected,group].astype(accumulator_dtype)
        if head_mask is None or head_mask.value[group]:
            linear+=rates*accumulator_dtype(value[group])
    return logits.astype(np.float32),np.tanh(linear.astype(np.float32))


def energy(logits,legal):
    return float(np.mean(np.square(relative_logits(logits,legal)).sum(axis=1)/legal.sum(axis=1)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--run-id',help='Disambiguate multiple head variants sharing one seed')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root;pin([56,57,58,59])
    args.output.resolve().relative_to(root.resolve())
    if args.output.exists():raise ValueError('Readout diagnostics are immutable')
    plan=json.loads(args.plan.read_text())
    cases=[c for c in plan['cases'] if c['seed']==args.seed and (not args.run_id or c.get('run_id')==args.run_id)]
    if len(cases)!=1:raise ValueError('Select exactly one registered seed/run')
    case=cases[0]
    checkpoint=root/case['checkpoint'];response=root/case['responses'];report_path=root/case['report']
    selection=json.loads((response.parent/'selection.json').read_text())
    legal_path=root/plan['legal_responses'];legal_selection=json.loads((legal_path.parent/'selection.json').read_text())
    if selection!=legal_selection:raise ValueError('Response positions and legal masks differ')
    report=json.loads(report_path.read_text())
    record=next(r for r in report['records'] if r['case']=='current')
    receipt=json.loads(checkpoint.with_suffix('.json').read_text())
    if (report['status']!='complete' or receipt['replica_status']!='verified'
            or sha256(checkpoint)!=case['checkpoint_sha256'] or receipt['sha256']!=case['checkpoint_sha256']
            or record['checkpoint_sha256']!=receipt['sha256'] or sha256(response)!=record['response_file_sha256']
            or sha256(legal_path)!=plan['legal_responses_sha256']):
        raise ValueError('Checkpoint or response provenance differs')
    with StorageBudget(root).reserve(files=4<<20,heap=2*GIB,purpose='learned readout contribution audit'):
        with np.load(response,allow_pickle=False) as data:
            motors=data['motors'];raw={k:data[k] for k in ('current','neutral')}
            observed={k:(data[k+'_logits'],data[k+'_value']) for k in raw}
        with np.load(legal_path,allow_pickle=False) as data:legal=data['legal']
        with np.load(checkpoint,allow_pickle=False) as data:
            from flygo.readout import HeadMask
            metadata=json.loads(data['metadata'].tobytes());config=metadata['model_config']
            head_mask=HeadMask.from_checkpoint(data)
            runtime=metadata.get('numerical_runtime','rust-fp32-f64-norm-v1')
            if runtime=='rust-fp32-circuit-f64-head-reductions-v3':accumulator_dtype=np.float64
            elif runtime in ('rust-fp32-f64-norm-v1','rust-fp32-target-mass-fixed-norm-v2'):accumulator_dtype=np.float32
            else:raise ValueError('Unqualified readout reconstruction precision')
            if (None if head_mask is None else head_mask.contract)!=case.get('head_mask'):
                raise ValueError('Checkpoint head mask differs from the diagnostic plan')
            if (config['seed']!=args.seed or config['groups']!=2129 or config['actions']!=82
                    or config['steps']!=8 or config.get('rate_softness',0)
                    or config.get('readout_mean_scale',1)!=1 or int(data['optimizer_step'])!=1000):
                raise ValueError('Expected the declared hard-rate current-input endpoint')
            if metadata['input_contract']!=plan['input_contract']:raise ValueError('Input contract differs')
            np.testing.assert_array_equal(data['port/output_group'][motors],np.arange(2129))
            scale=data['port/output_scale'][motors]*data['param/readout_gain'][motors]
            policy=data['param/policy_weight'].reshape(82,2129);pb=data['param/policy_bias']
            value=data['param/value_weight'];vb=data['param/value_bias']
        errors={}
        for name in raw:
            reconstructed=sequential_decode(raw[name],scale,policy,pb,value,vb,
                head_mask=head_mask,accumulator_dtype=accumulator_dtype)
            for component,a,b in zip(('logits','value'),reconstructed,observed[name]):
                np.testing.assert_allclose(a,b,rtol=3e-4,atol=3e-6)
                errors[name+'/'+component]=float(np.max(np.abs(a-b)))
        graph=Path(json.loads((root/'runs/m4/graph.json').read_text())['path'])
        annotation_path=root/plan['motor_annotations']
        if sha256(annotation_path)!=plan['motor_annotations_sha256']:
            raise ValueError('Motor annotation table differs')
        annotation=json.loads(annotation_path.read_text())
        if annotation['annotation_sha256']!=sha256(graph/'annotations.feather'):
            raise ValueError('Source anatomical annotations differ')
        visual=(raw['current']-raw['neutral']).astype(np.float64)
        centered=visual-visual.mean(axis=0);variance=np.square(centered).sum(axis=0)
        order=np.argsort(variance)[::-1]
        # Analyze the real-valued linear maps using the retained FP32 coefficients.
        # This keeps small contributions visible separately from FP32 reduction error.
        w=policy.astype(np.float64)*scale;v=value.astype(np.float64)*scale
        if head_mask is not None:
            w=np.where(head_mask.policy,w,0);v=np.where(head_mask.value,v,0)
        total_policy=centered@w.T;total_value=centered@v
        current_logits=raw['current'].astype(np.float64)@w.T+pb
        current_linear=raw['current'].astype(np.float64)@v+vb
        policy_energy=energy(total_policy,legal);value_energy=float(np.square(total_value).mean())
        groups=[]
        for count in plan['leading_cell_counts']:
            selected=order[:count];removed_policy=visual[:,selected]@w[:,selected].T
            removed_value=visual[:,selected]@v[selected]
            centered_policy=centered[:,selected]@w[:,selected].T
            centered_value=centered[:,selected]@v[selected]
            cells=[dict(node=int(motors[i]),raw_variance_fraction=float(variance[i]/variance.sum()),
                        **annotation['motors'][str(motors[i])]) for i in selected]
            groups.append(dict(cells=cells,
                policy_residual_energy_ratio=(energy(total_policy-centered_policy,legal)/policy_energy if policy_energy else None),
                value_residual_energy_ratio=(float(np.square(total_value-centered_value).mean())/value_energy if value_energy else None),
                neutralize_visual_component=dict(policy=policy_change(current_logits,current_logits-removed_policy,legal),
                    value_rms_change=float(np.sqrt(np.square(np.tanh(current_linear)-np.tanh(current_linear-removed_value)).mean())))))
        result=dict(status='complete',created=time.time(),seed=args.seed,run_id=case.get('run_id'),checkpoint_sha256=receipt['sha256'],
            head_mask=None if head_mask is None else head_mask.contract,numerical_runtime=runtime,
            input_contract=metadata['input_contract'],positions=len(visual),selection=selection,
            reconstruction_errors=errors,centered_visual_policy_energy=policy_energy,
            centered_visual_pre_tanh_value_energy=value_energy,groups=groups,
            source_sha256=sha256(Path(__file__)),plan_sha256=sha256(args.plan),
            response_sha256=sha256(response),report_sha256=sha256(report_path),
            annotation_sha256=sha256(graph/'annotations.feather'),
            scope='Existing trained B weights and 256 training-family responses. No labels, neural forward/update, fitted groups, '
                  'validation, test examples or TPU. Neutralizing only the chosen motors visual component is a decoder intervention, '
                  'not a physiological circuit lesion. Residual energy ratios are not additive variance attributions and may exceed one '
                  'because motor contributions can cancel. Influence does not establish predictive usefulness.')
        atomic_json(args.output,result)
        print(json.dumps(dict(seed=args.seed,status='complete',output=str(args.output),groups=groups)),flush=True)


if __name__=='__main__':main()
