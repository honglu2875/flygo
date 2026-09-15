#!/usr/bin/env python3
"""Fit converged, regularized policy decoders to an immutable motor bank."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.optimize import minimize, _lbfgsb, _lbfgsb_py

from flygo.attachments import runtime_hashes
from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import StorageBudget, GIB

try:
    import readout_probe as probe
    import readout_convex as convex
except ModuleNotFoundError:
    from flygo import readout_probe as probe, readout_convex as convex


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root;out=args.output
    plan=json.loads(args.plan.read_text());pin(plan['cpus'])
    out.resolve().relative_to(root.resolve())
    case=next(c for c in plan['banks'] if c['seed']==args.seed)
    bank_path=root/plan['source_bank']/f'seed-{args.seed}'
    bank=json.loads((bank_path/'bank.json').read_text())
    if (sha256(bank_path/'bank.json')!=case['bank_sha256'] or bank['plan_sha256']!=plan['source_plan_sha256']
            or bank['runtime_sha256']!=runtime_hashes() or scipy.__version__!=plan['optimizer']['scipy']):
        raise ValueError('The cached representation or solver differs from the registered source')
    with StorageBudget(root).reserve(files=192<<20,heap=8*GIB,purpose='convex motor decoder seed '+str(args.seed)):
        out.mkdir(parents=True,exist_ok=False);started=time.time()
        source=dict(worker_sha256=sha256(Path(__file__)),probe_sha256=sha256(Path(probe.__file__)),
            convex_sha256=sha256(Path(convex.__file__)),scipy=scipy.__version__,numpy=np.__version__,
            lbfgsb_sha256=sha256(Path(_lbfgsb.__file__)),lbfgsb_python_sha256=sha256(Path(_lbfgsb_py.__file__)),
            plan_sha256=sha256(args.plan),bank_sha256=sha256(bank_path/'bank.json'))
        atomic_json(out/'source.json',source)
        raw={};labels={}
        for split in ('train','validation'):
            for name,key in [(split+'-motors.npy','motors_sha256'),(split+'-labels.npz','labels_sha256')]:
                if sha256(bank_path/name)!=bank['selection'][split][key]:raise ValueError('Motor bank bytes changed')
            raw[split]=np.load(bank_path/(split+'-motors.npy'),mmap_mode='r',allow_pickle=False)
            with np.load(bank_path/(split+'-labels.npz'),allow_pickle=False) as saved:
                labels[split]={name:saved[name] for name in ('baseline','legal','policy')}
            if len(raw[split])!=plan['training_positions' if split=='train' else 'validation_positions']:
                raise ValueError('Unexpected bank size')
        if sha256(bank_path/'normalization.npz')!=bank['normalization_sha256']:raise ValueError('Training statistics changed')
        with np.load(bank_path/'normalization.npz',allow_pickle=False) as saved:mean,scale=saved['mean'],saved['scale']
        baseline={split:{k:float(v.mean()) for k,v in probe.position_metrics(d['baseline'],d['legal'],d['policy']).items()}
                  for split,d in labels.items()}
        trials=[]
        for variant in plan['variants']:
            features={split:np.empty((len(x),0),np.float64) if variant=='bias-only'
                      else probe.transform(x,variant,mean,scale,top_k=128) for split,x in raw.items()}
            center,values,vectors=convex.decompose(features['train'])
            directory=out/variant;directory.mkdir()
            np.savez(directory/'basis.npz',mean=center,eigenvalues=values,eigenvectors=vectors)
            for ridge in plan['ridge']:
                target=directory/('ridge-'+format(ridge,'.4g'));target.mkdir()
                began=time.time()
                x=convex.condition(features['train'],center,values,vectors,ridge)
                data=labels['train'];obj=convex.RidgePolicyObjective(x,data['baseline'],data['legal'],data['policy'],values,ridge)
                start=np.zeros(obj.shape,np.float64)
                np.testing.assert_array_equal(data['baseline']+x@start[:-1]+start[-1],data['baseline'])
                initial_objective=obj(start.ravel())[0];iterations=0
                def progress(current):
                    nonlocal iterations
                    iterations+=1
                    if iterations%25==0:
                        atomic_json(out/'status.json',dict(state='fitting',pid=os.getpid(),variant=variant,ridge=ridge,
                            iterations=iterations,evaluations=obj.calls,latest=obj.history[-1],updated=time.time()))
                settings={k:v for k,v in plan['optimizer'].items() if k not in ('method','scipy')}
                result=minimize(obj,start.ravel(),jac=True,method=plan['optimizer']['method'],options=settings,callback=progress)
                objective,gradient=obj(result.x)
                matrix=result.x.reshape(obj.shape);params=convex.coefficients(matrix,values,vectors,ridge)
                # Independently check the original objective and its gradient.
                centered=features['train']-center
                logits=probe.forward(params,centered,data['baseline'])
                white_logits=data['baseline']+x@matrix[:-1]+matrix[-1]
                np.testing.assert_allclose(logits,white_logits,rtol=1e-10,atol=1e-9)
                ce,raw_gradient=probe.loss_and_grad(params,centered,data['baseline'],data['legal'],data['policy'])
                original=np.concatenate((raw_gradient['weight']+ridge*params['weight'],
                                          (raw_gradient['bias']+ridge*params['bias'])[None]),axis=0)
                np.testing.assert_allclose(original,convex.original_gradient(gradient.reshape(obj.shape),values,vectors,ridge),rtol=1e-7,atol=1e-10)
                penalty=ridge/2*sum(float(np.sum(a*a)) for a in params.values())
                np.testing.assert_allclose(objective,ce+penalty,rtol=1e-11,atol=1e-10)
                gap=float(np.sum(original*original)/(2*ridge))
                metrics={};position={}
                for split,d in labels.items():
                    predicted=logits if split=='train' else probe.forward(params,features[split]-center,d['baseline'])
                    position[split]=probe.position_metrics(predicted,d['legal'],d['policy'])
                    metrics[split]={k:float(v.mean()) for k,v in position[split].items()}
                np.savez(target/'checkpoint.npz',weight=params['weight'],bias=params['bias'],conditioned=matrix,ridge=ridge)
                np.savez(target/'position-metrics.npz',**{split+'/'+k:v for split,metrics_ in position.items() for k,v in metrics_.items()})
                atomic_json(target/'curve.json',obj.history)
                trial=dict(variant=variant,ridge=ridge,status='complete',optimizer_success=bool(result.success),
                    optimizer_message=str(result.message),iterations=int(result.nit),solver_evaluations=int(result.nfev),
                    objective_calls=obj.calls,fit_label_exposures=obj.calls*len(x),gradient_audit_label_exposures=len(x),
                    train_metric_exposures=len(raw['train']),validation_metric_exposures=len(raw['validation']),
                    initial_objective=initial_objective,objective=objective,penalty=penalty,
                    original_gradient_inf=float(np.max(np.abs(original))),optimization_gap_bound=gap,
                    sufficiently_converged=gap<=plan['acceptance']['max_gradient_gap_bound'],
                    coordinate_logit_max_error=float(np.max(np.abs(logits-white_logits))),
                    metrics=metrics,seconds=time.time()-began,checkpoint_sha256=sha256(target/'checkpoint.npz'),
                    position_metrics_sha256=sha256(target/'position-metrics.npz'),basis_sha256=sha256(directory/'basis.npz'),
                    feature_abs_max={s:float(np.max(np.abs(f),initial=0)) for s,f in features.items()})
                atomic_json(target/'result.json',trial);trials.append(trial)
                complete=len(trials)==len(plan['variants'])*len(plan['ridge'])
                atomic_json(out/'result.json',dict(status='complete' if complete else 'running',seed=args.seed,source=source,
                    baseline=baseline,trials=trials,seconds=time.time()-started,scope=plan['scope']))
                print(json.dumps(dict(seed=args.seed,variant=variant,ridge=ridge,converged=trial['sufficiently_converged'],
                    gap_bound=gap,iterations=trial['iterations'],train_kl=metrics['train']['policy_kl'],validation_kl=metrics['validation']['policy_kl'])),flush=True)
                del x,obj,centered,logits,white_logits
        atomic_json(out/'status.json',dict(state='complete',pid=os.getpid(),updated=time.time()))


if __name__=='__main__':main()
