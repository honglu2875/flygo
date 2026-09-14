#!/usr/bin/env python3
"""Bounded, qualified motor-embedding pilot with fixed before/after response probes."""
import argparse
from dataclasses import asdict
import json,time,os
from pathlib import Path
import numpy as np
import flygo
from flygo import _native
from flygo.checkpoint import save_checkpoint,load_checkpoint
from flygo.contrastive import BranchObjective,VERSION as OBJECTIVE_VERSION
from flygo.data.branches import load_bank,family_sample,quartet_views
from flygo.data.corpus import atomic_json,identity
from flygo.fly import FlyConfig,RustFly,initialize,load_graph,PARAMETERS
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB,StorageBudget
from probe_biological_ports import correlations,residualize


class PairSampler:
    def __init__(self,features,pairs,renderer,seed):
        self.features,self.pairs,self.renderer=features,pairs,renderer
        self.rng=np.random.default_rng(seed);self.batches=0
    def batch(self,count):
        indices=family_sample(self.pairs,count,self.rng)
        self.batches+=1
        return quartet_views(self.renderer,self.features[indices],self.rng)
    def state(self):return dict(rng=self.rng.bit_generator.state,batches=self.batches)
    def restore(self,state):self.rng.bit_generator.state=state['rng'];self.batches=state['batches']


def predict(model,x,batch=32):
    response=[];score=[]
    for start in range(0,len(x),batch):
        pooled,value,_=model.native.embedding(model._input(x[start:start+batch]),model.config.steps)
        response.append(pooled.reshape(model.config.groups,-1).T.copy());score.append(value.copy())
    return np.concatenate(response),np.concatenate(score)


def evaluate(model,objective,view_probe,visual_probe,features,geometry,path,stage,families):
    r,v=predict(model,view_probe)
    metrics,_,_=objective(r,v)
    scores=v.astype(np.float64).reshape(-1,4)
    gap=scores[:,:2].mean(axis=1)-scores[:,2:].mean(axis=1)
    correct=(gap>0)+.5*(gap==0)
    families=np.asarray(families)
    by_family={name:float(correct[families==name].mean()) for name in sorted(set(families))}
    metrics.update(pair_accuracy_by_family=by_family,macro_family_accuracy=float(np.mean(list(by_family.values()))))
    response,_=predict(model,visual_probe)
    nuisance=np.column_stack([features[...,0].mean(axis=(1,2)),features[...,1].mean(axis=(1,2)),
                             features[:,0,0,8],features[:,0,0,10],features[...,11].mean(axis=(1,2))])
    raw,valid,summary=correlations(response)
    conditioned,design_rank=residualize(response,nuisance)
    corr,cvalid,cs=correlations(conditioned)
    np.savez(path/(stage+'-responses.npz'),rates=response,motor_body_id=geometry['motor_body_id'])
    np.savez(path/(stage+'-correlations.npz'),raw=raw,conditioned=corr,valid=valid,conditioned_valid=cvalid,
             motor_body_id=geometry['motor_body_id'])
    std=response.astype(np.float64).std(axis=0,ddof=1)
    metrics.update(response=summary,conditioned_response=cs,nuisance_rank=design_rank,
        varying_by_superclass={k:int(np.sum(std[geometry['motor_superclass']==k]>1e-6))
                               for k in sorted(set(geometry['motor_superclass']))})
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=['learned','frozen'],required=True)
    p.add_argument('--cpus',default=','.join(map(str,range(32,56))))
    args=p.parse_args();cpus=list(map(int,args.cpus.split(',')));pin(cpus)
    args.output.resolve().relative_to(args.root.resolve())
    with StorageBudget(args.root).reserve(files=192<<20,heap=12*GIB,purpose='bounded spherical motor-embedding pilot'):
        started=time.time();out=args.output
        bank,arrays,pairs,renderer,ports,geometry=load_bank(args.bank)
        qualification=json.loads(args.qualification.read_text())
        pyhash={name:sha256(Path(flygo.__file__).parent/name) for name in
                ('fly.py','vision.py','contrastive.py','data/branches.py','jax/model.py')}
        if (qualification['status']!='passed' or qualification.get('batch_size')!=32
                or qualification['bank_sha256']!=sha256(args.bank/'result.json')
                or qualification['native_sha256']!=sha256(Path(_native.__file__)) or qualification['python_sha256']!=pyhash):
            raise ValueError('This exact bank, objective and backend need a passing full-graph qualification')
        graph=load_graph(Path(json.loads((args.root/'runs/m4/graph.json').read_text())['path']))
        if graph['manifest']['graph_id']!=bank['graph_id']:raise ValueError('Graph differs from bank')
        cfg=FlyConfig(steps=8,features=len(geometry['sensors']),groups=len(geometry['motors']),actions=1,threads=len(cpus))
        _,params=initialize(graph,cfg);model=RustFly(graph,cfg,ports=ports,params=params)
        objective=BranchObjective()
        expected_model=asdict(cfg);qualified_model=qualification['model'].copy()
        expected_model.pop('threads');qualified_model.pop('threads')
        if expected_model!=qualified_model or qualification['objective']!=asdict(objective):
            raise ValueError('Model or objective configuration differs from qualification')
        epsilon=qualification['optimizer']['epsilon']
        if qualification['optimizer']['rate']!=.01 or qualification['optimizer']['clip']!=1.:
            raise ValueError('Qualified optimizer rate or clip differs from the pilot')
        scales=dict(bias=.01,readout_gain=0.,policy_weight=0.,policy_bias=0.,value_bias=0.)
        if args.variant=='frozen':scales.update(edge=0.,bias=0.,leak=0.,input_gain=0.)
        train=np.flatnonzero(arrays['split']=='train');probe=np.flatnonzero(arrays['split']=='probe')
        train_pairs=[pairs[i] for i in train]
        if set(p['opening_family'] for p in train_pairs)&set(pairs[i]['opening_family'] for i in probe):
            raise ValueError('Training/probe family overlap')
        probe_rng=np.random.default_rng(918426)
        probe=probe_rng.choice(probe,min(128,len(probe)),replace=False)
        probe_families=[pairs[i]['opening_family'] for i in probe]
        view_probe=quartet_views(renderer,arrays['features'][probe],np.random.default_rng(918427),augment=False)
        # Correlations use unique, unaugmented visual positions, not duplicate positive views.
        original=arrays['features'][probe].reshape(-1,9,9,12)
        _,indices=np.unique(np.ascontiguousarray(original[...,:8]).reshape(len(original),-1),axis=0,return_index=True)
        original=original[np.sort(indices)];visual_probe=renderer.render(original)
        sampler=PairSampler(arrays['features'][train],train_pairs,renderer,918425)
        contract=dict(schema_version=1,dataset_id=bank['dataset_id'],graph_id=bank['graph_id'],
            bank_sha256=sha256(args.bank/'result.json'),qualification_sha256=sha256(args.qualification),
            native_sha256=sha256(Path(_native.__file__)),python_sha256=pyhash,
            source_path=str(Path(flygo.__file__).parent),script_sha256=sha256(Path(__file__)),
            variant=args.variant,model=asdict(cfg),objective=asdict(objective),objective_version=OBJECTIVE_VERSION,
            updates=64,pairs_per_update=8,view_exposures=2048,pair_draws=512,
            optimizer=dict(rate=.01,clip=1.,epsilon=epsilon,rate_scales=scales),
            sampler_seed=918425,probe_indices=probe.tolist(),probe_positions=len(original),
            probe_families=len({pairs[i]['opening_family'] for i in probe}),cpus=cpus,
            training_work='two forwards and one backward per update; no TPU',
            scope='Single-seed pipeline pilot. Teacher-selected branch probes are not general validation or playing strength.')
        contract['contract_id']=identity(contract)
        out.mkdir(parents=True,exist_ok=False);atomic_json(out/'contract.json',contract)
        report=dict(status='running',contract_id=contract['contract_id'],records=[])
        atomic_json(out/'result.json',report)
        atomic_json(out/'status.json',dict(state='training',pid=os.getpid(),step=0,cpus=cpus))
        try:
            np.savez(out/'probe-inputs.npz',features=original,visual=visual_probe,views=view_probe)
            report['before']=evaluate(model,objective,view_probe,visual_probe,original,geometry,out,'before',probe_families)
            atomic_json(out/'result.json',report)
            for step in range(64):
                x=sampler.batch(8);begin=time.time()
                metrics=model.train_embedding(x,objective,rate=.01,rate_scales=scales,epsilon=epsilon)
                metrics['seconds']=time.time()-begin;report['records'].append(metrics)
                if (step+1)%8==0:
                    report['completed_updates']=step+1;atomic_json(out/'result.json',report)
                    atomic_json(out/'status.json',dict(state='training',pid=os.getpid(),step=step+1,
                                training_exposures=(step+1)*32,cpus=cpus))
                    print(json.dumps(dict(step=step+1,loss=metrics['loss'],seconds=metrics['seconds'])),flush=True)
            report['after']=evaluate(model,objective,view_probe,visual_probe,original,geometry,out,'after',probe_families)
            np.testing.assert_array_equal(model.parameters()['readout_gain'],params['readout_gain'])
            if args.variant=='frozen':
                for k in ('edge','leak','bias','input_gain'):np.testing.assert_array_equal(model.parameters()[k],params[k])
                with np.load(out/'before-responses.npz') as a,np.load(out/'after-responses.npz') as b:
                    np.testing.assert_array_equal(a['rates'],b['rates'])
            checkpoint=out/'checkpoints/step-00000064.npz'
            report['checkpoint']=save_checkpoint(model,sampler,checkpoint,
                dict(dataset_id=bank['dataset_id'],embedding_contract=contract),root=args.root)
            # One extra update only checks exact restoration; the published checkpoint stays at 64.
            next_x=sampler.batch(8);model.train_embedding(next_x,objective,rate=.01,rate_scales=scales,epsilon=epsilon)
            expected=model.checkpoint_arrays()
            restored=RustFly(graph,cfg,ports=ports,params=params)
            other=PairSampler(arrays['features'][train],train_pairs,renderer,0)
            load_checkpoint(checkpoint,restored,other,dataset_id=bank['dataset_id'])
            np.testing.assert_array_equal(other.batch(8),next_x)
            restored.train_embedding(next_x,objective,rate=.01,rate_scales=scales,epsilon=epsilon)
            for k,v in restored.checkpoint_arrays().items():np.testing.assert_array_equal(v,expected[k])
            report.update(status='complete',exact_continuation=True,
                diagnostic_extra_updates=2,diagnostic_extra_view_exposures=64,
                artifacts={name:sha256(out/name) for name in ('before-responses.npz','after-responses.npz',
                    'before-correlations.npz','after-correlations.npz','probe-inputs.npz')})
        except BaseException as e:
            report.update(status='failed',error=repr(e));raise
        finally:
            report['seconds']=time.time()-started;atomic_json(out/'result.json',report)
            atomic_json(out/'status.json',dict(state=report['status'],pid=os.getpid(),
                        step=report.get('completed_updates',0),cpus=cpus))
        print(json.dumps(dict(status=report['status'],before=report['before'],after=report['after'])),flush=True)


if __name__=='__main__':main()
