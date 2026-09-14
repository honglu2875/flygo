#!/usr/bin/env python3
"""Summarize frozen validation, paired seeds and counted search work."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_checkpoints import summarize
from flygo.cost import cnn_cost,fly_cost
from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.storage import GIB,StorageBudget


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--run-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pair',nargs=2,action='append',default=[],metavar=('REFERENCE','CANDIDATE'),
                   help='Paired runs; repeated pairs summarize a common contrast across seeds')
    p.add_argument('--reference',nargs=2,action='append',default=[],metavar=('STUDY','RUN'),
                   help='Adopt one completed run from another collected study; avoid repeating validation')
    p.add_argument('--partial',action='store_true',help='Clearly label a report before every registered stage finishes')
    args=p.parse_args();directory=args.root/'runs'/args.run_id
    args.output.resolve().relative_to(args.root.resolve())
    if args.output.exists():raise ValueError('Analysis reports are immutable; choose a new output name')
    collected=json.loads((directory/'summary.json').read_text())
    plan=json.loads((directory/'plan.json').read_text())
    complete=all(row['status'].get('state')=='complete' for row in collected['records'])
    if not complete and not args.partial:raise ValueError('Registered study is incomplete; collect again after all stages finish')
    entries=[(directory,plan,row,False) for row in collected['records']]
    references=[]
    for study,name in args.reference:
        if Path(study).name!=study or study in ('.','..'):raise ValueError('Reference study must be a plain run ID')
        other=args.root/'runs'/study
        rows=json.loads((other/'summary.json').read_text())['records']
        selected=[row for row in rows if row['run']==name]
        if len(selected)!=1 or selected[0]['status'].get('state')!='complete':
            raise ValueError('Adopt only an unambiguous completed reference run')
        entries.append((other,json.loads((other/'plan.json').read_text()),selected[0],True))
        references.append(dict(study=study,run=name,plan_sha256=sha256(other/'plan.json'),
                               validation_sha256=sha256(other/name/'validation/result.json')))
    graph=json.loads((args.root/'runs/m4/graph.json').read_text())
    with StorageBudget(args.root).reserve(files=16*(1<<20),heap=2*GIB,purpose='paired study analysis'):
        records=[];evidence={}
        for current,settings,row,adopted in entries:
            validation=row['results'].get('validation',{})
            if validation.get('status')!='complete':continue
            if len(validation['records'])!=1:raise ValueError('Each study job must identify one checkpoint')
            record=validation['records'][0];name=row['run'];config=record['model']
            if name in evidence:raise ValueError('A run cannot be included twice: '+name)
            path=current/name/'validation'/(Path(record['checkpoint']).stem+'-'+record['sha256'][:12]+'-metrics.npz')
            with np.load(path,allow_pickle=False) as data:arrays={key:data[key].copy() for key in data}
            natural=record['slices']['natural'];metrics=arrays['metrics']
            np.testing.assert_allclose(metrics.mean(axis=0),[natural[k]['mean'] for k in
                ('policy_kl','value_mse','teacher_top1_agreement')],atol=1e-12,rtol=0)
            if metrics.shape!=(natural['positions'],3):raise ValueError('Misaligned validation evidence')
            evidence[name]=dict(**arrays,dataset_id=record['dataset_id'],seed=config['seed'],
                                exposures=record['optimizer_step']*settings['batch_size'])
            family='cnn' if 'blocks' in config else 'fly'
            readout=None
            with np.load(record['checkpoint'],allow_pickle=False) as saved:
                training_contract=json.loads(saved['metadata'].tobytes()).get('training_contract',{})
                parameter_count=sum(saved[key].size for key in saved.files if key.startswith('param/'))
                if family=='fly':readout=int(np.count_nonzero(saved['port/output_group']>=0))
            cost=(cnn_cost(channels=config['channels'],blocks=config['blocks']) if family=='cnn' else
                  fly_cost(graph['neurons'],graph['edges'],graph['sensory_neurons'],passes=config['steps'],
                           groups=config['groups'],readout_neurons=readout,rate_softness=config.get('rate_softness',0.0),
                           readout_mean_scale=config.get('readout_mean_scale',1.0)))
            panels={}
            for mode in ('prior','puct','gumbel'):
                match=row['results'].get(mode)
                if not match or match.get('status')!='passed':continue
                evaluations=match.get('student_neural_evaluations',match['fly_neural_evaluations'])
                moves=sum(sum(1+ply%2==game.get('student_color',game['fly_color'])
                              for ply in range(4,len(game['actions']))) for game in match['games'])
                panels[mode]=dict(wins=match.get('student_wins',match['fly_wins']),completed=match['completed'],
                    capped=match['truncated'],opening_pairs=len(match['games'])//2,opening_seed=match['seed'],
                    neural_evaluations=evaluations,student_moves=moves,
                    evaluations_per_student_move=evaluations/moves if moves else None,
                    nominal_neural_flops=evaluations*cost['arithmetic_flops'],seconds=match['seconds'])
            records.append(dict(run=name,study=current.name,adopted_reference=adopted,family=family,seed=config['seed'],model=config,
                checkpoint_sha256=record['sha256'],metrics_sha256=sha256(path),dataset_id=record['dataset_id'],
                optimizer_updates=record['optimizer_step'],batch_size=settings['batch_size'],training_contract=training_contract,
                stored_parameter_scalars=parameter_count,
                training_position_exposures=record['optimizer_step']*settings['batch_size'],
                validation=record['slices'],inference_cost=cost,matches=panels))
        contrast=None
        if args.pair:
            deltas=[];pairs=[];reference=None;seeds=set()
            for left,right in args.pair:
                a,b=evidence[left],evidence[right]
                if a['dataset_id']!=b['dataset_id']:raise ValueError('Paired datasets differ')
                if a['seed']!=b['seed'] or a['exposures']!=b['exposures']:
                    raise ValueError('Paired training seeds or exposure horizons differ')
                if a['seed'] in seeds:raise ValueError('Use separate reports for different contrasts at the same seed')
                seeds.add(a['seed'])
                for key in ('indices','game_index'):
                    np.testing.assert_array_equal(a[key],b[key])
                    if reference is not None:np.testing.assert_array_equal(a[key],reference[key])
                reference=a;delta=b['metrics']-a['metrics'];deltas.append(delta)
                pairs.append(dict(reference=left,candidate=right,natural=summarize(delta,a['game_index'])))
            means=np.stack([delta.mean(axis=0) for delta in deltas])
            contrast=dict(direction='candidate minus reference',pairs=pairs,
                mean_over_registered_pairs=summarize(np.mean(deltas,axis=0),reference['game_index']),
                between_pair_sample_standard_deviation=means.std(axis=0,ddof=1).tolist() if len(means)>1 else None,
                metric_order=['policy_kl','value_mse','teacher_top1_agreement'],
                uncertainty='Game bootstrap conditions on these trained checkpoints. Sample SD describes variation across paired training seeds; a small seed set does not establish a precise seed-population confidence interval.')
        result=dict(status='complete' if complete else 'partial',study=args.run_id,records=records,contrast=contrast,
            collected_sha256=sha256(directory/'summary.json'),plan_sha256=sha256(directory/'plan.json'),adopted_references=references,
            compute_scope='Nominal neural arithmetic, including every counted search evaluation. Report CPU zero skipping, dense boundary/channel padding and latency separately. Excludes teacher generation, diagnostics, validation, gradient/backward work and earlier tuning trials; this is not total project compute.',
            match_scope='Raw completed-game counts with common paired openings; capped games reported separately. Small panels are not an Elo estimate.',
            final_test='not evaluated')
        atomic_json(args.output,result)
        print(json.dumps(dict(output=str(args.output),status=result['status'],runs=len(records),pairs=len(args.pair))))


if __name__=='__main__':main()
