#!/usr/bin/env python3
"""Compare fixed-horizon larger-retina endpoints with their adopted paired controls."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.feather as feather

from summarize_attachment_confirmation import complete, read, slices, seed_variation, concentration, METRICS
from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB, StorageBudget


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root;pin([56,57,58,59])
    args.output.resolve().relative_to(root.resolve())
    if args.output.exists():raise ValueError('Analysis reports are immutable')
    study=root/'runs/retinal-allocation-v1';plan=read(study/'plan.json')
    reference_path=root/plan['reference_analysis']
    if sha256(reference_path)!=plan['reference_analysis_sha256']:raise ValueError('Adopted analysis differs')
    adopted=complete(read(reference_path))
    if sha256(root/plan['input_map'])!=plan['input_map_sha256']:raise ValueError('Retinal map differs')
    if sorted(j['seed'] for j in plan['jobs'])!=[1,2,3]:raise ValueError('Expected three paired seeds')
    with StorageBudget(root).reserve(files=8<<20,heap=3*GIB,purpose='paired retinal allocation analysis'):
        manifest=read(root/'releases'/plan['release']/'manifest.json');dataset=manifest['dataset_id']
        if dataset!=adopted['dataset_id']:raise ValueError('Dataset differs from the reference')
        names=np.asarray([row['opening_family'] for row in manifest['records']])
        novelty_dir=root/'runs/attachment-input-novelty-v1';novelty=complete(read(novelty_dir/'result.json'))
        if novelty['dataset_id']!=dataset or sha256(novelty_dir/'masks.npz')!=novelty['masks_sha256']:
            raise ValueError('Reduced-source novelty evidence differs')
        with np.load(novelty_dir/'masks.npz',allow_pickle=False) as data:
            indices=data['indices'];common=data['current']&data['neutral']
        graph=Path(read(root/'runs/m4/graph.json')['path'])
        annotation=feather.read_table(graph/'annotations.feather',columns=['bodyId','type','superclass','subclass'])
        records=[];contrasts=[];matrices={};families=None;expected_motors=None
        for job in plan['jobs']:
            seed=job['seed'];run=job['run_id']
            control=next(r for r in adopted['records'] if r['mode']=='current' and r['seed']==seed)
            registered=next(r for r in plan['references'] if r['seed']==seed)
            if control['checkpoint_sha256']!=registered['checkpoint_sha256'] or control['run_id']!=registered['run_id']:
                raise ValueError('Adopted control checkpoint differs')
            endpoint_path=root/'runs/retinal-allocation-endpoints-v1'/(run+'.json')
            candidate=complete(read(endpoint_path))
            if (candidate['plan_sha256']!=sha256(study/'plan.json') or candidate['seed']!=seed
                    or candidate['input_contract']['mode']!='current'
                    or candidate['input_contract']['attachment_sha256']!=plan['input_map_sha256']
                    or any(r['replica_status']!='verified' for r in candidate['checkpoint_receipts'])):
                raise ValueError('Candidate endpoint or recovery contract differs')
            per_case=[]
            for kind,endpoint in [('reference',control),('large',candidate)]:
                if endpoint['labeled_training_exposures']!=plan['updates']*plan['batch_size']:
                    raise ValueError('Exposure horizons differ')
                if kind=='large':base=root/'runs/retinal-allocation-followup-v1'/run
                else:base=root/('runs/attachment-validation-v1' if seed==1 else 'runs/attachment-confirmation-validation-v1')/control['run_id']
                validation=base/'validation';report=complete(read(validation/'result.json'))
                if len(report['records'])!=1:raise ValueError('Expected one final checkpoint')
                row=report['records'][0];digest=endpoint['checkpoint_sha256']
                if (row['sha256']!=digest or row['dataset_id']!=dataset or row['optimizer_step']!=plan['updates']
                        or row['model']['seed']!=seed or row['model']['steps']!=plan['passes']
                        or row['model']['groups']!=plan['groups']):raise ValueError('Validation endpoint differs')
                path=validation/f'step-{plan["updates"]:08d}-{digest[:12]}-metrics.npz'
                with np.load(path,allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['indices'],indices)
                    metrics=data['metrics'];group=names[data['game_index']]
                if metrics.shape!=(len(indices),3) or not np.isfinite(metrics).all():raise ValueError('Invalid metrics')
                if families is None:families=group
                else:np.testing.assert_array_equal(families,group)
                np.testing.assert_allclose(metrics.mean(axis=0),[row['slices']['natural'][k]['mean'] for k in METRICS],rtol=0,atol=1e-12)
                summary=slices(metrics,families,common)
                if kind=='reference' and (sha256(path)!=control['metrics_sha256'] or any(summary[k]!=control[k] for k in summary)):
                    raise ValueError('Original paired control analysis no longer reproduces')
                matrices[seed,kind]=metrics
                per_case.append(dict(kind=kind,checkpoint_sha256=digest,metrics_sha256=sha256(path),
                    validation_record_sha256=sha256(validation/'result.json'),**summary))
            diagnostics=root/'runs/retinal-allocation-followup-v1'/run
            follow=read(diagnostics/'status.json')
            if follow['state']!='complete' or follow['checkpoint_sha256']!=candidate['checkpoint_sha256']:
                raise ValueError('Endpoint followups are incomplete')
            signal=complete(read(diagnostics/'signal/result.json'))
            selection=read(diagnostics/'signal/selection.json')
            if selection!=adopted['probe_selection']:raise ValueError('Motor probe positions differ')
            signal_row=next(r for r in signal['records'] if r['case']=='current')
            response=diagnostics/'signal/current-responses.npz'
            if (signal_row['checkpoint_sha256']!=candidate['checkpoint_sha256']
                    or sha256(response)!=signal_row['response_file_sha256']):raise ValueError('Motor response identity differs')
            concentrated,expected_motors=concentration(response,'current',annotation,expected_motors)
            count=complete(read(diagnostics/'count/result.json'))
            reference_count=next(r['counted_inference'] for r in adopted['diagnostics'] if r['seed']==seed and r['mode']=='current')
            # The source selection digest is common across every prior count arm.
            original_count=root/('runs/attachment-count-v1/result.json' if seed==1 else
                f'runs/attachment-confirmation-diagnostics-v1/{control["run_id"]}/count/result.json')
            if count['dataset_id']!=dataset or count['indices_sha256']!=read(original_count)['indices_sha256']:
                raise ValueError('Arithmetic probes differ')
            measured=next(r for r in count['results'] if r['checkpoint_sha256']==candidate['checkpoint_sha256'])
            records.append(dict(seed=seed,run_id=run,endpoint=candidate,endpoint_sha256=sha256(endpoint_path),
                comparisons=per_case,visual_concentration=concentrated,signal_record_sha256=sha256(diagnostics/'signal/result.json'),
                counted_inference=dict(reference=reference_count,large=measured),count_record_sha256=sha256(diagnostics/'count/result.json')))
            contrasts.append(dict(seed=seed,**slices(matrices[seed,'large']-matrices[seed,'reference'],families,common)))
        differences=[matrices[s,'large']-matrices[s,'reference'] for s in (1,2,3)]
        result=dict(status='complete',created=time.time(),plan=plan,plan_sha256=sha256(study/'plan.json'),
            reference_analysis_sha256=sha256(reference_path),dataset_id=dataset,records=records,contrasts=contrasts,
            pooled=slices(np.stack(differences).mean(axis=0),families,common),seed_variation=seed_variation(differences,common),
            contrast_direction='larger retina minus paired current-board montage',novelty=novelty,
            source_sha256=sha256(Path(__file__)),helper_sha256={name:sha256(Path(__file__).with_name(name)) for name in
                ('summarize_attachment_confirmation.py','summarize_attachments.py','compare_checkpoints.py')},
            limitations=['Three paired head/sampler seeds; initial core strengths are not independently randomized.',
                'Conditional opening-family intervals hold the fitted weights fixed; they are not training-seed confidence intervals.',
                'Same 9x9 board and information, different angular allocation/aspect ratio/drive; no gain correction.',
                'Reduced-source novelty precedes rendering and does not certify absence of FP32 image collisions.',
                'Raw response variance and decoder influence alone do not prove predictive usefulness.',
                'No final-test evaluation, new matched CNN result, playing-strength claim, head constraint, memory, optimizer change or TPU use.'])
        atomic_json(args.output,result)
        print(json.dumps(dict(status='complete',output=str(args.output),pooled=result['pooled'],seed_variation=result['seed_variation']),indent=2),flush=True)


if __name__=='__main__':main()
