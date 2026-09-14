#!/usr/bin/env python3
"""Close the registered spherical input screen from retained, paired evidence."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.feather as feather

from compare_checkpoints import summarize
from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB,StorageBudget


def family_summary(metrics,families):
    result=summarize(metrics,families)
    result['families']=result.pop('games')
    for value in result.values():
        if isinstance(value,dict):value['family_bootstrap_95']=value.pop('game_bootstrap_95')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();pin([56,57,58,59])
    args.output.resolve().relative_to(args.root.resolve())
    if args.output.exists():raise ValueError('Analysis reports are immutable')
    root=args.root;study=root/'runs/attachment-study-v1';validation=root/'runs/attachment-validation-v1'
    plan=json.loads((study/'plan.json').read_text())
    collected=json.loads((validation/'summary.json').read_text())
    if not all(row['status']['state']=='complete' for row in collected['records']):
        raise ValueError('Full validation has not completed')
    with StorageBudget(root).reserve(files=4<<20,heap=3*GIB,purpose='paired attachment endpoint analysis'):
        release=root/'releases/v0-1m/manifest.json';manifest=json.loads(release.read_text())
        if manifest['dataset_id']!=plan['dataset_id']:raise ValueError('Release identity differs')
        families=np.asarray([record['opening_family'] for record in manifest['records']])
        novelty=root/'runs/attachment-input-novelty-v1'
        novelty_report=json.loads((novelty/'result.json').read_text())
        if (novelty_report['status']!='complete' or novelty_report['dataset_id']!=plan['dataset_id']
                or sha256(novelty/'masks.npz')!=novelty_report['masks_sha256']):
            raise ValueError('Novelty evidence differs')
        with np.load(novelty/'masks.npz',allow_pickle=False) as data:
            indices=data['indices'];common=data['current'] & data['neutral']
        evidence={};records=[];common_groups=None
        for job in plan['jobs']:
            run=root/'runs'/job['run_id'];mode=job['mode']
            checkpoint=run/'checkpoints'/f"step-{plan['updates']:08d}.npz"
            record=json.loads((validation/job['run_id']/'validation/result.json').read_text())['records'][0]
            if (record['sha256']!=sha256(checkpoint) or record['optimizer_step']!=plan['updates']
                    or record['dataset_id']!=plan['dataset_id']):raise ValueError('Wrong endpoint')
            metric_path=validation/job['run_id']/'validation'/(checkpoint.stem+'-'+record['sha256'][:12]+'-metrics.npz')
            with np.load(metric_path,allow_pickle=False) as data:
                np.testing.assert_array_equal(data['indices'],indices)
                metrics=data['metrics'];group=families[data['game_index']]
            if common_groups is None:common_groups=group
            else:np.testing.assert_array_equal(group,common_groups)
            if not np.isfinite(metrics).all():raise ValueError('Nonfinite validation metric')
            np.testing.assert_allclose(metrics.mean(axis=0),[record['slices']['natural'][key]['mean']
                for key in ('policy_kl','value_mse','teacher_top1_agreement')],rtol=0,atol=1e-12)
            evidence[mode]=metrics
            log=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            endpoints=[row for row in log if row.get('kind')=='validation']
            if endpoints[-1]['step']!=plan['updates']:raise ValueError('Missing final learner evaluation')
            changes={}
            with np.load(checkpoint,allow_pickle=False) as final,np.load(run/'checkpoints/step-00000000.npz',allow_pickle=False) as initial:
                metadata=json.loads(final['metadata'].tobytes())
                if metadata['input_contract']['mode']!=mode:raise ValueError('Wrong visual encoding')
                for key in final.files:
                    if key.startswith('param/'):
                        before,after=initial[key],final[key];delta=after.astype(np.float64)-before
                        changes[key[6:]]=dict(scalars=after.size,changed=int(np.count_nonzero(after!=before)),
                            delta_l2=float(np.linalg.norm(delta)),delta_max=float(np.abs(delta).max()))
                gain=final['param/input_gain'];original=initial['param/input_gain']
                for name,selection in [('visual',slice(None,3490)),('context',slice(3490,None))]:
                    changes[name+'_input_gain']=dict(changed=int(np.count_nonzero(gain[selection]!=original[selection])),
                        quantiles=np.quantile(gain[selection],[0,.1,.5,.9,1]).tolist())
            diagnostics=[row for row in log if row.get('kind')=='diagnostics']
            records.append(dict(mode=mode,checkpoint_sha256=record['sha256'],input_contract=metadata['input_contract'],
                natural=family_summary(metrics,group),common_reduced_source_novel=family_summary(metrics[common],group[common]),
                learning_curve=endpoints,parameter_changes=changes,
                diagnostic_forward_positions=sum(row['diagnostic_forward_positions'] for row in diagnostics),
                diagnostic_backward_positions=sum(row['diagnostic_backward_positions'] for row in diagnostics),
                labeled_training_exposures=plan['updates']*plan['batch_size'],
                validation_forward_positions=sum(
                    sum(row[key]['positions'] for key in ('natural','novel','train'))+
                    sum(value['positions'] for value in row.get('input_perturbations',{}).values()) for row in endpoints),
                metrics_sha256=sha256(metric_path),final_activity=diagnostics[-1] if diagnostics else None))
        contrasts=[dict(reference=left,candidate=right,direction='candidate minus reference',
            natural=family_summary(evidence[right]-evidence[left],group),
            common_reduced_source_novel=family_summary((evidence[right]-evidence[left])[common],group[common]))
            for left,right in [('neutral','history'),('neutral','current'),('history','current')]]
        signal=root/'runs/attachment-signal-v1';signal_report=json.loads((signal/'result.json').read_text())
        graph=Path(json.loads((root/'runs/m4/graph.json').read_text())['path'])
        table=feather.read_table(graph/'annotations.feather',columns=['bodyId','type','superclass','subclass'])
        concentration=[]
        for mode in ('history','current'):
            with np.load(signal/(mode+'-responses.npz'),allow_pickle=False) as data:
                y=(data[mode]-data['neutral']).astype(np.float64);motors=data['motors']
            y-=y.mean(axis=0);variance=np.square(y).sum(axis=0);order=np.argsort(variance)[::-1]
            use=np.sqrt(variance/(len(y)-1))>1e-6;unit=y[:,use]/np.sqrt(variance[use]);gram=unit@unit.T
            concentration.append(dict(mode=mode,standardized_participation_rank=float(np.trace(gram)**2/np.square(gram).sum()),
                top_cells=[dict(node=int(motors[i]),visual_variance_fraction=float(variance[i]/variance.sum()),
                    **{key:table[key][int(motors[i])].as_py() for key in table.column_names}) for i in order[:5]]))
        count=json.loads((root/'runs/attachment-count-v1/result.json').read_text())
        if signal_report['status']!='complete' or count['status']!='complete':raise ValueError('Diagnostics incomplete')
        atomic_json(args.output,dict(status='complete',created=time.time(),plan=plan,records=records,contrasts=contrasts,
            uncertainty='1000 paired opening-family bootstrap resamples, conditional on these seed-1 endpoints; not training-seed uncertainty.',
            visual_signal=signal_report,visual_concentration=concentration,
            source_novelty=novelty_report,
            counted_inference=count,checkpoint_replication=json.loads((study/'checkpoint-00001000-replication.json').read_text()),
            source_sha256=sha256(Path(__file__)),bootstrap_helper_sha256=sha256(Path(__file__).with_name('compare_checkpoints.py')),
            annotation_sha256=sha256(graph/'annotations.feather'),plan_sha256=sha256(study/'plan.json'),
            limitations=['One short training seed; no advantage over CNN or matched-FLOP control established.',
                'Perturbations are out-of-distribution diagnostics; neutral is also trained independently as a control.',
                'Reduced source novelty is before retinal rendering; current-image FP32 collisions are not certified absent.',
                'Signal concentration and correlations describe this model and probe; they are not physiological recordings.',
                'Arithmetic counts exclude backward, optimizer, mask/integer/memory work, teacher generation and search; trace latency is not production inference latency.',
                'Final test examples and labels remain closed. No TPU use.']))
        print(json.dumps(dict(status='complete',output=str(args.output),records=len(records))),flush=True)


if __name__=='__main__':main()
