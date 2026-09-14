#!/usr/bin/env python3
"""Measure trained motor responses and isolate visual drive with context held fixed."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from flygo.attachments import MODES
from flygo.data.corpus import atomic_json
from flygo.play import load_player
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB,StorageBudget
from probe_biological_ports import correlations,parameter_hash,policy_summary,probes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpus',default=','.join(map(str,range(32,56))))
    args=p.parse_args();cpus=list(map(int,args.cpus.split(',')));pin(cpus)
    args.output.resolve().relative_to(args.root.resolve())
    with StorageBudget(args.root).reserve(files=48<<20,heap=4*GIB,purpose='trained attachment motor signal probe'):
        args.output.mkdir(parents=True,exist_ok=False);started=time.time()
        plan=json.loads((args.study/'plan.json').read_text())
        x,legal,nuisance,selection=probes(args.root,256,972091)
        if selection['dataset_id']!=plan['dataset_id']:raise ValueError('Probe release differs')
        atomic_json(args.output/'selection.json',selection)
        graph=Path(json.loads((args.root/'runs/m4/graph.json').read_text())['path'])
        cases=[('initial',plan['jobs'][0]['run_id'],0)]+[(job['mode'],job['run_id'],plan['updates']) for job in plan['jobs']]
        results=[]
        for name,run_id,step in cases:
            path=args.root/'runs'/run_id/'checkpoints'/f'step-{step:08d}.npz'
            receipt=json.loads(path.with_suffix('.json').read_text())
            if receipt['replica_status']!='verified' or sha256(path)!=receipt['sha256']:
                raise ValueError('Checkpoint lacks a verified recovery copy')
            player,metadata=load_player(path,graph,threads=len(cpus))
            if (metadata['dataset_id']!=plan['dataset_id'] or player.config.steps!=plan['passes']
                    or int(player.checkpoint_arrays()['optimizer_step'])!=step):
                raise ValueError('Unexpected checkpoint contract')
            before=parameter_hash(player)
            selected=np.flatnonzero(player.ports['output_group']>=0)
            motors=selected[np.argsort(player.ports['output_group'][selected])]
            np.testing.assert_array_equal(player.ports['output_group'][motors],np.arange(player.config.groups))
            responses={};predictions={};summaries={}
            for mode in MODES:
                response=[];logits=[];values=[]
                for start in range(0,len(x),32):
                    encoded=player.adapter.encode(x[start:start+32],mode)
                    output=player.core.infer(encoded,trace=True)
                    response.append(np.maximum(output['states'][-1][motors].T,0).copy())
                    logits.append(output['logits']);values.append(output['value'])
                responses[mode]=np.concatenate(response)
                predictions[mode]=(np.concatenate(logits),np.concatenate(values))
                _,_,summary=correlations(responses[mode])
                summaries[mode]=dict(raw_motor=summary,policy=policy_summary(predictions[mode][0],legal))
            for mode in ('history','current'):
                evoked=responses[mode]-responses['neutral']
                _,_,summary=correlations(evoked)
                summaries[mode]['visual_minus_neutral']=dict(summary,
                    response_rms=float(np.sqrt(np.square(evoked.astype(np.float64)).mean())),
                    logit_rms=float(np.sqrt(np.square((predictions[mode][0]-predictions['neutral'][0]).astype(np.float64)).mean())),
                    value_rms=float(np.sqrt(np.square((predictions[mode][1]-predictions['neutral'][1]).astype(np.float64)).mean())))
            if parameter_hash(player)!=before:raise AssertionError('Probe changed parameters')
            np.savez(args.output/(name+'-responses.npz'),motors=motors,**responses,
                **{mode+'_logits':value[0] for mode,value in predictions.items()},
                **{mode+'_value':value[1] for mode,value in predictions.items()})
            results.append(dict(case=name,checkpoint=str(path),checkpoint_sha256=receipt['sha256'],
                optimizer_step=step,trained_input=metadata['input_contract']['mode'],responses=summaries,
                response_file_sha256=sha256(args.output/(name+'-responses.npz'))))
            atomic_json(args.output/'result.json',dict(status='complete' if len(results)==len(cases) else 'running',
                created=time.time(),seconds=time.time()-started,records=results,
                dataset_id=plan['dataset_id'],plan_sha256=sha256(args.study/'plan.json'),
                source_sha256=sha256(Path(__file__)),probe_helpers_sha256=sha256(Path(__file__).with_name('probe_biological_ports.py')),
                scope='256 positions, one per training family; same nonvisual context across visual perturbations. Raw motor rates precede learned readout gains. Correlation variance floor 1e-6; absolute amplitudes and response arrays retained. No labels, optimizer updates, grouping fit, topology changes, final-test examples or TPU use. Initial parameters are shared across the three qualified arms.'))
            print(json.dumps(dict(case=name,responses=summaries)),flush=True)
            del player


if __name__=='__main__':main()
