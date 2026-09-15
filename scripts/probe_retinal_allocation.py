#!/usr/bin/env python3
"""Compare retinal maps at fixed weights; retain raw motor responses and heads."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.feather as feather

from flygo.attachments import load_attachment
from flygo.data.corpus import atomic_json
from flygo.play import load_player
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB,StorageBudget
from probe_biological_ports import correlations,parameter_hash,probes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--checkpoint',type=Path,nargs='+',required=True)
    p.add_argument('--input-map',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpus',default=','.join(map(str,range(32,56))))
    args=p.parse_args();cpus=list(map(int,args.cpus.split(',')));pin(cpus);root=args.root
    args.output.resolve().relative_to(root.resolve())
    if args.output.exists():raise ValueError('Response probes are immutable')
    with StorageBudget(root).reserve(files=64<<20,heap=6*GIB,purpose='fixed-weight retinal allocation motor probe'):
        args.output.mkdir(parents=True);started=time.time()
        graph=Path(json.loads((root/'runs/m4/graph.json').read_text())['path'])
        graph_id=json.loads((graph/'manifest.json').read_text())['graph_id']
        candidate,ports,receipt=load_attachment(args.input_map,graph_id=graph_id)
        x,legal,_,selection=probes(root,256,972091)
        if selection['dataset_id']!=receipt['dataset_id']:raise ValueError('Probe data differ from geometry')
        atomic_json(args.output/'selection.json',selection)
        annotation=feather.read_table(graph/'annotations.feather',columns=['bodyId','type'])
        records=[]
        for number,path in enumerate(args.checkpoint):
            recovery=json.loads(path.with_suffix('.json').read_text())
            if recovery['replica_status']!='verified' or sha256(path)!=recovery['sha256']:
                raise ValueError('Checkpoint has no verified recovery copy')
            player,metadata=load_player(path,graph,threads=len(cpus))
            if (metadata['dataset_id']!=receipt['dataset_id'] or player.config.steps!=8
                    or player.config.rate_softness or player.config.readout_mean_scale!=1
                    or metadata['input_contract']['mode']!='current'):
                raise ValueError('Expected the declared current-input K8 reference')
            for key in ports:np.testing.assert_array_equal(ports[key],player.ports[key])
            if metadata['input_contract']['attachment_sha256']!=receipt['source_attachment_sha256']:
                raise ValueError('Checkpoint was trained with a different reference attachment')
            before=parameter_hash(player)
            selected=np.flatnonzero(ports['output_group']>=0)
            motors=selected[np.argsort(ports['output_group'][selected])]
            np.testing.assert_array_equal(ports['output_group'][motors],np.arange(player.config.groups))
            responses={};predictions={}
            for name,adapter,mode in [('current',player.adapter,'current'),('large',candidate,'current'),
                                       ('neutral',player.adapter,'neutral')]:
                raw=[];logits=[];values=[]
                for start in range(0,len(x),32):
                    output=player.core.infer(adapter.encode(x[start:start+32],mode),trace=True)
                    raw.append(np.maximum(output['states'][-1][motors].T,0).copy())
                    logits.append(output['logits']);values.append(output['value'])
                responses[name]=np.concatenate(raw)
                predictions[name+'_logits']=np.concatenate(logits)
                predictions[name+'_value']=np.concatenate(values)
            summaries={}
            for name in ('current','large'):
                evoked=responses[name]-responses['neutral']
                _,_,summary=correlations(evoked)
                centered=evoked.astype(np.float64)-evoked.mean(axis=0,dtype=np.float64)
                variance=np.square(centered).sum(axis=0);total=variance.sum()
                use=np.sqrt(variance/(len(x)-1))>1e-6
                unit=centered[:,use]/np.sqrt(variance[use]);gram=unit@unit.T
                denominator=np.square(gram).sum()
                summary['standardized_participation_rank']=float(np.trace(gram)**2/denominator) if denominator else None
                summary['top_cells']=[dict(node=int(motors[i]),bodyId=annotation['bodyId'][int(motors[i])].as_py(),
                    type=annotation['type'][int(motors[i])].as_py(),visual_variance_fraction=float(variance[i]/total) if total else None)
                    for i in np.argsort(variance)[::-1][:5]]
                summaries[name]=summary
            params=player.parameters()
            artifact=args.output/f'case-{number}-responses.npz'
            np.savez(artifact,motors=motors,legal=legal,**responses,**predictions,
                readout_scale=ports['output_scale'][motors]*params['readout_gain'][motors],
                policy_weight=params['policy_weight'].reshape(82,2129),policy_bias=params['policy_bias'],
                value_weight=params['value_weight'],value_bias=params['value_bias'])
            if parameter_hash(player)!=before:raise AssertionError('Response probing changed model parameters')
            records.append(dict(checkpoint=str(path),checkpoint_sha256=recovery['sha256'],
                optimizer_step=recovery['step'],seed=player.config.seed,responses=summaries,
                response_file=artifact.name,response_sha256=sha256(artifact)))
            atomic_json(args.output/'result.json',dict(status='complete' if len(records)==len(args.checkpoint) else 'running',
                records=records,created=time.time(),seconds=time.time()-started,cpus=cpus,
                graph_id=graph_id,dataset_id=receipt['dataset_id'],attachment_sha256=receipt['sha256'],
                source_sha256=sha256(Path(__file__)),helpers_sha256=sha256(Path(__file__).with_name('probe_biological_ports.py')),
                scope='Same 256 training-family positions and context, with weights fixed within each map comparison. Learned checkpoints were trained with the reference map; larger-map responses are a transfer diagnostic, not learned-C capacity or validation. No targets, new updates, action partition, test inputs or TPU.'))
            print(json.dumps(records[-1]),flush=True)
            del player,params,output


if __name__=='__main__':main()
