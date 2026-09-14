#!/usr/bin/env python3
"""Measure warm Rust kernels and full inference/gradients on frozen real Go batches."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from flygo import _native
from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release,Sampler
from flygo.fly import FlyConfig,RustFly,load_graph
from flygo.play import load_player
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import StorageBudget,GIB


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--cpus',default='117,118,119')
    p.add_argument('--batches',type=int,nargs='+',default=[1,32])
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--reference',type=Path,help='Require identical full outputs/gradients to this earlier profile')
    args=p.parse_args();cpus=list(map(int,args.cpus.split(',')));pin(cpus)
    if (not args.batches or min(args.batches)<1 or max(args.batches)>128
            or len(set(args.batches))!=len(args.batches) or not 1<=args.repetitions<=100):
        p.error('Choose unique batches in 1..128 and 1..100 repetitions')
    root=args.root;args.output.mkdir(parents=True,exist_ok=False)
    with StorageBudget(root).reserve(files=GIB,heap=12*GIB,purpose='read-only Rust kernel profile'):
        graph=Path(json.loads((root/'runs/m4/graph.json').read_text())['path'])
        manifest,arrays,indexes=load_release(root,root/'releases/v0-1m/manifest.json',cache=True)
        model=(load_player(args.checkpoint,graph,threads=len(cpus))[0] if args.checkpoint else
               RustFly(load_graph(graph),FlyConfig(steps=4,threads=len(cpus))))
        sampler=Sampler(arrays,indexes,714091);records=[]
        before=model.checkpoint_arrays()
        for batch_size in args.batches:
            batch=sampler.batch(batch_size)
            kernels=model.profile_sparse(batch[0],repetitions=args.repetitions)
            model.infer(batch[0])  # Prime the persistent weight-transform cache.
            samples=[]
            for _ in range(args.repetitions):
                begin=time.perf_counter();result=model.infer(batch[0],trace=True)
                inference=time.perf_counter()-begin
                begin=time.perf_counter();prediction=model.infer(batch[0])
                prediction_seconds=time.perf_counter()-begin
                for key in prediction:
                    if prediction[key].tobytes()!=result[key].tobytes():
                        raise AssertionError('Prediction differs from trace: '+key)
                begin=time.perf_counter();loss,gradient=model.loss_and_grad(*batch)
                samples.append(dict(inference=inference,prediction=prediction_seconds,
                                    loss_and_grad=time.perf_counter()-begin))
            actual={**result,**{'gradient/'+k:v for k,v in gradient.items()},
                    'loss':np.asarray([loss['policy_loss'],loss['value_loss']])}
            name=f'batch-{batch_size}.npz'
            np.savez(args.output/name,**actual)
            if args.reference:
                with np.load(args.reference/name) as reference:
                    for key,value in actual.items():
                        a,b=np.asarray(value),reference[key]
                        if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():
                            raise AssertionError('CPU reference differs: '+name+'/'+key)
            records.append(dict(batch_size=batch_size,kernels=kernels,full_model=samples,
                activity=[float(np.mean(state>0)) for state in result['states']],reference_sha256=sha256(args.output/name)))
            for key,array in model.checkpoint_arrays().items():
                if array.tobytes()!=before[key].tobytes():raise AssertionError('Profiler mutated '+key)
            atomic_json(args.output/'result.json',dict(status='complete' if batch_size==args.batches[-1] else 'running',
                source_native_sha256=sha256(Path(_native.__file__)),checkpoint=str(args.checkpoint) if args.checkpoint else None,
                dataset_id=manifest['dataset_id'],cpus=cpus,records=records,
                reference=str(args.reference) if args.reference else None,
                scope='Warm sparse kernels use a synthetic dense cotangent; full-model gradients use real targets. No optimizer updates.'))
            print(json.dumps(records[-1]),flush=True)


if __name__=='__main__':main()
