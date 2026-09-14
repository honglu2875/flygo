#!/usr/bin/env python3
"""Compare full-circuit prediction and unchanged learning with a frozen CPU implementation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from flygo import _native
from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release,Sampler
from flygo.play import load_player
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB,StorageBudget


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--updates',type=int,default=0)
    parser.add_argument('--prune',action='store_true')
    parser.add_argument('--cpus',default='32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55')
    args=parser.parse_args();pin(list(map(int,args.cpus.split(','))))
    if not 1<=args.batch_size<=32 or not 0<=args.updates<=3:
        parser.error('Choose B1..32 and 0..3 in-memory diagnostic updates')
    if args.output.exists():raise ValueError('Numerical qualification records are immutable')
    args.output.resolve().relative_to(args.root.resolve())
    with StorageBudget(args.root).reserve(files=4<<20,heap=24*GIB,purpose='full-circuit dependency qualification'):
        root=args.root;start=time.time()
        manifest,arrays,indexes=load_release(root,root/'releases/v0-1m/manifest.json',cache=True)
        graph=Path(json.loads((root/'runs/m4/graph.json').read_text())['path'])
        model,metadata=load_player(args.checkpoint,graph,threads=len(args.cpus.split(',')))
        core=getattr(model,'core',model)
        sampler=Sampler(arrays,indexes,714091)
        result=dict(status='running',checkpoint_sha256=sha256(args.checkpoint),
            dataset_id=manifest['dataset_id'],graph_id=core.graph['manifest']['graph_id'],
            model_config=asdict(core.config),source_native_sha256=sha256(Path(_native.__file__)),
            script_sha256=sha256(Path(__file__)),batch_size=args.batch_size,updates=args.updates,
            pruned=args.prune,records=[],scope='Fixed training inputs and in-memory diagnostic updates only; no scientific checkpoint or final-test evaluation. Exact prediction, state, gradient and checkpoint-array comparison with the frozen CPU reference. FP64 diagnostic norm differences are reported separately.')
        reference=json.loads(args.reference.read_text()) if args.reference else None
        if reference:
            if reference['status']!='passed':raise ValueError('Reference qualification did not complete')
            for key in ('checkpoint_sha256','dataset_id','graph_id','model_config','batch_size','updates'):
                if result[key]!=reference[key]:raise ValueError('Reference contract differs: '+key)
        try:
            for step in range(args.updates+1):
                batch=sampler.batch(args.batch_size)
                features=model.adapter.encode(batch[0],model.mode) if hasattr(model,'adapter') else batch[0]
                traced=model.infer(batch[0],trace=True)
                record=dict(batch_sha256=[digest(x) for x in batch],encoded_sha256=digest(features),
                    inference={key:digest(traced[key]) for key in ('logits','value')},
                    states_sha256=[digest(x) for x in traced['states']])
                if args.prune:
                    pruned=model.infer(batch[0],prune=True)
                    if {key:digest(value) for key,value in pruned.items()}!=record['inference']:
                        raise AssertionError('Pruned outputs differ from the full trace')
                del traced
                if step<args.updates:
                    loss,gradients=core.loss_and_grad(features,*batch[1:])
                    record.update(loss=loss,gradients_sha256={key:digest(value) for key,value in gradients.items()})
                    del gradients
                    metrics=core.train_step(features,*batch[1:],rate=.03,rate_scales={'bias':.01},epsilon=1e-6)
                    record['gradient_norm']=metrics.pop('gradient_norm')
                    record['update']=metrics
                    record['checkpoint_arrays_sha256']={key:digest(value) for key,value in core.checkpoint_arrays().items()}
                if reference:
                    expected=reference['records'][step]
                    for key,value in record.items():
                        if key!='gradient_norm' and value!=expected[key]:
                            raise AssertionError(f'Frozen CPU reference differs at step {step}: {key}')
                    if 'gradient_norm' in record:
                        record['reference_norm_absolute_difference']=abs(record['gradient_norm']-expected['gradient_norm'])
                result['records'].append(record)
                atomic_json(args.output,result)
            if args.prune:result['dependencies']=core.prediction_dependencies()
            result.update(status='passed',seconds=time.time()-start)
            if reference:result['reference_sha256']=sha256(args.reference)
            atomic_json(args.output,result)
            print(json.dumps(dict(status='passed',checkpoint=str(args.checkpoint),batch=args.batch_size,
                                 updates=args.updates,pruned=args.prune,seconds=result['seconds'])),flush=True)
        except BaseException as error:
            result.update(status='failed',error=repr(error),seconds=time.time()-start)
            atomic_json(args.output,result)
            raise


if __name__=='__main__':main()
