#!/usr/bin/env python3
"""Fresh-process full-model continuation fingerprint for checkpoint recovery tests."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from flygo.checkpoint import load_checkpoint,save_checkpoint
from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release,Sampler
from flygo.fly import FlyConfig,RustFly,load_graph
from flygo.runtime import pin
from flygo.storage import StorageBudget,GIB


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--release',default='pilot-v1')
    p.add_argument('--prepare',action='store_true',help='Make a two-update checkpoint and its uninterrupted next-update fingerprint')
    p.add_argument('--passes',type=int,default=4)
    p.add_argument('--peer',default='cubic27@t1v-n-a09f5679-w-1')
    p.add_argument('--cpus',default='92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107')
    args=p.parse_args();cpus=list(map(int,args.cpus.split(',')));pin(cpus)
    with StorageBudget(args.root).reserve(files=16*1024**2,heap=12*GIB,purpose='fresh full-model checkpoint continuation'):
        metadata={};ports=None
        if args.prepare:config=FlyConfig(steps=args.passes,threads=len(cpus))
        else:
            with np.load(args.checkpoint,allow_pickle=False) as f:
                metadata=json.loads(f['metadata'].tobytes())
                ports={k:f['port/'+k].copy() for k in ('input_index','output_group','output_scale')}
            config=FlyConfig(**{**metadata['model_config'],'threads':len(cpus)})
        graph_path=Path(json.loads((args.root/'runs/m4/graph.json').read_text())['path'])
        manifest,arrays,indexes=load_release(args.root,args.root/'releases'/args.release/'manifest.json',cache=True)
        sampler=Sampler(arrays,indexes,config.seed);model=RustFly(load_graph(graph_path),config,ports=ports)
        contract=metadata.get('training_contract',dict(batch_size=32,rate=.003,clip=1.0))
        settings=dict(rate=contract['rate'],clip=contract['clip'])
        if contract.get('rate_scales'):settings['rate_scales']=contract['rate_scales']
        if args.prepare:
            for _ in range(2):model.train_step(*sampler.batch(contract['batch_size']),**settings)
            save_checkpoint(model,sampler,args.checkpoint,dict(dataset_id=manifest['dataset_id'],training_contract=contract),
                            root=args.root,peer=args.peer or None)
        else:load_checkpoint(args.checkpoint,model,sampler,dataset_id=manifest['dataset_id'])
        batch=sampler.batch(contract['batch_size'])
        batch_hash=hashlib.sha256(b''.join(a.tobytes() for a in batch)).hexdigest()
        result=model.train_step(*batch,**settings)
        states=model.checkpoint_arrays()
        hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in states.items()}
        atomic_json(args.output,dict(status='passed',batch_sha256=batch_hash,next_update=result,state_hashes=hashes,
                    checkpoint=str(args.checkpoint),cpus=cpus,graph_id=model.graph['manifest']['graph_id'],dataset_id=manifest['dataset_id']))
        print(json.dumps(dict(status='passed',next_update=result,output=str(args.output))),flush=True)


if __name__=='__main__':main()
