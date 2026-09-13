#!/usr/bin/env python3
"""Fresh-process full-model continuation fingerprint for checkpoint recovery tests."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from flygo.checkpoint import load_checkpoint
from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release,Sampler
from flygo.fly import FlyConfig,RustFly,load_graph
from flygo.runtime import pin
from flygo.storage import StorageBudget,GIB


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--cpus',default='92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107')
    args=p.parse_args();cpus=list(map(int,args.cpus.split(',')));pin(cpus)
    with StorageBudget(args.root).reserve(files=16*1024**2,heap=12*GIB,purpose='fresh full-model checkpoint continuation'):
        with np.load(args.checkpoint,allow_pickle=False) as f:metadata=json.loads(f['metadata'].tobytes())
        config=FlyConfig(**{**metadata['model_config'],'threads':len(cpus)})
        graph_path=Path(json.loads((args.root/'runs/m4/graph.json').read_text())['path'])
        manifest,arrays,indexes=load_release(args.root,args.root/'releases/pilot-v1/manifest.json')
        sampler=Sampler(arrays,indexes,config.seed);model=RustFly(load_graph(graph_path),config)
        load_checkpoint(args.checkpoint,model,sampler,dataset_id=manifest['dataset_id'])
        batch=sampler.batch(32)
        batch_hash=hashlib.sha256(b''.join(a.tobytes() for a in batch)).hexdigest()
        result=model.train_step(*batch)
        states=model.checkpoint_arrays()
        hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in states.items()}
        atomic_json(args.output,dict(status='passed',batch_sha256=batch_hash,next_update=result,state_hashes=hashes,
                    checkpoint=str(args.checkpoint),cpus=cpus,graph_id=metadata['graph_id'],dataset_id=metadata['dataset_id']))
        print(json.dumps(dict(status='passed',next_update=result,output=str(args.output))),flush=True)


if __name__=='__main__':main()
