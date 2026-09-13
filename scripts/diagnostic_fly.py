#!/usr/bin/env python3
"""Input-dependent two-class learning on every neuron/edge of the fixed graph."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from flygo.data.corpus import atomic_json
from flygo.fly import FlyConfig,RustFly,load_graph
from flygo.runtime import pin
from flygo.storage import StorageBudget,GIB


def main():
    root=Path('/dev/shm/flygo');pin(list(range(100,116)))
    with StorageBudget(root).reserve(files=16*1024**2,heap=8*GIB,purpose='full-graph synthetic learning diagnostic'):
        graph=load_graph(Path(json.loads((root/'runs/m4/graph.json').read_text())['path']))
        config=FlyConfig(steps=2,features=1,groups=16,actions=2,threads=16)
        model=RustFly(graph,config)
        x=np.array([[-1.0],[1.0]]*4,np.float32)
        policy=np.tile(np.eye(2,dtype=np.float32),(4,1))
        value=np.array([-.8,.8]*4,np.float32)
        legal=np.ones((8,2),np.uint8)
        initial=model.parameters()['edge'];initial_hash=hashlib.sha256(initial.tobytes()).hexdigest()
        start=time.time();losses=[]
        for step in range(201):
            result=model.infer(x)
            logits=result['logits'];logits-=logits.max(axis=1,keepdims=True)
            ce=float(-(policy*(logits-np.log(np.exp(logits).sum(axis=1,keepdims=True)))).sum(axis=1).mean())
            mse=float(np.square(result['value']-value).mean())
            if step%10==0:
                record=dict(step=step,policy_loss=ce,value_mse=mse);losses.append(record);print(json.dumps(record),flush=True)
            if step==200:break
            model.train_step(x,legal,policy,value,rate=.003)
        changed=int(np.count_nonzero(model.parameters()['edge']!=initial))
        passed=losses[-1]['policy_loss']<losses[0]['policy_loss']*.5 and losses[-1]['value_mse']<.1 and changed>0
        atomic_json(root/'runs/m5/diagnostic.json',dict(status='passed' if passed else 'failed',config=asdict(config),
            graph_id=graph['manifest']['graph_id'],neurons=len(graph['type_id']),edges=len(graph['src']),
            initial_edge_sha256=initial_hash,changed_edges=changed,losses=losses,seconds=time.time()-start,
            scope='Synthetic input-dependent diagnostic only; not a Go-strength measurement'))
        if not passed:raise RuntimeError('Full-graph diagnostic did not learn')


if __name__=='__main__':main()
