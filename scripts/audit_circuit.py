#!/usr/bin/env python3
"""Measure sensory reach and visual-column coverage in the unchanged circuit."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from flygo.data.corpus import atomic_json
from flygo.fly import FlyConfig,initialize,load_graph
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB,StorageBudget


def distances(source,destination,neurons,inputs,max_hops):
    distance=np.full(neurons,-1,np.int16);distance[inputs]=0
    frontier=distance==0
    for hop in range(1,max_hops+1):
        targets=np.bincount(destination[frontier[source]],minlength=neurons)>0
        frontier=targets&(distance<0)
        distance[frontier]=hop
        if not frontier.any():break
    return distance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-hops',type=int,default=16)
    p.add_argument('--cpus',default='117,118,119')
    args=p.parse_args();pin(list(map(int,args.cpus.split(','))))
    if not 1<=args.max_hops<=64:p.error('Choose 1..64 structural hops')
    args.output.resolve().relative_to(args.root.resolve())
    with StorageBudget(args.root).reserve(files=32*(1<<20),heap=3*GIB,purpose='fixed-circuit information-route audit'):
        args.output.mkdir(parents=True,exist_ok=False)
        graph_path=Path(json.loads((args.root/'runs/m4/graph.json').read_text())['path'])
        graph=load_graph(graph_path);n=len(graph['type_id'])
        annotation=feather.read_table(graph_path/'annotations.feather')
        np.testing.assert_array_equal(annotation['bodyId'].to_numpy(),np.load(graph_path/'body_id.npy'))
        superclass=np.asarray([value or '<unknown>' for value in annotation['superclass'].to_pylist()])
        visual=np.asarray(annotation['class'].to_pylist())=='visual'
        coordinates=np.stack([annotation[key].to_numpy() for key in ('assignedOlHex1','assignedOlHex2')],axis=1)
        annotated=np.isfinite(coordinates).all(axis=1)
        types=np.asarray([value or '<unknown>' for value in annotation['type'].to_pylist()])
        root_side=annotation['rootSide'].to_numpy();soma_side=annotation['somaSide'].to_numpy()
        side=np.where(np.isin(soma_side,['L','R']),soma_side,root_side)
        ports,_=initialize(graph,FlyConfig(steps=4))
        readout=ports['output_group']>=0
        pool_count=np.bincount(ports['output_group'][readout],minlength=656)
        sources=dict(all_sensory=graph['sensory'],visual_sensory=graph['sensory'][visual[graph['sensory']]])
        routes={};arrays={}
        for name,inputs in sources.items():
            distance=distances(graph['src'],graph['dst'],n,inputs,args.max_hops)
            arrays[name+'_distance']=distance
            passes={}
            for k in (1,2,4,8,16):
                if k-1>args.max_hops:continue
                reached=(distance>=0)&(distance<k)
                pool=np.bincount(ports['output_group'][readout&reached],minlength=656)
                passes[k]=dict(neurons=int(reached.sum()),readout_neurons=int((reached&readout).sum()),
                    annotated_visual_neurons=int((reached&annotated).sum()),
                    readout_pool_reachable_fraction_quantiles=np.quantile(pool/pool_count,[0,.1,.5,.9,1]).tolist(),
                    by_superclass={str(kind):dict(reached=int(reached[superclass==kind].sum()),total=int((superclass==kind).sum()))
                                   for kind in np.unique(superclass)})
                assert sum(row['total'] for row in passes[k]['by_superclass'].values())==n
            routes[name]=dict(inputs=len(inputs),hop_histogram=np.bincount(distance[distance>=0],minlength=args.max_hops+1).tolist(),
                              unreached_within_limit=int((distance<0).sum()),passes=passes)
        arrays['annotated_columns']=annotated
        np.savez(args.output/'routes.npz',**arrays)
        incoming=np.bincount(graph['dst'],weights=np.abs(graph['strength']),minlength=n)
        counts=np.bincount(graph['type_id'])
        result=dict(status='passed',graph_id=graph['manifest']['graph_id'],neurons=n,edges=len(graph['src']),
            annotation_sha256=sha256(graph_path/'annotations.feather'),routes_sha256=sha256(args.output/'routes.npz'),
            routes=routes,types=len(counts),singleton_types=int((counts==1).sum()),
            initial_incoming_absolute_weight_quantiles=np.quantile(incoming[incoming>0],[0,.1,.5,.9,1]).tolist(),
            direct_column_cells=int(annotated.sum()),direct_column_types=len(np.unique(graph['type_id'][annotated])),
            direct_columns_by_type=dict(Counter(str(x) for x in types[annotated])),
            sensory_visual_types=dict(Counter(str(x) for x in types[sources['visual_sensory']])),
            direct_columns_by_side=dict(Counter(str(x) for x in side[annotated])),
            side_rule='somaSide when L/R, otherwise rootSide',
            pass_convention='Sensory drive first enters state at pass 1; a path with d directed edges can first affect pass d+1.',
            scope='Structural reach is an upper bound on causal influence. ReLU gates, signs, attenuation, learned weights and readout cancellation are not represented. Unreached means beyond the declared hop limit or disconnected; no topology is changed.')
        atomic_json(args.output/'result.json',result)
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
