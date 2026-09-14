#!/usr/bin/env python3
"""Count directed connectivity inside annotated groups; never alter topology."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.feather as feather

from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB, StorageBudget


def partition(src, dst, synapses, labels):
    names, ids, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    groups = len(names)
    internal = np.zeros(groups, np.int64)
    weighted = np.zeros(groups, np.int64)
    incoming = np.zeros(groups, np.int64)
    outgoing = np.zeros(groups, np.int64)
    for start in range(0, len(src), 1_000_000):
        a, b = src[start:start+1_000_000], dst[start:start+1_000_000]
        x, y = ids[a], ids[b]
        use = (x == y) & (a != b)
        internal += np.bincount(x[use], minlength=groups)
        weighted += np.bincount(x[use], weights=synapses[start:start+1_000_000][use], minlength=groups).astype(np.int64)
        outgoing += np.bincount(x, minlength=groups)
        incoming += np.bincount(y, minlength=groups)
    rows = [dict(group=str(name), neurons=int(n), edges=int(e), synapses=int(s),
                 density=float(e/(n*(n-1))) if n>1 else None,
                 internal_fraction_of_outgoing=float(e/o) if o else None,
                 internal_fraction_of_incoming=float(e/i) if i else None)
            for name,n,e,s,i,o in zip(names,sizes,internal,weighted,incoming,outgoing)]
    assert all(r['edges'] <= r['neurons']*(r['neurons']-1) for r in rows)
    return rows, ids, names, sizes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpus',default='58,59')
    args=p.parse_args();pin(list(map(int,args.cpus.split(','))))
    args.output.resolve().relative_to(args.root.resolve())
    if args.output.exists():raise ValueError('Use a new immutable audit output')
    started=time.time()
    with StorageBudget(args.root).reserve(files=16<<20,heap=2*GIB,purpose='read-only anatomical group density audit'):
        record=json.loads((args.root/'runs/m4/graph.json').read_text());graph=Path(record['path'])
        src,dst,synapses=[np.load(graph/(key+'.npy'),mmap_mode='r') for key in ('src','dst','synapse_count')]
        annotation=feather.read_table(graph/'annotations.feather')
        np.testing.assert_array_equal(annotation['bodyId'].to_numpy(),np.load(graph/'body_id.npy',mmap_mode='r'))
        n=len(annotation);side=np.asarray(annotation['somaSide'].to_pylist())
        roots=np.asarray(annotation['rootSide'].to_pylist())
        side=np.where(np.isin(side,['L','R']),side,np.where(np.isin(roots,['L','R']),roots,'?'))
        columns={k:np.asarray([v or '<unknown>' for v in annotation[k].to_pylist()])
                 for k in ('class','superclass','type')}
        results={}
        for key,values in columns.items():
            labels=np.char.add(np.char.add(values,'/'),side.astype(str))
            rows,ids,names,sizes=partition(src,dst,synapses,labels)
            results[key]=rows
            if key=='type':
                a,b=ids[src],ids[dst]
                eligible=(sizes[a]>=20)&(sizes[b]>=20)&(a!=b)&(src!=dst)
                keys=a[eligible].astype(np.int64)*len(names)+b[eligible]
                keys,counts=np.unique(keys,return_counts=True)
                pairs=[]
                for k,count in zip(keys,counts):
                    x,y=divmod(int(k),len(names))
                    if count<100 or '<unknown>' in names[x] or '<unknown>' in names[y]:continue
                    pairs.append(dict(source=str(names[x]),target=str(names[y]),source_neurons=int(sizes[x]),
                        target_neurons=int(sizes[y]),edges=int(count),density=float(count/(sizes[x]*sizes[y]))))
                results['type_pairs_top_density']=sorted(pairs,key=lambda r:(-r['density'],-r['edges']))[:100]
                del a,b,eligible,keys
        q,r=[annotation[key].to_numpy() for key in ('assignedOlHex1','assignedOlHex2')]
        valid=np.isfinite(q)&np.isfinite(r)&np.isin(side,['L','R'])
        labels=np.full(n,'<unknown>',dtype='U48')
        for index in np.flatnonzero(valid):labels[index]=f'{side[index]}/{q[index]:g}/{r[index]:g}'
        rows,_,_,_=partition(src,dst,synapses,labels)
        results['direct_visual_columns']=[row for row in rows if row['group']!='<unknown>']
        selected=[row for row in results['direct_visual_columns'] if row['neurons']>=2]
        summary=dict(columns=len(results['direct_visual_columns']),annotated_neurons=int(valid.sum()),
            columns_with_at_least_2_neurons=len(selected),
            neuron_count_quantiles=np.quantile([r['neurons'] for r in selected],[0,.1,.5,.9,1]).tolist() if selected else [],
            density_quantiles=np.quantile([r['density'] for r in selected],[0,.1,.5,.9,1]).tolist() if selected else [],
            internal_edges=sum(r['edges'] for r in selected),
            possible_pairs=sum(r['neurons']*(r['neurons']-1) for r in selected),
            pooled_density=sum(r['edges'] for r in selected)/sum(r['neurons']*(r['neurons']-1) for r in selected) if selected else None)
        loops=int(np.count_nonzero(src==dst))
        output=dict(status='passed',graph_id=record['graph_id'],neurons=n,edges=len(src),self_edges=loops,
            whole_graph_density=(len(src)-loops)/(n*(n-1)),annotation_sha256=sha256(graph/'annotations.feather'),
            results=results,visual_column_summary=summary,seconds=time.time()-started,
            definition='Directed neuron pairs with >=2 minconf-0.5 synapses in the fixed Traced MaleCNS v1.0 graph. Internal density excludes self edges and divides by n*(n-1). Pair blocks divide by source_n*target_n.',
            scope='Class labels are not full neuropil membership. Direct column annotations are incomplete and identify assigned columns, not all arborization overlap. Left/right comes from somaSide, falling back to rootSide. Small groups and degree differences confound raw density; these are descriptive counts, not a degree-controlled enrichment test. No graph change or task-dependent group selection.')
        atomic_json(args.output,output)
        print(json.dumps(dict(output=str(args.output),whole_graph_density=output['whole_graph_density'],visual_columns=summary)))


if __name__=='__main__':main()
