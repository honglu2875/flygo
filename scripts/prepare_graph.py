#!/usr/bin/env python3
"""Reproduce the fixed MaleCNS graph from cached public source tables, preserving metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.runtime import cpu_profile, pin
from flygo.storage import GIB, StorageBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/home/cubic27/go/tmp/fly-connectome-analysis'))
    parser.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    args = parser.parse_args()
    pin(cpu_profile()['research_cpus'][-16:])
    started = time.time()
    with StorageBudget(args.root).reserve(files=3*GIB, heap=12*GIB, purpose='canonical full MaleCNS graph'):
        source = args.source
        originals = {p.name:dict(sha256=sha256(p),bytes=p.stat().st_size) for p in sorted((source/'data').glob('*.feather'))}
        graph = np.load(source / 'graph.npz', allow_pickle=False)
        body = graph['body_id']
        n = len(body)
        src, dst, count = graph['src'], graph['dst'], graph['synapse_count']
        if n != 165122 or len(src) != 15270273 or not np.all(body[1:] > body[:-1]):
            raise ValueError('Fixed graph identity/count changed')
        # Independently stream source rows and match every retained pair/count.
        reader = ipc.open_file(pa.memory_map(str(source/'data/connectome-weights-male-cns-v1.0-minconf-0.5.feather'), 'r'))
        keys = dst.astype(np.int64) * n + src
        if not np.all(keys[1:] > keys[:-1]):
            raise ValueError('Canonical edges must be unique and destination/source sorted')
        matched = np.zeros(len(src), dtype=bool)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            pre, post, weights = [batch.column(i).to_numpy() for i in range(3)]
            a, b = np.searchsorted(body, pre).clip(0,n-1), np.searchsorted(body,post).clip(0,n-1)
            valid = (body[a] == pre) & (body[b] == post) & (weights >= 2)
            edge_keys = b[valid].astype(np.int64) * n + a[valid]
            offsets = np.searchsorted(keys, edge_keys)
            if (np.any(offsets >= len(src)) or not np.array_equal(keys[offsets],edge_keys)
                    or not np.array_equal(count[offsets],weights[valid]) or np.any(matched[offsets])):
                raise ValueError('Cached graph does not reproduce original connection table')
            matched[offsets] = True
        if not matched.all():
            raise ValueError('Cached graph contains edges absent from source')
        ann = feather.read_table(source/'data/body-annotations-male-cns-v1.0-minconf-0.5.feather')
        ann = ann.filter(pc.equal(ann['status'],'Traced')).sort_by([('bodyId','ascending')])
        np.testing.assert_array_equal(ann['bodyId'].to_numpy(),body)
        nt = feather.read_table(source/'data/body-neurotransmitters-male-cns-v1.0.feather')
        nt_map = dict(zip(nt['body'].to_pylist(),nt['consensus_nt'].to_pylist()))
        consensus = np.array([nt_map.get(int(b)) or 'unclear' for b in body])
        sign = np.where(np.isin(consensus,['gaba','glutamate','histamine']),-1,1).astype(np.float32)
        mag = np.log1p(count).astype(np.float32)
        expected = (mag * sign[src] / np.bincount(dst,weights=mag,minlength=n)[dst]).astype(np.float32)
        np.testing.assert_array_equal(expected,graph['strength'])
        edge_hash = hashlib.sha256(body.tobytes()+src.tobytes()+dst.tobytes()).hexdigest()
        out = args.root/'graphs'/edge_hash
        out.mkdir(parents=True,exist_ok=False)
        superclass = np.array([x or '' for x in ann['superclass'].to_pylist()])
        types = np.array([x or '<unknown>' for x in ann['type'].to_pylist()])
        type_names, type_id = np.unique(types,return_inverse=True)
        sensory = np.flatnonzero(np.char.find(superclass, 'sensory') >= 0).astype(np.int32)
        arrays = dict(body_id=body,src=src,dst=dst,indptr=graph['indptr'],synapse_count=count,
                      strength=expected,sign=sign,type_id=type_id.astype(np.int32),sensory=sensory,
                      type_names=type_names,consensus_nt=consensus,superclass=superclass)
        # Transpose keeps edge IDs, so trainable parameter order is immutable.
        transpose = np.argsort(src,kind='stable').astype(np.int32)
        arrays['transpose_edges'] = transpose
        arrays['transpose_indptr'] = np.r_[0,np.cumsum(np.bincount(src,minlength=n))].astype(np.int32)
        for name,array in arrays.items():
            np.save(out/(name+'.npy'),array,allow_pickle=False)
        feather.write_feather(ann,out/'annotations.feather')
        feather.write_feather(nt,out/'neurotransmitters.feather')
        manifest = dict(schema_version=1,graph_id=edge_hash,neurons=n,edges=len(src),sensory_neurons=len(sensory),
                        source_tables=originals,cached_npz_sha256=sha256(source/'graph.npz'),
                        definition='Traced MaleCNS v1.0 neurons; all directed pairs with at least two minconf-0.5 synapses',
                        topology_fixed=True,canonical_order='ascending body ID; edges destination then source',
                        initial_weights='sign[pre] * log1p(synapse_count) / destination sum(log1p(count))',
                        sign_prior='negative: gaba/glutamate/histamine; positive otherwise; approximate modeling prior',
                        metadata='Original traced annotation rows and full NT table retained with nulls/confidences',
                        arrays={name:dict(shape=list(array.shape),dtype=str(array.dtype),sha256=sha256(out/(name+'.npy')))
                                for name,array in arrays.items()}, elapsed_seconds=time.time()-started)
        atomic_json(out/'manifest.json',manifest)
        atomic_json(args.root/'runs/m4/graph.json',dict(path=str(out),**manifest))
        print(json.dumps({k:v for k,v in manifest.items() if k not in ('arrays','source_tables')},indent=2))


if __name__ == '__main__':
    main()
