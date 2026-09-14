#!/usr/bin/env python3
"""Audit reduced source-input overlap without reading final-test examples or labels."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB, StorageBudget


def source_keys(features, mode):
    """Exact compact source keys, canonical under the learner's eight D4 transforms.

    Current uses own/opponent occupancy and legality; neutral uses legality.
    Both retain turn, signed komi and passes. These keys precede spherical
    rendering: they do not assert the absence of numerical retinal collisions.
    """
    if mode not in ('current', 'neutral'):raise ValueError('Expected a reduced input mode')
    x=np.asarray(features, np.float32)
    if x.ndim!=4 or x.shape[1:]!=(9,9,12) or not np.isfinite(x).all():
        raise ValueError('Expected finite NHWC features')
    board=x[..., [0,1,11] if mode=='current' else [11]]
    if np.any((board!=0)&(board!=1)) or not np.all(x[...,8:11]==x[:,0:1,0:1,8:11]):
        raise ValueError('Expected binary spatial features and constant context')
    context=np.ascontiguousarray(x[:,0,0,8:11],dtype='<f4').view(np.uint8).reshape(len(x),12)
    variants=[]
    for flip in (False,True):
        image=np.flip(board,axis=2) if flip else board
        for turns in range(4):
            packed=np.packbits(np.rot90(image,turns,axes=(1,2)).reshape(len(x),-1).astype(bool),axis=1)
            variants.append([row.tobytes() for row in np.concatenate([packed,context],axis=1)])
    return [min(row) for row in zip(*variants)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--release',default='v0-1m')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpus',default='56,57,58,59')
    args=p.parse_args();pin(list(map(int,args.cpus.split(','))))
    args.output.resolve().relative_to(args.root.resolve())
    with StorageBudget(args.root).reserve(files=8<<20,heap=2*GIB,purpose='reduced input novelty audit'):
        args.output.mkdir(parents=True,exist_ok=False);started=time.time()
        manifest,arrays,indexes=load_release(args.root,args.root/'releases'/args.release/'manifest.json',cache=True)
        indices=indexes['validation'];records={};masks={}
        for mode in ('current','neutral'):
            train=set()
            for start in range(0,len(indexes['train']),4096):
                train.update(source_keys(arrays['features'][indexes['train'][start:start+4096]],mode))
            novel=[]
            for start in range(0,len(indices),4096):
                novel.extend(key not in train for key in source_keys(arrays['features'][indices[start:start+4096]],mode))
            mask=np.asarray(novel,bool);masks[mode]=mask
            records[mode]=dict(train_distinct_sources=len(train),validation_positions=len(mask),
                validation_novel_sources=int(mask.sum()),validation_source_overlap=int((~mask).sum()))
            atomic_json(args.output/'status.json',dict(state='auditing',completed=mode,updated=time.time()))
        np.savez(args.output/'masks.npz',indices=indices,**masks)
        atomic_json(args.output/'result.json',dict(status='complete',created=time.time(),seconds=time.time()-started,
            dataset_id=manifest['dataset_id'],source_sha256=sha256(Path(__file__)),
            masks_sha256=sha256(args.output/'masks.npz'),records=records,
            scope='Exact D4-canonical reduced source inputs before rendering. Neutral keys describe actual constant-vision inputs; current keys do not certify absence of FP32 retinal collisions. No labels or final-test examples inspected.'))
        atomic_json(args.output/'status.json',dict(state='complete',updated=time.time()))
        print(json.dumps(records),flush=True)


if __name__=='__main__':main()
