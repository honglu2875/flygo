"""Bounded immutable feature caches, shared by readers on one host."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import shutil
import time
import uuid

import numpy as np

from .corpus import atomic_json,identity
from ..qualify import sha256
from ..storage import StorageBudget,GIB


CACHE_VERSION='go-9x9-h4-fp32-d4-v1'


def load_cached(root:Path,manifest:dict,build):
    """A single writer publishes atomically; readers verify and map immutable arrays.

    These caches are admitted under the shared RAM budget. This module never
    evicts files: active readers and corpus/checkpoint ownership stay explicit.
    """
    from .loader import LAYOUT
    directory=root/'cache/features'/CACHE_VERSION/manifest['dataset_id']
    directory.parent.mkdir(parents=True,exist_ok=True)
    with directory.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not directory.exists():
            size=manifest['positions']*sum(np.dtype(dtype).itemsize*int(np.prod(shape))
                                           for dtype,shape in LAYOUT.values())
            with StorageBudget(root).reserve(files=size+manifest['positions']*24+16*1024**2,
                                             heap=2*GIB,purpose='shared feature cache '+manifest['dataset_id']):
                temporary=directory.with_name('.'+directory.name+'.'+uuid.uuid4().hex+'.staging')
                temporary.mkdir()
                try:
                    arrays,indexes=build(temporary)
                    for array in arrays.values():array.flush()
                    for name,index in indexes.items():
                        np.save(temporary/('index-'+name+'.npy'),index,allow_pickle=False)
                    files={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(temporary.glob('*.npy'))}
                    receipt=dict(version=CACHE_VERSION,dataset_id=manifest['dataset_id'],
                                 positions=manifest['positions'],arrays=list(arrays),indexes=list(indexes),
                                 files=files,created=time.time())
                    receipt['cache_id']=identity(receipt)
                    atomic_json(temporary/'manifest.json',receipt)
                    del arrays
                    temporary.rename(directory)
                finally:
                    if temporary.exists():shutil.rmtree(temporary)
    receipt=json.loads((directory/'manifest.json').read_text())
    if (receipt['version']!=CACHE_VERSION or receipt['dataset_id']!=manifest['dataset_id']
            or receipt['positions']!=manifest['positions']
            or identity({k:v for k,v in receipt.items() if k!='cache_id'})!=receipt['cache_id']):
        raise ValueError('Feature cache identity mismatch')
    for name,record in receipt['files'].items():
        if Path(name).name!=name or sha256(directory/name)!=record['sha256']:
            raise ValueError('Feature cache hash mismatch: '+name)
    load=lambda name:np.load(directory/(name+'.npy'),mmap_mode='r',allow_pickle=False)
    arrays={name:load(name) for name in receipt['arrays']}
    indexes={name:load('index-'+name) for name in receipt['indexes']}
    return arrays,indexes
