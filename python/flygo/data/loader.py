"""Frozen-corpus loading and bounded, deterministic sampling for the offline learner."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import _native
from ..go import GameConfig, _json
from ..qualify import sha256
from .corpus import identity,split_for_family


LAYOUT = dict(features=(np.float32,(9,9,12)),legal=(np.bool_,(82,)),
              raw_policy=(np.float32,(82,)),raw_value=(np.float32,()),
              ply=(np.int32,()),game_index=(np.int32,()),opponent_index=(np.int8,()))


def load_release(root: Path, manifest_path: Path, *, max_positions=2_000_000, cache=False):
    manifest=json.loads(manifest_path.read_text())
    if identity({k:v for k,v in manifest.items() if k!='dataset_id'})!=manifest['dataset_id']:
        raise ValueError('Frozen manifest identity mismatch')
    if manifest['positions']>max_positions:
        raise ValueError('Release exceeds the bounded loader capacity; prepare a smaller release or qualify a larger bound')
    replication=json.loads((manifest_path.parent/'replication.json').read_text())
    if replication.get('status')!='passed' or replication['dataset_id']!=manifest['dataset_id']:
        raise ValueError('Release has not completed peer-copy verification')
    if cache:
        from .cache import load_cached
        arrays,indexes=load_cached(root,manifest,lambda directory:_load_arrays(root,manifest,directory))
    else:
        arrays,indexes=_load_arrays(root,manifest)
    return manifest,arrays,indexes


def _load_arrays(root,manifest,directory=None):
    """Preallocate once; one game's replay is the only additional feature buffer."""
    count=manifest['positions']
    arrays={name:(np.empty((count,*shape),dtype=dtype) if directory is None else
                  np.lib.format.open_memmap(directory/(name+'.npy'),mode='w+',dtype=dtype,shape=(count,*shape)))
            for name,(dtype,shape) in LAYOUT.items()}
    splits=defaultdict(list);offset=0
    for game_index,record in enumerate(manifest['records']):
        path=root/record['path']
        if sha256(path)!=record['sha256']:
            raise ValueError('Frozen game hash mismatch')
        with np.load(path,allow_pickle=False) as game:
            metadata=json.loads(game['metadata'].tobytes())
            if any(metadata[key]!=record[key] for key in ('game_id','rows','opening_family','split','contract_id')):
                raise ValueError('Frozen game metadata differs from its manifest')
            if split_for_family(record['opening_family'])!=record['split']:
                raise ValueError('Frozen game family split differs from the split contract')
            actions=np.ascontiguousarray(game['actions'],dtype=np.int32)
            end=offset+len(actions)
            if len(actions)!=record['rows'] or end>count:
                raise ValueError('Frozen game row count differs from its manifest')
            features=_native.replay_features(_json(GameConfig()),actions,np.asarray([0,len(actions)],dtype=np.int64))
            arrays['features'][offset:end]=features.reshape(len(actions),9,9,12)
            arrays['ply'][offset:end]=np.arange(len(actions),dtype=np.int32)
            arrays['game_index'][offset:end]=game_index
            arrays['opponent_index'][offset:end]=record['opponent_index']
            for key in ('legal','raw_policy','raw_value'):
                if game[key].shape!=(len(actions),*LAYOUT[key][1]):
                    raise ValueError('Frozen game column has an unexpected shape: '+key)
                arrays[key][offset:end]=game[key]
            splits[record['split']].extend(range(offset,end))
            offset=end
    if offset!=count:raise ValueError('Frozen release position count differs from its records')
    indexes={key:np.asarray(value,dtype=np.int64) for key,value in splits.items()}
    if 'train' not in indexes or 'validation' not in indexes:
        raise ValueError('Training and validation games are required')
    # Hash the exact model inputs after all history/komi/legal channels are encoded.
    def key(index):
        board=arrays['features'][index]
        # Training uses D4 augmentation, so novelty must exclude every equivalent
        # orientation, including full history, player/komi and legal channels.
        variants=[]
        for flip in (False,True):
            transformed=np.flip(board,axis=1) if flip else board
            for turns in range(4):
                variants.append(hashlib.blake2b(np.rot90(transformed,turns,axes=(0,1)).tobytes(),digest_size=16).digest())
        return min(variants)
    train_inputs={key(index) for index in indexes['train']}
    indexes['validation_novel']=np.asarray([i for i in indexes['validation'] if key(i) not in train_inputs],np.int64)
    return arrays,indexes


class Sampler:
    def __init__(self,arrays,indexes,seed=1):
        self.arrays,self.indexes=arrays,indexes
        self.rng=np.random.default_rng(seed)
        self.batches=0

    def batch(self,count,*,augment=True):
        indices=self.rng.choice(self.indexes['train'],size=count,replace=True)
        x=self.arrays['features'][indices].copy()
        legal=self.arrays['legal'][indices].copy()
        policy=self.arrays['raw_policy'][indices].copy()
        value=self.arrays['raw_value'][indices].copy()
        if augment:
            for b in range(count):
                symmetry=int(self.rng.integers(8))
                flip,turns=symmetry//4,symmetry%4
                board=x[b]
                p=policy[b,:81].reshape(9,9)
                mask=legal[b,:81].reshape(9,9)
                if flip:
                    board=np.flip(board,axis=1);p=np.flip(p,axis=1);mask=np.flip(mask,axis=1)
                x[b]=np.rot90(board,turns,axes=(0,1))
                policy[b,:81]=np.rot90(p,turns).ravel()
                legal[b,:81]=np.rot90(mask,turns).ravel()
        self.batches+=1
        return x,legal,policy,value

    def state(self):
        return dict(rng=self.rng.bit_generator.state,batches=self.batches)

    def restore(self,state):
        self.rng.bit_generator.state=state['rng'];self.batches=state['batches']
