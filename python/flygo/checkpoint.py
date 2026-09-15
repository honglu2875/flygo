"""Portable NumPy checkpoints with atomic publication and verified peer recovery."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import uuid

import numpy as np

from .data.corpus import atomic_json
from .qualify import sha256
from .storage import GIB,StorageBudget
from .replication import replicate_file
from .fly import MODEL_VERSION, OPTIMIZER_VERSION
from .optimizer import DEFAULT_CLIP_MODE, optimizer_version, saved_clipping_mode


def save_checkpoint(model,sampler,path:Path,metadata:dict,*,root:Path,peer:str|None=None):
    with StorageBudget(root).reserve(files=GIB,heap=GIB,purpose='training checkpoint and peer staging'):
        arrays=model.checkpoint_arrays()
        info=dict(schema_version=1,model_config=asdict(model.config),sampler=sampler.state(),
                  graph_id=model.graph['manifest']['graph_id'],model_version=getattr(model,'model_version',MODEL_VERSION),
                  optimizer_version=OPTIMIZER_VERSION,**metadata)
        info['numerical_runtime']=getattr(model,'numerical_runtime','rust-fp32-f64-norm-v1')
        info['optimizer_clip_mode']=getattr(model,'clip_mode',DEFAULT_CLIP_MODE)
        info['optimizer_version']=optimizer_version(info['optimizer_clip_mode'])
        saved_clipping_mode(info)
        arrays['metadata']=np.frombuffer(json.dumps(info,sort_keys=True,allow_nan=False).encode(),np.uint8)
        for name,array in model.ports.items():
            arrays['port/'+name]=array
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            raise ValueError('Checkpoint names are immutable')
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with temporary.open('wb') as stream:
                np.savez(stream,**arrays)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        digest=sha256(path)
        receipt=dict(path=str(path),sha256=digest,bytes=path.stat().st_size,step=int(arrays['optimizer_step']),
                     dataset_id=info['dataset_id'],graph_id=info['graph_id'],peer=None,
                     replica_status='pending' if peer else 'local_only')
        # Publish a recoverable local receipt before attempting the network copy.
        # A failed peer must not hide the intact checkpoint or terminate learning.
        atomic_json(path.with_suffix('.json'),receipt)
        if peer:
            try:
                replicate_file(path,root,peer)
                receipt.update(peer=peer,replica_status='verified')
                atomic_json(path.with_suffix('.json'),receipt)
                replicate_file(path.with_suffix('.json'),root,peer)
            except (OSError,RuntimeError,subprocess.SubprocessError) as error:
                receipt.update(replica_status='retry',replica_error=str(error))
                atomic_json(path.with_suffix('.json'),receipt)
        return receipt


def load_checkpoint(path:Path,model,sampler=None,*,dataset_id=None,numerical_runtime=None):
    receipt=json.loads(path.with_suffix('.json').read_text())
    if sha256(path)!=receipt['sha256']:
        raise ValueError('Checkpoint hash mismatch')
    with np.load(path,allow_pickle=False) as arrays:
        info=json.loads(arrays['metadata'].tobytes())
        if (numerical_runtime is not None
                and info.get('numerical_runtime','rust-fp32-f64-norm-v1') != numerical_runtime):
            raise ValueError('Checkpoint numerical runtime differs; training continuation requires its original runtime')
        mode=saved_clipping_mode(info)
        if mode!=getattr(model,'clip_mode',DEFAULT_CLIP_MODE):
            raise ValueError('Checkpoint clipping mode differs from the learner')
        # Early schema-1 checkpoints predate explicit names and contain this same
        # first model/optimizer. Future variants must use distinct version names.
        if info.get('schema_version')!=1 or info.get('model_version',MODEL_VERSION)!=getattr(model,'model_version',MODEL_VERSION) \
                or info.get('optimizer_version',OPTIMIZER_VERSION)!=optimizer_version(mode):
            raise ValueError('Unsupported checkpoint model or optimizer contract')
        if info['graph_id']!=model.graph['manifest']['graph_id']:
            raise ValueError('Checkpoint topology differs from the fixed graph')
        if dataset_id is not None and info['dataset_id']!=dataset_id:
            raise ValueError('Checkpoint dataset differs from the frozen training release')
        expected=asdict(model.config);stored=info['model_config'].copy()
        if 'rate_softness' in expected:stored.setdefault('rate_softness',0.0)
        if 'readout_mean_scale' in expected:stored.setdefault('readout_mean_scale',1.0)
        expected.pop('threads');stored.pop('threads')
        if expected!=stored:
            raise ValueError('Checkpoint numerical model configuration differs')
        for name,value in model.ports.items():
            if not np.array_equal(arrays['port/'+name],value):
                raise ValueError('Checkpoint sensory/readout attachment differs')
        model.restore_arrays(arrays)
        if sampler is not None:
            sampler.restore(info['sampler'])
    return info
