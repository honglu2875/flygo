"""Small fixed-connectome model interface; backend choice does not change the equations."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from . import _native

PARAMETERS = ('edge', 'leak', 'bias', 'input_gain', 'readout_gain', 'policy_weight',
              'policy_bias', 'value_weight', 'value_bias')
MODEL_VERSION = 'leaky-rate-v1'
OPTIMIZER_VERSION = 'adam-fp32-v1'


@dataclass(frozen=True)
class FlyConfig:
    steps: int = 8
    features: int = 972
    groups: int = 656
    actions: int = 82
    threads: int = 16
    seed: int = 1


def load_graph(path: Path) -> dict:
    from .qualify import sha256
    manifest = json.loads((path / 'manifest.json').read_text())
    graph = {name: np.load(path / (name + '.npy'), mmap_mode='r', allow_pickle=False)
             for name in ('indptr','src','dst','type_id','sign','strength','sensory')}
    graph['manifest'] = manifest
    for name,array in graph.items():
        if name == 'manifest':
            continue
        if sha256(path/(name+'.npy')) != manifest['arrays'][name]['sha256']:
            raise ValueError('Fixed graph array hash mismatch: '+name)
    return graph


def initialize(graph: dict, config: FlyConfig):
    rng = np.random.default_rng(config.seed)
    n = len(graph['type_id'])
    types = int(graph['type_id'].max()) + 1
    sensory = graph['sensory']
    input_index = np.full(n, -1, np.int32)
    shuffled = rng.permutation(len(sensory))
    input_index[sensory] = (shuffled % config.features).astype(np.int32)
    output_group = np.full(n, -1, np.int32)
    readout = np.flatnonzero(input_index < 0)
    output_group[readout] = (rng.permutation(len(readout)) % config.groups).astype(np.int32)
    counts = np.bincount(output_group[readout], minlength=config.groups)
    output_scale = np.zeros(n, np.float32)
    output_scale[readout] = (1 / np.sqrt(counts[output_group[readout]])).astype(np.float32)
    ports = dict(input_index=input_index,output_group=output_group,output_scale=output_scale)
    magnitude = np.abs(graph['strength']).astype(np.float64)
    if np.any(magnitude <= 0):
        raise ValueError('Every fixed edge needs a strictly positive initial magnitude')
    params = dict(edge=np.log(np.expm1(magnitude)).astype(np.float32),
                  leak=np.zeros(types,np.float32),bias=np.full(types,0.02,np.float32),
                  input_gain=np.full(config.features,0.2,np.float32),
                  readout_gain=np.ones(n,np.float32),
                  policy_weight=rng.normal(0,0.02,(config.actions,config.groups)).astype(np.float32).ravel(),
                  policy_bias=np.zeros(config.actions,np.float32),
                  value_weight=rng.normal(0,0.02,config.groups).astype(np.float32),
                  value_bias=np.zeros(1,np.float32))
    return ports, params


def packed(params):
    return [np.ascontiguousarray(params[name],dtype=np.float32).reshape(-1) for name in PARAMETERS]


class RustFly:
    def __init__(self, graph: dict, config: FlyConfig = FlyConfig(), *, ports=None, params=None):
        self.graph, self.config = graph, config
        initial_ports, initial_params = initialize(graph, config)
        self.ports = initial_ports if ports is None else ports
        params = initial_params if params is None else params
        self.native = _native.FlyModel(graph['indptr'],graph['src'],graph['type_id'],graph['sign'],
            self.ports['input_index'],self.ports['output_group'],self.ports['output_scale'],
            config.features,config.groups,config.actions,config.threads,packed(params))

    def _input(self, features):
        array = np.asarray(features,dtype=np.float32)
        if array.ndim < 2 or int(np.prod(array.shape[1:])) != self.config.features:
            raise ValueError('Expected batch-major Go features with the configured feature count')
        return np.ascontiguousarray(array.reshape(len(array),self.config.features).T)

    def infer(self, features, *, trace=False):
        input_array = self._input(features)
        logits, value, states = self.native.infer(input_array,self.config.steps,trace)
        result = dict(logits=logits.reshape(-1,self.config.actions),value=value)
        if trace:
            result['states'] = [s.reshape(len(self.graph['type_id']),-1) for s in states]
        return result

    def parameters(self):
        return dict(zip(PARAMETERS,self.native.parameters()))

    def loss_and_grad(self, features, legal, policy, value):
        pl,vl,grad = self.native.loss_and_grad(self._input(features),self.config.steps,
            np.ascontiguousarray(legal,dtype=np.uint8),np.ascontiguousarray(policy,dtype=np.float32),
            np.ascontiguousarray(value,dtype=np.float32))
        return dict(policy_loss=pl,value_loss=vl),dict(zip(PARAMETERS,grad))

    def train_step(self, features, legal, policy, value, *, rate=0.003,clip=1.0,rate_scales=None):
        if rate_scales is not None and set(rate_scales)-set(PARAMETERS):
            raise ValueError('Unknown parameter group in learning-rate multipliers')
        extra=() if rate_scales is None else ([rate_scales.get(name,1.0) for name in PARAMETERS],)
        pl,vl,norm,step = self.native.train_step(self._input(features),self.config.steps,
            np.ascontiguousarray(legal,dtype=np.uint8),np.ascontiguousarray(policy,dtype=np.float32),
            np.ascontiguousarray(value,dtype=np.float32),rate,clip,*extra)
        return dict(policy_loss=pl,value_loss=vl,gradient_norm=norm,step=step)

    def checkpoint_arrays(self):
        params,first,second,step = self.native.checkpoint()
        return {**{prefix+name:array for prefix,group in [('param/',params),('first/',first),('second/',second)]
                   for name,array in zip(PARAMETERS,group)},'optimizer_step':np.asarray(step,dtype=np.uint64)}

    def restore_arrays(self, arrays):
        self.native.restore(*[[np.ascontiguousarray(arrays[prefix+name],dtype=np.float32) for name in PARAMETERS]
                              for prefix in ('param/','first/','second/')],int(arrays['optimizer_step']))
