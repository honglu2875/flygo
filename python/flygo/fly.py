"""Small fixed-connectome model interface; backend choice does not change the equations."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from . import _native
from .optimizer import (DEFAULT_EPSILON, validate_epsilon, DEFAULT_CLIP_MODE,
                        validate_clip_mode, clipping_arrays, check_clipping_restore, clipping_metrics,
                        OPTIMIZER_VERSION)
from .readout import check_restore

PARAMETERS = ('edge', 'leak', 'bias', 'input_gain', 'readout_gain', 'policy_weight',
              'policy_bias', 'value_weight', 'value_bias')
MODEL_VERSION = 'leaky-rate-v1'


@dataclass(frozen=True)
class FlyConfig:
    steps: int = 8
    features: int = 972
    groups: int = 656
    actions: int = 82
    threads: int = 16
    seed: int = 1
    rate_softness: float = 0.0
    readout_mean_scale: float = 1.0

    def __post_init__(self):
        s=self.rate_softness
        if not math.isfinite(s) or s<0 or s>np.finfo(np.float32).max or (s>0 and np.float32(s)==0):
            raise ValueError('Rate softness must be a finite, nonnegative FP32 value')
        m=self.readout_mean_scale
        if not math.isfinite(m) or not 0<m<=1 or np.float32(m)==0:
            raise ValueError('Readout mean scale must be an FP32 value in (0, 1]')

    @property
    def model_version(self):
        return ('leaky-rate'+('-softplus' if self.rate_softness else '')+
                ('-mean-scaled' if self.readout_mean_scale!=1 else '')+'-v1')


def firing_rate(voltage,softness=0.0):
    """NumPy rate for diagnostics; learning uses independent Rust/JAX equations."""
    voltage=np.asarray(voltage,np.float32)
    if not softness:return np.maximum(voltage,np.float32(0))
    scale=np.float32(softness)
    with np.errstate(over='ignore',under='ignore'):
        return np.maximum(voltage,np.float32(0))+scale*np.log1p(np.exp(-np.abs(voltage)/scale))


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
    numerical_runtime = 'rust-fp32-circuit-f64-head-reductions-v3'

    @property
    def clip_mode(self):
        return self._clip_mode

    @property
    def model_version(self):
        return self.config.model_version + ('+masked-readout-v1' if self.head_mask is not None else '')

    def __init__(self, graph: dict, config: FlyConfig = FlyConfig(), *, ports=None, params=None, head_mask=None,
                 clip_mode=DEFAULT_CLIP_MODE):
        validate_clip_mode(clip_mode)
        self._clip_mode = clip_mode
        self.graph, self.config = graph, config
        self.head_mask = head_mask
        if head_mask is not None: head_mask.validate_shape(actions=config.actions, groups=config.groups)
        initial_ports, initial_params = initialize(graph, config)
        self.ports = initial_ports if ports is None else ports
        params = initial_params if params is None else params
        mask_args = {} if head_mask is None else dict(policy_mask=head_mask.policy.ravel(), value_mask=head_mask.value)
        self.native = _native.FlyModel(graph['indptr'],graph['src'],graph['type_id'],graph['sign'],
            self.ports['input_index'],self.ports['output_group'],self.ports['output_scale'],
            config.features,config.groups,config.actions,config.threads,packed(params),
            config.rate_softness,config.readout_mean_scale,**mask_args)

    def _input(self, features):
        array = np.asarray(features,dtype=np.float32)
        if array.ndim < 2 or int(np.prod(array.shape[1:])) != self.config.features:
            raise ValueError('Expected batch-major Go features with the configured feature count')
        return np.ascontiguousarray(array.reshape(len(array),self.config.features).T)

    def infer(self, features, *, trace=False, prune=False):
        input_array = self._input(features)
        logits, value, states = self.native.infer(input_array,self.config.steps,trace,prune)
        result = dict(logits=logits.reshape(-1,self.config.actions),value=value)
        if trace:
            result['states'] = [s.reshape(len(self.graph['type_id']),-1) for s in states]
        return result

    def prediction_dependencies(self):
        """Required neurons/incoming edges per pass; counts precede zero skipping."""
        return [dict(neurons=neurons,edges=edges) for neurons,edges in
                self.native.prediction_dependencies(self.config.steps)]

    def parameters(self):
        return dict(zip(PARAMETERS,self.native.parameters()))

    def embedding(self, features):
        """Batch-major individual/pool rates, without exporting recurrent states."""
        pooled, _, _ = self.native.embedding(self._input(features), self.config.steps)
        return pooled.reshape(self.config.groups, -1).T.copy()

    def _embedding_objective(self, features, objective, update=None):
        x = self._input(features)
        pooled, value, revision = self.native.embedding(x, self.config.steps)
        embedding = pooled.reshape(self.config.groups, -1).T
        metrics, de, dv = objective(embedding, value)
        de, dv = np.asarray(de, np.float32), np.asarray(dv, np.float32)
        if de.shape != embedding.shape or dv.shape != value.shape:
            raise ValueError('Objective cotangents must match the embedding and value shapes')
        grad, result = self.native.embedding_backward(x, self.config.steps,
            np.ascontiguousarray(de.T), np.ascontiguousarray(dv), revision, update, self.clip_mode)
        return metrics, grad, result

    def embedding_loss_and_grad(self, features, objective):
        """Objective returns (metrics, d_embedding, d_linear_score). Recomputes the tape."""
        metrics, grad, _ = self._embedding_objective(features, objective)
        return metrics, dict(zip(PARAMETERS, grad))

    def train_embedding(self, features, objective, *, rate=0.003, clip=1.0,
                        rate_scales=None, epsilon=DEFAULT_EPSILON):
        """Prototype external objective: two forwards, one backward, one Rust Adam update."""
        validate_epsilon(epsilon)
        if rate_scales is not None and set(rate_scales)-set(PARAMETERS):
            raise ValueError('Unknown parameter group in learning-rate multipliers')
        scales = [1.0 if rate_scales is None else rate_scales.get(name, 1.0) for name in PARAMETERS]
        metrics, _, (norm, step) = self._embedding_objective(features, objective,
            (rate, clip, scales, epsilon))
        return dict(metrics, gradient_norm=norm, step=step, clipping=self._clipping_metrics())

    def _clipping_metrics(self):
        norms, factors = self.native.update_statistics()
        return clipping_metrics(self.clip_mode, dict(zip(PARAMETERS, norms)), dict(zip(PARAMETERS, factors)))

    def loss_and_grad(self, features, legal, policy, value):
        pl,vl,grad = self.native.loss_and_grad(self._input(features),self.config.steps,
            np.ascontiguousarray(legal,dtype=np.uint8),np.ascontiguousarray(policy,dtype=np.float32),
            np.ascontiguousarray(value,dtype=np.float32))
        return dict(policy_loss=pl,value_loss=vl),dict(zip(PARAMETERS,grad))

    def train_step(self, features, legal, policy, value, *, rate=0.003,clip=1.0,rate_scales=None,epsilon=DEFAULT_EPSILON):
        validate_epsilon(epsilon)
        if rate_scales is not None and set(rate_scales)-set(PARAMETERS):
            raise ValueError('Unknown parameter group in learning-rate multipliers')
        scales=None if rate_scales is None else [rate_scales.get(name,1.0) for name in PARAMETERS]
        pl,vl,norm,step = self.native.train_step(self._input(features),self.config.steps,
            np.ascontiguousarray(legal,dtype=np.uint8),np.ascontiguousarray(policy,dtype=np.float32),
            np.ascontiguousarray(value,dtype=np.float32),rate,clip,scales,epsilon,self.clip_mode)
        return dict(policy_loss=pl,value_loss=vl,gradient_norm=norm,step=step,clipping=self._clipping_metrics())

    def profile_sparse(self,features,*,repetitions=5):
        """Read-only warm kernel timings; synthetic cotangent, no optimizer update."""
        return dict(self.native.profile_sparse(self._input(features),self.config.steps,repetitions))

    def checkpoint_arrays(self):
        params,first,second,step = self.native.checkpoint()
        return {**{prefix+name:array for prefix,group in [('param/',params),('first/',first),('second/',second)]
                   for name,array in zip(PARAMETERS,group)},'optimizer_step':np.asarray(step,dtype=np.uint64),
                **({} if self.head_mask is None else self.head_mask.checkpoint_arrays()),
                **clipping_arrays(self.clip_mode)}

    def restore_arrays(self, arrays):
        check_clipping_restore(self.clip_mode, arrays)
        check_restore(self.head_mask, arrays)
        self.native.restore(*[[np.ascontiguousarray(arrays[prefix+name],dtype=np.float32) for name in PARAMETERS]
                              for prefix in ('param/','first/','second/')],int(arrays['optimizer_step']))
