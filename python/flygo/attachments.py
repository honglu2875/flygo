"""External visual/context maps; no change to the fixed recurrent circuit."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .qualify import sha256
from .vision import SphericalRenderer, individual_motor_ports

VERSION = 'spherical-context-h4-ablation-v1'
MODES = ('history', 'current', 'neutral')


def context_ports(neurons, sensors, motors, context_nodes, *, seed=918421, size=9):
    """Balanced fixed context assignment; anatomical selection is done by the atlas.

    Context features are 81 point legalities, turn, signed komi/area, and passes/2.
    The allocation seed is independent of the model initialization seed.
    """
    ports = individual_motor_ports(neurons, sensors, motors)
    nodes = np.asarray(context_nodes)
    count = size*size+3
    if (nodes.ndim != 1 or nodes.dtype.kind not in 'iu' or len(nodes) < count
            or len(np.unique(nodes)) != len(nodes) or nodes.min() < 0 or nodes.max() >= neurons
            or np.intersect1d(nodes, np.r_[sensors, motors]).size):
        raise ValueError('Context nodes must be distinct, disjoint, in range and cover every feature')
    ports['input_index'][nodes] = len(sensors) + np.random.default_rng(seed).permutation(len(nodes)) % count
    return ports


@dataclass(frozen=True)
class VisualContext:
    renderer: SphericalRenderer

    @property
    def features(self):
        return len(self.renderer.index) + self.renderer.size**2 + 3

    def encode(self, features, mode):
        if mode not in MODES:
            raise ValueError('Unknown visual input mode')
        x = np.asarray(features, np.float32)
        size = self.renderer.size
        if x.ndim != 4 or x.shape[1:] != (size,size,12) or not np.isfinite(x).all():
            raise ValueError('Expected finite four-board NHWC features')
        if np.any((x[...,:8] < 0) | (x[...,:8] > 1)):
            raise ValueError('Occupancy planes must lie in [0,1]')
        if not np.all(x[...,8:11] == x[:,0:1,0:1,8:11]):
            raise ValueError('Turn, komi and pass context must be spatially constant')
        if np.any((x[...,11] != 0) & (x[...,11] != 1)):
            raise ValueError('Point legality must be binary')
        if mode == 'neutral':
            visual = np.full((len(x),len(self.renderer.index)), .5, np.float32)
        else:
            image = x
            if mode == 'current':
                image = x.copy()
                for lag in range(1,4): image[...,2*lag:2*lag+2] = x[...,:2]
            visual = self.renderer.render(image)
        return np.concatenate([visual, x[...,11].reshape(len(x),size*size), x[:,0,0,8:11]],axis=1)


def load_attachment(path, *, graph_id, dataset_id=None):
    path = Path(path)
    receipt = json.loads(path.with_suffix('.json').read_text())
    if (receipt.get('version') != VERSION or receipt.get('status') != 'complete'
            or receipt['graph_id'] != graph_id or sha256(path) != receipt['sha256']
            or (dataset_id is not None and receipt['dataset_id'] != dataset_id)):
        raise ValueError('Visual/context attachment identity mismatch')
    with np.load(path,allow_pickle=False) as data:
        arrays = {k:data[k].copy() for k in data.files}
    renderer = SphericalRenderer(**{k:arrays[k] for k in ('index','weight','unit','side','patch')})
    adapter = VisualContext(renderer)
    ports = {k:arrays[k] for k in ('input_index','output_group','output_scale')}
    expected = context_ports(len(ports['input_index']), arrays['sensors'], arrays['motors'],
                             arrays['context_nodes'], seed=receipt['context_seed'])
    if (adapter.features != receipt['features'] or len(arrays['motors']) != receipt['groups']
            or renderer.index.shape != renderer.weight.shape or renderer.index.shape[1] != 4
            or np.any((renderer.index < 0) | (renderer.index >= 324))
            or not np.isfinite(renderer.weight).all() or np.any(renderer.weight < 0)
            or np.any(renderer.weight.sum(axis=1) > 1+1e-6)
            or len(renderer.index) != len(arrays['sensors'])
            or any(not np.array_equal(ports[k],expected[k]) for k in ports)):
        raise ValueError('Invalid visual/context attachment arrays')
    return adapter, ports, receipt


def input_contract(receipt, mode):
    if mode not in MODES: raise ValueError('Unknown visual input mode')
    return dict(version=VERSION, attachment_sha256=receipt['sha256'], mode=mode)


def runtime_hashes():
    from . import _native
    base = Path(__file__).parent
    names = ('attachments.py','vision.py','fly.py','readout.py','optimizer.py','objectives.py','train.py',
             'checkpoint.py','play.py','diagnostics.py','jax/learner.py','jax/model.py','jax/numerics.py')
    return {**{name:sha256(base/name) for name in names}, 'native':sha256(Path(_native.__file__))}


def require_qualification(path, contract, config, *, batch_size, rate, epsilon, clip, rate_scales, head_mask=None,
                          clip_mode='global',value_core_scale=1.0):
    from .optimizer import validate_clip_mode
    from .objectives import value_core_scale as canonical_scale
    validate_clip_mode(clip_mode)
    scale = canonical_scale(value_core_scale)
    report = json.loads(Path(path).read_text())
    expected = dict(rate=rate, epsilon=epsilon, clip=clip, rate_scales=rate_scales, clip_mode=clip_mode,
                    value_core_scale=scale)
    if report.get('status') != 'complete' or report.get('runtime_sha256') != runtime_hashes():
        raise ValueError('Attachment training requires a completed qualification of this runtime')
    for record in report['records']:
        if (record.get('input_contract') == contract and record['batch_size'] == batch_size
                and record.get('head_mask') == (None if head_mask is None else head_mask.contract)
                and record['updates'] == 3 and {'clip_mode':'global', 'value_core_scale':1.0, **record['optimizer']} == expected
                and all(record['model'][k] == getattr(config,k) for k in
                        ('seed','steps','features','groups','actions','rate_softness','readout_mean_scale'))):
            return
    raise ValueError('No matching full-circuit attachment/optimizer qualification')
