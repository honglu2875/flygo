"""Fixed external head masks, independent of the recurrent graph and optimizer."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


PREFIX = 'readout_mask/'
VERSION = 'binary-readout-v1'


@dataclass(frozen=True)
class HeadMask:
    policy: np.ndarray
    value: np.ndarray
    provenance: str

    def __post_init__(self):
        policy, value = np.asarray(self.policy), np.asarray(self.value)
        if (policy.ndim != 2 or value.ndim != 1 or policy.shape[1] != len(value)
                or min(policy.shape) < 1
                or any(a.dtype.kind not in 'biu' or np.any((a != 0) & (a != 1)) for a in (policy, value))):
            raise ValueError('Head masks must be binary action-by-group and group arrays')
        if not policy.any(axis=1).all() or not value.any():
            raise ValueError('Every head output must retain at least one input')
        provenance = json.loads(self.provenance)
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError('Head masks need a nonempty provenance object')
        object.__setattr__(self, 'provenance', json.dumps(provenance, sort_keys=True, separators=(',', ':'), allow_nan=False))
        for name, array in [('policy', policy), ('value', value)]:
            frozen = np.frombuffer(np.ascontiguousarray(array, dtype=np.uint8).tobytes(), np.uint8).reshape(array.shape)
            object.__setattr__(self, name, frozen)

    def validate_shape(self, *, actions, groups):
        if self.policy.shape != (actions, groups) or self.value.shape != (groups,):
            raise ValueError('Head mask dimensions differ from the model')

    @property
    def contract(self):
        digest = hashlib.sha256()
        digest.update(VERSION.encode())
        digest.update(np.asarray(self.policy.shape, dtype='<u8').tobytes())
        digest.update(self.policy.tobytes())
        digest.update(self.value.tobytes())
        digest.update(self.provenance.encode())
        return dict(version=VERSION, sha256=digest.hexdigest(),
                    policy_coefficients=int(self.policy.sum()), value_coefficients=int(self.value.sum()),
                    provenance=json.loads(self.provenance))

    def parameter_masks(self):
        return dict(policy_weight=self.policy.ravel(), value_weight=self.value)

    def checkpoint_arrays(self):
        return {PREFIX+'policy':self.policy.copy(), PREFIX+'value':self.value.copy(),
                PREFIX+'provenance':np.frombuffer(self.provenance.encode(), np.uint8).copy()}

    @classmethod
    def from_checkpoint(cls, arrays):
        keys = {k for k in arrays if k.startswith(PREFIX)}
        if not keys: return None
        if keys != {PREFIX+k for k in ('policy', 'value', 'provenance')}:
            raise ValueError('Incomplete or unknown head mask checkpoint fields')
        provenance = np.asarray(arrays[PREFIX+'provenance'])
        if provenance.ndim != 1 or provenance.dtype != np.uint8:
            raise ValueError('Invalid head mask provenance bytes')
        return cls(arrays[PREFIX+'policy'], arrays[PREFIX+'value'], provenance.tobytes().decode())


def check_restore(mask, arrays):
    """Check the immutable mask and disabled moments before any model mutation."""
    stored = HeadMask.from_checkpoint(arrays)
    if (None if mask is None else mask.contract) != (None if stored is None else stored.contract):
        raise ValueError('Checkpoint head mask or provenance differs')
    if mask is not None:
        for name, enabled in mask.parameter_masks().items():
            for prefix in ('first/', 'second/'):
                moment = np.asarray(arrays[prefix+name])
                if moment.shape != enabled.shape or np.any(moment[enabled == 0] != 0):
                    raise ValueError('Disabled head coefficients must have zero optimizer moments')


def load_head_mask(path, *, graph_id, attachment_sha256):
    from .qualify import sha256
    path = Path(path)
    receipt = json.loads(path.with_suffix('.json').read_text())
    if receipt.get('status') != 'complete' or sha256(path) != receipt['sha256']:
        raise ValueError('Head mask artifact hash or status differs')
    with np.load(path, allow_pickle=False) as data: mask = HeadMask.from_checkpoint(data)
    if mask is None or mask.contract != receipt['contract']:
        raise ValueError('Head mask artifact contract differs')
    provenance = mask.contract['provenance']
    if provenance.get('graph_id') != graph_id or provenance.get('attachment_sha256') != attachment_sha256:
        raise ValueError('Head mask topology or input attachment differs')
    return mask


def side_policy(sides, *, size=9):
    """Soma side is a declared proxy; the middle column, pass and midline are shared."""
    sides = np.asarray(sides)
    if sides.ndim != 1 or not len(sides) or not np.isin(sides, ['L', 'R', 'M']).all() or size < 3 or size % 2 != 1:
        raise ValueError('Need known L/R/M sides and an odd board size of at least three')
    mask = np.ones((size*size+1, len(sides)), np.uint8)
    for action in range(size*size):
        column = action % size
        if column < size//2: mask[action] = sides != 'R'
        elif column > size//2: mask[action] = sides != 'L'
    return mask


def shuffle_sides(sides, superclasses, *, seed):
    sides, superclasses = np.asarray(sides), np.asarray(superclasses)
    if sides.ndim != 1 or superclasses.shape != sides.shape or not np.isin(sides, ['L', 'R', 'M']).all():
        raise ValueError('Expected one known side and superclass per motor')
    result = sides.copy(); rng = np.random.default_rng(seed)
    for kind in np.unique(superclasses):
        indices = np.flatnonzero((superclasses == kind) & (sides != 'M'))
        result[indices] = rng.permutation(sides[indices])
    return result
