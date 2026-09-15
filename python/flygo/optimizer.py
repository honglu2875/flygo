"""Explicit optimizer choices; legacy checkpoints use Adam epsilon 1e-8."""
import numpy as np

DEFAULT_EPSILON = 1e-8
CLIP_MODES = ('global', 'parameter-group')
DEFAULT_CLIP_MODE = 'global'
OPTIMIZER_VERSION = 'adam-fp32-v1'


def validate_clip_mode(mode):
    if mode not in CLIP_MODES:
        raise ValueError('Unknown gradient clipping mode')


def optimizer_version(mode):
    validate_clip_mode(mode)
    return OPTIMIZER_VERSION if mode == DEFAULT_CLIP_MODE else 'adam-fp32-group-clipping-v1'


def saved_clipping_mode(metadata):
    mode = metadata.get('optimizer_clip_mode', DEFAULT_CLIP_MODE)
    validate_clip_mode(mode)
    contract = metadata.get('training_contract', {})
    if 'clip_mode' in contract and contract['clip_mode'] != mode:
        raise ValueError('Inconsistent checkpoint clipping contracts')
    return mode


def clipping_arrays(mode):
    """Legacy global checkpoints need no additional array; alternatives carry a tag."""
    validate_clip_mode(mode)
    return {} if mode == DEFAULT_CLIP_MODE else {'optimizer_clip_mode': np.asarray(CLIP_MODES.index(mode), np.uint8)}


def check_clipping_restore(mode, arrays):
    validate_clip_mode(mode)
    stored = np.asarray(arrays.get('optimizer_clip_mode', np.asarray(0, np.uint8)))
    if stored.shape != () or stored.dtype != np.uint8 or int(stored) >= len(CLIP_MODES):
        raise ValueError('Invalid checkpoint clipping mode')
    if CLIP_MODES[int(stored)] != mode:
        raise ValueError('Checkpoint clipping mode differs from the learner')


def check_clipping_resume(mode, training_contract):
    validate_clip_mode(mode)
    previous = training_contract.get('clip_mode', DEFAULT_CLIP_MODE)
    validate_clip_mode(previous)
    if previous != mode:
        raise ValueError('Resume clipping mode differs from the checkpoint')


def clipping_metrics(mode, norms, factors):
    """Actual pre-update norms and applied FP32 factors; no gradient recomputation."""
    return dict(mode=mode, group_norms={k: float(v) for k, v in norms.items()},
                factors={k: float(v) for k, v in factors.items()},
                clipped=any(v < 1.0 for v in factors.values()))


def validate_epsilon(epsilon):
    if (not np.isfinite(epsilon) or epsilon<=0 or epsilon>np.finfo(np.float32).max
            or np.float32(epsilon)==0):
        raise ValueError('Adam epsilon must be a finite positive FP32 value')


def check_epsilon_resume(epsilon,training_contract):
    validate_epsilon(epsilon)
    previous=training_contract.get('epsilon',DEFAULT_EPSILON)
    validate_epsilon(previous)
    if np.float32(epsilon)!=np.float32(previous):
        raise ValueError('Resume Adam epsilon differs from the checkpoint')
