"""Supervised gradient routing, separate from inference equations and Adam."""
import numpy as np

DEFAULT_VALUE_CORE_SCALE = 1.0
OBJECTIVE_VERSION = 'policy-ce-value-mse-core-scale-v1'


def value_core_scale(value):
    """Return the actual FP32 coefficient; reject silent overflow or underflow."""
    # argparse passes text; application code and saved arrays use real scalars.
    if isinstance(value, str):
        try: value = float(value)
        except ValueError: raise ValueError('Value core scale must be numeric') from None
    if (not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or value < 0
            or value > np.finfo(np.float32).max or (value > 0 and np.float32(value) == 0)):
        raise ValueError('Value core scale must be a finite nonnegative FP32 value')
    return float(np.float32(value))


def saved_value_core_scale(metadata):
    scale = value_core_scale(metadata.get('value_core_scale', DEFAULT_VALUE_CORE_SCALE))
    contract = metadata.get('training_contract', {})
    if 'value_core_scale' in contract and value_core_scale(contract['value_core_scale']) != scale:
        raise ValueError('Inconsistent checkpoint value core scale contracts')
    version = metadata.get('objective_version')
    if version not in (None, OBJECTIVE_VERSION) or (scale != 1 and version is None):
        raise ValueError('Unsupported checkpoint objective contract')
    return scale


def objective_arrays(scale):
    scale = value_core_scale(scale)
    # Keep legacy default numerical arrays intact; alternatives have an exact tag.
    return {} if scale == 1 else {'objective_value_core_scale': np.asarray(scale, np.float32)}


def check_objective_restore(scale, arrays):
    scale = value_core_scale(scale)
    stored = np.asarray(arrays.get('objective_value_core_scale', np.asarray(1, np.float32)))
    if stored.shape != () or stored.dtype != np.float32:
        raise ValueError('Invalid checkpoint value core scale')
    if value_core_scale(stored.item()) != scale:
        raise ValueError('Checkpoint value core scale differs from the learner')


def check_objective_resume(scale, training_contract):
    previous = value_core_scale(training_contract.get('value_core_scale', 1))
    if value_core_scale(scale) != previous:
        raise ValueError('Resume value core scale differs from the checkpoint')
