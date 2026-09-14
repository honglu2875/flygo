"""Explicit optimizer choices; legacy checkpoints use Adam epsilon 1e-8."""
import numpy as np

DEFAULT_EPSILON = 1e-8


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
