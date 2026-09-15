"""Condition a regularized policy fit without changing its affine function class."""
from __future__ import annotations

import numpy as np

try:  # Frozen standalone research workers keep both helpers beside their entry.
    from readout_probe import loss_and_grad
except ModuleNotFoundError:
    from .readout_probe import loss_and_grad


def decompose(training):
    """Training-only centering and a complete, untruncated covariance basis."""
    x = np.asarray(training, np.float64)
    mean = x.mean(axis=0)
    z = x-mean
    covariance = z.T@z/len(z)
    values, vectors = np.linalg.eigh((covariance+covariance.T)/2)
    return mean, np.maximum(values, 0), vectors


def condition(features, mean, values, vectors, ridge):
    if ridge <= 0:
        raise ValueError('The convex probe requires strictly positive regularization')
    return (np.asarray(features, np.float64)-mean)@(vectors/np.sqrt(values+ridge))


def coefficients(matrix, values, vectors, ridge):
    """Map conditioned coefficients back to the centered feature coordinates."""
    return dict(weight=vectors@(matrix[:-1]/np.sqrt(values+ridge)[:,None]),
                bias=matrix[-1].copy())


def original_gradient(gradient, values, vectors, ridge):
    return np.concatenate((vectors@(gradient[:-1]*np.sqrt(values+ridge)[:,None]),
                           gradient[-1:]), axis=0)


class RidgePolicyObjective:
    """Cross entropy + ridge/2 * (||original weight||² + ||bias||²).

    The feature basis changes coordinates only. The original objective is
    ridge-strongly-convex, giving a gradient-based suboptimality bound
    ||original gradient||² / (2*ridge). No validation enters optimization.
    """
    def __init__(self, features, baseline, legal, target, values, ridge):
        if ridge <= 0:
            raise ValueError('Regularization must be positive')
        self.features = np.asarray(features, np.float64)
        self.baseline = np.asarray(baseline, np.float64)
        self.legal = np.asarray(legal, bool)
        self.target = np.asarray(target, np.float64)
        if (not np.isfinite(self.target).all() or (self.target < 0).any()
                or self.target[~self.legal].any() or not self.legal.any(axis=1).all()):
            raise ValueError('Invalid policy target or legal mask')
        self.shape = (features.shape[1]+1, baseline.shape[1])
        self.penalty = np.r_[ridge/(values+ridge), ridge][:,None]
        self.calls = 0
        self.history = []

    def __call__(self, vector):
        matrix = np.asarray(vector, np.float64).reshape(self.shape)
        loss, gradient = loss_and_grad(dict(weight=matrix[:-1],bias=matrix[-1]),
            self.features, self.baseline, self.legal, self.target)
        penalty = .5*float(np.sum(self.penalty*matrix*matrix))
        grad = np.concatenate((gradient['weight'],gradient['bias'][None]),axis=0)+self.penalty*matrix
        if not np.isfinite(loss+penalty) or not np.isfinite(grad).all():
            raise FloatingPointError('Nonfinite convex policy objective')
        self.calls += 1
        self.history.append(dict(call=self.calls,cross_entropy=loss,penalty=penalty,objective=loss+penalty,
                                 gradient_inf=float(np.max(np.abs(grad)))))
        return loss+penalty, grad.ravel()
