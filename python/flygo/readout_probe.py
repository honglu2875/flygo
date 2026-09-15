"""Small frozen-feature policy probes; independent of the recurrent learner.

All heads learn a residual over the same frozen policy logits. Float64 arithmetic
keeps this representation diagnostic separate from full-circuit precision gates.
Normalization uses training features only; no value loss or core update occurs.
"""
from __future__ import annotations

import numpy as np


def normalizer(training, floor=1e-6):
    x = np.asarray(training, np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return mean, np.maximum(std, floor), std


def transform(features, variant, mean, scale, *, top_k=128):
    x = np.asarray(features, np.float64)
    if variant == 'raw-linear':
        return x
    z = (x-mean)/scale
    if variant in ('scaled-linear', 'scaled-mlp'):
        return z
    if variant != 'gated-linear':
        raise ValueError('Unknown readout probe: '+variant)
    # Positive deviations from each neuron's training mean compete. Stable sort
    # resolves equal activations by motor index; zero entries stay zero.
    z = np.maximum(z, 0)
    keep = np.argsort(-z, axis=1, kind='stable')[:, :min(top_k, z.shape[1])]
    result = np.zeros_like(z)
    np.put_along_axis(result, keep, np.take_along_axis(z, keep, axis=1), axis=1)
    return result


def initialize(features, actions, *, hidden=0, seed=1):
    rng = np.random.default_rng(seed)
    p = dict(weight=np.zeros((hidden or features, actions), np.float64),
             bias=np.zeros(actions, np.float64))
    if hidden:
        p.update(hidden_weight=rng.normal(0, 1/np.sqrt(features), (features, hidden)),
                 hidden_bias=np.zeros(hidden, np.float64))
    return p


def forward(params, features, baseline):
    x = features
    if 'hidden_weight' in params:
        x = np.maximum(x@params['hidden_weight']+params['hidden_bias'], 0)
    return np.asarray(baseline, np.float64)+x@params['weight']+params['bias']


def log_policy(logits, legal):
    legal = np.asarray(legal, bool)
    if not legal.any(axis=1).all():
        raise ValueError('Every policy needs a legal action')
    z = np.where(legal, logits, -np.inf)
    z = z-np.max(z, axis=1, keepdims=True)
    denominator = np.exp(z).sum(axis=1, keepdims=True)
    return np.where(legal, z-np.log(denominator), -1e30)


def loss_and_grad(params, features, baseline, legal, target):
    x = np.asarray(features, np.float64)
    target = np.asarray(target, np.float64)
    hidden = x@params['hidden_weight']+params['hidden_bias'] if 'hidden_weight' in params else None
    h = x if hidden is None else np.maximum(hidden, 0)
    logits = np.asarray(baseline, np.float64)+h@params['weight']+params['bias']
    logp = log_policy(logits, legal)
    loss = -np.sum(target*logp)/len(target)
    dz = (target.sum(axis=1, keepdims=True)*np.exp(logp)-target)/len(target)
    grad = dict(weight=h.T@dz, bias=dz.sum(axis=0))
    if hidden is not None:
        dh = (dz@params['weight'].T)*(hidden > 0)
        grad.update(hidden_weight=x.T@dh, hidden_bias=dh.sum(axis=0))
    return float(loss), grad


def adam(params, grad, first, second, step, *, rate, epsilon=1e-8, clip=1.):
    norm = np.sqrt(sum(np.sum(g*g) for g in grad.values()))
    if not np.isfinite(norm):
        raise FloatingPointError('Nonfinite probe gradient')
    scale = min(1., clip/max(float(norm), 1e-30))
    for key, value in params.items():
        g = grad[key]*scale
        first[key] = .9*first[key]+.1*g
        second[key] = .999*second[key]+.001*g*g
        value -= rate*(first[key]/(1-.9**step))/(np.sqrt(second[key]/(1-.999**step))+epsilon)
    return float(norm)


def position_metrics(logits, legal, target):
    target = np.asarray(target, np.float64)
    logp = log_policy(np.asarray(logits, np.float64), legal)
    p = np.exp(logp)
    teacher_entropy = -np.sum(target*np.log(np.maximum(target, 1e-300)), axis=1)
    kl = -np.sum(target*logp, axis=1)-teacher_entropy
    return dict(policy_kl=kl, policy_entropy=-np.sum(p*logp, axis=1),
                teacher_entropy=teacher_entropy, top_probability=p.max(axis=1),
                teacher_top1=(np.argmax(p, axis=1) == np.argmax(target, axis=1)).astype(np.float64))
