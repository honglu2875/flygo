"""Independent, read-only linearization of the hard-rate recurrent circuit.

The scientific learner stays in Rust. This FP64 diagnostic uses its recorded
states and independently reconstructed FP32 weights/leaks. Native VJP checks
quantify the precision difference before interpreting any gradient flow.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_array


QUANTILES = (0., .1, .5, .9, .99, 1.)


def summary(values):
    x = np.asarray(values, np.float64).ravel()
    if not len(x):
        return dict(count=0)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite diagnostic values')
    return dict(count=len(x), mean=float(x.mean()), rms=float(np.sqrt(np.mean(x*x))),
                nonzero=int(np.count_nonzero(x)), quantiles=np.quantile(x, QUANTILES).tolist())


def gradient_summary(values, epsilon=1e-6):
    x = np.asarray(values, np.float64).ravel()
    return dict(summary(np.abs(x)), norm=float(np.linalg.norm(x)),
                fraction_above_epsilon=float(np.mean(np.abs(x) > epsilon)) if len(x) else None)


def distances(indptr, source, starts, hops, *, reverse=False):
    """Shortest directed distances up to hops; -1 means beyond this horizon."""
    n = len(indptr)-1
    adjacency = csr_array((np.ones(len(source), np.float32), source, indptr), shape=(n,n))
    if reverse:
        adjacency = adjacency.T.tocsr()
    result = np.full(n, -1, np.int16)
    frontier = np.zeros(n, np.float32)
    frontier[starts] = 1
    result[starts] = 0
    for step in range(1, hops+1):
        next_frontier = (adjacency@frontier > 0) & (result < 0)
        result[next_frontier] = step
        frontier = next_frontier.astype(np.float32)
        if not next_frontier.any():
            break
    return result


class Linearization:
    def __init__(self, graph, params):
        self.source = np.asarray(graph['src'])
        self.destination = np.asarray(graph['dst'])
        self.types = np.asarray(graph['type_id'])
        self.type_count = len(params['bias'])
        theta = np.asarray(params['edge'], np.float32)
        # Mirror the explicit FP32 transforms, but use an independent library.
        magnitude = np.maximum(theta, np.float32(0)) + np.log1p(np.exp(-np.abs(theta)))
        self.weights = (np.asarray(graph['sign'])[self.source]*magnitude).astype(np.float64)
        self.edge_slope = (np.asarray(graph['sign'])[self.source]/(np.float32(1)+np.exp(-theta))).astype(np.float64)
        sigmoid = np.float32(1)/(np.float32(1)+np.exp(-np.asarray(params['leak'], np.float32)))
        self.alpha = (np.float32(.01)+np.float32(.98)*sigmoid)[self.types].astype(np.float64)
        self.leak_slope = (np.float32(.98)*sigmoid*(np.float32(1)-sigmoid))[self.types].astype(np.float64)
        n = len(self.types)
        self.matrix = csr_array((self.weights, self.source, graph['indptr']), shape=(n,n))
        self.transpose = self.matrix.T.tocsr()

    def adjoint(self, states, last, *, edge_chunk=65536):
        """Exact real-valued chain rule at supplied states; no optimizer step."""
        if len(states) < 2 or any(np.shape(s) != np.shape(last) for s in states):
            raise ValueError('Expected a complete node-major state tape')
        if not np.isfinite(last).all():
            raise ValueError('Invalid final cotangent')
        current = np.asarray(last, np.float64).copy()
        drive = np.zeros_like(current)
        edge = np.zeros(len(self.source), np.float64)
        bias = np.zeros(self.type_count, np.float64)
        leak = np.zeros_like(bias)
        cotangents = [None]*len(states)
        cotangents[-1] = current.copy()
        for step in range(len(states)-2, -1, -1):
            previous = np.asarray(states[step], np.float64)
            following = np.asarray(states[step+1], np.float64)
            message = self.alpha[:,None]*current
            drive += message
            bias += np.bincount(self.types, weights=message.sum(axis=1), minlength=self.type_count)
            leak += np.bincount(self.types, weights=(current*(following-previous)).sum(axis=1)
                               /self.alpha*self.leak_slope, minlength=self.type_count)
            rate = np.maximum(previous, 0)
            for begin in range(0, len(edge), edge_chunk):
                block = slice(begin, begin+edge_chunk)
                edge[block] += np.einsum('ij,ij->i', rate[self.source[block]], message[self.destination[block]])
            current = (1-self.alpha[:,None])*current + (previous > 0)*(self.transpose@message)
            cotangents[step] = current.copy()
        return dict(edge=edge*self.edge_slope, bias=bias, leak=leak), drive, cotangents

    def local_bound(self, state):
        """Upper bound on each row's absolute Jacobian sum, including leakage."""
        positive = (np.asarray(state) > 0).astype(np.float64)
        # Absolute weights are independent of batch/state; do not mutate W.
        absolute = csr_array((np.abs(self.weights), self.matrix.indices, self.matrix.indptr), shape=self.matrix.shape)
        return (1-self.alpha[:,None]) + self.alpha[:,None]*(absolute@positive)


def node_signal(current, neutral):
    rate = np.maximum(np.asarray(current, np.float64), 0)
    other = np.maximum(np.asarray(neutral, np.float64), 0)
    response = rate-other
    return dict(mean=rate.mean(axis=1), std=rate.std(axis=1),
                positive_fraction=(current > 0).mean(axis=1),
                visual_mean=response.mean(axis=1), visual_std=response.std(axis=1),
                visual_rms=np.sqrt(np.mean(response*response, axis=1)))


def input_gradient(drive_gradient, encoded, ports, feature_count):
    selected = np.flatnonzero(ports['input_index'] >= 0)
    features = ports['input_index'][selected]
    weights = (drive_gradient[selected]*np.asarray(encoded, np.float64).T[features]).sum(axis=1)
    return np.bincount(features, weights=weights, minlength=feature_count)


def pooled_cotangent(name, output, params, motors, policy, value, residual):
    """The native VJP receives a fixed cotangent; task heads are held fixed."""
    count = len(policy)
    weight = params['policy_weight'].reshape(output['logits'].shape[1], len(motors)).astype(float)
    gain = params['readout_gain'][motors].astype(float)
    logits = output['logits'].astype(float)
    if name == 'original-value':
        delta = output['value']-value
        score = (np.float32(2)*delta/np.float32(count))*(np.float32(1)-output['value']**2)
        pooled = (score[:,None].astype(float)*params['value_weight'].astype(float)).astype(np.float32)
        return np.zeros_like(pooled), score, pooled, dict(value_mse=float(np.mean(delta.astype(float)**2)))
    raw = np.maximum(output['states'][-1][motors].T, 0).astype(float)
    effective = weight.copy()
    if name == 'fitted-linear-policy':
        if np.min(np.abs(gain)) <= 1e-8:
            raise ValueError('A zero readout gain cannot transmit the raw residual cotangent through this API')
        logits += ((raw-residual['mean'])/residual['scale']-residual['center'])@residual['weight']+residual['bias']
        effective += (residual['weight']/residual['scale'][:,None]/gain[:,None]).T
    legal = residual['legal']
    shifted = np.where(legal, logits, -np.inf)
    shifted -= shifted.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    probability = exponential/exponential.sum(axis=1, keepdims=True)
    target = policy.astype(float)
    dz = (target.sum(axis=1, keepdims=True)*probability-target)/count
    if name == 'original-policy':
        dz = dz.astype(np.float32).astype(float)  # The native loss/circuit boundary.
    pooled = (dz@effective).astype(np.float32)
    logp = np.where(legal, shifted-np.log(exponential.sum(axis=1, keepdims=True)), -1e30)
    entropy = -np.sum(target*np.log(np.maximum(target,1e-300)),axis=1)
    return pooled, np.zeros(count,np.float32), pooled, dict(policy_kl=float(np.mean(-np.sum(target*logp,axis=1)-entropy)))
