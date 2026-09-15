"""Explicit accuracy for transcendental operations, including TPU lowering."""
import jax
import jax.numpy as jnp
from functools import partial

HIGHEST = jax.lax.AccuracyMode.HIGHEST


def sigmoid(x):
    return jax.lax.logistic(x, accuracy=HIGHEST)


@jax.custom_jvp
def softplus(x):
    return jnp.maximum(x, 0) + jax.lax.log1p(
        jax.lax.exp(-jnp.abs(x), accuracy=HIGHEST), accuracy=HIGHEST)


@softplus.defjvp
def _softplus_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    # Explicit derivative also handles x=0 without abs/max subgradient ambiguity.
    return softplus(x), sigmoid(x) * dx


def log_softmax(x):
    # The common shift cancels analytically. Differentiating its maximum can
    # reintroduce the rounded sum of cotangents into the winning logit.
    shifted = x - jax.lax.stop_gradient(jnp.max(x, axis=-1, keepdims=True))
    total = jax.lax.exp(shifted, accuracy=HIGHEST).sum(axis=-1, keepdims=True)
    return shifted - jax.lax.log(total, accuracy=HIGHEST)


@partial(jax.custom_jvp,nondiff_argnums=(1,))
def smooth_rate(x,softness):
    """Fixed threshold softness; stable even when x/softness overflows."""
    scale=jnp.asarray(softness,dtype=x.dtype)
    return jnp.maximum(x,0)+scale*jax.lax.log1p(
        jax.lax.exp(-jnp.abs(x)/scale,accuracy=HIGHEST),accuracy=HIGHEST)


@smooth_rate.defjvp
def _smooth_rate_jvp(softness,primals,tangents):
    (x,),(dx,)=primals,tangents
    return smooth_rate(x,softness),sigmoid(x/jnp.asarray(softness,x.dtype))*dx


def firing_rate(x,softness=0.0):
    return smooth_rate(x,softness) if softness else jax.nn.relu(x)


def _condition_readout(x,scale):
    scale=jnp.asarray(scale,x.dtype)
    origin=x[:1]
    delta=x-origin
    return delta+(scale*origin-(1-scale)*delta.mean(axis=0,keepdims=True))


@partial(jax.custom_vjp,nondiff_argnums=(1,))
def condition_readout(x,scale):
    """Stable evaluation of a symmetric, fixed common-component projection."""
    return _condition_readout(x,scale)


def _condition_readout_fwd(x,scale):
    return _condition_readout(x,scale),None


def _condition_readout_bwd(scale,residual,cotangent):
    return (_condition_readout(cotangent,scale),)


condition_readout.defvjp(_condition_readout_fwd,_condition_readout_bwd)
