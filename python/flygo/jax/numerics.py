"""Explicit accuracy for transcendental operations, including TPU lowering."""
import jax
import jax.numpy as jnp

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
    shifted = x - jnp.max(x, axis=-1, keepdims=True)
    total = jax.lax.exp(shifted, accuracy=HIGHEST).sum(axis=-1, keepdims=True)
    return shifted - jax.lax.log(total, accuracy=HIGHEST)
