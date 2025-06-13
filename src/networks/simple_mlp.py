"""Basic MLP utilities using JAX."""

from typing import Sequence, Callable, Tuple
import jax
import jax.numpy as jnp


Params = Sequence[Tuple[jnp.ndarray, jnp.ndarray]]


def init_mlp(layers: Sequence[int], key: jax.Array) -> Params:
    """Initialize weights and biases for a fully-connected network.

    Parameters
    ----------
    layers : Sequence[int]
        Layer sizes including input and output dimensions.
    key : jax.Array
        PRNG key for initialization.
    Returns
    -------
    list of (w, b)
        Weight matrices and bias vectors.
    """
    params = []
    keys = jax.random.split(key, len(layers) - 1)
    for k, (din, dout) in zip(keys, zip(layers[:-1], layers[1:])):
        w_key, b_key = jax.random.split(k)
        w = jax.random.normal(w_key, (din, dout)) / jnp.sqrt(din)
        b = jnp.zeros(dout)
        params.append((w, b))
    return params


def mlp_apply(params: Params, x: jnp.ndarray, activation: Callable = jax.nn.tanh) -> jnp.ndarray:
    """Apply an MLP to inputs ``x``."""
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = activation(x)
    return x
