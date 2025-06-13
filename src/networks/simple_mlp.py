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


def mlp_apply(
    params: Params, x: jnp.ndarray, activation: Callable = jax.nn.tanh
) -> jnp.ndarray:
    """Apply an MLP to inputs ``x``."""
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = activation(x)
    return x


def mlp_forward_activations(
    params: Params, x: jnp.ndarray, activation: Callable = jax.nn.tanh
) -> Tuple[jnp.ndarray, Sequence[jnp.ndarray], Sequence[jnp.ndarray]]:
    """Forward pass that also returns activations and pre-activations.

    Parameters
    ----------
    params : Params
        Weight matrices and biases.
    x : jnp.ndarray
        Input minibatch ``(batch, in_dim)``.
    activation : Callable, optional
        Non-linearity, by default ``jax.nn.tanh``.

    Returns
    -------
    Tuple containing ``output``, ``acts`` and ``preacts`` lists. ``acts[i]`` is
    the input to layer ``i`` (after activation of the previous layer) and
    ``preacts[i]`` is ``acts[i] @ w + b``.
    """

    acts = []
    preacts = []
    a = x
    for i, (w, b) in enumerate(params):
        acts.append(a)
        z = a @ w + b
        preacts.append(z)
        if i < len(params) - 1:
            a = activation(z)
        else:
            a = z
    return a, acts, preacts


def mlp_backprops(
    params: Params,
    preacts: Sequence[jnp.ndarray],
    loss_grad: jnp.ndarray,
    activation: Callable = jax.nn.tanh,
) -> Sequence[jnp.ndarray]:
    """Compute backpropagated gradients for each layer output.

    Parameters
    ----------
    params : Params
        Weights and biases of the network.
    preacts : Sequence[jnp.ndarray]
        Pre-activation values for each layer from :func:`mlp_forward_activations`.
    loss_grad : jnp.ndarray
        Gradient of the loss w.r.t. the network output.
    activation : Callable, optional
        Activation function used in the network, by default ``jax.nn.tanh``.

    Returns
    -------
    List of gradients w.r.t. each layer's pre-activation output, ordered from
    input to output layer.
    """

    g = loss_grad
    backprops = []
    for i in reversed(range(len(params))):
        w, _ = params[i]
        backprops.insert(0, g)
        if i > 0:
            g = (g @ w.T) * (1.0 - jnp.tanh(preacts[i - 1]) ** 2)
    return backprops
