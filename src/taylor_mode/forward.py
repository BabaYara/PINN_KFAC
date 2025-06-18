from typing import Callable, Tuple, Dict
import jax
import jax.numpy as jnp
from .jet import Jet


def forward_derivatives(
    f: Callable[[jnp.ndarray], jnp.ndarray], x: jnp.ndarray, order: int = 1
) -> Tuple[jnp.ndarray, Dict[int, jnp.ndarray]]:
    """Compute derivatives of ``f`` at ``x`` up to ``order`` using JAX.

    This routine repeatedly applies :func:`jax.jacfwd` to obtain higher-order
    derivatives. It is intended for scalar-output functions; the returned
    derivatives have shape ``(input_dim,) * n`` for the ``n``-th order.
    """

    value = f(x)
    derivs: Dict[int, jnp.ndarray] = {}

    if order < 1:
        return value, derivs

    jac_fn = jax.jacfwd(f)
    derivs[1] = jac_fn(x)

    cur_fn = jac_fn
    for n in range(2, order + 1):
        cur_fn = jax.jacfwd(cur_fn)
        derivs[n] = cur_fn(x)

    return value, derivs


def hessian(f: Callable[[jnp.ndarray], jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    """Convenience wrapper to return the Hessian matrix of ``f`` at ``x``."""
    _, derivs = forward_derivatives(f, x, order=2)
    return derivs[2]


def taylor_series_coefficients(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    order: int,
) -> list[jnp.ndarray]:
    """Return ``f`` and its derivatives up to ``order`` at ``x``.

    Parameters
    ----------
    f : Callable
        Scalar-valued function to differentiate.
    x : jnp.ndarray
        Point of evaluation.
    order : int
        Highest derivative order to compute.

    Returns
    -------
    list of jnp.ndarray
        ``[f(x), f'(x), ..., f^(order)(x)]``.
    """

    coeffs = []
    g = f
    for _ in range(order + 1):
        coeffs.append(g(x))
        g = jax.grad(g)
    return coeffs
