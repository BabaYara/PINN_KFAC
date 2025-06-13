from typing import Callable

import jax
import jax.numpy as jnp

from taylor_mode.forward import forward_derivatives


def laplacian(f: Callable[[jnp.ndarray], jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    """Compute Laplacian of scalar function at ``x`` using Taylor-mode."""
    _, derivs = forward_derivatives(f, x, order=2)
    hess = derivs[2]
    return jnp.trace(hess)


def gradient(f: Callable[[jnp.ndarray], jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    """Compute gradient of scalar function at ``x`` using Taylor-mode."""
    _, derivs = forward_derivatives(f, x, order=1)
    return derivs[1]


def poisson_residual(
    model: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    forcing_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """Return the Poisson residual ``Δu(x) - f(x)`` for ``model`` at ``x``.

    Parameters
    ----------
    model : Callable
        Neural network mapping inputs ``x`` to scalar outputs ``u(x)``.
    x : jnp.ndarray
        Collocation points where the residual is evaluated.
    forcing_fn : Callable
        Forcing term ``f(x)`` of the Poisson equation ``Δu = f``.

    Returns
    -------
    jnp.ndarray
        Residual values ``Δu(x) - f(x)`` at each point in ``x``.
    """

    lap_fn = lambda z: laplacian(lambda y: model(y).squeeze(), z)
    lap_vals = jax.vmap(lap_fn)(x)
    return lap_vals - jax.vmap(forcing_fn)(x).squeeze()
