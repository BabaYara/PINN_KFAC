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


def heat_residual(
    model: Callable[[jnp.ndarray], jnp.ndarray],
    xt: jnp.ndarray,
    diffusivity: float = 1.0,
    forcing_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Return the heat equation residual ``u_t - α u_xx - f``.

    Parameters
    ----------
    model : Callable
        Network taking ``(x, t)`` concatenated as a 1D array and returning
        ``u(x, t)``.
    xt : jnp.ndarray
        Points of shape ``(batch, 2)`` where the residual is evaluated.
    diffusivity : float, optional
        Diffusion coefficient ``α`` in the heat equation, by default ``1.0``.
    forcing_fn : Callable, optional
        Optional forcing term ``f(x, t)``. If ``None``, the equation is assumed
        homogeneous.

    Returns
    -------
    jnp.ndarray
        Residual values at each point in ``xt``.
    """

    def single_residual(z: jnp.ndarray) -> jnp.ndarray:
        val, derivs = forward_derivatives(lambda y: model(y).squeeze(), z, order=2)
        grad = derivs[1]
        hess = derivs[2]
        res = grad[1] - diffusivity * hess[0, 0]
        if forcing_fn is not None:
            res = res - forcing_fn(z)
        return res

    return jax.vmap(single_residual)(xt)


def burgers_residual(
    model: Callable[[jnp.ndarray], jnp.ndarray],
    xt: jnp.ndarray,
    viscosity: float = 0.01,
    forcing_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Return the 1D Burgers residual ``u_t + u u_x - ν u_xx - f``.

    Parameters
    ----------
    model : Callable
        Network mapping ``(x, t)`` to ``u(x, t)``.
    xt : jnp.ndarray
        Points ``(batch, 2)`` where the residual is evaluated.
    viscosity : float, optional
        Viscosity coefficient ``ν``. Defaults to ``0.01``.
    forcing_fn : Callable, optional
        Optional forcing term ``f(x, t)``. If ``None``, assumes zero forcing.

    Returns
    -------
    jnp.ndarray
        Residual values at each input ``(x, t)``.
    """

    def single_residual(z: jnp.ndarray) -> jnp.ndarray:
        val, derivs = forward_derivatives(lambda y: model(y).squeeze(), z, order=2)
        u = val
        grad = derivs[1]
        hess = derivs[2]
        res = grad[1] + u * grad[0] - viscosity * hess[0, 0]
        if forcing_fn is not None:
            res = res - forcing_fn(z)
        return res

    return jax.vmap(single_residual)(xt)
