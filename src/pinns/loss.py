from typing import Callable
import jax
import jax.numpy as jnp
from .operators import laplacian


def pinn_loss(
    model: Callable[[jnp.ndarray], jnp.ndarray],
    X_res: jnp.ndarray,
    X_bc: jnp.ndarray,
    bc_values: jnp.ndarray,
    forcing_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """Simple PINN loss for Poisson-like equations.

    Parameters
    ----------
    model : Callable
        Neural network mapping inputs ``x`` to scalar outputs ``u(x)``.
    X_res : jnp.ndarray
        Collocation points where the PDE residual is enforced.
    X_bc : jnp.ndarray
        Points where Dirichlet boundary conditions are enforced.
    bc_values : jnp.ndarray
        Boundary values ``u(x_bc)`` at ``X_bc``.
    forcing_fn : Callable
        Function representing the forcing term ``f(x)`` in ``\\Delta u = f``.
    Returns
    -------
    jnp.ndarray
        Scalar loss equal to mean squared residual plus boundary error.
    """

    # PDE residual: Laplacian of model minus forcing term
    lap_fn = lambda x: laplacian(lambda z: model(z).squeeze(), x)
    res = jax.vmap(lap_fn)(X_res) - jax.vmap(forcing_fn)(X_res).squeeze()
    res_loss = jnp.mean(res**2)

    # Boundary condition error
    bc_pred = jax.vmap(model)(X_bc).squeeze()
    bc_loss = jnp.mean((bc_pred - bc_values) ** 2)

    return res_loss + bc_loss
