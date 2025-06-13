from typing import Callable
import jax
import jax.numpy as jnp
from taylor_mode.forward import forward_derivatives


def laplacian(f: Callable[[jnp.ndarray], jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    """Compute Laplacian of scalar function at ``x`` using Taylor-mode."""
    _, derivs = forward_derivatives(f, x, order=2)
    hess = derivs[2]
    return jnp.trace(hess)
