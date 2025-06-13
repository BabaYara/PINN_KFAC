from typing import Callable
import jax
import jax.numpy as jnp
from .forward import forward_derivatives


def stochastic_laplacian(
    f: Callable[[jnp.ndarray], jnp.ndarray], x: jnp.ndarray, samples: int = 10
) -> jnp.ndarray:
    """Approximate Laplacian of ``f`` at ``x`` using random projections."""

    key = jax.random.PRNGKey(0)
    vs = jax.random.normal(key, (samples,) + x.shape)

    grad_f = jax.grad(f)

    def second_directional(v_i: jnp.ndarray) -> jnp.ndarray:
        g = lambda z: jnp.vdot(grad_f(z), v_i)
        _, d = forward_derivatives(g, x, order=1)
        return jnp.vdot(v_i, d[1])

    hv_estimates = jax.vmap(second_directional)(vs)
    return hv_estimates.mean()
