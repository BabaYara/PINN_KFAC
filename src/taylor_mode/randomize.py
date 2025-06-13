from typing import Callable
import jax
import jax.numpy as jnp
from .forward import forward_derivatives


def stochastic_laplacian(f: Callable[[jnp.ndarray], jnp.ndarray], x: jnp.ndarray, samples: int = 10) -> jnp.ndarray:
    """Approximate Laplacian using random projections."""
    v = jax.random.normal(jax.random.PRNGKey(0), (samples,) + x.shape)
    hvp = jax.vmap(lambda v_i: jax.jvp(lambda z: forward_derivatives(f, z, order=2)[1][2], (x,), (v_i,))[1])(v)
    return hvp.mean()
