from typing import Callable, Tuple, Dict
import jax
import jax.numpy as jnp
from .jet import Jet


def forward_derivatives(f: Callable[[jnp.ndarray], jnp.ndarray], x: jnp.ndarray, order: int = 1) -> Tuple[jnp.ndarray, Dict[int, jnp.ndarray]]:
    """Compute derivatives of ``f`` at ``x`` up to ``order`` using JAX.

    Parameters
    ----------
    f : Callable
        Scalar-output function.
    x : jnp.ndarray
        Input array.
    order : int
        Highest derivative order to compute.

    Returns
    -------
    value : jnp.ndarray
        ``f(x)``.
    derivs : Dict[int, jnp.ndarray]
        Mapping from derivative order to derivative array.
    """
    value = f(x)
    derivs: Dict[int, jnp.ndarray] = {}
    if order >= 1:
        derivs[1] = jax.jacfwd(f)(x)
    if order >= 2:
        derivs[2] = jax.jacfwd(jax.jacfwd(f))(x)
    return value, derivs
