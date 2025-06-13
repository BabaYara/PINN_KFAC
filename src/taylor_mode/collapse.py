from typing import Callable, Tuple, Dict
import jax
import jax.numpy as jnp
from .forward import forward_derivatives


def forward_derivatives_collapsed(
    f: Callable[[jnp.ndarray], jnp.ndarray], x: jnp.ndarray, order: int = 1
) -> Tuple[jnp.ndarray, Dict[int, jnp.ndarray]]:
    """Collapsed Taylor-mode derivatives.

    Currently this is just a thin wrapper around :func:`forward_derivatives`.
    A more efficient implementation could collapse terms following Dangel et al.
    """

    return forward_derivatives(f, x, order=order)
