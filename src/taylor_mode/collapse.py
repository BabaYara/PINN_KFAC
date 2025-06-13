from typing import Callable, Tuple, Dict
import jax
import jax.numpy as jnp


def forward_derivatives_collapsed(f: Callable[[jnp.ndarray], jnp.ndarray], x: jnp.ndarray, order: int = 1) -> Tuple[jnp.ndarray, Dict[int, jnp.ndarray]]:
    """Placeholder for collapsed Taylor-mode derivatives."""
    return f(x), {}
