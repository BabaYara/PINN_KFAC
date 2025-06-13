from dataclasses import dataclass
from typing import Dict
import jax.numpy as jnp

@dataclass
class Jet:
    """Representation of a function's value and derivatives at a point."""

    value: jnp.ndarray
    derivatives: Dict[int, jnp.ndarray]
