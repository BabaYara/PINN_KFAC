from dataclasses import dataclass, field
from typing import Dict
import jax.numpy as jnp

@dataclass
class Jet:
    """Representation of a function's value and its derivatives.

    Parameters
    ----------
    value : jnp.ndarray
        Value of the function.
    derivatives : Dict[int, jnp.ndarray], optional
        Mapping from derivative order to array. The default is an empty dict.
    """

    value: jnp.ndarray
    derivatives: Dict[int, jnp.ndarray] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - simple convenience
        return f"Jet(value={self.value}, orders={list(self.derivatives)})"
