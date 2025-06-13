from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import jax.numpy as jnp


@dataclass
class KFACOptimizer:
    lr: float = 1e-3
    damping: float = 1e-2

    def step(self, params: Iterable[jnp.ndarray], grads: Iterable[jnp.ndarray]) -> Iterable[jnp.ndarray]:
        """A dummy step that applies gradient descent."""
        return [p - self.lr * g for p, g in zip(params, grads)]
