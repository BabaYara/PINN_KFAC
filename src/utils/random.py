"""Random number utility functions."""

from __future__ import annotations

import random
import numpy as np
import jax


def seed_rng(seed: int) -> jax.Array:
    """Seed Python, NumPy, and return a JAX PRNG key."""
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    return key
