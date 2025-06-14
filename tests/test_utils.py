import random
import numpy as np
import jax
from utils import seed_rng


def test_seed_rng_determinism():
    key1 = seed_rng(0)
    py_rand1 = random.random()
    np_rand1 = np.random.rand()
    jax_rand1 = jax.random.uniform(key1)

    key2 = seed_rng(0)
    py_rand2 = random.random()
    np_rand2 = np.random.rand()
    jax_rand2 = jax.random.uniform(key2)

    assert py_rand1 == py_rand2
    assert np.allclose(np_rand1, np_rand2)
    assert jax_rand1 == jax_rand2
