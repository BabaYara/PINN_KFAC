import jax
import jax.numpy as jnp
from networks import init_mlp, mlp_apply


def test_init_and_apply_mlp():
    key = jax.random.PRNGKey(0)
    params = init_mlp([1, 4, 1], key)
    x = jnp.array([[0.0]])
    y = mlp_apply(params, x)
    assert y.shape == (1, 1)
