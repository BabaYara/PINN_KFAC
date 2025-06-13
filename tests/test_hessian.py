import jax.numpy as jnp
from taylor_mode import hessian


def test_hessian_quadratic():
    f = lambda x: jnp.sum(x**2)
    x0 = jnp.array([1.0, -1.0])
    hess = hessian(f, x0)
    assert jnp.allclose(hess, 2 * jnp.eye(2))
