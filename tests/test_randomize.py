import jax.numpy as jnp
from taylor_mode import stochastic_laplacian


def test_stochastic_laplacian_quadratic():
    f = lambda x: jnp.sum(x**2)
    x0 = jnp.zeros(2)
    est = stochastic_laplacian(f, x0, samples=50)
    # Laplacian of x^2 + y^2 is 4
    assert jnp.allclose(est, 4.0, atol=0.5)
