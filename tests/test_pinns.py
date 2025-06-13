import jax.numpy as jnp
from pinns import gradient, laplacian


def test_gradient_and_laplacian():
    f = lambda x: jnp.sin(x).sum()
    x0 = jnp.array([0.0])
    grad = gradient(f, x0)
    lap = laplacian(f, x0)
    assert jnp.allclose(grad, 1.0)
    assert jnp.allclose(lap, 0.0)
