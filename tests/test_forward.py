import jax.numpy as jnp
from taylor_mode import forward_derivatives

def test_forward_derivatives():
    f = lambda x: jnp.sin(x)
    val, derivs = forward_derivatives(f, jnp.array(0.0), order=2)
    assert jnp.allclose(val, 0.0)
    assert jnp.allclose(derivs[1], 1.0)
    assert jnp.allclose(derivs[2], 0.0)
