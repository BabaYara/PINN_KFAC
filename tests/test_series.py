import jax.numpy as jnp
from taylor_mode import taylor_series_coefficients


def test_taylor_series_coeffs_sin():
    coeffs = taylor_series_coefficients(jnp.sin, jnp.array(0.0), order=3)
    expected = [0.0, 1.0, 0.0, -1.0]
    for c, e in zip(coeffs, expected):
        assert jnp.allclose(c, e)
