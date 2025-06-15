import jax.numpy as jnp
from pinns import gradient, laplacian, divergence
from pinns.operators import poisson_residual


def test_gradient_and_laplacian():
    f = lambda x: jnp.sin(x).sum()
    x0 = jnp.array([0.0])
    grad = gradient(f, x0)
    lap = laplacian(f, x0)
    assert jnp.allclose(grad, 1.0)
    assert jnp.allclose(lap, 0.0)


def test_divergence_simple_linear_field():
    f = lambda x: jnp.stack([2 * x[0], x[0] + x[1]])
    x0 = jnp.array([1.0, 2.0])
    div = divergence(f, x0)
    assert jnp.allclose(div, 3.0)


def test_pinn_loss_zero_for_exact_solution():
    # Poisson equation u'' = -sin(x) with u(x) = sin(x)
    model = lambda x: jnp.sin(x)
    forcing = lambda x: -jnp.sin(x)
    X_res = jnp.linspace(0.0, jnp.pi, 5).reshape(-1, 1)
    X_bc = jnp.array([[0.0], [jnp.pi]])
    bc_vals = jnp.array([0.0, 0.0])
    from pinns import pinn_loss

    loss = pinn_loss(model, X_res, X_bc, bc_vals, forcing)
    assert loss < 1e-6


def test_poisson_residual_zero_for_exact_solution():
    model = lambda x: jnp.sin(x)
    forcing = lambda x: -jnp.sin(x)
    x = jnp.linspace(0.0, jnp.pi, 5).reshape(-1, 1)
    res = poisson_residual(model, x, forcing)
    assert jnp.allclose(res, 0.0)


def test_heat_residual_zero_for_exact_solution():
    from pinns import heat_residual

    model = lambda xt: jnp.exp(-xt[1]) * jnp.sin(xt[0])
    xt = jnp.array([[0.1, 0.2], [0.3, 0.0]])
    res = heat_residual(model, xt)
    assert jnp.allclose(res, 0.0, atol=1e-6)


def test_burgers_residual_zero_for_constant_solution():
    from pinns.operators import burgers_residual

    model = lambda xt: jnp.array(0.0)
    xt = jnp.array([[0.0, 0.0], [0.5, 0.2]])
    res = burgers_residual(model, xt)
    assert jnp.allclose(res, 0.0)
