import jax.numpy as jnp
from pinns import gradient, laplacian


def test_gradient_and_laplacian():
    f = lambda x: jnp.sin(x).sum()
    x0 = jnp.array([0.0])
    grad = gradient(f, x0)
    lap = laplacian(f, x0)
    assert jnp.allclose(grad, 1.0)
    assert jnp.allclose(lap, 0.0)


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
