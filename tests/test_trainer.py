import jax
import jax.numpy as jnp
from networks import init_mlp, mlp_apply
from kron_utils import KFACOptimizer
from pinns import pinn_loss, train_pinn


def test_train_pinn_reduces_loss():
    key = jax.random.PRNGKey(0)
    params = init_mlp([1, 4, 1], key)

    X_res = jnp.linspace(0.0, jnp.pi, 5).reshape(-1, 1)
    X_bc = jnp.array([[0.0], [jnp.pi]])
    bc_vals = jnp.array([0.0, 0.0])
    forcing = lambda x: -jnp.sin(x)

    opt = KFACOptimizer(lr=0.05)

    def model(ps, x):
        return mlp_apply(ps, x)

    loss_fn = lambda ps: pinn_loss(
        lambda z: model(ps, z), X_res, X_bc, bc_vals, forcing
    )

    loss_before = loss_fn(params)
    trained = train_pinn(params, X_res, X_bc, bc_vals, forcing, opt, steps=2)
    loss_after = loss_fn(trained)

    assert loss_after < loss_before
