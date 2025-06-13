import jax
import jax.numpy as jnp
from kron_utils import KFACOptimizer


def test_kfac_linear_regression():
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (5, 1))
    true_w = jnp.array([[2.0]])
    y = X @ true_w

    params = [jax.random.normal(key, (1, 1))]
    opt = KFACOptimizer(lr=0.1, damping=1e-2)
    state = opt.init(params)

    def net(ps, x):
        return x @ ps[0]

    def loss_fn(ps):
        preds = net(ps, X)
        return jnp.mean((preds - y) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    preds = net(params, X)
    acts = [X]
    backprops = [preds - y]

    new_params, _ = opt.step(params, grads, acts, backprops, state)
    assert loss_fn(new_params) < loss
