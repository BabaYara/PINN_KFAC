import jax
import jax.numpy as jnp
from networks import (
    init_mlp,
    mlp_apply,
    mlp_forward_activations,
    mlp_backprops,
)


def test_init_and_apply_mlp():
    key = jax.random.PRNGKey(0)
    params = init_mlp([1, 4, 1], key)
    x = jnp.array([[0.0]])
    y = mlp_apply(params, x)
    assert y.shape == (1, 1)


def test_forward_and_backprops_match_gradients():
    key = jax.random.PRNGKey(0)
    params = init_mlp([1, 2, 1], key)
    X = jnp.ones((3, 1))
    y_true = jnp.zeros((3, 1))

    def loss_fn(ps):
        preds = mlp_apply(ps, X)
        return jnp.mean((preds - y_true) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    preds, acts, preacts = mlp_forward_activations(params, X)
    loss_grad = (preds - y_true) * (2 / X.shape[0])
    backprops = mlp_backprops(params, preacts, loss_grad)
    weight_grads = [acts[i].T @ backprops[i] for i in range(len(params))]

    for g_est, g_true in zip(weight_grads, [g[0] for g in grads]):
        assert jnp.allclose(g_est, g_true, atol=1e-5)
