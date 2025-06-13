"""Simple training loop using the KFAC optimizer for PINNs."""

from typing import Callable, Sequence, Tuple
import jax
import jax.numpy as jnp

from networks import mlp_apply, mlp_forward_activations, mlp_backprops
from kron_utils import KFACOptimizer
from .loss import pinn_loss

Params = Sequence[Tuple[jnp.ndarray, jnp.ndarray]]


def train_pinn(
    params: Params,
    X_res: jnp.ndarray,
    X_bc: jnp.ndarray,
    bc_values: jnp.ndarray,
    forcing_fn: Callable[[jnp.ndarray], jnp.ndarray],
    optimizer: KFACOptimizer,
    steps: int = 100,
) -> Params:
    """Train ``params`` to solve Poisson-like equations using KFAC.

    The training is intentionally minimal and meant for demonstration.
    """

    opt_state = optimizer.init([w.T for w, _ in params])

    def model(ps, x):
        return mlp_apply(ps, x)

    for _ in range(steps):

        def loss_fn(ws):
            ps = [(ws[i], params[i][1]) for i in range(len(ws))]
            return pinn_loss(lambda z: model(ps, z), X_res, X_bc, bc_values, forcing_fn)

        w_list = [w.T for w, _ in params]
        loss, grads = jax.value_and_grad(loss_fn)([w for w, _ in params])
        grads = [g.T for g in grads]

        # Very rough approximations for Kronecker factors
        preds, acts, preacts = mlp_forward_activations(params, X_res)
        loss_grad = jnp.tile(loss, (X_res.shape[0], 1))
        backprops = mlp_backprops(params, preacts, loss_grad)

        new_ws, opt_state = optimizer.step(w_list, grads, acts, backprops, opt_state)
        params = [(new_ws[i].T, params[i][1]) for i in range(len(params))]

    return params
