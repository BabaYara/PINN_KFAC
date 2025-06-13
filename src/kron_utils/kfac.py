from __future__ import annotations
"""Minimal KFAC optimizer for fully-connected layers."""

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import jax.numpy as jnp


@dataclass
class LayerState:
    """Running estimates of Kronecker factors for one layer."""

    A: jnp.ndarray
    G: jnp.ndarray


@dataclass
class KFACOptimizer:
    """Very small KFAC optimizer for lists of weight matrices."""

    lr: float = 1e-3
    damping: float = 1e-2
    decay: float = 0.95

    def init(self, params: Iterable[jnp.ndarray]) -> List[LayerState]:
        """Initialise Kronecker factor estimates for ``params``."""
        states = []
        for p in params:
            out_dim, in_dim = p.shape
            states.append(
                LayerState(jnp.eye(in_dim), jnp.eye(out_dim))
            )
        return states

    def update(
        self,
        params: Iterable[jnp.ndarray],
        grads: Iterable[jnp.ndarray],
        acts: Iterable[jnp.ndarray],
        backprops: Iterable[jnp.ndarray],
        state: List[LayerState],
    ) -> Tuple[List[jnp.ndarray], List[LayerState]]:
        """Apply one KFAC update.

        Parameters
        ----------
        params : Iterable[jnp.ndarray]
            List of weight matrices.
        grads : Iterable[jnp.ndarray]
            List of gradients w.r.t. weight matrices.
        acts : Iterable[jnp.ndarray]
            Activations for each layer with shape ``(batch, in_dim)``.
        backprops : Iterable[jnp.ndarray]
            Backpropagated gradients with shape ``(batch, out_dim)``.
        state : List[LayerState]
            Running estimates of Kronecker factors.
        """

        new_params = []
        new_state = []
        for p, g, a, b, s in zip(params, grads, acts, backprops, state):
            batch_size = a.shape[0]
            A_batch = (a.T @ a) / batch_size
            G_batch = (b.T @ b) / batch_size
            A = self.decay * s.A + (1 - self.decay) * A_batch
            G = self.decay * s.G + (1 - self.decay) * G_batch

            inv_A = jnp.linalg.inv(A + self.damping * jnp.eye(A.shape[0]))
            inv_G = jnp.linalg.inv(G + self.damping * jnp.eye(G.shape[0]))

            precond_grad = inv_G @ g @ inv_A

            new_params.append(p - self.lr * precond_grad)
            new_state.append(LayerState(A, G))

        return new_params, new_state

    def step(
        self,
        params: Iterable[jnp.ndarray],
        grads: Iterable[jnp.ndarray],
        acts: Iterable[jnp.ndarray],
        backprops: Iterable[jnp.ndarray],
        state: List[LayerState],
    ) -> Tuple[List[jnp.ndarray], List[LayerState]]:
        """Convenience wrapper for :meth:`update`."""

        return self.update(params, grads, acts, backprops, state)

