"""Linear-Quadratic FBSDE example solver."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


def riccati_solution(
    mu: float, sigma: float, Q: float, R: float, G: float, T: float, N: int
) -> jnp.ndarray:
    """Solve the Riccati equation backward in time.

    Returns an array ``P`` of shape ``(N+1,)`` with ``P[i]`` approximating
    ``P(t_i)`` where ``t_i = i * T / N``.
    """

    dt = T / N
    P = jnp.zeros(N + 1)
    P = P.at[-1].set(G)
    def body(carry, _):
        p_next = carry
        dp = 2 * mu * p_next + sigma ** 2 * p_next ** 2 + Q - 2 * R * p_next
        p = p_next - dt * dp
        return p, p

    _, vals = jax.lax.scan(body, G, jnp.arange(N))
    P = P.at[:-1].set(vals[::-1])
    return P


def solve_lq_fbsde(
    mu: float,
    sigma: float,
    Q: float,
    R: float,
    G: float,
    x0: float,
    T: float,
    N: int,
    key: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Simulate a linear-quadratic FBSDE with Euler discretization."""

    dt = T / N
    times = jnp.linspace(0.0, T, N + 1)
    dW = jax.random.normal(key, (N,)) * jnp.sqrt(dt)

    def forward_step(x, dw):
        return x + mu * x * dt + sigma * dw

    X = jnp.zeros(N + 1)
    X = X.at[0].set(x0)
    def f_scan(carry, dw):
        x = forward_step(carry, dw)
        return x, x

    _, xs = jax.lax.scan(f_scan, x0, dW)
    X = X.at[1:].set(xs)

    P = riccati_solution(mu, sigma, Q, R, G, T, N)

    Y = P * X
    Z = sigma * P * X
    return times, X, Y, Z
