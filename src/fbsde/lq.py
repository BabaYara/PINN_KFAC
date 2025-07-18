"""Linear-Quadratic FBSDE example solver."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


def riccati_solution(
    mu: float,
    sigma: float,
    Q: float,
    R: float,
    G: float,
    T: float,
    N: int,
) -> jnp.ndarray:
    r"""Solve the scalar Riccati equation.

    The equation solved is

    .. math::

        -\dot{P}(t) = 2\mu P(t) + \sigma^2 P(t)^2 + Q - 2R P(t),

    with terminal condition ``P(T) = G``.  Euler integration with ``N`` steps
    is used to compute an approximation.

    Parameters
    ----------
    mu : float
        Drift coefficient of the forward SDE.
    sigma : float
        Diffusion coefficient of the forward SDE.
    Q : float
        State cost weight.
    R : float
        Control cost weight.
    G : float
        Terminal cost weight ``P(T)``.
    T : float
        Final time.
    N : int
        Number of time steps.

    Returns
    -------
    jnp.ndarray
        Array ``P`` of shape ``(N+1,)`` with ``P[i]`` approximating
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
    num_paths: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Simulate a linear-quadratic FBSDE with Euler discretisation.

    The forward process satisfies

    .. math::

        X_{t+\Delta t} = X_t + \mu X_t \Delta t + \sigma \Delta W_t,

    while the backward variables are

    .. math::

        Y_t = P(t) X_t, \qquad Z_t = \sigma P(t) X_t,

    where ``P`` is obtained from :func:`riccati_solution`.

    Parameters
    ----------
    mu : float
        Drift coefficient of the forward SDE.
    sigma : float
        Diffusion coefficient of the forward SDE.
    Q : float
        State cost weight.
    R : float
        Control cost weight.
    G : float
        Terminal cost weight ``P(T)``.
    x0 : float
        Initial state ``X_0``.
    T : float
        Final time.
    N : int
        Number of time steps.
    key : jax.Array
        PRNG key used to generate Brownian increments.
    num_paths : int, default=1
        Number of independent paths to simulate.

    Returns
    -------
    tuple of jnp.ndarray
        ``times`` of shape ``(N+1,)`` and ``X``, ``Y`` and ``Z`` arrays of
        shape ``(num_paths, N+1)``.

    Notes
    -----
    Setting ``num_paths`` greater than one will simulate multiple
    independent trajectories using vectorised operations.
    """

    dt = T / N
    times = jnp.linspace(0.0, T, N + 1)
    dW = jax.random.normal(key, (num_paths, N)) * jnp.sqrt(dt)

    def forward_step(x, dw):
        return x + mu * x * dt + sigma * dw

    def simulate_path(dw_path):
        X_path = jnp.zeros(N + 1)
        X_path = X_path.at[0].set(x0)

        def f_scan(carry, dw):
            x = forward_step(carry, dw)
            return x, x

        _, xs = jax.lax.scan(f_scan, x0, dw_path)
        X_path = X_path.at[1:].set(xs)
        return X_path

    X = jax.vmap(simulate_path)(dW)

    P = riccati_solution(mu, sigma, Q, R, G, T, N)

    Y = P * X
    Z = sigma * P * X
    return times, X, Y, Z
