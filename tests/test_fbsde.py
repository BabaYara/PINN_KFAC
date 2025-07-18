import jax
import jax.numpy as jnp
from fbsde import solve_lq_fbsde
from fbsde.lq import riccati_solution


def test_riccati_solution():
    mu = 0.1
    sigma = 0.2
    Q = 1.0
    R = 0.5
    G = 1.0
    T = 1.0
    N = 5

    P = riccati_solution(mu, sigma, Q, R, G, T, N)

    dt = T / N
    P_manual = jnp.zeros(N + 1)
    P_manual = P_manual.at[-1].set(G)
    for i in range(N, 0, -1):
        p_next = P_manual[i]
        dp = 2 * mu * p_next + sigma ** 2 * p_next ** 2 + Q - 2 * R * p_next
        P_manual = P_manual.at[i - 1].set(p_next - dt * dp)

    assert P.shape == (N + 1,)
    assert P[-1] == G
    assert jnp.allclose(P, P_manual)


def test_lq_fbsde_shapes():
    key = jax.random.PRNGKey(0)
    times, X, Y, Z = solve_lq_fbsde(
        mu=0.1,
        sigma=0.2,
        Q=1.0,
        R=0.5,
        G=1.0,
        x0=1.0,
        T=1.0,
        N=5,
        key=key,
        num_paths=3,
    )
    assert times.shape == (6,)
    assert X.shape == (3, 6)
    assert Y.shape == (3, 6)
    assert Z.shape == (3, 6)
    # Riccati consistency: Y should equal P*X
    dt = 1.0 / 5
    P = jnp.zeros(6)
    P = P.at[-1].set(1.0)
    for i in range(5, 0, -1):
        p_next = P[i]
        dp = 2 * 0.1 * p_next + 0.2 ** 2 * p_next ** 2 + 1.0 - 2 * 0.5 * p_next
        P = P.at[i - 1].set(p_next - dt * dp)
    assert jnp.allclose(Y, P * X, atol=1e-5)


def test_lq_fbsde_solution_consistency():
    key = jax.random.PRNGKey(1)
    params = dict(mu=0.1, sigma=0.2, Q=1.0, R=0.5, G=1.0, x0=1.0, T=1.0, N=50)
    times, X, Y, Z = solve_lq_fbsde(key=key, num_paths=2, **params)

    P = riccati_solution(
        mu=params["mu"],
        sigma=params["sigma"],
        Q=params["Q"],
        R=params["R"],
        G=params["G"],
        T=params["T"],
        N=params["N"],
    )

    assert times.shape == (params["N"] + 1,)
    assert X.shape == (2, params["N"] + 1)
    assert jnp.allclose(Y, P * X, atol=1e-5)
    assert jnp.allclose(Z, params["sigma"] * P * X, atol=1e-5)

