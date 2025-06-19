"""Computes kinetic-energy g-matrix for formaldehyde molecule (H2CO)"""

import jax
import jax.numpy as jnp
from .jet_prim import inv

jax.config.update("jax_enable_x64", True)

EPS = jnp.array(
    [
        [[int((i - j) * (j - k) * (k - i) * 0.5) for k in range(3)] for j in range(3)]
        for i in range(3)
    ],
    dtype=jnp.float64,
)

# masses of C, O, H, H atoms
MASSES = jnp.array([12.0, 15.990526, 1.00782505, 1.00782505])


def internal_to_cartesian(internal_coords):
    """Given internal valence coordinates of H2CO molecule,
    returns Cartesian coordinates of its atoms in the order: C, O, H, H
    """
    rCO, rCH1, rCH2, aOCH1, aOCH2, tau = internal_coords
    xyz = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, rCO],
            [
                rCH1 * jnp.sin(aOCH1) * jnp.cos(tau * 0.5),
                -rCH1 * jnp.sin(aOCH1) * jnp.sin(tau * 0.5),
                rCH1 * jnp.cos(aOCH1),
            ],
            [
                rCH2 * jnp.sin(aOCH2) * jnp.cos(tau * 0.5),
                rCH2 * jnp.sin(aOCH2) * jnp.sin(tau * 0.5),
                rCH2 * jnp.cos(aOCH2),
            ],
        ]
    )
    com = MASSES @ xyz / jnp.sum(MASSES)
    return xyz - com[None, :]


@jax.jit
def gmat(q):
    # xyz_g = jax.jacfwd(internal_to_cartesian)(jnp.asarray(q))
    xyz_g = jax.jacrev(internal_to_cartesian)(jnp.asarray(q))
    tvib = xyz_g
    xyz = internal_to_cartesian(jnp.asarray(q))
    trot = jnp.transpose(EPS @ xyz.T, (2, 0, 1))
    ttra = jnp.array([jnp.eye(3, dtype=jnp.float64) for _ in range(len(xyz))])
    tvec = jnp.concatenate((tvib, trot, ttra), axis=2)
    masses_sq = jnp.sqrt(jnp.asarray(MASSES))
    tvec = tvec * masses_sq[:, None, None]
    tvec = jnp.reshape(tvec, (len(xyz) * 3, len(q) + 6))
    return tvec.T @ tvec


@jax.jit
def Gmat(q):
    """Computes kinetic energy G-matrix as function
    of molecular internal coordinates `q`
    """
    return inv(gmat(q))