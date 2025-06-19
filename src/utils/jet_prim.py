import jax
import jax.numpy as jnp
from jax import lax
from jax.core import ShapedArray
from jax.experimental import jet
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

jax.config.update("jax_enable_x64", True)


def fact(n):
    return lax.exp(lax.lgamma(n + 1.0))


# Define JAX.jet rules for missing primitive functions


##################
# matrix inversion
##################

inv_p = Primitive("_inv")


def inv(a):
    return inv_p.bind(a)


def inv_impl(a):
    return jnp.linalg.inv(a)


inv_p.def_impl(inv_impl)


@jax.jit
def inv_jvp(primals, tangents):
    (a,) = primals
    (da,) = tangents
    a_inv = inv(a)
    da_inv = -a_inv @ da @ a_inv
    return a_inv, da_inv


ad.primitive_jvps[inv_p] = inv_jvp


def inv_abstact_eval(a):
    shape = a.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Input to 'inv' must be a square matrix")
    N = shape[0]
    dtype = a.dtype
    return ShapedArray((N, N), dtype)


inv_p.def_abstract_eval(inv_abstact_eval)


def inv_lowering(ctx, a):
    return mlir.lower_fun(jnp.linalg.inv, multiple_results=False)(ctx, a)


mlir.register_lowering(inv_p, inv_lowering)


def inv_batch_rule(args, dims):
    (mat,) = args
    (dim,) = dims
    if dim is None:
        return inv_p(mat), None
    m = jax.vmap(jnp.linalg.inv)(mat)
    return m, dim


batching.primitive_batchers[inv_p] = inv_batch_rule


@jax.jit
def _inverse_taylor_rule(primals_in, series_in):
    (x,) = primals_in
    (x_terms,) = series_in
    u = [x] + x_terms
    v = [None] * len(u)

    v[0] = inv(x)

    def scale(k, j):
        return 1.0 / (fact(k - j) * fact(j))

    for k in range(1, len(v)):
        v[k] = (
            -fact(k)
            * v[0]  # T?
            @ sum(scale(k, j) * u[j] @ v[k - j] for j in range(1, k + 1))
        )
    primal_out, *series_out = v
    return primal_out, series_out


jet.jet_rules[inv_p] = _inverse_taylor_rule