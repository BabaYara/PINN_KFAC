# Taylor Mode PINNs

This repository provides a small Python package offering
Taylor-mode automatic differentiation utilities and a simple
Kronecker-Factored Approximate Curvature (KFAC) optimizer for
Physics-informed neural networks (PINNs).

## Installation

The package follows a standard Python layout.  Install it in editable mode
with

```bash
pip install -e .
```

This will make the `taylor_mode`, `kron_utils`, `networks`, and `pinns` modules
available. The `pinns` module now also exposes a simple `train_pinn` routine
for quick experiments with KFAC training.  The `pinns.operators` submodule
includes helpers like `poisson_residual` for assembling common PDE losses,
as well as convenience functions such as `divergence`.

The operators module now also provides `heat_residual` for the 1D heat
equation, and `burgers_residual` for Burgers' equation, making it easy
to experiment with time-dependent problems.

## Example

Several notebooks in the `notebooks/` folder demonstrate the library.
`02_gradient_operator.ipynb` demonstrates computing gradients using Taylor-mode utilities.
`04_PINN_loss_demo.ipynb` shows building a simple Poisson PINN using `pinn_loss`.
`08_KFAC_implementation.ipynb` shows a short linear-regression example using the `KFACOptimizer`.
`10_pinn_with_kfac.ipynb` demonstrates training a tiny PINN using the KFAC optimizer and the simple MLP utilities from `networks`.
`11_kfac_training.ipynb` shows the `train_pinn` helper in action.
`12_poisson_residual_demo.ipynb` demonstrates computing Poisson residuals with
the new convenience function.
`13_heat_residual_demo.ipynb` shows how to evaluate the heat equation residual
using `heat_residual`.
`14_burgers_residual_demo.ipynb` illustrates evaluating the Burgers equation
residual with `burgers_residual`.
`examples/v6_two_tree_solver.py` provides an end-to-end solver in one file.

### 13 End-to-end example (two agents, two trees, EZW preferences)

```python
"""
examples/v6_two_tree_solver.py
---------------------------------------------------------
A *single-file replica* of the full v6 solver we prototyped in
the chat.  This script does **not** import the still-to-be-built
`deepbsde/` package; instead it lays out every step explicitly,
with line-by-line comments so Codex agents can translate each
block into its future modular home.

Run:
  python examples/v6_two_tree_solver.py --device cpu
Estimated wall-time on CPU ≈ 3 min (N_FINE=150, batch=256)
"""

# ︙ 1. Imports -------------------------------------------------------------
import jax, jax.numpy as jnp
import equinox as eqx, optax, functools, argparse, time

# ︙ 2. Command-line flags --------------------------------------------------
P = argparse.ArgumentParser()
P.add_argument("--device", default="cpu")
P.add_argument("--depth", type=int, default=8)
P.add_argument("--width", type=int, default=128)
P.add_argument("--steps", type=int, default=5_000)
ARGS = P.parse_args()
jax.config.update("jax_platform_name", ARGS.device)

# ︙ 3. Economic primitives -------------------------------------------------
γA, ψA, ρA = 7.0, .9, .02       # agent A EZW
γB, ψB, ρB =10.0,1.3, .02       # agent B EZW
θA = (1-γA)/(1-1/ψA)            # homogeneity coeffs
θB = (1-γB)/(1-1/ψB)

κ_u, μ_u, σ_u = 2.5, 0.0, 0.35  # OU drift in *logit* of dividend share
def logistic(u): return jnp.exp(u)/(1+jnp.exp(u))

T, N = 20., 150                 # time-horizon & grid
BATCH = 256
eta = 0.0                       # log Pareto weight λA/λB   ← will stay fixed here

# helper to cache algebraic functions
def cache_eta(eta):
    λA, λB = jnp.exp(.5*eta), jnp.exp(-.5*eta)
    λAψ, λBψ = λA**ψA, λB**ψB
    ratio = (λA/λB)**(1/θB)
    def share_JB(JA):
        JB = ratio * JA**(-θA/θB)
        num = λAψ * JA**(ψA*θA)
        den = num + λBψ * JB**(ψB*θB)
        return num/den, JB
    β_lin = ρA + (γA-1)*σ_u**2/(2*ψA)  # linearised discount exponent
    return λA, λB, share_JB, β_lin
λA, λB, share_JB, β_lin = cache_eta(eta)

# ︙ 4. Residual network (depth & width from CLI) ---------------------------
def resblock(width, key):
    k1,k2 = jax.random.split(key)
    return (eqx.nn.Linear(width,width,key=k1),
            eqx.nn.Linear(width,width,key=k2))
class ResNet(eqx.Module):
    inproj : eqx.nn.Linear
    blocks : tuple
    headY  : eqx.nn.Linear
    headZ  : eqx.nn.Linear
    def __call__(self,t,u):
        h = jax.nn.silu(self.inproj(jnp.stack([t,u],-1)))
        for w1,w2 in self.blocks:
            h = h + jax.nn.silu(w2(jax.nn.silu(w1(h))))
        y = self.headY(h)[:,0]
        z = self.headZ(h)[:,0]
        return y,z

key = jax.random.PRNGKey(0)
k0,*ks = jax.random.split(key, ARGS.depth+3)
blocks = tuple(resblock(ARGS.width,k) for k in ks[:ARGS.depth])
net = ResNet(eqx.nn.Linear(2,ARGS.width,key=k0), blocks,
             eqx.nn.Linear(ARGS.width,1,key=ks[-2]),
             eqx.nn.Linear(ARGS.width,1,key=ks[-1]))

# ︙ 5. Generator f(u,y,z) (social-planner HJB ⇔ BSDE driver) --------------
@functools.partial(jax.jit, static_argnums=0)
def make_f(eta):
    λA, _, share_JB, _ = cache_eta(eta)
    @jax.jit
    def f(u,y,z):
        x = logistic(u)
        shareA,_ = share_JB(y)
        M = λA*shareA**(-γA)*y**θA                 # marginal utility index
        def M_u(uu):
            s,_ = share_JB(y)                     # JA treated const wrt u
            return λA*s**(-γA)*y**θA
        dM   = jax.grad(M_u)(u); d2M = jax.grad(jax.grad(M_u))(u)
        μu   = κ_u*(μ_u-u);  LuM = μu*dM + .5*σ_u**2*d2M
        r    = -LuM/M
        k    = -σ_u*dM/M
        σx   = σ_u * x*(1-x)
        σlnC = σx * (1-shareA)
        return θA/(1-γA)*(ρA-γA*r-.5*(θA-1)*(z/σx)**2)*y \
             + θA*y*(z/σx)*(k-σlnC)
    return f
f = make_f(eta)

# ︙ 6. Brownian sampler (Sobol + antithetic) ------------------------------
def brownian_batch(N, key):
    sob = jax.random.sobol_sample(N, BATCH//2, dtype=jnp.float32)
    sob = jnp.clip(sob,1e-6,1-1e-6)
    g   = jax.scipy.stats.norm.ppf(sob)
    g   = jnp.concatenate([g,-g],0)
    return jnp.sqrt(T/N)*g                      # scale by √Δt

# ︙ 7. Loss (Euler + full control-variate) -------------------------------
def bsde_loss(net, key, step):
    dW = brownian_batch(N, key)
    dt = T/N
    def body(carry, inp):
        u,y,yl = carry; i, dwi = inp
        t = i*dt*jnp.ones_like(u)
        y_hat, z_hat = net(t,u)
        # control-variate pair
        ycv = y_hat - yl
        # tamed Euler
        μu = κ_u*(μ_u-u); μu = μu/(1+dt*jnp.abs(μu))
        u1 = u + μu*dt + σ_u*dwi
        y1 = y_hat - f(u,y_hat,z_hat)*dt + z_hat*dwi
        yl1= yl - β_lin*yl*dt
        return (u1,y1,yl1), ycv[-1]             # pen = last ycv
    (uT,yT,ylT), pen = jax.lax.scan(
        body, (jnp.zeros((BATCH,)), net(0.,0.)[0], jnp.ones((BATCH,))), 
        (jnp.arange(N), dW.T))
    return jnp.mean((yT-ylT)**2) + .01*jnp.mean(pen**2)

optim = optax.adam(learning_rate=3e-4)
opt_state = optim.init(net)

# ︙ 8. Training loop ------------------------------------------------------
for step in range(ARGS.steps):
    key, sub = jax.random.split(key)
    loss, grads = eqx.filter_value_and_grad(bsde_loss)(net, sub, step)
    updates, opt_state = optim.update(grads, opt_state, net)
    net = eqx.apply_updates(net, updates)
    if step % 500 == 0:
        print(f"step {step:5d}  loss {float(loss):.3e}")

# ︙ 9. Quick PDE residual check ------------------------------------------
grid = 2.5*jnp.cos(jnp.linspace(0,jnp.pi,400))
def residual(u):
    y,_ = net(0.,u)
    z   = jax.grad(lambda uu: net(0.,uu)[0])(u)
    return f(u,y,z)
print("max PDE residual", float(jnp.max(jnp.abs(jax.vmap(residual)(grid)))))
```
