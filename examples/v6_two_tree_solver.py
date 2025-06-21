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
Estimated wall-time on CPU \u2248 3 min (N_FINE=150, batch=256)
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
\u03b3A, \u03c8A, \u03c1A = 7.0, .9, .02       # agent A EZW
\u03b3B, \u03c8B, \u03c1B =10.0,1.3, .02       # agent B EZW
\u03b8A = (1-\u03b3A)/(1-1/\u03c8A)            # homogeneity coeffs
\u03b8B = (1-\u03b3B)/(1-1/\u03c8B)

\u03ba_u, \u03bc_u, \u03c3_u = 2.5, 0.0, 0.35  # OU drift in *logit* of dividend share
def logistic(u): return jnp.exp(u)/(1+jnp.exp(u))

T, N = 20., 150                 # time-horizon & grid
BATCH = 256
eta = 0.0                       # log Pareto weight \u03bbA/\u03bbB   \u2190 will stay fixed here

# helper to cache algebraic functions
def cache_eta(eta):
    \u03bbA, \u03bbB = jnp.exp(.5*eta), jnp.exp(-.5*eta)
    \u03bbA\u03c8, \u03bbB\u03c8 = \u03bbA**\u03c8A, \u03bbB**\u03c8B
    ratio = (\u03bbA/\u03bbB)**(1/\u03b8B)
    def share_JB(JA):
        JB = ratio * JA**(-\u03b8A/\u03b8B)
        num = \u03bbA\u03c8 * JA**(\u03c8A*\u03b8A)
        den = num + \u03bbB\u03c8 * JB**(\u03c8B*\u03b8B)
        return num/den, JB
    \u03b2_lin = \u03c1A + (\u03b3A-1)*\u03c3_u**2/(2*\u03c8A)  # linearised discount exponent
    return \u03bbA, \u03bbB, share_JB, \u03b2_lin
\u03bbA, \u03bbB, share_JB, \u03b2_lin = cache_eta(eta)

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
    \u03bbA, _, share_JB, _ = cache_eta(eta)
    @jax.jit
    def f(u,y,z):
        x = logistic(u)
        shareA,_ = share_JB(y)
        M = \u03bbA*shareA**(-\u03b3A)*y**\u03b8A                 # marginal utility index
        def M_u(uu):
            s,_ = share_JB(y)                     # JA treated const wrt u
            return \u03bbA*s**(-\u03b3A)*y**\u03b8A
        dM   = jax.grad(M_u)(u); d2M = jax.grad(jax.grad(M_u))(u)
        \u03bcu   = \u03ba_u*(\u03bc_u-u);  LuM = \u03bcu*dM + .5*\u03c3_u**2*d2M
        r    = -LuM/M
        k    = -\u03c3_u*dM/M
        \u03c3x   = \u03c3_u * x*(1-x)
        \u03c3lnC = \u03c3x * (1-shareA)
        return \u03b8A/(1-\u03b3A)*(\u03c1A-\u03b3A*r-.5*(\u03b8A-1)*(z/\u03c3x)**2)*y \
             + \u03b8A*y*(z/\u03c3x)*(k-\u03c3lnC)
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
        \u03bcu = \u03ba_u*(\u03bc_u-u); \u03bcu = \u03bcu/(1+dt*jnp.abs(\u03bcu))
        u1 = u + \u03bcu*dt + \u03c3_u*dwi
        y1 = y_hat - f(u,y_hat,z_hat)*dt + z_hat*dwi
        yl1= yl - \u03b2_lin*yl*dt
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
