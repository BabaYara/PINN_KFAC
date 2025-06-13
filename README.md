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
includes helpers like `poisson_residual` for assembling common PDE losses.

## Example

Several notebooks in the `notebooks/` folder demonstrate the library.
`02_gradient_operator.ipynb` demonstrates computing gradients using Taylor-mode utilities.
`04_PINN_loss_demo.ipynb` shows building a simple Poisson PINN using `pinn_loss`.
`08_KFAC_implementation.ipynb` shows a short linear-regression example using the `KFACOptimizer`.
`10_pinn_with_kfac.ipynb` demonstrates training a tiny PINN using the KFAC optimizer and the simple MLP utilities from `networks`.
`11_kfac_training.ipynb` shows the `train_pinn` helper in action.
`12_poisson_residual_demo.ipynb` demonstrates computing Poisson residuals with
the new convenience function.
