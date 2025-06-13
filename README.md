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

This will make the `taylor_mode`, `kron_utils`, and `pinns` modules
available.

## Example

Several notebooks in the `notebooks/` folder demonstrate the library.
Run `08_KFAC_implementation.ipynb` to see a very small KFAC step.

## Project Plan

See `Plan.tex` for the complete roadmap and repository layout.
