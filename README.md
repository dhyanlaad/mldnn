# MLDNN – Multi‑Layer Deep Neural Networks for Stochastic Differential Equations

## Overview

This repository implements **MLDNN** solvers for fractional stochastic differential equations (SDEs) using the Muntz‑Legendre basis. Two families of solvers are provided:

1. **Affine (linear) solvers** – classic least‑squares and Gauss‑Newton methods for problems where the drift and diffusion are affine in the state variable.
2. **Deep (non‑linear) solvers** – an *Extreme Learning Machine* (ELM) style augmentation that adds a single random hidden layer (tanh activation) to the basis. The hidden weights are drawn once (deterministic seed) and remain fixed, enabling fast linear solves via least‑squares or Gauss‑Newton.

The core components are in `solver/core_mldnn.py`:
* Basis evaluation (`basis_eval`), operational matrix construction, and the `Blocks` helper.
* `solve_affine`, `solve_gauss_newton` for affine problems.
* `solve_affine_deep`, `solve_gauss_newton_deep` for the deep‑ELM variant.
* `evaluate_solution` automatically reconstructs the hidden features when given a deep coefficient vector.

## Experiments

Four experiments showcase the methods:

| Experiment | Goal | Key Parameters | Output |
|------------|------|----------------|--------|
| **Exp 1** – Deterministic drift | Verify deterministic case against analytical solution. | Muntz‑Legendre order `m` up to 24. | Expectation PDF, Pareto table. |
| **Exp 2** – Linear diffusion | Compare against a reference Monte‑Carlo solution. | Hidden layer size 64, `Nq=256`. | Expectation, Q‑Q plots, variance PDF. |
| **Exp 3** – Non‑linear drift (`-y³`) | Demonstrate deep‑ELM solver on a non‑linear drift. | Reduced hidden size (16), `Nq=128`, maxit = 30 for speed. | Expectation PDF, Pareto, Q‑Q (t = 0.3/0.6), variance PDF. |
| **Exp 4** – Geometric Brownian Motion | Test against the analytic GBM solution. | Same deep‑ELM configuration as Exp 3. | Expectation PDF, Pareto, Q‑Q, variance PDF. |

All figures are saved in the `exports/` directory.

## Usage

```bash
# Activate the virtual environment (already created)
source .venv/bin/activate

# Run a specific experiment
PYTHONPATH=. ./.venv/bin/python3 experiments/exp3_nonlinear_drift.py
```

The `run_all.sh` script sequentially runs all experiments.

## Clean‑up

Temporary debugging scripts in `scratch/` have been removed.

---

*Author: Arthur*
