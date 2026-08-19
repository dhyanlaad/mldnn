# Müntz–Legendre Operational Neural Networks for Fractional Stochastic Differential Equations

A spectral neural framework for solving **Caputo Fractional Stochastic Differential Equations (CFSDEs)** via **Müntz–Legendre orthogonal polynomials** and **pathwise operational matrices of integration**.

---

## Mathematical Overview

We consider general Caputo fractional stochastic differential equations of order $\alpha \in (0.5, 1]$ on $t \in [0, 1]$:
$$D_t^\alpha y(t) = b(t, y(t)) + \sigma(t, y(t)) \frac{dB_t}{dt}, \quad y(0) = y_0$$

### Key Algorithmic Components
1. **Müntz–Legendre Fractional Feature Layer:**  
   The state trajectory is represented on an orthogonal Müntz–Legendre polynomial basis $\mathbf{M}^\Lambda(t) = [M_0^\Lambda(t), M_1^\Lambda(t), \dots, M_{\hat{m}}^\Lambda(t)]^\top$ with exponent sequence $\lambda_k = k\alpha$, natively capturing the singularity profile $t^\alpha$.
2. **Deterministic Operational Matrix $\mathbb{P}_\alpha(t)$:**  
   Evaluates the fractional Riemann–Liouville integral algebraically via exact Gamma function scaling:
   $$\mathcal{I}^\alpha [\mathbf{M}^\Lambda(t)] = \mathbb{P}_\alpha(t) \mathbf{M}^\Lambda(t), \quad \mathbb{P}_\alpha(t) = t^\alpha \mathbf{C} \mathbf{D}_\alpha \mathbf{C}^{-1}$$
3. **Stochastic Operational Matrix $\mathbb{S}_\alpha$ (Stochastic Fubini Contraction):**  
   Evaluates the weakly singular fractional Itô integral $\mathcal{J}^\alpha [\mathbf{M}^\Lambda(t)] \approx \mathbb{S}_\alpha \mathbf{M}^\Lambda(t)$ via an exact adjoint contraction tensor, eliminating the need for step-by-step numerical stochastic convolution during training.
4. **Instantaneous Algebraic Solves:**  
   Reduces linear and affine SDEs to a regularized Moore–Penrose pseudoinverse solve across thousands of sample paths simultaneously via vectorized PyTorch BLAS operations. Nonlinear SDEs are solved via fast Gauss–Newton / Levenberg–Marquardt collocation.

---

## Repository Structure

```text
.
├── cache/                  # Archived simulation data, raw .npz arrays, and .csv tables
│   ├── exp1_deterministic/ # Example 1: Deterministic Mittag-Leffler ODE
│   ├── exp2_ou/            # Example 1: Fractional Ornstein-Uhlenbeck Process
│   ├── exp3_gbm/           # Example 2: Fractional Geometric Brownian Motion
│   ├── exp4_cir/           # Example 3: Fractional Cox-Ingersoll-Ross Process
│   └── exp5_trig/          # Example 4: Nonlinear Trigonometric SDE
├── experiments/            # Official manuscript experiment drivers
│   ├── exp1_deterministic.py  # Mittag-Leffler deterministic benchmark
│   ├── exp2_stochastic_ou.py  # Ornstein-Uhlenbeck (theta=0.3, sigma=0.15)
│   ├── exp3_gbm.py            # Geometric Brownian Motion (mu=0.3, sigma=0.15)
│   ├── exp4_cir.py            # Cox-Ingersoll-Ross (mu=0.3, sigma=0.15)
│   ├── exp5_trig.py           # Nonlinear Trigonometric SDE (mu=0.3, sigma=0.15)
│   └── common.py              # Shared C-accelerated fEM/Milstein wrapper
├── solver/                 # Core spectral solver library
│   ├── core_mldnn.py       # Muntz basis evaluation, operational blocks, collocation
│   └── parallel.py         # Batched PyTorch solver & Stochastic Fubini tensor engine
├── benchmark/              # High-performance C kernels for reference benchmarks
│   ├── fast_fem_all.c      # Multithreaded Fractional Euler-Maruyama (ARM NEON)
│   └── fast_milstein.c     # High-resolution classical Milstein solver (N=65,536)
├── tex/                    # Manuscript LaTeX source
│   ├── numerics/           # Numerical experiments section
│   │   ├── figures/        # Publication figures
│   │   ├── numerics.tex    # Section 4 LaTeX source
│   │   └── references.bib  # BibLaTeX bibliography
│   └── old/                # Legacy manuscript draft
├── config.py               # Global experiment configurations and paths
└── requirements.txt        # Python package dependencies
```

---

## Benchmark Problems

All simulations evaluate an ensemble of $R = 5{,}000$ independent Brownian paths against fine-mesh reference benchmarks ($N = 65{,}536 = 2^{16}$ steps):

| Index | Model | SDE Formulation | Parameters | Benchmark Method |
|:---|:---|:---|:---|:---|
| **Example 1** | **Deterministic ODE** | $D_t^\alpha y(t) = -y(t)$ | $y_0 = 1.0$ | Exact Mittag–Leffler $E_\alpha(-t^\alpha)$ |
| **Example 1** | **Ornstein–Uhlenbeck** | $D_t^\alpha y(t) = \theta(\mu - y(t)) + \sigma \frac{dB_t}{dt}$ | $\theta = 0.3, \mu = 0, \sigma = 0.15$ | Exact analytical Itô integral ($\alpha = 1$) / fEM |
| **Example 2** | **Geometric Brownian Motion** | $D_t^\alpha y(t) = \mu y(t) + \sigma y(t) \frac{dB_t}{dt}$ | $\mu = 0.3, \sigma = 0.15, y_0 = 1.0$ | Exact geometric Itô formula ($\alpha = 1$) / fEM |
| **Example 3** | **Cox–Ingersoll–Ross** | $D_t^\alpha y(t) = \mu y(t) + \sigma \sqrt{y(t)} \frac{dB_t}{dt}$ | $\mu = 0.3, \sigma = 0.15, y_0 = 1.0$ | Milstein ($\alpha = 1$, order 1.0) / fEM |
| **Example 4** | **Nonlinear Trigonometric** | $D_t^\alpha y(t) = \mu \cos(y(t)) + \sigma \sin(y(t)) \frac{dB_t}{dt}$ | $\mu = 0.3, \sigma = 0.15, y_0 = 1.0$ | Milstein ($\alpha = 1$, order 1.0) / fEM |




