# Müntz–Legendre Operational Neural Networks for Fractional SDEs

A spectral, physics-informed functional-link framework for Caputo fractional stochastic differential equations (CFSDEs), using Müntz–Legendre bases and deterministic/pathwise operational matrices.

The implementation targets the Apple M3 Ultra Mac Studio used for the manuscript experiments: macOS on ARM64, 20 performance cores, and float64 linear algebra through Apple Accelerate. The stochastic reference kernels use pthreads and explicit ARM NEON SIMD. PyTorch is deliberately run on CPU because these workloads use small batched float64 systems; MPS is neither required nor used.

## Method

For

$$
D_t^\alpha y(t)=b(t,y(t))+\sigma(t,y(t))\frac{dB_t}{dt},
\qquad y(0)=y_0,
$$

the state is represented as

$$
y_n(t)=\mathbf c^\top\mathbf M^\Lambda(t),
\qquad \lambda_k=k\alpha.
$$

The main numerical components are:

1. A Müntz–Legendre feature basis adapted to fractional powers $t^{k\alpha}$.
2. A deterministic fractional integration matrix
   $$\mathbb P_\alpha(t)=t^\alpha\mathbf C\mathbf D_\alpha\mathbf C^{-1}.$$
3. A stochastic Fubini contraction tensor that constructs every pathwise stochastic operational matrix with one batched matrix multiplication.
4. The filtered frozen-coefficient multipass (FCMP) solver (`solver/fcmp.py`). Each pass fits the diffusion coefficients by a weighted Gauss-rule projection of the previous state, applies the stochastic block with the exact finite-dimensional Malliavin (Skorokhod) trace differentiated through all earlier passes, and solves the deterministic drift equation. Affine drift uses one linear solve; nonlinear drift uses Picard iterations polished by Newton.
5. An exact initial-value pin: the output is the $L^2$ projection onto $\{v\in V_n : v(0)=y_0\}$.

The experiments use the unfiltered scheme: one pass for additive noise, three for GBM, and two for the nonlinear models. Its convergence theorem covers globally Lipschitz coefficients with bounded derivatives, which includes every experiment below.

## Verified Mac configuration

The final verification was performed on:

- Apple M3 Ultra Mac Studio, ARM64
- 28 CPU cores: 20 performance and 8 efficiency cores
- 256 GB unified memory
- macOS 15.7.4
- Python 3.12.4
- NumPy 2.5.2 linked to Apple Accelerate with NEON/ASIMD
- SciPy 1.18.0
- PyTorch 2.13.0, CPU float64 path, 20 compute threads

The C reference kernels cap themselves at 20 threads so work stays on the performance cores. The Python batched solvers use PyTorch/Accelerate BLAS and process operator-trace paths in memory-bounded chunks.

## Installation

Python 3.12 and the Xcode command-line tools are recommended.

```bash
xcode-select --install                 # only if clang/make are unavailable
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
make -C benchmark check
```

The native build uses `clang -O3 -mcpu=native`, fast fused floating-point contraction, pthreads, and ARM NEON. Generated `.dylib` files are intentionally ignored by Git and must be built after cloning.

## Run and verify

Run the unit/regression suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Compile the numerical manuscript independently:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -cd tex/numerics/numerics.tex
```

Regenerate every manuscript experiment in order:

```bash
./run_all.sh
```

Set `PYTHON_BIN` to use a different compatible interpreter. The runner builds the native libraries first and places Matplotlib caches under `/tmp` by default. Raw arrays, CSV files, and auxiliary plots are written under `exports/`; Brownian increments are cached under `cache/`. Both locations avoid committing large generated data.

Individual experiments can also be run with:

```bash
.venv/bin/python -m experiments.exp1_deterministic
.venv/bin/python -m experiments.exp2_stochastic_ou
.venv/bin/python -m experiments.exp3_gbm
.venv/bin/python -m experiments.exp5_trig
```

## Manuscript experiment protocol

The published tables use seed 42, $R=500$ paired Brownian paths, $N^*=65{,}536$ increments and degrees $\hat m\in\{2,4,8,16,24,32\}$ for stochastic convergence sweeps.

| Example | Equation | Parameters | Reference |
|:--|:--|:--|:--|
| Deterministic relaxation | $D_t^\alpha y=-y$ | $y_0=1$ | Mittag–Leffler solution |
| Ornstein–Uhlenbeck | $D_t^\alpha y=\theta(\mu-y)+\sigma\dot B$ | $\theta=0.3,\mu=0,\sigma=0.15$ | Exact Itô solution at $\alpha=1$; fEM below 1 |
| Geometric Brownian motion | $D_t^\alpha y=\mu y+\sigma y\dot B$ | $\mu=0.3,\sigma=0.15,y_0=1$ | Exact Itô solution at $\alpha=1$; fEM below 1 |
| Trigonometric SDE | $D_t^\alpha y=\mu\cos y+\sigma\sin y\dot B$ | $\mu=0.3,\sigma=0.15,y_0=1$ | Milstein at $\alpha=1$; fEM below 1 |

At $\alpha=1$ and $\hat m=32$, the regenerated supremum trajectory MSE values are:

| Model | $\mathcal E_\infty$ |
|:--|--:|
| OU | $1.5991\times10^{-4}$ |
| GBM | $2.5477\times10^{-4}$ |
| Trigonometric | $1.2657\times10^{-4}$ |

The complete definitions, tables, and Q–Q figures are in [`tex/numerics/numerics.tex`](tex/numerics/numerics.tex).

## Repository layout

```text
benchmark/       ARM64 NEON C reference kernels and native Makefile
cache/           ignored Brownian/reference caches
experiments/     four manuscript experiment drivers and shared fEM bindings
exports/         ignored regenerated CSV, NPZ, JSON, and auxiliary figures
solver/          basis, operational matrices, and the FCMP solver
tests/           FCMP and fEM reference regression tests
tex/main/        main manuscript source
tex/numerics/    regenerated numerical section and publication figures
config.py        paths, precision, platform, and experiment defaults
run_all.sh       native build plus complete experiment regeneration
```

## Numerical notes

- The Malliavin trace is exact for the discretized Fubini matrix and is differentiated through every earlier pass; its cost grows quickly with the degree and pass count.
- Coefficient functions passed to the FCMP solver must be torch callables, because the trace differentiates through them.
- All primary computation uses float64.
- The fast fEM kernel is $O(N^2)$ in the fine-grid length per path; the native NEON implementation and cached paired Brownian paths are essential for manuscript-scale reruns.
