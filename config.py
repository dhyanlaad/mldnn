"""
config.py
=========
Central configuration module for the Muntz-Legendre Spectral / Operational Matrix solver (ML-SDE),
Fractional Euler-Maruyama (fEM) baseline solver, and numerical experiments.

Pure spectral representation with zero hidden layers:
    N(t) = c^T M^Lambda(t) = sum_{k=0}^{mhat} c_k M_k^Lambda(t)

Optimized for Apple Silicon / macOS (multi-core ARM64 parallelism & float64 SIMD).
"""

from __future__ import annotations
import os
import platform
import torch
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================================
# 1. Platform & Hardware Detection (Apple Silicon / Multi-core CPU)
# ============================================================================

SYSTEM_OS: str = platform.system()
IS_MACOS: bool = (SYSTEM_OS == "Darwin")
IS_ARM64: bool = (platform.machine() in ("arm64", "aarch64"))
CPU_COUNT: int = os.cpu_count() or 4


# ============================================================================
# 2. Project Paths & Shared Libraries
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SOLVER_DIR = PROJECT_ROOT / "solver"
EXPORTS_DIR = PROJECT_ROOT / "exports"
RESULTS_DIR = EXPORTS_DIR / "results"
FIGURES_DIR = EXPORTS_DIR / "figures"
CACHE_DIR = PROJECT_ROOT / "cache"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
SCRATCH_DIR = BENCHMARK_DIR

# Fast FEM dynamic library path (.dylib for macOS, .so for Linux)
LIBFAST_FEM_DYLIB = BENCHMARK_DIR / "libfast_fem.dylib"
LIBFAST_FEM_SO = BENCHMARK_DIR / "libfast_fem.so"
LIBFAST_FEM_PATH = LIBFAST_FEM_DYLIB if IS_MACOS else LIBFAST_FEM_SO

# Ensure runtime directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 3. Global Protocol Constants
# ============================================================================

SEED: int = 42
TORCH_DTYPE: torch.dtype = torch.float64
NUMPY_DTYPE: np.dtype = np.float64

# CPU is optimal for float64 spectral operations & small-matrix linear solves on Apple Silicon
DEVICE: str = "cpu"

# Precision for arbitrary-precision mpmath precomputations (e.g. matrix A)
DPS: int = 50


# ============================================================================
# 4. Fractional Differential Equation Parameters
# ============================================================================

ALPHA_DEFAULT: float = 0.75
ALPHA_SWEEP: list[float] = [0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
T_START: float = 0.0
T_END: float = 1.0
Y0_DEFAULT: float = 1.0


# ============================================================================
# 5. Muntz-Legendre Basis & Spectral Grid Parameters
# ============================================================================

# Truncation order for Muntz polynomial basis {t^{k*alpha}}_{k=0..mhat} (No hidden layers)
MHAT_DEFAULT: int = 8
MHAT_SWEEP: list[int] = [4, 6, 8, 10, 12]

# Number of shifted Chebyshev collocation points (shifted roots on [0, 1])
NQ_DEFAULT: int = 64
NQ_SWEEP: list[int] = [16, 32, 64, 128]

# Fine discretization mesh for stochastic operational matrix S_alpha construction (n*)
N_MAX_BROWNIAN: int = 2**16  # 65,536 steps


# ============================================================================
# 6. Optimization & Loss Hyperparameters (Least Squares & Gauss-Newton)
# ============================================================================

LAMBDA_B: float = 1.0  # Drift projection loss weight
LAMBDA_S: float = 1.0  # Diffusion projection loss weight
GN_MAX_ITER: int = 50   # Max iterations for Gauss-Newton solver on nonlinear problems
GN_TOL: float = 1e-13   # Convergence tolerance for Gauss-Newton updates
LAPACK_DRIVER: str = "gelsd"  # High-precision SVD-based linear least-squares driver


# ============================================================================
# 7. Fractional Euler-Maruyama (fEM) Baseline Parameters
# ============================================================================

# High-resolution fine-grid for ground truth reference generation
FEM_N_FINE: int = 2**18  # 262,144 steps

# Standard/coarse grid for baseline comparison
FEM_N_COARSE: int = 2**14  # 16,384 steps

# Discretization sweep for Pareto analysis
FEM_N_SWEEP: list[int] = [2**10, 2**12, 2**14, 2**16, 2**18]

# Monte Carlo paths & parallelization
MC_NUM_PATHS: int = 1000
MC_NUM_PATHS_LARGE: int = 10000
MC_BATCH_SIZE: int = 100

# Leverage all Apple Silicon performance & efficiency cores
NUM_WORKER_THREADS: int = CPU_COUNT


# ============================================================================
# 8. Typed Configuration Dataclasses
# ============================================================================

@dataclass
class BasisConfig:
    """Parameters for the Muntz-Legendre polynomial basis and operational matrices."""
    alpha: float = ALPHA_DEFAULT
    mhat: int = MHAT_DEFAULT
    nq: int = NQ_DEFAULT
    n_max_brownian: int = N_MAX_BROWNIAN
    dps: int = DPS


@dataclass
class SolverOptConfig:
    """Parameters for linear least-squares and analytic Gauss-Newton solvers."""
    lambda_b: float = LAMBDA_B
    lambda_s: float = LAMBDA_S
    gn_max_iter: int = GN_MAX_ITER
    gn_tol: float = GN_TOL
    lapack_driver: str = LAPACK_DRIVER


@dataclass
class FEMConfig:
    """Parameters for Fractional Euler-Maruyama reference solvers."""
    n_fine: int = FEM_N_FINE
    n_coarse: int = FEM_N_COARSE
    num_threads: int = NUM_WORKER_THREADS
    lib_path: Path = LIBFAST_FEM_PATH
    use_c_extension: bool = True


@dataclass
class SolverConfig:
    """Master configuration encompassing all solver components."""
    seed: int = SEED
    device: str = DEVICE
    dtype: torch.dtype = TORCH_DTYPE
    t_span: tuple[float, float] = (T_START, T_END)
    y0: float = Y0_DEFAULT
    basis: BasisConfig = field(default_factory=BasisConfig)
    opt: SolverOptConfig = field(default_factory=SolverOptConfig)
    fem: FEMConfig = field(default_factory=FEMConfig)

    def __post_init__(self):
        torch.set_default_dtype(self.dtype)


# Default master configuration instance
DEFAULT_CONFIG = SolverConfig()
