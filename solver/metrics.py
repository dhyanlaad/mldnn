"""
metrics.py
==========
Standardized error metrics and statistical diagnostics for the ML-PIFLNN solver.

Distinct error metrics defined and implemented:
1. Paired Log-Ratio (Drift-Level Diagnostic):
   Delta_r(t) = log(y_solver^(r)(t)) - log(y_ref^(r)(t)) on matched Brownian paths.
   Cancels the common Brownian driver and achieves ~400x higher sensitivity for drift/trace effects.
2. Trajectory MSE / RMSE:
   MSE(t) = (1/R) sum_{r=1}^R (y_solver^(r)(t) - y_ref^(r)(t))^2.
3. Terminal RMSE:
   RMSE(1) = sqrt( (1/R) sum_{r=1}^R (y_solver^(r)(1) - y_ref^(r)(1))^2 ).
4. Mean Trajectory Discrete L2 Error (Expectation Error):
   Discrete_L2 = (1/N_t) * sqrt( sum_{k=1}^{N_t} ( (1/R) sum_{r=1}^R y_solver^(r)(t_k) - E[y(t_k)] )^2 ).
5. Discrete Linf:
   Discrete_Linf = max_k | (1/R) sum_{r=1}^R y_solver^(r)(t_k) - E[y(t_k)] |.
"""

from __future__ import annotations
import numpy as np


def paired_log_ratio(
    solver_paths: np.ndarray,
    reference_paths: np.ndarray,
    t_idx: int = -1,
    n_boot: int = 10000,
    ci_alpha: float = 0.05,
    seed: int = 12345
) -> tuple[float, tuple[float, float], np.ndarray]:
    """
    Compute the paired terminal log-ratio on matched Brownian paths.

    Returns:
        mean_delta: float, average log-ratio across sample paths
        (ci_low, ci_high): tuple of float, 95% bootstrap percentile confidence interval
        deltas: np.ndarray, per-path log differences
    """
    if solver_paths.ndim == 1:
        s_eval = solver_paths
        r_eval = reference_paths
    else:
        s_eval = solver_paths[:, t_idx]
        r_eval = reference_paths[:, t_idx]

    deltas = np.log(np.clip(s_eval, 1e-300, None)) - np.log(np.clip(r_eval, 1e-300, None))
    mean_delta = float(np.mean(deltas))

    rng = np.random.default_rng(seed)
    n = len(deltas)
    boot_means = np.empty(n_boot, dtype=np.float64)
    batch_size = 256
    for start in range(0, n_boot, batch_size):
        stop = min(start + batch_size, n_boot)
        idx = rng.integers(0, n, size=(stop - start, n))
        boot_means[start:stop] = np.mean(deltas[idx], axis=1)

    lo = float(np.percentile(boot_means, 100.0 * ci_alpha / 2.0))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - ci_alpha / 2.0)))
    return mean_delta, (lo, hi), deltas


def terminal_rmse(solver_paths: np.ndarray, reference_paths: np.ndarray) -> float:
    """Terminal RMSE across paths at t = 1.0."""
    s_t1 = solver_paths[:, -1] if solver_paths.ndim > 1 else solver_paths
    r_t1 = reference_paths[:, -1] if reference_paths.ndim > 1 else reference_paths
    return float(np.sqrt(np.mean((s_t1 - r_t1) ** 2)))


def trajectory_mse(solver_paths: np.ndarray, reference_paths: np.ndarray) -> np.ndarray:
    """Mean squared error at each time evaluation node: MSE(t) = (1/R) sum_r (y_r(t) - ref_r(t))^2."""
    diff = solver_paths - reference_paths
    return np.mean(diff ** 2, axis=0)


def mean_trajectory_l2(solver_paths: np.ndarray, exact_mean: np.ndarray) -> float:
    """
    Discrete L2 error of the empirical Monte Carlo mean trajectory against analytic expectation:
    (1/N_t) * sqrt( sum_k ( y_bar(t_k) - E[y(t_k)] )^2 ).
    """
    num_mean = np.mean(solver_paths, axis=0)
    n_t = len(exact_mean)
    return float((1.0 / n_t) * np.sqrt(np.sum((num_mean - exact_mean) ** 2)))


def discrete_linf(solver_paths: np.ndarray, exact_mean: np.ndarray) -> float:
    """Max pointwise difference between empirical Monte Carlo mean trajectory and expectation."""
    num_mean = np.mean(solver_paths, axis=0)
    return float(np.max(np.abs(num_mean - exact_mean)))


def absolute_error(exact: np.ndarray, approx: np.ndarray) -> np.ndarray:
    return np.abs(exact - approx)


def max_error(exact: np.ndarray, approx: np.ndarray) -> float:
    return float(np.max(np.abs(exact - approx)))


def mean_absolute_error(exact: np.ndarray, approx: np.ndarray) -> float:
    return float(np.mean(np.abs(exact - approx)))


def strong_error(fine: np.ndarray, coarse: np.ndarray) -> float:
    return float(np.sqrt(np.mean((fine - coarse) ** 2)))
