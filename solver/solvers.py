"""
solvers.py
==========
High-level solver classes for Fractional Stochastic Differential Equations:
1. FEMSolver: Fractional Euler-Maruyama baseline solver.
2. MLDNNSolver: Muntz-Legendre Operational Matrix Solver (pure spectral representation,
   zero hidden layers, N(t) = c^T M^Lambda(t)).
"""

from __future__ import annotations
import numpy as np
from .core_mldnn import (
    basis_eval,
    chebyshev_nodes,
    em_caputo,
    solve_affine,
    solve_gauss_newton,
    evaluate_solution
)
from .milstein import MilsteinSolver, FastMilsteinSolver, solve_milstein_trig



class FEMSolver:
    """Fractional Euler-Maruyama (fEM) baseline solver."""
    def __init__(self, alpha: float, bfun, sfun):
        self.alpha = alpha
        self.bfun = bfun
        self.sfun = sfun

    def solve(self, y0: float, dB: np.ndarray, t_eval: np.ndarray | None = None) -> np.ndarray:
        """Solve a single or batch of Brownian paths using fEM."""
        return em_caputo(self.alpha, self.bfun, self.sfun, y0, dB, t_eval=t_eval)


class MLDNNSolver:
    """
    Muntz-Legendre Operational Matrix Solver for Caputo Fractional SDEs.
    
    Pure spectral representation (L = 2 layers: input basis -> linear output, 0 hidden layers):
        N(t) = c^T M^Lambda(t) = sum_{k=0}^{mhat} c_k M_k^Lambda(t)
    
    Parameters are solely the (mhat + 1) expansion coefficients c, and the (mhat + 1)
    projection vectors theta_b, theta_s.
    
    Training:
    - Affine systems: Exact linear least-squares (LAPACK gelsd) in closed-form (<1 ms).
    - Nonlinear systems: Damped Gauss-Newton with analytic Jacobian.
    - Zero hidden weights, zero backpropagation, machine precision (~1e-14).
    """
    def __init__(
        self,
        alpha: float,
        mhat: int,
        y0: float,
        S: np.ndarray | None = None,
        Nq: int = 64,
        lam_b: float = 1.0,
        lam_s: float = 1.0
    ):
        self.alpha = alpha
        self.mhat = mhat
        self.y0 = y0
        self.S = S
        self.Nq = Nq
        self.lam_b = lam_b
        self.lam_s = lam_s
        
        self.c: np.ndarray | None = None
        self.theta_b: np.ndarray | None = None
        self.theta_s: np.ndarray | None = None
        self.residual_norm: float | None = None
        self.n_iters: int = 0

    def solve_affine(
        self,
        b0,
        b1: float,
        s0,
        s1: float,
        trace_order: int = 1
    ) -> MLDNNSolver:
        """Solve affine drift b(t, y) = b0(t) + b1*y and diffusion sigma(t, y) = s0(t) + s1*y
        via closed-form linear least squares with Malliavin trace correction.
        """
        c, tb, ts, res = solve_affine(
            alpha=self.alpha,
            mhat=self.mhat,
            S=self.S,
            y0=self.y0,
            b0=b0,
            b1=b1,
            s0=s0,
            s1=s1,
            Nq=self.Nq,
            lam_b=self.lam_b,
            lam_s=self.lam_s,
            trace_order=trace_order
        )
        self.c = c
        self.theta_b = tb
        self.theta_s = ts
        self.residual_norm = res
        self.n_iters = 1
        return self

    def solve_nonlinear(
        self,
        bfun,
        bprime,
        sfun,
        sprime,
        sprime2=None,
        maxit: int = 50,
        tol: float = 1e-13,
        verbose: bool = False
    ) -> MLDNNSolver:
        """Solve general nonlinear CFSDE via analytic Gauss-Newton on basis coefficients c."""
        # solve_gauss_newton in core_mldnn returns (c, tb, ts, res)
        out = solve_gauss_newton(
            alpha=self.alpha,
            mhat=self.mhat,
            S=self.S,
            y0=self.y0,
            bfun=bfun,
            bprime=bprime,
            sfun=sfun,
            sprime=sprime,
            Nq=self.Nq,
            lam_b=self.lam_b,
            lam_s=self.lam_s,
            tol=tol,
            maxit=maxit,
            verbose=verbose,
            sprime2=sprime2
        )
        self.c = out[0]
        self.theta_b = out[1]
        self.theta_s = out[2]
        self.residual_norm = out[3]
        return self

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the continuous solution N(t) = c^T M^Lambda(t) at points t in [0, 1]."""
        if self.c is None:
            raise RuntimeError("Solver has not been solved yet. Call solve_affine or solve_nonlinear first.")
        return evaluate_solution(self.alpha, self.mhat, self.c, t)


# Alias for backward compatibility
MLSpectralSolver = MLDNNSolver
