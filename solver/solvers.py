"""
solvers.py
==========
High-level solver classes for Fractional Stochastic Differential Equations:
1. FEMSolver: Fractional Euler-Maruyama baseline solver.
2. Milstein reference solvers (re-exported from milstein.py).
The MLDNN solver itself is solver/fcmp.py.
"""

from __future__ import annotations
import numpy as np
from .core_mldnn import em_caputo
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
