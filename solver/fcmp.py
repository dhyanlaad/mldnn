"""
fcmp.py
=======
Filtered frozen-coefficient multipass (FCMP) solver.

This is the variant of the MLDNN scheme covered by the convergence theorem in
``agent/docs/notes/fcmp-convergence-proof.md`` (Theorem 1, alpha in (1/2, 1]).
It keeps the Muntz feature space, the deterministic operational matrix ``A``
(``get_A``) and the stochastic Fubini tensor (``build_fubini_tensor``), and
changes three things relative to ``solve_nonlinear_fubini_batch``:

1. the diffusion coefficients are *lagged*: pass k uses theta^(k-1), fitted
   from the previous state, so the only per-pass solve is the deterministic
   drift equation (no joint least squares, no random inverse Hessian);
2. all fits are filtered projections F_n = diag(f(j/n)) Pi_n with a smooth
   cutoff f (f = 1 on [0, 1/2], f(1) = 0), which is uniformly bounded on L^p;
3. the stochastic block of pass k is the pulled-out term minus the *exact*
   finite-dimensional Malliavin trace, differentiated through all earlier
   passes (Gaussian integration by parts over the Fubini matrix G).

The output is pinned to y(0) = y0 by the L^2-orthogonal projection onto
{v in V_n : v(0) = y0}, i.e. y + (y0 - y(0)) K_n(., 0) / K_n(0, 0) with K_n
the reproducing kernel of V_n.  This projection is non-expansive, so the
convergence theorem is unaffected (proof note, Section 8).

Scheme, in orthonormal coefficients phi_j = sqrt(2 j alpha + 1) M_j:

    theta^(0) = F sigma(., y0)
    s^(k)     = f * (G^T theta^(k-1) - tau^(k-1)),   tau^(0) = 0
    c^(k)     = f * (y0 e_0 + D F b(., y^(k))) + s^(k)
    theta^(k) = F sigma(., y^(k))

with D_ij = <phi_i, I^alpha phi_j>, G_ij = int phi_i(s) psi_j(s) dB_s,
psi_j(s) = int_s^1 K(t-s) phi_j(t) dt, and
tau_j = sum_{i,k,l} d theta_i / d G_kl Cov(G_kl, G_ij).

Coefficient functions must accept and return torch tensors (``torch.cos``,
not ``np.cos``) because the trace differentiates through them.

Cost: the nested derivative for pass k grows quickly with mhat and n_passes.
Keep ``chunk_size`` small (8 at mhat = 32, n_passes = 3 is ~1 GB).
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import torch
from scipy.special import roots_jacobi
from torch.func import grad, jacrev, vmap

from .core_mldnn import basis_eval, get_A
from .parallel import build_fubini_tensor

TorchFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _smooth_step(y: np.ndarray) -> np.ndarray:
    """C-infinity step: 0 for y <= 0, 1 for y >= 1."""
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    inside = (y > 0) & (y < 1)
    a = np.exp(-1.0 / y[inside])
    c = np.exp(-1.0 / (1.0 - y[inside]))
    out[inside] = a / (a + c)
    out[y >= 1] = 1.0
    return out


def fcmp_filter(mhat: int, kind: str = "smooth") -> np.ndarray:
    """Coefficient filter f(j/mhat), j = 0..mhat ('smooth' or 'none')."""
    if kind == "none" or mhat == 0:
        return np.ones(mhat + 1)
    if kind == "smooth":
        x = np.arange(mhat + 1, dtype=float) / mhat
        return _smooth_step((1.0 - x) / 0.5)
    raise ValueError(f"unknown filter {kind!r}")


def _t_rule(alpha: float, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss rule for int_0^1 f(t) dt, exact for f = p(t^alpha), deg p <= 2q-1."""
    beta = 1.0 / alpha - 1.0
    x, w = roots_jacobi(q, 0.0, beta)
    t = ((x + 1.0) / 2.0) ** (1.0 / alpha)
    return t, w / (alpha * 2.0 ** (beta + 1.0))


class FCMPOperators:
    """Path-independent pieces for a given (alpha, mhat, n_steps)."""

    def __init__(self, alpha: float, mhat: int, n_steps: int, filt: str = "smooth",
                 quad_order: int | None = None, M_tens: np.ndarray | None = None):
        self.alpha, self.mhat, self.n_steps = alpha, mhat, n_steps
        m1 = self.m1 = mhat + 1
        self.d = np.sqrt(2.0 * alpha * np.arange(m1) + 1.0)
        self.f = fcmp_filter(mhat, filt)

        tq, wq = _t_rule(alpha, quad_order or (2 * mhat + 40))
        self.wq = wq
        Mq = basis_eval(alpha, mhat, tq)                       # (m1, Q)
        Phi_q = self.d[:, None] * Mq
        self.Pw = Phi_q * wq[None, :]                          # coeff = Pw @ g(tq)
        self.PqT = Phi_q.T                                     # y(tq) = PqT @ c
        A = get_A(alpha, mhat)                                 # I^a M(t) = t^a A M(t)
        IaPhi_q = (tq ** alpha)[:, None] * (Mq.T @ A.T) * self.d[None, :]
        self.D = self.Pw @ IaPhi_q                             # <phi_i, I^a phi_j>, exact
        self.tq = tq

        # Xi_ij = int M_i K_j dB  ->  G_ij = d_i d_j Xi_ij  (orthonormal basis);
        # scaled in place when built here: at mhat=40, n_steps=65536 one copy is ~0.9 GB
        if M_tens is None:
            self.T, _ = build_fubini_tensor(alpha, mhat, n_steps)
            self.T *= np.outer(self.d, self.d).reshape(-1)[:, None]
        else:
            self.T = M_tens * np.outer(self.d, self.d).reshape(-1)[:, None]
        C = (self.T @ self.T.T) / n_steps                     # Cov(G), dB ~ N(0, 1/n_steps)
        self.C4 = C.reshape(m1, m1, m1, m1)

        tt = lambda a: torch.as_tensor(a, dtype=torch.float64)
        self.tf, self.tPw, self.tPqT, self.tD = tt(self.f), tt(self.Pw), tt(self.PqT), tt(self.D)
        self.tC4, self.ttq = tt(self.C4), tt(tq)
        self.tfD = self.tf[:, None] * self.tD

    def G_from_dB(self, dB: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(dB) @ self.T.T).reshape(-1, self.m1, self.m1)

    def eval_matrix(self, t: np.ndarray) -> np.ndarray:
        """(len(t), m1) matrix mapping orthonormal coefficients to values."""
        return (self.d[:, None] * basis_eval(self.alpha, self.mhat, np.asarray(t, float))).T

    def pin(self, c: np.ndarray, y0: float) -> np.ndarray:
        """L^2 projection of coefficient rows c onto {v in V_n : v(0) = y0}."""
        v0 = self.eval_matrix(np.array([0.0]))[0]           # phi_j(0); K_n(., 0) = sum phi_j(0) phi_j
        return c + ((y0 - c @ v0) / (v0 @ v0))[..., None] * v0

    def contraction_certificate(self, L: float, lambdas: np.ndarray | None = None) -> tuple[float, float]:
        """kappa = min_lambda L * ||X||_lambda for the discrete drift map.

        X = PqT diag(f) D diag(f) Pw maps node values b(y(t_q)) to the drift
        contribution to y(t_q); ||.||_lambda is the node norm
        sum_q w_q exp(-2 lambda t_q) v_q^2.  If |d_y b| <= L and kappa < 1, the
        per-pass state equation has a unique solution for every path and the
        Picard iteration converges from any start at rate kappa.  Returns
        (kappa, argmin lambda).
        """
        X = self.PqT @ (self.f[:, None] * self.D * self.f[None, :]) @ self.Pw
        lambdas = np.linspace(0.0, 40.0, 161) if lambdas is None else lambdas
        sw = np.sqrt(self.wq)
        best = (np.inf, 0.0)
        for lam in lambdas:
            scale = (sw[:, None] / sw[None, :]) * np.exp(-lam * (self.tq[:, None] - self.tq[None, :]))
            k = L * np.linalg.norm(scale * X, 2)
            best = min(best, (k, lam))
        return best


class _FCMPMap:
    """The scheme as a differentiable function of one Fubini matrix G."""

    def __init__(self, ops: FCMPOperators, y0: float, bfun: TorchFn, sfun: TorchFn,
                 bprime: TorchFn | None, affine_b: tuple | None, n_passes: int,
                 trace: str, newton_iters: int, picard_iters: int = 20):
        if trace not in ("exact", "none"):
            raise ValueError("trace must be 'exact' or 'none'")
        self.o, self.b, self.s, self.NT, self.trace = ops, bfun, sfun, n_passes, trace
        self.newton_iters, self.picard_iters = newton_iters, picard_iters
        if bprime is None:
            bprime = lambda t, y: vmap(grad(lambda yy, tt: bfun(tt, yy)))(y, t)
        self.bp = bprime
        m1 = ops.m1
        e0 = torch.zeros(m1, dtype=torch.float64)
        e0[0] = 1.0
        self.c0 = y0 * e0                                      # phi_0 = 1
        self.base = ops.tf * self.c0
        self.theta0 = self.fit(sfun, self.c0)
        self.Aff = None
        if affine_b is not None:
            b0, b1 = affine_b
            b0q = torch.as_tensor(np.broadcast_to(b0(ops.tq) if callable(b0) else b0, ops.tq.shape).copy())
            self.Aff = torch.eye(m1, dtype=torch.float64) - b1 * ops.tfD * ops.tf[None, :]
            self.rhs0 = self.base + ops.tfD @ (ops.tf * (ops.tPw @ b0q))

    def fit(self, fun: TorchFn, c: torch.Tensor) -> torch.Tensor:
        o = self.o
        return o.tf * (o.tPw @ fun(o.ttq, o.tPqT @ c))

    def residual(self, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return c - self.base - self.o.tfD @ self.fit(self.b, c) - s

    def solve_state(self, s: torch.Tensor) -> torch.Tensor:
        o = self.o
        if self.Aff is not None:
            return torch.linalg.solve(self.Aff, self.rhs0 + s)
        c = self.base + s
        # Picard first: a proven contraction whenever ops.contraction_certificate(L) < 1;
        # Newton then polishes quadratically from inside the basin
        for _ in range(self.picard_iters):
            c = self.base + o.tfD @ self.fit(self.b, c) + s
        for _ in range(self.newton_iters):
            bp = self.bp(o.ttq, o.tPqT @ c)
            Jr = torch.eye(o.m1, dtype=torch.float64) - o.tfD @ (o.tf[:, None] * (o.tPw * bp[None, :]) @ o.tPqT)
            c = c - torch.linalg.solve(Jr, self.residual(c, s))
        return c

    def passes(self) -> list:
        """fns[k](G) -> (theta^(k), c^(k), s^(k))."""
        o = self.o
        fns = [lambda G: (self.theta0, self.c0, torch.zeros(o.m1, dtype=torch.float64))]

        def make(k):
            prev = fns[k - 1]

            def fn(G):
                if k == 1 or self.trace == "none":
                    th_prev = prev(G)[0]
                    tau = torch.zeros(o.m1, dtype=torch.float64)
                else:
                    J, th_prev = jacrev(lambda g: (prev(g)[0],) * 2, has_aux=True)(G)
                    tau = torch.einsum("ikl,klij->j", J, o.tC4)
                s = o.tf * (G.T @ th_prev - tau)
                c = self.solve_state(s)
                return self.fit(self.s, c), c, s
            return fn

        for k in range(1, self.NT + 1):
            fns.append(make(k))
        return fns


def solve_fcmp_batch(
    alpha: float,
    mhat: int,
    dB: np.ndarray,
    y0: float,
    bfun: TorchFn,
    sfun: TorchFn,
    bprime: TorchFn | None = None,
    affine_b: tuple | None = None,
    n_passes: int = 3,
    filt: str = "smooth",
    trace: str = "exact",
    t_eval: np.ndarray | None = None,
    newton_iters: int = 6,
    picard_iters: int = 20,
    pin_initial: bool = True,
    newton_tol: float = 1e-10,
    chunk_size: int = 8,
    quad_order: int | None = None,
    ops: FCMPOperators | None = None,
    return_coeffs: bool = False,
):
    """FCMP solve for a batch of Brownian increment paths ``dB`` (n_paths, n_steps).

    ``bfun``/``sfun``/``bprime`` take torch tensors ``(t, y)``.  For a drift affine
    in y, pass ``affine_b=(b0, b1)`` (b0 constant or numpy callable of t, b1
    constant) to use an exact linear solve; ``bfun`` is then unused.
    Returns values at ``t_eval`` (n_paths, len(t_eval)); with ``t_eval=None``,
    coefficients in the production basis M_j (compatible with
    ``evaluate_solution``).  ``return_coeffs=True`` also returns the orthonormal
    coefficients of every pass, shape (n_passes, n_paths, mhat+1).
    ``pin_initial`` projects each output onto {v : v(0) = y0} (internal passes
    are not pinned).
    """
    dB = np.atleast_2d(np.asarray(dB, dtype=float))
    n_paths, n_steps = dB.shape
    if not 0.5 < alpha <= 1.0:
        warnings.warn("FCMP convergence theory covers alpha in (1/2, 1] only")
    if ops is None:
        ops = FCMPOperators(alpha, mhat, n_steps, filt, quad_order)
    if affine_b is not None and bfun is None:
        bfun = lambda t, y: torch.zeros_like(y)
    scheme = _FCMPMap(ops, y0, bfun, sfun, bprime, affine_b, n_passes, trace, newton_iters, picard_iters)
    fns = scheme.passes()
    G = torch.as_tensor(ops.G_from_dB(dB))

    coeffs = np.empty((n_passes, n_paths, ops.m1))
    worst = 0.0
    # pass k recomputes passes 1..k-1 internally, so only run the ones requested
    for k in (range(1, n_passes + 1) if return_coeffs else [n_passes]):
        run = vmap(lambda g: fns[k](g)[1:])
        for i in range(0, n_paths, chunk_size):
            c, s = run(G[i:i + chunk_size])
            coeffs[k - 1, i:i + chunk_size] = c.detach().numpy()
            if affine_b is None:
                r = vmap(scheme.residual)(c, s).abs().max().item()
                worst = max(worst, r)
        if pin_initial:
            coeffs[k - 1] = ops.pin(coeffs[k - 1], y0)
    if worst > newton_tol:
        warnings.warn(f"FCMP Newton residual {worst:.2e} exceeds {newton_tol:.0e}; "
                      "increase newton_iters")

    final = coeffs[-1]
    if t_eval is not None:
        out = final @ ops.eval_matrix(t_eval).T
    else:
        out = final * ops.d[None, :]                           # M_j-basis coefficients
    return (out, coeffs) if return_coeffs else out


# ----------------------------------------------------------------------------
# Experiment-facing wrappers
# ----------------------------------------------------------------------------

_OPS_CACHE: dict = {}


def get_operators(alpha: float, mhat: int, n_steps: int, filt: str = "none") -> FCMPOperators:
    """FCMPOperators for (alpha, mhat, n_steps, filt), reusing the last one built.

    Only one set is kept: at mhat = 40, n_steps = 65536 the Fubini tensor alone is ~0.9 GB.
    """
    key = (float(alpha), int(mhat), int(n_steps), filt)
    if key not in _OPS_CACHE:
        _OPS_CACHE.clear()
        _OPS_CACHE[key] = FCMPOperators(alpha, mhat, n_steps, filt)
    return _OPS_CACHE[key]


def _default_chunk(mhat: int, n_passes: int) -> int:
    # the nested trace Jacobian of pass k has ~(mhat+1)^(2k) entries per path;
    # 8 paths at mhat = 32 with 3 passes is ~1 GB
    if n_passes <= 2:
        return 64
    return int(np.clip(8 * (33.0 / (mhat + 1)) ** (2 * (n_passes - 2)), 1, 64))


def solve_affine_batch(alpha: float, mhat: int, dB: np.ndarray, y0: float,
                       b0=0.0, b1: float = 0.0, s0: float = 0.0, s1: float = 0.0,
                       t_eval: np.ndarray | None = None, n_passes: int | None = None,
                       filt: str = "none", chunk_size: int | None = None, **kw) -> np.ndarray:
    """FCMP for b = b0 + b1 y, sigma = s0 + s1 y (b0 constant or numpy callable of t).

    Additive noise (s1 = 0) needs one pass (every pass is identical); otherwise
    three.  Unfiltered FCMP is covered by Theorem 1 for affine coefficients at any
    pass count.
    """
    dB = np.atleast_2d(np.asarray(dB, dtype=float))
    if n_passes is None:
        n_passes = 1 if s1 == 0.0 else 3
    ops = get_operators(alpha, mhat, dB.shape[1], filt)
    return solve_fcmp_batch(alpha, mhat, dB, y0, None, lambda t, y: s0 + s1 * y,
                            affine_b=(b0, b1), n_passes=n_passes, filt=filt, t_eval=t_eval, ops=ops,
                            chunk_size=chunk_size or _default_chunk(mhat, n_passes), **kw)


def solve_nonlinear_batch(alpha: float, mhat: int, dB: np.ndarray, y0: float,
                          bfun: TorchFn, sfun: TorchFn, bprime: TorchFn | None = None,
                          t_eval: np.ndarray | None = None, n_passes: int = 2,
                          filt: str = "none", chunk_size: int | None = None, **kw) -> np.ndarray:
    """FCMP for general b(t, y), sigma(t, y) given as torch callables.

    Default: two unfiltered passes, the largest pass count Theorem 1 covers
    without the filter for C^1 coefficients.
    """
    dB = np.atleast_2d(np.asarray(dB, dtype=float))
    ops = get_operators(alpha, mhat, dB.shape[1], filt)
    return solve_fcmp_batch(alpha, mhat, dB, y0, bfun, sfun, bprime=bprime,
                            n_passes=n_passes, filt=filt, t_eval=t_eval, ops=ops,
                            chunk_size=chunk_size or _default_chunk(mhat, n_passes), **kw)
