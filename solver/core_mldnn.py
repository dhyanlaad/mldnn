"""
core_mldnn.py
=============
Core numerical machinery for the Muntz-Legendre network solver for Caputo SDEs

    D^alpha y(t) = b(t, y(t)) + sigma(t, y(t)) dB_t,   t in [0,1],  y(0)=y0,
    alpha in (1/2, 1),

following the manuscript exactly:

*  Basis (Sec. 1.3): Muntz exponents Lambda = {k*alpha}, k = 0..mhat.  The
   Muntz-Legendre polynomials (orthogonal on [0,1] w.r.t. the unweighted L^2
   inner product of Def. 1.1, normalised so that M_n(1)=1) are

       M_n(t) = P_n^{(0, 1/alpha - 1)}(2 t^alpha - 1),

   i.e. shifted Jacobi polynomials in u = t^alpha.  They are evaluated by the
   stable three-term Jacobi recurrence (scipy.special.eval_jacobi); the
   ill-conditioned monomial transform C is *never* formed in float64.
   Exact norms:  ||M_n||^2 = 1/(2*alpha*n + 1).

*  Deterministic operational matrix (Sec. 2.1):  since lambda_j + alpha =
   lambda_{j+1}, the identity  I^alpha M(t) = t^alpha * A * M(t)  with
   A = C D_alpha C^{-1} holds pointwise with no truncation.  A is assembled in
   mpmath at DPS decimal digits (default 50) and cast to float64.

*  Stochastic operational matrix (Sec. 2.1 / Appendix B):
   S_alpha = Xi * Omega^{-1}.  The weakly singular Wiener integrals I_i(t) are
   evaluated on a uniform fine mesh by a kernel-exact rule: the kernel
   (t - tau)^{alpha-1} is integrated exactly on every cell against the linear
   interpolant of the (deterministic) basis function, with the Brownian
   increment entering through the cell-average density dB_j / dtau.  Because
   the integrands M_i are deterministic, this is a valid L^2(Omega)
   approximation of the Wiener integral (same limit as the left-point rule of
   Appendix B, smaller constant).  Assembly is a causal convolution done with
   FFTs: O(mhat * n log n).  Xi is then integrated with the trapezoidal rule
   on the same mesh; Omega^{-1} = diag(2*alpha*k + 1) is exact.

*  Network / training (Sec. 1.3, 2.2, Algorithm 1): we instantiate the
   architecture with L = 2 (fractional feature layer + linear output layer),
   N_theta(t) = c^T M(t).  For coefficients affine in y the total loss of
   Sec. 2.2 is an exact linear least-squares problem (solved with LAPACK
   gelsd = fully converged training); for nonlinear drift we use Gauss-Newton
   with the analytic Jacobian.

*  Fractional Euler-Maruyama reference scheme (Doan et al. 2020, ref. [1]).

*  Mittag-Leffler function: float64 vectorised series with cancellation
   guard + arbitrary-precision mpmath fallback (needed at lambda = 16, where
   float64 loses ~14 digits to cancellation).

*  Graded composite Gauss-Legendre + Gauss-Jacobi quadrature for the
   smoothed-driver ablation (self-consistent to ~1e-12).

Author: generated for the Section-4 numerical study.  Seed = 42 throughout.
"""

from __future__ import annotations
import csv, json, os, time
import numpy as np
from scipy.special import eval_jacobi, roots_jacobi, gammaln, gamma as sgamma
from scipy.linalg import lstsq

# ----------------------------------------------------------------------------
# global protocol constants
# ----------------------------------------------------------------------------
SEED    = 42
ALPHA0  = 0.75
NQ      = 64          # collocation points (shifted Chebyshev roots, Alg. 1)
DPS     = 50          # mpmath working precision for A = C D_alpha C^{-1}
N_MAX   = 2**16       # finest Brownian mesh; coarser meshes by block summation

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "results")
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# ============================================================================
# 1. Muntz-Legendre basis  M_n(t) = P_n^{(0,c)}(2 t^alpha - 1),  c = 1/alpha-1
# ============================================================================

def cpar(alpha: float) -> float:
    return 1.0 / alpha - 1.0

def basis_eval(alpha: float, mhat: int, t: np.ndarray) -> np.ndarray:
    """M[n, j] = M_n(t_j),  n = 0..mhat.  Stable Jacobi recurrence."""
    t = np.asarray(t, dtype=float)
    x = 2.0 * np.power(np.clip(t, 0.0, None), alpha) - 1.0
    c = cpar(alpha)
    M = np.empty((mhat + 1, t.size))
    for n in range(mhat + 1):
        M[n] = eval_jacobi(n, 0.0, c, x)
    return M

def basis_norm2(alpha: float, mhat: int) -> np.ndarray:
    """Exact ||M_n||^2_{L^2[0,1]} = 1/(2 alpha n + 1)."""
    n = np.arange(mhat + 1, dtype=float)
    return 1.0 / (2.0 * alpha * n + 1.0)

# ----------------------------------------------------------------------------
# deterministic operational matrix  A  with  I^alpha M(t) = t^alpha A M(t)
# (exact pointwise; assembled at high precision, cached on disk)
# ----------------------------------------------------------------------------

def _compute_A_mpmath(alpha: float, mhat: int, dps: int = DPS) -> np.ndarray:
    import mpmath as mp
    dps_eff = max(dps, 30 + int(0.9 * mhat))   # C has condition ~ 5.8^mhat
    with mp.workdps(dps_eff):
        one = mp.mpf(1)
        al  = mp.mpf(repr(alpha))
        c   = one / al - one
        m   = mhat + 1
        # coefficient rows of p_n(u) in monomials u^k (lower triangular)
        C = [[mp.mpf(0)] * m for _ in range(m)]
        C[0][0] = one
        if mhat >= 1:
            C[1][0] = -(c + 1); C[1][1] = c + 2          # p1(u) = (c+2)u-(c+1)
        for n in range(2, m):
            nn = mp.mpf(n)
            a1 = 2 * nn * (nn + c) * (2 * nn + c - 2)                    # * p_n
            b1 = (2 * nn + c - 1) * (2 * nn + c) * (2 * nn + c - 2)      # * x p_{n-1}
            b0 = -(2 * nn + c - 1) * c * c                               # * p_{n-1}
            d1 = 2 * (nn - 1) * (nn + c - 1) * (2 * nn + c)              # * p_{n-2}
            row = [mp.mpf(0)] * m
            for k in range(n):        # x = 2u - 1  acting on p_{n-1}
                pk = C[n - 1][k]
                if pk != 0:
                    row[k + 1] += 2 * b1 * pk / a1
                    row[k]     += (-b1 + b0) * pk / a1
            for k in range(n - 1):
                row[k] -= d1 * C[n - 2][k] / a1
            C[n] = row
        # g_k = Gamma(k a + 1)/Gamma((k+1) a + 1)
        g = [mp.gamma(k * al + 1) / mp.gamma((k + 1) * al + 1) for k in range(m)]
        # Cg[n,k] = C[n,k] * g_k ;  solve A C = Cg  (C lower-triangular)
        A = [[mp.mpf(0)] * m for _ in range(m)]
        for n in range(m):
            for j in range(n, -1, -1):         # back-substitute (C^T upper-tri)
                s = C[n][j] * g[j]
                for l in range(j + 1, n + 1):
                    s -= A[n][l] * C[l][j]
                A[n][j] = s / C[j][j]
        return np.array([[float(A[i][j]) for j in range(m)] for i in range(m)])

_A_CACHE: dict = {}

def get_A(alpha: float, mhat: int) -> np.ndarray:
    """Deterministic operational matrix A (float64), disk+memory cached.
    Leading principal submatrices are valid (A is lower triangular)."""
    key = (round(alpha, 12), mhat)
    if key in _A_CACHE:
        return _A_CACHE[key]
    fn = os.path.join(RESULTS, "cache_A_%0.6f_%d.npy" % (alpha, mhat))
    if os.path.exists(fn):
        A = np.load(fn)
    else:
        A = _compute_A_mpmath(alpha, mhat)
        np.save(fn, A)
    _A_CACHE[key] = A
    return A

def frac_int_exact_power(alpha: float, p: float, t: np.ndarray) -> np.ndarray:
    """I^alpha t^p = Gamma(p+1)/Gamma(p+alpha+1) t^{p+alpha}  (beta identity)."""
    cst = np.exp(gammaln(p + 1.0) - gammaln(p + alpha + 1.0))
    return cst * np.power(t, p + alpha)

# ============================================================================
# 2. Brownian paths (seed 42, nested by block summation)
# ============================================================================

def brownian_increments(n: int, rng=None) -> np.ndarray:
    rng = np.random.default_rng(SEED) if rng is None else rng
    return rng.standard_normal(n) * np.sqrt(1.0 / n)

def brownian_paths(n_steps: int, n_paths: int, rng=None) -> np.ndarray:
    rng = np.random.default_rng(SEED) if rng is None else rng
    return rng.standard_normal((n_paths, n_steps)) * np.sqrt(1.0 / n_steps)

def coarsen(dB_fine: np.ndarray, n: int) -> np.ndarray:
    f = dB_fine.size // n
    assert dB_fine.size % n == 0
    return dB_fine.reshape(n, f).sum(axis=1)

# ============================================================================
# 3. Stochastic operational matrix  S_alpha = Xi Omega^{-1}
# ============================================================================

def _pow_diff(d: np.ndarray, p: float) -> np.ndarray:
    """d^p - (d-1)^p, stable for large d (d >= 1)."""
    with np.errstate(divide="ignore"):
        return -np.expm1(p * np.log1p(-1.0 / d)) * np.power(d, p)

def _causal_conv(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    """out[:, k] = sum_{j<k} F[:, j] K[k-j];  F (m,n), K (n+1,) with K[0]=0."""
    n = F.shape[1]
    nf = 1 << int(np.ceil(np.log2(2 * (n + 1))))
    out = np.fft.irfft(np.fft.rfft(F, nf, axis=1) * np.fft.rfft(K, nf), nf, axis=1)
    return out[:, : n + 1]

def stochastic_integrals(alpha: float, Mmesh: np.ndarray, dB: np.ndarray,
                         rule: str = "kel") -> np.ndarray:
    """
    I[i, k] ~= (1/Gamma(a)) int_0^{t_k} (t_k - tau)^{a-1} M_i(tau) dB_tau
    on the uniform mesh t_k = k/n.   rule = 'kel' (kernel-exact linear, default)
    or 'lp' (left-point rule of Appendix B).
    """
    n = dB.size
    D = 1.0 / n
    d = np.arange(1, n + 1, dtype=float)
    if rule == "kel":
        m0   = D**alpha * _pow_diff(d, alpha) / alpha                 # int cell w^{a-1}
        A1oD = D**alpha * _pow_diff(d, alpha + 1.0) / (alpha + 1.0)   # int cell w^a / D
        KL = np.concatenate(([0.0], A1oD - (d - 1.0) * m0))   # weight of left node
        KR = np.concatenate(([0.0], d * m0 - A1oD))           # weight of right node
        F = Mmesh[:, :-1] * (dB / D)
        H = Mmesh[:, 1:]  * (dB / D)
        I = _causal_conv(F, KL) + _causal_conv(H, KR)
    elif rule == "lp":
        kappa = np.concatenate(([0.0], np.power(d * D, alpha - 1.0)))
        I = _causal_conv(Mmesh[:, :-1] * dB, kappa)
    else:
        raise ValueError(rule)
    return I / sgamma(alpha)

def build_S(alpha: float, mhat: int, dB: np.ndarray, rule: str = "kel",
            return_I: bool = False):
    """S_alpha = Xi Omega^{-1};  Xi by trapezoid on the fine mesh (Sec. 2.1)."""
    n = dB.size
    tm = np.linspace(0.0, 1.0, n + 1)
    Mmesh = basis_eval(alpha, mhat, tm)
    I = stochastic_integrals(alpha, Mmesh, dB, rule=rule)
    w = np.full(n + 1, 1.0 / n); w[0] *= 0.5; w[-1] *= 0.5
    Xi = (I * w) @ Mmesh.T                       # (m+1, m+1)
    S = Xi * (2.0 * alpha * np.arange(mhat + 1) + 1.0)[None, :]   # Xi Omega^{-1}
    if return_I:
        return S, I, Mmesh, tm
    return S

# ============================================================================
# 4. Collocation + solvers (Sec. 2.2 / Algorithm 1, Phase II)
# ============================================================================

def chebyshev_nodes(Nq: int = NQ) -> np.ndarray:
    i = np.arange(1, Nq + 1)
    return 0.5 * (1.0 - np.cos((2 * i - 1) * np.pi / (2 * Nq)))

class Blocks:
    """Precomputed collocation blocks (Alg. 1, line 11)."""
    def __init__(self, alpha, mhat, S, Nq=NQ):
        self.alpha, self.mhat = alpha, mhat
        self.t = chebyshev_nodes(Nq)
        self.Phi = basis_eval(alpha, mhat, self.t)          # (m+1, Nq)
        A = get_A(alpha, mhat)
        self.PhiT = self.Phi.T                               # (Nq, m+1)
        self.DetT = (self.t ** alpha)[:, None] * (self.PhiT @ A.T)  # rows t^a M^T A^T
        self.StoT = self.PhiT @ S.T

def solve_affine(alpha, mhat, S, y0, b0, b1, s0, s1, Nq=NQ,
                 lam_b: float = 1.0, lam_s: float = 1.0):
    """
    Exact least-squares solution of the total loss (Sec. 2.2) for
    b(t,y) = b0(t) + b1*y,  sigma(t,y) = s0(t) + s1*y.
    Returns (c, theta_b, theta_s, residual_norm).
    """
    B = Blocks(alpha, mhat, S, Nq)
    t, PhiT = B.t, B.PhiT
    m1 = mhat + 1
    Z = np.zeros((Nq, m1))
    b0v = b0(t) if callable(b0) else np.full(Nq, float(b0))
    s0v = s0(t) if callable(s0) else np.full(Nq, float(s0))
    wb, ws = np.sqrt(lam_b), np.sqrt(lam_s)
    R1 = np.hstack([PhiT, -B.DetT, -B.StoT])
    R2 = wb * np.hstack([-b1 * PhiT, PhiT, Z])
    R3 = ws * np.hstack([-s1 * PhiT, Z, PhiT])
    Amat = np.vstack([R1, R2, R3])
    rhs = np.concatenate([np.full(Nq, y0), wb * b0v, ws * s0v])
    z, res, rank, sv = lstsq(Amat, rhs, lapack_driver="gelsd")
    r = Amat @ z - rhs
    return z[:m1], z[m1:2 * m1], z[2 * m1:], float(np.linalg.norm(r) / np.sqrt(Nq))

def solve_gauss_newton(alpha, mhat, S, y0, bfun, bprime, sfun, sprime,
                       Nq=NQ, lam_b=1.0, lam_s=1.0, tol=1e-13, maxit=60,
                       z0=None, verbose=False):
    """
    Gauss-Newton (analytic Jacobian, damped) on the total loss of Sec. 2.2 for
    general b(t,y), sigma(t,y).  bprime/sprime = partial_y derivatives.
    """
    B = Blocks(alpha, mhat, S, Nq)
    t, PhiT = B.t, B.PhiT
    m1 = mhat + 1
    Z = np.zeros((Nq, m1))
    wb, ws = np.sqrt(lam_b), np.sqrt(lam_s)

    def residual(z):
        c, tb, ts = z[:m1], z[m1:2 * m1], z[2 * m1:]
        N = PhiT @ c
        r1 = N - y0 - B.DetT @ tb - B.StoT @ ts
        r2 = wb * (PhiT @ tb - bfun(t, N))
        r3 = ws * (PhiT @ ts - sfun(t, N))
        return np.concatenate([r1, r2, r3]), N

    def jac(z, N):
        J1 = np.hstack([PhiT, -B.DetT, -B.StoT])
        J2 = wb * np.hstack([-(bprime(t, N))[:, None] * PhiT, PhiT, Z])
        J3 = ws * np.hstack([-(sprime(t, N))[:, None] * PhiT, Z, PhiT])
        return np.vstack([J1, J2, J3])

    if z0 is None:
        # initialise from the affine solve with b linearised about y0
        b1_0 = float(bprime(np.array([0.0]), np.array([y0]))[0])
        s1_0 = float(sprime(np.array([0.0]), np.array([y0]))[0])
        b0f = lambda tt: bfun(tt, np.full_like(tt, y0)) - b1_0 * y0
        s0f = lambda tt: sfun(tt, np.full_like(tt, y0)) - s1_0 * y0
        c, tb, ts, _ = solve_affine(alpha, mhat, S, y0, b0f, b1_0, s0f, s1_0,
                                    Nq, lam_b, lam_s)
        z = np.concatenate([c, tb, ts])
    else:
        z = z0.copy()
    r, N = residual(z)
    for it in range(maxit):
        J = jac(z, N)
        dz, *_ = lstsq(J, -r, lapack_driver="gelsd")
        step = 1.0
        for _ in range(30):
            r_new, N_new = residual(z + step * dz)
            if np.linalg.norm(r_new) <= np.linalg.norm(r) * (1 + 1e-14) or step < 1e-8:
                break
            step *= 0.5
        z, r, N = z + step * dz, r_new, N_new
        if verbose:
            print("  GN it %2d |r| = %.3e step %.2g" % (it, np.linalg.norm(r), step))
        if np.linalg.norm(step * dz) < tol * max(1.0, np.linalg.norm(z)):
            break
    c, tb, ts = z[:m1], z[m1:2 * m1], z[2 * m1:]
    return c, tb, ts, float(np.linalg.norm(r) / np.sqrt(Nq)), it + 1


def evaluate_solution(alpha, mhat, c, t):
    """Evaluate solution at points t.
    Supports both affine (size mhat+1) and deep (augmented) coefficient vectors.
    """
    expected_len = mhat + 1
    c_arr = np.asarray(c, float)
    # Basis matrix (Nq, m+1)
    PhiT = basis_eval(alpha, mhat, np.asarray(t, float)).T
    if c_arr.shape[0] == expected_len:
        # Standard affine solution
        return PhiT @ c_arr
    # Deep solution: extra hidden features were added
    hidden = c_arr.shape[0] - expected_len
    PhiT_aug = _random_hidden_features(PhiT, hidden, None)
    return PhiT_aug @ c_arr




# ============================================================================
# 5b. Deep ELM solvers (random hidden layer) for nonlinear problems
# ============================================================================

def _random_hidden_features(PhiT: np.ndarray, hidden: int = 64, rng: np.random.Generator = None) -> np.ndarray:
    """Generate random hidden-layer features via a tanh activation.
    PhiT has shape (Nq, m+1). The function returns an augmented feature matrix
    of shape (Nq, m+1 + hidden).
    """
    rng = np.random.default_rng(SEED) if rng is None else rng
    m1 = PhiT.shape[1]
    # Random projection
    W = rng.standard_normal((m1, hidden))
    b = rng.standard_normal(hidden)
    hidden_feat = np.tanh(PhiT @ W + b)
    return np.hstack([PhiT, hidden_feat])

def solve_affine_deep(alpha, mhat, S, y0, b0, b1, s0, s1, Nq=NQ,
                     lam_b=1.0, lam_s=1.0, hidden=64, rng=None):
    """Affine solve with an extra random hidden layer (ELM style)."""
    B = Blocks(alpha, mhat, S, Nq)
    t, PhiT = B.t, B.PhiT
    PhiT_aug = _random_hidden_features(PhiT, hidden, rng)
    m1_aug = PhiT_aug.shape[1]
    m1 = PhiT.shape[1]
    Z = np.zeros((Nq, m1))
    Z_aug = np.zeros((Nq, m1_aug))
    b0v = b0(t) if callable(b0) else np.full(Nq, float(b0))
    s0v = s0(t) if callable(s0) else np.full(Nq, float(s0))
    wb, ws = np.sqrt(lam_b), np.sqrt(lam_s)
    # R1: Phi c - DetT tb - StoT ts = y0
    R1 = np.hstack([PhiT_aug, -B.DetT, -B.StoT])
    # R2: wb * (Phi tb - b1*Phi c) = wb * b0
    R2 = np.hstack([-b1 * wb * PhiT_aug, wb * PhiT, Z])
    # R3: ws * (Phi ts - s1*Phi c) = ws * s0
    R3 = np.hstack([-s1 * ws * PhiT_aug, Z, ws * PhiT])
    Amat = np.vstack([R1, R2, R3])
    rhs = np.concatenate([np.full(Nq, y0), wb * b0v, ws * s0v])
    z, _, _, _ = lstsq(Amat, rhs, lapack_driver="gelsd")
    c = z[:m1_aug]
    tb = z[m1_aug:m1_aug + m1]
    ts = z[m1_aug + m1:]
    r = Amat @ z - rhs
    return c, tb, ts, float(np.linalg.norm(r) / np.sqrt(Nq))

def solve_gauss_newton_deep(alpha: float, mhat: int, S: np.ndarray, y0: float,
                            bfun, bprime, sfun, sprime,
                            Nq: int = NQ, lam_b: float = 1.0, lam_s: float = 1.0,
                            tol: float = 1e-13, maxit: int = 60,
                            hidden: int = 64, rng: np.random.Generator = None,
                            z0=None, verbose=False):
    """Gauss‑Newton solver that uses a random hidden‑layer augmentation.
    The hidden features are generated once and kept fixed throughout the
    iterations, mirroring the ELM philosophy.
    """
    B = Blocks(alpha, mhat, S, Nq)
    t, PhiT = B.t, B.PhiT
    PhiT_aug = _random_hidden_features(PhiT, hidden, rng)
    m1_aug = PhiT_aug.shape[1]
    m1 = PhiT.shape[1]
    Z = np.zeros((Nq, m1))
    wb, ws = np.sqrt(lam_b), np.sqrt(lam_s)

    def residual(z):
        c, tb, ts = z[:m1_aug], z[m1_aug:m1_aug + m1], z[m1_aug + m1:]
        N = PhiT_aug @ c
        r1 = N - y0 - B.DetT @ tb - B.StoT @ ts
        r2 = wb * (PhiT @ tb - bfun(t, N))
        r3 = ws * (PhiT @ ts - sfun(t, N))
        return np.concatenate([r1, r2, r3]), N

    def jac(z, N):
        J1 = np.hstack([PhiT_aug, -B.DetT, -B.StoT])
        J2 = wb * np.hstack([-(bprime(t, N))[:, None] * PhiT_aug, PhiT, Z])
        J3 = ws * np.hstack([-(sprime(t, N))[:, None] * PhiT_aug, Z, PhiT])
        return np.vstack([J1, J2, J3])

    if z0 is None:
        # initialise from affine deep solve (linearising about y0)
        b1_0 = float(bprime(np.array([0.0]), np.array([y0]))[0])
        s1_0 = float(sprime(np.array([0.0]), np.array([y0]))[0])
        b0f = lambda tt: bfun(tt, np.full_like(tt, y0)) - b1_0 * y0
        s0f = lambda tt: sfun(tt, np.full_like(tt, y0)) - s1_0 * y0
        c0, tb0, ts0, _ = solve_affine_deep(alpha, mhat, S, y0, b0f, b1_0, s0f, s1_0,
                                             Nq, lam_b, lam_s, hidden, rng)
        z = np.concatenate([c0, tb0, ts0])
    else:
        z = z0.copy()
    r, N = residual(z)
    for it in range(maxit):
        J = jac(z, N)
        dz, *_ = lstsq(J, -r, lapack_driver="gelsd")
        step = 1.0
        for _ in range(30):
            r_new, N_new = residual(z + step * dz)
            if np.linalg.norm(r_new) <= np.linalg.norm(r) * (1 + 1e-14) or step < 1e-8:
                break
            step *= 0.5
        z, r, N = z + step * dz, r_new, N_new
        if verbose:
            print(f"  GN it {it:2d} |r| = {np.linalg.norm(r):.3e} step {step:g}")
        if np.linalg.norm(step * dz) < tol * max(1.0, np.linalg.norm(z)):
            break
# ============================================================================

# End of Gauss‑Newton iterations
    c, tb, ts = z[:m1_aug], z[m1_aug:m1_aug + m1], z[m1_aug + m1:]
    return c, tb, ts, float(np.linalg.norm(r) / np.sqrt(Nq)), it + 1

def em_caputo(alpha, bfun, sfun, y0, dB, t_eval=None):
    """
    y_{k+1} = y0 + (1/G(a)) sum_{j<=k} [ b(t_j,y_j) w_{k+1-j} +
                                         sigma(t_j,y_j) (t_{k+1}-t_j)^{a-1} dB_j ],
    w_d = int_{cell} (t_{k+1}-s)^{a-1} ds = D^a (d^a-(d-1)^a)/a  (exact),
    left-point kernel for the diffusion (non-anticipating, Appendix B analogue).
    dB shape (n,) or (R, n).  Returns y on the full mesh (R, n+1) or (n+1,).
    """
    one_path = dB.ndim == 1
    dBm = dB[None, :] if one_path else dB
    R, n = dBm.shape
    D = 1.0 / n
    tm = np.linspace(0.0, 1.0, n + 1)
    d = np.arange(1, n + 1, dtype=float)
    wdet = D**alpha * _pow_diff(d, alpha) / alpha
    ksto = np.power(d * D, alpha - 1.0)
    ga = sgamma(alpha)
    y = np.empty((R, n + 1)); y[:, 0] = y0
    Bh = np.empty((R, n)); Sh = np.empty((R, n))
    for k in range(n):
        Bh[:, k] = bfun(tm[k], y[:, k])
        Sh[:, k] = sfun(tm[k], y[:, k]) * dBm[:, k]
        wd = wdet[k::-1]      # w_{k+1-j}, j = 0..k
        ks = ksto[k::-1]
        y[:, k + 1] = y0 + (Bh[:, :k + 1] @ wd + Sh[:, :k + 1] @ ks) / ga
    return y[0] if one_path else y

# ============================================================================
# 6. Mittag-Leffler  E_{a,b}(z)
# ============================================================================

def ml_series_f64(alpha, beta, z, kmax=400):
    """Vectorised float64 series with cancellation tracking.
    Returns (value, max_term_magnitude)."""
    z = np.asarray(z, dtype=float)
    s = np.zeros_like(z); term = np.ones_like(z) / sgamma(beta)
    mx = np.abs(term).copy()
    s += term
    for k in range(1, kmax):
        term = np.power(z, k) / np.exp(gammaln(alpha * k + beta))
        s += term
        mx = np.maximum(mx, np.abs(term))
        if np.all(np.abs(term) <= 1e-18 * (np.abs(s) + 1e-300)) and k > 5:
            break
    return s, mx

def ml_mp(alpha, beta, z, dps=60):
    """Arbitrary-precision E_{a,b}(z) by direct summation with adaptive dps."""
    import mpmath as mp
    az = abs(float(z))
    loss = 0.0
    if az > 1.0:  # estimate digits lost to cancellation via the peak term
        kpk = max(1.0, az ** (1.0 / alpha) / alpha)
        loss = max(0.0, (kpk * np.log(az) - gammaln(alpha * kpk + beta)) / np.log(10.0))
    with mp.workdps(int(dps + loss + 10)):
        zz = mp.mpf(repr(float(z)))
        s = mp.mpf(0); k = 0
        while True:
            t = zz ** k / mp.gamma(alpha * k + beta)
            s += t
            if k > 5 and abs(t) < mp.mpf(10) ** (-(dps + 5)) * (abs(s) + mp.mpf(10) ** (-50)):
                break
            k += 1
            if k > 20000:
                break
        return float(s)

def ml_vec(alpha, beta, z, safe_thresh=1e10):
    """Float64 vectorised E_{a,b} with automatic mpmath fallback where the
    series cancellation would exceed ~6 digits."""
    z = np.asarray(z, dtype=float)
    val, mx = ml_series_f64(alpha, beta, z)
    bad = mx / (np.abs(val) + 1e-300) > safe_thresh
    if np.any(bad):
        idx = np.where(bad)[0] if z.ndim == 1 else np.argwhere(bad)
        flat = val.ravel(); zf = z.ravel(); badf = bad.ravel()
        for i in np.where(badf)[0]:
            flat[i] = ml_mp(alpha, beta, zf[i])
        val = flat.reshape(z.shape)
    return val

# ============================================================================
# 7. Graded composite Gauss-Legendre + Gauss-Jacobi quadrature (ablation)
# ============================================================================

def graded_gl(a, b_end, q=0.35, levels=26, order=16):
    """Nodes/weights for int_a^{b} f, geometric grading of [a,b] toward a
    (endpoint algebraic singularity of type (x-a)^gamma, gamma > -1)."""
    xg, wg = np.polynomial.legendre.leggauss(order)
    L = b_end - a
    pts = np.concatenate([[a], a + L * q ** np.arange(levels, -1, -1.0)])
    xs, ws = [], []
    for lo, hi in zip(pts[:-1], pts[1:]):
        h = 0.5 * (hi - lo)
        xs.append(lo + h * (xg + 1.0)); ws.append(h * wg)
    return np.concatenate(xs), np.concatenate(ws)

def inner_frac_quad(alpha, q=0.35, levels=26, order=16, ngj=24, delta=0.5):
    """
    Fixed reference rule in s for  I^a phi(t) = (t^a/G(a)) int_0^1 (1-s)^{a-1} phi(t s) ds:
    graded GL toward s=0 on [0, 1-delta] (weight carried in the integrand) +
    Gauss-Jacobi with weight (1-s)^{a-1} on [1-delta, 1].
    Returns (s_nodes, weights) such that  I^a phi(t) ~ (t^a/G(a)) * sum w_i phi(t s_i).
    """
    s1, w1 = graded_gl(0.0, 1.0 - delta, q=q, levels=levels, order=order)
    w1 = w1 * np.power(1.0 - s1, alpha - 1.0)
    xj, wj = roots_jacobi(ngj, alpha - 1.0, 0.0)     # weight (1-x)^{a-1} on [-1,1]
    s2 = 1.0 - delta * (1.0 - xj) / 2.0
    w2 = (delta / 2.0) ** alpha * wj
    return np.concatenate([s1, s2]), np.concatenate([w1, w2])

# ============================================================================
# 8. KL smoothing of the sampled path (ablation driver)
# ============================================================================

def kl_coeffs_from_path(dB_fine, K=8):
    """Z_k = int_0^1 sqrt(2) cos((k-1/2) pi t) dB(t), left-point on the fine mesh
    of the *same* sampled path (seed 42) -> the smooth driver is the K-term
    Karhunen-Loeve truncation of that very path."""
    n = dB_fine.size
    tl = np.arange(n) / n
    k = np.arange(1, K + 1)
    return (np.sqrt(2.0) * np.cos(np.outer(k - 0.5, np.pi * tl))) @ dB_fine

def kl_deriv(Z, t):
    """B_K'(t) = sum_k Z_k sqrt(2) cos((k-1/2) pi t)."""
    t = np.asarray(t, float)
    k = np.arange(1, Z.size + 1)
    return (np.sqrt(2.0) * np.cos(np.outer(k - 0.5, np.pi * t)) * Z[:, None]).sum(axis=0)

# ============================================================================
# 9. error metrics + misc
# ============================================================================

def l2_err(f, g, tgrid):
    return float(np.sqrt(np.trapezoid((f - g) ** 2, tgrid)))

def linf_err(f, g):
    return float(np.max(np.abs(f - g)))

def sci(x):
    return "%.3e" % x

def save_csv(name, header, rows):
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        for r in rows:
            wr.writerow([v if isinstance(v, str) else ("%.16e" % v) for v in r])
    return path

def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "figure.dpi": 150, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    })
    return plt
