"""
parallel.py
===========
Batched building blocks: the Stochastic Fubini Adjoint Tensor contraction for
S_alpha (used by the FCMP solver in fcmp.py) and a process-parallel fractional
Euler-Maruyama reference.
"""

from __future__ import annotations
import os
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from scipy.special import gamma as sgamma
from .core_mldnn import basis_eval

def build_fubini_tensor(alpha: float, mhat: int, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the deterministic Stochastic Fubini Adjoint Contraction Tensor M_tens.
    By Stochastic Fubini, the cross-covariance Xi_ik = int_0^1 K_k(tau) M_i(tau) dB_tau
    is an exact linear map from the Brownian increments dB to the matrix Xi.
    
    Returns:
        M_tens: ( (mhat+1)^2, n_steps ) matrix such that (M_tens @ dB.T).T gives flattened Xi.
        omega_inv: diagonal weights (2*alpha*k + 1)
    """
    tm = np.linspace(0.0, 1.0, n_steps + 1)
    Mmesh = basis_eval(alpha, mhat, tm)
    m1 = mhat + 1
    D = 1.0 / n_steps
    d = np.arange(1, n_steps + 1, dtype=float)
    with np.errstate(divide="ignore"):
        pdiff_a = -np.expm1(alpha * np.log1p(-1.0 / d)) * np.power(d, alpha)
        pdiff_a1 = -np.expm1((alpha + 1.0) * np.log1p(-1.0 / d)) * np.power(d, alpha + 1.0)
        
    m0 = D**alpha * pdiff_a / alpha
    A1oD = D**alpha * pdiff_a1 / (alpha + 1.0)
    KL = np.concatenate(([0.0], A1oD - (d - 1.0) * m0))
    KR = np.concatenate(([0.0], d * m0 - A1oD))
    ga = sgamma(alpha)
    w_trapz = np.full(n_steps + 1, 1.0 / n_steps)
    w_trapz[0] *= 0.5
    w_trapz[-1] *= 0.5
    wM = (w_trapz[None, :] * Mmesh) / ga

    nf = 1 << int(np.ceil(np.log2(2 * (n_steps + 1))))
    wM_rev = wM[:, ::-1]
    rfft_KL = np.fft.rfft(KL, nf)
    rfft_KR = np.fft.rfft(KR, nf)

    conv_L = np.fft.irfft(np.fft.rfft(wM_rev, nf, axis=1) * rfft_KL[None, :], nf, axis=1)[:, :n_steps + 1]
    conv_R = np.fft.irfft(np.fft.rfft(wM_rev, nf, axis=1) * rfft_KR[None, :], nf, axis=1)[:, :n_steps + 1]

    AL = conv_L[:, n_steps:0:-1].T
    AR = conv_R[:, n_steps:0:-1].T

    # in place: at mhat = 40, n_steps = 65536 each (n_steps, m1, m1) array is ~0.9 GB
    T_tensor = Mmesh[:, :-1].T[:, :, None] * AL[:, None, :]
    for i in range(m1):
        T_tensor[:, i, :] += Mmesh[i, 1:, None] * AR
    T_tensor /= D
    M_tens = T_tensor.reshape(n_steps, m1 * m1).T  # shape ((mhat+1)^2, n_steps)
    omega_inv = (2.0 * alpha * np.arange(m1) + 1.0)
    return M_tens, omega_inv

def build_S_fubini_batch(alpha: float, mhat: int, dB: np.ndarray, M_tens: np.ndarray | None = None) -> np.ndarray:
    """
    Evaluate S_alpha for an arbitrary batch of paths dB (shape (N_paths, n_steps))
    via a single BLAS matrix multiplication.
    """
    if dB.ndim == 1:
        dB = dB[None, :]
    n_paths, n_steps = dB.shape
    m1 = mhat + 1
    
    if M_tens is None:
        M_tens, omega_inv = build_fubini_tensor(alpha, mhat, n_steps)
    else:
        omega_inv = (2.0 * alpha * np.arange(m1) + 1.0)
        
    # Hardware-accelerated with PyTorch (Apple Accelerate / NEON / AMX)
    M_t = torch.from_numpy(M_tens)
    dB_t = torch.from_numpy(dB)
    Xi_t = torch.matmul(dB_t, M_t.T)
    Xi_all = Xi_t.numpy().reshape(n_paths, m1, m1)
    S_all = Xi_all * omega_inv[None, None, :]
    return S_all

def _em_caputo_chunk(args):
    chunk_dB, alpha, bfun, sfun, y0, t_eval = args
    R, n = chunk_dB.shape
    D = 1.0 / n
    tm = np.linspace(0.0, 1.0, n + 1)
    d = np.arange(1, n + 1, dtype=float)
    with np.errstate(divide="ignore"):
        pdiff_a = -np.expm1(alpha * np.log1p(-1.0 / d)) * np.power(d, alpha)
    wdet = D**alpha * pdiff_a / alpha
    ksto = np.power(d * D, alpha - 1.0)
    ga = sgamma(alpha)
    
    y = np.empty((R, n + 1))
    y[:, 0] = y0
    Bh = np.empty((R, n))
    Sh = np.empty((R, n))
    
    for k in range(n):
        Bh[:, k] = bfun(tm[k], y[:, k])
        Sh[:, k] = sfun(tm[k], y[:, k]) * chunk_dB[:, k]
        wd = wdet[k::-1]
        ks = ksto[k::-1]
        y[:, k + 1] = y0 + (Bh[:, :k + 1] @ wd + Sh[:, :k + 1] @ ks) / ga
        
    if t_eval is not None:
        return np.array([np.interp(t_eval, tm, y[i]) for i in range(R)])
    return y

def parallel_em_caputo(
    alpha: float,
    bfun,
    sfun,
    y0: float,
    dB: np.ndarray,
    t_eval: np.ndarray | None = None,
    n_workers: int | None = None
) -> np.ndarray:
    """Parallelized Fractional Euler-Maruyama solver across Monte Carlo paths."""
    if dB.ndim == 1:
        dB = dB[None, :]
    n_paths, n_steps = dB.shape
    
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, n_paths)
        
    chunks = np.array_split(dB, n_workers)
    worker_args = [(chunk, alpha, bfun, sfun, y0, t_eval) for chunk in chunks if len(chunk) > 0]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results_nested = list(executor.map(_em_caputo_chunk, worker_args))
        
    return np.vstack(results_nested)
